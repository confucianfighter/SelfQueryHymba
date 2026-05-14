"""Autoregressive sampler for HybridHymba2xPairCharLM.

This sampler uses only pair slot 0 as the next-character head.
Slot 1 is treated as an auxiliary training head and is never sampled.

Expected usage from repo root:

    python scripts/sample/sample_pair2x.py \
        --checkpoint experiments/.../checkpoint.pt \
        --text-file data/tinyshakespeare/input.txt \
        --prompt "ROMEO:" \
        --steps 500 \
        --temperature 0.9 \
        --top-k 40

Notes:
- The current HybridHymba2xPairCharLM implementation can produce a compressed
  slot whose causal time equals the last real token even for odd context lengths.
  Therefore this script does NOT right-pad by default.
- Generation is intentionally slow and conservative: one full model forward per
  generated character. That is the cleanest qualitative sampler for this setup.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


def _repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_repo_imports(repo_root: Path) -> None:
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Could not find a model state dict. Expected checkpoint['model_state_dict'], "
        "checkpoint['state_dict'], checkpoint['model'], or a raw state_dict."
    )


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _extract_chars(checkpoint: Any, text_file: str | None, vocab_file: str | None) -> tuple[str, ...]:
    if vocab_file:
        path = Path(vocab_file)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "chars" in data:
                return tuple(data["chars"])
            if "itos" in data:
                itos = data["itos"]
                if isinstance(itos, dict):
                    return tuple(itos[str(i)] if str(i) in itos else itos[i] for i in range(len(itos)))
                return tuple(itos)
        if isinstance(data, list):
            return tuple(data)
        raise ValueError(f"Unsupported vocab JSON format in {path}")

    if isinstance(checkpoint, dict):
        for key in ("chars", "vocab_chars"):
            chars = checkpoint.get(key)
            if chars is not None:
                return tuple(chars)
        vocab = checkpoint.get("vocab")
        if isinstance(vocab, dict):
            if "chars" in vocab:
                return tuple(vocab["chars"])
            if "itos" in vocab:
                itos = vocab["itos"]
                if isinstance(itos, dict):
                    return tuple(itos[str(i)] if str(i) in itos else itos[i] for i in range(len(itos)))
                return tuple(itos)

    if text_file:
        text = Path(text_file).read_text(encoding="utf-8")
        if not text:
            raise ValueError("text_file is empty; cannot build vocabulary")
        return tuple(sorted(set(text)))

    raise ValueError(
        "Could not determine vocabulary. Provide --text-file or --vocab-file, "
        "or save chars/vocab in the checkpoint."
    )


def _extract_config_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    for key in ("config", "model_config", "hparams", "args"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
    return {}


def _sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    greedy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"logits must be [batch, vocab], got {tuple(logits.shape)}")
    if greedy:
        probs = torch.softmax(logits, dim=-1)
        return logits.argmax(dim=-1, keepdim=True), probs

    temperature = max(float(temperature), 1e-6)
    logits = logits / temperature

    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        vals, idx = torch.topk(logits, top_k, dim=-1)
        masked = torch.full_like(logits, -float("inf"))
        logits = masked.scatter(dim=-1, index=idx, src=vals)

    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    return next_id, probs


def _top_probs_text(probs: torch.Tensor, itos: dict[int, str], *, k: int = 5) -> str:
    vals, idx = torch.topk(probs[0], min(k, probs.shape[-1]))
    parts = []
    for prob, token_id in zip(vals.detach().cpu().tolist(), idx.detach().cpu().tolist()):
        ch = itos[int(token_id)]
        escaped = ch.encode("unicode_escape").decode("ascii")
        parts.append(f"{escaped}:{prob:.3f}")
    return " ".join(parts)


def _find_slot_for_last_real_token(memory_times: torch.Tensor, last_real_t: int) -> int:
    times = memory_times
    if times.ndim == 2:
        times = times[0]
    if times.ndim != 1:
        raise ValueError(f"memory_times must be [slots] or [batch, slots], got {tuple(memory_times.shape)}")
    matches = (times == last_real_t).nonzero(as_tuple=False).flatten()
    if matches.numel() == 0:
        raise RuntimeError(
            f"No compressed slot has causal_time == last real token index {last_real_t}. "
            f"Available tail times: {times[-8:].detach().cpu().tolist()}"
        )
    return int(matches[-1].item())


@torch.no_grad()
def generate_pair2x(
    model: torch.nn.Module,
    context: torch.Tensor,
    *,
    steps: int,
    temperature: float,
    top_k: int | None,
    greedy: bool,
    itos: dict[int, str],
    debug: bool,
) -> torch.Tensor:
    """Generate one character per forward pass using pair_logits slot 0 only."""
    model.eval()

    for step in range(steps):
        real_len = context.shape[1]
        last_real_t = real_len - 1

        out = model(context)
        if out.pair_logits is None or out.memory_times is None:
            raise RuntimeError("Model output does not contain pair_logits and memory_times")

        slot = _find_slot_for_last_real_token(out.memory_times, last_real_t)
        logits = out.pair_logits[:, slot, 0, :]  # slot 0 is the valid t+1 generator
        next_id, probs = _sample_from_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )
        context = torch.cat([context, next_id], dim=1)

        if debug:
            times = out.memory_times[0] if out.memory_times.ndim == 2 else out.memory_times
            sampled = itos[int(next_id[0, 0].item())].encode("unicode_escape").decode("ascii")
            entropy = -(probs[0] * probs[0].clamp_min(1e-12).log()).sum().item()
            print(
                f"step={step:04d} real_len={real_len} last_t={last_real_t} "
                f"slot={slot} compressed_time={int(times[slot].item())} "
                f"sample={sampled!r} entropy={entropy:.3f} top={_top_probs_text(probs, itos)}",
                file=sys.stderr,
            )

    return context


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from HybridHymba2xPairCharLM using slot-0 autoregression.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--text-file", help="Training text file used to rebuild sorted character vocabulary")
    parser.add_argument("--vocab-file", help="Optional JSON vocabulary with chars or itos")
    parser.add_argument("--prompt", default="", help="Seed text")
    parser.add_argument("--prompt-file", help="Read seed text from file instead of --prompt")
    parser.add_argument("--steps", type=int, default=500, help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repo-root", default=None, help="Repo root; defaults to two dirs above this script")
    parser.add_argument("--output", help="Optional output text file")
    parser.add_argument("--debug", action="store_true", help="Print slot/time/top-prob debug info to stderr")

    # Fallback config values if checkpoint does not store a model config.
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--pair-model-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--full-rate-layers", type=int, default=8)
    parser.add_argument("--compressed-layers", type=int, default=8)
    parser.add_argument("--ssm-kernel-size", type=int, default=3)
    parser.add_argument("--no-tie-head", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root_from_this_file()
    _ensure_repo_imports(repo_root)

    from models.CST import HybridHymba2xPairCharLM, HybridHymba2xPairCharLMConfig

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    chars = _extract_chars(checkpoint, args.text_file, args.vocab_file)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    config_dict = _extract_config_dict(checkpoint)
    allowed_keys = {
        "vocab_size",
        "d_model",
        "pair_model_dim",
        "num_heads",
        "full_rate_layers",
        "compressed_layers",
        "ssm_kernel_size",
        "tie_head",
    }
    filtered = {k: v for k, v in config_dict.items() if k in allowed_keys}
    filtered.setdefault("vocab_size", len(chars))
    filtered.setdefault("d_model", args.d_model)
    filtered.setdefault("pair_model_dim", args.pair_model_dim)
    filtered.setdefault("num_heads", args.num_heads)
    filtered.setdefault("full_rate_layers", args.full_rate_layers)
    filtered.setdefault("compressed_layers", args.compressed_layers)
    filtered.setdefault("ssm_kernel_size", args.ssm_kernel_size)
    filtered.setdefault("tie_head", not args.no_tie_head)

    if int(filtered["vocab_size"]) != len(chars):
        raise ValueError(
            f"Config vocab_size={filtered['vocab_size']} but vocabulary has {len(chars)} chars. "
            "Use the exact training vocabulary."
        )

    model = HybridHymba2xPairCharLM(HybridHymba2xPairCharLMConfig(**filtered)).to(args.device)
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing[:12]}{'...' if len(missing) > 12 else ''}", file=sys.stderr)
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected[:12]}{'...' if len(unexpected) > 12 else ''}", file=sys.stderr)

    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    if not prompt:
        prompt = chars[0]
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(f"Prompt contains chars not in vocabulary: {unknown!r}")

    ids = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=args.device)
    sampled = generate_pair2x(
        model,
        ids,
        steps=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
        itos=itos,
        debug=args.debug,
    )
    text = "".join(itos[int(i)] for i in sampled[0].detach().cpu().tolist())

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
