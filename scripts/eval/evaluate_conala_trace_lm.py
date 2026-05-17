from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.CST import (  # noqa: E402
    BranchedLossQueryFastHymbaCharLM,
    CharVocabulary,
    FastHymbaCharLM,
    FastHymbaCharLMConfig,
    LossContextInjectedFastHymbaCharLM,
    LossQueryFastHymbaCharLM,
)
from scripts.data.prepare_conala_corpus import (  # noqa: E402
    DEFAULT_CURATED_TEST,
    DEFAULT_CURATED_TRAIN,
    read_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a char-LM checkpoint on CoNaLa intent-to-snippet examples.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_CURATED_TEST)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--max-new-chars", type=int, default=220)
    parser.add_argument("--task", default="conala_python")
    parser.add_argument("--sentinel", default="<END>")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=10)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    vocab = CharVocabulary.from_chars(checkpoint["vocab_chars"])
    model_config = FastHymbaCharLMConfig(
        vocab_size=vocab.size,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["layers"],
        ssm_kernel_size=config.get("ssm_kernel_size", 3),
        state_branch=config.get("state_branch", "conv"),
    )
    architecture = config.get("architecture", "fast_hymba")
    if architecture == "fast_hymba":
        model = FastHymbaCharLM(model_config).to(device)
    elif architecture == "loss_query_hymba":
        model = LossQueryFastHymbaCharLM(model_config).to(device)
    elif architecture == "branched_loss_query_hymba":
        model = BranchedLossQueryFastHymbaCharLM(model_config).to(device)
    elif architecture == "loss_context_injected_hymba":
        model = LossContextInjectedFastHymbaCharLM(model_config).to(device)
    else:
        raise ValueError(f"unsupported architecture: {architecture!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab, config


def load_rows(path: Path) -> list[dict[str, str]]:
    examples = []
    for row in read_json(path):
        intent = row.get("rewritten_intent") or row.get("intent")
        snippet = row.get("snippet")
        if intent and snippet:
            examples.append({"intent": intent.strip(), "snippet": snippet.strip()})
    return examples


def prompt_for(intent: str, *, task: str) -> str:
    return f"Task: {task}\nInput:\n{intent}\n\nOutput:\n"


def normalize_snippet(snippet: str) -> str:
    return "\n".join(line.rstrip() for line in snippet.strip().splitlines())


@torch.no_grad()
def generate_completion(
    model: torch.nn.Module,
    vocab: CharVocabulary,
    prompt: str,
    *,
    seq_len: int,
    max_new_chars: int,
    sentinel: str,
    device: torch.device,
) -> str:
    ids = vocab.encode(prompt, device=device).tolist()
    prompt_len = len(ids)
    for _ in range(max_new_chars):
        context = torch.tensor([ids[-seq_len:]], dtype=torch.long, device=device)
        logits = model(context, pad_to_length=seq_len).logits[:, -1, :]
        ids.append(int(logits.argmax(dim=-1).item()))
        generated = vocab.decode(ids[prompt_len:])
        if sentinel in generated:
            break
    completion = vocab.decode(ids[prompt_len:])
    if sentinel in completion:
        completion = completion.split(sentinel, 1)[0]
    return completion.strip()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint if Path(args.checkpoint).is_absolute() else ROOT / args.checkpoint
    data_path = args.data_path if args.data_path.is_absolute() else ROOT / args.data_path
    device = torch.device(args.device)
    model, vocab, config = load_model(checkpoint_path, device)
    seq_len = int(config["seq_len"])

    rows = load_rows(data_path)
    results = []
    skipped_oov = 0
    for row in rows:
        if len(results) >= args.examples:
            break
        prompt = prompt_for(row["intent"], task=args.task)
        try:
            vocab.encode(prompt)
        except ValueError:
            skipped_oov += 1
            continue
        generated = generate_completion(
            model,
            vocab,
            prompt,
            seq_len=seq_len,
            max_new_chars=args.max_new_chars,
            sentinel=args.sentinel,
            device=device,
        )
        expected = normalize_snippet(row["snippet"])
        predicted = normalize_snippet(generated)
        results.append(
            {
                "intent": row["intent"],
                "prompt": prompt,
                "expected": expected,
                "generated": predicted,
                "exact": predicted == expected,
            }
        )

    exact = sum(row["exact"] for row in results)
    summary = {
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
        "examples": len(results),
        "skipped_oov_prompts": skipped_oov,
        "exact": exact,
        "exact_accuracy": exact / len(results) if results else 0.0,
        "samples": results[: args.samples],
    }
    payload = {"summary": summary, "rows": results}
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
