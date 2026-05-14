from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.CST import (  # noqa: E402
    AlternatingClockTokenFastHymbaCharLM,
    DEFAULT_CHAR_LM_DATA_PATH,
    CharVocabulary,
    ClockConditionedFastHymbaCharLM,
    CurrentClockFusionFastHymbaCharLM,
    EphemeralClockSidecarFastHymbaCharLM,
    EfficientStrictAlternatingClockTokenFastHymbaCharLM,
    FastHymbaCharLM,
    FastHymbaCharLMConfig,
    IsolatedAlternatingClockTokenFastHymbaCharLM,
    LossQueryFastHymbaCharLM,
    MSHymbaCharLM,
    MSHymbaCharLMConfig,
    PreviousClockConditionedFastHymbaCharLM,
    StrictAlternatingClockTokenFastHymbaCharLM,
    TypedIsolatedAlternatingClockTokenFastHymbaCharLM,
    TwoSideHymbaCharLM,
    TwoSideHymbaCharLMConfig,
    next_token_accuracy,
    next_token_loss,
    weighted_next_token_loss,
)


@dataclass(frozen=True)
class TrainConfig:
    data_path: str
    mod_id: str
    run_name: str
    seed: int
    device: str
    architecture: str
    resume_checkpoint: str | None
    vocab_checkpoint: str | None
    replaced_oov_chars: dict[str, int]
    allow_vocab_expansion: bool
    expanded_vocab_chars: list[str]
    steps: int
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    layers: int
    ssm_kernel_size: int
    state_branch: str
    num_scales: int
    scale_kernel_size: int
    scale_block_mlp: bool
    global_mlp: bool
    loss_kind: str
    digit_loss_weight: float
    nondigit_loss_weight: float
    loss_prediction_alpha: float
    set_clock_gate: float | None
    freeze_clock_gates: bool
    lr: float
    plateau_recovery: bool
    plateau_patience_steps: int
    plateau_lr_factor: float
    plateau_min_lr: float
    plateau_max_lr: float | None
    plateau_min_delta: float
    grad_clip: float
    eval_every: int
    checkpoint_every: int
    sample_every: int
    sample_prompt: str | None
    sample_prompts: list[str] | None
    sample_prompt_file: str | None
    sample_chars: int
    sample_temperature: float
    train_only_layers: str | None
    unfreeze_all_after_step: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-XL fast Hymba char-LM baseline.")
    parser.add_argument("--data-path", default=DEFAULT_CHAR_LM_DATA_PATH)
    parser.add_argument("--mod-id", default="001_first_pass_hymba_cst")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--architecture",
        choices=(
            "fast_hymba",
            "ms_hymba",
            "two_side_hymba",
            "loss_query_hymba",
            "current_clock_hymba",
            "ephemeral_clock_hymba",
            "clock_conditioned_hymba",
            "isolated_alternating_clock_hymba",
            "typed_isolated_alternating_clock_hymba",
            "alternating_clock_hymba",
            "strict_alternating_clock_hymba",
            "efficient_strict_alternating_clock_hymba",
            "previous_clock_hymba",
        ),
        default="fast_hymba",
    )
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--vocab-checkpoint", default=None)
    parser.add_argument("--replace-oov-with", default=None)
    parser.add_argument(
        "--allow-vocab-expansion",
        action="store_true",
        help="Append dataset-only chars to the checkpoint vocabulary and initialize their rows fresh.",
    )
    parser.add_argument("--no-resume-optimizer", action="store_true")
    parser.add_argument("--override-resume-lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--ssm-kernel-size", type=int, default=3)
    parser.add_argument("--state-branch", choices=("conv", "multistride_1_2"), default="conv")
    parser.add_argument("--num-scales", type=int, default=1)
    parser.add_argument("--scale-kernel-size", type=int, default=3)
    parser.add_argument("--scale-block-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--global-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-kind", choices=("next_token", "digit_weighted"), default="next_token")
    parser.add_argument("--digit-loss-weight", type=float, default=1.0)
    parser.add_argument("--nondigit-loss-weight", type=float, default=0.05)
    parser.add_argument("--loss-prediction-alpha", type=float, default=0.05)
    parser.add_argument(
        "--set-clock-gate",
        type=float,
        default=None,
        help="If set, initialize clock gate parameters to this value after loading/grafting.",
    )
    parser.add_argument(
        "--freeze-clock-gates",
        action="store_true",
        help="Freeze parameters whose names end with _gate after optional --set-clock-gate.",
    )
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--plateau-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plateau-patience-steps", type=int, default=500)
    parser.add_argument("--plateau-lr-factor", type=float, default=0.5)
    parser.add_argument("--plateau-min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--plateau-max-lr",
        type=float,
        default=None,
        help="Maximum LR used by plateau recovery; defaults to the LR active when training starts.",
    )
    parser.add_argument("--plateau-min-delta", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--sample-every", type=int, default=0)
    parser.add_argument("--sample-prompt", default=None)
    parser.add_argument(
        "--sample-prompts",
        action="append",
        default=None,
        help="Additional sample prompt. May be passed multiple times.",
    )
    parser.add_argument(
        "--sample-prompt-file",
        default=None,
        help="Text file containing sample prompts separated by blank lines.",
    )
    parser.add_argument("--sample-chars", type=int, default=400)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument(
        "--train-only-layers",
        default=None,
        help="Comma-separated 1-indexed layer numbers to train initially; all other parameters are frozen.",
    )
    parser.add_argument(
        "--unfreeze-all-after-step",
        type=int,
        default=None,
        help="If set, unfreeze all model parameters after this many training steps and recreate the optimizer.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    max_start = data.numel() - seq_len - 1
    if max_start <= 0:
        raise ValueError("dataset is too short for requested seq_len")
    starts = torch.randint(0, max_start, (batch_size,))
    batch = torch.stack([data[start : start + seq_len + 1] for start in starts])
    return batch.to(device)


def parse_layer_numbers(value: str | None, *, num_layers: int) -> list[int] | None:
    if value is None:
        return None
    layer_numbers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        layer_number = int(part)
        if layer_number < 1 or layer_number > num_layers:
            raise ValueError(f"layer number {layer_number} is outside 1..{num_layers}")
        layer_numbers.append(layer_number)
    if not layer_numbers:
        raise ValueError("--train-only-layers did not contain any layer numbers")
    return sorted(set(layer_numbers))


def resolve_sample_prompts(args: argparse.Namespace, *, fallback_prompt: str) -> list[str]:
    prompts: list[str] = []
    if args.sample_prompt is not None:
        prompts.append(args.sample_prompt)
    if args.sample_prompts:
        prompts.extend(args.sample_prompts)
    if args.sample_prompt_file is not None:
        prompt_path = Path(args.sample_prompt_file)
        if not prompt_path.is_absolute():
            prompt_path = ROOT / prompt_path
        prompt_text = prompt_path.read_text(encoding="utf-8")
        separator = "\n---\n" if "\n---\n" in prompt_text else "\n\n"
        prompts.extend(part.strip("\n") for part in prompt_text.split(separator) if part.strip())
    if not prompts:
        prompts.append(fallback_prompt)
    return prompts


def set_fast_hymba_trainable_layers(model: torch.nn.Module, layer_numbers: list[int] | None) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(layer_numbers is None)
    if layer_numbers is None:
        return
    if not hasattr(model, "layers"):
        raise ValueError("--train-only-layers currently requires an architecture with model.layers")
    layers = getattr(model, "layers")
    for layer_number in layer_numbers:
        for parameter in layers[layer_number - 1].parameters():
            parameter.requires_grad_(True)


def make_optimizer(model: torch.nn.Module, *, lr: float) -> torch.optim.Optimizer:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters are enabled")
    return torch.optim.AdamW(trainable_parameters, lr=lr)


def clamp_lr(lr: float, *, min_lr: float, max_lr: float) -> float:
    return max(min_lr, min(max_lr, lr))


def optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def configure_clock_gates(
    model: torch.nn.Module,
    *,
    set_value: float | None,
    freeze: bool,
) -> dict[str, float]:
    configured: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if not name.endswith("_gate"):
            continue
        if set_value is not None:
            parameter.data.fill_(set_value)
        if freeze:
            parameter.requires_grad_(False)
        configured[name] = float(parameter.detach().cpu().item())
    if (set_value is not None or freeze) and not configured:
        raise ValueError("requested clock gate configuration, but model has no *_gate parameters")
    return configured


def expand_tensor_to_shape(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    if source.shape == target.shape:
        return source
    if source.ndim != target.ndim:
        return None
    if source.shape[0] >= target.shape[0] or source.shape[1:] != target.shape[1:]:
        return None
    expanded = target.detach().clone()
    expanded[: source.shape[0]].copy_(source)
    if source.ndim == 1:
        expanded[source.shape[0] :].zero_()
    else:
        std = float(source.float().std().item())
        if std == 0.0:
            std = 0.02
        expanded[source.shape[0] :].normal_(mean=0.0, std=std)
    return expanded


def adapt_checkpoint_state_dict(
    model: torch.nn.Module,
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    model_state = model.state_dict()
    adapted = dict(checkpoint_state_dict)
    expanded_keys: list[str] = []
    shared_token_embedding = None
    if "token_embedding.weight" in checkpoint_state_dict and "token_embedding.weight" in model_state:
        shared_token_embedding = expand_tensor_to_shape(
            checkpoint_state_dict["token_embedding.weight"],
            model_state["token_embedding.weight"],
        )
        if shared_token_embedding is not None and shared_token_embedding.shape != checkpoint_state_dict["token_embedding.weight"].shape:
            adapted["token_embedding.weight"] = shared_token_embedding
            expanded_keys.append("token_embedding.weight")
    for key, value in checkpoint_state_dict.items():
        if key not in model_state:
            continue
        if key == "token_embedding.weight":
            continue
        if key == "lm_head.weight" and shared_token_embedding is not None and shared_token_embedding.shape == model_state[key].shape:
            adapted[key] = shared_token_embedding
            if value.shape != model_state[key].shape:
                expanded_keys.append(key)
            continue
        expanded = expand_tensor_to_shape(value, model_state[key])
        if expanded is not None and expanded.shape != value.shape:
            adapted[key] = expanded
            expanded_keys.append(key)
    return adapted, expanded_keys


def load_resume_state(
    model: torch.nn.Module,
    checkpoint: dict,
    *,
    target_architecture: str,
) -> tuple[bool, dict[str, list[str]]]:
    """Load checkpoint weights, allowing fast_hymba -> loss_query_hymba grafts."""

    checkpoint_architecture = checkpoint.get("config", {}).get("architecture", "fast_hymba")
    checkpoint_state_dict, expanded_keys = adapt_checkpoint_state_dict(model, checkpoint["model_state_dict"])
    if checkpoint_architecture == "fast_hymba" and target_architecture == "loss_query_hymba":
        result = model.load_state_dict(checkpoint_state_dict, strict=False)
        allowed_missing_prefixes = (
            "loss_query",
            "loss_query_attn.",
            "loss_query_norm.",
            "loss_memory_norm.",
            "loss_head.",
        )
        unexpected = list(result.unexpected_keys)
        disallowed_missing = [
            key
            for key in result.missing_keys
            if not any(key == prefix.rstrip(".") or key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                "incompatible checkpoint graft: "
                f"missing={disallowed_missing!r}, unexpected={unexpected!r}"
            )
        return True, {"missing_keys": list(result.missing_keys), "unexpected_keys": unexpected, "expanded_keys": expanded_keys}

    if checkpoint_architecture == "loss_query_hymba" and target_architecture == "current_clock_hymba":
        result = model.load_state_dict(checkpoint_state_dict, strict=False)
        allowed_missing_prefixes = (
            "current_clock_adapter.",
            "current_clock_gate",
        )
        unexpected = list(result.unexpected_keys)
        disallowed_missing = [
            key
            for key in result.missing_keys
            if not any(key == prefix.rstrip(".") or key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                "incompatible current-clock graft: "
                f"missing={disallowed_missing!r}, unexpected={unexpected!r}"
            )
        return True, {"missing_keys": list(result.missing_keys), "unexpected_keys": unexpected, "expanded_keys": expanded_keys}

    if checkpoint_architecture == "loss_query_hymba" and target_architecture == "ephemeral_clock_hymba":
        result = model.load_state_dict(checkpoint_state_dict, strict=False)
        allowed_missing_prefixes = (
            "clock_sidecar_attn.",
            "readout_sidecar_attn.",
            "clock_query_norm.",
            "clock_memory_norm.",
            "readout_query_norm.",
            "readout_memory_norm.",
            "readout_sidecar_gate",
        )
        unexpected = list(result.unexpected_keys)
        disallowed_missing = [
            key
            for key in result.missing_keys
            if not any(key == prefix.rstrip(".") or key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                "incompatible ephemeral-clock graft: "
                f"missing={disallowed_missing!r}, unexpected={unexpected!r}"
            )
        return True, {"missing_keys": list(result.missing_keys), "unexpected_keys": unexpected, "expanded_keys": expanded_keys}

    if checkpoint_architecture == "loss_query_hymba" and target_architecture == "clock_conditioned_hymba":
        result = model.load_state_dict(checkpoint_state_dict, strict=False)
        allowed_missing_prefixes = (
            "previous_clock_adapter.",
            "previous_clock_gate",
        )
        unexpected = list(result.unexpected_keys)
        disallowed_missing = [
            key
            for key in result.missing_keys
            if not any(key == prefix.rstrip(".") or key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                "incompatible clock-conditioned graft: "
                f"missing={disallowed_missing!r}, unexpected={unexpected!r}"
            )
        return True, {"missing_keys": list(result.missing_keys), "unexpected_keys": unexpected, "expanded_keys": expanded_keys}

    if checkpoint_architecture == "loss_query_hymba" and target_architecture == "alternating_clock_hymba":
        result = model.load_state_dict(checkpoint_state_dict, strict=False)
        allowed_missing_prefixes = (
            "clock_embedding",
        )
        allowed_missing_suffixes = (
            "loss_head.weight",
            "loss_head.bias",
        )
        unexpected = list(result.unexpected_keys)
        allowed_unexpected_prefixes = (
            "loss_query",
            "loss_query_attn.",
            "loss_query_norm.",
            "loss_memory_norm.",
        )
        unexpected = [
            key for key in unexpected if not any(key == prefix.rstrip(".") or key.startswith(prefix) for prefix in allowed_unexpected_prefixes)
        ]
        disallowed_missing = [
            key
            for key in result.missing_keys
            if not any(key == prefix.rstrip(".") or key.startswith(prefix) for prefix in allowed_missing_prefixes)
            and not any(key.endswith(suffix) for suffix in allowed_missing_suffixes)
        ]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                "incompatible alternating-clock graft: "
                f"missing={disallowed_missing!r}, unexpected={unexpected!r}"
            )
        return True, {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys), "expanded_keys": expanded_keys}

    if checkpoint_architecture == "loss_query_hymba" and target_architecture == "previous_clock_hymba":
        result = model.load_state_dict(checkpoint_state_dict, strict=False)
        allowed_missing_prefixes = (
            "previous_clock_adapter.",
            "previous_clock_gate",
        )
        unexpected = list(result.unexpected_keys)
        disallowed_missing = [
            key
            for key in result.missing_keys
            if not any(key == prefix.rstrip(".") or key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                "incompatible previous-clock graft: "
                f"missing={disallowed_missing!r}, unexpected={unexpected!r}"
            )
        return True, {"missing_keys": list(result.missing_keys), "unexpected_keys": unexpected, "expanded_keys": expanded_keys}

    model.load_state_dict(checkpoint_state_dict)
    return False, {"missing_keys": [], "unexpected_keys": [], "expanded_keys": expanded_keys}


def digit_target_weights(
    batch: torch.Tensor,
    vocab: CharVocabulary,
    *,
    digit_weight: float,
    nondigit_weight: float,
) -> torch.Tensor:
    if digit_weight < 0 or nondigit_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    targets = batch[:, 1:]
    weights = torch.full(targets.shape, nondigit_weight, dtype=torch.float32, device=batch.device)
    digit_ids = [vocab.stoi[str(digit)] for digit in range(10) if str(digit) in vocab.stoi]
    if digit_ids:
        digit_id_tensor = torch.tensor(digit_ids, dtype=targets.dtype, device=batch.device)
        digit_mask = (targets.unsqueeze(-1) == digit_id_tensor).any(dim=-1)
        weights = torch.where(digit_mask, torch.full_like(weights, digit_weight), weights)
    return weights


def model_loss(
    model: FastHymbaCharLM,
    batch: torch.Tensor,
    vocab: CharVocabulary,
    *,
    loss_kind: str,
    digit_weight: float,
    nondigit_weight: float,
    loss_prediction_alpha: float,
) -> torch.Tensor:
    output = model(batch)
    logits = output.logits
    if loss_kind == "digit_weighted":
        lm_loss = weighted_next_token_loss(
            logits,
            batch,
            digit_target_weights(
                batch,
                vocab,
                digit_weight=digit_weight,
                nondigit_weight=nondigit_weight,
            ),
        )
    else:
        lm_loss = next_token_loss(logits, batch)
    if output.loss_predictions is None:
        return lm_loss
    token_ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        batch[:, 1:].reshape(-1),
        reduction="none",
    ).view(batch.shape[0], -1)
    loss_pred = output.loss_predictions[:, :-1]
    aux_loss = F.mse_loss(loss_pred, token_ce.detach())
    return lm_loss + loss_prediction_alpha * aux_loss


@torch.no_grad()
def evaluate(
    model: FastHymbaCharLM,
    held_batch: torch.Tensor,
    vocab: CharVocabulary,
    *,
    loss_kind: str,
    digit_weight: float,
    nondigit_weight: float,
    loss_prediction_alpha: float,
) -> dict[str, float]:
    was_training = model.training

    model.train()
    train_mode_output = model(held_batch)
    train_mode_lm_loss = (
        weighted_next_token_loss(
            train_mode_output.logits,
            held_batch,
            digit_target_weights(
                held_batch,
                vocab,
                digit_weight=digit_weight,
                nondigit_weight=nondigit_weight,
            ),
        )
        if loss_kind == "digit_weighted"
        else next_token_loss(train_mode_output.logits, held_batch)
    )
    train_mode_loss = train_mode_lm_loss
    train_loss_pred_mse = None
    if train_mode_output.loss_predictions is not None:
        train_token_ce = F.cross_entropy(
            train_mode_output.logits[:, :-1].reshape(-1, train_mode_output.logits.shape[-1]),
            held_batch[:, 1:].reshape(-1),
            reduction="none",
        ).view(held_batch.shape[0], -1)
        train_loss_pred_mse = F.mse_loss(train_mode_output.loss_predictions[:, :-1], train_token_ce.detach())
        train_mode_loss = train_mode_lm_loss + loss_prediction_alpha * train_loss_pred_mse

    model.eval()
    eval_mode_output = model(held_batch)
    eval_mode_lm_loss = (
        weighted_next_token_loss(
            eval_mode_output.logits,
            held_batch,
            digit_target_weights(
                held_batch,
                vocab,
                digit_weight=digit_weight,
                nondigit_weight=nondigit_weight,
            ),
        )
        if loss_kind == "digit_weighted"
        else next_token_loss(eval_mode_output.logits, held_batch)
    )
    eval_mode_loss = eval_mode_lm_loss
    eval_loss_pred_mse = None
    if eval_mode_output.loss_predictions is not None:
        eval_token_ce = F.cross_entropy(
            eval_mode_output.logits[:, :-1].reshape(-1, eval_mode_output.logits.shape[-1]),
            held_batch[:, 1:].reshape(-1),
            reduction="none",
        ).view(held_batch.shape[0], -1)
        eval_loss_pred_mse = F.mse_loss(eval_mode_output.loss_predictions[:, :-1], eval_token_ce.detach())
        eval_mode_loss = eval_mode_lm_loss + loss_prediction_alpha * eval_loss_pred_mse
    eval_unweighted_loss = next_token_loss(eval_mode_output.logits, held_batch)
    eval_acc = next_token_accuracy(eval_mode_output.logits, held_batch)

    if was_training:
        model.train()

    result = {
        "train_mode_loss": float(train_mode_loss.item()),
        "eval_mode_loss": float(eval_mode_loss.item()),
        "eval_unweighted_loss": float(eval_unweighted_loss.item()),
        "loss_delta": float(abs(train_mode_loss.item() - eval_mode_loss.item())),
        "eval_next_char_accuracy": float(eval_acc.item()),
    }
    if eval_loss_pred_mse is not None and train_loss_pred_mse is not None:
        result["train_loss_prediction_mse"] = float(train_loss_pred_mse.item())
        result["eval_loss_prediction_mse"] = float(eval_loss_pred_mse.item())
    return result


@torch.no_grad()
def generate_sample(
    model: FastHymbaCharLM,
    vocab: CharVocabulary,
    prompt: str,
    *,
    max_new_chars: int,
    seq_len: int,
    temperature: float,
    device: torch.device,
) -> str:
    if max_new_chars <= 0:
        return prompt
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    was_training = model.training
    model.eval()
    ids = vocab.encode(prompt, device=device).tolist()
    for _ in range(max_new_chars):
        context = torch.tensor([ids[-seq_len:]], dtype=torch.long, device=device)
        logits = model(context, pad_to_length=seq_len).logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        ids.append(int(torch.multinomial(probs, num_samples=1).item()))
    if was_training:
        model.train()
    return vocab.decode(ids)


def generate_samples(
    model: FastHymbaCharLM,
    vocab: CharVocabulary,
    prompts: list[str],
    *,
    max_new_chars: int,
    seq_len: int,
    temperature: float,
    device: torch.device,
) -> str:
    parts = []
    for index, prompt in enumerate(prompts, start=1):
        sample = generate_sample(
            model,
            vocab,
            prompt,
            max_new_chars=max_new_chars,
            seq_len=seq_len,
            temperature=temperature,
            device=device,
        )
        parts.append(f"--- sample {index}: {prompt!r} ---\n{sample}")
    return "\n\n".join(parts)


def save_checkpoint(
    path: Path,
    *,
    model: FastHymbaCharLM,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    vocab: CharVocabulary,
    step: int,
    final_eval: dict[str, float] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "vocab_chars": list(vocab.chars),
        "step": step,
    }
    if final_eval is not None:
        payload["final_eval"] = final_eval
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_fast_hymba_no_xl"
    run_dir = ROOT / "experiments" / "mods" / args.mod_id / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    config = TrainConfig(
        data_path=args.data_path,
        mod_id=args.mod_id,
        run_name=run_name,
        seed=args.seed,
        device=args.device,
        architecture=args.architecture,
        resume_checkpoint=args.resume_checkpoint,
        vocab_checkpoint=args.vocab_checkpoint,
        replaced_oov_chars={},
        allow_vocab_expansion=args.allow_vocab_expansion,
        expanded_vocab_chars=[],
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        layers=args.layers,
        ssm_kernel_size=args.ssm_kernel_size,
        state_branch=args.state_branch,
        num_scales=args.num_scales,
        scale_kernel_size=args.scale_kernel_size,
        scale_block_mlp=args.scale_block_mlp,
        global_mlp=args.global_mlp,
        loss_kind=args.loss_kind,
        digit_loss_weight=args.digit_loss_weight,
        nondigit_loss_weight=args.nondigit_loss_weight,
        loss_prediction_alpha=args.loss_prediction_alpha,
        set_clock_gate=args.set_clock_gate,
        freeze_clock_gates=args.freeze_clock_gates,
        lr=args.lr,
        plateau_recovery=args.plateau_recovery,
        plateau_patience_steps=args.plateau_patience_steps,
        plateau_lr_factor=args.plateau_lr_factor,
        plateau_min_lr=args.plateau_min_lr,
        plateau_max_lr=args.plateau_max_lr,
        plateau_min_delta=args.plateau_min_delta,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        sample_every=args.sample_every,
        sample_prompt=args.sample_prompt,
        sample_prompts=args.sample_prompts,
        sample_prompt_file=args.sample_prompt_file,
        sample_chars=args.sample_chars,
        sample_temperature=args.sample_temperature,
        train_only_layers=args.train_only_layers,
        unfreeze_all_after_step=args.unfreeze_all_after_step,
    )
    write_json(run_dir / "run_config.json", asdict(config))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    if args.plateau_patience_steps <= 0:
        raise ValueError("--plateau-patience-steps must be positive")
    if not 0.0 < args.plateau_lr_factor <= 1.0:
        raise ValueError("--plateau-lr-factor must be in (0, 1]")
    if args.plateau_min_lr <= 0:
        raise ValueError("--plateau-min-lr must be positive")
    if args.plateau_max_lr is not None and args.plateau_max_lr < args.plateau_min_lr:
        raise ValueError("--plateau-max-lr must be >= --plateau-min-lr")
    if args.plateau_min_delta < 0:
        raise ValueError("--plateau-min-delta must be nonnegative")

    text = (ROOT / args.data_path).read_text(encoding="utf-8")
    vocab_checkpoint = args.vocab_checkpoint or args.resume_checkpoint
    if vocab_checkpoint is not None:
        vocab_checkpoint_path = Path(vocab_checkpoint)
        if not vocab_checkpoint_path.is_absolute():
            vocab_checkpoint_path = ROOT / vocab_checkpoint_path
        vocab_payload = torch.load(vocab_checkpoint_path, map_location="cpu", weights_only=False)
        vocab = CharVocabulary.from_chars(vocab_payload["vocab_chars"])
    else:
        vocab = CharVocabulary.from_text(text)
    missing_chars = sorted(set(text) - set(vocab.chars))
    replaced_oov_chars: dict[str, int] = {}
    expanded_vocab_chars: list[str] = []
    if missing_chars:
        if args.allow_vocab_expansion:
            expanded_vocab_chars = missing_chars
            vocab = CharVocabulary.from_chars(tuple(list(vocab.chars) + expanded_vocab_chars))
        elif args.replace_oov_with is None:
            raise ValueError(
                f"dataset contains chars not in checkpoint vocabulary: {missing_chars!r}; "
                "set --replace-oov-with or --allow-vocab-expansion for continuation runs"
            )
        else:
            if args.replace_oov_with not in vocab.stoi:
                raise ValueError(f"--replace-oov-with character is not in vocabulary: {args.replace_oov_with!r}")
            for ch in missing_chars:
                replaced_oov_chars[ch] = text.count(ch)
                text = text.replace(ch, args.replace_oov_with)
    if replaced_oov_chars or expanded_vocab_chars:
        config_payload = asdict(config)
        config_payload["replaced_oov_chars"] = replaced_oov_chars
        config_payload["expanded_vocab_chars"] = expanded_vocab_chars
        write_json(run_dir / "run_config.json", config_payload)
    encoded = vocab.encode(text)
    split = int(0.9 * encoded.numel())
    train_data = encoded[:split]
    val_data = encoded[split:]
    fallback_prompt = text[split : split + min(args.seq_len, 80)]
    sample_prompts = resolve_sample_prompts(args, fallback_prompt=fallback_prompt)
    for prompt in sample_prompts:
        vocab.encode(prompt)

    device = torch.device(args.device)
    if args.architecture == "fast_hymba":
        model = FastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.layers,
                ssm_kernel_size=args.ssm_kernel_size,
                state_branch=args.state_branch,
            )
        ).to(device)
        model_source = "models/CST/lm.py:FastHymbaCharLM"
        inner_hymba_blocks = args.layers
    elif args.architecture == "loss_query_hymba":
        model = LossQueryFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.layers,
                ssm_kernel_size=args.ssm_kernel_size,
                state_branch=args.state_branch,
            )
        ).to(device)
        model_source = "models/CST/lm.py:LossQueryFastHymbaCharLM"
        inner_hymba_blocks = args.layers
    elif args.architecture == "current_clock_hymba":
        model = CurrentClockFusionFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.layers,
                ssm_kernel_size=args.ssm_kernel_size,
                state_branch=args.state_branch,
            )
        ).to(device)
        model_source = "models/CST/lm.py:CurrentClockFusionFastHymbaCharLM"
        inner_hymba_blocks = args.layers
    elif args.architecture == "ephemeral_clock_hymba":
        model = EphemeralClockSidecarFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.layers,
                ssm_kernel_size=args.ssm_kernel_size,
                state_branch=args.state_branch,
            )
        ).to(device)
        model_source = "models/CST/lm.py:EphemeralClockSidecarFastHymbaCharLM"
        inner_hymba_blocks = args.layers
    elif args.architecture == "clock_conditioned_hymba":
        model = ClockConditionedFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.layers,
                ssm_kernel_size=args.ssm_kernel_size,
                state_branch=args.state_branch,
            )
        ).to(device)
        model_source = "models/CST/lm.py:ClockConditionedFastHymbaCharLM"
        inner_hymba_blocks = args.layers
    elif args.architecture == "alternating_clock_hymba":
        model = AlternatingClockTokenFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.layers,
                ssm_kernel_size=args.ssm_kernel_size,
                state_branch=args.state_branch,
            )
        ).to(device)
        model_source = "models/CST/lm.py:AlternatingClockTokenFastHymbaCharLM"
        inner_hymba_blocks = args.layers
    elif args.architecture == "isolated_alternating_clock_hymba":
        model = IsolatedAlternatingClockTokenFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.layers,
                ssm_kernel_size=args.ssm_kernel_size,
                state_branch=args.state_branch,
            )
        ).to(device)
        model_source = "models/CST/lm.py:IsolatedAlternatingClockTokenFastHymbaCharLM"
        inner_hymba_blocks = args.layers
    elif args.architecture == "typed_isolated_alternating_clock_hymba":
        model = TypedIsolatedAlternatingClockTokenFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.layers,
                ssm_kernel_size=args.ssm_kernel_size,
                state_branch=args.state_branch,
            )
        ).to(device)
        model_source = "models/CST/lm.py:TypedIsolatedAlternatingClockTokenFastHymbaCharLM"
        inner_hymba_blocks = args.layers
    elif args.architecture == "strict_alternating_clock_hymba":
        model = StrictAlternatingClockTokenFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.layers,
                ssm_kernel_size=args.ssm_kernel_size,
                state_branch=args.state_branch,
            )
        ).to(device)
        model_source = "models/CST/lm.py:StrictAlternatingClockTokenFastHymbaCharLM"
        inner_hymba_blocks = args.layers
    elif args.architecture == "efficient_strict_alternating_clock_hymba":
        model = EfficientStrictAlternatingClockTokenFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.layers,
                ssm_kernel_size=args.ssm_kernel_size,
                state_branch=args.state_branch,
            )
        ).to(device)
        model_source = "models/CST/lm.py:EfficientStrictAlternatingClockTokenFastHymbaCharLM"
        inner_hymba_blocks = args.layers
    elif args.architecture == "previous_clock_hymba":
        model = PreviousClockConditionedFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.layers,
                ssm_kernel_size=args.ssm_kernel_size,
                state_branch=args.state_branch,
            )
        ).to(device)
        model_source = "models/CST/lm.py:PreviousClockConditionedFastHymbaCharLM"
        inner_hymba_blocks = args.layers
    elif args.architecture == "two_side_hymba":
        if args.layers % 2 != 0:
            raise ValueError("two_side_hymba requires an even --layers value")
        model = TwoSideHymbaCharLM(
            TwoSideHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                side_layers=args.layers // 2,
                ssm_kernel_size=args.ssm_kernel_size,
                state_branch=args.state_branch,
            )
        ).to(device)
        model_source = "models/CST/lm.py:TwoSideHymbaCharLM"
        inner_hymba_blocks = args.layers
    else:
        model = MSHymbaCharLM(
            MSHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.layers,
                num_scales=args.num_scales,
                scale_kernel_size=args.scale_kernel_size,
                ssm_kernel_size=args.ssm_kernel_size,
                scale_block_mlp=args.scale_block_mlp,
                global_mlp=args.global_mlp,
            )
        ).to(device)
        model_source = "models/CST/lm.py:MSHymbaCharLM"
        inner_hymba_blocks = args.layers * (args.num_scales + 2)
    resumed_from = None
    loaded_optimizer_state = None
    grafted_resume = False
    resume_load_info: dict[str, list[str]] = {"missing_keys": [], "unexpected_keys": []}
    if args.resume_checkpoint is not None:
        checkpoint_path = Path(args.resume_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        grafted_resume, resume_load_info = load_resume_state(
            model,
            checkpoint,
            target_architecture=args.architecture,
        )
        if not grafted_resume and not expanded_vocab_chars and not args.no_resume_optimizer and "optimizer_state_dict" in checkpoint:
            loaded_optimizer_state = checkpoint["optimizer_state_dict"]
        resumed_from = str(checkpoint_path)

    train_only_layers = parse_layer_numbers(args.train_only_layers, num_layers=args.layers)
    if args.unfreeze_all_after_step is not None and args.unfreeze_all_after_step < 0:
        raise ValueError("--unfreeze-all-after-step must be nonnegative")
    set_fast_hymba_trainable_layers(model, train_only_layers)
    configured_clock_gates = configure_clock_gates(
        model,
        set_value=args.set_clock_gate,
        freeze=args.freeze_clock_gates,
    )
    optimizer = make_optimizer(model, lr=args.lr)
    if loaded_optimizer_state is not None:
        optimizer.load_state_dict(loaded_optimizer_state)
    if args.override_resume_lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = args.override_resume_lr
    plateau_max_lr = args.plateau_max_lr if args.plateau_max_lr is not None else optimizer_lr(optimizer)
    plateau_restore_lr = clamp_lr(optimizer_lr(optimizer), min_lr=args.plateau_min_lr, max_lr=plateau_max_lr)
    set_optimizer_lr(optimizer, plateau_restore_lr)
    plateau_stage = "tracking"
    plateau_stage_start_step = 0
    best_eval_loss = float("inf")
    best_eval_step = 0
    best_checkpoint = run_dir / "checkpoint_best.pt"
    plateau_event_count = 0

    status = {
        "status": "running",
        "run_dir": str(run_dir),
        "vocab_size": vocab.size,
        "train_chars": int(train_data.numel()),
        "val_chars": int(val_data.numel()),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "model_source": model_source,
        "architecture": args.architecture,
        "xl_memory": False,
        "compression": False,
        "state_branch": "MSHymbaBlock" if args.architecture == "ms_hymba" else "FastCausalConvBranch",
        "state_branch_mode": args.state_branch,
        "stride_channels": [[1, args.d_model // 2], [2, args.d_model // 2]] if args.state_branch == "multistride_1_2" else [[1, args.d_model]],
        "num_scales": args.num_scales if args.architecture == "ms_hymba" else None,
        "scale_outputs": args.num_scales + 2 if args.architecture == "ms_hymba" else None,
        "inner_hymba_blocks": inner_hymba_blocks,
        "side_layers": args.layers // 2 if args.architecture == "two_side_hymba" else None,
        "cross_skip_pairs": args.layers // 2 if args.architecture == "two_side_hymba" else None,
        "scale_block_mlp": args.scale_block_mlp if args.architecture == "ms_hymba" else None,
        "global_mlp": args.global_mlp if args.architecture == "ms_hymba" else None,
        "uses_recurrent_causal_ssm_loop": False,
        "loss_kind": args.loss_kind,
        "digit_loss_weight": args.digit_loss_weight,
        "nondigit_loss_weight": args.nondigit_loss_weight,
        "loss_prediction_alpha": args.loss_prediction_alpha,
        "configured_clock_gates": configured_clock_gates,
        "freeze_clock_gates": args.freeze_clock_gates,
        "resumed_from": resumed_from,
        "grafted_resume": grafted_resume,
        "resume_load_info": resume_load_info,
        "vocab_checkpoint": str(vocab_checkpoint_path) if vocab_checkpoint is not None else None,
        "replaced_oov_chars": replaced_oov_chars,
        "allow_vocab_expansion": args.allow_vocab_expansion,
        "expanded_vocab_chars": expanded_vocab_chars,
        "train_only_layers": train_only_layers,
        "unfreeze_all_after_step": args.unfreeze_all_after_step,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "plateau_recovery": {
            "enabled": args.plateau_recovery,
            "patience_steps": args.plateau_patience_steps,
            "lr_factor": args.plateau_lr_factor,
            "min_lr": args.plateau_min_lr,
            "max_lr": plateau_max_lr,
            "min_delta": args.plateau_min_delta,
            "stage": plateau_stage,
            "stage_start_step": plateau_stage_start_step,
            "best_eval_loss": None,
            "best_eval_step": None,
            "current_lr": optimizer_lr(optimizer),
            "event_count": plateau_event_count,
        },
        "last_checkpoint": None,
        "last_checkpoint_step": None,
        "last_sample": None,
        "last_sample_step": None,
    }
    write_json(run_dir / "run_status.json", status)
    write_json(run_dir / "vocab_summary.json", {"vocab_size": vocab.size, "chars": list(vocab.chars)})
    (run_dir / "sample_prompts.txt").write_text("\n\n".join(sample_prompts), encoding="utf-8")

    held_batch = make_batch(val_data, args.batch_size, args.seq_len, device)
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        initial_eval = evaluate(
            model,
            held_batch,
            vocab,
            loss_kind=args.loss_kind,
            digit_weight=args.digit_loss_weight,
            nondigit_weight=args.nondigit_loss_weight,
            loss_prediction_alpha=args.loss_prediction_alpha,
        )
        metrics_file.write(json.dumps({"step": 0, **initial_eval, "lr": optimizer_lr(optimizer)}) + "\n")
        best_eval_loss = initial_eval["eval_mode_loss"]
        save_checkpoint(
            best_checkpoint,
            model=model,
            optimizer=optimizer,
            config=config,
            vocab=vocab,
            step=0,
            final_eval=initial_eval,
        )
        status["plateau_recovery"].update(
            {
                "best_eval_loss": best_eval_loss,
                "best_eval_step": best_eval_step,
                "best_checkpoint": str(best_checkpoint),
            }
        )
        write_json(run_dir / "run_status.json", status)

        for step in range(1, args.steps + 1):
            if (
                args.unfreeze_all_after_step is not None
                and train_only_layers is not None
                and step == args.unfreeze_all_after_step + 1
            ):
                set_fast_hymba_trainable_layers(model, None)
                optimizer = make_optimizer(model, lr=args.override_resume_lr or args.lr)
                plateau_restore_lr = clamp_lr(optimizer_lr(optimizer), min_lr=args.plateau_min_lr, max_lr=plateau_max_lr)
                set_optimizer_lr(optimizer, plateau_restore_lr)
                plateau_stage = "tracking"
                plateau_stage_start_step = step
                train_only_layers = None
                status.update(
                    {
                        "unfroze_all_at_step": step,
                        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                    }
                )
                write_json(run_dir / "run_status.json", status)
            model.train()
            batch = make_batch(train_data, args.batch_size, args.seq_len, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model_loss(
                model,
                batch,
                vocab,
                loss_kind=args.loss_kind,
                digit_weight=args.digit_loss_weight,
                nondigit_weight=args.nondigit_loss_weight,
                loss_prediction_alpha=args.loss_prediction_alpha,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            row = {
                "step": step,
                "train_loss": float(loss.item()),
                "grad_norm": float(grad_norm.item()),
                "lr": optimizer_lr(optimizer),
            }
            if step == args.steps or step % args.eval_every == 0:
                row.update(
                    evaluate(
                        model,
                        held_batch,
                        vocab,
                        loss_kind=args.loss_kind,
                        digit_weight=args.digit_loss_weight,
                        nondigit_weight=args.nondigit_loss_weight,
                        loss_prediction_alpha=args.loss_prediction_alpha,
                    )
                )
                eval_loss = row["eval_mode_loss"]
                improved = eval_loss < best_eval_loss - args.plateau_min_delta
                if improved:
                    best_eval_loss = eval_loss
                    best_eval_step = step
                    save_checkpoint(
                        best_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        vocab=vocab,
                        step=step,
                        final_eval={
                            key: row[key]
                            for key in (
                                "train_mode_loss",
                                "eval_mode_loss",
                                "eval_unweighted_loss",
                                "loss_delta",
                                "eval_next_char_accuracy",
                            )
                            if key in row
                        },
                    )
                    row["plateau_best_improved"] = True
                    if args.plateau_recovery and plateau_stage == "lr_reduced":
                        restored_lr = clamp_lr(plateau_restore_lr, min_lr=args.plateau_min_lr, max_lr=plateau_max_lr)
                        set_optimizer_lr(optimizer, restored_lr)
                        plateau_stage = "tracking"
                        plateau_stage_start_step = step
                        row["plateau_action"] = "restore_lr"
                        row["plateau_lr"] = restored_lr
                    elif args.plateau_recovery:
                        plateau_stage = "tracking"
                        plateau_stage_start_step = step
                elif args.plateau_recovery and step - plateau_stage_start_step >= args.plateau_patience_steps:
                    plateau_event_count += 1
                    if plateau_stage == "tracking":
                        plateau_restore_lr = optimizer_lr(optimizer)
                        reduced_lr = clamp_lr(
                            plateau_restore_lr * args.plateau_lr_factor,
                            min_lr=args.plateau_min_lr,
                            max_lr=plateau_max_lr,
                        )
                        set_optimizer_lr(optimizer, reduced_lr)
                        plateau_stage = "lr_reduced"
                        plateau_stage_start_step = step
                        row["plateau_action"] = "reduce_lr"
                        row["plateau_lr"] = reduced_lr
                    elif plateau_stage == "lr_reduced":
                        optimizer = make_optimizer(model, lr=optimizer_lr(optimizer))
                        plateau_stage = "adam_reset"
                        plateau_stage_start_step = step
                        row["plateau_action"] = "reset_adam"
                        row["plateau_lr"] = optimizer_lr(optimizer)
                    else:
                        best_payload = torch.load(best_checkpoint, map_location=device, weights_only=False)
                        model.load_state_dict(best_payload["model_state_dict"])
                        optimizer = make_optimizer(model, lr=plateau_restore_lr)
                        set_optimizer_lr(
                            optimizer,
                            clamp_lr(plateau_restore_lr, min_lr=args.plateau_min_lr, max_lr=plateau_max_lr),
                        )
                        plateau_stage = "tracking"
                        plateau_stage_start_step = step
                        row["plateau_action"] = "reload_best_fresh_adam"
                        row["plateau_lr"] = optimizer_lr(optimizer)
                        row["plateau_reloaded_best_step"] = best_eval_step
                status["plateau_recovery"].update(
                    {
                        "stage": plateau_stage,
                        "stage_start_step": plateau_stage_start_step,
                        "best_eval_loss": best_eval_loss,
                        "best_eval_step": best_eval_step,
                        "best_checkpoint": str(best_checkpoint),
                        "current_lr": optimizer_lr(optimizer),
                        "event_count": plateau_event_count,
                        "steps_since_best": step - best_eval_step,
                    }
                )
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            if args.checkpoint_every > 0 and (step % args.checkpoint_every == 0 or step == args.steps):
                latest_checkpoint = run_dir / "checkpoint_latest.pt"
                step_checkpoint = run_dir / f"checkpoint_step_{step:06d}.pt"
                final_eval = {
                    key: row[key]
                    for key in (
                        "train_mode_loss",
                        "eval_mode_loss",
                        "eval_unweighted_loss",
                        "loss_delta",
                        "eval_next_char_accuracy",
                    )
                    if key in row
                }
                save_checkpoint(
                    step_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    vocab=vocab,
                    step=step,
                    final_eval=final_eval,
                )
                save_checkpoint(
                    latest_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    vocab=vocab,
                    step=step,
                    final_eval=final_eval,
                )
                status.update(
                    {
                        "status": "running",
                        "last_checkpoint": str(latest_checkpoint),
                        "last_checkpoint_step": step,
                        "last_logged_train_loss": row["train_loss"],
                    }
                )
                write_json(run_dir / "run_status.json", status)
            if args.sample_every > 0 and (step % args.sample_every == 0 or step == args.steps):
                sample_path = run_dir / f"sample_step_{step:04d}.txt"
                sample = generate_samples(
                    model,
                    vocab,
                    sample_prompts,
                    max_new_chars=args.sample_chars,
                    seq_len=args.seq_len,
                    temperature=args.sample_temperature,
                    device=device,
                )
                sample_path.write_text(sample, encoding="utf-8")
                status.update(
                    {
                        "status": "running",
                        "last_sample": str(sample_path),
                        "last_sample_step": step,
                    }
                )
                write_json(run_dir / "run_status.json", status)
                print(f"\n--- sample step {step} ---\n{sample}\n--- end sample step {step} ---", flush=True)

    final_eval = evaluate(
        model,
        held_batch,
        vocab,
        loss_kind=args.loss_kind,
        digit_weight=args.digit_loss_weight,
        nondigit_weight=args.nondigit_loss_weight,
        loss_prediction_alpha=args.loss_prediction_alpha,
    )
    sample = generate_samples(
        model,
        vocab,
        sample_prompts,
        max_new_chars=args.sample_chars,
        seq_len=args.seq_len,
        temperature=args.sample_temperature,
        device=device,
    )
    (run_dir / f"sample_step_{args.steps:04d}.txt").write_text(sample, encoding="utf-8")
    status.update(
        {
            "status": "completed",
            "final_eval": final_eval,
            "last_checkpoint": str(run_dir / "checkpoint_final.pt"),
            "last_checkpoint_step": args.steps,
        }
    )
    write_json(run_dir / "run_status.json", status)
    save_checkpoint(
        run_dir / "checkpoint_final.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        vocab=vocab,
        step=args.steps,
        final_eval=final_eval,
    )

    print(json.dumps({"run_dir": str(run_dir), "vocab_size": vocab.size, "final_eval": final_eval}, indent=2))


if __name__ == "__main__":
    main()
