from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.CST import (
    AlternatingClockTokenFastHymbaCharLM,
    BranchedLossQueryFastHymbaCharLM,
    CharVocabulary,
    ClockConditionedFastHymbaCharLM,
    CurrentClockFusionFastHymbaCharLM,
    EphemeralClockSidecarFastHymbaCharLM,
    EfficientStrictAlternatingClockTokenFastHymbaCharLM,
    FastHymbaCharLM,
    FastHymbaCharLMConfig,
    InterleavedLogitConditionedFastHymbaCharLM,
    IsolatedAlternatingClockTokenFastHymbaCharLM,
    LayerCrossLogitConditionedFastHymbaCharLM,
    LayerCrossTokenLogitConditionedFastHymbaCharLM,
    LossContextInjectedFastHymbaCharLM,
    LossQueryFastHymbaCharLM,
    LogitConditionedFastHymbaCharLM,
    LogitConditionedFastHymbaCharLMConfig,
    MSHymbaCharLM,
    MSHymbaCharLMConfig,
    PreviousClockConditionedFastHymbaCharLM,
    PreviousLossScalarInjectedFastHymbaCharLM,
    PreviousLossScalarConditionedFastHymbaCharLM,
    StrictAlternatingClockTokenFastHymbaCharLM,
    TypedIsolatedAlternatingClockTokenFastHymbaCharLM,
    TwoSideHymbaCharLM,
    TwoSideHymbaCharLMConfig,
)
from scripts.data.prepare_addition_traces import (
    AdditionTraceConfig,
    format_addition_trace,
    format_strict_addition_trace,
    random_examples,
)


ANSWER_RE = re.compile(r"^answer: (\d+)\s*$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a char-LM checkpoint on generated addition traces.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-new-chars", type=int, default=220)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--trace-format", choices=("prose", "strict"), default="prose")
    parser.add_argument("--show-failures", type=int, default=10)
    parser.add_argument("--include-fixed-example", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    vocab = CharVocabulary.from_chars(checkpoint["vocab_chars"])
    architecture = config.get("architecture", "fast_hymba")
    activation_kwargs = {
        "projection_type": config.get("projection_type", "dense"),
        "attention_qkv_projection_type": config.get("attention_qkv_projection_type", "dense"),
        "ssm_activation_type": config.get("ssm_activation_type", "silu"),
        "block_mlp_multiplier": config.get("block_mlp_multiplier", 4),
        "block_mlp_activation_type": config.get("block_mlp_activation_type", "gelu"),
        "block_mlp_up_projection_type": config.get("block_mlp_up_projection_type", "dense"),
        "block_mlp_down_projection_type": config.get("block_mlp_down_projection_type", "dense"),
        "activation_type": config.get("activation_type", "identity"),
        "basin_min_width": config.get("basin_min_width", 0.35),
        "basin_max_width": config.get("basin_max_width", 3.0),
        "basin_floor": config.get("basin_floor", 0.08),
        "basin_zag_amp": config.get("basin_zag_amp", 0.12),
        "basin_sharpness": config.get("basin_sharpness", 2.0),
        "basin_eps": config.get("basin_eps", 1e-6),
    }
    if "teacher_checkpoint" in config:
        conditioning_mode = config.get("conditioning_mode", "concat")
        if conditioning_mode == "interleave":
            student_cls = InterleavedLogitConditionedFastHymbaCharLM
        elif conditioning_mode == "layer_cross_logits":
            student_cls = LayerCrossLogitConditionedFastHymbaCharLM
        elif conditioning_mode == "layer_cross_token_logits":
            student_cls = LayerCrossTokenLogitConditionedFastHymbaCharLM
        else:
            student_cls = LogitConditionedFastHymbaCharLM
        model = student_cls(
            LogitConditionedFastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
                teacher_logit_temperature=config.get("teacher_logit_temperature", 1.0),
            )
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        teacher_path = Path(config["teacher_checkpoint"])
        if not teacher_path.is_absolute():
            teacher_path = ROOT / teacher_path
        teacher, teacher_vocab, _teacher_config = load_model(teacher_path, device)
        if teacher_vocab.chars != vocab.chars:
            raise ValueError("conditioned student and teacher vocabularies differ")
        return ConditionedModelBundle(student=model, teacher=teacher), vocab, config
    if architecture == "fast_hymba":
        model = FastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "loss_query_hymba":
        model = LossQueryFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "branched_loss_query_hymba":
        model = BranchedLossQueryFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "loss_context_injected_hymba":
        model = LossContextInjectedFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "current_clock_hymba":
        model = CurrentClockFusionFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "ephemeral_clock_hymba":
        model = EphemeralClockSidecarFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "clock_conditioned_hymba":
        model = ClockConditionedFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "alternating_clock_hymba":
        model = AlternatingClockTokenFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "isolated_alternating_clock_hymba":
        model = IsolatedAlternatingClockTokenFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "typed_isolated_alternating_clock_hymba":
        model = TypedIsolatedAlternatingClockTokenFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "strict_alternating_clock_hymba":
        model = StrictAlternatingClockTokenFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "efficient_strict_alternating_clock_hymba":
        model = EfficientStrictAlternatingClockTokenFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "previous_clock_hymba":
        model = PreviousClockConditionedFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "previous_loss_scalar_hymba":
        model = PreviousLossScalarConditionedFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    elif architecture == "previous_loss_scalar_injected_hymba":
        model = PreviousLossScalarInjectedFastHymbaCharLM(
            FastHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
                **activation_kwargs,
            )
        ).to(device)
    elif architecture == "ms_hymba":
        model = MSHymbaCharLM(
            MSHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                num_layers=config["layers"],
                num_scales=config.get("num_scales", 1),
                scale_kernel_size=config.get("scale_kernel_size", 3),
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                scale_block_mlp=config.get("scale_block_mlp", True),
                global_mlp=config.get("global_mlp", True),
            )
        ).to(device)
    elif architecture == "two_side_hymba":
        model = TwoSideHymbaCharLM(
            TwoSideHymbaCharLMConfig(
                vocab_size=vocab.size,
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                side_layers=config["layers"] // 2,
                ssm_kernel_size=config.get("ssm_kernel_size", 3),
                state_branch=config.get("state_branch", "conv"),
            )
        ).to(device)
    else:
        raise ValueError(f"unsupported architecture: {architecture!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab, config


class ConditionedModelBundle:
    def __init__(self, *, student: LogitConditionedFastHymbaCharLM, teacher: torch.nn.Module) -> None:
        self.student = student
        self.teacher = teacher

    def eval(self):
        self.student.eval()
        self.teacher.eval()
        return self

    def conditioned_logits(self, input_ids: torch.Tensor, *, pad_to_length: int | None = None) -> torch.Tensor:
        teacher_logits = model_logits(self.teacher, input_ids, pad_to_length=pad_to_length)
        return self.student(input_ids, teacher_logits=teacher_logits, pad_to_length=pad_to_length).logits


def model_logits(model, input_ids: torch.Tensor, *, pad_to_length: int | None = None) -> torch.Tensor:
    if isinstance(model, ConditionedModelBundle):
        return model.conditioned_logits(input_ids, pad_to_length=pad_to_length)
    return model(input_ids, pad_to_length=pad_to_length).logits


@torch.no_grad()
def generate_greedy(
    model: torch.nn.Module,
    vocab: CharVocabulary,
    prompt: str,
    *,
    seq_len: int,
    max_new_chars: int,
    device: torch.device,
) -> str:
    ids = vocab.encode(prompt, device=device).tolist()
    for _ in range(max_new_chars):
        context = torch.tensor([ids[-seq_len:]], dtype=torch.long, device=device)
        logits = model_logits(model, context, pad_to_length=seq_len)[:, -1, :]
        next_id = int(logits.argmax(dim=-1).item())
        ids.append(next_id)
        text = vocab.decode(ids)
        if re.search(r"\nanswer: \d+\n", text):
            break
    return vocab.decode(ids)


def first_trace_block(text: str) -> str:
    match = re.search(r"\n\nadd \d+ \+ \d+", text)
    if match is None:
        return text.strip()
    return text[: match.start()].strip()


def parse_answer(text: str) -> int | None:
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        return None
    return int(matches[0].group(1))


def evaluate_pair(model, vocab, left: int, right: int, *, seq_len: int, max_new_chars: int, device: torch.device) -> dict:
    return evaluate_pair_with_format(
        model,
        vocab,
        left,
        right,
        trace_format="prose",
        seq_len=seq_len,
        max_new_chars=max_new_chars,
        device=device,
    )


def evaluate_pair_with_format(
    model,
    vocab,
    left: int,
    right: int,
    *,
    trace_format: str,
    seq_len: int,
    max_new_chars: int,
    device: torch.device,
) -> dict:
    if trace_format == "strict":
        prompt = f"{left} + {right}\n"
        expected = format_strict_addition_trace(left, right)
    else:
        prompt = f"add {left} + {right}\n"
        expected = format_addition_trace(left, right)
    generated = first_trace_block(
        generate_greedy(
            model,
            vocab,
            prompt,
            seq_len=seq_len,
            max_new_chars=max_new_chars,
            device=device,
        )
    )
    predicted_answer = parse_answer(generated)
    return {
        "left": left,
        "right": right,
        "expected_answer": left + right,
        "predicted_answer": predicted_answer,
        "answer_correct": predicted_answer == left + right,
        "trace_exact": generated == expected,
        "expected": expected,
        "generated": generated,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    model, vocab, config = load_model(checkpoint_path, device)
    seq_len = int(config["seq_len"])

    pairs = random_examples(
        AdditionTraceConfig(
            examples=args.examples,
            seed=args.seed,
            min_digits=1,
            max_digits=3,
            exhaustive=False,
            include_zero=False,
            trace_format=args.trace_format,
        )
    )
    if args.include_fixed_example:
        pairs = [(347, 586)] + [pair for pair in pairs if pair != (347, 586)]

    rows = [
        evaluate_pair_with_format(
            model,
            vocab,
            left,
            right,
            trace_format=args.trace_format,
            seq_len=seq_len,
            max_new_chars=args.max_new_chars,
            device=device,
        )
        for left, right in pairs[: args.examples]
    ]
    failures = [row for row in rows if not row["answer_correct"]]
    exact_failures = [row for row in rows if not row["trace_exact"]]
    summary = {
        "checkpoint": str(checkpoint_path),
        "examples": len(rows),
        "answer_correct": sum(row["answer_correct"] for row in rows),
        "answer_accuracy": sum(row["answer_correct"] for row in rows) / len(rows),
        "trace_exact": sum(row["trace_exact"] for row in rows),
        "trace_exact_accuracy": sum(row["trace_exact"] for row in rows) / len(rows),
        "failures_shown": failures[: args.show_failures],
        "exact_failures_shown": exact_failures[: args.show_failures],
    }
    payload = {"summary": summary, "rows": rows}
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
