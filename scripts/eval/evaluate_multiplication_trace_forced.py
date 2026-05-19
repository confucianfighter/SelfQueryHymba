from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.prepare_multiplication_traces import (  # noqa: E402
    MultiplicationTraceConfig,
    format_multiplication_trace,
    random_examples,
)
from scripts.eval.evaluate_addition_trace_forced import answer_span, pad_sequences, replace_oov  # noqa: E402
from scripts.eval.evaluate_addition_trace_lm import load_model, model_logits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast teacher-forced multiplication answer-only evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--replace-oov-with", default=None)
    parser.add_argument("--task-prefix", default=None, help="Optional leading task line, e.g. 'Task: multiplication_answer'.")
    parser.add_argument("--shared-output-format", action="store_true")
    parser.add_argument("--question-only", action="store_true", help="Evaluate the bare question-plus-answer math format.")
    parser.add_argument("--sentinel", default=None)
    return parser.parse_args()


def tag_trace(
    trace: str,
    task_prefix: str | None,
    *,
    shared_output_format: bool = False,
    question_only: bool = False,
    sentinel: str | None = None,
) -> str:
    if question_only:
        tagged = trace
    elif task_prefix is None:
        tagged = trace
    else:
        lines = trace.splitlines()
        if shared_output_format:
            tagged = task_prefix.rstrip("\n") + "\nInput:\n" + lines[0] + "\n\nOutput:\n" + "\n".join(lines[1:])
        else:
            tagged = task_prefix.rstrip("\n") + "\nInput:\n" + lines[0] + "\nTrace:\n" + "\n".join(lines[1:])
    if sentinel is not None:
        tagged += "\n" + sentinel
    return tagged


@torch.no_grad()
def evaluate_batch(model, vocab, traces: list[str], device: torch.device, pad_id: int) -> list[dict]:
    encoded = [vocab.encode(replace_oov(trace + "\n", vocab, getattr(model, "_replace_oov_with", None))) for trace in traces]
    batch = pad_sequences(encoded, pad_id).to(device)
    logits = model_logits(model, batch)
    predictions = logits[:, :-1].argmax(dim=-1).cpu()
    targets = batch[:, 1:].cpu()
    rows = []
    for idx, trace in enumerate(traces):
        length = encoded[idx].numel()
        pred = predictions[idx, : length - 1]
        target = targets[idx, : length - 1]
        full_exact = bool(torch.equal(pred, target))
        start, end = answer_span(trace)
        answer_positions = torch.arange(start - 1, end - 1)
        answer_exact = bool(torch.equal(pred[answer_positions], target[answer_positions]))
        rows.append(
            {
                "trace": trace,
                "full_exact": full_exact,
                "answer_exact": answer_exact,
                "token_accuracy": float((pred == target).float().mean().item()),
                "answer": trace[start:end],
                "predicted_answer": vocab.decode(pred[answer_positions]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    model, vocab, _config = load_model(checkpoint, device)
    model._replace_oov_with = args.replace_oov_with
    pad_id = vocab.stoi["\n"]

    pairs = random_examples(
        MultiplicationTraceConfig(
            examples=args.examples,
            seed=args.seed,
            min_digits=2,
            max_digits=2,
            exhaustive=False,
            include_zero=False,
            trace_format="answer_only",
        )
    )
    traces = [
        tag_trace(
            format_multiplication_trace(left, right),
            args.task_prefix,
            shared_output_format=args.shared_output_format,
            question_only=args.question_only,
            sentinel=args.sentinel,
        )
        for left, right in pairs[: args.examples]
    ]

    rows = []
    for start in range(0, len(traces), args.batch_size):
        rows.extend(evaluate_batch(model, vocab, traces[start : start + args.batch_size], device, pad_id))

    summary = {
        "checkpoint": str(checkpoint),
        "examples": len(rows),
        "answer_exact": sum(row["answer_exact"] for row in rows),
        "answer_exact_accuracy": sum(row["answer_exact"] for row in rows) / len(rows),
        "full_exact": sum(row["full_exact"] for row in rows),
        "full_exact_accuracy": sum(row["full_exact"] for row in rows) / len(rows),
        "mean_token_accuracy": sum(row["token_accuracy"] for row in rows) / len(rows),
        "failures_shown": [row for row in rows if not row["answer_exact"]][:10],
    }
    payload = {"summary": summary, "rows": rows}
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
