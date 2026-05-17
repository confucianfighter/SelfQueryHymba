from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.prepare_addition_traces import AdditionTraceConfig, format_addition_trace, random_examples
from scripts.eval.evaluate_addition_trace_lm import load_model, model_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast teacher-forced addition trace evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--include-fixed-example", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replace-oov-with", default=None)
    parser.add_argument("--task-prefix", default=None, help="Optional leading task line, e.g. 'Task: addition_prose'.")
    parser.add_argument("--shared-output-format", action="store_true")
    parser.add_argument("--question-only", action="store_true", help="Evaluate the bare question-plus-trace math format.")
    parser.add_argument("--sentinel", default=None)
    return parser.parse_args()


def answer_span(trace: str) -> tuple[int, int]:
    marker = "answer: "
    start = trace.index(marker) + len(marker)
    end = trace.index("\n", start) if "\n" in trace[start:] else len(trace)
    return start, end


def pad_sequences(sequences: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(seq.numel() for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for idx, seq in enumerate(sequences):
        batch[idx, : seq.numel()] = seq
    return batch


def replace_oov(text: str, vocab, replacement: str | None) -> str:
    if replacement is None:
        return text
    if replacement not in vocab.stoi:
        raise ValueError(f"--replace-oov-with character is not in vocabulary: {replacement!r}")
    return "".join(ch if ch in vocab.stoi else replacement for ch in text)


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
        # Target positions are shifted left by one relative to trace character offsets.
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
        AdditionTraceConfig(
            examples=args.examples,
            seed=args.seed,
            min_digits=1,
            max_digits=3,
            exhaustive=False,
            include_zero=False,
            trace_format="prose",
        )
    )
    if args.include_fixed_example:
        pairs = [(347, 586)] + [pair for pair in pairs if pair != (347, 586)]
    traces = [
        tag_trace(
            format_addition_trace(left, right),
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
