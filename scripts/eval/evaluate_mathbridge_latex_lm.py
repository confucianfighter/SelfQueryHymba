from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.CST import CharVocabulary  # noqa: E402
from scripts.eval.evaluate_addition_trace_lm import load_model, model_logits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MathBridge spoken-English to LaTeX generations.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--examples-file", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task", default="mathbridge_latex")
    parser.add_argument("--sentinel", default="<END>")
    parser.add_argument("--with-il", action="store_true", help="Prompt at the IL block and extract the generated Output block.")
    parser.add_argument("--max-new-chars", type=int, default=160)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show-failures", type=int, default=10)
    return parser.parse_args()


def read_examples(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prompt_for(spoken: str, *, task: str, with_il: bool) -> str:
    suffix = "IL:\n" if with_il else "Output:\n"
    return f"Task: {task}\nInput:\n{spoken}\n\n{suffix}"


def normalize_latex(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def extract_answer(generated: str, *, with_il: bool) -> str:
    if not with_il:
        return generated.strip()
    if "\nOutput:\n" in generated:
        return generated.split("\nOutput:\n", 1)[1].strip()
    if "Output:\n" in generated:
        return generated.split("Output:\n", 1)[1].strip()
    return generated.strip()


@torch.no_grad()
def generate_greedy(
    model,
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
        logits = model_logits(model, context, pad_to_length=seq_len)[:, -1, :]
        ids.append(int(logits.argmax(dim=-1).item()))
        generated = vocab.decode(ids[prompt_len:])
        if sentinel in generated:
            break
        if "\n\nTask:" in generated:
            break
    generated = vocab.decode(ids[prompt_len:])
    if sentinel in generated:
        generated = generated.split(sentinel, 1)[0]
    if "\n\nTask:" in generated:
        generated = generated.split("\n\nTask:", 1)[0]
    return generated.strip()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    examples_file = args.examples_file if args.examples_file.is_absolute() else ROOT / args.examples_file
    output = args.output if args.output is None or args.output.is_absolute() else ROOT / args.output

    device = torch.device(args.device)
    model, vocab, config = load_model(checkpoint, device)
    seq_len = int(config["seq_len"])
    examples = read_examples(examples_file)

    rows = []
    skipped_oov = 0
    for example in examples:
        if len(rows) >= args.examples:
            break
        spoken = str(example["input"])
        expected = str(example["output"])
        prompt = prompt_for(spoken, task=args.task, with_il=args.with_il)
        try:
            vocab.encode(prompt)
        except ValueError:
            skipped_oov += 1
            continue
        generated = generate_greedy(
            model,
            vocab,
            prompt,
            seq_len=seq_len,
            max_new_chars=args.max_new_chars,
            sentinel=args.sentinel,
            device=device,
        )
        answer = extract_answer(generated, with_il=args.with_il)
        rows.append(
            {
                "row_idx": example.get("row_idx"),
                "input": spoken,
                "expected": expected,
                "generated": generated,
                "answer": answer,
                "exact": answer == expected,
                "normalized_exact": normalize_latex(answer) == normalize_latex(expected),
            }
        )

    exact = sum(row["exact"] for row in rows)
    normalized_exact = sum(row["normalized_exact"] for row in rows)
    failures = [row for row in rows if not row["normalized_exact"]]
    summary = {
        "checkpoint": str(checkpoint),
        "examples_file": str(examples_file),
        "examples": len(rows),
        "skipped_oov_prompts": skipped_oov,
        "exact": exact,
        "exact_accuracy": exact / len(rows) if rows else 0.0,
        "normalized_exact": normalized_exact,
        "normalized_exact_accuracy": normalized_exact / len(rows) if rows else 0.0,
        "failures_shown": failures[: args.show_failures],
    }
    payload = {"summary": summary, "rows": rows}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
