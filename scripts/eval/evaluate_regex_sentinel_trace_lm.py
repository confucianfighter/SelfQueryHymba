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

from scripts.eval.evaluate_addition_trace_lm import load_model, model_logits  # noqa: E402


IL_RE = re.compile(r"^IL:\n(.+?)\n\nTemplate:", re.MULTILINE | re.DOTALL)
TEMPLATE_RE = re.compile(r"^Template:\n([^\n]+)", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sentinel-wrapped regex IL/template generation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="data/sentinel_regex_v7_dream_queries_120k.txt")
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9393)
    parser.add_argument("--seq-len", type=int, default=384)
    parser.add_argument("--max-new-chars", type=int, default=360)
    parser.add_argument("--sentinel", default="<END>")
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show-failures", type=int, default=10)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def split_sentinel_examples(text: str, sentinel: str) -> list[str]:
    parts = text.split(sentinel)
    examples = [part.strip() + "\n" + sentinel for part in parts[:-1] if part.strip()]
    trailing = parts[-1].strip()
    if trailing:
        raise ValueError(f"trailing text after final sentinel: {trailing[:80]!r}")
    if not examples:
        raise ValueError(f"no examples ending with {sentinel!r}")
    return examples


def choose_split(examples: list[str], split_name: str) -> list[str]:
    split = int(0.9 * len(examples))
    if split_name == "train":
        return examples[:split]
    if split_name == "val":
        return examples[split:]
    return examples


def prompt_and_expected(example: str, sentinel: str) -> tuple[str, str]:
    marker = "\nOutput:\n"
    if marker not in example:
        raise ValueError(f"example has no Output marker: {example[:120]!r}")
    prompt, rest = example.split(marker, 1)
    expected = rest.rsplit("\n" + sentinel, 1)[0].strip()
    return prompt + marker, expected


@torch.no_grad()
def generate_greedy(
    model,
    vocab,
    prompt: str,
    *,
    seq_len: int,
    max_new_chars: int,
    device: torch.device,
    sentinel: str,
) -> str:
    ids = vocab.encode(prompt, device=device).tolist()
    prompt_len = len(prompt)
    for _ in range(max_new_chars):
        context = torch.tensor([ids[-seq_len:]], dtype=torch.long, device=device)
        logits = model_logits(model, context, pad_to_length=seq_len)[:, -1, :]
        ids.append(int(logits.argmax(dim=-1).item()))
        text = vocab.decode(ids)
        suffix = text[prompt_len:]
        if sentinel in suffix:
            break
        if re.search(r"\nTemplate:\n[^\n]+\n", suffix):
            break
    return vocab.decode(ids)


def strip_generated_output(generated: str, prompt: str, sentinel: str) -> str:
    suffix = generated[len(prompt) :]
    if sentinel in suffix:
        suffix = suffix.split(sentinel, 1)[0]
    return suffix.strip()


def extract_il(text: str) -> str | None:
    match = IL_RE.search(text)
    return match.group(1).strip() if match else None


def extract_template(text: str) -> str | None:
    match = TEMPLATE_RE.search(text)
    return match.group(1).strip() if match else None


def expand_template(template: str | None, prompt: str) -> str | None:
    if template is None:
        return None
    refs = re.findall(r'"([^"\n]*)"', prompt)
    expanded = template
    for idx, value in enumerate(refs):
        expanded = expanded.replace(f"<{idx}>", re.escape(value))
    return expanded


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = resolve_path(args.checkpoint)
    data_path = resolve_path(args.data_path)
    model, vocab, _config = load_model(checkpoint, device)

    all_examples = split_sentinel_examples(data_path.read_text(encoding="utf-8"), args.sentinel)
    pool = choose_split(all_examples, args.split)
    if args.examples > len(pool):
        raise ValueError(f"requested {args.examples} examples, but split has {len(pool)}")
    rng = random.Random(args.seed)
    sampled = rng.sample(pool, args.examples)

    exact_output = 0
    il_exact = 0
    template_exact = 0
    expanded_exact = 0
    parsed_il = 0
    parsed_template = 0
    failures: list[dict[str, object]] = []

    for index, example in enumerate(sampled, start=1):
        prompt, expected = prompt_and_expected(example, args.sentinel)
        generated = generate_greedy(
            model,
            vocab,
            prompt,
            seq_len=args.seq_len,
            max_new_chars=args.max_new_chars,
            device=device,
            sentinel=args.sentinel,
        )
        predicted = strip_generated_output(generated, prompt, args.sentinel)
        expected_il = extract_il(expected)
        predicted_il = extract_il(predicted)
        expected_template = extract_template(expected)
        predicted_template = extract_template(predicted)
        output_ok = predicted == expected
        il_ok = predicted_il == expected_il
        template_ok = predicted_template == expected_template
        expanded_ok = expand_template(predicted_template, prompt) == expand_template(expected_template, prompt)
        exact_output += int(output_ok)
        il_exact += int(il_ok)
        template_exact += int(template_ok)
        expanded_exact += int(expanded_ok)
        parsed_il += int(predicted_il is not None)
        parsed_template += int(predicted_template is not None)
        if (not output_ok or not il_ok or not template_ok) and len(failures) < args.show_failures:
            failures.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "expected": expected,
                    "predicted": predicted,
                    "expected_il": expected_il,
                    "predicted_il": predicted_il,
                    "expected_template": expected_template,
                    "predicted_template": predicted_template,
                    "expected_regex": expand_template(expected_template, prompt),
                    "predicted_regex": expand_template(predicted_template, prompt),
                }
            )

    result = {
        "checkpoint": str(checkpoint),
        "data_path": str(data_path),
        "split": args.split,
        "available_examples": len(pool),
        "examples": args.examples,
        "exact_output": exact_output,
        "exact_output_accuracy": exact_output / args.examples,
        "il_exact": il_exact,
        "il_exact_accuracy": il_exact / args.examples,
        "template_exact": template_exact,
        "template_exact_accuracy": template_exact / args.examples,
        "expanded_regex_exact": expanded_exact,
        "expanded_regex_exact_accuracy": expanded_exact / args.examples,
        "parsed_il": parsed_il,
        "parsed_il_accuracy": parsed_il / args.examples,
        "parsed_template": parsed_template,
        "parsed_template_accuracy": parsed_template / args.examples,
        "failures_shown": failures,
    }
    if args.output is not None:
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
