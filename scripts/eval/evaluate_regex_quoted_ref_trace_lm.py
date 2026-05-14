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

from scripts.data.prepare_regex_quoted_ref_traces import format_example, make_example  # noqa: E402
from scripts.eval.evaluate_addition_trace_lm import load_model, model_logits  # noqa: E402


TEMPLATE_RE = re.compile(r"^Template:\n([^\n]+)", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw-quoted regex-template generation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9393)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-new-chars", type=int, default=240)
    parser.add_argument("--component-counts", default="2,2,3")
    parser.add_argument("--language-style", choices=("simple", "v2"), default="simple")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show-failures", type=int, default=10)
    parser.add_argument("--task-prefix", default=None, help="Optional leading task line, e.g. 'Task: regex_v2'.")
    return parser.parse_args()


def parse_component_counts(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


@torch.no_grad()
def generate_greedy(model, vocab, prompt: str, *, seq_len: int, max_new_chars: int, device: torch.device) -> str:
    ids = vocab.encode(prompt, device=device).tolist()
    for _ in range(max_new_chars):
        context = torch.tensor([ids[-seq_len:]], dtype=torch.long, device=device)
        logits = model_logits(model, context, pad_to_length=seq_len)[:, -1, :]
        ids.append(int(logits.argmax(dim=-1).item()))
        text = vocab.decode(ids)
        if "\n\nInput:\n" in text[len(prompt) :] or re.search(r"\nTemplate:\n[^\n]+\n", text):
            break
    return vocab.decode(ids)


def extract_template(text: str) -> str | None:
    match = TEMPLATE_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def expand_template(template: str, refs: list[str]) -> str:
    expanded = template
    for idx, value in enumerate(refs):
        expanded = expanded.replace(f"<{idx}>", re.escape(value))
    return expanded


def prompt_from_example(example: dict[str, object], task_prefix: str | None = None) -> str:
    rendered = format_example(example)
    prompt = rendered.split("\nTemplate:\n", 1)[0] + "\nTemplate:\n"
    if task_prefix is not None:
        prompt = task_prefix.rstrip("\n") + "\n" + prompt
    return prompt


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, vocab, _config = load_model(Path(args.checkpoint), device)
    rng = random.Random(args.seed)
    component_counts = parse_component_counts(args.component_counts)

    failures = []
    template_exact = 0
    expanded_exact = 0
    parsed = 0
    for _idx in range(args.examples):
        example = make_example(rng, component_choices=component_counts, language_style=args.language_style)
        prompt = prompt_from_example(example, args.task_prefix)
        generated = generate_greedy(
            model,
            vocab,
            prompt,
            seq_len=args.seq_len,
            max_new_chars=args.max_new_chars,
            device=device,
        )
        predicted_template = extract_template(generated)
        if predicted_template is not None:
            parsed += 1
        expected_template = str(example["template"])
        refs = example["refs"]
        if not isinstance(refs, list):
            raise TypeError("refs must be a list")
        exact = predicted_template == expected_template
        template_exact += int(exact)
        if predicted_template is not None:
            expanded_exact += int(expand_template(predicted_template, refs) == expand_template(expected_template, refs))
        if not exact and len(failures) < args.show_failures:
            failures.append(
                {
                    "input": example["input"],
                    "expected_plan": example["plan"],
                    "expected_template": expected_template,
                    "predicted_template": predicted_template,
                    "expected_regex": expand_template(expected_template, refs),
                    "predicted_regex": expand_template(predicted_template, refs) if predicted_template is not None else None,
                    "generated": generated,
                }
            )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "examples": args.examples,
        "template_exact": template_exact,
        "template_exact_accuracy": template_exact / args.examples,
        "expanded_regex_exact": expanded_exact,
        "expanded_regex_exact_accuracy": expanded_exact / args.examples,
        "parsed_template": parsed,
        "parsed_template_accuracy": parsed / args.examples,
        "failures_shown": failures,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
