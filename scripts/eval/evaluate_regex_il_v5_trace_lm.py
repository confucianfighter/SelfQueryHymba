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

from scripts.data.prepare_regex_il_v3_traces import format_example, parse_component_counts  # noqa: E402
from scripts.data.prepare_regex_il_v5_traces import make_example  # noqa: E402
from scripts.eval.evaluate_addition_trace_lm import load_model, model_logits  # noqa: E402


IL_RE = re.compile(r"^IL:\n(.+?)\n\nTemplate:", re.MULTILINE | re.DOTALL)
TEMPLATE_RE = re.compile(r"^Template:\n([^\n]+)", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v5 regex IL/template generation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9393)
    parser.add_argument("--seq-len", type=int, default=384)
    parser.add_argument("--max-new-chars", type=int, default=300)
    parser.add_argument("--component-counts", default="2,3,3,4")
    parser.add_argument("--string-min-len", type=int, default=2)
    parser.add_argument("--string-max-len", type=int, default=9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show-failures", type=int, default=10)
    parser.add_argument("--task-prefix", default=None, help="Optional leading task line, e.g. 'Task: regex_v5'.")
    parser.add_argument("--shared-output-format", action="store_true")
    parser.add_argument("--stop-token", default=None)
    return parser.parse_args()


@torch.no_grad()
def generate_greedy(
    model,
    vocab,
    prompt: str,
    *,
    seq_len: int,
    max_new_chars: int,
    device: torch.device,
    stop_token: str | None = None,
) -> str:
    ids = vocab.encode(prompt, device=device).tolist()
    for _ in range(max_new_chars):
        context = torch.tensor([ids[-seq_len:]], dtype=torch.long, device=device)
        logits = model_logits(model, context, pad_to_length=seq_len)[:, -1, :]
        ids.append(int(logits.argmax(dim=-1).item()))
        text = vocab.decode(ids)
        if stop_token is not None and stop_token in text[len(prompt) :]:
            break
        if "\n\nInput:\n" in text[len(prompt) :] or re.search(r"\nTemplate:\n[^\n]+\n", text):
            break
    return vocab.decode(ids)


def prompt_from_example(
    example: dict[str, object],
    task_prefix: str | None = None,
    *,
    shared_output_format: bool = False,
) -> str:
    rendered = format_example(example)
    if shared_output_format:
        prompt = rendered.split("\nIL:\n", 1)[0] + "\nOutput:\n"
    else:
        prompt = rendered.split("\nIL:\n", 1)[0] + "\nIL:\n"
    if task_prefix is not None:
        prompt = task_prefix.rstrip("\n") + "\n" + prompt
    return prompt


def extract_il(text: str) -> str | None:
    match = IL_RE.search(text)
    return match.group(1).strip() if match else None


def extract_template(text: str) -> str | None:
    match = TEMPLATE_RE.search(text)
    return match.group(1).strip() if match else None


def expand_template(template: str | None, refs: list[str]) -> str | None:
    if template is None:
        return None
    expanded = template
    for idx, value in enumerate(refs):
        expanded = expanded.replace(f"<{idx}>", re.escape(value))
    return expanded


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, vocab, _config = load_model(Path(args.checkpoint), device)
    rng = random.Random(args.seed)
    component_counts = parse_component_counts(args.component_counts)

    failures = []
    il_exact = 0
    template_exact = 0
    expanded_exact = 0
    parsed_il = 0
    parsed_template = 0
    band_counts: dict[str, int] = {}
    band_exact: dict[str, int] = {}
    for _idx in range(args.examples):
        example = make_example(
            rng,
            component_counts=component_counts,
            string_min_len=args.string_min_len,
            string_max_len=args.string_max_len,
        )
        band = str(example["band"])
        band_counts[band] = band_counts.get(band, 0) + 1
        prompt = prompt_from_example(example, args.task_prefix, shared_output_format=args.shared_output_format)
        generated = generate_greedy(
            model,
            vocab,
            prompt,
            seq_len=args.seq_len,
            max_new_chars=args.max_new_chars,
            device=device,
            stop_token=args.stop_token,
        )
        predicted_il = extract_il(generated)
        predicted_template = extract_template(generated)
        expected_il = str(example["il"])
        expected_template = str(example["template"])
        refs = example["refs"]
        if not isinstance(refs, list):
            raise TypeError("refs must be a list")

        parsed_il += int(predicted_il is not None)
        parsed_template += int(predicted_template is not None)
        il_ok = predicted_il == expected_il
        template_ok = predicted_template == expected_template
        expanded_ok = expand_template(predicted_template, refs) == expand_template(expected_template, refs)
        il_exact += int(il_ok)
        template_exact += int(template_ok)
        expanded_exact += int(expanded_ok)
        band_exact[band] = band_exact.get(band, 0) + int(expanded_ok)
        if (not template_ok or not il_ok) and len(failures) < args.show_failures:
            failures.append(
                {
                    "band": band,
                    "input": example["input"],
                    "expected_il": expected_il,
                    "predicted_il": predicted_il,
                    "expected_template": expected_template,
                    "predicted_template": predicted_template,
                    "expected_regex": expand_template(expected_template, refs),
                    "predicted_regex": expand_template(predicted_template, refs),
                    "generated": generated,
                }
            )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "examples": args.examples,
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
        "band_counts": band_counts,
        "band_expanded_exact": band_exact,
        "band_expanded_exact_accuracy": {
            band: band_exact.get(band, 0) / count for band, count in sorted(band_counts.items())
        },
        "failures_shown": failures,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
