from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.prepare_addition_traces import AdditionTraceConfig, format_addition_trace, random_examples as addition_examples
from scripts.data.prepare_regex_il_v3_traces import format_example, parse_component_counts
from scripts.data.prepare_regex_il_v5_traces import make_example as make_v5_example
from scripts.data.prepare_subtraction_traces import (
    SubtractionTraceConfig,
    format_subtraction_trace,
    random_examples as subtraction_examples,
)
from scripts.eval.evaluate_addition_trace_lm import load_model, model_logits
from scripts.eval.evaluate_regex_il_v5_trace_lm import expand_template, extract_il, extract_template


SENTINEL = "<END>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write sample generations for the sentinel multitask corpus.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9393)
    parser.add_argument("--math-seed", type=int, default=2026)
    parser.add_argument("--seq-len", type=int, default=384)
    parser.add_argument("--max-new-chars", type=int, default=340)
    parser.add_argument("--component-counts", default="2,3,3,4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    stop_token: str = "<END>",
) -> str:
    ids = vocab.encode(prompt, device=device).tolist()
    for _ in range(max_new_chars):
        context = torch.tensor([ids[-seq_len:]], dtype=torch.long, device=device)
        logits = model_logits(model, context, pad_to_length=seq_len)[:, -1, :]
        ids.append(int(logits.argmax(dim=-1).item()))
        text = vocab.decode(ids)
        if stop_token in text[len(prompt) :]:
            break
    return vocab.decode(ids)


def completion_from_generated(prompt: str, generated: str) -> str:
    if not generated.startswith(prompt):
        return generated
    return generated[len(prompt) :]


def split_math_trace(trace: str) -> tuple[str, str]:
    lines = trace.splitlines()
    return lines[0], "\n".join(lines[1:])


def math_prompt(task: str, trace: str) -> tuple[str, str]:
    input_line, output = split_math_trace(trace)
    prompt = f"Task: {task}\nInput:\n{input_line}\n\nOutput:\n"
    expected = output + f"\n{SENTINEL}"
    return prompt, expected


def answer_from_text(text: str) -> str | None:
    match = re.search(r"^answer: ([^\n]+)", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def result_block(*, correct: bool, expected_answer: str | None) -> list[str]:
    lines = ["**3. Correctness**", "", f"- correct: `{correct}`"]
    if not correct:
        lines.append(f"- expected answer: `{expected_answer}`")
    lines.append("")
    return lines


def write_regex_samples(lines: list[str], model, vocab, args: argparse.Namespace, device: torch.device) -> None:
    rng = random.Random(args.seed)
    component_counts = parse_component_counts(args.component_counts)
    lines.append("## regex_v5")
    for idx in range(1, args.examples + 1):
        example = make_v5_example(rng, component_counts=component_counts, string_min_len=2, string_max_len=9)
        rendered = format_example(example)
        prompt = "Task: regex_v5\n" + rendered.split("\nIL:\n", 1)[0] + "\nOutput:\n"
        generated = generate_greedy(
            model,
            vocab,
            prompt,
            seq_len=args.seq_len,
            max_new_chars=args.max_new_chars,
            device=device,
        )
        predicted_il = extract_il(generated)
        predicted_template = extract_template(generated)
        refs = example["refs"]
        if not isinstance(refs, list):
            raise TypeError("refs must be a list")
        expected_il = str(example["il"])
        expected_template = str(example["template"])
        expected_expanded = expand_template(expected_template, refs)
        predicted_expanded = expand_template(predicted_template, refs)
        exact = predicted_expanded == expected_expanded
        generated_completion = completion_from_generated(prompt, generated)
        lines.extend(
            [
                f"### regex_v5 sample {idx}",
                "**1. Input**",
                "```text",
                prompt,
                "```",
                "",
                "**2. Model Answer**",
                "",
                f"- predicted expanded regex: `{predicted_expanded}`",
                f"- predicted template: `{predicted_template}`",
                f"- predicted IL: `{predicted_il}`",
                "",
                "```text",
                generated_completion,
                "```",
                "",
            ]
        )
        lines.extend(result_block(correct=exact, expected_answer=expected_expanded))


def write_math_samples(
    lines: list[str],
    *,
    title: str,
    task: str,
    traces: list[str],
    model,
    vocab,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    lines.append(f"## {title}")
    for idx, trace in enumerate(traces[: args.examples], start=1):
        prompt, expected = math_prompt(task, trace)
        generated = generate_greedy(
            model,
            vocab,
            prompt,
            seq_len=args.seq_len,
            max_new_chars=args.max_new_chars,
            device=device,
        )
        expected_answer = answer_from_text(expected)
        generated_completion = completion_from_generated(prompt, generated)
        predicted_answer = answer_from_text(generated_completion)
        correct = predicted_answer == expected_answer
        lines.extend(
            [
                f"### {title} sample {idx}",
                "**1. Input**",
                "```text",
                prompt,
                "```",
                "",
                "**2. Model Answer**",
                "",
                f"- predicted answer: `{predicted_answer}`",
                "",
                "```text",
                generated_completion,
                "```",
                "",
            ]
        )
        lines.extend(result_block(correct=correct, expected_answer=expected_answer))


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    output = args.output if args.output.is_absolute() else ROOT / args.output
    device = torch.device(args.device)
    model, vocab, _config = load_model(checkpoint, device)

    additions = [
        format_addition_trace(left, right)
        for left, right in addition_examples(
            AdditionTraceConfig(
                examples=args.examples,
                seed=args.math_seed,
                min_digits=1,
                max_digits=3,
                exhaustive=False,
                include_zero=False,
                trace_format="prose",
            )
        )
    ]
    subtractions = [
        format_subtraction_trace(left, right)
        for left, right in subtraction_examples(
            SubtractionTraceConfig(
                examples=args.examples,
                seed=args.math_seed,
                min_digits=1,
                max_digits=3,
                exhaustive=False,
                include_zero=False,
            )
        )
    ]

    lines = [
        f"# Sentinel Multitask Samples",
        f"checkpoint: {checkpoint}",
        "",
    ]
    write_regex_samples(lines, model, vocab, args, device)
    write_math_samples(
        lines,
        title="addition_prose",
        task="addition_prose",
        traces=additions,
        model=model,
        vocab=vocab,
        args=args,
        device=device,
    )
    write_math_samples(
        lines,
        title="subtraction_prose",
        task="subtraction_prose",
        traces=subtractions,
        model=model,
        vocab=vocab,
        args=args,
        device=device,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
