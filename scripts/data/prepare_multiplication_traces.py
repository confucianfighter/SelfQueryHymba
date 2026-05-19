from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "multiplication_traces_2_digit.txt"
DEFAULT_METADATA_OUTPUT = ROOT / "data" / "multiplication_traces_2_digit.sources.json"
PLACE_NAMES = ("ones", "tens", "hundreds")


@dataclass(frozen=True)
class MultiplicationTraceConfig:
    examples: int
    seed: int
    min_digits: int
    max_digits: int
    exhaustive: bool
    include_zero: bool
    trace_format: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate worked multiplication traces.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--examples", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--min-digits", type=int, default=2)
    parser.add_argument("--max-digits", type=int, default=2)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--include-zero", action="store_true")
    parser.add_argument("--trace-format", choices=("prose", "answer_only"), default="prose")
    return parser.parse_args()


def digit_range(digits: int, *, include_zero: bool) -> tuple[int, int]:
    if digits == 1:
        return (0 if include_zero else 1), 9
    return 10 ** (digits - 1), 10**digits - 1


def format_answer_only_multiplication_trace(left: int, right: int) -> str:
    return f"multiply {left} * {right}\nanswer: {left * right}"


def format_multiplication_trace(left: int, right: int) -> str:
    answer = left * right
    lines = [f"multiply {left} * {right}"]
    partials: list[int] = []
    for idx, digit_char in enumerate(reversed(str(right))):
        digit = int(digit_char)
        raw_partial = left * digit
        shifted_partial = raw_partial * (10**idx)
        partials.append(shifted_partial)
        line = f"{PLACE_NAMES[idx]}: {left} * {digit} = {raw_partial}"
        if idx:
            line += f", shift {idx} place = {shifted_partial}"
        lines.append(line)
    if len(partials) > 1:
        terms = " + ".join(str(partial) for partial in partials)
        lines.append(f"sum: {terms} = {sum(partials)}")
    lines.append(f"answer: {answer}")
    return "\n".join(lines)


def random_examples(config: MultiplicationTraceConfig) -> list[tuple[int, int]]:
    rng = random.Random(config.seed)
    digit_pairs = [
        (left_digits, right_digits)
        for left_digits in range(config.min_digits, config.max_digits + 1)
        for right_digits in range(config.min_digits, config.max_digits + 1)
    ]
    examples = []
    for idx in range(config.examples):
        left_digits, right_digits = digit_pairs[idx % len(digit_pairs)]
        left_min, left_max = digit_range(left_digits, include_zero=config.include_zero)
        right_min, right_max = digit_range(right_digits, include_zero=config.include_zero)
        examples.append((rng.randint(left_min, left_max), rng.randint(right_min, right_max)))
    rng.shuffle(examples)
    return examples


def exhaustive_examples(config: MultiplicationTraceConfig) -> list[tuple[int, int]]:
    values_by_digits = {}
    for digits in range(config.min_digits, config.max_digits + 1):
        start, stop = digit_range(digits, include_zero=config.include_zero)
        values_by_digits[digits] = range(start, stop + 1)
    return [
        (left, right)
        for left_digits in range(config.min_digits, config.max_digits + 1)
        for right_digits in range(config.min_digits, config.max_digits + 1)
        for left in values_by_digits[left_digits]
        for right in values_by_digits[right_digits]
    ]


def main() -> None:
    args = parse_args()
    if args.min_digits < 1 or args.max_digits > 3 or args.min_digits > args.max_digits:
        raise ValueError("expected 1 <= min_digits <= max_digits <= 3")
    if args.examples <= 0 and not args.exhaustive:
        raise ValueError("--examples must be positive unless --exhaustive is set")

    config = MultiplicationTraceConfig(
        examples=args.examples,
        seed=args.seed,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        exhaustive=args.exhaustive,
        include_zero=args.include_zero,
        trace_format=args.trace_format,
    )
    pairs = exhaustive_examples(config) if args.exhaustive else random_examples(config)
    formatter = format_answer_only_multiplication_trace if args.trace_format == "answer_only" else format_multiplication_trace
    corpus = "\n\n".join(formatter(left, right) for left, right in pairs) + "\n"
    metadata = {
        "config": asdict(config),
        "output": str(args.output.relative_to(ROOT) if args.output.is_absolute() else args.output),
        "total_examples": len(pairs),
        "total_chars": len(corpus),
        "unique_chars": len(set(corpus)),
        "first_examples": [{"left": left, "right": right, "answer": left * right} for left, right in pairs[:10]],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8")
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "examples": len(pairs), "chars": len(corpus)}, indent=2))


if __name__ == "__main__":
    main()
