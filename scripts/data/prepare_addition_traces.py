from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "addition_traces_1to3_digit.txt"
DEFAULT_METADATA_OUTPUT = ROOT / "data" / "addition_traces_1to3_digit.sources.json"
PLACE_NAMES = ("ones", "tens", "hundreds", "thousands")


@dataclass(frozen=True)
class AdditionTraceConfig:
    examples: int
    seed: int
    min_digits: int
    max_digits: int
    exhaustive: bool
    include_zero: bool
    trace_format: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate worked 1-to-3 digit addition traces.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--examples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--min-digits", type=int, default=1)
    parser.add_argument("--max-digits", type=int, default=3)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--include-zero", action="store_true")
    parser.add_argument("--trace-format", choices=("prose", "strict", "copy_strict"), default="prose")
    return parser.parse_args()


def digit_range(digits: int, *, include_zero: bool) -> tuple[int, int]:
    if digits == 1:
        return (0 if include_zero else 1), 9
    return 10 ** (digits - 1), 10**digits - 1


def digits_reversed(value: int, width: int) -> list[int]:
    return [(value // (10**idx)) % 10 for idx in range(width)]


def format_addition_trace(left: int, right: int) -> str:
    answer = left + right
    width = max(len(str(left)), len(str(right)))
    left_digits = digits_reversed(left, width)
    right_digits = digits_reversed(right, width)
    carry = 0

    lines = [f"add {left} + {right}"]
    for idx in range(width):
        total = left_digits[idx] + right_digits[idx] + carry
        write_digit = total % 10
        next_carry = total // 10
        terms = f"{left_digits[idx]} + {right_digits[idx]}"
        if carry:
            terms += f" + carry {carry}"
        line = f"{PLACE_NAMES[idx]}: {terms} = {total}, write {write_digit}"
        if next_carry:
            line += f" carry {next_carry}"
        lines.append(line)
        carry = next_carry

    if carry:
        lines.append(f"{PLACE_NAMES[width]}: carry {carry}, write {carry}")
    lines.append(f"answer: {answer}")
    return "\n".join(lines)


def format_strict_addition_trace(left: int, right: int) -> str:
    answer = left + right
    width = max(len(str(left)), len(str(right)))
    left_digits = digits_reversed(left, width)
    right_digits = digits_reversed(right, width)
    carry = 0

    lines = [f"{left} + {right}"]
    for idx in range(width):
        total = left_digits[idx] + right_digits[idx] + carry
        digit = total % 10
        next_carry = total // 10
        lines.append(
            f"c{idx}: a={left_digits[idx]} b={right_digits[idx]} c={carry} -> s={total} d={digit} k={next_carry}"
        )
        carry = next_carry
    if carry:
        lines.append(f"c{width}: a=0 b=0 c={carry} -> s={carry} d={carry} k=0")
    lines.append(f"ans: {answer}")
    return "\n".join(lines)


def format_copy_strict_addition_trace(left: int, right: int) -> str:
    answer = left + right
    width = max(len(str(left)), len(str(right)))
    left_digits = digits_reversed(left, width)
    right_digits = digits_reversed(right, width)
    carry = 0

    lines = [
        f"x: {left}",
        f"y: {right}",
        "xd: " + " ".join(f"x{idx}={digit}" for idx, digit in enumerate(left_digits)),
        "yd: " + " ".join(f"y{idx}={digit}" for idx, digit in enumerate(right_digits)),
    ]
    for idx in range(width):
        total = left_digits[idx] + right_digits[idx] + carry
        digit = total % 10
        next_carry = total // 10
        lines.append(
            f"c{idx}: a=x{idx}={left_digits[idx]} b=y{idx}={right_digits[idx]} c={carry} -> s={total} d={digit} k={next_carry}"
        )
        carry = next_carry
    if carry:
        lines.append(f"c{width}: a=0 b=0 c={carry} -> s={carry} d={carry} k=0")
    lines.append(f"ans: {answer}")
    return "\n".join(lines)


def random_examples(config: AdditionTraceConfig) -> list[tuple[int, int]]:
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


def exhaustive_examples(config: AdditionTraceConfig) -> list[tuple[int, int]]:
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

    config = AdditionTraceConfig(
        examples=args.examples,
        seed=args.seed,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        exhaustive=args.exhaustive,
        include_zero=args.include_zero,
        trace_format=args.trace_format,
    )
    pairs = exhaustive_examples(config) if args.exhaustive else random_examples(config)
    if args.trace_format == "strict":
        formatter = format_strict_addition_trace
    elif args.trace_format == "copy_strict":
        formatter = format_copy_strict_addition_trace
    else:
        formatter = format_addition_trace
    corpus = "\n\n".join(formatter(left, right) for left, right in pairs) + "\n"
    metadata = {
        "config": asdict(config),
        "output": str(args.output.relative_to(ROOT) if args.output.is_absolute() else args.output),
        "total_examples": len(pairs),
        "total_chars": len(corpus),
        "unique_chars": len(set(corpus)),
        "first_examples": [{"left": left, "right": right, "answer": left + right} for left, right in pairs[:10]],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8")
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "examples": len(pairs), "chars": len(corpus)}, indent=2))


if __name__ == "__main__":
    main()
