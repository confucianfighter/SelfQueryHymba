from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "subtraction_traces_1to3_digit.txt"
DEFAULT_METADATA_OUTPUT = ROOT / "data" / "subtraction_traces_1to3_digit.sources.json"
PLACE_NAMES = ("ones", "tens", "hundreds", "thousands")


@dataclass(frozen=True)
class SubtractionTraceConfig:
    examples: int
    seed: int
    min_digits: int
    max_digits: int
    exhaustive: bool
    include_zero: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate worked nonnegative 1-to-3 digit subtraction traces.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--examples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--min-digits", type=int, default=1)
    parser.add_argument("--max-digits", type=int, default=3)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--include-zero", action="store_true")
    return parser.parse_args()


def digit_range(digits: int, *, include_zero: bool) -> tuple[int, int]:
    if digits == 1:
        return (0 if include_zero else 1), 9
    return 10 ** (digits - 1), 10**digits - 1


def digits_reversed(value: int, width: int) -> list[int]:
    return [(value // (10**idx)) % 10 for idx in range(width)]


def borrow_description(idx: int, borrow_idx: int) -> str:
    if borrow_idx == idx + 1:
        return f"take 1 from {PLACE_NAMES[borrow_idx]}"
    through = " through ".join(PLACE_NAMES[j] for j in range(idx + 1, borrow_idx))
    return f"borrow through {through} from {PLACE_NAMES[borrow_idx]}"


def format_subtraction_trace(left: int, right: int) -> str:
    if left < right:
        raise ValueError("subtraction traces require left >= right")
    answer = left - right
    width = max(len(str(left)), len(str(right)))
    top_digits = digits_reversed(left, width)
    bottom_digits = digits_reversed(right, width)

    lines = [f"subtract {left} - {right}"]
    for idx in range(width):
        top = top_digits[idx]
        bottom = bottom_digits[idx]
        if top < bottom:
            borrow_idx = idx + 1
            while borrow_idx < width and top_digits[borrow_idx] == 0:
                borrow_idx += 1
            if borrow_idx >= width:
                raise ValueError(f"could not borrow in {left} - {right}")
            top_digits[borrow_idx] -= 1
            for fill_idx in range(borrow_idx - 1, idx, -1):
                top_digits[fill_idx] = 9
            top += 10
            top_digits[idx] = top
            line = (
                f"{PLACE_NAMES[idx]}: {top - 10} - {bottom} needs borrow, "
                f"{borrow_description(idx, borrow_idx)}, {top} - {bottom} = {top - bottom}"
            )
        else:
            line = f"{PLACE_NAMES[idx]}: {top} - {bottom} = {top - bottom}"
        lines.append(line)
    lines.append(f"answer: {answer}")
    return "\n".join(lines)


def random_examples(config: SubtractionTraceConfig) -> list[tuple[int, int]]:
    rng = random.Random(config.seed)
    digit_pairs = [
        (left_digits, right_digits)
        for left_digits in range(config.min_digits, config.max_digits + 1)
        for right_digits in range(config.min_digits, config.max_digits + 1)
    ]
    examples = []
    while len(examples) < config.examples:
        left_digits, right_digits = digit_pairs[len(examples) % len(digit_pairs)]
        left_min, left_max = digit_range(left_digits, include_zero=config.include_zero)
        right_min, right_max = digit_range(right_digits, include_zero=config.include_zero)
        left = rng.randint(left_min, left_max)
        right = rng.randint(right_min, right_max)
        if left < right:
            left, right = right, left
        examples.append((left, right))
    rng.shuffle(examples)
    return examples


def exhaustive_examples(config: SubtractionTraceConfig) -> list[tuple[int, int]]:
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
        if left >= right
    ]


def main() -> None:
    args = parse_args()
    if args.min_digits < 1 or args.max_digits > 3 or args.min_digits > args.max_digits:
        raise ValueError("expected 1 <= min_digits <= max_digits <= 3")
    if args.examples <= 0 and not args.exhaustive:
        raise ValueError("--examples must be positive unless --exhaustive is set")

    config = SubtractionTraceConfig(
        examples=args.examples,
        seed=args.seed,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        exhaustive=args.exhaustive,
        include_zero=args.include_zero,
    )
    pairs = exhaustive_examples(config) if args.exhaustive else random_examples(config)
    corpus = "\n\n".join(format_subtraction_trace(left, right) for left, right in pairs) + "\n"
    metadata = {
        "config": asdict(config),
        "output": str(args.output.relative_to(ROOT) if args.output.is_absolute() else args.output),
        "total_examples": len(pairs),
        "total_chars": len(corpus),
        "unique_chars": len(set(corpus)),
        "first_examples": [{"left": left, "right": right, "answer": left - right} for left, right in pairs[:10]],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8")
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "examples": len(pairs), "chars": len(corpus)}, indent=2))


if __name__ == "__main__":
    main()
