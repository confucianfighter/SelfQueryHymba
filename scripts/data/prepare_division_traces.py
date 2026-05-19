from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "division_traces_2to3_by_1to2_digit.txt"
DEFAULT_METADATA_OUTPUT = ROOT / "data" / "division_traces_2to3_by_1to2_digit.sources.json"


@dataclass(frozen=True)
class DivisionTraceConfig:
    examples: int
    seed: int
    min_dividend_digits: int
    max_dividend_digits: int
    min_divisor_digits: int
    max_divisor_digits: int
    include_zero_remainder: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate worked long-division traces.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--examples", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--min-dividend-digits", type=int, default=2)
    parser.add_argument("--max-dividend-digits", type=int, default=3)
    parser.add_argument("--min-divisor-digits", type=int, default=1)
    parser.add_argument("--max-divisor-digits", type=int, default=2)
    parser.add_argument("--include-zero-remainder", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def digit_range(digits: int, *, include_zero: bool = False) -> tuple[int, int]:
    if digits == 1:
        return (0 if include_zero else 1), 9
    return 10 ** (digits - 1), 10**digits - 1


def format_division_trace(dividend: int, divisor: int) -> str:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    quotient = dividend // divisor
    remainder = dividend % divisor
    lines = [f"divide {dividend} / {divisor}"]
    current = 0
    quotient_digits: list[str] = []
    for digit_char in str(dividend):
        current = current * 10 + int(digit_char)
        q_digit = current // divisor
        product = q_digit * divisor
        new_remainder = current - product
        if quotient_digits or q_digit:
            quotient_digits.append(str(q_digit))
        shown_quotient = "".join(quotient_digits) if quotient_digits else "0"
        lines.append(
            f"step: {divisor} goes into {current} {q_digit} times, "
            f"{q_digit} * {divisor} = {product}, subtract {current} - {product} = {new_remainder}, quotient so far {shown_quotient}"
        )
        current = new_remainder
    if quotient_digits:
        lines.append(f"answer: {quotient} remainder {remainder}")
    else:
        lines.append(f"answer: 0 remainder {dividend}")
    return "\n".join(lines)


def random_examples(config: DivisionTraceConfig) -> list[tuple[int, int]]:
    rng = random.Random(config.seed)
    digit_pairs = [
        (dividend_digits, divisor_digits)
        for dividend_digits in range(config.min_dividend_digits, config.max_dividend_digits + 1)
        for divisor_digits in range(config.min_divisor_digits, config.max_divisor_digits + 1)
    ]
    examples = []
    while len(examples) < config.examples:
        dividend_digits, divisor_digits = digit_pairs[len(examples) % len(digit_pairs)]
        dividend_min, dividend_max = digit_range(dividend_digits)
        divisor_min, divisor_max = digit_range(divisor_digits)
        dividend = rng.randint(dividend_min, dividend_max)
        divisor = rng.randint(divisor_min, min(divisor_max, dividend))
        if not config.include_zero_remainder and dividend % divisor == 0:
            continue
        examples.append((dividend, divisor))
    rng.shuffle(examples)
    return examples


def main() -> None:
    args = parse_args()
    if not 1 <= args.min_dividend_digits <= args.max_dividend_digits <= 3:
        raise ValueError("expected 1 <= min dividend digits <= max dividend digits <= 3")
    if not 1 <= args.min_divisor_digits <= args.max_divisor_digits <= 2:
        raise ValueError("expected 1 <= min divisor digits <= max divisor digits <= 2")
    if args.examples <= 0:
        raise ValueError("--examples must be positive")

    config = DivisionTraceConfig(
        examples=args.examples,
        seed=args.seed,
        min_dividend_digits=args.min_dividend_digits,
        max_dividend_digits=args.max_dividend_digits,
        min_divisor_digits=args.min_divisor_digits,
        max_divisor_digits=args.max_divisor_digits,
        include_zero_remainder=args.include_zero_remainder,
    )
    pairs = random_examples(config)
    corpus = "\n\n".join(format_division_trace(dividend, divisor) for dividend, divisor in pairs) + "\n"
    metadata = {
        "config": asdict(config),
        "output": str(args.output.relative_to(ROOT) if args.output.is_absolute() else args.output),
        "total_examples": len(pairs),
        "total_chars": len(corpus),
        "unique_chars": len(set(corpus)),
        "first_examples": [
            {
                "dividend": dividend,
                "divisor": divisor,
                "quotient": dividend // divisor,
                "remainder": dividend % divisor,
            }
            for dividend, divisor in pairs[:10]
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8")
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "examples": len(pairs), "chars": len(corpus)}, indent=2))


if __name__ == "__main__":
    main()
