from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROWS_URL = "https://datasets-server.huggingface.co/rows"


@dataclass(frozen=True)
class MathBridgeConfig:
    dataset: str
    config: str
    split: str
    examples: int
    seed: int
    batch_size: int
    min_offset: int
    max_offset: int
    task: str
    sentinel: str
    strip_dollar: bool
    min_input_chars: int
    max_input_chars: int
    min_output_chars: int
    max_output_chars: int
    offset_strategy: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MathBridge spoken-English to LaTeX char-LM corpus.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--examples-output", type=Path, default=None)
    parser.add_argument("--source-examples-jsonl", type=Path, default=None)
    parser.add_argument("--dataset", default="Kyudan/MathBridge")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--examples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--min-offset", type=int, default=0)
    parser.add_argument("--max-offset", type=int, default=23_195_831)
    parser.add_argument("--task", default="mathbridge_latex")
    parser.add_argument("--sentinel", default="<END>")
    parser.add_argument("--il-format", action="store_true", help="Include an intermediate LaTeX analysis block before Output.")
    parser.add_argument("--strip-dollar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-input-chars", type=int, default=3)
    parser.add_argument("--max-input-chars", type=int, default=160)
    parser.add_argument("--min-output-chars", type=int, default=1)
    parser.add_argument("--max-output-chars", type=int, default=120)
    parser.add_argument("--offset-strategy", choices=("random", "sequential"), default="sequential")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--retry-limit", type=int, default=8)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def fetch_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    retry_limit: int,
    retry_base_seconds: float,
) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{ROWS_URL}?{query}"
    for attempt in range(retry_limit + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.load(response)
            break
        except (HTTPError, URLError) as exc:
            retryable = isinstance(exc, URLError) or getattr(exc, "code", None) in {429, 500, 502, 503, 504}
            if not retryable or attempt >= retry_limit:
                raise
            time.sleep(retry_base_seconds * (2**attempt))
    return payload.get("rows", [])


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_equation(value: object, *, strip_dollar: bool) -> str:
    equation = clean_text(value)
    if strip_dollar and len(equation) >= 2 and equation.startswith("$") and equation.endswith("$"):
        equation = equation[1:-1].strip()
    return equation


def valid_pair(
    spoken: str,
    equation: str,
    *,
    min_input_chars: int,
    max_input_chars: int,
    min_output_chars: int,
    max_output_chars: int,
) -> bool:
    return (
        min_input_chars <= len(spoken) <= max_input_chars
        and min_output_chars <= len(equation) <= max_output_chars
        and "\n" not in spoken
        and "\n" not in equation
    )


def latex_tokens(equation: str) -> list[str]:
    return re.findall(r"\\[A-Za-z]+|\\[^\sA-Za-z]|[A-Za-z]+|\d+(?:\.\d+)?|[_^{}()[\],=+\-*/<>|]|\S", equation)


def latex_features(tokens: list[str]) -> list[str]:
    features = []
    commands = [token for token in tokens if token.startswith("\\") and len(token) > 1 and token[1:].isalpha()]
    variables = [token for token in tokens if re.fullmatch(r"[A-Za-z]+", token)]
    numbers = [token for token in tokens if re.fullmatch(r"\d+(?:\.\d+)?", token)]
    relations = [token for token in tokens if token in {"=", "<", ">", "\\leq", "\\geq", "\\neq", "\\approx", "\\subset", "\\subseteq"}]
    if commands:
        features.append("COMMANDS(" + ",".join(commands[:8]) + ")")
    if variables:
        features.append("VARS(" + ",".join(variables[:8]) + ")")
    if numbers:
        features.append("NUMBERS(" + ",".join(numbers[:8]) + ")")
    if relations:
        features.append("RELATIONS(" + ",".join(relations[:8]) + ")")
    if "_" in tokens:
        features.append("HAS_SUBSCRIPT")
    if "^" in tokens:
        features.append("HAS_SUPERSCRIPT")
    if "\\frac" in tokens:
        features.append("HAS_FRACTION")
    if not features:
        features.append("DIRECT_SYMBOL")
    return features


def format_example(spoken: str, equation: str, *, task: str, sentinel: str, il_format: bool) -> str:
    if not il_format:
        return f"Task: {task}\nInput:\n{spoken}\n\nOutput:\n{equation}\n{sentinel}"
    tokens = latex_tokens(equation)
    normalized_spoken = spoken.rstrip(".").lower()
    return (
        f"Task: {task}\n"
        f"Input:\n{spoken}\n\n"
        "IL:\n"
        f"SPOKEN({normalized_spoken})\n"
        f"STRUCTURE({';'.join(latex_features(tokens))})\n"
        f"TOKENS({' '.join(tokens)})\n\n"
        f"Output:\n{equation}\n{sentinel}"
    )


def main() -> None:
    args = parse_args()
    if args.examples <= 0:
        raise ValueError("--examples must be positive")
    if args.batch_size <= 0 or args.batch_size > 100:
        raise ValueError("--batch-size must be in 1..100 for the dataset rows API")
    if args.max_offset <= args.min_offset:
        raise ValueError("--max-offset must be greater than --min-offset")

    output = resolve(args.output)
    metadata_output = resolve(args.metadata_output) if args.metadata_output else output.with_suffix(".sources.json")
    examples_output = resolve(args.examples_output) if args.examples_output else output.with_suffix(".examples.jsonl")

    rng = random.Random(args.seed)
    attempts = 0
    if args.source_examples_jsonl is not None:
        source_examples_path = resolve(args.source_examples_jsonl)
        examples = [
            json.loads(line)
            for line in source_examples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.examples > len(examples):
            raise ValueError(f"{source_examples_path} has only {len(examples)} examples; requested {args.examples}")
        examples = rng.sample(examples, args.examples)
    else:
        examples: list[dict[str, object]] = []
        seen_row_ids: set[int] = set()
        max_start = max(args.min_offset, args.max_offset - args.batch_size)
        next_offset = args.min_offset
        while len(examples) < args.examples:
            attempts += 1
            if args.offset_strategy == "random":
                offset = rng.randint(args.min_offset, max_start)
            else:
                offset = next_offset
                next_offset += args.batch_size
                if next_offset > max_start:
                    next_offset = args.min_offset
            rows = fetch_rows(
                dataset=args.dataset,
                config=args.config,
                split=args.split,
                offset=offset,
                length=args.batch_size,
                retry_limit=args.retry_limit,
                retry_base_seconds=args.retry_base_seconds,
            )
            for wrapper in rows:
                if len(examples) >= args.examples:
                    break
                row_idx = int(wrapper["row_idx"])
                if row_idx in seen_row_ids:
                    continue
                row = wrapper.get("row", {})
                spoken = clean_text(row.get("spoken_English"))
                equation = clean_equation(row.get("equation"), strip_dollar=args.strip_dollar)
                if not valid_pair(
                    spoken,
                    equation,
                    min_input_chars=args.min_input_chars,
                    max_input_chars=args.max_input_chars,
                    min_output_chars=args.min_output_chars,
                    max_output_chars=args.max_output_chars,
                ):
                    continue
                seen_row_ids.add(row_idx)
                examples.append({"row_idx": row_idx, "input": spoken, "output": equation})
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    rng.shuffle(examples)
    corpus = "\n\n".join(
        format_example(
            str(example["input"]),
            str(example["output"]),
            task=args.task,
            sentinel=args.sentinel,
            il_format=args.il_format,
        )
        for example in examples
    ) + "\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    examples_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(corpus, encoding="utf-8")
    with examples_output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    config = MathBridgeConfig(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        examples=args.examples,
        seed=args.seed,
        batch_size=args.batch_size,
        min_offset=args.min_offset,
        max_offset=args.max_offset,
        task=args.task,
        sentinel=args.sentinel,
        strip_dollar=args.strip_dollar,
        min_input_chars=args.min_input_chars,
        max_input_chars=args.max_input_chars,
        min_output_chars=args.min_output_chars,
        max_output_chars=args.max_output_chars,
        offset_strategy=args.offset_strategy,
    )
    metadata = {
        "config": asdict(config),
        "output": str(output.relative_to(ROOT)),
        "examples_output": str(examples_output.relative_to(ROOT)),
        "source_examples_jsonl": str(resolve(args.source_examples_jsonl).relative_to(ROOT)) if args.source_examples_jsonl else None,
        "total_examples": len(examples),
        "fetch_attempts": attempts,
        "total_chars": len(corpus),
        "unique_chars": len(set(corpus)),
        "chars": sorted(set(corpus)),
        "il_format": args.il_format,
        "first_examples": examples[:10],
    }
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "examples": len(examples), "chars": len(corpus)}, indent=2))


if __name__ == "__main__":
    main()
