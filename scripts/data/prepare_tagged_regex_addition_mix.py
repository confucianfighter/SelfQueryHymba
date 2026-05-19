from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TaggedSourceSpec:
    path: str
    count: int
    task: str


@dataclass(frozen=True)
class TaggedMixConfig:
    sources: list[TaggedSourceSpec]
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tagged example-level mix for regex, math, and LaTeX curricula.")
    parser.add_argument("--v5-input", type=Path, default=Path("data/regex_il_v5_clear_capture_120k.txt"))
    parser.add_argument("--v2-input", type=Path, default=Path("data/regex_quoted_ref_tokens_v2_easy_50k.txt"))
    parser.add_argument("--addition-input", type=Path, default=Path("data/addition_traces_1to3_digit.txt"))
    parser.add_argument("--subtraction-input", type=Path, default=Path("data/subtraction_traces_1to3_digit.txt"))
    parser.add_argument("--multiplication-input", type=Path, default=Path("data/multiplication_traces_2_digit.txt"))
    parser.add_argument("--division-input", type=Path, default=Path("data/division_traces_2to3_by_1to2_digit.txt"))
    parser.add_argument("--latex-input", type=Path, default=Path("data/mathbridge_latex_il_train_50000.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--examples-per-source", type=int, default=33333)
    parser.add_argument("--v5-examples", type=int, default=None)
    parser.add_argument("--v2-examples", type=int, default=None)
    parser.add_argument("--addition-examples", type=int, default=None)
    parser.add_argument("--subtraction-examples", type=int, default=0)
    parser.add_argument("--multiplication-examples", type=int, default=0)
    parser.add_argument("--division-examples", type=int, default=0)
    parser.add_argument("--latex-examples", type=int, default=0)
    parser.add_argument("--sentinel-format", action="store_true")
    parser.add_argument("--sentinel", default="<END>")
    parser.add_argument(
        "--math-format",
        choices=("tagged", "question_only"),
        default="tagged",
        help="Use the normal task/input/output wrapper for math, or train math examples as bare question-plus-trace.",
    )
    parser.add_argument("--seed", type=int, default=20260513)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_examples(path: Path, task: str) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("Task: "):
        return [match.group(0).strip() for match in re.finditer(r"(?ms)^Task: .*?(?=^Task: |\Z)", text)]
    if task.startswith("regex_"):
        return [match.group(0).strip() for match in re.finditer(r"(?ms)^Input:\n.*?(?=^Input:\n|\Z)", text)]
    return [example.strip() for example in text.split("\n\n") if example.strip()]


def tag_regex(example: str, task: str) -> str:
    return f"Task: {task}\n{example}"


def tag_regex_sentinel(example: str, task: str, sentinel: str) -> str:
    if "\n\nIL:\n" in example:
        body = example.replace("\n\nIL:\n", "\n\nOutput:\nIL:\n", 1)
    elif "\n\nTemplate:\n" in example:
        body = example.replace("\n\nTemplate:\n", "\n\nOutput:\nTemplate:\n", 1)
    else:
        raise ValueError(f"unsupported regex example format for task {task!r}")
    return f"Task: {task}\n{body}\n{sentinel}"


def tag_math(
    example: str,
    task: str,
    *,
    sentinel_format: bool = False,
    sentinel: str = "<END>",
    math_format: str = "tagged",
) -> str:
    lines = example.splitlines()
    if not lines:
        raise ValueError(f"empty {task} example")
    if math_format == "question_only":
        body = "\n".join(lines)
        return body + (f"\n{sentinel}" if sentinel_format else "")
    if sentinel_format:
        return f"Task: {task}\nInput:\n" + lines[0] + "\n\nOutput:\n" + "\n".join(lines[1:]) + f"\n{sentinel}"
    return f"Task: {task}\nInput:\n" + lines[0] + "\nTrace:\n" + "\n".join(lines[1:])


def sample_tagged(
    *,
    rng: random.Random,
    path: Path,
    count: int,
    task: str,
    sentinel_format: bool,
    sentinel: str,
    math_format: str,
) -> tuple[list[str], dict[str, object]]:
    examples = read_examples(path, task)
    if count > len(examples):
        raise ValueError(f"{path} has only {len(examples)} examples; requested {count}")
    sampled = rng.sample(examples, count)
    if task == "mathbridge_latex":
        tagged = sampled
    elif task in {"addition_prose", "subtraction_prose", "multiplication_prose", "division_prose"}:
        tagged = [
            tag_math(
                example,
                task,
                sentinel_format=sentinel_format,
                sentinel=sentinel,
                math_format=math_format,
            )
            for example in sampled
        ]
    elif sentinel_format:
        tagged = [tag_regex_sentinel(example, task, sentinel) for example in sampled]
    else:
        tagged = [tag_regex(example, task) for example in sampled]
    return tagged, {
        "path": str(path.relative_to(ROOT)),
        "available_examples": len(examples),
        "used_examples": count,
        "task": task,
        "format": math_format if task in {"addition_prose", "subtraction_prose", "multiplication_prose", "division_prose"} else "tagged",
        "chars": path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source_defs = [
        (resolve_path(args.v5_input), "regex_v5", args.v5_examples if args.v5_examples is not None else args.examples_per_source),
        (resolve_path(args.v2_input), "regex_v2", args.v2_examples if args.v2_examples is not None else args.examples_per_source),
        (
            resolve_path(args.addition_input),
            "addition_prose",
            args.addition_examples if args.addition_examples is not None else args.examples_per_source,
        ),
        (resolve_path(args.subtraction_input), "subtraction_prose", args.subtraction_examples),
        (resolve_path(args.multiplication_input), "multiplication_prose", args.multiplication_examples),
        (resolve_path(args.division_input), "division_prose", args.division_examples),
        (resolve_path(args.latex_input), "mathbridge_latex", args.latex_examples),
    ]
    mixed: list[str] = []
    metadata_sources = []
    config_sources = []
    for path, task, count in source_defs:
        if count <= 0:
            continue
        tagged, metadata = sample_tagged(
            rng=rng,
            path=path,
            count=count,
            task=task,
            sentinel_format=args.sentinel_format,
            sentinel=args.sentinel,
            math_format=args.math_format,
        )
        mixed.extend(tagged)
        metadata_sources.append(metadata)
        config_sources.append(TaggedSourceSpec(path=str(path.relative_to(ROOT)), count=count, task=task))
    rng.shuffle(mixed)
    corpus = "\n\n".join(mixed) + "\n"
    output_path = resolve_path(args.output)
    metadata_path = resolve_path(args.metadata_output) if args.metadata_output is not None else output_path.with_suffix(".sources.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(corpus, encoding="utf-8")
    metadata = {
        "config": asdict(TaggedMixConfig(sources=config_sources, seed=args.seed)),
        "math_format": args.math_format,
        "sentinel_format": args.sentinel_format,
        "sentinel": args.sentinel,
        "output": str(output_path.relative_to(ROOT)),
        "total_examples": len(mixed),
        "sources": metadata_sources,
        "total_chars": len(corpus),
        "unique_chars": len(set(corpus)),
        "chars": sorted(set(corpus)),
        "first_examples": mixed[:10],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "examples": len(mixed), "chars": len(corpus)}, indent=2))


if __name__ == "__main__":
    main()
