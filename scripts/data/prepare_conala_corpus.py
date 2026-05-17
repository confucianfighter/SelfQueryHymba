from __future__ import annotations

import argparse
import json
import random
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "https://www.phontron.com/download/conala-corpus-v1.1.zip"
DEFAULT_ARCHIVE = ROOT / "data" / "raw" / "conala-corpus-v1.1.zip"
DEFAULT_EXTRACT_DIR = ROOT / "data" / "raw" / "conala-corpus-v1.1"
DEFAULT_CURATED_TRAIN_URL = "https://huggingface.co/datasets/neulab/conala/resolve/main/data/conala-paired-train.json"
DEFAULT_CURATED_TEST_URL = "https://huggingface.co/datasets/neulab/conala/resolve/main/data/conala-paired-test.json"
DEFAULT_CURATED_TRAIN = ROOT / "data" / "raw" / "conala-paired-train.json"
DEFAULT_CURATED_TEST = ROOT / "data" / "raw" / "conala-paired-test.json"


@dataclass(frozen=True)
class ConalaCorpusConfig:
    split: str
    include_mined: bool
    mined_limit: int
    mined_min_prob: float
    seed: int
    task: str
    sentinel: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CoNaLa intent-to-Python snippet char-LM corpus.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--curated-train-url", default=DEFAULT_CURATED_TRAIN_URL)
    parser.add_argument("--curated-test-url", default=DEFAULT_CURATED_TEST_URL)
    parser.add_argument("--curated-train", type=Path, default=DEFAULT_CURATED_TRAIN)
    parser.add_argument("--curated-test", type=Path, default=DEFAULT_CURATED_TEST)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--include-mined", action="store_true")
    parser.add_argument("--mined-limit", type=int, default=0)
    parser.add_argument("--mined-min-prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--task", default="conala_python")
    parser.add_argument("--sentinel", default="<END>")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def ensure_archive(url: str, archive: Path, extract_dir: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)


def ensure_download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)


def find_file(extract_dir: Path, name: str) -> Path:
    matches = list(extract_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(f"could not find {name!r} under {extract_dir}")
    return matches[0]


def read_json(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list or JSONL records")
    return rows


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_example(intent: str, snippet: str, *, task: str, sentinel: str) -> str:
    return f"Task: {task}\nInput:\n{intent.strip()}\n\nOutput:\n{snippet.strip()}\n{sentinel}"


def curated_examples(path: Path, *, task: str, sentinel: str) -> list[str]:
    examples = []
    for row in read_json(path):
        intent = row.get("rewritten_intent") or row.get("intent")
        snippet = row.get("snippet")
        if intent and snippet:
            examples.append(format_example(intent, snippet, task=task, sentinel=sentinel))
    return examples


def mined_examples(
    extract_dir: Path,
    *,
    limit: int,
    min_prob: float,
    seed: int,
    task: str,
    sentinel: str,
) -> list[str]:
    mined_path = find_file(extract_dir, "conala-mined.jsonl")
    rows = [row for row in read_jsonl(mined_path) if float(row.get("prob", 0.0)) >= min_prob]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    examples = []
    for row in rows:
        intent = row.get("intent")
        snippet = row.get("snippet")
        if intent and snippet:
            examples.append(format_example(intent, snippet, task=task, sentinel=sentinel))
    return examples


def main() -> None:
    args = parse_args()
    archive = resolve(args.archive)
    extract_dir = resolve(args.extract_dir)
    curated_train = resolve(args.curated_train)
    curated_test = resolve(args.curated_test)
    output = resolve(args.output)
    metadata_output = resolve(args.metadata_output) if args.metadata_output else output.with_suffix(".sources.json")

    curated_path = curated_train if args.split == "train" else curated_test
    curated_url = args.curated_train_url if args.split == "train" else args.curated_test_url
    ensure_download(curated_url, curated_path)
    examples = curated_examples(curated_path, task=args.task, sentinel=args.sentinel)
    curated_count = len(examples)
    mined_count = 0
    if args.include_mined:
        ensure_archive(args.url, archive, extract_dir)
        mined = mined_examples(
            extract_dir,
            limit=args.mined_limit,
            min_prob=args.mined_min_prob,
            seed=args.seed,
            task=args.task,
            sentinel=args.sentinel,
        )
        mined_count = len(mined)
        examples.extend(mined)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    corpus = "\n\n".join(examples) + "\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(corpus, encoding="utf-8")
    config = ConalaCorpusConfig(
        split=args.split,
        include_mined=args.include_mined,
        mined_limit=args.mined_limit,
        mined_min_prob=args.mined_min_prob,
        seed=args.seed,
        task=args.task,
        sentinel=args.sentinel,
    )
    metadata = {
        "config": asdict(config),
        "output": str(output.relative_to(ROOT)),
        "curated_source": str(curated_path.relative_to(ROOT)),
        "curated_examples": curated_count,
        "mined_examples": mined_count,
        "total_examples": len(examples),
        "total_chars": len(corpus),
        "unique_chars": len(set(corpus)),
        "chars": sorted(set(corpus)),
        "first_examples": examples[:10],
    }
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "examples": len(examples), "chars": len(corpus)}, indent=2))


if __name__ == "__main__":
    main()
