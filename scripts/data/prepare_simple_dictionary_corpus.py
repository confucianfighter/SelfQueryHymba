from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DICTIONARY_URL = (
    "https://raw.githubusercontent.com/nightblade9/simple-english-dictionary/main/processed/filtered.json"
)
DEFAULT_CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
DEFAULT_FREQUENCY_URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"
)


@dataclass(frozen=True)
class SimpleDictionaryConfig:
    source: str
    cmudict_source: str | None
    examples: int
    seed: int
    task: str
    sentinel: str
    include_pronunciation: bool
    require_synonyms: bool
    require_pronunciation: bool
    max_synonyms: int
    max_definition_chars: int
    frequency_source: str | None
    top_words: int | None


PHONE_TEXT = {
    "AA": "ah",
    "AE": "a",
    "AH": "uh",
    "AO": "aw",
    "AW": "ow",
    "AY": "eye",
    "B": "b",
    "CH": "ch",
    "D": "d",
    "DH": "th",
    "EH": "eh",
    "ER": "er",
    "EY": "ay",
    "F": "f",
    "G": "g",
    "HH": "h",
    "IH": "ih",
    "IY": "ee",
    "JH": "j",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ng",
    "OW": "oh",
    "OY": "oy",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "sh",
    "T": "t",
    "TH": "th",
    "UH": "uu",
    "UW": "oo",
    "V": "v",
    "W": "w",
    "Y": "y",
    "Z": "z",
    "ZH": "zh",
}
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SimpleDictionary word-definition corpus.")
    parser.add_argument("--dictionary-json", type=Path, default=None)
    parser.add_argument("--dictionary-url", default=DEFAULT_DICTIONARY_URL)
    parser.add_argument("--cmudict-path", type=Path, default=None)
    parser.add_argument("--cmudict-url", default=DEFAULT_CMUDICT_URL)
    parser.add_argument("--frequency-path", type=Path, default=None)
    parser.add_argument("--frequency-url", default=DEFAULT_FREQUENCY_URL)
    parser.add_argument(
        "--top-words",
        type=int,
        default=None,
        help="If set, keep only dictionary entries whose headword appears in the top N ranked words.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--examples-output", type=Path, default=None)
    parser.add_argument("--examples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--task", default="simple_dictionary")
    parser.add_argument("--sentinel", default="<END>")
    parser.add_argument("--include-pronunciation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-synonyms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-pronunciation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-synonyms", type=int, default=6)
    parser.add_argument("--max-definition-chars", type=int, default=180)
    parser.add_argument("--min-word-len", type=int, default=2)
    parser.add_argument("--max-word-len", type=int, default=24)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_text_or_url(path: Path | None, url: str) -> tuple[str, str]:
    if path is not None:
        resolved = resolve(path)
        return resolved.read_text(encoding="utf-8"), str(resolved)
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8"), url


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_word(value: str) -> str:
    return clean_text(value).lower()


def valid_word(word: str, *, min_len: int, max_len: int) -> bool:
    return (
        min_len <= len(word) <= max_len
        and re.fullmatch(r"[a-z][a-z '-]*[a-z]", word) is not None
        and "  " not in word
    )


def parse_cmudict(text: str) -> dict[str, list[str]]:
    pronunciations: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";;;"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word = re.sub(r"\(\d+\)$", "", parts[0]).lower()
        if not re.fullmatch(r"[a-z][a-z'-]*", word):
            continue
        pronunciations.setdefault(word, parts[1:])
    return pronunciations


def parse_frequency_ranks(text: str) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for raw_line in text.splitlines():
        line = clean_word(raw_line.split()[0] if raw_line.split() else "")
        if line and line not in ranks:
            ranks[line] = len(ranks) + 1
    return ranks


def split_phone(phone: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Z]+)([012])?", phone)
    if match is None:
        return phone, 0
    return match.group(1), int(match.group(2) or "0")


def phone_base(phone: str) -> str:
    return split_phone(phone)[0]


def is_vowel(phone: str) -> bool:
    return phone_base(phone) in VOWELS


def syllables_from_phones(phones: list[str]) -> list[tuple[list[str], int]]:
    vowel_positions = [idx for idx, phone in enumerate(phones) if is_vowel(phone)]
    if not vowel_positions:
        return [(phones, 0)]
    syllables: list[tuple[list[str], int]] = []
    start = 0
    for vowel_index, vowel_pos in enumerate(vowel_positions):
        next_vowel = vowel_positions[vowel_index + 1] if vowel_index + 1 < len(vowel_positions) else len(phones)
        if vowel_index + 1 < len(vowel_positions):
            consonants_between = next_vowel - vowel_pos - 1
            end = vowel_pos + 1 + max(0, consonants_between - 1)
        else:
            end = len(phones)
        chunk = phones[start:end]
        stress = max((split_phone(phone)[1] for phone in chunk if is_vowel(phone)), default=0)
        syllables.append((chunk, stress))
        start = end
    return syllables


def respell_pronunciation(phones: list[str]) -> str:
    parts = []
    for chunk, stress in syllables_from_phones(phones):
        text = "".join(PHONE_TEXT.get(phone_base(phone), phone_base(phone).lower()) for phone in chunk)
        if stress == 1:
            text = text.upper()
        parts.append(text)
    return "-".join(part for part in parts if part)


def meaning_rows(word: str, entry: dict[str, Any], pronunciation: str | None, max_synonyms: int) -> list[dict[str, object]]:
    entry_synonyms = [clean_word(value) for value in entry.get("SYNONYMS", []) if clean_text(value)]
    rows = []
    for meaning in entry.get("MEANINGS", []):
        if not isinstance(meaning, list) or len(meaning) < 2:
            continue
        pos = clean_text(meaning[0]).lower()
        definition = clean_text(meaning[1])
        meaning_synonyms = []
        if len(meaning) >= 3 and isinstance(meaning[2], list):
            meaning_synonyms = [clean_word(value) for value in meaning[2] if clean_text(value)]
        synonyms = []
        seen = {word}
        for synonym in meaning_synonyms + entry_synonyms:
            if synonym and synonym not in seen and valid_word(synonym, min_len=2, max_len=40):
                synonyms.append(synonym)
                seen.add(synonym)
            if len(synonyms) >= max_synonyms:
                break
        rows.append(
            {
                "word": word,
                "part_of_speech": pos,
                "definition": definition,
                "synonyms": synonyms,
                "pronunciation": pronunciation,
            }
        )
    return rows


def format_example(row: dict[str, object], *, task: str, sentinel: str, include_pronunciation: bool) -> str:
    synonyms = row["synonyms"]
    synonym_text = "; ".join(synonyms) if synonyms else "none"
    output_lines = [
        f"part_of_speech: {row['part_of_speech']}",
        f"definition: {row['definition']}",
    ]
    if include_pronunciation and row.get("pronunciation"):
        output_lines.append(f"pronunciation: {row['pronunciation']}")
    return (
        f"Task: {task}\n"
        "Input:\n"
        f"word: {row['word']}\n"
        f"synonyms: {synonym_text}\n\n"
        "Output:\n"
        + "\n".join(output_lines)
        + f"\n{sentinel}"
    )


def main() -> None:
    args = parse_args()
    dictionary_text, dictionary_source = read_text_or_url(args.dictionary_json, args.dictionary_url)
    dictionary = json.loads(dictionary_text)
    frequency_source = None
    frequency_ranks: dict[str, int] = {}
    if args.top_words is not None:
        frequency_text, frequency_source = read_text_or_url(args.frequency_path, args.frequency_url)
        frequency_ranks = parse_frequency_ranks(frequency_text)
    cmudict_source = None
    pronunciations: dict[str, list[str]] = {}
    if args.include_pronunciation:
        cmudict_text, cmudict_source = read_text_or_url(args.cmudict_path, args.cmudict_url)
        pronunciations = parse_cmudict(cmudict_text)

    rows = []
    for raw_word, entry in dictionary.items():
        word = clean_word(raw_word)
        if not valid_word(word, min_len=args.min_word_len, max_len=args.max_word_len):
            continue
        rank = frequency_ranks.get(word)
        if args.top_words is not None and (rank is None or rank > args.top_words):
            continue
        if not isinstance(entry, dict):
            continue
        pronunciation = respell_pronunciation(pronunciations[word]) if word in pronunciations else None
        if args.require_pronunciation and pronunciation is None:
            continue
        for row in meaning_rows(word, entry, pronunciation, args.max_synonyms):
            if args.require_synonyms and not row["synonyms"]:
                continue
            if len(str(row["definition"])) > args.max_definition_chars:
                continue
            row["frequency_rank"] = rank
            rows.append(row)

    if not rows:
        raise ValueError("no usable dictionary rows after filtering")
    if args.examples > len(rows):
        raise ValueError(f"only {len(rows)} usable rows; requested {args.examples}")

    rng = random.Random(args.seed)
    rows = sorted(rows, key=lambda row: (row["frequency_rank"] is None, row["frequency_rank"] or 10**9, str(row["word"])))
    sampled = rows[: args.examples]
    rng.shuffle(sampled)
    corpus = "\n\n".join(
        format_example(row, task=args.task, sentinel=args.sentinel, include_pronunciation=args.include_pronunciation)
        for row in sampled
    ) + "\n"

    output = resolve(args.output)
    metadata_output = resolve(args.metadata_output) if args.metadata_output else output.with_suffix(".sources.json")
    examples_output = resolve(args.examples_output) if args.examples_output else output.with_suffix(".examples.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    examples_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(corpus, encoding="utf-8")
    examples_output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in sampled) + "\n", encoding="utf-8")

    metadata = {
        "config": asdict(
            SimpleDictionaryConfig(
                source=dictionary_source,
                cmudict_source=cmudict_source,
                examples=args.examples,
                seed=args.seed,
                task=args.task,
                sentinel=args.sentinel,
                include_pronunciation=args.include_pronunciation,
                require_synonyms=args.require_synonyms,
                require_pronunciation=args.require_pronunciation,
                max_synonyms=args.max_synonyms,
                max_definition_chars=args.max_definition_chars,
                frequency_source=frequency_source,
                top_words=args.top_words,
            )
        ),
        "output": str(output.relative_to(ROOT)),
        "examples_output": str(examples_output.relative_to(ROOT)),
        "total_available_rows": len(rows),
        "total_examples": len(sampled),
        "total_chars": len(corpus),
        "unique_chars": len(set(corpus)),
        "chars": sorted(set(corpus)),
        "first_examples": corpus.split("\n\n")[:5],
    }
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "examples": len(sampled), "chars": len(corpus)}, indent=2))


if __name__ == "__main__":
    main()
