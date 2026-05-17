from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.prepare_regex_il_v3_traces import (  # noqa: E402
    CLASS_SPECS,
    Part,
    QuoteBuilder,
    class_words,
    count_word,
    make_class_part,
    make_part,
    make_ref_part,
    parse_component_counts,
    random_string,
    seq_il,
    seq_template,
)
from scripts.data.prepare_regex_il_v5_traces import make_example as make_v5_example  # noqa: E402
from scripts.data.prepare_regex_traces import ROOT  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "regex_il_v6_natural_queries_120k.txt"
DEFAULT_METADATA_OUTPUT = ROOT / "data" / "regex_il_v6_natural_queries_120k.sources.json"
CAPTURE_NAMES = ["a", "b", "c", "A", "B", "C"]


@dataclass(frozen=True)
class RegexIlV6Config:
    examples: int
    seed: int
    string_min_len: int
    string_max_len: int
    component_counts: str
    preview: int
    band_weights: dict[str, int]
    noise_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v6 regex IL traces with natural /regex query phrasing.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--examples", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=62026)
    parser.add_argument("--string-min-len", type=int, default=2)
    parser.add_argument("--string-max-len", type=int, default=9)
    parser.add_argument("--component-counts", default="2,3,3,4")
    parser.add_argument("--noise-rate", type=float, default=0.18)
    parser.add_argument("--preview", type=int, default=0)
    return parser.parse_args()


def literal_part(rng: random.Random, quotes: QuoteBuilder) -> Part:
    if rng.random() < 0.72:
        return make_ref_part(rng, quotes)
    return make_class_part(rng, quant="")


def capturable_part(rng: random.Random, quotes: QuoteBuilder) -> Part:
    if rng.random() < 0.78:
        kind = rng.choice(tuple(CLASS_SPECS))
        quant = rng.choice(["+", "+", "*", "?", "{2}", "{3}", "{2,4}", "{3,5}"])
        return Part(class_words(kind, quant, rng), kind + quant, CLASS_SPECS[kind]["regex"] + quant)
    return literal_part(rng, quotes)


def named_cap(part: Part, index: int) -> Part:
    return Part(part.words, f"CAP{index}({part.il})", f"({part.template})")


def q(value: str) -> str:
    return f"'{value}'"


def maybe_prefix(rng: random.Random) -> str:
    return rng.choice(["/regex ", "/regex ", "/r ", "/r ", "/regex please ", ""])


def maybe_noise(text: str, rng: random.Random, noise_rate: float) -> str:
    if rng.random() >= noise_rate:
        return text
    inserts = [
        "please ",
        "basically ",
        "if possible ",
        "for now ",
        "case sensitive ",
        "as a regex ",
        "in normal regex ",
    ]
    suffixes = [
        "",
        " thanks",
        " please",
        " if possible",
        " and keep it simple",
    ]
    return rng.choice(inserts) + text + rng.choice(suffixes)


def scope_phrase(anchor: str, rng: random.Random) -> str:
    if anchor == "FIND":
        return rng.choice(["In the text, ", "Find in the text: ", "Match in text: ", ""])
    if anchor == "START":
        return rng.choice(["At the start, ", "Match from the start: ", "Begin with "])
    if anchor == "END":
        return rng.choice(["At the end, ", "Match near the end: ", "End with "])
    if anchor == "FULL":
        return rng.choice(["Match on the whole string: ", "Whole string: ", "The whole thing should be "])
    raise ValueError(anchor)


def part_phrase(part: Part, rng: random.Random) -> str:
    if part.il.startswith("REF"):
        return rng.choice([part.words, f"match {part.words}"])
    if part.il.startswith("LIT"):
        return part.words
    return rng.choice([part.words, f"match {part.words}", f"one chunk of {part.words}"])


def capture_phrase(part: Part, name: str, rng: random.Random) -> str:
    return rng.choice(
        [
            f"capture {part_phrase(part, rng)} as {q(name)}",
            f"save {part_phrase(part, rng)} as {q(name)}",
            f"call {part_phrase(part, rng)} {q(name)}",
            f"remember {part_phrase(part, rng)} as {q(name)}",
            f"extract {part_phrase(part, rng)} into {q(name)}",
            f"name {part_phrase(part, rng)} {q(name)}",
            f"grab {part_phrase(part, rng)} as {q(name)}",
        ]
    )


def select_phrase(names: list[str], rng: random.Random) -> str:
    quoted = [q(name) for name in names]
    joined = ", ".join(quoted)
    if len(names) == 1:
        return rng.choice(
            [
                f"Select {quoted[0]}.",
                f"Return {quoted[0]}.",
                f"Output {quoted[0]}.",
                f"Give back the captured {quoted[0]}.",
            ]
        )
    return rng.choice(
        [
            f"Select {joined}.",
            f"Return {joined}.",
            f"Output {joined}.",
            f"Give back the captured parts {joined}.",
        ]
    )


def join_steps(steps: list[str], rng: random.Random) -> str:
    text = steps[0]
    for step in steps[1:]:
        text += rng.choice([", then ", ", followed by ", ", and then ", " then "]) + step
    return text


def make_named_capture_query(rng: random.Random, quotes: QuoteBuilder, component_counts: list[int], noise_rate: float) -> dict[str, object]:
    count = max(2, rng.choice(component_counts))
    capture_count = min(rng.choice([1, 1, 2, 2, 3]), count)
    capture_slots = sorted(rng.sample(range(count), k=capture_count))
    names = CAPTURE_NAMES[:capture_count]
    parts = []
    name_by_slot = dict(zip(capture_slots, names))
    cap_index = 1
    for slot in range(count):
        base = capturable_part(rng, quotes) if slot in name_by_slot else make_part(rng, quotes, allow_alt=True)
        if slot in name_by_slot:
            base = named_cap(base, cap_index)
            cap_index += 1
        parts.append(base)

    anchor = rng.choices(["FIND", "START", "END", "FULL"], weights=[45, 20, 15, 20], k=1)[0]
    template = seq_template(parts)
    if anchor == "START":
        template = "^" + template
    elif anchor == "END":
        template = template + "$"
    elif anchor == "FULL":
        template = "^" + template + "$"

    steps = []
    cap_number = 0
    selected_names = []
    for slot, part in enumerate(parts):
        if slot in name_by_slot:
            cap_number += 1
            selected_names.append(name_by_slot[slot])
            steps.append(capture_phrase(Part(part.words, part.il, part.template), name_by_slot[slot], rng))
        else:
            steps.append(part_phrase(part, rng))
    query = maybe_prefix(rng) + scope_phrase(anchor, rng) + join_steps(steps, rng) + ". " + select_phrase(selected_names, rng)
    query = maybe_noise(query, rng, noise_rate)
    return {
        "band": "named_capture",
        "refs": quotes.refs,
        "input": query,
        "il": f"{anchor}; {seq_il(parts)}; SELECT({','.join(f'CAP{i}' for i in range(1, cap_number + 1))});",
        "template": template,
    }


def make_word_query(rng: random.Random, quotes: QuoteBuilder, noise_rate: float) -> dict[str, object]:
    name = rng.choice(["a", "A"])
    prefix = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 4)))
    suffix = rng.choice(["ing", "ed", "tion", "ly", "er"])
    body_kind = rng.choice(["ANY_LETTER", "LOWER", "UPPER", "VOWEL"])
    body = Part(class_words(body_kind, "+", rng), body_kind + "+", CLASS_SPECS[body_kind]["regex"] + "+")
    pattern_type = rng.choice(["starts", "ends", "contains"])
    if pattern_type == "starts":
        parts = [Part(f'the word "{prefix}"', "REF0", "<0>"), named_cap(body, 1)]
        quotes.refs.append(prefix)
        phrase = rng.choice(
            [
                f"/regex Match on words: start with {q(prefix)}, then capture the rest as {q(name)}. Select {q(name)}.",
                f"/regex Find words that start with {q(prefix)}, then save the rest as {q(name)}. Return {q(name)}.",
            ]
        )
    elif pattern_type == "ends":
        parts = [named_cap(body, 1), Part(f'the ending "{suffix}"', "REF0", "<0>")]
        quotes.refs.append(suffix)
        phrase = rng.choice(
            [
                f"/regex Match on words: capture the start as {q(name)}, then end with {q(suffix)}. Select {q(name)}.",
                f"/regex Find words that end with {q(suffix)} and remember the part before it as {q(name)}. Output {q(name)}.",
            ]
        )
    else:
        middle = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(2, 4)))
        tail_kind = rng.choice(["ANY_LETTER", "LOWER", "UPPER", "VOWEL"])
        tail = Part(class_words(tail_kind, "*", rng), tail_kind + "*", CLASS_SPECS[tail_kind]["regex"] + "*")
        parts = [named_cap(body, 1), Part(f'the middle "{middle}"', "REF0", "<0>"), tail]
        quotes.refs.append(middle)
        phrase = rng.choice(
            [
                f"/regex Match on words: capture letters as {q(name)}, then {q(middle)}, then any letters. Select {q(name)}.",
                f"/regex Find words that contain {q(middle)} after captured letters named {q(name)}. Return {q(name)}.",
            ]
        )
    template = r"\b" + seq_template(parts) + r"\b"
    return {
        "band": "words_that",
        "refs": quotes.refs,
        "input": maybe_noise(phrase, rng, noise_rate),
        "il": f"WORD; {seq_il(parts)}; SELECT(CAP1);",
        "template": template,
    }


def make_replace_query(rng: random.Random, quotes: QuoteBuilder, noise_rate: float) -> dict[str, object]:
    first = named_cap(capturable_part(rng, quotes), 1)
    sep = rng.choice([Part("whitespace", "SPACE+", r"\s+"), Part('","', 'LIT(",")', ",")])
    second = named_cap(capturable_part(rng, quotes), 2)
    parts = [first, sep, second]
    repl = rng.choice(
        [
            "'b', \", \", 'a'",
            "'a', \"-\", 'b'",
            "\"[\", 'a', \"]\", 'b'",
            "'b', \" #\", 'a'",
        ]
    )
    whole_string = rng.random() < 0.35
    if whole_string:
        query = f"/regex Match on the whole string: capture {first.words} as 'a', then {sep.words}, then capture {second.words} as 'b'. Replace with {repl}."
        anchor = "FULL"
        template = "^" + seq_template(parts) + "$"
    else:
        query = rng.choice(
            [
                f"/regex In the text, save {first.words} as 'a', then match {sep.words}, then save {second.words} as 'b'. Rewrite as {repl}.",
                f"/r capture {first.words} as 'a', then {sep.words}, then capture {second.words} as 'b'; replace the match with {repl}.",
            ]
        )
        anchor = "FIND"
        template = seq_template(parts)
    return {
        "band": "replace",
        "refs": quotes.refs,
        "input": maybe_noise(query, rng, noise_rate),
        "il": f"{anchor}; {seq_il(parts)}; REPLACE({repl});",
        "template": template + " => " + repl,
    }


def make_order_query(rng: random.Random, quotes: QuoteBuilder, noise_rate: float) -> dict[str, object]:
    left = literal_part(rng, quotes)
    right = literal_part(rng, quotes)
    mode = rng.choice(["BEFORE", "AFTER"])
    if mode == "BEFORE":
        phrase = rng.choice(
            [
                f"/regex In the text, match {left.words} before {right.words}. Select the whole match.",
                f"/regex Find text where {left.words} appears before {right.words}.",
            ]
        )
        template = f"{left.template}.*{right.template}"
    else:
        phrase = rng.choice(
            [
                f"/regex In the text, match {left.words} after {right.words}. Select the whole match.",
                f"/regex Find text where {left.words} comes after {right.words}.",
            ]
        )
        template = f"{right.template}.*{left.template}"
    return {
        "band": "natural_order",
        "refs": quotes.refs,
        "input": maybe_noise(phrase, rng, noise_rate),
        "il": f"{mode}; LEFT({left.il}); RIGHT({right.il});",
        "template": template,
    }


def make_v5_naturalized(rng: random.Random, component_counts: list[int], string_min_len: int, string_max_len: int, noise_rate: float) -> dict[str, object]:
    example = make_v5_example(
        rng,
        component_counts=component_counts,
        string_min_len=string_min_len,
        string_max_len=string_max_len,
    )
    prefix = maybe_prefix(rng)
    text = str(example["input"])
    variants = [
        prefix + text,
        prefix + "In the text, " + text,
        prefix + "Match " + text,
        prefix + "Find " + text,
    ]
    return {**example, "band": "v5_naturalized", "input": maybe_noise(rng.choice(variants), rng, noise_rate)}


def make_example(
    rng: random.Random,
    *,
    component_counts: list[int],
    string_min_len: int,
    string_max_len: int,
    noise_rate: float,
) -> dict[str, object]:
    quotes = QuoteBuilder(rng, min_len=string_min_len, max_len=string_max_len)
    bands = {
        "v5_naturalized": 35,
        "named_capture": 34,
        "words_that": 12,
        "replace": 10,
        "natural_order": 9,
    }
    band = rng.choices(tuple(bands), weights=tuple(bands.values()), k=1)[0]
    if band == "v5_naturalized":
        return make_v5_naturalized(rng, component_counts, string_min_len, string_max_len, noise_rate)
    if band == "named_capture":
        return make_named_capture_query(rng, quotes, component_counts, noise_rate)
    if band == "words_that":
        return make_word_query(rng, quotes, noise_rate)
    if band == "replace":
        return make_replace_query(rng, quotes, noise_rate)
    if band == "natural_order":
        return make_order_query(rng, quotes, noise_rate)
    raise ValueError(f"unknown band: {band}")


def format_example(example: dict[str, object]) -> str:
    return f"Input:\n{example['input']}\n\nIL:\n{example['il']}\n\nTemplate:\n{example['template']}"


def main() -> None:
    args = parse_args()
    if args.examples <= 0:
        raise ValueError("--examples must be positive")
    if not 0 <= args.noise_rate <= 1:
        raise ValueError("--noise-rate must be between 0 and 1")
    component_counts = parse_component_counts(args.component_counts)
    band_weights = {"v5_naturalized": 35, "named_capture": 34, "words_that": 12, "replace": 10, "natural_order": 9}
    config = RegexIlV6Config(
        examples=args.examples,
        seed=args.seed,
        string_min_len=args.string_min_len,
        string_max_len=args.string_max_len,
        component_counts=args.component_counts,
        preview=args.preview,
        band_weights=band_weights,
        noise_rate=args.noise_rate,
    )
    rng = random.Random(args.seed)
    examples = [
        make_example(
            rng,
            component_counts=component_counts,
            string_min_len=args.string_min_len,
            string_max_len=args.string_max_len,
            noise_rate=args.noise_rate,
        )
        for _ in range(args.examples)
    ]
    corpus = "\n\n".join(format_example(example) for example in examples) + "\n"
    actual_bands = sorted({str(example["band"]) for example in examples})
    metadata = {
        "config": asdict(config),
        "output": str(args.output.relative_to(ROOT) if args.output.is_absolute() else args.output),
        "total_examples": len(examples),
        "total_chars": len(corpus),
        "unique_chars": len(set(corpus)),
        "chars": sorted(set(corpus)),
        "band_counts": {band: sum(1 for example in examples if example["band"] == band) for band in actual_bands},
        "generator_band_weights": band_weights,
        "first_examples": examples[:30],
        "note": "v6 uses /regex-style natural queries, named capture wording, word-scope examples, replacement tasks, and noisy phrasing. IL still uses numbered CAPn internally.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8")
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.preview:
        for index, example in enumerate(examples[: args.preview], start=1):
            print(f"--- example {index} [{example['band']}] ---")
            print(format_example(example))
    else:
        print(json.dumps({"output": str(args.output), "metadata_output": str(args.metadata_output), "examples": len(examples), "chars": len(corpus)}, indent=2))


if __name__ == "__main__":
    main()
