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
    format_example,
    make_class_part,
    make_example as make_v3_example,
    make_part,
    make_ref_part,
    parse_component_counts,
    random_string,
    seq_il,
    seq_template,
)
from scripts.data.prepare_regex_traces import ROOT  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "regex_il_v5_clear_capture_120k.txt"
DEFAULT_METADATA_OUTPUT = ROOT / "data" / "regex_il_v5_clear_capture_120k.sources.json"


@dataclass(frozen=True)
class RegexIlV5Config:
    examples: int
    seed: int
    string_min_len: int
    string_max_len: int
    component_counts: str
    preview: int
    band_weights: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v5 regex IL traces with clearer capture/select examples.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--examples", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=51226)
    parser.add_argument("--string-min-len", type=int, default=2)
    parser.add_argument("--string-max-len", type=int, default=9)
    parser.add_argument("--component-counts", default="2,3,3,4")
    parser.add_argument("--preview", type=int, default=0)
    return parser.parse_args()


def capture_class_part(rng: random.Random, *, quant: str | None = None) -> Part:
    kind = rng.choice(tuple(CLASS_SPECS))
    if quant is None:
        quant = rng.choice(["+", "+", "{2}", "{3}", "{2,4}", "*"])
    il = kind + quant
    template = CLASS_SPECS[kind]["regex"] + quant
    return Part(class_words(kind, quant, rng), il, template)


def capture_ref_part(rng: random.Random, quotes: QuoteBuilder) -> Part:
    ref, value = quotes.add()
    words = rng.choice(
        [
            f'the text "{value}"',
            f'the string "{value}"',
            f'the exact text "{value}"',
            f'literal "{value}"',
        ]
    )
    return Part(words, f"REF{ref}", f"<{ref}>")


def capturable_part(rng: random.Random, quotes: QuoteBuilder) -> Part:
    if rng.random() < 0.78:
        return capture_class_part(rng)
    return capture_ref_part(rng, quotes)


def cap(part: Part, index: int) -> Part:
    return Part(part.words, f"CAP{index}({part.il})", f"({part.template})")


def capture_command(rng: random.Random) -> str:
    return rng.choice(["capture", "select", "extract", "return"])


def capture_words(part: Part, rng: random.Random) -> str:
    command = capture_command(rng)
    return rng.choice(
        [
            f"{command} {part.words}",
            f"{command} the part matching {part.words}",
        ]
    )


def join_requested_words(parts: list[Part], capture_slots: set[int], rng: random.Random) -> str:
    text = capture_words(parts[0], rng) if 0 in capture_slots else parts[0].words
    for slot, part in enumerate(parts[1:], start=1):
        linker = rng.choice([" followed by ", " then ", " and then "])
        if slot in capture_slots:
            linker = rng.choice([" followed by ", ", then ", ", and then "])
        text += linker
        text += capture_words(part, rng) if slot in capture_slots else part.words
    return text


def make_capture_sequence_example(
    rng: random.Random,
    quotes: QuoteBuilder,
    component_counts: list[int],
) -> dict[str, object]:
    count = rng.choice(component_counts)
    count = max(2, count)
    capture_count = 1 if rng.random() < 0.82 else 2
    capture_count = min(capture_count, count)
    capture_slots = rng.sample(range(count), k=capture_count)
    capture_slots.sort()
    capture_slot_set = set(capture_slots)
    parts: list[Part] = []
    capture_index = 1
    for slot in range(count):
        part = capturable_part(rng, quotes) if slot in capture_slot_set else make_part(rng, quotes, allow_alt=True)
        if slot in capture_slot_set:
            part = cap(part, capture_index)
            capture_index += 1
        parts.append(part)

    anchor = rng.choices(["FIND", "START", "END", "FULL"], weights=[30, 35, 20, 15], k=1)[0]
    if anchor == "FIND":
        body = seq_template(parts)
        scope = rng.choice(["lines containing ", "lines with ", "items with "])
    elif anchor == "START":
        body = "^" + seq_template(parts)
        scope = rng.choice(["lines starting with ", "lines that begin with ", "items beginning with "])
    elif anchor == "END":
        body = seq_template(parts) + "$"
        scope = rng.choice(["lines ending with ", "lines that end with ", "items ending in "])
    else:
        body = "^" + seq_template(parts) + "$"
        scope = rng.choice(["the whole line is ", "the entire line is ", "lines made of "])

    selected = ",".join(f"CAP{idx}" for idx in range(1, capture_index))
    requested = join_requested_words(parts, capture_slot_set, rng)
    input_text = rng.choice(
        [
            f"{scope}{requested}",
            f"{scope}{requested}; output the captured part"
            if len(capture_slots) == 1
            else f"{scope}{requested}; output both captured parts",
        ]
    )

    return {
        "band": "capture_sequence",
        "refs": quotes.refs,
        "input": input_text,
        "il": f"{anchor}; {seq_il(parts)}; SELECT({selected});",
        "template": body,
    }


def join_capture_words(parts: list[Part], rng: random.Random) -> str:
    if len(parts) == 1:
        return parts[0].words
    text = parts[0].words
    for part in parts[1:]:
        text += rng.choice([" followed by ", " then ", " and then "]) + part.words
    return text


def make_capture_between_example(rng: random.Random, quotes: QuoteBuilder) -> dict[str, object]:
    left = make_ref_part(rng, quotes) if rng.random() < 0.65 else make_class_part(rng, quant="")
    middle = cap(capturable_part(rng, quotes), 1)
    right = make_ref_part(rng, quotes) if rng.random() < 0.65 else make_class_part(rng, quant="")
    anchor = rng.choice(["FIND", "START", "END"])
    if anchor == "FIND":
        template = left.template + middle.template + right.template
        scope = rng.choice(["lines with ", "lines containing "])
    elif anchor == "START":
        template = "^" + left.template + middle.template + right.template
        scope = rng.choice(["lines starting with ", "items beginning with "])
    else:
        template = left.template + middle.template + right.template + "$"
        scope = rng.choice(["lines ending with ", "items ending in "])
    input_text = rng.choice(
        [
            f"{scope}{left.words} then {capture_words(middle, rng)} then {right.words}",
            f"{scope}{left.words} followed by {capture_words(middle, rng)} followed by {right.words}",
            f"{capture_words(middle, rng)} between {left.words} and {right.words}",
        ]
    )
    return {
        "band": "capture_between",
        "refs": quotes.refs,
        "input": input_text,
        "il": f"{anchor}; SEQ({left.il},{middle.il},{right.il}); SELECT(CAP1);",
        "template": template,
    }


def make_capture_repeat_example(rng: random.Random, quotes: QuoteBuilder) -> dict[str, object]:
    kind = rng.choice(tuple(CLASS_SPECS))
    count = rng.choice([2, 3, 4])
    count_text = count_word(count, rng)
    words = class_words(kind, f"{{{count}}}", rng)
    il_atom = f"{kind}{{{count}}}"
    template_atom = f"{CLASS_SPECS[kind]['regex']}{{{count}}}"
    anchor = rng.choice(["FIND", "START", "FULL"])
    captured_template = f"({template_atom})"
    if anchor == "FIND":
        template = captured_template
        scope = "lines with "
    elif anchor == "START":
        template = "^" + captured_template
        scope = "lines starting with "
    else:
        template = "^" + captured_template + "$"
        scope = "lines made of "
    if anchor == "FIND":
        input_text = rng.choice(
            [
                f"{scope}capture {words}",
                f"{scope}select the run of {words}",
                f"{scope}return the whole run of {words}",
            ]
        )
    elif anchor == "START":
        input_text = rng.choice(
            [
                f"{scope}capture {words}",
                f"{scope}select the run of {words}",
                f"{scope}return the whole run of {words}",
            ]
        )
    else:
        input_text = rng.choice(
            [
                f"{scope}capture {words} only",
                f"{scope}select the run of {words}",
                f"{scope}return the whole run of {words}",
            ]
        )
    return {
        "band": "capture_repeat",
        "refs": quotes.refs,
        "input": input_text,
        "il": f"{anchor}; CAP1({il_atom}); SELECT(CAP1);",
        "template": template,
    }


def make_example(
    rng: random.Random,
    *,
    component_counts: list[int],
    string_min_len: int,
    string_max_len: int,
) -> dict[str, object]:
    quotes = QuoteBuilder(rng, min_len=string_min_len, max_len=string_max_len)
    bands = {
        "v3": 60,
        "capture_sequence": 28,
        "capture_between": 8,
        "capture_repeat": 4,
    }
    band = rng.choices(tuple(bands), weights=tuple(bands.values()), k=1)[0]
    if band == "v3":
        return make_v3_example(
            rng,
            component_counts=component_counts,
            string_min_len=string_min_len,
            string_max_len=string_max_len,
        )
    if band == "capture_sequence":
        example = make_capture_sequence_example(rng, quotes, component_counts)
    elif band == "capture_between":
        example = make_capture_between_example(rng, quotes)
    elif band == "capture_repeat":
        example = make_capture_repeat_example(rng, quotes)
    else:
        raise ValueError(f"unknown band: {band}")
    return {**example, "refs": quotes.refs}


def main() -> None:
    args = parse_args()
    if args.examples <= 0:
        raise ValueError("--examples must be positive")
    component_counts = parse_component_counts(args.component_counts)
    band_weights = {"v3": 60, "capture_sequence": 28, "capture_between": 8, "capture_repeat": 4}
    config = RegexIlV5Config(
        examples=args.examples,
        seed=args.seed,
        string_min_len=args.string_min_len,
        string_max_len=args.string_max_len,
        component_counts=args.component_counts,
        preview=args.preview,
        band_weights=band_weights,
    )
    rng = random.Random(args.seed)
    examples = [
        make_example(
            rng,
            component_counts=component_counts,
            string_min_len=args.string_min_len,
            string_max_len=args.string_max_len,
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
        "note": "v5 extends v3 with explicit capture/select wording. Inputs contain raw quoted strings and no REF tokens; IL uses REFn placeholders assigned by quote order and CAPn for captures.",
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
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "metadata_output": str(args.metadata_output),
                    "examples": len(examples),
                    "chars": len(corpus),
                    "band_counts": metadata["band_counts"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
