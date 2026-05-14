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

from scripts.data.prepare_regex_traces import ROOT


DEFAULT_OUTPUT = ROOT / "data" / "regex_il_v3_linefilter_100k.txt"
DEFAULT_METADATA_OUTPUT = ROOT / "data" / "regex_il_v3_linefilter_100k.sources.json"

ALPHANUM = "abcdefghijklmnopqrstuvwxyz0123456789"
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

CLASS_SPECS = {
    "DIGIT": {
        "regex": r"\d",
        "singular": ["digit", "number", "numeral", "numeric character"],
        "plural": ["digits", "numbers", "numerals", "numeric characters"],
    },
    "ANY_LETTER": {
        "regex": "[A-Za-z]",
        "singular": ["letter of any case", "alphabetic character"],
        "plural": ["letters of any case", "alphabetic characters"],
    },
    "LOWER": {
        "regex": "[a-z]",
        "singular": ["lowercase letter", "lower-case letter", "small letter"],
        "plural": ["lowercase letters", "lower-case letters", "small letters"],
    },
    "UPPER": {
        "regex": "[A-Z]",
        "singular": ["uppercase letter", "capital letter", "upper-case letter"],
        "plural": ["uppercase letters", "capital letters", "upper-case letters"],
    },
    "WORD": {
        "regex": r"\w",
        "singular": ["word character", "letter, digit, or underscore"],
        "plural": ["word characters", "letters, digits, or underscores"],
    },
    "SPACE": {
        "regex": r"\s",
        "singular": ["space", "whitespace", "blank character"],
        "plural": ["spaces", "whitespace characters", "blank characters"],
    },
    "VOWEL": {
        "regex": "[AEIOUaeiou]",
        "singular": ["vowel"],
        "plural": ["vowels"],
    },
}

VISIBLE_LITERALS = {
    "dash": ("-", "-"),
    "dot": (".", r"\."),
    "underscore": ("_", "_"),
    "slash": ("/", r"\/"),
    "colon": (":", ":"),
    "at sign": ("@", "@"),
    "plus sign": ("+", r"\+"),
}


@dataclass(frozen=True)
class RegexIlV3Config:
    examples: int
    seed: int
    string_min_len: int
    string_max_len: int
    component_counts: str
    preview: int
    band_weights: dict[str, int]


@dataclass(frozen=True)
class Part:
    words: str
    il: str
    template: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v3 natural-language regex traces with a compact IL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--examples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=12512)
    parser.add_argument("--string-min-len", type=int, default=2)
    parser.add_argument("--string-max-len", type=int, default=8)
    parser.add_argument("--component-counts", default="2,2,3")
    parser.add_argument("--preview", type=int, default=0)
    return parser.parse_args()


def parse_component_counts(value: str) -> list[int]:
    counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not counts or any(count < 1 for count in counts):
        raise ValueError("--component-counts must contain positive integers")
    return counts


class QuoteBuilder:
    def __init__(self, rng: random.Random, *, min_len: int, max_len: int) -> None:
        self.rng = rng
        self.min_len = min_len
        self.max_len = max_len
        self.refs: list[str] = []

    def add(self) -> tuple[int, str]:
        value = random_string(self.rng, self.min_len, self.max_len)
        while value in self.refs or value.lower().startswith("ref"):
            value = random_string(self.rng, self.min_len, self.max_len)
        self.refs.append(value)
        return len(self.refs) - 1, value


def random_string(rng: random.Random, min_len: int, max_len: int) -> str:
    if min_len < 1 or max_len < min_len:
        raise ValueError("invalid string length bounds")
    first = rng.choice("abcdefghijklmnopqrstuvwxyz")
    length = rng.randint(min_len, max_len)
    return first + "".join(rng.choice(ALPHANUM) for _ in range(length - 1))


def count_word(n: int, rng: random.Random) -> str:
    return rng.choice([str(n), NUMBER_WORDS.get(n, str(n))])


def quant_choice(rng: random.Random) -> str:
    return rng.choice(["", "", "+", "*", "?", "{2}", "{3}", "{4}", "{2,4}", "{3,5}"])


def apply_quant(base_il: str, base_regex: str, quant: str) -> tuple[str, str]:
    if quant == "":
        return base_il, base_regex
    return f"{base_il}{quant}", f"{base_regex}{quant}"


def class_words(kind: str, quant: str, rng: random.Random) -> str:
    spec = CLASS_SPECS[kind]
    singular = rng.choice(spec["singular"])
    plural = rng.choice(spec["plural"])
    article = "an" if singular[0].lower() in "aeiou" else "a"
    if quant == "":
        return rng.choice([f"{article} {singular}", f"one {singular}", f"a single {singular}"])
    if quant == "+":
        return rng.choice([f"one or more {plural}", f"at least one {singular}", f"{plural} one or more times"])
    if quant == "*":
        return rng.choice([f"zero or more {plural}", f"any number of {plural}", f"{plural} zero or more times"])
    if quant == "?":
        return rng.choice([f"an optional {singular}", f"zero or one {singular}", f"maybe a {singular}"])
    if quant.startswith("{") and "," not in quant:
        count = int(quant.strip("{}"))
        text = count_word(count, rng)
        return rng.choice([f"exactly {text} {plural}", f"{text} {plural}", f"precisely {text} {plural}"])
    if quant.startswith("{"):
        lo, hi = quant.strip("{}").split(",")
        lo_text = count_word(int(lo), rng)
        hi_text = count_word(int(hi), rng)
        return rng.choice(
            [
                f"between {lo_text} and {hi_text} {plural}",
                f"{lo_text} to {hi_text} {plural}",
                f"from {lo_text} through {hi_text} {plural}",
            ]
        )
    raise ValueError(f"unknown quantifier: {quant}")


def make_class_part(rng: random.Random, *, quant: str | None = None) -> Part:
    kind = rng.choice(tuple(CLASS_SPECS))
    quant = quant_choice(rng) if quant is None else quant
    il, template = apply_quant(kind, CLASS_SPECS[kind]["regex"], quant)
    return Part(class_words(kind, quant, rng), il, template)


def make_ref_part(rng: random.Random, quotes: QuoteBuilder) -> Part:
    ref, value = quotes.add()
    phrase = rng.choice(
        [
            f'"{value}"',
            f"the string \"{value}\"",
            f"the text \"{value}\"",
            f"the word \"{value}\"",
            f"literal \"{value}\"",
            f"the exact text \"{value}\"",
        ]
    )
    return Part(phrase, f"REF{ref}", f"<{ref}>")


def make_punctuation_part(rng: random.Random) -> Part:
    name, (literal, template) = rng.choice(tuple(VISIBLE_LITERALS.items()))
    phrase = rng.choice([f"a {name}", f"the literal {name}", f'literal "{literal}"'])
    return Part(phrase, f'LIT("{literal}")', template)


def make_alt_part(rng: random.Random, quotes: QuoteBuilder, *, max_choices: int = 3) -> Part:
    count = rng.randint(2, max_choices)
    choices = [quotes.add() for _ in range(count)]
    refs = [ref for ref, _value in choices]
    values = [value for _ref, value in choices]
    if count == 2:
        words = rng.choice(
            [
                f'either "{values[0]}" or "{values[1]}"',
                f'"{values[0]}" or "{values[1]}"',
                f'the string "{values[0]}" or the string "{values[1]}"',
                f'one of "{values[0]}" and "{values[1]}"',
            ]
        )
    else:
        words = rng.choice(
            [
                f'either "{values[0]}", "{values[1]}", or "{values[2]}"',
                f'one of "{values[0]}", "{values[1]}", and "{values[2]}"',
            ]
        )
    il = "ALT(" + ",".join(f"REF{ref}" for ref in refs) + ")"
    template = "(?:" + "|".join(f"<{ref}>" for ref in refs) + ")"
    return Part(words, il, template)


def make_part(rng: random.Random, quotes: QuoteBuilder, *, allow_alt: bool = True) -> Part:
    roll = rng.random()
    if allow_alt and roll < 0.16:
        return make_alt_part(rng, quotes)
    if roll < 0.46:
        return make_ref_part(rng, quotes)
    if roll < 0.66:
        return make_punctuation_part(rng)
    return make_class_part(rng)


def join_words(parts: list[Part], rng: random.Random) -> str:
    if len(parts) == 1:
        return parts[0].words
    text = parts[0].words
    for part in parts[1:]:
        text += rng.choice([" followed by ", " then ", " and then "]) + part.words
    return text


def seq_il(parts: list[Part]) -> str:
    return "SEQ(" + ",".join(part.il for part in parts) + ")"


def seq_template(parts: list[Part]) -> str:
    return "".join(part.template for part in parts)


def make_sequence_example(rng: random.Random, quotes: QuoteBuilder, component_counts: list[int]) -> dict[str, object]:
    count = rng.choice(component_counts)
    parts = [make_part(rng, quotes) for _ in range(count)]
    anchor = rng.choices(["FIND", "START", "END", "FULL"], weights=[35, 35, 15, 15], k=1)[0]
    phrase = join_words(parts, rng)
    if anchor == "FIND":
        intro = rng.choice(["lines with ", "lines containing ", "lines that have ", "items with "])
        template = seq_template(parts)
    elif anchor == "START":
        intro = rng.choice(["lines starting with ", "lines that begin with ", "items beginning with "])
        template = "^" + seq_template(parts)
    elif anchor == "END":
        intro = rng.choice(["lines ending with ", "lines that end with ", "items ending in "])
        template = seq_template(parts) + "$"
    else:
        intro = rng.choice(["lines made of ", "the whole line is ", "the entire line is "])
        template = "^" + seq_template(parts) + "$"
    return {
        "band": "sequence",
        "refs": quotes.refs,
        "input": intro + phrase,
        "il": f"{anchor}; {seq_il(parts)};",
        "template": template,
    }


def make_order_example(rng: random.Random, quotes: QuoteBuilder) -> dict[str, object]:
    left = make_part(rng, quotes, allow_alt=False)
    right = make_part(rng, quotes, allow_alt=False)
    order = rng.choice(["BEFORE", "AFTER"])
    if order == "BEFORE":
        first, second = left, right
    else:
        first, second = right, left
    if order == "BEFORE":
        text = rng.choice(
            [
                f"lines with {left.words} before {right.words}",
                f"lines where {left.words} comes before {right.words}",
            ]
        )
    elif order == "AFTER":
        text = rng.choice(
            [
                f"lines with {left.words} after {right.words}",
                f"lines where {left.words} comes after {right.words}",
            ]
        )
    return {
        "band": "order",
        "refs": quotes.refs,
        "input": text,
        "il": f"{order}; LEFT({left.il}); RIGHT({right.il});",
        "template": first.template + ".*" + second.template,
    }


def make_only_example(rng: random.Random, quotes: QuoteBuilder) -> dict[str, object]:
    # Keep this band intentionally simple: one repeatable class or one quoted string.
    if rng.random() < 0.78:
        kind = rng.choice(tuple(CLASS_SPECS))
        words = rng.choice(CLASS_SPECS[kind]["plural"])
        part = Part(words, f"{kind}+", f"{CLASS_SPECS[kind]['regex']}+")
    else:
        part = make_ref_part(rng, quotes)
        part = Part(part.words, part.il + "+", f"(?:{part.template})+")
    text = rng.choice(
        [
            f"lines containing only {part.words}",
            f"lines made up of only {part.words}",
            f"items that consist only of {part.words}",
        ]
    )
    return {
        "band": "only",
        "refs": quotes.refs,
        "input": text,
        "il": f"ONLY; {part.il};",
        "template": f"^{part.template}$",
    }


def make_negative_example(rng: random.Random, quotes: QuoteBuilder) -> dict[str, object]:
    if rng.random() < 0.65:
        part = make_ref_part(rng, quotes)
    else:
        kind = rng.choice(tuple(CLASS_SPECS))
        words = "any " + rng.choice(CLASS_SPECS[kind]["singular"])
        part = Part(words, kind, CLASS_SPECS[kind]["regex"])
    text = rng.choice(
        [
            f"lines that do not contain {part.words}",
            f"lines without {part.words}",
            f"items not containing {part.words}",
        ]
    )
    return {
        "band": "negative",
        "refs": quotes.refs,
        "input": text,
        "il": f"NOT_CONTAIN; {part.il};",
        "template": f"^(?!.*{part.template}).*$",
    }


def make_repeat_example(rng: random.Random, quotes: QuoteBuilder) -> dict[str, object]:
    part = make_ref_part(rng, quotes) if rng.random() < 0.55 else make_class_part(rng, quant="")
    count = rng.choice([2, 3, 4, 5])
    count_text = count_word(count, rng)
    mode = rng.choice(["AT_LEAST", "EXACTLY"])
    if mode == "AT_LEAST":
        text = rng.choice(
            [
                f"lines with {part.words} at least {count_text} times",
                f"lines containing {part.words} {count_text} or more times",
            ]
        )
        quant = f"{{{count},}}"
    else:
        text = rng.choice(
            [
                f"lines with {part.words} exactly {count_text} times",
                f"items containing exactly {count_text} copies of {part.words}",
            ]
        )
        quant = f"{{{count}}}"
    return {
        "band": "repeat",
        "refs": quotes.refs,
        "input": text,
        "il": f"{mode}; {part.il}; COUNT({count});",
        "template": f"(?:{part.template}){quant}",
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
        "sequence": 45,
        "order": 25,
        "only": 12,
        "negative": 8,
        "repeat": 10,
    }
    band = rng.choices(tuple(bands), weights=tuple(bands.values()), k=1)[0]
    if band == "sequence":
        example = make_sequence_example(rng, quotes, component_counts)
    elif band == "order":
        example = make_order_example(rng, quotes)
    elif band == "only":
        example = make_only_example(rng, quotes)
    elif band == "negative":
        example = make_negative_example(rng, quotes)
    elif band == "repeat":
        example = make_repeat_example(rng, quotes)
    else:
        raise ValueError(f"unknown band: {band}")
    return {**example, "refs": quotes.refs}


def format_example(example: dict[str, object]) -> str:
    return f"Input:\n{example['input']}\n\nIL:\n{example['il']}\n\nTemplate:\n{example['template']}"


def main() -> None:
    args = parse_args()
    if args.examples <= 0:
        raise ValueError("--examples must be positive")
    component_counts = parse_component_counts(args.component_counts)
    band_weights = {"sequence": 45, "order": 25, "only": 12, "negative": 8, "repeat": 10}
    config = RegexIlV3Config(
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
    metadata = {
        "config": asdict(config),
        "output": str(args.output.relative_to(ROOT) if args.output.is_absolute() else args.output),
        "total_examples": len(examples),
        "total_chars": len(corpus),
        "unique_chars": len(set(corpus)),
        "chars": sorted(set(corpus)),
        "band_counts": {band: sum(1 for example in examples if example["band"] == band) for band in band_weights},
        "first_examples": examples[:20],
        "note": "Inputs contain raw quoted strings. IL and Template use REFn/<n> placeholders assigned by quote order.",
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
                    "examples": len(examples),
                    "chars": len(corpus),
                    "band_counts": metadata["band_counts"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
