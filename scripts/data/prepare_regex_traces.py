from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "regex_traces.txt"
DEFAULT_METADATA_OUTPUT = ROOT / "data" / "regex_traces.sources.json"


@dataclass(frozen=True)
class RegexTraceConfig:
    examples: int
    seed: int


ALPHANUM = "abcdefghijklmnopqrstuvwxyz0123456789"
NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
}
LITERAL_CHARS = {
    "dash": ("-", "-"),
    "dot": (".", r"\."),
    "underscore": ("_", "_"),
    "slash": ("/", r"\/"),
    "space": (" ", r"\s"),
    "colon": (":", ":"),
    "at sign": ("@", "@"),
}
CLASS_SPECS = {
    "digit": ("DIGIT", r"\d"),
    "letter": ("LETTER", "[A-Za-z]"),
    "lowercase letter": ("LOWER", "[a-z]"),
    "uppercase letter": ("UPPER", "[A-Z]"),
    "word character": ("WORD", r"\w"),
    "whitespace": ("SPACE", r"\s"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate natural-language to regex traces.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--examples", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=4242)
    return parser.parse_args()


def count_words(count: int, rng: random.Random) -> str:
    return rng.choice([str(count), NUMBER_WORDS[count]])


def quant_words(base: str, quant: str, rng: random.Random) -> str:
    if quant == "":
        return f"a {base}" if base[0] not in "aeiou" else f"an {base}"
    if quant == "+":
        return rng.choice([f"one or more {base}s", f"at least one {base}"])
    if quant == "*":
        return rng.choice([f"zero or more {base}s", f"any number of {base}s"])
    if quant == "?":
        return rng.choice([f"an optional {base}", f"zero or one {base}"])
    if quant.startswith("{") and "," not in quant:
        count = count_words(int(quant.strip("{}")), rng)
        return rng.choice([f"exactly {count} {base}s", f"{count} {base}s"])
    if quant.startswith("{"):
        lo, hi = quant.strip("{}").split(",")
        lo_word = count_words(int(lo), rng)
        hi_word = count_words(int(hi), rng)
        return rng.choice([f"between {lo_word} and {hi_word} {base}s", f"{lo_word} to {hi_word} {base}s"])
    raise ValueError(f"unknown quantifier: {quant}")


def apply_quant(regex: str, plan: str, quant: str) -> tuple[str, str]:
    if quant == "":
        return plan, regex
    return f"{plan}{quant}", f"{regex}{quant}"


def make_class_component(rng: random.Random) -> tuple[str, str, str]:
    name = rng.choice(tuple(CLASS_SPECS))
    plan_base, regex_base = CLASS_SPECS[name]
    quant = rng.choice(["", "+", "*", "?", "{2}", "{3}", "{4}", "{2,4}", "{3,5}"])
    words = quant_words(name, quant, rng)
    plan, regex = apply_quant(regex_base, plan_base, quant)
    return words, plan, regex


def random_literal(rng: random.Random) -> str:
    first = rng.choice("abcdefghijklmnopqrstuvwxyz")
    length = rng.randint(2, 7)
    return first + "".join(rng.choice(ALPHANUM) for _ in range(length - 1))


def make_literal_component(rng: random.Random) -> tuple[str, str, str]:
    if rng.random() < 0.55:
        word = random_literal(rng)
        phrase = rng.choice([f'the word "{word}"', f'literal "{word}"', f'{word!r}'])
        return phrase, f'LITERAL("{word}")', word
    name, (literal, regex) = rng.choice(tuple(LITERAL_CHARS.items()))
    phrase = rng.choice([f"a {name}", f'the literal {name}', f'literal "{literal}"'])
    display = literal if literal != " " else "space"
    return phrase, f'LITERAL("{display}")', regex


def make_alt_component(rng: random.Random) -> tuple[str, str, str]:
    choices = []
    choice_count = rng.choice([2, 3])
    while len(choices) < choice_count:
        value = random_literal(rng)
        if value not in choices:
            choices.append(value)
    if len(choices) == 2:
        phrase = rng.choice(
            [
                f'either "{choices[0]}" or "{choices[1]}"',
                f'"{choices[0]}" or "{choices[1]}"',
                f'the word "{choices[0]}" or the word "{choices[1]}"',
                f"{choices[0]} or {choices[1]}",
            ]
        )
    else:
        phrase = rng.choice(
            [
                f'"{choices[0]}", "{choices[1]}", or "{choices[2]}"',
                f"{choices[0]}, {choices[1]}, or {choices[2]}",
            ]
        )
    plan = "ALT(" + ", ".join(f'LITERAL("{choice}")' for choice in choices) + ")"
    regex = "(?:" + "|".join(choices) + ")"
    return phrase, plan, regex


def make_component(rng: random.Random) -> tuple[str, str, str]:
    roll = rng.random()
    if roll < 0.55:
        return make_class_component(rng)
    if roll < 0.85:
        return make_literal_component(rng)
    return make_alt_component(rng)


def join_phrases(phrases: list[str], rng: random.Random) -> str:
    if len(phrases) == 1:
        return phrases[0]
    connectors = [" then ", " followed by ", " and then "]
    text = phrases[0]
    for phrase in phrases[1:]:
        text += rng.choice(connectors) + phrase
    return text


def make_example(rng: random.Random) -> dict[str, str]:
    start_anchor = rng.random() < 0.65
    end_anchor = rng.random() < 0.30
    count = rng.choice([2, 3, 3, 4])
    components = [make_component(rng) for _ in range(count)]

    phrase = join_phrases([component[0] for component in components], rng)
    if start_anchor and end_anchor:
        intro = rng.choice(["matches exactly ", "is exactly ", "the whole string is "])
    elif start_anchor:
        intro = rng.choice(["starts with ", "begins with ", "has prefix "])
    elif end_anchor:
        intro = rng.choice(["ends with ", "has suffix "])
    else:
        intro = rng.choice(["contains ", "has ", "matches "])

    plan_parts = []
    regex = ""
    if start_anchor:
        plan_parts.append("START")
        regex += "^"
    for _words, plan, regex_part in components:
        plan_parts.append(plan)
        regex += regex_part
    if end_anchor:
        plan_parts.append("END")
        regex += "$"

    return {
        "input": intro + phrase,
        "plan": "; ".join(plan_parts) + ";",
        "regex": regex,
    }


def main() -> None:
    args = parse_args()
    if args.examples <= 0:
        raise ValueError("--examples must be positive")
    config = RegexTraceConfig(examples=args.examples, seed=args.seed)
    rng = random.Random(config.seed)
    examples = [make_example(rng) for _ in range(config.examples)]
    corpus = "\n\n".join(
        f"Input:\n{example['input']}\n\nPlan:\n{example['plan']}\n\nRegex:\n{example['regex']}"
        for example in examples
    ) + "\n"
    metadata = {
        "config": asdict(config),
        "output": str(args.output.relative_to(ROOT) if args.output.is_absolute() else args.output),
        "total_examples": len(examples),
        "total_chars": len(corpus),
        "unique_chars": len(set(corpus)),
        "chars": sorted(set(corpus)),
        "first_examples": examples[:10],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8")
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "examples": len(examples), "chars": len(corpus)}, indent=2))


if __name__ == "__main__":
    main()
