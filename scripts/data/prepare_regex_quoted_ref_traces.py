from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.data.prepare_regex_traces import (
    CLASS_SPECS,
    LITERAL_CHARS,
    ROOT,
    apply_quant,
    join_phrases,
    quant_words,
    random_literal,
)


DEFAULT_OUTPUT = ROOT / "data" / "regex_quoted_ref_traces.txt"
DEFAULT_METADATA_OUTPUT = ROOT / "data" / "regex_quoted_ref_traces.sources.json"


@dataclass(frozen=True)
class RegexQuotedRefTraceConfig:
    examples: int
    seed: int
    component_counts: str
    language_style: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw-quoted NL to referenced regex-template traces.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--examples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=8383)
    parser.add_argument("--component-counts", default="2,2,3")
    parser.add_argument("--language-style", choices=("simple", "v2"), default="simple")
    return parser.parse_args()


def parse_component_counts(value: str) -> list[int]:
    counts = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        count = int(part)
        if count < 1:
            raise ValueError("component counts must be positive")
        counts.append(count)
    if not counts:
        raise ValueError("--component-counts must contain at least one count")
    return counts


class QuoteBuilder:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.refs: list[str] = []

    def add_quote(self) -> tuple[int, str]:
        value = random_literal(self.rng)
        while value in self.refs:
            value = random_literal(self.rng)
        self.refs.append(value)
        return len(self.refs) - 1, value


def make_class_component(rng: random.Random) -> tuple[str, str, str]:
    name = rng.choice(tuple(CLASS_SPECS))
    plan_base, regex_base = CLASS_SPECS[name]
    quant = rng.choice(["", "+", "*", "?", "{2}", "{3}", "{4}", "{2,4}", "{3,5}"])
    words = quant_words(name, quant, rng)
    plan, template = apply_quant(regex_base, plan_base, quant)
    return words, plan, template


def class_paraphrase(name: str, quant: str, rng: random.Random, *, language_style: str) -> str:
    if language_style == "simple":
        return quant_words(name, quant, rng)

    plural = f"{name}s"
    if quant == "":
        return rng.choice([f"a {name}", f"one {name}", f"a single {name}"])
    if quant == "+":
        return rng.choice([f"one or more {plural}", f"at least one {name}", f"one plus {plural}", f"one or many {plural}"])
    if quant == "*":
        return rng.choice([f"zero or more {plural}", f"any number of {plural}", f"possibly many {plural}", f"as many {plural} as needed"])
    if quant == "?":
        return rng.choice([f"an optional {name}", f"zero or one {name}", f"maybe a {name}", f"with or without a {name}"])
    if quant.startswith("{") and "," not in quant:
        count = quant.strip("{}")
        count_text = rng.choice([count, {"2": "two", "3": "three", "4": "four"}.get(count, count)])
        return rng.choice([f"exactly {count_text} {plural}", f"{count_text} {plural}", f"precisely {count_text} {plural}"])
    if quant.startswith("{"):
        lo, hi = quant.strip("{}").split(",")
        words = {"2": "two", "3": "three", "4": "four", "5": "five"}
        lo_text = rng.choice([lo, words.get(lo, lo)])
        hi_text = rng.choice([hi, words.get(hi, hi)])
        return rng.choice(
            [
                f"between {lo_text} and {hi_text} {plural}",
                f"{lo_text} to {hi_text} {plural}",
                f"from {lo_text} through {hi_text} {plural}",
            ]
        )
    raise ValueError(f"unknown quantifier: {quant}")


def make_class_component_v2(rng: random.Random, *, language_style: str) -> tuple[str, str, str]:
    name = rng.choice(tuple(CLASS_SPECS))
    plan_base, regex_base = CLASS_SPECS[name]
    quant = rng.choice(["", "+", "*", "?", "{2}", "{3}", "{4}", "{2,4}", "{3,5}"])
    words = class_paraphrase(name, quant, rng, language_style=language_style)
    plan, template = apply_quant(regex_base, plan_base, quant)
    return words, plan, template


def make_literal_component(rng: random.Random, quotes: QuoteBuilder, *, language_style: str) -> tuple[str, str, str]:
    if rng.random() < 0.55:
        ref, value = quotes.add_quote()
        if language_style == "v2":
            phrase = rng.choice(
                [
                    f'"{value}"',
                    f'the word "{value}"',
                    f'literal "{value}"',
                    f'the exact text "{value}"',
                    f'the quoted text "{value}"',
                    f'the string "{value}"',
                ]
            )
        else:
            phrase = rng.choice([f'"{value}"', f'the word "{value}"', f'literal "{value}"'])
        return phrase, f"REF{ref}", f"<{ref}>"
    name, (literal, regex) = rng.choice(tuple(LITERAL_CHARS.items()))
    phrase = rng.choice([f"a {name}", f"the literal {name}", f'literal "{literal}"'])
    display = literal if literal != " " else "space"
    return phrase, f'LITERAL("{display}")', regex


def make_alt_component(rng: random.Random, quotes: QuoteBuilder, *, language_style: str) -> tuple[str, str, str]:
    choice_count = rng.choice([2, 3])
    choices = [quotes.add_quote() for _ in range(choice_count)]
    refs = [ref for ref, _value in choices]
    values = [value for _ref, value in choices]
    if len(choices) == 2:
        options = [
            f'either "{values[0]}" or "{values[1]}"',
            f'"{values[0]}" or "{values[1]}"',
            f'the word "{values[0]}" or the word "{values[1]}"',
        ]
        if language_style == "v2":
            options.extend(
                [
                    f'one of "{values[0]}" and "{values[1]}"',
                    f'one of "{values[0]}" or "{values[1]}"',
                    f'the exact text "{values[0]}" or "{values[1]}"',
                ]
            )
        phrase = rng.choice(options)
    else:
        options = [f'"{values[0]}", "{values[1]}", or "{values[2]}"']
        if language_style == "v2":
            options.extend(
                [
                    f'one of "{values[0]}", "{values[1]}", and "{values[2]}"',
                    f'either "{values[0]}", "{values[1]}", or "{values[2]}"',
                ]
            )
        phrase = rng.choice(options)
    plan = "ALT(" + ", ".join(f"REF{ref}" for ref in refs) + ")"
    template = "(?:" + "|".join(f"<{ref}>" for ref in refs) + ")"
    return phrase, plan, template


def make_component(rng: random.Random, quotes: QuoteBuilder, *, language_style: str) -> tuple[str, str, str]:
    roll = rng.random()
    if roll < 0.55:
        return make_class_component_v2(rng, language_style=language_style)
    if roll < 0.85:
        return make_literal_component(rng, quotes, language_style=language_style)
    return make_alt_component(rng, quotes, language_style=language_style)


def make_example(
    rng: random.Random,
    *,
    component_choices: list[int] | None = None,
    language_style: str = "simple",
) -> dict[str, object]:
    quote_builder = QuoteBuilder(rng)
    start_anchor = rng.random() < 0.65
    end_anchor = rng.random() < 0.30
    count = rng.choice(component_choices or [2, 2, 3])
    components = [make_component(rng, quote_builder, language_style=language_style) for _ in range(count)]

    phrase = join_phrases([component[0] for component in components], rng)
    if start_anchor and end_anchor:
        intros = ["matches exactly ", "is exactly ", "the whole string is "]
        if language_style == "v2":
            intros += ["the entire value is ", "should be exactly "]
        intro = rng.choice(intros)
    elif start_anchor:
        intros = ["starts with ", "begins with ", "has prefix "]
        if language_style == "v2":
            intros += ["should start with ", "must begin with ", "is prefixed by "]
        intro = rng.choice(intros)
    elif end_anchor:
        intros = ["ends with ", "has suffix "]
        if language_style == "v2":
            intros += ["should end with ", "must finish with ", "is suffixed by "]
        intro = rng.choice(intros)
    else:
        intros = ["contains ", "has ", "matches "]
        if language_style == "v2":
            intros += ["includes ", "has somewhere ", "should include "]
        intro = rng.choice(intros)

    plan_parts = []
    template = ""
    if start_anchor:
        plan_parts.append("START")
        template += "^"
    for _words, plan, template_part in components:
        plan_parts.append(plan)
        template += template_part
    if end_anchor:
        plan_parts.append("END")
        template += "$"

    return {
        "refs": quote_builder.refs,
        "input": intro + phrase,
        "plan": "; ".join(plan_parts) + ";",
        "template": template,
    }


def format_example(example: dict[str, object]) -> str:
    return f"Input:\n{example['input']}\n\nPlan:\n{example['plan']}\n\nTemplate:\n{example['template']}"


def main() -> None:
    args = parse_args()
    if args.examples <= 0:
        raise ValueError("--examples must be positive")
    component_counts = parse_component_counts(args.component_counts)
    config = RegexQuotedRefTraceConfig(
        examples=args.examples,
        seed=args.seed,
        component_counts=args.component_counts,
        language_style=args.language_style,
    )
    rng = random.Random(config.seed)
    examples = [
        make_example(rng, component_choices=component_counts, language_style=args.language_style)
        for _ in range(config.examples)
    ]
    corpus = "\n\n".join(format_example(example) for example in examples) + "\n"
    metadata = {
        "config": asdict(config),
        "output": str(args.output.relative_to(ROOT) if args.output.is_absolute() else args.output),
        "total_examples": len(examples),
        "total_chars": len(corpus),
        "unique_chars": len(set(corpus)),
        "chars": sorted(set(corpus)),
        "first_examples": examples[:10],
        "note": "Inputs contain raw quoted strings. Targets refer to quote order with REF0 and <0> placeholders.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus, encoding="utf-8")
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "examples": len(examples), "chars": len(corpus)}, indent=2))


if __name__ == "__main__":
    main()
