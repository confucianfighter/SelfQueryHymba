from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.prepare_regex_il_v3_traces import QuoteBuilder, parse_component_counts, random_string  # noqa: E402
from scripts.data.prepare_regex_traces import ROOT  # noqa: E402


DEFAULT_OUTPUT = ROOT / "data" / "regex_il_v7_dream_queries_120k.txt"
DEFAULT_METADATA_OUTPUT = ROOT / "data" / "regex_il_v7_dream_queries_120k.sources.json"
CAPTURE_NAMES = ["a", "b", "c", "A", "B", "C"]
CODE_IDENTIFIERS = [
    "d_model",
    "num_heads",
    "mlp_multiplier",
    "hidden",
    "states",
    "CausalSSMBranch",
    "FastCausalConvBranch",
    "HybridHymbaBlock",
    "MultiScaleCausalDecomposition",
]


@dataclass(frozen=True)
class RegexIlV7Config:
    examples: int
    seed: int
    string_min_len: int
    string_max_len: int
    component_counts: str
    preview: int
    band_weights: dict[str, int]
    polite_rate: float


@dataclass(frozen=True)
class Atom:
    words: str
    il: str
    template: str


def random_quant(rng: random.Random, *, allow_empty: bool = True, allow_zero: bool = True) -> str:
    roll = rng.random()
    if roll < 0.34:
        choices = ["+"]
        if allow_zero:
            choices.extend(["*", "?"])
        if allow_empty:
            choices.append("")
        return rng.choice(choices)
    if roll < 0.62:
        return "{" + str(rng.randint(1, 12)) + "}"
    if roll < 0.88:
        left = rng.randint(1, 9)
        right = rng.randint(left + 1, min(20, left + rng.randint(2, 9)))
        return "{" + str(left) + "," + str(right) + "}"
    if roll < 0.95:
        return "{" + str(rng.randint(1, 9)) + ",}"
    if allow_zero:
        return "{0," + str(rng.randint(1, 12)) + "}"
    return "{" + str(rng.randint(1, 12)) + "}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clean /regex dream-language IL traces.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--examples", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=72026)
    parser.add_argument("--string-min-len", type=int, default=2)
    parser.add_argument("--string-max-len", type=int, default=9)
    parser.add_argument("--component-counts", default="2,3,3,4")
    parser.add_argument("--polite-rate", type=float, default=0.08)
    parser.add_argument("--preview", type=int, default=0)
    return parser.parse_args()


def quote(value: str) -> str:
    return f'"{value}"'


def cap_name(name: str) -> str:
    return f"'{name}'"


def word_literal(rng: random.Random, *, min_len: int = 2, max_len: int = 6) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(min_len, max_len)))


def bare_literal_atom(rng: random.Random, *, min_len: int = 2, max_len: int = 8) -> Atom:
    value = word_literal(rng, min_len=min_len, max_len=max_len)
    return Atom(words=value, il=f'LIT("{value}")', template=re.escape(value))


def literal_atom(rng: random.Random, quotes: QuoteBuilder) -> Atom:
    ref, value = quotes.add()
    words = rng.choice([quote(value), f"the text {quote(value)}"])
    return Atom(words=words, il=f"REF{ref}", template=f"<{ref}>")


def punctuation_atom(rng: random.Random) -> Atom:
    value, words, regex = rng.choice(
        [
            ("-", "a dash", "-"),
            ("_", "an underscore", "_"),
            (".", "a dot", r"\."),
            (",", "a comma", ","),
            (":", "a colon", ":"),
            ("@", "an at sign", "@"),
            ("+", "a plus sign", r"\+"),
            ("/", "a slash", r"\/"),
        ]
    )
    return Atom(words=words, il=f'LIT("{value}")', template=regex)


def alt_atom(rng: random.Random, quotes: QuoteBuilder, *, word_safe: bool = False) -> Atom:
    if word_safe:
        left = class_atom(rng, quant=rng.choice(["", "{2}", "{3}", "+"]), word_safe=True)
        right = class_atom(rng, quant=rng.choice(["", "{2}", "{3}", "+"]), word_safe=True)
    else:
        left = literal_atom(rng, quotes) if rng.random() < 0.7 else punctuation_atom(rng)
        right = literal_atom(rng, quotes) if rng.random() < 0.7 else punctuation_atom(rng)
    words = rng.choice(
        [
            f"either {left.words} or {right.words}",
            f"one of {left.words} or {right.words}",
            f"one of {left.words} and {right.words}",
            f"{left.words} or {right.words}",
        ]
    )
    return Atom(words=words, il=f"ALT({left.il},{right.il})", template=f"(?:{left.template}|{right.template})")


def class_atom(rng: random.Random, *, quant: str | None = None, word_safe: bool = False) -> Atom:
    classes = [
        ("DIGIT", r"\d", "digit", "digits"),
        ("ANY_LETTER", "[A-Za-z]", "letter", "letters"),
        ("LOWER", "[a-z]", "lowercase letter", "lowercase letters"),
        ("UPPER", "[A-Z]", "uppercase letter", "uppercase letters"),
        ("WORD", r"\w", "word character", "word characters"),
        ("SPACE", r"\s", "space", "spaces"),
        ("VOWEL", "[AEIOUaeiou]", "vowel", "vowels"),
    ]
    if word_safe:
        classes = [item for item in classes if item[0] in {"ANY_LETTER", "LOWER", "UPPER", "VOWEL"}]
    kind, regex, singular, plural = rng.choice(classes)
    if quant is None:
        quant = random_quant(rng)
    if quant == "":
        words = rng.choice([f"one {singular}", f"a single {singular}", singular])
    elif quant == "+":
        words = rng.choice([plural, f"one or more {plural}"])
    elif quant == "*":
        words = rng.choice([f"zero or more {plural}", f"any number of {plural}"])
    elif quant == "?":
        words = rng.choice([f"maybe {singular}", f"an optional {singular}"])
    elif quant.startswith("{") and "," not in quant:
        count = int(quant.strip("{}"))
        noun = singular if count == 1 else plural
        words = rng.choice([f"exactly {count} {noun}", f"{count} {noun}"])
    elif quant.startswith("{") and quant.endswith(",}"):
        left = int(quant.strip("{,}"))
        if left == 1:
            words = rng.choice([f"at least one {singular}", f"one or more {plural}"])
        else:
            words = rng.choice([f"at least {left} {plural}", f"{left} or more {plural}"])
    elif quant.startswith("{0,"):
        right = int(quant.strip("{}").split(",")[1])
        noun = singular if right == 1 else plural
        words = rng.choice([f"at most {right} {noun}", f"up to {right} {noun}", f"no more than {right} {noun}"])
    else:
        left, right = quant.strip("{}").split(",")
        words = rng.choice([f"{left} to {right} {plural}", f"between {left} and {right} {plural}", f"from {left} through {right} {plural}"])
    return Atom(words=words, il=kind + quant, template=regex + quant)


def plain_atom(rng: random.Random, quotes: QuoteBuilder, *, word_safe: bool = False) -> Atom:
    roll = rng.random()
    if roll < 0.14:
        return alt_atom(rng, quotes, word_safe=word_safe)
    if roll < 0.5:
        return class_atom(rng, word_safe=word_safe)
    if roll < 0.8 and not word_safe:
        return literal_atom(rng, quotes)
    if roll < 0.92 and not word_safe:
        return punctuation_atom(rng)
    return class_atom(rng, quant=random_quant(rng, allow_empty=False), word_safe=word_safe)


def capture_atom(atom: Atom, index: int) -> Atom:
    return Atom(words=atom.words, il=f"CAP{index}({atom.il})", template=f"({atom.template})")


def capture_phrase(atom: Atom, name: str, rng: random.Random) -> str:
    obj = atom.words
    return rng.choice(
        [
            f"capture {obj} as {cap_name(name)}",
            f"save {obj} as {cap_name(name)}",
            f"remember {obj} as {cap_name(name)}",
            f"call {obj} {cap_name(name)}",
            f"start capture {cap_name(name)}: {obj}; end capture {cap_name(name)}",
            f"begin capture {cap_name(name)}: {obj}; end capture {cap_name(name)}",
            f"capture starts as {cap_name(name)}: {obj}; capture ends",
        ]
    )


def capture_action(names: list[str], rng: random.Random) -> str:
    quoted = [cap_name(name) for name in names]
    joined = ", ".join(quoted)
    if len(names) == 1:
        return rng.choice([f"give me {joined}", f"show {joined}", f"keep {joined}", f"output {joined}"])
    return rng.choice([f"give me {joined}", f"show {joined}", f"keep {joined}", f"output {joined}"])


def selection_token(rng: random.Random) -> str:
    return rng.choice(["selection", "selected text", "selected part"])


def select_phrase(atom: Atom, rng: random.Random) -> str:
    obj = atom.words
    if obj.startswith("the text "):
        obj = obj.removeprefix("the text ")
    elif obj.startswith("literal "):
        obj = obj.removeprefix("literal ")
    variants = [
        f"start select: {obj}; end select",
        f"begin select: {obj}; end select",
        f"start selection: {obj}; end selection",
        f"select starts: {obj}; select ends",
        f"select {obj}",
        f"select the part that is {obj}",
    ]
    return rng.choice(variants)


def connector_join(parts: list[str], rng: random.Random) -> str:
    if len(parts) == 1:
        return parts[0]
    style = rng.choice(["then", "followed", "comma_then"])
    if style == "then":
        return ", then ".join(parts)
    if style == "followed":
        return " followed by ".join(parts)
    text = parts[0]
    for part in parts[1:-1]:
        text += ", " + part
    return text + ", and then " + parts[-1]


def scope_prefix(scope: str, rng: random.Random) -> str:
    options = {
        "text": ["text: ", "in text: ", ""],
        "line": ["line: ", "lines: ", "in each line: "],
        "sentence": ["sentence: ", "sentences: ", "in each sentence: "],
        "whole": ["whole string: ", "the whole string: "],
        "words": ["words: "],
    }
    return rng.choice(options[scope])


def scoped_query(scope: str, phrase: str, rng: random.Random) -> str:
    command_starts = (
        "capture ",
        "save ",
        "remember ",
        "call ",
        "start capture ",
        "begin capture ",
        "capture starts ",
        "start select",
        "begin select",
        "start selection",
        "select starts",
        "select ",
    )
    command_like = phrase.startswith(command_starts) or any(part in phrase for part in command_starts)
    if command_like:
        return scope_prefix(scope, rng) + phrase
    if scope == "text":
        return rng.choice(
            [
                scope_prefix(scope, rng) + phrase,
                f"text that has {phrase}",
                f"text containing {phrase}",
                f"text with {phrase}",
            ]
        )
    if scope == "line":
        return rng.choice(
            [
                scope_prefix(scope, rng) + phrase,
                f"lines that have {phrase}",
                f"lines containing {phrase}",
                f"lines with {phrase}",
                f"each line with {phrase}",
            ]
        )
    if scope == "sentence":
        return rng.choice(
            [
                scope_prefix(scope, rng) + phrase,
                f"sentences that have {phrase}",
                f"sentences containing {phrase}",
                f"sentences with {phrase}",
                f"each sentence with {phrase}",
            ]
        )
    if scope == "whole":
        return rng.choice(
            [
                scope_prefix(scope, rng) + phrase,
                f"whole string matching {phrase}",
                f"the whole string should be {phrase}",
            ]
        )
    raise ValueError(f"unknown scope: {scope}")


def regex_prefix(rng: random.Random) -> str:
    return rng.choice(["/regex ", "/regex ", "/regex ", "/r "])


def apply_polite_noise(query: str, rng: random.Random, rate: float) -> str:
    if rng.random() >= rate:
        return query
    return rng.choice([f"please {query}", f"{query} please", f"as a regex, {query}", f"{query} if possible"])


def seq_il(atoms: list[Atom]) -> str:
    return "SEQ(" + ",".join(atom.il for atom in atoms) + ")"


def seq_template(atoms: list[Atom]) -> str:
    return "".join(atom.template for atom in atoms)


def common_word_atom(value: str) -> Atom:
    return Atom(words=value, il=f'LIT("{value}")', template=re.escape(value))


def ref_value_atom(quotes: QuoteBuilder, value: str, *, words: str | None = None) -> Atom:
    quotes.refs.append(value)
    ref = len(quotes.refs) - 1
    return Atom(words=words if words is not None else quote(value), il=f"REF{ref}", template=f"<{ref}>")


def selected_sequence(rng: random.Random, quotes: QuoteBuilder) -> tuple[list[Atom], str]:
    head = common_word_atom(rng.choice(["page", "line", "word", "class", "self", "total", "section"]))
    connector = Atom(words="space", il="SPACE+", template=r"\s+")
    if rng.random() < 0.7:
        ref, value = quotes.add()
        tail = Atom(words=rng.choice([quote(value), f"the word {quote(value)}", f"the text {quote(value)}"]), il=f"REF{ref}", template=f"<{ref}>")
        tail_plain = quote(value)
    else:
        value = rng.choice(["count", "total", "title", "name", "value", "id", "page"])
        tail = common_word_atom(value)
        tail_plain = value
    words = rng.choice(
        [
            f"{head.words} followed by {tail.words}",
            f"{head.words} then {tail.words}",
            f"{head.words} followed by the word {tail_plain}",
        ]
    )
    return [head, connector, tail], words


def make_match_query(rng: random.Random, quotes: QuoteBuilder, counts: list[int], polite_rate: float) -> dict[str, object]:
    scope = rng.choices(["text", "line", "sentence", "whole"], weights=[43, 22, 12, 23], k=1)[0]
    count = rng.choice(counts)
    atoms = [plain_atom(rng, quotes) for _ in range(count)]
    phrase = connector_join([atom.words for atom in atoms], rng)
    anchor = {"text": "FIND", "line": "FIND", "sentence": "FIND", "whole": "FULL"}[scope]
    template = seq_template(atoms)
    if anchor == "FULL":
        template = "^" + template + "$"
    query = regex_prefix(rng) + scoped_query(scope, phrase, rng)
    return {
        "band": "match",
        "refs": quotes.refs,
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": f"{anchor}; {seq_il(atoms)};",
        "template": template,
    }


def make_literal_query(rng: random.Random, polite_rate: float) -> dict[str, object]:
    atom = bare_literal_atom(rng)
    query = regex_prefix(rng) + rng.choice(
        [
            f"match {atom.words}",
            f"find {atom.words}",
            f"text: {atom.words}",
            f"match the text {atom.words}",
        ]
    )
    return {
        "band": "literal",
        "refs": [],
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": f"FIND; SEQ({atom.il});",
        "template": atom.template,
    }


def make_anchor_query(rng: random.Random, quotes: QuoteBuilder, polite_rate: float) -> dict[str, object]:
    roll = rng.random()
    if roll < 0.45:
        atom = bare_literal_atom(rng)
    elif roll < 0.8:
        atom = literal_atom(rng, quotes)
    else:
        atom = class_atom(rng, quant=random_quant(rng, allow_empty=False))
    mode = rng.choice(["starts", "ends"])
    if mode == "starts":
        query = regex_prefix(rng) + rng.choice(
            [
                f"starts with {atom.words}",
                f"match starts with {atom.words}",
                f"text starts with {atom.words}",
                f"line starts with {atom.words}",
                f"lines that start with {atom.words}",
            ]
        )
        il = f"LINE_START; {atom.il};"
        template = "(?m)^" + atom.template
    else:
        query = regex_prefix(rng) + rng.choice(
            [
                f"ends with {atom.words}",
                f"match ends with {atom.words}",
                f"text ends with {atom.words}",
                f"line ends with {atom.words}",
                f"lines that end with {atom.words}",
            ]
        )
        il = f"LINE_END; {atom.il};"
        template = "(?m)" + atom.template + "$"
    return {
        "band": "anchor",
        "refs": quotes.refs,
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": il,
        "template": template,
    }


def make_not_followed_query(rng: random.Random, quotes: QuoteBuilder, polite_rate: float) -> dict[str, object]:
    left = literal_atom(rng, quotes) if rng.random() < 0.75 else bare_literal_atom(rng, min_len=1, max_len=4)
    right = literal_atom(rng, quotes) if rng.random() < 0.75 else bare_literal_atom(rng, min_len=1, max_len=5)
    mode = rng.choice(["word_start", "text"])
    if mode == "word_start":
        query = regex_prefix(rng) + rng.choice(
            [
                f"word starts with {left.words} not followed by {right.words}",
                f"words start with {left.words} not followed by {right.words}",
                f"words that start with {left.words} not followed by {right.words}",
                f"word starts with {left.words} not immediately followed by {right.words}",
                f"words starting with {left.words} but not followed by {right.words}",
            ]
        )
        il = f"WORD_NOT_FOLLOWED; LEFT({left.il}); RIGHT({right.il});"
        template = r"\b" + left.template + "(?!" + right.template + r")[A-Za-z]*\b"
    else:
        query = regex_prefix(rng) + rng.choice(
            [
                f"{left.words} not followed by {right.words}",
                f"match {left.words} not followed by {right.words}",
                f"text: {left.words} not followed by {right.words}",
                f"{left.words} not immediately followed by {right.words}",
            ]
        )
        il = f"NOT_FOLLOWED; LEFT({left.il}); RIGHT({right.il});"
        template = left.template + "(?!" + right.template + ")"
    return {
        "band": "not_followed",
        "refs": quotes.refs,
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": il,
        "template": template,
    }


def make_prefix_select_query(rng: random.Random, quotes: QuoteBuilder, polite_rate: float) -> dict[str, object]:
    prefix = literal_atom(rng, quotes) if rng.random() < 0.65 else bare_literal_atom(rng, min_len=1, max_len=4)
    selected = literal_atom(rng, quotes) if rng.random() < 0.65 else bare_literal_atom(rng, min_len=1, max_len=5)
    query = regex_prefix(rng) + rng.choice(
        [
            f"starts with {prefix.words} select {selected.words}",
            f"word starts with {prefix.words} select {selected.words}",
            f"words that start with {prefix.words}; select {selected.words}",
            f"match words starting with {prefix.words} and select {selected.words}",
            f"starts with {prefix.words}, then select {selected.words}",
        ]
    )
    return {
        "band": "prefix_select",
        "refs": quotes.refs,
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": f"WORD_PREFIX_SELECT; LEFT({prefix.il}); SELECT({selected.il});",
        "template": r"\b" + prefix.template + "(" + selected.template + r")[A-Za-z]*\b",
    }


def make_selection_grammar_query(rng: random.Random, quotes: QuoteBuilder, polite_rate: float) -> dict[str, object]:
    selected_atoms, selected_words = selected_sequence(rng, quotes)
    selected = Atom(words=selected_words, il=seq_il(selected_atoms), template=seq_template(selected_atoms))
    mode = rng.choice(["select_only", "prefix_select"])
    if mode == "select_only":
        query = regex_prefix(rng) + rng.choice(
            [
                f"select {selected.words}",
                f"select {selected.words}; end select",
                f"start select: {selected.words}; end select",
                f"begin select {selected.words} end select",
                f"select {selected.words} as the match",
            ]
        )
        return {
            "band": "selection_grammar",
            "refs": quotes.refs,
            "input": apply_polite_noise(query, rng, polite_rate),
            "il": f"FIND; SEQ(CAP1({selected.il})); SELECT(CAP1);",
            "template": "(" + selected.template + ")",
        }

    prefix = literal_atom(rng, quotes) if rng.random() < 0.7 else bare_literal_atom(rng, min_len=1, max_len=7)
    query = regex_prefix(rng) + rng.choice(
        [
            f"starts with {prefix.words}, select {selected.words}",
            f"starts with {prefix.words}; select {selected.words}",
            f"starts with {prefix.words} then select {selected.words}",
            f"word starts with {prefix.words}, select {selected.words}",
            f"words that start with {prefix.words}; start select: {selected.words}; end select",
        ]
    )
    return {
        "band": "selection_grammar",
        "refs": quotes.refs,
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": f"WORD_PREFIX_SELECT; LEFT({prefix.il}); SELECT({selected.il});",
        "template": r"\b" + prefix.template + "(" + selected.template + r")[A-Za-z]*\b",
    }


def make_code_line_query(rng: random.Random, quotes: QuoteBuilder, polite_rate: float) -> dict[str, object]:
    name = ref_value_atom(quotes, rng.choice(CODE_IDENTIFIERS))
    value = ref_value_atom(quotes, rng.choice(["int", "float", "Tensor", "None", "True", "False"]))
    mode = rng.choice(["annotation", "assignment", "literal"])
    if mode == "annotation":
        query = regex_prefix(rng) + rng.choice(
            [
                f"line starts with whitespace then {name.words} then a colon followed by whitespace and then {value.words}",
                f"line starts with white space then {name.words} then a colon followed by whitespace and then {value.words}",
                f"lines that start with whitespace then {name.words}: whitespace {value.words}",
                f"starts with whitespace then {name.words} colon whitespace {value.words}",
            ]
        )
        il = f"LINE_START; SEQ(SPACE+,{name.il},LIT(\":\"),SPACE+,{value.il});"
        template = r"(?m)^\s+" + name.template + r":\s+" + value.template
    elif mode == "assignment":
        query = regex_prefix(rng) + rng.choice(
            [
                f"line starts with whitespace then {name.words} followed by optional whitespace, equals, optional whitespace",
                f"starts with whitespace then {name.words} then equals",
                f"lines that start with whitespace then {name.words} equals",
            ]
        )
        il = f"LINE_START; SEQ(SPACE+,{name.il},SPACE*,LIT(\"=\"),SPACE*);"
        template = r"(?m)^\s+" + name.template + r"\s*=\s*"
    else:
        query = regex_prefix(rng) + rng.choice(
            [
                f"match {name.words}",
                f"find {name.words}",
                f"text: {name.words}",
            ]
        )
        il = f"FIND; SEQ({name.il});"
        template = name.template
    return {
        "band": "code_line",
        "refs": quotes.refs,
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": il,
        "template": template,
    }


def make_capture_query(rng: random.Random, quotes: QuoteBuilder, counts: list[int], polite_rate: float) -> dict[str, object]:
    scope = rng.choices(["text", "line", "sentence", "whole"], weights=[43, 22, 12, 23], k=1)[0]
    count = max(2, rng.choice(counts))
    capture_count = min(rng.choice([1, 1, 1, 2, 2, 3]), count - 1)
    capture_slots = sorted(rng.sample(range(count), capture_count))
    names = CAPTURE_NAMES[:capture_count]
    atoms: list[Atom] = []
    rendered = []
    cap_idx = 1
    name_idx = 0
    for slot in range(count):
        atom = plain_atom(rng, quotes)
        if slot in capture_slots:
            name = names[name_idx]
            rendered.append(capture_phrase(atom, name, rng))
            atoms.append(capture_atom(atom, cap_idx))
            cap_idx += 1
            name_idx += 1
        else:
            rendered.append(atom.words)
            atoms.append(atom)
    anchor = {"text": "FIND", "line": "FIND", "sentence": "FIND", "whole": "FULL"}[scope]
    template = seq_template(atoms)
    if anchor == "FULL":
        template = "^" + template + "$"
    action = capture_action(names, rng)
    query = regex_prefix(rng) + scoped_query(scope, connector_join(rendered, rng), rng) + f". {action}"
    return {
        "band": "capture",
        "refs": quotes.refs,
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": f"{anchor}; {seq_il(atoms)}; SELECT({','.join(f'CAP{i}' for i in range(1, cap_idx))});",
        "template": template,
    }


def make_select_query(rng: random.Random, quotes: QuoteBuilder, counts: list[int], polite_rate: float) -> dict[str, object]:
    scope = rng.choices(["text", "line", "sentence", "whole"], weights=[48, 22, 12, 18], k=1)[0]
    count = max(2, rng.choice(counts))
    select_slot = rng.randrange(count)
    atoms: list[Atom] = []
    rendered = []
    for slot in range(count):
        atom = plain_atom(rng, quotes)
        if slot == select_slot:
            rendered.append(select_phrase(atom, rng))
            atoms.append(capture_atom(atom, 1))
        else:
            rendered.append(atom.words)
            atoms.append(atom)
    anchor = {"text": "FIND", "line": "FIND", "sentence": "FIND", "whole": "FULL"}[scope]
    template = seq_template(atoms)
    if anchor == "FULL":
        template = "^" + template + "$"
    query = regex_prefix(rng) + scoped_query(scope, connector_join(rendered, rng), rng)
    return {
        "band": "select",
        "refs": quotes.refs,
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": f"{anchor}; {seq_il(atoms)}; SELECT(CAP1);",
        "template": template,
    }


def make_word_query(rng: random.Random, quotes: QuoteBuilder, polite_rate: float) -> dict[str, object]:
    name = rng.choice(["a", "b", "A"])
    mode = rng.choice(["starts", "starts_class", "ends", "contains", "length", "class"])
    if mode == "starts":
        value = word_literal(rng)
        ref = len(quotes.refs)
        quotes.refs.append(value)
        rest = Atom(words="letters", il="ANY_LETTER*", template="[A-Za-z]*")
        capture = rng.random() < 0.55
        atoms = [Atom(quote(value), f"REF{ref}", f"<{ref}>"), capture_atom(rest, 1) if capture else rest]
        base = rng.choice([f"words that start with {quote(value)}", f"words starting with {quote(value)}"])
        if capture:
            base += rng.choice(
                [
                    "; give me the rest",
                    "; keep what comes after " + quote(value),
                    "; start select: the rest; end select",
                ]
            )
    elif mode == "starts_class":
        quant = random_quant(rng, allow_empty=False, allow_zero=False)
        first = class_atom(rng, quant=quant, word_safe=True)
        rest = Atom(words="letters", il="ANY_LETTER*", template="[A-Za-z]*")
        capture = rng.random() < 0.45
        atoms = [first, capture_atom(rest, 1) if capture else rest]
        base = rng.choice(
            [
                f"words that start with {first.words}",
                f"words starting with {first.words}",
            ]
        )
        if capture:
            base += rng.choice(["; give me the rest", "; keep the rest", "; start select: the rest; end select"])
    elif mode == "ends":
        value = rng.choice(["ing", "ed", "ly", "tion", "er"])
        ref = len(quotes.refs)
        quotes.refs.append(value)
        start = Atom(words="letters", il="ANY_LETTER*", template="[A-Za-z]*")
        capture = rng.random() < 0.55
        atoms = [capture_atom(start, 1) if capture else start, Atom(quote(value), f"REF{ref}", f"<{ref}>")]
        base = rng.choice([f"words that end with {quote(value)}", f"words ending in {quote(value)}"])
        if capture:
            base += rng.choice(
                [
                    f"; give me what comes before {quote(value)}",
                    f"; keep the part before {quote(value)}",
                    "; start select: the part before the ending; end select",
                ]
            )
    elif mode == "contains":
        value = word_literal(rng, min_len=2, max_len=4)
        ref = len(quotes.refs)
        quotes.refs.append(value)
        left = Atom(words="letters", il="ANY_LETTER*", template="[A-Za-z]*")
        right = Atom(words="letters", il="ANY_LETTER*", template="[A-Za-z]*")
        atoms = [left, Atom(quote(value), f"REF{ref}", f"<{ref}>"), right]
        base = rng.choice([f"words containing {quote(value)}", f"words that contain {quote(value)}"])
    elif mode == "length":
        n = rng.randint(2, 12)
        atom = Atom(words=f"{n} letters", il=f"ANY_LETTER{{{n}}}", template=f"[A-Za-z]{{{n}}}")
        atoms = [atom]
        base = rng.choice([f"words with {n} letters", f"words of {n} letters"])
    else:
        kind, template, phrase, word_phrase = rng.choice(
            [
                ("LOWER+", "[a-z]+", "lowercase letters", "lowercase words"),
                ("UPPER+", "[A-Z]+", "uppercase letters", "uppercase words"),
                ("VOWEL+", "[AEIOUaeiou]+", "vowels", "words made only of vowels"),
                ("ANY_LETTER+", "[A-Za-z]+", "letters", "words made of letters"),
            ]
        )
        atom = Atom(words=phrase, il=kind, template=template)
        atoms = [atom]
        base = rng.choice([f"words made of {atom.words}", word_phrase])
    template = r"\b" + seq_template(atoms) + r"\b"
    select = "; SELECT(CAP1);" if any(atom.il.startswith("CAP1(") for atom in atoms) else ""
    query = regex_prefix(rng) + rng.choice(
        [
            "words: " + base,
            base,
            base.replace("words ", "all words ", 1) if base.startswith("words ") else base,
        ]
    )
    return {
        "band": "words",
        "refs": quotes.refs,
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": f"WORD; {seq_il(atoms)}{select if select else ';'}",
        "template": template,
    }


def make_order_query(rng: random.Random, quotes: QuoteBuilder, polite_rate: float) -> dict[str, object]:
    left = class_atom(rng, quant=random_quant(rng, allow_empty=False, allow_zero=False)) if rng.random() < 0.45 else literal_atom(rng, quotes)
    right = class_atom(rng, quant=random_quant(rng, allow_empty=False, allow_zero=False)) if rng.random() < 0.45 else literal_atom(rng, quotes)
    mode = rng.choice(["BEFORE", "AFTER"])
    if mode == "BEFORE":
        query = regex_prefix(rng) + rng.choice(
            [
                f"text: {left.words} before {right.words}",
                f"lines: {left.words} before {right.words}",
                f"sentences: {left.words} before {right.words}",
                f"text that has {left.words} before {right.words}",
                f"lines that have {left.words} before {right.words}",
                f"sentences that have {left.words} before {right.words}",
            ]
        )
        template = left.template + ".*" + right.template
    else:
        query = regex_prefix(rng) + rng.choice(
            [
                f"text: {left.words} after {right.words}",
                f"lines: {left.words} after {right.words}",
                f"sentences: {left.words} after {right.words}",
                f"text that has {left.words} after {right.words}",
                f"lines that have {left.words} after {right.words}",
                f"sentences that have {left.words} after {right.words}",
            ]
        )
        template = right.template + ".*" + left.template
    return {
        "band": "order",
        "refs": quotes.refs,
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": f"{mode}; LEFT({left.il}); RIGHT({right.il});",
        "template": template,
    }


def replacement_template(names: list[str], rng: random.Random) -> str:
    if len(names) == 1:
        choices = [
            ['"["', cap_name(names[0]), '"]"'],
            ['"prefix-"', cap_name(names[0])],
            [cap_name(names[0]), '"-suffix"'],
        ]
    elif len(names) == 2:
        choices = [
            [cap_name(names[1]), '", "', cap_name(names[0])],
            [cap_name(names[0]), '"-"', cap_name(names[1])],
            ['"["', cap_name(names[0]), '"]"', cap_name(names[1])],
            [cap_name(names[1]), '" #"', cap_name(names[0])],
        ]
    else:
        choices = [
            [cap_name(names[0]), '"-"', cap_name(names[1]), '"-"', cap_name(names[2])],
            [cap_name(names[2]), '", "', cap_name(names[1]), '", "', cap_name(names[0])],
            ['"["', cap_name(names[0]), '"]"', cap_name(names[1]), '"("', cap_name(names[2]), '")"'],
        ]
    return ", ".join(rng.choice(choices))


def add_literal_ref(quotes: QuoteBuilder, value: str) -> int:
    quotes.refs.append(value)
    return len(quotes.refs) - 1


def ref_replacement_literals(replacement: str, quotes: QuoteBuilder) -> tuple[str, str]:
    refs: list[tuple[str, int]] = []

    def collect(match: re.Match[str]) -> str:
        ref = add_literal_ref(quotes, match.group(1))
        refs.append((match.group(0), ref))
        return f"REF{ref}"

    il = re.sub(r'"([^"\n]*)"', collect, replacement)
    template = replacement
    for quoted, ref in refs:
        template = template.replace(quoted, f"<{ref}>", 1)
    return il, template


def make_replace_query(rng: random.Random, quotes: QuoteBuilder, counts: list[int], polite_rate: float) -> dict[str, object]:
    capture_count = rng.choice([1, 2, 2, 3])
    count = max(capture_count, rng.choice(counts))
    capture_slots = sorted(rng.sample(range(count), capture_count))
    names = CAPTURE_NAMES[:capture_count]
    atoms: list[Atom] = []
    rendered = []
    cap_idx = 1
    name_idx = 0
    for slot in range(count):
        atom = plain_atom(rng, quotes)
        if slot in capture_slots:
            atoms.append(capture_atom(atom, cap_idx))
            if capture_count == 1:
                rendered.append(select_phrase(atom, rng))
            else:
                name = names[name_idx]
                rendered.append(capture_phrase(atom, name, rng))
            cap_idx += 1
            name_idx += 1
        else:
            atoms.append(atom)
            rendered.append(atom.words)
    action = rng.choice(["replace with", "rewrite as", "change the match to"])
    if capture_count == 1:
        replacement_words = rng.choice(
            [
                selection_token(rng),
                '"["' + ", " + selection_token(rng) + ', "]"',
                '"prefix-", ' + selection_token(rng),
                selection_token(rng) + ', "-suffix"',
            ]
        )
    else:
        replacement_words = replacement_template(names, rng)
    replacement_il, replacement_template_text = ref_replacement_literals(replacement_words, quotes)
    scope = rng.choices(["text", "sentence", "whole"], weights=[55, 15, 30], k=1)[0]
    anchor = "FULL" if scope == "whole" else "FIND"
    template = seq_template(atoms)
    if anchor == "FULL":
        template = "^" + template + "$"
    query = regex_prefix(rng) + scoped_query(scope, connector_join(rendered, rng), rng) + f". {action} {replacement_words}"
    return {
        "band": "replace",
        "refs": quotes.refs,
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": f"{anchor}; {seq_il(atoms)}; REPLACE({replacement_il});",
        "template": template + " => " + replacement_template_text,
    }


def make_delete_insert_query(rng: random.Random, quotes: QuoteBuilder, polite_rate: float) -> dict[str, object]:
    atom = plain_atom(rng, quotes)
    action = rng.choice(["delete", "remove", "append", "prepend", "surround"])
    subject, verb = rng.choice([("text", "has"), ("lines", "have"), ("sentences", "have")])
    prefix = rng.choice([f"{subject}: {atom.words}", f"{subject} that {verb} {atom.words}", f"{subject} containing {atom.words}"])
    if action in {"delete", "remove"}:
        query = regex_prefix(rng) + f"{prefix}. {action} the match"
        il_action = "DELETE();"
        template_action = "DELETE"
    elif action == "append":
        value = word_literal(rng, min_len=1, max_len=3)
        ref = add_literal_ref(quotes, value)
        query = regex_prefix(rng) + f"{prefix}. append {quote(value)}"
        il_action = f"APPEND(REF{ref});"
        template_action = f"APPEND <{ref}>"
    elif action == "prepend":
        value = word_literal(rng, min_len=1, max_len=3)
        ref = add_literal_ref(quotes, value)
        query = regex_prefix(rng) + f"{prefix}. prepend {quote(value)}"
        il_action = f"PREPEND(REF{ref});"
        template_action = f"PREPEND <{ref}>"
    else:
        left, right = quote("["), quote("]")
        left_ref = add_literal_ref(quotes, "[")
        right_ref = add_literal_ref(quotes, "]")
        query = regex_prefix(rng) + f"{prefix}. surround with {left} and {right}"
        il_action = f"SURROUND(REF{left_ref},REF{right_ref});"
        template_action = f"SURROUND <{left_ref}> <{right_ref}>"
    return {
        "band": "edit",
        "refs": quotes.refs,
        "input": apply_polite_noise(query, rng, polite_rate),
        "il": f"FIND; SEQ({atom.il}); {il_action}",
        "template": atom.template + " => " + template_action,
    }


def make_example(
    rng: random.Random,
    *,
    component_counts: list[int],
    string_min_len: int,
    string_max_len: int,
    polite_rate: float,
) -> dict[str, object]:
    quotes = QuoteBuilder(rng, min_len=string_min_len, max_len=string_max_len)
    bands = {
        "match": 22,
        "literal": 10,
        "anchor": 14,
        "not_followed": 10,
        "prefix_select": 8,
        "selection_grammar": 10,
        "code_line": 8,
        "capture": 16,
        "select": 14,
        "words": 18,
        "order": 12,
        "replace": 14,
        "edit": 6,
    }
    band = rng.choices(tuple(bands), weights=tuple(bands.values()), k=1)[0]
    if band == "match":
        return make_match_query(rng, quotes, component_counts, polite_rate)
    if band == "literal":
        return make_literal_query(rng, polite_rate)
    if band == "anchor":
        return make_anchor_query(rng, quotes, polite_rate)
    if band == "not_followed":
        return make_not_followed_query(rng, quotes, polite_rate)
    if band == "prefix_select":
        return make_prefix_select_query(rng, quotes, polite_rate)
    if band == "selection_grammar":
        return make_selection_grammar_query(rng, quotes, polite_rate)
    if band == "code_line":
        return make_code_line_query(rng, quotes, polite_rate)
    if band == "capture":
        return make_capture_query(rng, quotes, component_counts, polite_rate)
    if band == "select":
        return make_select_query(rng, quotes, component_counts, polite_rate)
    if band == "words":
        return make_word_query(rng, quotes, polite_rate)
    if band == "order":
        return make_order_query(rng, quotes, polite_rate)
    if band == "replace":
        return make_replace_query(rng, quotes, component_counts, polite_rate)
    if band == "edit":
        return make_delete_insert_query(rng, quotes, polite_rate)
    raise ValueError(f"unknown band: {band}")


def format_example(example: dict[str, object]) -> str:
    return f"Input:\n{example['input']}\n\nIL:\n{example['il']}\n\nTemplate:\n{example['template']}"


def main() -> None:
    args = parse_args()
    if args.examples <= 0:
        raise ValueError("--examples must be positive")
    if not 0 <= args.polite_rate <= 1:
        raise ValueError("--polite-rate must be in 0..1")
    component_counts = parse_component_counts(args.component_counts)
    band_weights = {"match": 22, "literal": 10, "anchor": 14, "not_followed": 10, "prefix_select": 8, "selection_grammar": 10, "code_line": 8, "capture": 16, "select": 14, "words": 18, "order": 12, "replace": 14, "edit": 6}
    config = RegexIlV7Config(
        examples=args.examples,
        seed=args.seed,
        string_min_len=args.string_min_len,
        string_max_len=args.string_max_len,
        component_counts=args.component_counts,
        preview=args.preview,
        band_weights=band_weights,
        polite_rate=args.polite_rate,
    )
    rng = random.Random(args.seed)
    examples = [
        make_example(
            rng,
            component_counts=component_counts,
            string_min_len=args.string_min_len,
            string_max_len=args.string_max_len,
            polite_rate=args.polite_rate,
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
        "note": "v7 dream-language regex corpus. Semantic structures are generated first, then rendered through compatible phrase templates.",
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
