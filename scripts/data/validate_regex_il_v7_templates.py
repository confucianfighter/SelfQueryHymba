from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_METADATA = ROOT / "data" / "regex_il_v7_dream_queries_120k.sources.json"

CLASS_REGEX = {
    "DIGIT": r"\d",
    "ANY_LETTER": "[A-Za-z]",
    "LOWER": "[a-z]",
    "UPPER": "[A-Z]",
    "WORD": r"\w",
    "SPACE": r"\s",
    "VOWEL": "[AEIOUaeiou]",
}

LITERAL_REGEX = {
    "-": "-",
    "_": "_",
    ".": r"\.",
    ",": ",",
    ":": ":",
    "@": "@",
    "+": r"\+",
    "/": r"\/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate v7 regex IL/template consistency.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--regenerate", action="store_true", help="Regenerate all examples from metadata config and validate them.")
    parser.add_argument("--max-errors", type=int, default=20)
    return parser.parse_args()


def split_top_level(value: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_quote = False
    escape = False
    for index, char in enumerate(value):
        if in_quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_quote = False
            continue
        if char == '"':
            in_quote = True
        elif char in "({":
            depth += 1
        elif char in ")}":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def split_statements(il: str) -> list[str]:
    return [part.strip() for part in il.strip().rstrip(";").split(";") if part.strip()]


def unwrap_call(value: str, name: str) -> str:
    prefix = name + "("
    if not value.startswith(prefix) or not value.endswith(")"):
        raise ValueError(f"expected {name}(...), got {value!r}")
    return value[len(prefix) : -1]


def class_regex(expr: str) -> str | None:
    for kind, regex in CLASS_REGEX.items():
        if expr.startswith(kind):
            return regex + expr[len(kind) :]
    return None


def expr_regex(expr: str) -> str:
    expr = expr.strip()
    if expr.startswith("SEQ("):
        return "".join(expr_regex(part) for part in split_top_level(unwrap_call(expr, "SEQ")))
    if expr.startswith("ALT("):
        return "(?:" + "|".join(expr_regex(part) for part in split_top_level(unwrap_call(expr, "ALT"))) + ")"
    if re.fullmatch(r"CAP\d+\(.+\)", expr):
        name_end = expr.index("(")
        inner = expr[name_end + 1 : -1]
        return "(" + expr_regex(inner) + ")"
    if re.fullmatch(r"REF\d+", expr):
        return f"<{expr[3:]}>"
    if expr.startswith('LIT("') and expr.endswith('")'):
        value = expr[5:-2]
        return LITERAL_REGEX.get(value, re.escape(value))
    result = class_regex(expr)
    if result is not None:
        return result
    raise ValueError(f"unsupported IL expression: {expr!r}")


def expected_template(example: dict[str, object]) -> str:
    il = str(example["il"])
    statements = split_statements(il)
    op = statements[0]
    if op in {"FIND", "FULL"}:
        base = expr_regex(statements[1])
        if op == "FULL":
            base = "^" + base + "$"
        if len(statements) == 2 or statements[2].startswith("SELECT("):
            return base
        if statements[2].startswith("REPLACE("):
            return base + " => " + action_template(unwrap_call(statements[2], "REPLACE"))
        if statements[2].startswith("DELETE("):
            return base + " => DELETE"
        if statements[2].startswith("APPEND("):
            return base + " => APPEND " + action_template(unwrap_call(statements[2], "APPEND"))
        if statements[2].startswith("PREPEND("):
            return base + " => PREPEND " + action_template(unwrap_call(statements[2], "PREPEND"))
        if statements[2].startswith("SURROUND("):
            return base + " => SURROUND " + " ".join(action_template(part) for part in split_top_level(unwrap_call(statements[2], "SURROUND")))
        raise ValueError(f"unsupported action statement: {statements[2]!r}")
    if op == "WORD":
        base = r"\b" + expr_regex(statements[1]) + r"\b"
        return base
    if op == "LINE_START":
        return "(?m)^" + expr_regex(statements[1])
    if op == "LINE_END":
        return "(?m)" + expr_regex(statements[1]) + "$"
    if op == "NOT_FOLLOWED":
        left = expr_regex(unwrap_call(statements[1], "LEFT"))
        right = expr_regex(unwrap_call(statements[2], "RIGHT"))
        return left + "(?!" + right + ")"
    if op == "WORD_NOT_FOLLOWED":
        left = expr_regex(unwrap_call(statements[1], "LEFT"))
        right = expr_regex(unwrap_call(statements[2], "RIGHT"))
        return r"\b" + left + "(?!" + right + r")[A-Za-z]*\b"
    if op == "WORD_PREFIX_SELECT":
        left = expr_regex(unwrap_call(statements[1], "LEFT"))
        selected = expr_regex(unwrap_call(statements[2], "SELECT"))
        return r"\b" + left + "(" + selected + r")[A-Za-z]*\b"
    if op in {"BEFORE", "AFTER"}:
        left = expr_regex(unwrap_call(statements[1], "LEFT"))
        right = expr_regex(unwrap_call(statements[2], "RIGHT"))
        return left + ".*" + right if op == "BEFORE" else right + ".*" + left
    raise ValueError(f"unsupported op: {op!r}")


def action_template(value: str) -> str:
    return re.sub(r"\bREF(\d+)\b", r"<\1>", value)


def regex_part(template: str) -> str:
    return template.split(" => ", 1)[0]


def expand_refs(pattern: str, refs: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index >= len(refs):
            raise ValueError(f"REF{index} not present in refs {refs!r}")
        return re.escape(refs[index])

    return re.sub(r"<(\d+)>", replace, pattern)


def validate_example(example: dict[str, object], index: int) -> list[str]:
    errors: list[str] = []
    actual = str(example["template"])
    try:
        expected = expected_template(example)
    except Exception as exc:
        return [f"#{index}: failed to derive template: {exc}; il={example.get('il')!r}"]
    if actual != expected:
        errors.append(f"#{index}: template mismatch\n  expected: {expected}\n  actual:   {actual}\n  input:    {example.get('input')}")
    try:
        re.compile(expand_refs(regex_part(actual), list(example.get("refs", []))))
    except Exception as exc:
        errors.append(f"#{index}: template regex does not compile: {exc}\n  template: {actual}\n  refs:     {example.get('refs')}")
    return errors


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if args.regenerate:
        from scripts.data.prepare_regex_il_v3_traces import parse_component_counts
        from scripts.data.prepare_regex_il_v7_dream_queries import make_example

        config = metadata["config"]
        component_counts = parse_component_counts(config["component_counts"])
        rng = random.Random(int(config["seed"]))
        examples = [
            make_example(
                rng,
                component_counts=component_counts,
                string_min_len=int(config["string_min_len"]),
                string_max_len=int(config["string_max_len"]),
                polite_rate=float(config["polite_rate"]),
            )
            for _ in range(int(config["examples"]))
        ]
    else:
        examples = metadata.get("examples") or metadata.get("first_examples")
        if examples is None:
            raise ValueError(f"{args.metadata} has no examples/first_examples field")
    total = len(examples)
    errors: list[str] = []
    for index, example in enumerate(examples, start=1):
        errors.extend(validate_example(example, index))
        if len(errors) >= args.max_errors:
            break
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(json.dumps({"metadata": str(args.metadata), "validated_examples": total, "errors": 0}, indent=2))


if __name__ == "__main__":
    main()
