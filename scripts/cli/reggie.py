from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_addition_trace_lm import load_model, model_logits  # noqa: E402


DEFAULT_CHECKPOINT = (
    ROOT
    / "experiments"
    / "mods"
    / "001_first_pass_hymba_cst"
    / "runs"
    / "2026-05-17_alpine_regex_v7_code_line_continue_1000_alpha02_lr0002"
    / "checkpoint_best.pt"
)
SENTINEL = "<END>"

RESET = "\x1b[0m"
DIM = "\x1b[2m"
MATCH_BG = "\x1b[30;43m"
REPLACE_BG = "\x1b[30;46m"
PATH_FG = "\x1b[36m"
ERR_FG = "\x1b[31m"


@dataclass(frozen=True)
class ReggieProgram:
    query: str
    prompt: str
    raw_output: str
    il: str | None
    template: str
    pattern: str
    replacement: str | None


@dataclass(frozen=True)
class ModelBundle:
    model: object
    vocab: object
    device: torch.device


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    examples = f"""examples:
  reggie ./notes
      Start interactive mode, then type: words that start with "todo"

  reggie ./notes --interactive
      Explicitly start repeated interactive mode.

  reggie app.log
      Start interactive mode, then type: lines that have "ERROR" before digits

  reggie app.log 'lines that have "ERROR" before digits'
      Show matching lines, with the matched span highlighted.

  reggie src
      Start interactive mode, then type: text: begin select: "user_"; end select followed by digits

  reggie src 'text: begin select: "user_"; end select followed by digits'
      Highlight only the selected part when the model emits a capture/select template.

  reggie names.txt
      Start interactive mode, then type: whole string: capture letters as 'a', then a comma, then capture letters as 'b'. replace with 'b', ", ", 'a'

  reggie names.txt 'whole string: capture letters as ''a'', then a comma, then capture letters as ''b''. replace with ''b'', ", ", ''a'''
      Show a replacement preview. Matches use yellow; replacement text uses cyan.

  reggie . --show-regex
      Start interactive mode, then type: either "cat" or "dog"

  reggie . 'either "cat" or "dog"' --show-regex
      Print the generated IL/template/regex before scanning files.

  reggie README.md --checkpoint {DEFAULT_CHECKPOINT}
      Use an explicit checkpoint.
"""
    parser = argparse.ArgumentParser(
        prog="reggie",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Grep-style search powered by the Alpine regex model. "
            "Pass a file or folder and a plain-English regex request; Reggie prepends /r, "
            "asks the model for a regex template, expands quoted refs, and highlights matches."
        ),
        epilog=examples,
    )
    parser.add_argument("path", help="File or folder to scan.")
    parser.add_argument(
        "query",
        nargs="*",
        help="Plain-English regex request. Omit it for interactive mode. /r or /regex is optional.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Model checkpoint to use. Defaults to the current Alpine regex-v7 ALT checkpoint.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq-len", type=int, default=384, help="Model context length.")
    parser.add_argument("--max-new-chars", type=int, default=260, help="Generation budget for IL/template output.")
    parser.add_argument("--ignore-case", "-i", action="store_true", help="Compile the generated regex with IGNORECASE.")
    parser.add_argument("--multiline", action="store_true", help="Compile the generated regex with MULTILINE.")
    parser.add_argument("--all-text", action="store_true", help="Try to read every file instead of using a text-extension filter.")
    parser.add_argument("--max-matches", type=int, default=200, help="Stop after this many matches. Use 0 for no limit.")
    parser.add_argument("--context", "-C", type=int, default=0, help="Characters of context to show around each match.")
    parser.add_argument("--show-regex", action="store_true", help="Print generated IL, template, and expanded regex before scanning.")
    parser.add_argument("--interactive", action="store_true", help="Start a repeated prompt for queries. Type exit or quit to stop.")
    parser.add_argument("--no-feedback", action="store_true", help="Do not ask whether interactive results were correct.")
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=ROOT / "SampleOutputs" / "reggie_failures.jsonl",
        help="Where interactive incorrect-result reports are saved.",
    )
    parser.add_argument(
        "--debug-input",
        action="store_true",
        help="Print argv, parsed query, exact model prompt, raw model output, and regex before scanning.",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    return parser


def normalize_query(query_parts: list[str]) -> str:
    stripped = " ".join(query_parts).strip()
    stripped = canonicalize_query_text(stripped)
    lowered = stripped.lower()
    if lowered.startswith("/r ") or lowered.startswith("/regex "):
        return stripped
    return "/r " + stripped


GRAMMAR_WORDS = {
    "a",
    "all",
    "an",
    "and",
    "any",
    "as",
    "at",
    "before",
    "begin",
    "between",
    "by",
    "capture",
    "colon",
    "comma",
    "containing",
    "contains",
    "digit",
    "digits",
    "dot",
    "end",
    "ends",
    "followed",
    "from",
    "line",
    "lines",
    "match",
    "not",
    "one",
    "or",
    "page",
    "regex",
    "select",
    "space",
    "starts",
    "text",
    "then",
    "the",
    "to",
    "whitespace",
    "white",
    "with",
    "word",
    "words",
}


def canonicalize_query_text(query: str) -> str:
    parts = re.split(r'("[^"\n]*")', query)
    for index in range(0, len(parts), 2):
        parts[index] = canonicalize_unquoted_text(parts[index])
    return "".join(parts)


def canonicalize_unquoted_text(text: str) -> str:
    text = re.sub(r"\bwhite\s+space\b", "whitespace", text, flags=re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        core = token.rstrip(":")
        trailing = token[len(core) :]
        if should_quote_bare_token(core, trailing):
            return f'"{core}{trailing}"'
        return token

    return re.sub(r"\b[A-Za-z][A-Za-z0-9_]*:?", replace, text)


def should_quote_bare_token(core: str, trailing: str) -> bool:
    if not core:
        return False
    lowered = core.lower()
    if lowered in GRAMMAR_WORDS:
        return False
    if "_" in core:
        return True
    if trailing == ":" and lowered not in {"text", "line", "lines", "words", "regex"}:
        return True
    has_lower = any(char.islower() for char in core)
    has_upper = any(char.isupper() for char in core)
    if has_lower and has_upper:
        return True
    return False


@torch.no_grad()
def generate_completion(model, vocab, prompt: str, *, seq_len: int, max_new_chars: int, device: torch.device) -> str:
    ids = vocab.encode(prompt, device=device).tolist()
    prompt_len = len(prompt)
    model.eval()
    for _ in range(max_new_chars):
        context = torch.tensor([ids[-seq_len:]], dtype=torch.long, device=device)
        logits = model_logits(model, context, pad_to_length=seq_len)[:, -1, :]
        ids.append(int(logits.argmax(dim=-1).item()))
        suffix = vocab.decode(ids)[prompt_len:]
        if SENTINEL in suffix:
            break
        if re.search(r"\nTemplate:\n[^\n]+\n", suffix):
            break
    suffix = vocab.decode(ids)[prompt_len:]
    return suffix.split(SENTINEL, 1)[0].strip()


def extract_block(label: str, text: str) -> str | None:
    match = re.search(rf"^{re.escape(label)}:\n(.+?)(?:\n\n[A-Z][A-Za-z ]*:\n|\Z)", text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else None


def extract_template(text: str) -> str:
    match = re.search(r"^Template:\n([^\n]+)", text, re.MULTILINE)
    if not match:
        raise ValueError("model output did not include a Template block")
    return match.group(1).strip()


def expand_refs(template: str, query: str) -> str:
    return expand_template_refs(template, query, escape=True, quote_values=False)


def expand_replacement_refs(template: str | None, query: str) -> str | None:
    if template is None:
        return None
    return expand_template_refs(template, query, escape=False, quote_values=True)


def expand_template_refs(template: str, query: str, *, escape: bool, quote_values: bool) -> str:
    refs = re.findall(r'"([^"\n]*)"', query)
    expanded = template
    for index, value in enumerate(refs):
        replacement = re.escape(value) if escape else value
        if quote_values:
            replacement = quote_replacement_literal(replacement)
        expanded = expanded.replace(f"<{index}>", replacement)
    unresolved = re.findall(r"<\d+>", expanded)
    if unresolved:
        raise ValueError(f"template contains unresolved refs: {', '.join(sorted(set(unresolved)))}")
    return expanded


def quote_replacement_literal(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def load_bundle(args: argparse.Namespace) -> ModelBundle:
    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    device = torch.device(args.device)
    model, vocab, _config = load_model(checkpoint, device)
    return ModelBundle(model=model, vocab=vocab, device=device)


def compile_program(query_parts: list[str], args: argparse.Namespace, bundle: ModelBundle) -> ReggieProgram:
    query = normalize_query(query_parts)
    prompt = f"Task: regex_v5\nInput:\n{query}\n\nOutput:\n"
    raw_output = generate_completion(
        bundle.model,
        bundle.vocab,
        prompt,
        seq_len=args.seq_len,
        max_new_chars=args.max_new_chars,
        device=bundle.device,
    )
    il = extract_block("IL", raw_output)
    template = extract_template(raw_output)
    pattern_template, replacement = split_template_action(template)
    pattern = expand_refs(pattern_template, query)
    replacement = expand_replacement_refs(replacement, query)
    return ReggieProgram(
        query=query,
        prompt=prompt,
        raw_output=raw_output,
        il=il,
        template=template,
        pattern=pattern,
        replacement=replacement,
    )


def infer_program(args: argparse.Namespace, bundle: ModelBundle | None = None) -> ReggieProgram:
    if not args.query:
        raise ValueError("no query provided; use --interactive or pass a query after the path")
    if bundle is None:
        bundle = load_bundle(args)
    return compile_program(args.query, args, bundle)


def split_template_action(template: str) -> tuple[str, str | None]:
    if " => " not in template:
        return template, None
    pattern, action = template.split(" => ", 1)
    return pattern.strip(), action.strip()


def replacement_from_action(action: str, match: re.Match[str]) -> str:
    if action == "DELETE":
        return ""
    if action.startswith("APPEND "):
        return match.group(0) + literal_arg(action.removeprefix("APPEND ").strip())
    if action.startswith("PREPEND "):
        return literal_arg(action.removeprefix("PREPEND ").strip()) + match.group(0)
    if action.startswith("SURROUND "):
        pieces = re.findall(r'"([^"]*)"', action)
        if len(pieces) >= 2:
            return pieces[0] + match.group(0) + pieces[1]
        return match.group(0)
    return render_replacement_expr(action, match)


def literal_arg(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return token[1:-1]
    return token


def render_replacement_expr(expr: str, match: re.Match[str]) -> str:
    parts: list[str] = []
    for token in split_replacement_tokens(expr):
        if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
            parts.append(token[1:-1])
        elif len(token) >= 2 and token[0] == "'" and token[-1] == "'":
            group_index = capture_name_to_index(token[1:-1])
            parts.append(match.group(group_index) if group_index <= len(match.groups()) else "")
        elif token in {"selection", "selected text", "selected part"}:
            parts.append(match.group(1) if match.groups() else match.group(0))
        else:
            parts.append(token)
    return "".join(parts)


def split_replacement_tokens(expr: str) -> list[str]:
    tokens = []
    for match in re.finditer(r'"[^"]*"|' + r"'[^']*'|[^,]+", expr):
        token = match.group(0).strip()
        if token:
            tokens.append(token)
    return tokens


def capture_name_to_index(name: str) -> int:
    order = ["a", "b", "c", "A", "B", "C"]
    return order.index(name) + 1 if name in order else 1


TEXT_EXTENSIONS = {
    ".bat",
    ".c",
    ".cfg",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".log",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def iter_files(path: Path, *, all_text: bool) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for root, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if name not in {".git", "__pycache__", ".pytest_cache"}]
        for name in files:
            file_path = Path(root) / name
            if all_text or file_path.suffix.lower() in TEXT_EXTENSIONS or not file_path.suffix:
                yield file_path


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None
    for encoding in ("utf-8", "utf-16", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def colorize(text: str, color: str, *, enabled: bool) -> str:
    return f"{color}{text}{RESET}" if enabled else text


def mark_match(text: str, *, color: bool) -> str:
    return colorize(text, MATCH_BG, enabled=color) if color else f"[[{text}]]"


def highlighted_match_line(text: str, match: re.Match[str], *, color: bool) -> tuple[int, str]:
    spans = selected_spans(match)
    line_start = text.rfind("\n", 0, min(start for start, _end in spans)) + 1
    line_end = text.find("\n", max(end for _start, end in spans))
    if line_end == -1:
        line_end = len(text)
    line_number = text.count("\n", 0, line_start) + 1
    rendered = []
    cursor = line_start
    for start, end in spans:
        rendered.append(text[cursor:start])
        rendered.append(mark_match(text[start:end], color=color))
        cursor = end
    rendered.append(text[cursor:line_end])
    return line_number, "".join(rendered)


def selected_spans(match: re.Match[str]) -> list[tuple[int, int]]:
    spans = [match.span(index) for index in range(1, len(match.groups()) + 1) if match.span(index) != (-1, -1)]
    if not spans:
        spans = [match.span(0)]
    return merge_spans(spans)


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def scan(program: ReggieProgram, target: Path, args: argparse.Namespace) -> tuple[int, list[dict[str, object]]]:
    flags = re.MULTILINE if args.multiline else 0
    if args.ignore_case:
        flags |= re.IGNORECASE
    regex = re.compile(program.pattern, flags)
    color = not args.no_color
    shown = 0
    shown_matches: list[dict[str, object]] = []
    for file_path in iter_files(target, all_text=args.all_text):
        text = read_text(file_path)
        if text is None:
            continue
        for match in regex.finditer(text):
            rel = file_path if file_path.is_absolute() else file_path
            label = colorize(str(rel), PATH_FG, enabled=color)
            line_number, rendered = highlighted_match_line(text, match, color=color)
            print(f"{label}:{line_number}: {rendered}")
            if len(shown_matches) < 20:
                shown_matches.append(
                    {
                        "path": str(rel),
                        "line_number": line_number,
                        "line": rendered if not color else text[text.rfind("\n", 0, match.start()) + 1 : (text.find("\n", match.end()) if text.find("\n", match.end()) != -1 else len(text))],
                        "match": match.group(0),
                        "groups": list(match.groups()),
                    }
                )
            if program.replacement is not None:
                replacement = replacement_from_action(program.replacement, match)
                replace_label = colorize("replace:", DIM, enabled=color)
                print(f"{replace_label} {colorize(replacement, REPLACE_BG, enabled=color)}")
            shown += 1
            if args.max_matches and shown >= args.max_matches:
                return shown, shown_matches
    return shown, shown_matches


def print_program(program: ReggieProgram, args: argparse.Namespace) -> None:
    if args.debug_input:
        print("debug argv:", repr(sys.argv))
        print("debug path:", repr(args.path))
        print("debug query args:", repr(args.query))
        print("debug normalized query:", repr(program.query))
        print("debug model prompt:", repr(program.prompt))
        print("debug raw model output:", repr(program.raw_output))
        print("debug parsed il:", repr(program.il))
        print("debug parsed template:", repr(program.template))
        print("debug expanded regex:", repr(program.pattern))
        if program.replacement is not None:
            print("debug replacement:", repr(program.replacement))
        print()
    elif args.show_regex or args.interactive:
        print(f"query: {program.query}")
        if program.il:
            print(f"il: {program.il}")
        print(f"template: {program.template}")
        print(f"regex: {program.pattern}")
        if program.replacement is not None:
            print(f"replacement: {program.replacement}")
        print()


def run_interactive(args: argparse.Namespace, target: Path) -> int:
    print("Reggie interactive mode. Type a query and press Enter; type exit or quit to stop.", flush=True)
    bundle = load_bundle(args)
    last_status = 1
    while True:
        try:
            query = input("reggie> ").strip()
        except EOFError:
            print()
            return last_status
        if query.lower() in {"exit", "quit", ":q"}:
            return last_status
        if not query:
            continue
        try:
            program = compile_program([query], args, bundle)
            print_program(program, args)
            count, matches = scan(program, target, args)
            if not args.no_feedback:
                maybe_log_failure(args, program, count=count, matches=matches)
            last_status = 0 if count else 1
        except Exception as exc:  # noqa: BLE001
            color = not getattr(args, "no_color", False)
            print(colorize(f"reggie error: {exc}", ERR_FG, enabled=color), file=sys.stderr)
            last_status = 1


def maybe_log_failure(args: argparse.Namespace, program: ReggieProgram, *, count: int, matches: list[dict[str, object]]) -> None:
    try:
        answer = input("correct? [Y/n] ").strip().lower()
    except EOFError:
        print()
        return
    if answer not in {"n", "no"}:
        return
    log_path = resolve_path(args.failure_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "query": program.query,
        "prompt": program.prompt,
        "raw_output": program.raw_output,
        "il": program.il,
        "template": program.template,
        "regex": program.pattern,
        "replacement": program.replacement,
        "match_count": count,
        "matches_shown": matches,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"saved failure case: {log_path}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    target = resolve_path(args.path)
    if not target.exists():
        print(f"{ERR_FG}path not found:{RESET} {target}", file=sys.stderr)
        return 2
    if args.interactive or not args.query:
        return run_interactive(args, target)
    try:
        program = infer_program(args)
        print_program(program, args)
        count, _matches = scan(program, target, args)
    except Exception as exc:  # noqa: BLE001
        color = not getattr(args, "no_color", False)
        print(colorize(f"reggie error: {exc}", ERR_FG, enabled=color), file=sys.stderr)
        if getattr(args, "show_regex", False):
            raise
        return 1
    if count == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
