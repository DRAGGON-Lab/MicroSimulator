"""Separate statement groups without changing tokens or guessing numerical intent."""

from __future__ import annotations

import difflib
from functools import cache
from pathlib import Path

import tree_sitter_cpp
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_typescript
from quality_files import NATIVE, ROOT
from tree_sitter import Language, Node, Parser

BLOCKS = {"compound_statement", "statement_block", "block"}
CONTROL = {
    "if_statement",
    "for_statement",
    "for_range_loop",
    "for_in_statement",
    "while_statement",
    "do_statement",
    "switch_statement",
    "switch_expression",
    "try_statement",
    "with_statement",
    "match_statement",
}
DEFINITIONS = {"function_definition", "class_definition", "function_declaration"}


@cache
def parser_for(suffix: str) -> Parser | None:
    if suffix in NATIVE:
        language = tree_sitter_cpp.language()
    elif suffix in {".py", ".pyi"}:
        language = tree_sitter_python.language()
    elif suffix in {".ts", ".tsx"}:
        language = (
            tree_sitter_typescript.language_tsx()
            if suffix == ".tsx"
            else tree_sitter_typescript.language_typescript()
        )
    elif suffix in {".js", ".mjs", ".cjs"}:
        language = tree_sitter_javascript.language()
    else:
        return None

    return Parser(Language(language))


def needs_paragraph(previous: Node, current: Node) -> bool:
    if previous.type == "ERROR" or current.type == "ERROR":
        return False

    if previous.type in DEFINITIONS or current.type in DEFINITIONS:
        return False

    return previous.type in CONTROL or current.type in CONTROL | {"return_statement"}


def block_boundaries(block: Node, lines: list[str]) -> set[int]:
    boundaries = set()
    previous = None
    comments: list[Node] = []

    for node in block.named_children:
        if node.type == "comment":
            comments.append(node)
            continue

        if previous is not None and needs_paragraph(previous, node):
            leading = [c for c in comments if c.start_point.row > previous.end_point.row]
            row = (leading[0] if leading else node).start_point.row

            # A newline may only be inserted between complete physical lines.
            # Never split one-line bodies, trailing comments, or multiline tokens.
            if row > previous.end_point.row and row > 0 and lines[row - 1].strip():
                boundaries.add(row)

        previous = node
        comments = []

    return boundaries


def format_code(path: Path, code: str) -> str:
    parser = parser_for(path.suffix)

    if parser is None:
        return code

    lines = code.splitlines(keepends=True)
    tree = parser.parse(code.encode("utf-8"))
    pending = [tree.root_node]
    boundaries = set()

    while pending:
        node = pending.pop()

        # Unsupported Objective-C/CUDA syntax is opaque. It must not expose
        # text inside an error recovery node as candidate statements.
        if node.type == "ERROR" or node.type.startswith("preproc_"):
            continue

        if node.type in BLOCKS:
            boundaries.update(block_boundaries(node, lines))

        pending.extend(node.named_children)

    return "".join(("\n" if i in boundaries else "") + line for i, line in enumerate(lines))


def check(paths: list[Path], *, write: bool) -> int:
    changed = 0

    for path in paths:
        if parser_for(path.suffix) is None:
            continue

        original = (ROOT / path).read_text(encoding="utf-8")
        formatted = format_code(path, original)

        if formatted == original:
            continue

        changed += 1

        if write:
            (ROOT / path).write_text(formatted, encoding="utf-8")
        else:
            print(
                "".join(
                    difflib.unified_diff(
                        original.splitlines(keepends=True),
                        formatted.splitlines(keepends=True),
                        fromfile=str(path),
                        tofile=f"{path} (logical spacing)",
                    )
                ),
                end="",
            )

    print(f"Code paragraphs: {changed} files {'updated' if write else 'need formatting'}")

    return int(changed > 0 and not write)
