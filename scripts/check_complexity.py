"""Structural limits shared by native, Python, and viewer source."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import lizard
from quality_files import NATIVE, ROOT, source_files

MAX_LINES = 100
MAX_COMPLEXITY = 15


def analysis_name(path: Path) -> str | None:
    if path.suffix in NATIVE:
        return str(path.with_suffix(".mm" if path.suffix == ".mm" else ".cpp"))

    if path.suffix in {".py", ".pyi"}:
        return str(path.with_suffix(".py"))

    if path.suffix in {".ts", ".tsx"}:
        return str(path.with_suffix(".ts"))

    if path.suffix in {".js", ".mjs", ".cjs"}:
        return str(path.with_suffix(".js"))

    return None


def violations(path: Path, code: str) -> list[str]:
    name = analysis_name(path)

    if name is None:
        return []

    findings = []

    if re.search(r"(?m)^\s*(?://|#|/\*|\*)\s*#?\s*lizard\s+forgiv", code):
        findings.append(f"{path}: complexity suppressions are not permitted")

    result = lizard.analyze_file.analyze_source_code(name, code)

    for function in result.function_list:
        if function.nloc > MAX_LINES or function.cyclomatic_complexity > MAX_COMPLEXITY:
            findings.append(
                f"{path}:{function.start_line}: {function.name}: "
                f"{function.nloc}/{MAX_LINES} code lines, "
                f"complexity {function.cyclomatic_complexity}/{MAX_COMPLEXITY}"
            )

    return findings


def check(paths: list[Path]) -> int:
    findings = []
    count = 0

    for path in paths:
        if analysis_name(path) is not None:
            count += 1
            findings.extend(violations(path, (ROOT / path).read_text(encoding="utf-8")))

    for finding in findings:
        print(finding)

    print(f"Complexity: {len(findings)} violations in {count} source files")

    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(check(source_files(sys.argv[1:])))
