"""Run the same pinned formatters and checks locally, in hooks, and in CI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from check_complexity import check as check_complexity
from code_paragraphs import check as check_paragraphs
from quality_files import ROOT, formatter, source_files


def prettier() -> list[str]:
    script = ROOT / "viewer/node_modules/prettier/bin/prettier.cjs"

    if not script.is_file():
        raise RuntimeError(
            "Install viewer tools first: pnpm --dir viewer install --frozen-lockfile"
        )

    return ["node", str(script)]


def commands(kind: str, write: bool) -> list[list[str]]:
    if kind == "native":
        return [["clang-format", "-i"] if write else ["clang-format", "--dry-run", "--Werror"]]

    if kind == "python":
        if write:
            return [["ruff", "format"], ["ruff", "check", "--fix"], ["ruff", "format"]]

        return [
            ["ruff", "check"],
            ["ruff", "format", "--check"],
        ]

    if kind in {"web", "yaml"}:
        return [
            [
                *prettier(),
                "--write" if write else "--check",
                *(["--parser", "yaml"] if kind == "yaml" else []),
            ]
        ]

    if kind == "toml":
        return [["taplo", "fmt", *([] if write else ["--check"])]]

    if kind == "shell":
        return [["shfmt", "-i", "2", "-w" if write else "-d"]]

    if kind == "cmake":
        return [["cmake-format", "-i" if write else "--check"]]

    raise ValueError(f"unknown formatter: {kind}")


def run_formatters(paths: list[Path], write: bool) -> int:
    groups: dict[str, list[str]] = defaultdict(list)

    for path in paths:
        kind = formatter(path)

        if kind is not None:
            groups[kind].append(str(path))

    failed = False

    for kind, names in sorted(groups.items()):
        for command in commands(kind, write):
            if shutil.which(command[0]) is None:
                raise RuntimeError(
                    f"Missing {command[0]}; run with the locked quality dependency group"
                )

            # Bound command size on Windows as well as Unix. Always use argv, never a shell.
            for start in range(0, len(names), 50):
                result = subprocess.run([*command, *names[start : start + 50]], cwd=ROOT)
                failed |= result.returncode != 0

    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["format", "check", "complexity"])
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()
    paths = source_files(args.files)

    if args.command == "complexity":
        return check_complexity(paths)

    status = run_formatters(paths, write=args.command == "format")
    status |= check_paragraphs(paths, write=args.command == "format")

    if args.command == "check":
        status |= check_complexity(paths)

    return status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
