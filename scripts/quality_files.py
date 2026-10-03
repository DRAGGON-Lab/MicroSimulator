"""Shared source discovery for formatting and structural checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".cu", ".cuh", ".mm", ".metal"}
PYTHON = {".py", ".pyi"}
WEB = {".ts", ".tsx", ".js", ".mjs", ".cjs", ".html", ".css", ".md", ".json", ".yml", ".yaml"}
YAML_NAMES = {".clang-format", ".clang-tidy"}
EXCLUDED_PARTS = {".git", ".venv", "node_modules", "build", "dist", "__pycache__"}
LOCKFILES = {"uv.lock", "pnpm-lock.yaml"}


def maintained(path: Path) -> bool:
    return not (set(path.parts) & EXCLUDED_PARTS) and path.name not in LOCKFILES


def source_files(names: list[str] | None = None) -> list[Path]:
    if not names:
        output = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
        )
        names = output.decode().rstrip("\0").split("\0")

    paths = set()
    discovered = None

    for name in names:
        path = Path(name)

        if path.is_absolute():
            path = path.relative_to(ROOT)

        if ".." in path.parts:
            raise ValueError(f"path must be inside the repository: {path}")

        if maintained(path) and (ROOT / path).is_dir():
            if discovered is None:
                discovered = source_files()

            paths.update(candidate for candidate in discovered if path in candidate.parents)

        if maintained(path) and (ROOT / path).is_file():
            paths.add(path)

    return sorted(paths)


def formatter(path: Path) -> str | None:
    if path.suffix in NATIVE:
        return "native"

    if path.suffix in PYTHON:
        return "python"

    if path.name in YAML_NAMES:
        return "yaml"

    if path.suffix in WEB:
        return "web"

    if path.suffix == ".toml":
        return "toml"

    if path.suffix == ".sh":
        return "shell"

    if path.suffix == ".cmake" or path.name == "CMakeLists.txt":
        return "cmake"

    return None
