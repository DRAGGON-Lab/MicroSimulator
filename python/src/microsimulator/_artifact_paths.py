"""Shared path conventions for generated MicroSimulator artifacts."""

from __future__ import annotations

from pathlib import Path

_CHECKPOINT_SUFFIX = ".json"


def periodic_checkpoint_parts(output: Path) -> tuple[Path, str, str]:
    """Return the directory, stem, and suffix used by periodic checkpoints."""

    name = output.name
    stem = name.removesuffix(_CHECKPOINT_SUFFIX)
    return output.parent, stem, _CHECKPOINT_SUFFIX


def periodic_checkpoint_path(output: Path, step: int) -> Path:
    """Derive the path of the periodic checkpoint written at one step."""

    parent, stem, suffix = periodic_checkpoint_parts(output)
    return parent / f"{stem}.step-{step:08d}{suffix}"
