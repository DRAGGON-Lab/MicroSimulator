"""Shared path conventions for generated MicroSimulator artifacts."""

from __future__ import annotations

from pathlib import Path

_CHECKPOINT_SUFFIX = ".json"
_LEGACY_CHECKPOINT_SUFFIX = ".cm2.json"


def periodic_checkpoint_parts(output: Path) -> tuple[Path, str, str]:
    """Return the directory, stem, and suffix used by periodic checkpoints."""

    name = output.name
    suffix = (
        _LEGACY_CHECKPOINT_SUFFIX
        if name.endswith(_LEGACY_CHECKPOINT_SUFFIX)
        else _CHECKPOINT_SUFFIX
    )
    stem = name[: -len(suffix)] if name.endswith(suffix) else name
    return output.parent, stem, suffix


def periodic_checkpoint_path(output: Path, step: int) -> Path:
    """Derive one periodic checkpoint path while preserving legacy suffixes."""

    parent, stem, suffix = periodic_checkpoint_parts(output)
    return parent / f"{stem}.step-{step:08d}{suffix}"
