"""Versioned, non-executable MicroSimulator checkpoints.

The native core encodes, validates, and atomically writes checkpoint files.
This module pairs that state with the Python-owned controller and provenance
data stored beside it.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    MAX_CHECKPOINT_BYTES,
    BackendKind,
    CheckpointError,
    Simulation,
    _load_checkpoint,
    _save_checkpoint,
)
from .channels import UNNAMED_CHANNELS, ChannelMetadata

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]

__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "MAX_CHECKPOINT_BYTES",
    "CheckpointBundle",
    "CheckpointError",
    "CheckpointSourceBackend",
    "JSONScalar",
    "JSONValue",
    "load_checkpoint",
    "load_checkpoint_bundle",
    "save_checkpoint",
]


@dataclass(frozen=True, slots=True)
class CheckpointSourceBackend:
    """Backend identity recorded by the checkpoint producer."""

    kind: str
    name: str
    device: str
    device_index: int
    native: bool


@dataclass(frozen=True, slots=True)
class CheckpointBundle:
    """Validated native state plus optional data-only controller state."""

    simulation: Simulation
    controller: JSONValue
    provenance: dict[str, JSONValue]
    schema_version: int
    source_backend: CheckpointSourceBackend
    channel_metadata: ChannelMetadata = UNNAMED_CHANNELS


def save_checkpoint(
    simulation: Simulation,
    path: str | os.PathLike[str],
    *,
    provenance: Mapping[str, JSONValue] | None = None,
    controller: JSONValue = None,
    channel_metadata: ChannelMetadata = UNNAMED_CHANNELS,
) -> None:
    """Atomically save a complete simulation checkpoint as validated JSON."""

    _save_checkpoint(
        simulation,
        path,
        dict(provenance) if provenance is not None else {},
        controller,
        channel_metadata,
    )


def load_checkpoint_bundle(
    path: str | os.PathLike[str],
    *,
    backend: BackendKind = BackendKind.CPU,
    device_index: int = 0,
) -> CheckpointBundle:
    """Load native and optional controller state without evaluating executable content."""

    document = _load_checkpoint(path)
    source = document.source_backend
    return CheckpointBundle(
        simulation=Simulation(backend, document.simulation, device_index),
        controller=document.controller,
        provenance=document.provenance,
        schema_version=CHECKPOINT_VERSION,
        source_backend=CheckpointSourceBackend(
            kind=source.kind.name.lower(),
            name=source.name,
            device=source.device,
            device_index=source.device_index,
            native=source.native,
        ),
        channel_metadata=document.channel_metadata,
    )


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    backend: BackendKind = BackendKind.CPU,
    device_index: int = 0,
) -> Simulation:
    """Load a native checkpoint, rejecting controller state that would be discarded."""

    bundle = load_checkpoint_bundle(path, backend=backend, device_index=device_index)
    if bundle.controller is not None:
        raise CheckpointError(
            "checkpoint contains controller state; load it with load_checkpoint_bundle"
        )
    if any(
        label is not None
        for group in (bundle.channel_metadata.species, bundle.channel_metadata.signals)
        for label in (group or ())
    ):
        raise CheckpointError(
            "checkpoint contains channel metadata; load it with load_checkpoint_bundle"
        )
    return bundle.simulation
