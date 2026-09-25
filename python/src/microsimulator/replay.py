"""Data-only replay export from an explicitly ordered checkpoint sequence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import rfc8785

from ._core import BackendKind  # pyright: ignore[reportMissingModuleSource]
from .checkpoint import MAX_CHECKPOINT_BYTES, CheckpointError, JSONValue, load_checkpoint_bundle
from .scene import MAX_SCENE_BYTES, SceneBackend, SceneBackendKind, capture_scene, dumps_scene

REPLAY_FORMAT = "microsimulator-replay"
REPLAY_VERSION = 1
MAX_REPLAY_FRAMES = 100_000
MAX_REPLAY_MANIFEST_BYTES = 16 * 1024 * 1024


class ReplayExportError(ValueError):
    """Raised when an ordered recording cannot be exported without loss."""


@dataclass(frozen=True, slots=True)
class ReplayExportSummary:
    output: Path
    frame_count: int


def export_replay(
    checkpoints: Sequence[str | os.PathLike[str]], output: str | os.PathLike[str]
) -> ReplayExportSummary:
    """Export exact snapshots on CPU, never loading model source or stepping biology.

    Input order is authoritative. Equal times remain distinct frames; decreasing
    time is rejected. The destination must not exist, including an empty folder.
    """

    if not 1 <= len(checkpoints) <= MAX_REPLAY_FRAMES:
        raise ReplayExportError(f"expected 1 to {MAX_REPLAY_FRAMES} ordered checkpoints")
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise ReplayExportError(f"output already exists: {destination}")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ReplayExportError(
            f"could not prepare replay destination {destination}: {error}"
        ) from error
    entries: list[JSONValue] = []
    previous_time = -1.0
    export_backend: JSONValue = None
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.", dir=destination.parent
        ) as temporary:
            stage = Path(temporary)
            frames_dir = stage / "frames"
            frames_dir.mkdir()
            snapshot = stage / ".checkpoint.json"
            for ordinal, checkpoint_path in enumerate(checkpoints):
                try:
                    # Parse exactly the bytes whose digest is recorded, even if a
                    # running producer replaces the original checkpoint concurrently.
                    digest = hashlib.sha256()
                    size = 0
                    with Path(checkpoint_path).open("rb") as source, snapshot.open("wb") as target:
                        while chunk := source.read(1024 * 1024):
                            size += len(chunk)
                            if size > MAX_CHECKPOINT_BYTES:
                                raise ReplayExportError("checkpoint exceeds its byte limit")
                            target.write(chunk)
                            digest.update(chunk)
                    bundle = load_checkpoint_bundle(snapshot, backend=BackendKind.CPU)
                    captured = capture_scene(
                        bundle.simulation, channel_metadata=bundle.channel_metadata
                    )
                    if captured.time < previous_time:
                        raise ReplayExportError(
                            f"time {captured.time} precedes previous frame time {previous_time}"
                        )
                    previous_time = captured.time
                    if export_backend is None:
                        export_backend = cast(JSONValue, asdict(captured.backend))
                    source_backend = bundle.source_backend
                    # The displayed scene describes its source run, not the CPU
                    # used solely to deserialize portable state during export.
                    frame = replace(
                        captured,
                        backend=SceneBackend(
                            kind=cast(SceneBackendKind, source_backend.kind),
                            name=source_backend.name,
                            device=source_backend.device,
                            device_index=source_backend.device_index,
                            native=source_backend.native,
                        ),
                    )
                    encoded = dumps_scene(frame).encode("utf-8")
                    if len(encoded) > MAX_SCENE_BYTES:
                        raise ReplayExportError("scene exceeds its byte limit")
                    relative = f"frames/{ordinal:08d}.scene.json"
                    (stage / relative).write_bytes(encoded)
                    entries.append(
                        {
                            "ordinal": ordinal,
                            "time": frame.time,
                            "file": relative,
                            "bytes": len(encoded),
                            "sha256": hashlib.sha256(encoded).hexdigest(),
                            "checkpoint_sha256": digest.hexdigest(),
                            "source_backend": cast(JSONValue, asdict(source_backend)),
                        }
                    )
                except (OSError, ValueError, RuntimeError) as error:
                    raise ReplayExportError(
                        f"frame {ordinal} ({checkpoint_path}): {error}"
                    ) from error
            snapshot.unlink()
            recording: dict[str, JSONValue] = {"export_backend": export_backend, "frames": entries}
            document: dict[str, JSONValue] = {
                "format": REPLAY_FORMAT,
                "version": REPLAY_VERSION,
                "integrity": {
                    "algorithm": "sha256",
                    "recording": hashlib.sha256(rfc8785.dumps(recording)).hexdigest(),
                },
                "recording": recording,
            }
            encoded_manifest = (
                json.dumps(document, allow_nan=False, ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")
            if len(encoded_manifest) > MAX_REPLAY_MANIFEST_BYTES:
                raise ReplayExportError("replay manifest exceeds the 16 MiB limit")
            (stage / "manifest.json").write_bytes(encoded_manifest)
            if destination.exists() or destination.is_symlink():
                raise ReplayExportError(f"output already exists: {destination}")
            stage.rename(destination)
    except (OSError, CheckpointError) as error:
        raise ReplayExportError(f"could not export replay to {destination}: {error}") from error
    return ReplayExportSummary(destination, len(entries))
