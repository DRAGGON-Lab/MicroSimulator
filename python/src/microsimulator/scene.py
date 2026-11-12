"""Immutable, versioned presentation snapshots for independent viewers.

The native core captures, validates, encodes, and decodes scene frames. Each
scene type is a read-only record compared by value; ``replace(**changes)``
returns a copy with the named fields substituted.
"""

from __future__ import annotations

from typing import Literal

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    MAX_SCENE_BYTES,
    MAX_SCENE_CHANNELS,
    SCENE_FORMAT,
    SCENE_VERSION,
    SceneBackend,
    SceneBoxConstraint,
    SceneCell,
    SceneConstraints,
    SceneCylinderConstraint,
    SceneError,
    SceneFrame,
    SceneGridBoundary,
    ScenePlaneConstraint,
    SceneSignalGrid,
    SceneSphereConstraint,
    capture_scene,
    dumps_scene,
    load_scene,
    parse_scene,
    save_scene,
)

type SceneBackendKind = Literal["cpu", "metal", "cuda"]
type SceneBoundaryKind = Literal["no_flux", "periodic", "fixed"]
type SceneRegionKind = Literal["outside", "inside"]

__all__ = [
    "MAX_SCENE_BYTES",
    "MAX_SCENE_CHANNELS",
    "SCENE_FORMAT",
    "SCENE_VERSION",
    "SceneBackend",
    "SceneBackendKind",
    "SceneBoundaryKind",
    "SceneBoxConstraint",
    "SceneCell",
    "SceneConstraints",
    "SceneCylinderConstraint",
    "SceneError",
    "SceneFrame",
    "SceneGridBoundary",
    "ScenePlaneConstraint",
    "SceneRegionKind",
    "SceneSignalGrid",
    "SceneSphereConstraint",
    "capture_scene",
    "dumps_scene",
    "load_scene",
    "parse_scene",
    "save_scene",
]
