"""Immutable, versioned presentation snapshots for independent viewers."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, NoReturn, cast

import rfc8785

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    GridBoundary,
    GridBoundaryKind,
    Simulation,
    Vec3,
)
from .checkpoint import JSONValue

SCENE_FORMAT = "microsimulator-scene"
SCENE_VERSION = 1
MAX_SCENE_BYTES = 1 << 30

_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_FLOAT32_MAX = 3.4028234663852886e38

type SceneBackendKind = Literal["cpu", "metal", "cuda"]
type SceneBoundaryKind = Literal["no_flux", "periodic", "fixed"]

_BACKEND_NAMES: dict[BackendKind, SceneBackendKind] = {
    BackendKind.CPU: "cpu",
    BackendKind.METAL: "metal",
    BackendKind.CUDA: "cuda",
}
_BOUNDARY_NAMES: dict[GridBoundaryKind, SceneBoundaryKind] = {
    GridBoundaryKind.NO_FLUX: "no_flux",
    GridBoundaryKind.PERIODIC: "periodic",
    GridBoundaryKind.FIXED: "fixed",
}
_BACKEND_KINDS = frozenset(_BACKEND_NAMES.values())
_BOUNDARY_KINDS = frozenset(_BOUNDARY_NAMES.values())


class SceneError(ValueError):
    """Raised when a scene frame cannot be safely encoded or decoded."""


@dataclass(frozen=True, slots=True)
class SceneBackend:
    kind: SceneBackendKind
    name: str
    device: str
    device_index: int
    native: bool


@dataclass(frozen=True, slots=True)
class SceneCell:
    id: int
    parent_id: int | None
    slot: int
    position: tuple[float, float, float]
    direction: tuple[float, float, float]
    length: float
    radius: float
    growth_rate: float
    cell_type: int
    fixed: bool
    species: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SceneGridBoundary:
    kind: SceneBoundaryKind
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SceneSignalGrid:
    signal_count: int
    shape: tuple[int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    x_lower: SceneGridBoundary
    x_upper: SceneGridBoundary
    y_lower: SceneGridBoundary
    y_upper: SceneGridBoundary
    z_lower: SceneGridBoundary
    z_upper: SceneGridBoundary
    levels: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SceneFrame:
    time: float
    backend: SceneBackend
    species_count: int
    cells: tuple[SceneCell, ...]
    signal_grid: SceneSignalGrid | None


def _installed_version() -> str:
    try:
        return version("microsimulator")
    except PackageNotFoundError:
        return "0+unknown"


def _tuple3(value: Vec3) -> tuple[float, float, float]:
    return (value.x, value.y, value.z)


def _capture_boundary(boundary: GridBoundary) -> SceneGridBoundary:
    return SceneGridBoundary(
        kind=_BOUNDARY_NAMES[boundary.kind],
        values=tuple(boundary.values),
    )


def capture_scene(simulation: Simulation) -> SceneFrame:
    """Capture a complete immutable presentation frame after a simulation step."""

    checkpoint = simulation._checkpoint()
    checkpoint.validate()
    backend = simulation.backend_info
    lineage = {entry.child: entry.parent for entry in checkpoint.world.lineage}
    cells = tuple(
        SceneCell(
            id=cell.id,
            parent_id=lineage.get(cell.id),
            slot=cell.slot,
            position=_tuple3(cell.position),
            direction=_tuple3(cell.direction),
            length=cell.length,
            radius=cell.radius,
            growth_rate=cell.growth_rate,
            cell_type=cell.cell_type,
            fixed=cell.fixed,
            species=tuple(cell.species),
        )
        for cell in checkpoint.world.cells
    )
    signal_grid = None
    if checkpoint.signal_grid is not None:
        grid = checkpoint.signal_grid
        spec = grid.spec
        signal_grid = SceneSignalGrid(
            signal_count=spec.signal_count,
            shape=(spec.shape.x, spec.shape.y, spec.shape.z),
            origin=_tuple3(spec.origin),
            spacing=_tuple3(spec.spacing),
            x_lower=_capture_boundary(spec.x_lower),
            x_upper=_capture_boundary(spec.x_upper),
            y_lower=_capture_boundary(spec.y_lower),
            y_upper=_capture_boundary(spec.y_upper),
            z_lower=_capture_boundary(spec.z_lower),
            z_upper=_capture_boundary(spec.z_upper),
            levels=tuple(grid.levels),
        )
    frame = SceneFrame(
        time=checkpoint.time,
        backend=SceneBackend(
            kind=_BACKEND_NAMES[backend.kind],
            name=backend.name,
            device=backend.device,
            device_index=backend.device_index,
            native=backend.native,
        ),
        species_count=checkpoint.world.species_count,
        cells=cells,
        signal_grid=signal_grid,
    )
    _validate_frame(frame)
    return frame


def _boundary_to_json(boundary: SceneGridBoundary) -> dict[str, JSONValue]:
    return {"kind": boundary.kind, "values": list(boundary.values)}


def _frame_to_json(frame: SceneFrame) -> dict[str, JSONValue]:
    cells: list[JSONValue] = [
        {
            "id": str(cell.id),
            "parent_id": str(cell.parent_id) if cell.parent_id is not None else None,
            "slot": cell.slot,
            "position": list(cell.position),
            "direction": list(cell.direction),
            "length": cell.length,
            "radius": cell.radius,
            "growth_rate": cell.growth_rate,
            "cell_type": cell.cell_type,
            "fixed": cell.fixed,
            "species": list(cell.species),
        }
        for cell in frame.cells
    ]
    grid: JSONValue = None
    if frame.signal_grid is not None:
        value = frame.signal_grid
        grid = {
            "signal_count": value.signal_count,
            "shape": list(value.shape),
            "origin": list(value.origin),
            "spacing": list(value.spacing),
            "boundaries": {
                "x_lower": _boundary_to_json(value.x_lower),
                "x_upper": _boundary_to_json(value.x_upper),
                "y_lower": _boundary_to_json(value.y_lower),
                "y_upper": _boundary_to_json(value.y_upper),
                "z_lower": _boundary_to_json(value.z_lower),
                "z_upper": _boundary_to_json(value.z_upper),
            },
            "levels": list(value.levels),
        }
    return {
        "time": frame.time,
        "backend": {
            "kind": frame.backend.kind,
            "name": frame.backend.name,
            "device": frame.backend.device,
            "device_index": frame.backend.device_index,
            "native": frame.backend.native,
        },
        "species_count": frame.species_count,
        "cells": cells,
        "signal_grid": grid,
    }


def _canonical_json(value: JSONValue) -> bytes:
    try:
        return rfc8785.dumps(value)
    except rfc8785.CanonicalizationError as error:
        raise SceneError(f"scene cannot be canonicalized: {error}") from error


def dumps_scene(frame: SceneFrame) -> str:
    """Encode a validated scene frame as deterministic, human-readable JSON."""

    _validate_frame(frame)
    payload = _frame_to_json(frame)
    document: dict[str, JSONValue] = {
        "format": SCENE_FORMAT,
        "version": SCENE_VERSION,
        "producer": {"name": "microsimulator", "version": _installed_version()},
        "integrity": {
            "algorithm": "sha256",
            "frame": hashlib.sha256(_canonical_json(payload)).hexdigest(),
        },
        "frame": payload,
    }
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def save_scene(frame: SceneFrame, path: str | os.PathLike[str]) -> None:
    """Atomically save an immutable scene frame."""

    encoded = dumps_scene(frame).encode("utf-8")
    if len(encoded) > MAX_SCENE_BYTES:
        raise SceneError(f"scene exceeds the {MAX_SCENE_BYTES}-byte limit")
    destination = Path(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    except OSError as error:
        raise SceneError(f"could not write scene {destination}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fail(path: str, message: str) -> NoReturn:
    raise SceneError(f"{path}: {message}")


def _reject_constant(value: str) -> NoReturn:
    raise SceneError(f"scene contains non-finite JSON number {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SceneError(f"scene contains duplicate key {key!r}")
        result[key] = value
    return result


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        _fail(path, "expected string object keys")
    return cast(dict[str, object], mapping)


def _array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    return cast(list[object], value)


def _keys(value: dict[str, object], path: str, required: set[str]) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required
    if missing:
        _fail(path, f"missing keys {sorted(missing)}")
    if unknown:
        _fail(path, f"unknown keys {sorted(unknown)}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "expected a boolean")
    return value


def _integer(value: object, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected an integer")
    if value < minimum or value > maximum:
        _fail(path, f"integer is outside [{minimum}, {maximum}]")
    return value


def _identifier(value: object, path: str) -> int:
    encoded = _string(value, path)
    if not encoded.isascii() or not encoded.isdecimal() or encoded.startswith("0"):
        _fail(path, "expected a canonical positive decimal uint64 string")
    result = int(encoded)
    if result <= 0 or result > _UINT64_MAX:
        _fail(path, "identifier is outside the positive uint64 range")
    return result


def _number(value: object, path: str, *, float32: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(path, "expected a number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _fail(path, "number is outside the finite float64 range")
    if not math.isfinite(result):
        _fail(path, "number must be finite")
    if float32 and abs(result) > _FLOAT32_MAX:
        _fail(path, "number is outside the finite float32 range")
    return result


def _tuple3_from_json(value: object, path: str) -> tuple[float, float, float]:
    items = _array(value, path)
    if len(items) != 3:
        _fail(path, "expected exactly three values")
    return (
        _number(items[0], f"{path}[0]", float32=True),
        _number(items[1], f"{path}[1]", float32=True),
        _number(items[2], f"{path}[2]", float32=True),
    )


def _float_tuple(value: object, path: str) -> tuple[float, ...]:
    return tuple(
        _number(item, f"{path}[{index}]", float32=True)
        for index, item in enumerate(_array(value, path))
    )


def _boundary(value: object, path: str, signal_count: int) -> SceneGridBoundary:
    data = _object(value, path)
    _keys(data, path, {"kind", "values"})
    kind_value = _string(data["kind"], f"{path}.kind")
    if kind_value not in _BOUNDARY_KINDS:
        _fail(f"{path}.kind", f"unknown boundary kind {kind_value!r}")
    kind = kind_value
    values = _float_tuple(data["values"], f"{path}.values")
    expected = signal_count if kind == "fixed" else 0
    if len(values) != expected:
        _fail(f"{path}.values", f"expected {expected} values for {kind} boundary")
    return SceneGridBoundary(kind=kind, values=values)


def _backend(value: object, path: str) -> SceneBackend:
    data = _object(value, path)
    _keys(data, path, {"kind", "name", "device", "device_index", "native"})
    kind_value = _string(data["kind"], f"{path}.kind")
    if kind_value not in _BACKEND_KINDS:
        _fail(f"{path}.kind", f"unknown backend kind {kind_value!r}")
    name = _string(data["name"], f"{path}.name")
    device = _string(data["device"], f"{path}.device")
    if not name:
        _fail(f"{path}.name", "must not be empty")
    if not device:
        _fail(f"{path}.device", "must not be empty")
    return SceneBackend(
        kind=kind_value,
        name=name,
        device=device,
        device_index=_integer(data["device_index"], f"{path}.device_index", 0, _UINT32_MAX),
        native=_boolean(data["native"], f"{path}.native"),
    )


def _cell(value: object, path: str, species_count: int) -> SceneCell:
    data = _object(value, path)
    _keys(
        data,
        path,
        {
            "id",
            "parent_id",
            "slot",
            "position",
            "direction",
            "length",
            "radius",
            "growth_rate",
            "cell_type",
            "fixed",
            "species",
        },
    )
    identifier = _identifier(data["id"], f"{path}.id")
    parent_value = data["parent_id"]
    parent_id = None if parent_value is None else _identifier(parent_value, f"{path}.parent_id")
    species = _float_tuple(data["species"], f"{path}.species")
    if len(species) != species_count:
        _fail(f"{path}.species", f"expected {species_count} values")
    return SceneCell(
        id=identifier,
        parent_id=parent_id,
        slot=_integer(data["slot"], f"{path}.slot", 0, _UINT32_MAX - 1),
        position=_tuple3_from_json(data["position"], f"{path}.position"),
        direction=_tuple3_from_json(data["direction"], f"{path}.direction"),
        length=_number(data["length"], f"{path}.length", float32=True),
        radius=_number(data["radius"], f"{path}.radius", float32=True),
        growth_rate=_number(data["growth_rate"], f"{path}.growth_rate", float32=True),
        cell_type=_integer(data["cell_type"], f"{path}.cell_type", _INT32_MIN, _INT32_MAX),
        fixed=_boolean(data["fixed"], f"{path}.fixed"),
        species=species,
    )


def _signal_grid(value: object, path: str) -> SceneSignalGrid | None:
    if value is None:
        return None
    data = _object(value, path)
    _keys(
        data,
        path,
        {"signal_count", "shape", "origin", "spacing", "boundaries", "levels"},
    )
    signal_count = _integer(data["signal_count"], f"{path}.signal_count", 1, _UINT32_MAX)
    shape_values = _array(data["shape"], f"{path}.shape")
    if len(shape_values) != 3:
        _fail(f"{path}.shape", "expected exactly three dimensions")
    shape = cast(
        tuple[int, int, int],
        tuple(
            _integer(item, f"{path}.shape[{index}]", 1, _UINT32_MAX)
            for index, item in enumerate(shape_values)
        ),
    )
    boundaries = _object(data["boundaries"], f"{path}.boundaries")
    boundary_names = {"x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper"}
    _keys(boundaries, f"{path}.boundaries", boundary_names)
    levels = _float_tuple(data["levels"], f"{path}.levels")
    expected_levels = signal_count * shape[0] * shape[1] * shape[2]
    if len(levels) != expected_levels:
        _fail(f"{path}.levels", f"expected {expected_levels} values")
    return SceneSignalGrid(
        signal_count=signal_count,
        shape=shape,
        origin=_tuple3_from_json(data["origin"], f"{path}.origin"),
        spacing=_tuple3_from_json(data["spacing"], f"{path}.spacing"),
        x_lower=_boundary(boundaries["x_lower"], f"{path}.boundaries.x_lower", signal_count),
        x_upper=_boundary(boundaries["x_upper"], f"{path}.boundaries.x_upper", signal_count),
        y_lower=_boundary(boundaries["y_lower"], f"{path}.boundaries.y_lower", signal_count),
        y_upper=_boundary(boundaries["y_upper"], f"{path}.boundaries.y_upper", signal_count),
        z_lower=_boundary(boundaries["z_lower"], f"{path}.boundaries.z_lower", signal_count),
        z_upper=_boundary(boundaries["z_upper"], f"{path}.boundaries.z_upper", signal_count),
        levels=levels,
    )


def _frame(value: object, path: str) -> SceneFrame:
    data = _object(value, path)
    _keys(data, path, {"time", "backend", "species_count", "cells", "signal_grid"})
    species_count = _integer(data["species_count"], f"{path}.species_count", 0, _UINT32_MAX)
    frame = SceneFrame(
        time=_number(data["time"], f"{path}.time"),
        backend=_backend(data["backend"], f"{path}.backend"),
        species_count=species_count,
        cells=tuple(
            _cell(item, f"{path}.cells[{index}]", species_count)
            for index, item in enumerate(_array(data["cells"], f"{path}.cells"))
        ),
        signal_grid=_signal_grid(data["signal_grid"], f"{path}.signal_grid"),
    )
    _validate_frame(frame)
    return frame


def _validate_boundary(boundary: SceneGridBoundary, signal_count: int, path: str) -> None:
    if boundary.kind not in _BOUNDARY_KINDS:
        _fail(f"{path}.kind", f"unknown boundary kind {boundary.kind!r}")
    expected = signal_count if boundary.kind == "fixed" else 0
    if len(boundary.values) != expected:
        _fail(f"{path}.values", f"expected {expected} values for {boundary.kind} boundary")
    for index, value in enumerate(boundary.values):
        _number(value, f"{path}.values[{index}]", float32=True)


def _validate_frame(frame: SceneFrame) -> None:
    _number(frame.time, "$.frame.time")
    if frame.time < 0.0:
        _fail("$.frame.time", "must be non-negative")
    if frame.backend.kind not in _BACKEND_KINDS:
        _fail("$.frame.backend.kind", f"unknown backend kind {frame.backend.kind!r}")
    if not frame.backend.name or not frame.backend.device:
        _fail("$.frame.backend", "name and device must not be empty")
    _integer(frame.backend.device_index, "$.frame.backend.device_index", 0, _UINT32_MAX)
    _boolean(frame.backend.native, "$.frame.backend.native")
    _integer(frame.species_count, "$.frame.species_count", 0, _UINT32_MAX)

    identifiers: set[int] = set()
    for index, cell in enumerate(frame.cells):
        path = f"$.frame.cells[{index}]"
        if cell.slot != index:
            _fail(f"{path}.slot", "cells must be compact and ordered by slot")
        _integer(cell.id, f"{path}.id", 1, _UINT64_MAX)
        if cell.id in identifiers:
            _fail(f"{path}.id", "duplicate cell identifier")
        identifiers.add(cell.id)
        if cell.parent_id is not None:
            _integer(cell.parent_id, f"{path}.parent_id", 1, _UINT64_MAX)
            if cell.parent_id >= cell.id:
                _fail(f"{path}.parent_id", "must precede the child identifier")
        for vector_name, vector in (("position", cell.position), ("direction", cell.direction)):
            if len(vector) != 3:
                _fail(f"{path}.{vector_name}", "expected exactly three values")
            for component, value in enumerate(vector):
                _number(value, f"{path}.{vector_name}[{component}]", float32=True)
        direction_norm = math.sqrt(sum(value * value for value in cell.direction))
        if abs(direction_norm - 1.0) > 1.0e-5:
            _fail(f"{path}.direction", "must be normalized")
        length = _number(cell.length, f"{path}.length", float32=True)
        radius = _number(cell.radius, f"{path}.radius", float32=True)
        if length < 0.0:
            _fail(f"{path}.length", "must be non-negative")
        if radius <= 0.0:
            _fail(f"{path}.radius", "must be positive")
        _number(cell.growth_rate, f"{path}.growth_rate", float32=True)
        _integer(cell.cell_type, f"{path}.cell_type", _INT32_MIN, _INT32_MAX)
        _boolean(cell.fixed, f"{path}.fixed")
        if len(cell.species) != frame.species_count:
            _fail(f"{path}.species", f"expected {frame.species_count} values")
        for species_index, level in enumerate(cell.species):
            _number(level, f"{path}.species[{species_index}]", float32=True)

    grid = frame.signal_grid
    if grid is None:
        return
    _integer(grid.signal_count, "$.frame.signal_grid.signal_count", 1, _UINT32_MAX)
    if len(grid.shape) != 3:
        _fail("$.frame.signal_grid.shape", "expected exactly three dimensions")
    for index, dimension in enumerate(grid.shape):
        _integer(dimension, f"$.frame.signal_grid.shape[{index}]", 1, _UINT32_MAX)
    for vector_name, vector in (("origin", grid.origin), ("spacing", grid.spacing)):
        if len(vector) != 3:
            _fail(f"$.frame.signal_grid.{vector_name}", "expected exactly three values")
        for component, value in enumerate(vector):
            number = _number(
                value,
                f"$.frame.signal_grid.{vector_name}[{component}]",
                float32=True,
            )
            if vector_name == "spacing" and number <= 0.0:
                _fail(f"$.frame.signal_grid.{vector_name}[{component}]", "must be positive")
    for name in ("x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper"):
        _validate_boundary(
            cast(SceneGridBoundary, getattr(grid, name)),
            grid.signal_count,
            f"$.frame.signal_grid.boundaries.{name}",
        )
    expected_levels = grid.signal_count * grid.shape[0] * grid.shape[1] * grid.shape[2]
    if len(grid.levels) != expected_levels:
        _fail("$.frame.signal_grid.levels", f"expected {expected_levels} values")
    for index, level in enumerate(grid.levels):
        _number(level, f"$.frame.signal_grid.levels[{index}]", float32=True)


def parse_scene(source: str | bytes) -> SceneFrame:
    """Decode a scene document without importing models or executable state."""

    if not source:
        raise SceneError("scene is empty")
    if len(source) > MAX_SCENE_BYTES:
        raise SceneError(f"scene exceeds the {MAX_SCENE_BYTES}-byte limit")
    try:
        decoded = json.loads(
            source,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except SceneError:
        raise
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise SceneError(f"scene is not valid UTF-8 JSON: {error}") from error

    root = _object(cast(object, decoded), "$")
    _keys(root, "$", {"format", "version", "producer", "integrity", "frame"})
    if _string(root["format"], "$.format") not in (SCENE_FORMAT, "cellmodeller2-scene"):
        _fail("$.format", "not a MicroSimulator scene")
    schema_version = _integer(root["version"], "$.version", 0, _UINT32_MAX)
    if schema_version != SCENE_VERSION:
        _fail("$.version", f"unsupported scene version {schema_version}")
    producer = _object(root["producer"], "$.producer")
    _keys(producer, "$.producer", {"name", "version"})
    _string(producer["name"], "$.producer.name")
    _string(producer["version"], "$.producer.version")
    integrity = _object(root["integrity"], "$.integrity")
    _keys(integrity, "$.integrity", {"algorithm", "frame"})
    if _string(integrity["algorithm"], "$.integrity.algorithm") != "sha256":
        _fail("$.integrity.algorithm", "unsupported integrity algorithm")
    expected_digest = _string(integrity["frame"], "$.integrity.frame")
    actual_digest = hashlib.sha256(
        _canonical_json(cast(JSONValue, root["frame"]))
    ).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        _fail("$.integrity.frame", "frame digest does not match")
    return _frame(root["frame"], "$.frame")


def load_scene(path: str | os.PathLike[str]) -> SceneFrame:
    """Load and validate a bounded scene document."""

    source = Path(path)
    try:
        with source.open("rb") as stream:
            encoded = stream.read(MAX_SCENE_BYTES + 1)
        if not encoded:
            raise SceneError("scene is empty")
        if len(encoded) > MAX_SCENE_BYTES:
            raise SceneError(f"scene exceeds the {MAX_SCENE_BYTES}-byte limit")
    except OSError as error:
        raise SceneError(f"could not read scene {source}") from error
    return parse_scene(encoded)
