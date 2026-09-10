"""One-way import of trusted CellModeller 1 pickle snapshots."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import io
import math
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
from numpy._core.multiarray import (  # pyright: ignore[reportPrivateUsage]
    _reconstruct as _numpy_reconstruct,
)
from numpy._core.multiarray import scalar as _numpy_scalar  # pyright: ignore[reportPrivateUsage]

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    CellSnapshot,
    Simulation,
    SpeciesRatePlan,
    Vec3,
    _ConstraintSetCheckpoint,
    _LineageEntry,
    _SimulationCheckpoint,
    _WorldStateCheckpoint,
)
from .checkpoint import JSONValue

MAX_LEGACY_PICKLE_BYTES = 1 << 30
_FLOAT32_MAX = 3.4028234663852886e38
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1

_MIGRATED_CELL_FIELDS = {
    "id",
    "idx",
    "pos",
    "dir",
    "length",
    "radius",
    "growthRate",
    "cellType",
    "species",
}


class LegacyPickleError(ValueError):
    """Raised when a legacy snapshot cannot be migrated without ambiguity."""


class _LegacyCellRecord(dict[str, object]):
    """Inert target for both object-style and historical dict-style CellState values."""

    def __setstate__(self, state: object) -> None:
        if not isinstance(state, dict):
            raise LegacyPickleError("legacy CellState has unsupported instance state")
        mapping = cast(dict[object, object], state)
        if not all(isinstance(key, str) for key in mapping):
            raise LegacyPickleError("legacy CellState has unsupported instance state")
        vars(self).update(cast(dict[str, object], mapping))


def _legacy_reconstructor(cls: object, base: object, state: object) -> _LegacyCellRecord:
    if cls is not _LegacyCellRecord or (base is not object and base is not dict):
        raise LegacyPickleError("legacy pickle requested an unsupported class reconstruction")
    del state
    return _LegacyCellRecord()


def _legacy_bytes(value: object, encoding: object) -> bytes:
    if not isinstance(value, str) or encoding != "latin1":
        raise LegacyPickleError("legacy pickle requested an unsupported byte encoding")
    return value.encode("latin1")


class _LegacyUnpickler(pickle.Unpickler):
    _SAFE_GLOBALS: ClassVar[dict[tuple[str, str], object]] = {
        ("CellModeller.CellState", "CellState"): _LegacyCellRecord,
        ("_codecs", "encode"): _legacy_bytes,
        ("__builtin__", "dict"): dict,
        ("__builtin__", "object"): object,
        ("builtins", "complex"): complex,
        ("builtins", "dict"): dict,
        ("builtins", "frozenset"): frozenset,
        ("builtins", "object"): object,
        ("builtins", "set"): set,
        ("builtins", "slice"): slice,
        ("copy_reg", "_reconstructor"): _legacy_reconstructor,
        ("copyreg", "_reconstructor"): _legacy_reconstructor,
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy._core.multiarray", "_reconstruct"): _numpy_reconstruct,
        ("numpy._core.multiarray", "scalar"): _numpy_scalar,
        ("numpy.core.multiarray", "_reconstruct"): _numpy_reconstruct,
        ("numpy.core.multiarray", "scalar"): _numpy_scalar,
    }

    def find_class(self, module: str, name: str) -> Any:
        try:
            return self._SAFE_GLOBALS[(module, name)]
        except KeyError as error:
            raise LegacyPickleError(
                f"legacy pickle requests forbidden global {module}.{name}"
            ) from error


@dataclass(frozen=True, slots=True)
class LegacyPickleImport:
    simulation: Simulation
    provenance: dict[str, JSONValue]
    dropped_cell_fields: tuple[str, ...]


def _read_pickle(path: Path) -> tuple[object, bytes]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise LegacyPickleError(f"could not inspect legacy pickle {path}") from error
    if size > MAX_LEGACY_PICKLE_BYTES:
        raise LegacyPickleError("legacy pickle exceeds the 1 GiB import limit")
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise LegacyPickleError(f"could not read legacy pickle {path}") from error
    if len(encoded) > MAX_LEGACY_PICKLE_BYTES:
        raise LegacyPickleError("legacy pickle exceeds the 1 GiB import limit")
    stream = io.BytesIO(encoded)
    try:
        value = _LegacyUnpickler(stream, fix_imports=True, encoding="latin1").load()
    except LegacyPickleError:
        raise
    except (EOFError, pickle.UnpicklingError, AttributeError, ImportError, IndexError) as error:
        raise LegacyPickleError("legacy pickle is malformed or unsupported") from error
    if stream.read(1):
        raise LegacyPickleError("legacy pickle contains trailing data")
    return value, encoded


def _mapping(value: object, path: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise LegacyPickleError(f"{path} must be a mapping")
    return cast(Mapping[object, object], value)


def _integer(value: object, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise LegacyPickleError(f"{path} must be an integer")
    result = int(cast(Any, value))
    if result < minimum or result > maximum:
        raise LegacyPickleError(f"{path} is outside [{minimum}, {maximum}]")
    return result


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise LegacyPickleError(f"{path} must be a number")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise LegacyPickleError(f"{path} must be a finite float32 value") from error
    if not math.isfinite(result) or abs(result) > _FLOAT32_MAX:
        raise LegacyPickleError(f"{path} must be a finite float32 value")
    return result


def _sequence(value: object, path: str) -> Sequence[object]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise LegacyPickleError(f"{path} must be one-dimensional")
        return cast(list[object], value.tolist())
    if isinstance(value, list | tuple):
        return cast(Sequence[object], value)
    raise LegacyPickleError(f"{path} must be a sequence")


def _vec3(value: object, path: str) -> Vec3:
    sequence = _sequence(value, path)
    if len(sequence) != 3:
        raise LegacyPickleError(f"{path} must contain three coordinates")
    return Vec3(
        _number(sequence[0], f"{path}[0]"),
        _number(sequence[1], f"{path}[1]"),
        _number(sequence[2], f"{path}[2]"),
    )


def _attributes(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, _LegacyCellRecord):
        raise LegacyPickleError(f"{path} is not a legacy CellState")
    attributes: dict[str, object] = dict(value)
    attributes.update(vars(value))
    return attributes


def _root_parts(
    value: object,
) -> tuple[Mapping[object, object], Mapping[object, object], dict[str, object], str]:
    if isinstance(value, dict):
        root = cast(dict[str, object], value)
        cells = _mapping(root.get("cellStates"), "legacy.cellStates")
        lineage = _mapping(root.get("lineage", {}), "legacy.lineage")
        return cells, lineage, root, "mapping"
    if isinstance(value, tuple):
        sequence = cast(tuple[object, ...], value)
        if len(sequence) not in {2, 3}:
            raise LegacyPickleError("legacy tuple snapshot must have two or three fields")
        cells = _mapping(sequence[0], "legacy[0]")
        lineage_index = 1 if len(sequence) == 2 else 2
        lineage = _mapping(sequence[lineage_index], f"legacy[{lineage_index}]")
        return cells, lineage, {}, f"tuple-v{len(sequence)}"
    raise LegacyPickleError("legacy pickle root must be a snapshot mapping or tuple")


def _physical_time(
    root: Mapping[str, object], time: float | None, dt: float | None
) -> tuple[float, int | None, str]:
    if time is not None and dt is not None:
        raise LegacyPickleError("supply either physical time or legacy dt, not both")
    step_value = root.get("stepNum")
    step_number = (
        _integer(step_value, "legacy.stepNum", 0, _UINT64_MAX)
        if step_value is not None
        else None
    )
    if time is not None:
        if not math.isfinite(time) or time < 0.0:
            raise LegacyPickleError("physical time must be finite and non-negative")
        return time, step_number, "explicit-time"
    if dt is not None:
        if not math.isfinite(dt) or dt < 0.0:
            raise LegacyPickleError("legacy dt must be finite and non-negative")
        if step_number is None:
            raise LegacyPickleError("tuple snapshots require explicit physical time")
        result = float(step_number) * dt
        if not math.isfinite(result):
            raise LegacyPickleError("derived physical time is not finite")
        return result, step_number, "step-number-times-dt"
    raise LegacyPickleError("physical time is required; supply time or legacy dt")


def import_legacy_pickle(
    path: str | Path,
    *,
    time: float | None = None,
    dt: float | None = None,
    trusted: bool = False,
    native_state_only: bool = False,
) -> LegacyPickleImport:
    """Migrate geometry, species, IDs, and lineage from a trusted legacy snapshot."""

    if not trusted:
        raise LegacyPickleError("legacy pickle import requires trusted=True")
    if not native_state_only:
        raise LegacyPickleError("legacy pickle import requires native_state_only=True")

    source = Path(path).resolve()
    value, encoded = _read_pickle(source)
    cell_values, lineage_values, root, legacy_format = _root_parts(value)
    physical_time, step_number, time_basis = _physical_time(root, time, dt)

    indexed_cells: list[tuple[int, CellSnapshot]] = []
    dropped_fields: set[str] = set()
    species_count: int | None = None
    active_ids: set[int] = set()
    for raw_id, record in cell_values.items():
        cell_id = _integer(raw_id, "legacy.cellStates key", 1, _UINT64_MAX)
        attributes = _attributes(record, f"legacy.cellStates[{cell_id}]")
        record_id = _integer(
            attributes.get("id"),
            f"legacy.cellStates[{cell_id}].id",
            1,
            _UINT64_MAX,
        )
        if record_id != cell_id or cell_id in active_ids:
            raise LegacyPickleError("legacy cell identifiers are inconsistent or duplicated")
        active_ids.add(cell_id)
        slot = _integer(
            attributes.get("idx"), f"legacy.cellStates[{cell_id}].idx", 0, _UINT32_MAX
        )
        species_value = attributes.get("species", [])
        species = [
            _number(item, f"legacy.cellStates[{cell_id}].species[{index}]")
            for index, item in enumerate(
                _sequence(species_value, f"legacy.cellStates[{cell_id}].species")
            )
        ]
        if species_count is None:
            species_count = len(species)
        elif len(species) != species_count:
            raise LegacyPickleError("legacy cells have inconsistent species counts")

        snapshot = CellSnapshot()
        snapshot.id = cell_id
        snapshot.slot = slot
        snapshot.position = _vec3(
            attributes.get("pos"), f"legacy.cellStates[{cell_id}].pos"
        )
        direction = _vec3(
            attributes.get("dir"), f"legacy.cellStates[{cell_id}].dir"
        )
        magnitude = math.sqrt(
            direction.x * direction.x + direction.y * direction.y + direction.z * direction.z
        )
        if not math.isfinite(magnitude) or magnitude <= 0.0:
            raise LegacyPickleError(f"legacy.cellStates[{cell_id}].dir must be non-zero")
        snapshot.direction = Vec3(
            direction.x / magnitude,
            direction.y / magnitude,
            direction.z / magnitude,
        )
        snapshot.length = _number(
            attributes.get("length"), f"legacy.cellStates[{cell_id}].length"
        )
        snapshot.radius = _number(
            attributes.get("radius"), f"legacy.cellStates[{cell_id}].radius"
        )
        snapshot.growth_rate = _number(
            attributes.get("growthRate", 1.0),
            f"legacy.cellStates[{cell_id}].growthRate",
        )
        snapshot.cell_type = _integer(
            attributes.get("cellType", 0),
            f"legacy.cellStates[{cell_id}].cellType",
            _INT32_MIN,
            _INT32_MAX,
        )
        snapshot.species = species
        dropped_fields.update(attributes.keys() - _MIGRATED_CELL_FIELDS)
        indexed_cells.append((slot, snapshot))

    indexed_cells.sort(key=lambda item: item[0])
    if [slot for slot, _ in indexed_cells] != list(range(len(indexed_cells))):
        raise LegacyPickleError("legacy active cell slots must be compact and unique")
    cells = [cell for _, cell in indexed_cells]

    lineage: list[_LineageEntry] = []
    all_ids = set(active_ids)
    for raw_child, raw_parent in lineage_values.items():
        child = _integer(raw_child, "legacy.lineage child", 1, _UINT64_MAX)
        parent = _integer(raw_parent, f"legacy.lineage[{child}]", 1, _UINT64_MAX)
        if parent >= child:
            raise LegacyPickleError("legacy lineage must have monotonic parent-child IDs")
        entry = _LineageEntry()
        entry.child = child
        entry.parent = parent
        lineage.append(entry)
        all_ids.update((child, parent))
    lineage.sort(key=lambda entry: entry.child)
    maximum_id = max(all_ids, default=0)
    if maximum_id == _UINT64_MAX:
        raise LegacyPickleError("legacy cell identifier space is exhausted")

    world = _WorldStateCheckpoint()
    world.species_count = species_count or 0
    world.next_id = maximum_id + 1
    world.cells = cells
    world.lineage = lineage
    try:
        world.validate()
        native = _SimulationCheckpoint()
        native.time = physical_time
        native.world = world
        native.constraints = _ConstraintSetCheckpoint()
        native.species_rate_plan = SpeciesRatePlan.zero(world.species_count)
        native.signal_grid = None
        native.coupled_rate_plan = None
        native.validate()
        simulation = Simulation(BackendKind.CPU, native)
    except (ValueError, OverflowError, RuntimeError) as error:
        raise LegacyPickleError(f"legacy snapshot state is invalid: {error}") from error

    module_name = root.get("moduleName")
    module_source = root.get("moduleStr")
    source_digest: JSONValue = None
    if isinstance(module_source, str):
        source_digest = hashlib.sha256(module_source.encode("utf-8")).hexdigest()
    dropped_values: list[JSONValue] = [field for field in sorted(dropped_fields)]
    limitations: list[JSONValue] = [
        "legacy constraints were not stored in pickle output",
        "callback random state was not stored",
        "rate plans and signal transport were not reconstructed",
    ]
    legacy_provenance: dict[str, JSONValue] = {
        "path": str(source),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "legacy_format": legacy_format,
        "step_number": step_number,
        "time_basis": time_basis,
        "module_name": module_name if isinstance(module_name, str) else None,
        "module_source_sha256": source_digest,
        "migration_mode": "geometry-species",
        "dropped_cell_fields": dropped_values,
        "limitations": limitations,
    }
    provenance: dict[str, JSONValue] = {"legacy_pickle": legacy_provenance}
    return LegacyPickleImport(
        simulation=simulation,
        provenance=provenance,
        dropped_cell_fields=tuple(sorted(dropped_fields)),
    )
