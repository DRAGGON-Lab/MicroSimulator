"""Versioned, non-executable MicroSimulator checkpoints."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NoReturn, cast

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    CellSnapshot,
    ConstraintRegion,
    CoupledRatePlan,
    GridBoundary,
    GridBoundaryKind,
    GridShape,
    RateInstruction,
    RateOp,
    SignalGridAffineReaction,
    SignalGridSpec,
    SignalGridVelocityField,
    SignalIntegrationKind,
    Simulation,
    SpeciesRatePlan,
    Vec3,
    _BoxConstraint,
    _ConstraintSetCheckpoint,
    _CylinderConstraint,
    _LineageEntry,
    _PlaneConstraint,
    _SignalGridCheckpoint,
    _SimulationCheckpoint,
    _SphereConstraint,
    _WorldStateCheckpoint,
)
from .channels import UNNAMED_CHANNELS, ChannelMetadata, ChannelMetadataError

CHECKPOINT_FORMAT = "microsimulator-checkpoint"
CHECKPOINT_VERSION = 9
MAX_CHECKPOINT_BYTES = 1 << 30
_NATIVE_CHECKPOINT_VERSION = 4

_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_FLOAT32_MAX = 3.4028234663852886e38

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be safely decoded or validated."""


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


_RATE_OP_NAMES = {
    RateOp.CONSTANT: "constant",
    RateOp.SPECIES: "species",
    RateOp.POSITION_X: "position_x",
    RateOp.POSITION_Y: "position_y",
    RateOp.POSITION_Z: "position_z",
    RateOp.CELL_LENGTH: "cell_length",
    RateOp.CELL_RADIUS: "cell_radius",
    RateOp.GROWTH_RATE: "growth_rate",
    RateOp.CELL_TYPE: "cell_type",
    RateOp.CELL_VOLUME: "cell_volume",
    RateOp.CELL_VOLUME_CHANGE_RATE: "cell_volume_change_rate",
    RateOp.CELL_SURFACE_AREA: "cell_surface_area",
    RateOp.ADD: "add",
    RateOp.SUBTRACT: "subtract",
    RateOp.MULTIPLY: "multiply",
    RateOp.DIVIDE: "divide",
    RateOp.POWER: "power",
    RateOp.MINIMUM: "minimum",
    RateOp.MAXIMUM: "maximum",
    RateOp.NEGATE: "negate",
    RateOp.EXPONENTIAL: "exponential",
    RateOp.LOGARITHM: "logarithm",
    RateOp.LESS: "less",
    RateOp.LESS_EQUAL: "less_equal",
    RateOp.GREATER: "greater",
    RateOp.GREATER_EQUAL: "greater_equal",
    RateOp.EQUAL: "equal",
    RateOp.SELECT: "select",
    RateOp.SIGNAL: "signal",
}
_RATE_OPS = {name: operation for operation, name in _RATE_OP_NAMES.items()}
_CONSTRAINT_REGION_NAMES = {
    ConstraintRegion.OUTSIDE: "outside",
    ConstraintRegion.INSIDE: "inside",
}
_CONSTRAINT_REGIONS = {name: region for region, name in _CONSTRAINT_REGION_NAMES.items()}
_BACKEND_NAMES = {
    BackendKind.CPU: "cpu",
    BackendKind.METAL: "metal",
    BackendKind.CUDA: "cuda",
}
_GRID_BOUNDARY_NAMES = {
    GridBoundaryKind.NO_FLUX: "no_flux",
    GridBoundaryKind.PERIODIC: "periodic",
    GridBoundaryKind.FIXED: "fixed",
}
_GRID_BOUNDARIES = {name: kind for kind, name in _GRID_BOUNDARY_NAMES.items()}
_SIGNAL_INTEGRATION_NAMES = {
    SignalIntegrationKind.FORWARD_EULER: "forward_euler",
    SignalIntegrationKind.CRANK_NICOLSON: "crank_nicolson",
    SignalIntegrationKind.BACKWARD_EULER: "backward_euler",
}
_SIGNAL_INTEGRATIONS = {name: kind for kind, name in _SIGNAL_INTEGRATION_NAMES.items()}


def _installed_version() -> str:
    try:
        return version("microsimulator")
    except PackageNotFoundError:
        return "0+unknown"


def _vec3_to_json(value: Vec3) -> list[JSONValue]:
    return [value.x, value.y, value.z]


def _boundary_to_json(boundary: GridBoundary) -> dict[str, JSONValue]:
    return {
        "kind": _GRID_BOUNDARY_NAMES[boundary.kind],
        "values": list(boundary.values),
    }


def _signal_grid_to_json(checkpoint: _SignalGridCheckpoint | None) -> JSONValue:
    if checkpoint is None:
        return None
    spec = checkpoint.spec
    return {
        "spec": {
            "signal_count": spec.signal_count,
            "shape": [spec.shape.x, spec.shape.y, spec.shape.z],
            "origin": _vec3_to_json(spec.origin),
            "spacing": _vec3_to_json(spec.spacing),
            "diffusion": list(spec.diffusion),
            "advection": [_vec3_to_json(value) for value in spec.advection],
            "reaction": (
                {
                    "source_rates": list(spec.reaction.source_rates),
                    "loss_rates": list(spec.reaction.loss_rates),
                }
                if spec.reaction is not None
                else None
            ),
            "obstacles": list(spec.obstacles),
            "velocity_field": (
                {
                    "x_faces": list(spec.velocity_field.x_faces),
                    "y_faces": list(spec.velocity_field.y_faces),
                    "z_faces": list(spec.velocity_field.z_faces),
                }
                if spec.velocity_field is not None
                else None
            ),
            "integration": _SIGNAL_INTEGRATION_NAMES[spec.integration],
            "solver": {
                "max_iterations": spec.solver.max_iterations,
                "absolute_tolerance": spec.solver.absolute_tolerance,
                "relative_tolerance": spec.solver.relative_tolerance,
            },
            "boundaries": {
                "x_lower": _boundary_to_json(spec.x_lower),
                "x_upper": _boundary_to_json(spec.x_upper),
                "y_lower": _boundary_to_json(spec.y_lower),
                "y_upper": _boundary_to_json(spec.y_upper),
                "z_lower": _boundary_to_json(spec.z_lower),
                "z_upper": _boundary_to_json(spec.z_upper),
            },
        },
        "levels": list(checkpoint.levels),
    }


def _instructions_to_json(instructions: list[RateInstruction]) -> list[JSONValue]:
    return [
        {
            "operation": _RATE_OP_NAMES[instruction.operation],
            "first": instruction.first,
            "second": instruction.second,
            "third": instruction.third,
            "value": instruction.value,
        }
        for instruction in instructions
    ]


def _coupled_rate_plan_to_json(plan: CoupledRatePlan | None) -> JSONValue:
    if plan is None:
        return None
    return {
        "species_count": plan.species_count,
        "signal_count": plan.signal_count,
        "instructions": _instructions_to_json(plan.instructions),
        "species_outputs": list(plan.species_outputs),
        "signal_outputs": list(plan.signal_outputs),
    }


def _simulation_to_json(checkpoint: _SimulationCheckpoint) -> dict[str, JSONValue]:
    cells: list[JSONValue] = []
    for cell in checkpoint.world.cells:
        cells.append(
            {
                "id": cell.id,
                "slot": cell.slot,
                "position": _vec3_to_json(cell.position),
                "direction": _vec3_to_json(cell.direction),
                "length": cell.length,
                "radius": cell.radius,
                "growth_rate": cell.growth_rate,
                "cell_type": cell.cell_type,
                "fixed": cell.fixed,
                "species": list(cell.species),
            }
        )

    lineage: list[JSONValue] = [
        {"child": entry.child, "parent": entry.parent} for entry in checkpoint.world.lineage
    ]
    planes: list[JSONValue] = [
        {
            "id": plane.id,
            "point": _vec3_to_json(plane.point),
            "inward_normal": _vec3_to_json(plane.inward_normal),
            "coefficient": plane.coefficient,
        }
        for plane in checkpoint.constraints.planes
    ]
    spheres: list[JSONValue] = [
        {
            "id": sphere.id,
            "center": _vec3_to_json(sphere.center),
            "radius": sphere.radius,
            "coefficient": sphere.coefficient,
            "allowed_region": _CONSTRAINT_REGION_NAMES[sphere.allowed_region],
        }
        for sphere in checkpoint.constraints.spheres
    ]
    boxes: list[JSONValue] = [
        {
            "id": box.id,
            "center": _vec3_to_json(box.center),
            "half_extents": _vec3_to_json(box.half_extents),
            "coefficient": box.coefficient,
            "allowed_region": _CONSTRAINT_REGION_NAMES[box.allowed_region],
        }
        for box in checkpoint.constraints.boxes
    ]
    cylinders: list[JSONValue] = [
        {
            "id": cylinder.id,
            "center": _vec3_to_json(cylinder.center),
            "radius": cylinder.radius,
            "half_height": cylinder.half_height,
            "coefficient": cylinder.coefficient,
            "allowed_region": _CONSTRAINT_REGION_NAMES[cylinder.allowed_region],
        }
        for cylinder in checkpoint.constraints.cylinders
    ]
    instructions = _instructions_to_json(checkpoint.species_rate_plan.instructions)
    return {
        "time": checkpoint.time,
        "world": {
            "species_count": checkpoint.world.species_count,
            "next_id": checkpoint.world.next_id,
            "cells": cells,
            "lineage": lineage,
        },
        "constraints": {
            "next_id": checkpoint.constraints.next_id,
            "planes": planes,
            "spheres": spheres,
            "boxes": boxes,
            "cylinders": cylinders,
        },
        "species_rate_plan": {
            "species_count": checkpoint.species_rate_plan.species_count,
            "instructions": instructions,
            "outputs": list(checkpoint.species_rate_plan.outputs),
        },
        "signal_grid": _signal_grid_to_json(checkpoint.signal_grid),
        "coupled_rate_plan": _coupled_rate_plan_to_json(checkpoint.coupled_rate_plan),
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def save_checkpoint(
    simulation: Simulation,
    path: str | os.PathLike[str],
    *,
    provenance: Mapping[str, JSONValue] | None = None,
    controller: JSONValue = None,
    channel_metadata: ChannelMetadata = UNNAMED_CHANNELS,
) -> None:
    """Atomically save a complete simulation checkpoint as validated JSON."""

    checkpoint = simulation._checkpoint()
    checkpoint.validate()
    state = _simulation_to_json(checkpoint)
    try:
        labels = channel_metadata.to_json(simulation.species_count, simulation.signal_count)
    except ChannelMetadataError as error:
        raise CheckpointError(str(error)) from error
    digest = hashlib.sha256(_canonical_json(state)).hexdigest()
    controller_digest = hashlib.sha256(_canonical_json(controller)).hexdigest()
    backend = simulation.backend_info
    document: dict[str, JSONValue] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "producer": {"name": "microsimulator", "version": _installed_version()},
        "source_backend": {
            "kind": _BACKEND_NAMES[backend.kind],
            "name": backend.name,
            "device": backend.device,
            "device_index": backend.device_index,
            "native": backend.native,
        },
        "provenance": dict(provenance) if provenance is not None else {},
        "integrity": {
            "algorithm": "sha256",
            "simulation": digest,
            "controller": controller_digest,
            "channel_metadata": hashlib.sha256(_canonical_json(labels)).hexdigest(),
        },
        "simulation": state,
        "controller": controller,
        "channel_metadata": labels,
    }
    try:
        encoded = (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise CheckpointError("checkpoint provenance is not finite JSON data") from error

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
        raise CheckpointError(f"could not write checkpoint {destination}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fail(path: str, message: str) -> NoReturn:
    raise CheckpointError(f"{path}: {message}")


def _reject_constant(value: str) -> NoReturn:
    raise CheckpointError(f"checkpoint contains non-finite JSON number {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"checkpoint contains duplicate key {key!r}")
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


def _keys(
    value: dict[str, object], path: str, required: set[str], optional: set[str] | None = None
) -> None:
    allowed = required | (optional or set())
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing:
        _fail(path, f"missing keys {sorted(missing)}")
    if unknown:
        _fail(path, f"unknown keys {sorted(unknown)}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    return value


def _integer(value: object, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected an integer")
    if value < minimum or value > maximum:
        _fail(path, f"integer is outside [{minimum}, {maximum}]")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "expected a boolean")
    return value


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


def _vec3(value: object, path: str) -> Vec3:
    values = _array(value, path)
    if len(values) != 3:
        _fail(path, "expected exactly three coordinates")
    return Vec3(
        _number(values[0], f"{path}[0]", float32=True),
        _number(values[1], f"{path}[1]", float32=True),
        _number(values[2], f"{path}[2]", float32=True),
    )


def _cell(value: object, path: str, schema_version: int) -> CellSnapshot:
    data = _object(value, path)
    required = {
        "id",
        "slot",
        "position",
        "direction",
        "length",
        "radius",
        "growth_rate",
        "cell_type",
        "species",
    }
    if schema_version >= 6:
        required.add("fixed")
    _keys(
        data,
        path,
        required,
    )
    cell = CellSnapshot()
    cell.id = _integer(data["id"], f"{path}.id", 1, _UINT64_MAX)
    cell.slot = _integer(data["slot"], f"{path}.slot", 0, _UINT32_MAX)
    cell.position = _vec3(data["position"], f"{path}.position")
    cell.direction = _vec3(data["direction"], f"{path}.direction")
    cell.length = _number(data["length"], f"{path}.length", float32=True)
    cell.radius = _number(data["radius"], f"{path}.radius", float32=True)
    cell.growth_rate = _number(data["growth_rate"], f"{path}.growth_rate", float32=True)
    cell.cell_type = _integer(data["cell_type"], f"{path}.cell_type", _INT32_MIN, _INT32_MAX)
    cell.fixed = _boolean(data["fixed"], f"{path}.fixed") if schema_version >= 6 else False
    cell.species = [
        _number(item, f"{path}.species[{index}]", float32=True)
        for index, item in enumerate(_array(data["species"], f"{path}.species"))
    ]
    return cell


def _lineage_entry(value: object, path: str) -> _LineageEntry:
    data = _object(value, path)
    _keys(data, path, {"child", "parent"})
    entry = _LineageEntry()
    entry.child = _integer(data["child"], f"{path}.child", 1, _UINT64_MAX)
    entry.parent = _integer(data["parent"], f"{path}.parent", 1, _UINT64_MAX)
    return entry


def _plane(value: object, path: str) -> _PlaneConstraint:
    data = _object(value, path)
    _keys(data, path, {"id", "point", "inward_normal", "coefficient"})
    plane = _PlaneConstraint()
    plane.id = _integer(data["id"], f"{path}.id", 1, _UINT64_MAX)
    plane.point = _vec3(data["point"], f"{path}.point")
    plane.inward_normal = _vec3(data["inward_normal"], f"{path}.inward_normal")
    plane.coefficient = _number(data["coefficient"], f"{path}.coefficient", float32=True)
    return plane


def _sphere(value: object, path: str) -> _SphereConstraint:
    data = _object(value, path)
    _keys(data, path, {"id", "center", "radius", "coefficient", "allowed_region"})
    sphere = _SphereConstraint()
    sphere.id = _integer(data["id"], f"{path}.id", 1, _UINT64_MAX)
    sphere.center = _vec3(data["center"], f"{path}.center")
    sphere.radius = _number(data["radius"], f"{path}.radius", float32=True)
    sphere.coefficient = _number(data["coefficient"], f"{path}.coefficient", float32=True)
    region_name = _string(data["allowed_region"], f"{path}.allowed_region")
    try:
        sphere.allowed_region = _CONSTRAINT_REGIONS[region_name]
    except KeyError:
        _fail(f"{path}.allowed_region", f"unknown sphere region {region_name!r}")
    return sphere


def _box(value: object, path: str) -> _BoxConstraint:
    data = _object(value, path)
    _keys(data, path, {"id", "center", "half_extents", "coefficient", "allowed_region"})
    box = _BoxConstraint()
    box.id = _integer(data["id"], f"{path}.id", 1, _UINT64_MAX)
    box.center = _vec3(data["center"], f"{path}.center")
    box.half_extents = _vec3(data["half_extents"], f"{path}.half_extents")
    box.coefficient = _number(data["coefficient"], f"{path}.coefficient", float32=True)
    region_name = _string(data["allowed_region"], f"{path}.allowed_region")
    try:
        box.allowed_region = _CONSTRAINT_REGIONS[region_name]
    except KeyError:
        _fail(f"{path}.allowed_region", f"unknown box region {region_name!r}")
    return box


def _cylinder(value: object, path: str) -> _CylinderConstraint:
    data = _object(value, path)
    _keys(data, path, {"id", "center", "radius", "half_height", "coefficient", "allowed_region"})
    cylinder = _CylinderConstraint()
    cylinder.id = _integer(data["id"], f"{path}.id", 1, _UINT64_MAX)
    cylinder.center = _vec3(data["center"], f"{path}.center")
    cylinder.radius = _number(data["radius"], f"{path}.radius", float32=True)
    cylinder.half_height = _number(data["half_height"], f"{path}.half_height", float32=True)
    cylinder.coefficient = _number(data["coefficient"], f"{path}.coefficient", float32=True)
    region_name = _string(data["allowed_region"], f"{path}.allowed_region")
    try:
        cylinder.allowed_region = _CONSTRAINT_REGIONS[region_name]
    except KeyError:
        _fail(f"{path}.allowed_region", f"unknown cylinder region {region_name!r}")
    return cylinder


def _instruction(value: object, path: str) -> RateInstruction:
    data = _object(value, path)
    _keys(data, path, {"operation", "first", "second", "third", "value"})
    operation_name = _string(data["operation"], f"{path}.operation")
    instruction = RateInstruction()
    try:
        instruction.operation = _RATE_OPS[operation_name]
    except KeyError:
        _fail(f"{path}.operation", f"unknown rate operation {operation_name!r}")
    instruction.first = _integer(data["first"], f"{path}.first", 0, _UINT32_MAX)
    instruction.second = _integer(data["second"], f"{path}.second", 0, _UINT32_MAX)
    instruction.third = _integer(data["third"], f"{path}.third", 0, _UINT32_MAX)
    instruction.value = _number(data["value"], f"{path}.value", float32=True)
    return instruction


def _boundary(value: object, path: str) -> GridBoundary:
    data = _object(value, path)
    _keys(data, path, {"kind", "values"})
    kind_name = _string(data["kind"], f"{path}.kind")
    boundary = GridBoundary()
    try:
        boundary.kind = _GRID_BOUNDARIES[kind_name]
    except KeyError:
        _fail(f"{path}.kind", f"unknown grid boundary kind {kind_name!r}")
    boundary.values = [
        _number(item, f"{path}.values[{index}]", float32=True)
        for index, item in enumerate(_array(data["values"], f"{path}.values"))
    ]
    return boundary


def _affine_reaction(value: object, path: str) -> SignalGridAffineReaction | None:
    if value is None:
        return None
    data = _object(value, path)
    _keys(data, path, {"source_rates", "loss_rates"})
    reaction = SignalGridAffineReaction()
    reaction.source_rates = [
        _number(item, f"{path}.source_rates[{index}]", float32=True)
        for index, item in enumerate(_array(data["source_rates"], f"{path}.source_rates"))
    ]
    reaction.loss_rates = [
        _number(item, f"{path}.loss_rates[{index}]", float32=True)
        for index, item in enumerate(_array(data["loss_rates"], f"{path}.loss_rates"))
    ]
    return reaction


def _signal_grid(value: object, path: str, schema_version: int) -> _SignalGridCheckpoint | None:
    if value is None:
        return None
    data = _object(value, path)
    _keys(data, path, {"spec", "levels"})
    spec_data = _object(data["spec"], f"{path}.spec")
    spec_keys = {
        "signal_count",
        "shape",
        "origin",
        "spacing",
        "diffusion",
        "advection",
        "boundaries",
    }
    if schema_version >= 5:
        spec_keys.update({"integration", "solver"})
    if schema_version >= 7:
        spec_keys.add("reaction")
    if schema_version >= 8:
        spec_keys.update({"obstacles", "velocity_field"})
    _keys(
        spec_data,
        f"{path}.spec",
        spec_keys,
    )
    shape_values = _array(spec_data["shape"], f"{path}.spec.shape")
    if len(shape_values) != 3:
        _fail(f"{path}.spec.shape", "expected exactly three dimensions")
    shape = GridShape()
    shape.x = _integer(shape_values[0], f"{path}.spec.shape[0]", 1, _UINT32_MAX)
    shape.y = _integer(shape_values[1], f"{path}.spec.shape[1]", 1, _UINT32_MAX)
    shape.z = _integer(shape_values[2], f"{path}.spec.shape[2]", 1, _UINT32_MAX)

    boundaries = _object(spec_data["boundaries"], f"{path}.spec.boundaries")
    boundary_names = {"x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper"}
    _keys(boundaries, f"{path}.spec.boundaries", boundary_names)

    spec = SignalGridSpec()
    spec.signal_count = _integer(
        spec_data["signal_count"], f"{path}.spec.signal_count", 1, _UINT32_MAX
    )
    spec.shape = shape
    spec.origin = _vec3(spec_data["origin"], f"{path}.spec.origin")
    spec.spacing = _vec3(spec_data["spacing"], f"{path}.spec.spacing")
    spec.diffusion = [
        _number(item, f"{path}.spec.diffusion[{index}]", float32=True)
        for index, item in enumerate(_array(spec_data["diffusion"], f"{path}.spec.diffusion"))
    ]
    spec.advection = [
        _vec3(item, f"{path}.spec.advection[{index}]")
        for index, item in enumerate(_array(spec_data["advection"], f"{path}.spec.advection"))
    ]
    if schema_version >= 7:
        spec.reaction = _affine_reaction(spec_data["reaction"], f"{path}.spec.reaction")
    if schema_version >= 8:
        spec.obstacles = [
            _integer(item, f"{path}.spec.obstacles[{index}]", 0, 1)
            for index, item in enumerate(_array(spec_data["obstacles"], f"{path}.spec.obstacles"))
        ]
        field_value = spec_data["velocity_field"]
        if field_value is not None:
            field_data = _object(field_value, f"{path}.spec.velocity_field")
            _keys(field_data, f"{path}.spec.velocity_field", {"x_faces", "y_faces", "z_faces"})
            field = SignalGridVelocityField()
            field.x_faces = [
                _number(item, f"{path}.spec.velocity_field.x_faces[{index}]", float32=True)
                for index, item in enumerate(
                    _array(field_data["x_faces"], f"{path}.spec.velocity_field.x_faces")
                )
            ]
            field.y_faces = [
                _number(item, f"{path}.spec.velocity_field.y_faces[{index}]", float32=True)
                for index, item in enumerate(
                    _array(field_data["y_faces"], f"{path}.spec.velocity_field.y_faces")
                )
            ]
            field.z_faces = [
                _number(item, f"{path}.spec.velocity_field.z_faces[{index}]", float32=True)
                for index, item in enumerate(
                    _array(field_data["z_faces"], f"{path}.spec.velocity_field.z_faces")
                )
            ]
            spec.velocity_field = field
    if schema_version >= 5:
        integration_name = _string(spec_data["integration"], f"{path}.spec.integration")
        if integration_name not in _SIGNAL_INTEGRATIONS:
            _fail(f"{path}.spec.integration", f"unknown integration {integration_name!r}")
        spec.integration = _SIGNAL_INTEGRATIONS[integration_name]
        solver_data = _object(spec_data["solver"], f"{path}.spec.solver")
        _keys(
            solver_data,
            f"{path}.spec.solver",
            {"max_iterations", "absolute_tolerance", "relative_tolerance"},
        )
        spec.solver.max_iterations = _integer(
            solver_data["max_iterations"],
            f"{path}.spec.solver.max_iterations",
            1,
            _UINT32_MAX,
        )
        spec.solver.absolute_tolerance = _number(
            solver_data["absolute_tolerance"],
            f"{path}.spec.solver.absolute_tolerance",
            float32=True,
        )
        spec.solver.relative_tolerance = _number(
            solver_data["relative_tolerance"],
            f"{path}.spec.solver.relative_tolerance",
            float32=True,
        )
    spec.x_lower = _boundary(boundaries["x_lower"], f"{path}.spec.boundaries.x_lower")
    spec.x_upper = _boundary(boundaries["x_upper"], f"{path}.spec.boundaries.x_upper")
    spec.y_lower = _boundary(boundaries["y_lower"], f"{path}.spec.boundaries.y_lower")
    spec.y_upper = _boundary(boundaries["y_upper"], f"{path}.spec.boundaries.y_upper")
    spec.z_lower = _boundary(boundaries["z_lower"], f"{path}.spec.boundaries.z_lower")
    spec.z_upper = _boundary(boundaries["z_upper"], f"{path}.spec.boundaries.z_upper")

    checkpoint = _SignalGridCheckpoint()
    checkpoint.spec = spec
    checkpoint.levels = [
        _number(item, f"{path}.levels[{index}]", float32=True)
        for index, item in enumerate(_array(data["levels"], f"{path}.levels"))
    ]
    return checkpoint


def _coupled_rate_plan(value: object, path: str) -> CoupledRatePlan | None:
    if value is None:
        return None
    data = _object(value, path)
    _keys(
        data,
        path,
        {
            "species_count",
            "signal_count",
            "instructions",
            "species_outputs",
            "signal_outputs",
        },
    )
    species_count = _integer(data["species_count"], f"{path}.species_count", 0, _UINT64_MAX)
    signal_count = _integer(data["signal_count"], f"{path}.signal_count", 1, _UINT64_MAX)
    instructions = [
        _instruction(item, f"{path}.instructions[{index}]")
        for index, item in enumerate(_array(data["instructions"], f"{path}.instructions"))
    ]
    species_outputs = [
        _integer(item, f"{path}.species_outputs[{index}]", 0, _UINT32_MAX)
        for index, item in enumerate(_array(data["species_outputs"], f"{path}.species_outputs"))
    ]
    signal_outputs = [
        _integer(item, f"{path}.signal_outputs[{index}]", 0, _UINT32_MAX)
        for index, item in enumerate(_array(data["signal_outputs"], f"{path}.signal_outputs"))
    ]
    try:
        return CoupledRatePlan(
            species_count,
            signal_count,
            instructions,
            species_outputs,
            signal_outputs,
        )
    except (ValueError, OverflowError) as error:
        raise CheckpointError(f"{path}: invalid coupled rate plan: {error}") from error


def _native_checkpoint(value: object, schema_version: int) -> _SimulationCheckpoint:
    data = _object(value, "$.simulation")
    required = {"time", "world", "constraints", "species_rate_plan"}
    if schema_version >= 2:
        required.add("signal_grid")
    if schema_version >= 3:
        required.add("coupled_rate_plan")
    _keys(data, "$.simulation", required)

    world_data = _object(data["world"], "$.simulation.world")
    _keys(world_data, "$.simulation.world", {"species_count", "next_id", "cells", "lineage"})
    world = _WorldStateCheckpoint()
    world.species_count = _integer(
        world_data["species_count"], "$.simulation.world.species_count", 0, _UINT64_MAX
    )
    world.next_id = _integer(world_data["next_id"], "$.simulation.world.next_id", 1, _UINT64_MAX)
    world.cells = [
        _cell(item, f"$.simulation.world.cells[{index}]", schema_version)
        for index, item in enumerate(_array(world_data["cells"], "$.simulation.world.cells"))
    ]
    world.lineage = [
        _lineage_entry(item, f"$.simulation.world.lineage[{index}]")
        for index, item in enumerate(_array(world_data["lineage"], "$.simulation.world.lineage"))
    ]

    constraint_data = _object(data["constraints"], "$.simulation.constraints")
    constraint_keys = {"next_id", "planes", "spheres"}
    if schema_version >= 8:
        constraint_keys.update({"boxes", "cylinders"})
    _keys(constraint_data, "$.simulation.constraints", constraint_keys)
    constraints = _ConstraintSetCheckpoint()
    constraints.next_id = _integer(
        constraint_data["next_id"], "$.simulation.constraints.next_id", 1, _UINT64_MAX
    )
    constraints.planes = [
        _plane(item, f"$.simulation.constraints.planes[{index}]")
        for index, item in enumerate(
            _array(constraint_data["planes"], "$.simulation.constraints.planes")
        )
    ]
    constraints.spheres = [
        _sphere(item, f"$.simulation.constraints.spheres[{index}]")
        for index, item in enumerate(
            _array(constraint_data["spheres"], "$.simulation.constraints.spheres")
        )
    ]
    if schema_version >= 8:
        constraints.boxes = [
            _box(item, f"$.simulation.constraints.boxes[{index}]")
            for index, item in enumerate(
                _array(constraint_data["boxes"], "$.simulation.constraints.boxes")
            )
        ]
        constraints.cylinders = [
            _cylinder(item, f"$.simulation.constraints.cylinders[{index}]")
            for index, item in enumerate(
                _array(constraint_data["cylinders"], "$.simulation.constraints.cylinders")
            )
        ]

    plan_data = _object(data["species_rate_plan"], "$.simulation.species_rate_plan")
    _keys(
        plan_data,
        "$.simulation.species_rate_plan",
        {"species_count", "instructions", "outputs"},
    )
    plan_species_count = _integer(
        plan_data["species_count"],
        "$.simulation.species_rate_plan.species_count",
        0,
        _UINT64_MAX,
    )
    instructions = [
        _instruction(item, f"$.simulation.species_rate_plan.instructions[{index}]")
        for index, item in enumerate(
            _array(plan_data["instructions"], "$.simulation.species_rate_plan.instructions")
        )
    ]
    outputs = [
        _integer(item, f"$.simulation.species_rate_plan.outputs[{index}]", 0, _UINT32_MAX)
        for index, item in enumerate(
            _array(plan_data["outputs"], "$.simulation.species_rate_plan.outputs")
        )
    ]

    checkpoint = _SimulationCheckpoint()
    checkpoint.schema_version = _NATIVE_CHECKPOINT_VERSION
    checkpoint.time = _number(data["time"], "$.simulation.time")
    checkpoint.world = world
    checkpoint.constraints = constraints
    checkpoint.species_rate_plan = SpeciesRatePlan(plan_species_count, instructions, outputs)
    checkpoint.signal_grid = (
        _signal_grid(data["signal_grid"], "$.simulation.signal_grid", schema_version)
        if schema_version >= 2
        else None
    )
    checkpoint.coupled_rate_plan = (
        _coupled_rate_plan(data["coupled_rate_plan"], "$.simulation.coupled_rate_plan")
        if schema_version >= 3
        else None
    )
    try:
        checkpoint.validate()
    except (ValueError, OverflowError) as error:
        raise CheckpointError(f"checkpoint state is invalid: {error}") from error
    return checkpoint


def load_checkpoint_bundle(
    path: str | os.PathLike[str],
    *,
    backend: BackendKind = BackendKind.CPU,
    device_index: int = 0,
) -> CheckpointBundle:
    """Load native and optional controller state without evaluating executable content."""

    source = Path(path)
    try:
        with source.open("rb") as stream:
            encoded = stream.read(MAX_CHECKPOINT_BYTES + 1)
        if not encoded:
            raise CheckpointError("checkpoint is empty")
        if len(encoded) > MAX_CHECKPOINT_BYTES:
            raise CheckpointError(f"checkpoint exceeds the {MAX_CHECKPOINT_BYTES}-byte limit")
    except OSError as error:
        raise CheckpointError(f"could not read checkpoint {source}") from error

    try:
        decoded = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except CheckpointError:
        raise
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise CheckpointError(f"checkpoint is not valid UTF-8 JSON: {error}") from error

    root = _object(cast(object, decoded), "$")
    if "version" not in root:
        _fail("$", "missing keys ['version']")
    schema_version = _integer(root["version"], "$.version", 0, _UINT32_MAX)
    supported_versions = {1, 2, 3, 4, 5, 6, 7, 8, CHECKPOINT_VERSION}
    if schema_version not in supported_versions:
        _fail("$.version", f"unsupported checkpoint version {schema_version}")
    required = {
        "format",
        "version",
        "producer",
        "source_backend",
        "provenance",
        "integrity",
        "simulation",
    }
    if schema_version >= 4:
        required.add("controller")
    if schema_version >= 9:
        required.add("channel_metadata")
    _keys(
        root,
        "$",
        required,
    )
    if _string(root["format"], "$.format") not in (CHECKPOINT_FORMAT, "cellmodeller2-checkpoint"):
        _fail("$.format", "not a MicroSimulator checkpoint")
    _object(root["producer"], "$.producer")
    source_backend_data = _object(root["source_backend"], "$.source_backend")
    _keys(
        source_backend_data,
        "$.source_backend",
        {"kind", "name", "device", "device_index", "native"},
    )
    source_backend_kind = _string(source_backend_data["kind"], "$.source_backend.kind")
    if source_backend_kind not in _BACKEND_NAMES.values():
        _fail("$.source_backend.kind", f"unknown backend kind {source_backend_kind!r}")
    source_backend = CheckpointSourceBackend(
        kind=source_backend_kind,
        name=_string(source_backend_data["name"], "$.source_backend.name"),
        device=_string(source_backend_data["device"], "$.source_backend.device"),
        device_index=_integer(
            source_backend_data["device_index"],
            "$.source_backend.device_index",
            0,
            _UINT32_MAX,
        ),
        native=_boolean(source_backend_data["native"], "$.source_backend.native"),
    )
    provenance = cast(dict[str, JSONValue], _object(root["provenance"], "$.provenance"))

    integrity = _object(root["integrity"], "$.integrity")
    integrity_keys = {"algorithm", "simulation"}
    if schema_version >= 4:
        integrity_keys.add("controller")
    if schema_version >= 9:
        integrity_keys.add("channel_metadata")
    _keys(integrity, "$.integrity", integrity_keys)
    if _string(integrity["algorithm"], "$.integrity.algorithm") != "sha256":
        _fail("$.integrity.algorithm", "unsupported integrity algorithm")
    expected_digest = _string(integrity["simulation"], "$.integrity.simulation")
    actual_digest = hashlib.sha256(_canonical_json(root["simulation"])).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        _fail("$.integrity.simulation", "state digest does not match")

    controller = cast(JSONValue, root["controller"]) if schema_version >= 4 else None
    if schema_version >= 4:
        expected_controller_digest = _string(integrity["controller"], "$.integrity.controller")
        actual_controller_digest = hashlib.sha256(_canonical_json(controller)).hexdigest()
        if not hmac.compare_digest(actual_controller_digest, expected_controller_digest):
            _fail("$.integrity.controller", "controller digest does not match")

    if schema_version >= 9:
        expected_labels_digest = _string(
            integrity["channel_metadata"], "$.integrity.channel_metadata"
        )
        actual_labels_digest = hashlib.sha256(_canonical_json(root["channel_metadata"])).hexdigest()
        if not hmac.compare_digest(actual_labels_digest, expected_labels_digest):
            _fail("$.integrity.channel_metadata", "channel metadata digest does not match")
    checkpoint = _native_checkpoint(root["simulation"], schema_version)
    species_count = checkpoint.world.species_count
    signal_count = checkpoint.signal_grid.spec.signal_count if checkpoint.signal_grid else 0
    try:
        labels = (
            ChannelMetadata.from_json(root["channel_metadata"], species_count, signal_count)
            if schema_version >= 9
            else ChannelMetadata().resolved(species_count, signal_count)
        )
    except ChannelMetadataError as error:
        raise CheckpointError(str(error)) from error
    return CheckpointBundle(
        simulation=Simulation(backend, checkpoint, device_index),
        controller=controller,
        provenance=provenance,
        schema_version=schema_version,
        source_backend=source_backend,
        channel_metadata=labels,
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
