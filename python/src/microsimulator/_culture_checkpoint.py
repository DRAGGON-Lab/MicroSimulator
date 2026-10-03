"""Closed, data-only schema for the native culture state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

from . import _core as core  # pyright: ignore[reportMissingModuleSource, reportPrivateUsage]

type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None


@dataclass(frozen=True)
class _Record:
    factory: type[object]
    fields: dict[str, Codec]


@dataclass(frozen=True)
class _Array:
    element: Codec
    length: int | None = None


@dataclass(frozen=True)
class _Enum:
    values: dict[str, object]


type Codec = Literal["float", "u32", "u64", "bool", "str"] | _Record | _Array | _Enum


def _encode(value: object, codec: Codec) -> JSONValue:
    if isinstance(codec, _Record):
        return {
            key: _encode(cast(object, getattr(value, key)), item)
            for key, item in codec.fields.items()
        }

    if isinstance(codec, _Array):
        return [_encode(item, codec.element) for item in cast(list[object], value)]

    if isinstance(codec, _Enum):
        for name, item in codec.values.items():
            if value == item:
                return name

        raise ValueError("unknown native culture enum value")

    return cast(JSONValue, value)


def _decode_record(value: object, codec: _Record, path: str, *, validate_only: bool) -> object:
    if not isinstance(value, dict) or set(cast(dict[object, object], value)) != set(codec.fields):
        raise ValueError(f"{path}: culture object fields do not match the schema")

    data = cast(dict[str, object], value)
    result = None if validate_only else codec.factory()

    for key, item in codec.fields.items():
        decoded = _decode(data[key], item, f"{path}.{key}", validate_only=validate_only)

        if result is not None:
            setattr(result, key, decoded)

    return result


def _decode_array(value: object, codec: _Array, path: str, *, validate_only: bool) -> list[object]:
    if not isinstance(value, list) or (
        codec.length is not None and len(cast(list[object], value)) != codec.length
    ):
        raise ValueError(f"{path}: invalid culture array")

    return [
        _decode(item, codec.element, f"{path}[{i}]", validate_only=validate_only)
        for i, item in enumerate(cast(list[object], value))
    ]


def _decode_scalar(
    value: object, codec: Literal["float", "u32", "u64", "bool", "str"], path: str
) -> object:
    if codec == "bool":
        if type(value) is not bool:
            raise ValueError(f"{path}: expected a Boolean")
    elif codec == "str":
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected a string")
    elif codec in ("u32", "u64"):
        maximum = (1 << (32 if codec == "u32" else 64)) - 1

        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError(f"{path}: unsigned integer out of range")
    elif codec == "float":
        if type(value) not in (int, float) or not math.isfinite(cast(float, value)):
            raise ValueError(f"{path}: expected a finite number")

        return float(cast(float, value))

    return value


def _decode(value: object, codec: Codec, path: str, *, validate_only: bool = False) -> object:
    if isinstance(codec, _Record):
        return _decode_record(value, codec, path, validate_only=validate_only)

    if isinstance(codec, _Array):
        return _decode_array(value, codec, path, validate_only=validate_only)

    if isinstance(codec, _Enum):
        if not isinstance(value, str) or value not in codec.values:
            raise ValueError(f"{path}: invalid culture enum")

        return codec.values[value]

    return _decode_scalar(value, codec, path)


_axis = _Enum({"x": core.FlowAxis.X, "y": core.FlowAxis.Y, "z": core.FlowAxis.Z})
_port_kind = _Enum(
    {"pressure": core.FlowPortKind.PRESSURE, "flow_rate": core.FlowPortKind.FLOW_RATE}
)
_shape = _Record(core.GridShape, {"x": "u32", "y": "u32", "z": "u32"})
_vector = _Record(core.Vec3, {"x": "float", "y": "float", "z": "float"})
_grid = _Record(
    core.FluidGridSpec,
    {
        "shape": _shape,
        "origin": _vector,
        "spacing": "float",
        "length_unit_m": "float",
        "time_unit_s": "float",
        "obstacles": _Array("u32"),
    },
)
_fluid = _Record(core.FluidProperties, {"viscosity_pa_s": "float", "density_kg_m3": "float"})
_port = _Record(
    core.FlowPort,
    {
        "name": "str",
        "axis": _axis,
        "upper": "bool",
        "kind": _port_kind,
        "value": "float",
        "sites": _Array("u32"),
    },
)
_solver = _Record(
    core.LinearSolveParameters,
    {
        "relative_tolerance": "float",
        "absolute_tolerance": "float",
        "max_iterations": "u32",
        "memory_limit_bytes": "u64",
    },
)
_stepping = _Record(
    core.FluidBodyStepParameters,
    {
        "minimum_gap_m": "float",
        "maximum_displacement_fraction": "float",
        "max_halvings": "u32",
        "max_contact_iterations": "u32",
    },
)
_geometry = _Record(
    core.FluidGeometryParameters,
    {
        "surface_resolution": "u32",
        "maximum_surface_error_fraction": "float",
        "memory_limit_bytes": "u64",
    },
)
_solute = _Record(core.Solute, {"name": "str", "diffusion": "float", "amount_unit": "str"})
_reservoir = _Record(
    core.ChemicalBoundary,
    {
        "port": "str",
        "concentrations": _Array("float"),
        "kind": _Enum(
            {
                "reservoir_contact": core.ChemicalBoundaryKind.RESERVOIR_CONTACT,
                "advective": core.ChemicalBoundaryKind.ADVECTIVE,
                "outflow": core.ChemicalBoundaryKind.OUTFLOW,
            }
        ),
        "allow_backflow": "bool",
    },
)
_exchange = _Record(
    core.CellSurfaceExchange,
    {
        "body_id": "u64",
        "solute": "u32",
        "species": "u32",
        "uptake_velocity": "float",
        "secretion_rate": "float",
    },
)
_requirement = _Record(core.ReserveRequirement, {"species": "u32", "amount_per_biomass": "float"})
_growth_requirement = _Record(
    core.GrowthRequirement,
    {
        "solute": "u32",
        "half_saturation": "float",
        "biomass_yield": "float",
    },
)
_growth_model = _Record(
    core.CellGrowthModel,
    {
        "cell_id": "u64",
        "kind": _Enum({"monod": core.GrowthKind.MONOD, "essential": core.GrowthKind.ESSENTIAL}),
        "mu_max": "float",
        "biomass_density": "float",
        "volume_ratio": "float",
        "requirements": _Array(_growth_requirement),
    },
)
_event = _Record(
    core.CultureEvent,
    {
        "time": "float",
        "ports": _Array(_port),
        "reservoirs": _Array(_reservoir),
    },
)
_configuration = _Record(
    core.CultureConfiguration,
    {
        "grid": _grid,
        "fluid": _fluid,
        "ports": _Array(_port),
        "solutes": _Array(_solute),
        "reservoirs": _Array(_reservoir),
        "exchange": _Array(_exchange),
        "biomass_requirements": _Array(_requirement),
        "biomass_per_geometric_volume": "float",
        "solver": _solver,
        "stepping": _stepping,
        "geometry": _geometry,
        "maximum_substeps": "u32",
        "maximum_retries": "u32",
        "growth": _Array(_growth_model),
        "events": _Array(_event),
        "authoring_json": "str",
        "coupling_tolerance": "float",
        "maximum_coupling_iterations": "u32",
    },
)
_body = _Record(
    core.CapsuleBody,
    {
        "id": "u64",
        "position": _Array("float", 3),
        "orientation": _Array("float", 4),
        "length": "float",
        "radius": "float",
        "length_rate": "float",
        "fixed": "bool",
        "force_n": _Array("float", 3),
        "torque_nm": _Array("float", 3),
    },
)
_cell = _Record(
    core.CultureCellState,
    {
        "body": _body,
        "biochemical_volume": "float",
        "species_amounts": _Array("float"),
        "uptake_totals": _Array("float"),
        "realized_specific_rate": "float",
        "biomass_produced": "float",
    },
)
_transfer = _Record(
    core.ChemicalTransfer, {"port": "str", "body_id": "u64", "amounts": _Array("float")}
)
_flow_report = _Record(
    core.FluidSolveReport,
    {
        "iterations": "u32",
        "relative_residual": "float",
        "absolute_residual": "float",
        "divergence_rms_per_s": "float",
        "continuity_rms_per_s": "float",
        "source_volume_rate_m3_s": "float",
        "max_speed_m_s": "float",
        "reynolds_number": "float",
        "viscous_relaxation_time_s": "float",
        "net_flow_rate_m3_s": "float",
        "estimated_memory_bytes": "u64",
    },
)
_transport_report = _Record(
    core.SoluteTransportReport,
    {
        "projection_iterations": "u32",
        "transport_iterations": "u32",
        "maximum_volume_residual": "float",
        "mass_balance_error": _Array("float"),
    },
)
_report = _Record(
    core.CultureReport,
    {"substeps": "u32", "retries": "u32", "flow": _flow_report, "transport": _transport_report},
)
_checkpoint = _Record(
    core.CultureCheckpoint,
    {
        "configuration": _configuration,
        "cells": _Array(_cell),
        "extracellular_amounts": _Array("float"),
        "reservoir_totals": _Array(_transfer),
        "last_report": _report,
        "time": "float",
        "event_index": "u32",
    },
)


# The original version-10 shape is checked before migration. Unknown new fields
# must not be silently overwritten by defaults or bypass the old closed schema.
_v10_solute = _Record(core.Solute, {k: v for k, v in _solute.fields.items() if k != "amount_unit"})
_v10_boundary = _Record(
    core.ChemicalBoundary,
    {k: v for k, v in _reservoir.fields.items() if k not in {"kind", "allow_backflow"}},
)
_v10_configuration = _Record(
    core.CultureConfiguration,
    {
        **{
            k: v
            for k, v in _configuration.fields.items()
            if k
            not in {
                "fluid",
                "growth",
                "events",
                "authoring_json",
                "coupling_tolerance",
                "maximum_coupling_iterations",
            }
        },
        "medium": _fluid,
        "solutes": _Array(_v10_solute),
        "reservoirs": _Array(_v10_boundary),
    },
)
_v10_cell = _Record(
    core.CultureCellState,
    {
        k: v
        for k, v in _cell.fields.items()
        if k not in {"uptake_totals", "realized_specific_rate", "biomass_produced"}
    },
)
_v10_checkpoint = _Record(
    core.CultureCheckpoint,
    {
        **{k: v for k, v in _checkpoint.fields.items() if k not in {"time", "event_index"}},
        "configuration": _v10_configuration,
        "cells": _Array(_v10_cell),
    },
)


def encode_culture_checkpoint(value: core.CultureCheckpoint | None) -> JSONValue:
    return None if value is None else _encode(value, _checkpoint)


def decode_culture_checkpoint(
    value: object, schema_version: int = 11, time: float = 0.0
) -> core.CultureCheckpoint | None:
    if value is not None and schema_version == 10:
        import copy

        _decode(value, _v10_checkpoint, "$.simulation.media_flow", validate_only=True)
        data = copy.deepcopy(cast(dict[str, JSONValue], value))
        config = cast(dict[str, JSONValue], data["configuration"])
        config["fluid"] = config.pop("medium")
        config.update(
            growth=[],
            events=[],
            authoring_json="",
            coupling_tolerance=2e-6,
            maximum_coupling_iterations=64,
        )
        solutes = cast(list[dict[str, JSONValue]], config["solutes"])

        for solute in solutes:
            solute["amount_unit"] = "model"

        for boundary in cast(list[dict[str, JSONValue]], config["reservoirs"]):
            boundary.update(kind="reservoir_contact", allow_backflow=True)

        for cell in cast(list[dict[str, JSONValue]], data["cells"]):
            cell.update(
                uptake_totals=cast(JSONValue, [0.0] * len(solutes)),
                realized_specific_rate=0.0,
                biomass_produced=0.0,
            )

        data.update(time=time, event_index=0)
        value = data

    return (
        None
        if value is None
        else cast(core.CultureCheckpoint, _decode(value, _checkpoint, "$.simulation.culture"))
    )
