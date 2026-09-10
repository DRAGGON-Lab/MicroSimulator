"""Explicit compatibility layer for legacy Python regulation callbacks."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendFeature,
    CellInit,
    CellSnapshot,
    MechanicsParameters,
    MechanicsSolveResult,
    Simulation,
    Vec3,
)
from .checkpoint import JSONValue


class LegacyCompatibilityError(RuntimeError):
    """Raised when a callback requests behavior outside the compatibility contract."""


class LegacyCell:
    """Mutable CellState-shaped view used by legacy callbacks.

    Arbitrary user attributes are intentionally supported. Engine-owned geometry
    is refreshed from the native simulation after every completed step.
    """

    def __init__(self, snapshot: CellSnapshot) -> None:
        self.id = snapshot.id
        self.idx = snapshot.slot
        self.pos = _vector(snapshot.position)
        self.dir = _vector(snapshot.direction)
        self.length = snapshot.length
        self.radius = snapshot.radius
        self.growthRate = snapshot.growth_rate
        self.cellType = snapshot.cell_type
        self.species = list(snapshot.species)
        self.signals: list[float] = []
        self.volume = snapshot.length
        self.oldLen = snapshot.length
        self.strainRate = 0.0
        self.effGrowth = 0.0
        self.cellAge = 0
        self.time = 0.0
        self.neighbours: list[int] = []
        self.cts = 0
        self.divideFlag = False
        self.color: Any = [0.5, 0.5, 0.5]
        self.asymm = [1.0, 1.0]
        self.vel = [0.0, 0.0, 0.0]
        self.ends = _ends(self.pos, self.dir, self.length)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


type InitCallback = Callable[[LegacyCell], None]
type UpdateCallback = Callable[[dict[int, LegacyCell]], None]
type DivideCallback = Callable[[LegacyCell, LegacyCell, LegacyCell], None]


def _vector(value: Any) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _ends(
    position: list[float], direction: list[float], length: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    half = length * 0.5
    return (
        (
            position[0] - direction[0] * half,
            position[1] - direction[1] * half,
            position[2] - direction[2] * half,
        ),
        (
            position[0] + direction[0] * half,
            position[1] + direction[1] * half,
            position[2] + direction[2] * half,
        ),
    )


def _division_fraction(value: object) -> float:
    try:
        weights = list(cast(Any, value))
    except TypeError as error:
        raise LegacyCompatibilityError("legacy asymm must contain two positive weights") from error
    if len(weights) != 2:
        raise LegacyCompatibilityError("legacy asymm must contain two positive weights")
    try:
        first = float(weights[0])
        second = float(weights[1])
    except (TypeError, ValueError, OverflowError) as error:
        raise LegacyCompatibilityError("legacy asymm must contain two positive weights") from error
    total = first + second
    if (
        not math.isfinite(first)
        or not math.isfinite(second)
        or not math.isfinite(total)
        or first <= 0.0
        or second <= 0.0
    ):
        raise LegacyCompatibilityError("legacy asymm must contain two finite positive weights")
    fraction = first / total
    if fraction <= 0.0 or fraction >= 1.0:
        raise LegacyCompatibilityError("legacy asymm weights cannot produce a valid split")
    return fraction


def _encoded(value: object, path: str) -> JSONValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LegacyCompatibilityError(f"{path} must be finite")
        return value
    if isinstance(value, np.generic):
        scalar = cast(Any, value)
        return _encoded(cast(object, scalar.item()), path)
    if isinstance(value, np.ndarray):
        array = cast(Any, value)
        if array.dtype.kind not in "biuf":
            raise LegacyCompatibilityError(f"{path} has unsupported NumPy dtype {array.dtype}")
        if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
            raise LegacyCompatibilityError(f"{path} must contain finite values")
        return {
            "$type": "ndarray",
            "dtype": cast(str, array.dtype.str),
            "shape": cast(list[JSONValue], list(array.shape)),
            "items": _encoded(cast(object, array.tolist()), f"{path}.items"),
        }
    if isinstance(value, list):
        sequence = cast(list[object], value)
        return {
            "$type": "list",
            "items": [
                _encoded(item, f"{path}[{index}]") for index, item in enumerate(sequence)
            ],
        }
    if isinstance(value, tuple):
        sequence = cast(tuple[object, ...], value)
        return {
            "$type": "tuple",
            "items": [
                _encoded(item, f"{path}[{index}]") for index, item in enumerate(sequence)
            ],
        }
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        items: list[JSONValue] = []
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise LegacyCompatibilityError(f"{path} dictionary keys must be strings")
            items.append([key, _encoded(item, f"{path}[{key!r}]")])
        return {"$type": "dict", "items": items}
    raise LegacyCompatibilityError(f"{path} has unsupported value type {type(value).__name__}")


def _decoded(value: JSONValue, path: str) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LegacyCompatibilityError(f"{path} must be finite")
        return value
    if not isinstance(value, dict):
        raise LegacyCompatibilityError(f"{path} is not a tagged legacy value")
    kind = value.get("$type")
    if kind in {"list", "tuple"}:
        if set(value) != {"$type", "items"} or not isinstance(value["items"], list):
            raise LegacyCompatibilityError(f"{path} has an invalid {kind} encoding")
        items = [
            _decoded(item, f"{path}.items[{index}]")
            for index, item in enumerate(value["items"])
        ]
        return items if kind == "list" else tuple(items)
    if kind == "dict":
        if set(value) != {"$type", "items"} or not isinstance(value["items"], list):
            raise LegacyCompatibilityError(f"{path} has an invalid dictionary encoding")
        result: dict[str, Any] = {}
        for index, entry in enumerate(value["items"]):
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or entry[0] in result
            ):
                raise LegacyCompatibilityError(f"{path}.items[{index}] is invalid")
            result[entry[0]] = _decoded(entry[1], f"{path}[{entry[0]!r}]")
        return result
    if kind == "ndarray":
        if set(value) != {"$type", "dtype", "shape", "items"}:
            raise LegacyCompatibilityError(f"{path} has an invalid NumPy encoding")
        dtype_value = value["dtype"]
        shape_value = value["shape"]
        if not isinstance(dtype_value, str) or not isinstance(shape_value, list):
            raise LegacyCompatibilityError(f"{path} has invalid NumPy metadata")
        if not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in shape_value
        ):
            raise LegacyCompatibilityError(f"{path} has an invalid NumPy shape")
        shape = [cast(int, item) for item in shape_value]
        try:
            dtype = np.dtype(dtype_value)
            if dtype.kind not in "biuf":
                raise LegacyCompatibilityError(f"{path} has unsupported NumPy dtype {dtype}")
            array = cast(Any, np.asarray(_decoded(value["items"], f"{path}.items"), dtype=dtype))
            return array.reshape(tuple(shape))
        except (TypeError, ValueError, OverflowError) as error:
            raise LegacyCompatibilityError(f"{path} has invalid NumPy data") from error
    raise LegacyCompatibilityError(f"{path} has an unknown tagged value type")


class LegacyModelAdapter:
    """Drive maintained ``init/update/divide`` callbacks over a native simulation."""

    def __init__(
        self,
        simulation: Simulation,
        *,
        init: InitCallback,
        update: UpdateCallback,
        divide: DivideCallback | None = None,
        mechanics: bool = True,
        compute_neighbors: bool = False,
        division_jitter_z: bool | None = None,
        alternate_divisions: bool = False,
        max_substeps: int = 8,
        rng: random.Random | None = None,
        mechanics_parameters: MechanicsParameters | None = None,
    ) -> None:
        if simulation.cell_count != 0:
            raise LegacyCompatibilityError("legacy adapter requires an empty simulation")
        self._configure(
            simulation,
            init=init,
            update=update,
            divide=divide,
            mechanics=mechanics,
            compute_neighbors=compute_neighbors,
            division_jitter_z=division_jitter_z,
            alternate_divisions=alternate_divisions,
            max_substeps=max_substeps,
            rng=rng,
            mechanics_parameters=mechanics_parameters,
        )
        self._cells: dict[int, LegacyCell] = {}
        self._setup_cell_ids: list[int] = []

    def _configure(
        self,
        simulation: Simulation,
        *,
        init: InitCallback,
        update: UpdateCallback,
        divide: DivideCallback | None,
        mechanics: bool,
        compute_neighbors: bool,
        division_jitter_z: bool | None,
        alternate_divisions: bool,
        max_substeps: int,
        rng: random.Random | None,
        mechanics_parameters: MechanicsParameters | None,
    ) -> None:
        if mechanics and not simulation.supports(BackendFeature.CELL_MECHANICS):
            raise LegacyCompatibilityError("backend does not implement cell mechanics")
        if compute_neighbors and not simulation.supports(BackendFeature.CELL_CONTACTS):
            raise LegacyCompatibilityError("backend does not implement cell contacts")
        if division_jitter_z is not None and rng is None:
            raise LegacyCompatibilityError(
                "legacy division jitter requires an explicit random stream"
            )
        if division_jitter_z is not None and alternate_divisions:
            raise LegacyCompatibilityError(
                "legacy random jitter and alternating division axes are mutually exclusive"
            )
        max_substeps_value = cast(object, max_substeps)
        if (
            not isinstance(max_substeps_value, int)
            or isinstance(max_substeps_value, bool)
            or max_substeps_value < 0
            or max_substeps_value > (1 << 32) - 1
        ):
            raise LegacyCompatibilityError("legacy max_substeps must be an unsigned 32-bit integer")
        self.simulation = simulation
        self._init = init
        self._update = update
        self._divide = divide
        self._mechanics = mechanics
        self._compute_neighbors = compute_neighbors
        self._division_jitter_z = division_jitter_z
        self._alternate_divisions = alternate_divisions
        self._max_substeps = max_substeps_value
        self._rng = rng
        self._mechanics_parameters = mechanics_parameters or MechanicsParameters()
        self._last_mechanics_reports: tuple[MechanicsSolveResult, ...] = ()

    @property
    def last_mechanics_reports(self) -> tuple[MechanicsSolveResult, ...]:
        """Return reports from the most recent contact-frontier relaxation."""

        return self._last_mechanics_reports

    def controller_state(self) -> dict[str, JSONValue]:
        """Return complete data-only callback and random-stream state."""

        snapshots = self.simulation.cells()
        if {snapshot.id for snapshot in snapshots} != self._cells.keys():
            raise LegacyCompatibilityError("legacy cells and native simulation identities disagree")
        cells: list[JSONValue] = []
        for snapshot in snapshots:
            cell = self._cells[snapshot.id]
            attributes = {
                name: _encoded(value, f"legacy cell {snapshot.id}.{name}")
                for name, value in vars(cell).items()
            }
            cells.append({"id": snapshot.id, "attributes": attributes})
        parameters = self._mechanics_parameters
        return {
            "kind": "microsimulator-legacy-python",
            "version": 4,
            "options": {
                "mechanics": self._mechanics,
                "compute_neighbors": self._compute_neighbors,
                "division_jitter_z": self._division_jitter_z,
                "alternate_divisions": self._alternate_divisions,
                "max_substeps": self._max_substeps,
                "mechanics_parameters": {
                    "mu_a": parameters.mu_a,
                    "gamma": parameters.gamma,
                    "residual_rms_tolerance": parameters.residual_rms_tolerance,
                    "max_iterations": parameters.max_iterations,
                },
            },
            "random_state": _encoded(self._rng.getstate(), "legacy random state")
            if self._rng is not None
            else None,
            "setup_cell_ids": list(self._setup_cell_ids),
            "cells": cells,
        }

    @classmethod
    def from_controller_state(
        cls,
        simulation: Simulation,
        controller: JSONValue,
        *,
        init: InitCallback,
        update: UpdateCallback,
        divide: DivideCallback | None = None,
        rng: random.Random | None = None,
    ) -> LegacyModelAdapter:
        """Restore callback state onto an already-restored native simulation."""

        data = cls._controller_object(controller, "controller")
        if set(data) != {
            "kind",
            "version",
            "options",
            "random_state",
            "setup_cell_ids",
            "cells",
        }:
            raise LegacyCompatibilityError("legacy controller has unexpected fields")
        version = data["version"]
        if (
            data["kind"] not in ("microsimulator-legacy-python", "cellmodeller2-legacy-python")
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version not in (2, 3, 4)
        ):
            raise LegacyCompatibilityError("legacy controller kind or version is unsupported")
        options = cls._controller_object(data["options"], "controller.options")
        expected_options = {
            "mechanics",
            "compute_neighbors",
            "division_jitter_z",
            "mechanics_parameters",
        }
        if version >= 3:
            expected_options.add("alternate_divisions")
        if version >= 4:
            expected_options.add("max_substeps")
        if set(options) != expected_options:
            raise LegacyCompatibilityError("legacy controller options are invalid")
        mechanics = options["mechanics"]
        compute_neighbors = options["compute_neighbors"]
        division_jitter_z = options["division_jitter_z"]
        alternate_divisions = options["alternate_divisions"] if version >= 3 else False
        max_substeps = options["max_substeps"] if version >= 4 else 2
        if not isinstance(mechanics, bool) or not isinstance(compute_neighbors, bool):
            raise LegacyCompatibilityError("legacy controller Boolean options are invalid")
        if division_jitter_z is not None and not isinstance(division_jitter_z, bool):
            raise LegacyCompatibilityError("legacy division jitter option is invalid")
        if not isinstance(alternate_divisions, bool):
            raise LegacyCompatibilityError("legacy alternating division option is invalid")
        if (
            not isinstance(max_substeps, int)
            or isinstance(max_substeps, bool)
            or max_substeps < 0
            or max_substeps > (1 << 32) - 1
        ):
            raise LegacyCompatibilityError("legacy max_substeps option is invalid")
        mechanics_data = cls._controller_object(
            options["mechanics_parameters"], "controller.options.mechanics_parameters"
        )
        if set(mechanics_data) != {
            "mu_a",
            "gamma",
            "residual_rms_tolerance",
            "max_iterations",
        }:
            raise LegacyCompatibilityError("legacy mechanics parameters are invalid")
        parameters = MechanicsParameters()
        try:
            parameters.mu_a = float(cast(Any, mechanics_data["mu_a"]))
            parameters.gamma = float(cast(Any, mechanics_data["gamma"]))
            parameters.residual_rms_tolerance = float(
                cast(Any, mechanics_data["residual_rms_tolerance"])
            )
            parameters.max_iterations = int(cast(Any, mechanics_data["max_iterations"]))
        except (TypeError, ValueError, OverflowError) as error:
            raise LegacyCompatibilityError("legacy mechanics parameters are invalid") from error

        random_state = data["random_state"]
        restored_rng = rng
        if random_state is not None:
            restored_rng = restored_rng or random.Random()
            decoded_random_state = _decoded(random_state, "controller.random_state")
            if not isinstance(decoded_random_state, tuple):
                raise LegacyCompatibilityError("legacy random state is invalid")
            try:
                restored_rng.setstate(cast(tuple[Any, ...], decoded_random_state))
            except (TypeError, ValueError) as error:
                raise LegacyCompatibilityError("legacy random state is invalid") from error

        instance = cls.__new__(cls)
        instance._configure(
            simulation,
            init=init,
            update=update,
            divide=divide,
            mechanics=mechanics,
            compute_neighbors=compute_neighbors,
            division_jitter_z=division_jitter_z,
            alternate_divisions=alternate_divisions,
            max_substeps=max_substeps,
            rng=restored_rng,
            mechanics_parameters=parameters,
        )
        setup_cell_ids = data["setup_cell_ids"]
        if not isinstance(setup_cell_ids, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in setup_cell_ids
        ):
            raise LegacyCompatibilityError("legacy setup cell identifiers are invalid")
        instance._setup_cell_ids = [cast(int, item) for item in setup_cell_ids]
        instance._cells = instance._restore_cells(data["cells"])
        return instance

    @staticmethod
    def _controller_object(value: JSONValue, path: str) -> dict[str, JSONValue]:
        if not isinstance(value, dict):
            raise LegacyCompatibilityError(f"{path} must be an object")
        return value

    def _restore_cells(self, value: JSONValue) -> dict[int, LegacyCell]:
        if not isinstance(value, list):
            raise LegacyCompatibilityError("controller.cells must be an array")
        records: dict[int, dict[str, JSONValue]] = {}
        for index, item in enumerate(value):
            record = self._controller_object(item, f"controller.cells[{index}]")
            if set(record) != {"id", "attributes"}:
                raise LegacyCompatibilityError(f"controller.cells[{index}] is invalid")
            cell_id = record["id"]
            attributes = record["attributes"]
            if (
                not isinstance(cell_id, int)
                or isinstance(cell_id, bool)
                or cell_id <= 0
                or cell_id in records
                or not isinstance(attributes, dict)
            ):
                raise LegacyCompatibilityError(f"controller.cells[{index}] is invalid")
            records[cell_id] = attributes
        snapshots = self.simulation.cells()
        if set(records) != {snapshot.id for snapshot in snapshots}:
            raise LegacyCompatibilityError("legacy controller and native cell identities disagree")

        cells: dict[int, LegacyCell] = {}
        for snapshot in snapshots:
            decoded_attributes = {
                name: _decoded(item, f"controller.cells[{snapshot.id}].attributes.{name}")
                for name, item in records[snapshot.id].items()
            }
            cell = LegacyCell(snapshot)
            vars(cell).clear()
            vars(cell).update(decoded_attributes)
            if cell.id != snapshot.id or cell.idx != snapshot.slot:
                raise LegacyCompatibilityError("legacy controller cell identity is invalid")
            self._validate_engine_owned_geometry(cell, snapshot)
            self._validate_mutable_state(cell)
            if (
                float(cell.growthRate) != snapshot.growth_rate
                or int(cell.cellType) != snapshot.cell_type
                or (
                    self.simulation.species_count != 0
                    and [float(item) for item in cell.species] != list(snapshot.species)
                )
            ):
                raise LegacyCompatibilityError(
                    "legacy controller mutable state disagrees with native state"
                )
            cells[snapshot.id] = cell
        return cells

    @property
    def cells(self) -> Mapping[int, LegacyCell]:
        return MappingProxyType(self._cells)

    def add_cell(self, cell: CellInit) -> int:
        """Add a native cell, then run the legacy ``init(cell)`` callback."""

        cell_id = self.simulation.add_cell(cell)
        legacy_cell = LegacyCell(self.simulation.cell(cell_id))
        self._init(legacy_cell)
        self._validate_engine_owned_geometry(legacy_cell, self.simulation.cell(cell_id))
        self._apply_mutable_state(legacy_cell)
        self._cells[cell_id] = legacy_cell
        return cell_id

    def step(self, dt: float) -> None:
        """Run regulation, division, native integration, mechanics, and state refresh."""

        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("time step must be finite and non-negative")

        self._update(self._cells)
        snapshots = {snapshot.id: snapshot for snapshot in self.simulation.cells()}
        for cell_id, cell in self._cells.items():
            self._validate_engine_owned_geometry(cell, snapshots[cell_id])
            self._validate_mutable_state(cell)
        for cell in self._cells.values():
            self._apply_mutable_state(cell)
            cell.time = self.simulation.time

        dividing = [cell_id for cell_id, cell in self._cells.items() if cell.divideFlag]
        for parent_id in dividing:
            self._divide_cell(parent_id)

        self.simulation.step(dt)
        self._last_mechanics_reports = ()
        if self._mechanics and self.simulation.cell_count != 0:
            self._last_mechanics_reports = self._relax_new_contact_frontier()
        self._refresh_cells()

    def _relax_new_contact_frontier(self) -> tuple[MechanicsSolveResult, ...]:
        seen: set[tuple[object, ...]] = set()
        reports: list[MechanicsSolveResult] = []
        for _ in range(max(0, self._max_substeps - 1)):
            cell_contacts = self.simulation.find_cell_contacts().contacts
            external_contacts = self.simulation.find_external_contacts().contacts
            current: set[tuple[object, ...]] = {
                ("cell", contact.first_id, contact.second_id, contact.ordinal)
                for contact in cell_contacts
            }
            current.update(
                (
                    "constraint",
                    contact.cell_id,
                    contact.constraint_id,
                    contact.constraint_kind,
                    contact.endpoint,
                )
                for contact in external_contacts
            )
            if not current.difference(seen):
                break
            seen.update(current)
            reports.append(self.simulation.relax_cell_mechanics(self._mechanics_parameters))
        return tuple(reports)

    def _divide_cell(self, parent_id: int) -> None:
        parent = self._cells[parent_id]
        first_fraction = _division_fraction(parent.asymm)
        parent.divideFlag = False
        first_id, second_id = self.simulation.divide(parent_id, first_fraction)
        self._apply_division_orientation(first_id)
        self._apply_division_orientation(second_id)
        first = copy.deepcopy(parent)
        second = copy.deepcopy(parent)
        first.cellAge = 0
        second.cellAge = 0
        self._set_identity(first, self.simulation.cell(first_id))
        self._set_identity(second, self.simulation.cell(second_id))
        if self._divide is not None:
            self._divide(parent, first, second)
        self._validate_engine_owned_geometry(first, self.simulation.cell(first_id))
        self._validate_engine_owned_geometry(second, self.simulation.cell(second_id))
        self._validate_mutable_state(first)
        self._validate_mutable_state(second)
        self._apply_mutable_state(first)
        self._apply_mutable_state(second)
        del self._cells[parent_id]
        self._cells[first_id] = first
        self._cells[second_id] = second

    def _apply_division_orientation(self, cell_id: int) -> None:
        snapshot = self.simulation.cell(cell_id)
        if self._alternate_divisions:
            direction = Vec3(
                -snapshot.direction.y,
                snapshot.direction.x,
                snapshot.direction.z,
            )
            self.simulation.set_cell_geometry(
                cell_id, snapshot.position, direction, snapshot.length
            )
            return
        if self._division_jitter_z is None:
            return
        if self._rng is None:
            raise AssertionError("division jitter random stream is missing")
        jitter = [self._rng.uniform(-0.001, 0.001) for _ in range(3)]
        if not self._division_jitter_z:
            jitter[2] = 0.0
        direction = Vec3(
            snapshot.direction.x + jitter[0],
            snapshot.direction.y + jitter[1],
            snapshot.direction.z + jitter[2],
        )
        self.simulation.set_cell_geometry(cell_id, snapshot.position, direction, snapshot.length)

    @staticmethod
    def _set_identity(cell: LegacyCell, snapshot: CellSnapshot) -> None:
        cell.id = snapshot.id
        cell.idx = snapshot.slot
        cell.pos = _vector(snapshot.position)
        cell.dir = _vector(snapshot.direction)
        cell.length = snapshot.length
        cell.volume = snapshot.length
        cell.oldLen = snapshot.length
        cell.ends = _ends(cell.pos, cell.dir, cell.length)

    @staticmethod
    def _validate_engine_owned_geometry(cell: LegacyCell, snapshot: CellSnapshot) -> None:
        actual = (
            tuple(float(value) for value in cell.pos),
            tuple(float(value) for value in cell.dir),
            float(cell.length),
            float(cell.radius),
        )
        expected = (
            tuple(_vector(snapshot.position)),
            tuple(_vector(snapshot.direction)),
            snapshot.length,
            snapshot.radius,
        )
        if actual != expected:
            raise LegacyCompatibilityError(
                "legacy callbacks may not mutate native position, direction, length, or radius"
            )

    def _validate_mutable_state(self, cell: LegacyCell) -> None:
        growth_rate = float(cell.growthRate)
        if not math.isfinite(growth_rate):
            raise LegacyCompatibilityError("legacy growthRate must be finite")
        cell_type = cast(object, cell.cellType)
        if not isinstance(cell_type, int):
            raise LegacyCompatibilityError("legacy cellType must be an integer")
        if (
            self.simulation.species_count != 0
            and len(cell.species) != self.simulation.species_count
        ):
            raise LegacyCompatibilityError("legacy species count does not match the simulation")
        if not all(math.isfinite(float(value)) for value in cell.species):
            raise LegacyCompatibilityError("legacy species levels must be finite")

    def _apply_mutable_state(self, cell: LegacyCell) -> None:
        self._validate_mutable_state(cell)
        self.simulation.set_cell_attributes(cell.id, float(cell.growthRate), int(cell.cellType))
        if self.simulation.species_count != 0:
            self.simulation.set_species(cell.id, [float(value) for value in cell.species])

    def _refresh_cells(self) -> None:
        graph = self.simulation.find_cell_contacts() if self._compute_neighbors else None
        for snapshot in self.simulation.cells():
            cell = self._cells[snapshot.id]
            previous_position = cell.pos
            previous_length = cell.oldLen
            cell.idx = snapshot.slot
            cell.pos = _vector(snapshot.position)
            cell.dir = _vector(snapshot.direction)
            cell.length = snapshot.length
            cell.radius = snapshot.radius
            cell.growthRate = snapshot.growth_rate
            cell.cellType = snapshot.cell_type
            if self.simulation.species_count != 0:
                cell.species = list(snapshot.species)
            cell.signals = (
                self.simulation.sample_signals(snapshot.position)
                if self.simulation.has_signal_grid
                else []
            )
            cell.vel = [cell.pos[index] - previous_position[index] for index in range(3)]
            cell.strainRate = (
                (cell.length - previous_length) / previous_length if previous_length != 0.0 else 0.0
            )
            cell.effGrowth = (
                cell.effGrowth * cell.cellAge + cell.strainRate * previous_length
            ) / (cell.cellAge + 1)
            cell.cellAge += 1
            cell.oldLen = cell.length
            cell.volume = cell.length
            cell.ends = _ends(cell.pos, cell.dir, cell.length)
            cell.neighbours = list(graph.neighbor_ids(snapshot.slot)) if graph is not None else []
            cell.cts = len(cell.neighbours)
            cell.divideFlag = False
