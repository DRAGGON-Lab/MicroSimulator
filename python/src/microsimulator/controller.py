"""Restartable Python controller contract for native simulations."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Protocol, cast, runtime_checkable

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    CellSnapshot,
    ConstraintContactParameters,
    ContactParameters,
    MechanicsIntegrationParameters,
    MechanicsParameters,
    MechanicsSolveResult,
    Simulation,
)
from .checkpoint import CheckpointBundle, JSONValue

_RANDOM_STATE_KIND = "python-random-mt19937"
_RANDOM_STATE_VERSION = 1
_MT_STATE_WORDS = 624
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_FLOAT32_MAX = 3.4028234663852886e38
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_NATIVE_CONTROLLER_KIND = "microsimulator-native-controller"
_NATIVE_CONTROLLER_VERSION = 1


class ControllerStateError(ValueError):
    """Raised when persisted controller state is malformed or unsupported."""


class ControllerPlanError(ValueError):
    """Raised when a native controller returns an invalid step plan."""


@runtime_checkable
class SimulationController(Protocol):
    """Structural contract for Python orchestration around a native simulation.

    The controller owns runtime policy and must return all state needed by its
    model module's ``resume(context, checkpoint)`` function as finite JSON.
    """

    @property
    def simulation(self) -> Simulation:
        """Return the native simulation that owns checkpointed engine state."""

        ...

    def step(self, dt: float) -> None:
        """Advance exactly one biological step."""

        ...

    def controller_state(self) -> JSONValue:
        """Return complete non-null data-only state needed for exact resume."""

        ...


@dataclass(frozen=True, slots=True)
class CellUpdate:
    """Atomic mutable-state update for one cell before native integration."""

    cell_id: int
    growth_rate: float | None = None
    cell_type: int | None = None
    fixed: bool | None = None
    species: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class DivisionRequest:
    """Deterministic parent and axial fraction for one topology event."""

    parent_id: int
    first_fraction: float = 0.5


@dataclass(frozen=True, slots=True)
class StepPlan:
    """Complete host-side mutations to apply before one native step."""

    updates: tuple[CellUpdate, ...] = ()
    divisions: tuple[DivisionRequest, ...] = ()


@dataclass(frozen=True, slots=True)
class ControllerStep:
    """Read view plus explicit mutable model state and random stream."""

    simulation: Simulation
    cells: tuple[CellSnapshot, ...]
    completed_steps: int
    time: float
    rng: random.Random
    state: dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class DivisionEvent:
    """Native identities and snapshots produced by a division request."""

    parent: CellSnapshot
    first: CellSnapshot
    second: CellSnapshot


type RegulationCallback = Callable[[ControllerStep], StepPlan]
type DivisionCallback = Callable[[ControllerStep, DivisionEvent], None]


def _finite_number(value: object, path: str) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or abs(value) > _FLOAT32_MAX
    ):
        raise ControllerStateError(f"{path} must be a finite float32 value")
    return float(value)


def _integer(value: object, path: str, lower: int, upper: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < lower or value > upper:
        raise ControllerStateError(f"{path} must be an integer in [{lower}, {upper}]")
    return value


def _plan_number(value: object, path: str) -> float:
    try:
        return _finite_number(value, path)
    except ControllerStateError as error:
        raise ControllerPlanError(str(error)) from error


def _plan_integer(value: object, path: str, lower: int, upper: int) -> int:
    try:
        return _integer(value, path, lower, upper)
    except ControllerStateError as error:
        raise ControllerPlanError(str(error)) from error


def _json_object(value: Mapping[str, JSONValue], path: str) -> dict[str, JSONValue]:
    import json

    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = cast(object, json.loads(encoded))
    except (TypeError, ValueError, RecursionError) as error:
        raise ControllerStateError(f"{path} must be finite JSON data") from error
    if not isinstance(decoded, dict):
        raise ControllerStateError(f"{path} must be a JSON object")
    return cast(dict[str, JSONValue], decoded)


@dataclass(frozen=True, slots=True)
class MechanicsConfig:
    """Checkpointable parameters for exact mechanics passes after each step."""

    passes: int = 1
    mu_a: float = 1.0
    gamma: float = 10.0
    residual_rms_tolerance: float = 5.0e-3
    max_iterations: int = 0
    contact_activation_margin: float = 0.01
    contact_parallel_sine_threshold: float = 0.1
    contact_degeneracy_epsilon: float = 1.0e-6
    max_rotation_radians: float = 0.0872664626
    require_convergence: bool = True
    constraint_activation_margin: float = 0.0
    constraint_degeneracy_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        _integer(self.passes, "mechanics.passes", 1, _UINT32_MAX)
        _integer(self.max_iterations, "mechanics.max_iterations", 0, _UINT32_MAX)
        mu_a = _finite_number(self.mu_a, "mechanics.mu_a")
        gamma = _finite_number(self.gamma, "mechanics.gamma")
        tolerance = _finite_number(self.residual_rms_tolerance, "mechanics.residual_rms_tolerance")
        contact_margin = _finite_number(
            self.contact_activation_margin, "mechanics.contact_activation_margin"
        )
        parallel_threshold = _finite_number(
            self.contact_parallel_sine_threshold,
            "mechanics.contact_parallel_sine_threshold",
        )
        contact_epsilon = _finite_number(
            self.contact_degeneracy_epsilon, "mechanics.contact_degeneracy_epsilon"
        )
        rotation = _finite_number(self.max_rotation_radians, "mechanics.max_rotation_radians")
        constraint_margin = _finite_number(
            self.constraint_activation_margin,
            "mechanics.constraint_activation_margin",
        )
        constraint_epsilon = _finite_number(
            self.constraint_degeneracy_epsilon,
            "mechanics.constraint_degeneracy_epsilon",
        )
        if mu_a <= 0.0 or gamma <= 0.0:
            raise ControllerStateError("mechanics mu_a and gamma must be positive")
        if tolerance < 0.0 or contact_margin < 0.0 or rotation < 0.0:
            raise ControllerStateError("mechanics tolerances, margins, and limits are invalid")
        if parallel_threshold < 0.0 or parallel_threshold > 1.0:
            raise ControllerStateError("mechanics contact parallel threshold is invalid")
        if contact_epsilon <= 0.0 or constraint_epsilon <= 0.0 or constraint_margin < 0.0:
            raise ControllerStateError("mechanics constraint/contact parameters are invalid")
        if not isinstance(cast(object, self.require_convergence), bool):
            raise ControllerStateError("mechanics.require_convergence must be Boolean")

    def native_parameters(
        self,
    ) -> tuple[
        MechanicsParameters,
        ContactParameters,
        MechanicsIntegrationParameters,
        ConstraintContactParameters,
    ]:
        mechanics = MechanicsParameters()
        mechanics.mu_a = self.mu_a
        mechanics.gamma = self.gamma
        mechanics.residual_rms_tolerance = self.residual_rms_tolerance
        mechanics.max_iterations = self.max_iterations
        contacts = ContactParameters()
        contacts.activation_margin = self.contact_activation_margin
        contacts.parallel_sine_threshold = self.contact_parallel_sine_threshold
        contacts.degeneracy_epsilon = self.contact_degeneracy_epsilon
        integration = MechanicsIntegrationParameters()
        integration.max_rotation_radians = self.max_rotation_radians
        integration.require_convergence = self.require_convergence
        constraints = ConstraintContactParameters()
        constraints.activation_margin = self.constraint_activation_margin
        constraints.degeneracy_epsilon = self.constraint_degeneracy_epsilon
        return mechanics, contacts, integration, constraints

    def to_json(self) -> dict[str, JSONValue]:
        return {
            field.name: cast(JSONValue, getattr(self, field.name))
            for field in fields(MechanicsConfig)
        }

    @classmethod
    def from_json(cls, value: JSONValue) -> MechanicsConfig:
        if not isinstance(value, dict) or set(value) != {
            field.name for field in fields(MechanicsConfig)
        }:
            raise ControllerStateError("controller mechanics configuration is invalid")
        require_convergence = value["require_convergence"]
        if not isinstance(require_convergence, bool):
            raise ControllerStateError("mechanics.require_convergence must be Boolean")
        return cls(
            passes=_integer(value["passes"], "mechanics.passes", 1, _UINT32_MAX),
            mu_a=_finite_number(value["mu_a"], "mechanics.mu_a"),
            gamma=_finite_number(value["gamma"], "mechanics.gamma"),
            residual_rms_tolerance=_finite_number(
                value["residual_rms_tolerance"], "mechanics.residual_rms_tolerance"
            ),
            max_iterations=_integer(
                value["max_iterations"], "mechanics.max_iterations", 0, _UINT32_MAX
            ),
            contact_activation_margin=_finite_number(
                value["contact_activation_margin"],
                "mechanics.contact_activation_margin",
            ),
            contact_parallel_sine_threshold=_finite_number(
                value["contact_parallel_sine_threshold"],
                "mechanics.contact_parallel_sine_threshold",
            ),
            contact_degeneracy_epsilon=_finite_number(
                value["contact_degeneracy_epsilon"],
                "mechanics.contact_degeneracy_epsilon",
            ),
            max_rotation_radians=_finite_number(
                value["max_rotation_radians"], "mechanics.max_rotation_radians"
            ),
            require_convergence=require_convergence,
            constraint_activation_margin=_finite_number(
                value["constraint_activation_margin"],
                "mechanics.constraint_activation_margin",
            ),
            constraint_degeneracy_epsilon=_finite_number(
                value["constraint_degeneracy_epsilon"],
                "mechanics.constraint_degeneracy_epsilon",
            ),
        )


def _model_identity(model_id: object, model_version: object) -> tuple[str, int]:
    if not isinstance(model_id, str) or _MODEL_ID.fullmatch(model_id) is None:
        raise ControllerStateError("native controller model ID is invalid")
    return model_id, _integer(model_version, "native controller model version", 1, _UINT32_MAX)


class NativeController:
    """Typed deterministic orchestration over a native simulation."""

    def __init__(
        self,
        simulation: Simulation,
        *,
        model_id: str,
        model_version: int,
        rng: random.Random,
        regulate: RegulationCallback | None = None,
        on_division: DivisionCallback | None = None,
        mechanics: MechanicsConfig | None = None,
        state: Mapping[str, JSONValue] | None = None,
        completed_steps: int = 0,
    ) -> None:
        self._model_id, self._model_version = _model_identity(model_id, model_version)
        if not isinstance(cast(object, simulation), Simulation):
            raise TypeError("native controller simulation must be a Simulation")
        if not isinstance(cast(object, rng), random.Random):
            raise TypeError("native controller requires an explicit random.Random stream")
        self.simulation = simulation
        self._rng = rng
        self._regulate = regulate
        self._on_division = on_division
        self._mechanics = mechanics
        self._state = _json_object(state or {}, "native controller state")
        self._completed_steps = _integer(
            completed_steps, "native controller completed steps", 0, _UINT64_MAX
        )
        self._last_mechanics_reports: tuple[MechanicsSolveResult, ...] = ()

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def state(self) -> dict[str, JSONValue]:
        return self._state

    @property
    def last_mechanics_reports(self) -> tuple[MechanicsSolveResult, ...]:
        return self._last_mechanics_reports

    def _context(self) -> ControllerStep:
        return ControllerStep(
            simulation=self.simulation,
            cells=tuple(self.simulation.cells()),
            completed_steps=self._completed_steps,
            time=self.simulation.time,
            rng=self._rng,
            state=self._state,
        )

    def _validate_plan(self, plan: object) -> StepPlan:
        if not isinstance(plan, StepPlan):
            raise ControllerPlanError("regulation callback must return a StepPlan")
        snapshots = {cell.id: cell for cell in self.simulation.cells()}
        updated: set[int] = set()
        updates_value = cast(object, plan.updates)
        if not isinstance(updates_value, tuple):
            raise ControllerPlanError("step plan updates must be a tuple")
        updates = cast(tuple[object, ...], updates_value)
        for update_value in updates:
            if not isinstance(update_value, CellUpdate):
                raise ControllerPlanError("step plan contains an invalid cell update")
            update = update_value
            cell_id = _plan_integer(update.cell_id, "cell update ID", 1, _UINT64_MAX)
            if cell_id not in snapshots or cell_id in updated:
                raise ControllerPlanError("step plan updates an unknown or duplicate cell")
            updated.add(cell_id)
            if update.growth_rate is not None:
                _plan_number(update.growth_rate, "cell update growth rate")
            if update.cell_type is not None:
                _plan_integer(update.cell_type, "cell update cell type", _INT32_MIN, _INT32_MAX)
            if update.fixed is not None and not isinstance(cast(object, update.fixed), bool):
                raise ControllerPlanError("cell update fixed value must be Boolean")
            if update.species is not None:
                species_value = cast(object, update.species)
                if not isinstance(species_value, tuple):
                    raise ControllerPlanError("cell update species shape is invalid")
                species = cast(tuple[object, ...], species_value)
                if len(species) != self.simulation.species_count:
                    raise ControllerPlanError("cell update species shape is invalid")
                for value in species:
                    _plan_number(value, "cell update species value")
        dividing: set[int] = set()
        divisions_value = cast(object, plan.divisions)
        if not isinstance(divisions_value, tuple):
            raise ControllerPlanError("step plan divisions must be a tuple")
        divisions = cast(tuple[object, ...], divisions_value)
        for request_value in divisions:
            if not isinstance(request_value, DivisionRequest):
                raise ControllerPlanError("step plan contains an invalid division request")
            request = request_value
            parent_id = _plan_integer(request.parent_id, "division parent ID", 1, _UINT64_MAX)
            parent = snapshots.get(parent_id)
            if parent is None or parent_id in dividing:
                raise ControllerPlanError("step plan divides an unknown or duplicate parent")
            dividing.add(parent_id)
            fraction = _plan_number(request.first_fraction, "division fraction")
            if fraction <= 0.0 or fraction >= 1.0:
                raise ControllerPlanError("division fraction must be strictly between zero and one")
            if parent.length < 2.0 * parent.radius:
                raise ControllerPlanError("division parent is shorter than its cap diameter")
        return plan

    def _apply_update(self, update: CellUpdate) -> None:
        current = self.simulation.cell(update.cell_id)
        self.simulation.set_cell_attributes(
            update.cell_id,
            current.growth_rate if update.growth_rate is None else update.growth_rate,
            current.cell_type if update.cell_type is None else update.cell_type,
        )
        if update.fixed is not None:
            self.simulation.set_cell_fixed(update.cell_id, update.fixed)
        if update.species is not None:
            self.simulation.set_species(update.cell_id, list(update.species))

    def step(self, dt: float) -> None:
        """Apply regulation, division, integration, and configured mechanics."""

        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("time step must be finite and non-negative")
        if self._completed_steps == _UINT64_MAX:
            raise ControllerPlanError("native controller completed-step counter is exhausted")
        plan = StepPlan() if self._regulate is None else self._regulate(self._context())
        plan = self._validate_plan(plan)
        for update in plan.updates:
            self._apply_update(update)
        for request in plan.divisions:
            parent = self.simulation.cell(request.parent_id)
            first_id, second_id = self.simulation.divide(request.parent_id, request.first_fraction)
            if self._on_division is not None:
                event = DivisionEvent(
                    parent=parent,
                    first=self.simulation.cell(first_id),
                    second=self.simulation.cell(second_id),
                )
                self._on_division(self._context(), event)

        self.simulation.step(dt)
        reports: list[MechanicsSolveResult] = []
        if self._mechanics is not None and self.simulation.cell_count != 0:
            parameters = self._mechanics.native_parameters()
            for _ in range(self._mechanics.passes):
                reports.append(self.simulation.relax_cell_mechanics(*parameters))
        self._last_mechanics_reports = tuple(reports)
        self._completed_steps += 1

    def controller_state(self) -> dict[str, JSONValue]:
        """Return the closed native-controller checkpoint payload."""

        return {
            "kind": _NATIVE_CONTROLLER_KIND,
            "version": _NATIVE_CONTROLLER_VERSION,
            "model": {"id": self._model_id, "version": self._model_version},
            "completed_steps": self._completed_steps,
            "random": capture_random_state(self._rng),
            "state": _json_object(self._state, "native controller state"),
            "mechanics": self._mechanics.to_json() if self._mechanics is not None else None,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: CheckpointBundle,
        *,
        model_id: str,
        model_version: int,
        regulate: RegulationCallback | None = None,
        on_division: DivisionCallback | None = None,
    ) -> NativeController:
        """Restore the standard controller state around checkpoint native state."""

        expected_identity = _model_identity(model_id, model_version)
        value = checkpoint.controller
        if not isinstance(value, dict) or set(value) != {
            "kind",
            "version",
            "model",
            "completed_steps",
            "random",
            "state",
            "mechanics",
        }:
            raise ControllerStateError("native controller state is invalid")
        if (
            value["kind"] not in (_NATIVE_CONTROLLER_KIND, "cellmodeller2-native-controller")
            or value["version"] != _NATIVE_CONTROLLER_VERSION
        ):
            raise ControllerStateError("native controller kind or version is unsupported")
        identity = value["model"]
        if not isinstance(identity, dict) or set(identity) != {"id", "version"}:
            raise ControllerStateError("native controller model identity is invalid")
        actual_identity = _model_identity(identity["id"], identity["version"])
        if actual_identity != expected_identity:
            raise ControllerStateError("native controller model identity does not match")
        state = value["state"]
        if not isinstance(state, dict):
            raise ControllerStateError("native controller model state must be an object")
        mechanics_value = value["mechanics"]
        mechanics = None if mechanics_value is None else MechanicsConfig.from_json(mechanics_value)
        return cls(
            checkpoint.simulation,
            model_id=model_id,
            model_version=model_version,
            rng=restore_random_state(value["random"]),
            regulate=regulate,
            on_division=on_division,
            mechanics=mechanics,
            state=state,
            completed_steps=_integer(
                value["completed_steps"],
                "native controller completed steps",
                0,
                _UINT64_MAX,
            ),
        )


def capture_random_state(stream: random.Random) -> dict[str, JSONValue]:
    """Encode a dedicated Python random stream as closed-schema JSON data."""

    state_version, internal_state, gaussian = stream.getstate()
    if state_version != 3 or len(internal_state) != _MT_STATE_WORDS + 1:
        raise ControllerStateError("Python random stream uses an unsupported state format")
    if gaussian is not None and not math.isfinite(gaussian):
        raise ControllerStateError("Python random stream has a non-finite Gaussian cache")
    return {
        "kind": _RANDOM_STATE_KIND,
        "version": _RANDOM_STATE_VERSION,
        "state_version": state_version,
        "state": list(internal_state),
        "gauss_next": gaussian,
    }


def restore_random_state(value: JSONValue) -> random.Random:
    """Restore a random stream produced by :func:`capture_random_state`."""

    if not isinstance(value, dict):
        raise ControllerStateError("random state must be an object")
    if set(value) != {"kind", "version", "state_version", "state", "gauss_next"}:
        raise ControllerStateError("random state has unexpected fields")
    if value["kind"] != _RANDOM_STATE_KIND or value["version"] != _RANDOM_STATE_VERSION:
        raise ControllerStateError("random state kind or version is unsupported")
    if value["state_version"] != 3:
        raise ControllerStateError("Python random state version is unsupported")
    words = value["state"]
    if not isinstance(words, list) or len(words) != _MT_STATE_WORDS + 1:
        raise ControllerStateError("random state vector is invalid")
    for index, word in enumerate(words):
        upper = _MT_STATE_WORDS if index == _MT_STATE_WORDS else _UINT32_MAX
        if not isinstance(word, int) or isinstance(word, bool) or word < 0 or word > upper:
            raise ControllerStateError("random state vector is invalid")
    gaussian = value["gauss_next"]
    if gaussian is not None and (
        not isinstance(gaussian, int | float)
        or isinstance(gaussian, bool)
        or not math.isfinite(gaussian)
    ):
        raise ControllerStateError("random state Gaussian cache is invalid")

    stream = random.Random()
    try:
        stream.setstate(
            (
                3,
                tuple(cast(list[int], words)),
                float(gaussian) if gaussian is not None else None,
            )
        )
    except (TypeError, ValueError) as error:
        raise ControllerStateError("random state is invalid") from error
    return stream
