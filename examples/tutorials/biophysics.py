"""Native ports of the CellModeller biophysics and simple-growth tutorials.

Select a lesson with ``--parameter scenario='"basics"'``.  The supported
scenarios are ``basics``, ``two_types``, ``short_cells``, ``competition``,
and ``box``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from microsimulator import (
    CellInit,
    CellUpdate,
    ControllerStep,
    DivisionEvent,
    DivisionRequest,
    MechanicsConfig,
    ModelContext,
    NativeController,
    PlaneConstraintInit,
    Simulation,
    StepPlan,
    Vec3,
)
from microsimulator.checkpoint import CheckpointBundle, JSONValue

MODEL_ID = "tutorials.biophysics"
MODEL_VERSION = 1
_SCENARIOS = frozenset({"basics", "two_types", "short_cells", "competition", "box"})


def _scenario(parameters: Mapping[str, JSONValue]) -> str:
    value = parameters.get("scenario", "basics")
    if not isinstance(value, str) or value not in _SCENARIOS:
        raise ValueError(f"scenario must be one of {sorted(_SCENARIOS)}")
    return value


def _target_range(scenario: str, cell_type: int, *, founder: bool = False) -> tuple[float, float]:
    if scenario == "competition":
        lower = {0: 1.0, 1: 2.0, 2: 3.5}[cell_type]
        return lower, lower + 0.5
    if scenario == "two_types":
        return 2.5, 3.0
    if scenario == "short_cells":
        return (2.5, 3.0) if founder else (1.0, 1.5)
    if scenario == "box":
        return 3.0, 3.5
    return 3.5, 4.0


def _growth_rate(scenario: str, cell_type: int) -> float:
    if scenario == "competition":
        return {0: 2.0, 1: 1.1, 2: 0.8}[cell_type]
    if scenario in {"two_types", "short_cells"}:
        return 2.0
    return 1.0


def _targets(step: ControllerStep) -> dict[str, JSONValue]:
    value = step.state.get("division_targets")
    if not isinstance(value, dict):
        raise ValueError("division target state is invalid")
    return value


def _callbacks(scenario: str):
    def regulate(step: ControllerStep) -> StepPlan:
        targets = _targets(step)
        if set(targets) != {str(cell.id) for cell in step.cells}:
            raise ValueError("division targets do not match active cells")
        max_y = max((cell.position.y for cell in step.cells), default=0.0)
        updates: list[CellUpdate] = []
        divisions: list[DivisionRequest] = []
        for cell in step.cells:
            growth_rate = _growth_rate(scenario, cell.cell_type)
            if scenario == "competition" and max_y - cell.position.y >= 5.0:
                growth_rate = 0.0
            updates.append(CellUpdate(cell.id, growth_rate=growth_rate))
            target = targets[str(cell.id)]
            if not isinstance(target, int | float) or isinstance(target, bool):
                raise ValueError("division target state is invalid")
            if cell.length > float(target):
                divisions.append(DivisionRequest(cell.id))
        return StepPlan(updates=tuple(updates), divisions=tuple(divisions))

    def divided(step: ControllerStep, event: DivisionEvent) -> None:
        targets = _targets(step)
        del targets[str(event.parent.id)]
        lower, upper = _target_range(scenario, event.parent.cell_type)
        jitter_z = scenario in {"short_cells", "box"}
        for daughter in (event.first, event.second):
            targets[str(daughter.id)] = step.rng.uniform(lower, upper)
            jitter = [step.rng.uniform(-1.0e-3, 1.0e-3) for _ in range(3)]
            if not jitter_z:
                jitter[2] = 0.0
            step.simulation.set_cell_geometry(
                daughter.id,
                daughter.position,
                Vec3(
                    daughter.direction.x + jitter[0],
                    daughter.direction.y + jitter[1],
                    daughter.direction.z + jitter[2],
                ),
                daughter.length,
            )

    return regulate, divided


def _add_box(simulation: Simulation) -> None:
    planes = (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((10.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ((-10.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 10.0, 0.0), (0.0, -1.0, 0.0)),
        ((0.0, -10.0, 0.0), (0.0, 1.0, 0.0)),
    )
    for point, normal in planes:
        plane = PlaneConstraintInit()
        plane.point = Vec3(*point)
        plane.inward_normal = Vec3(*normal)
        plane.coefficient = 1.0
        simulation.add_plane_constraint(plane)


def _founder_specs(scenario: str) -> tuple[tuple[int, Vec3], ...]:
    if scenario == "two_types":
        return ((0, Vec3(-10.0, 0.0, 0.0)), (1, Vec3(10.0, 0.0, 0.0)))
    if scenario == "competition":
        return (
            (0, Vec3(0.0, 0.0, 0.0)),
            (1, Vec3(6.0, 0.0, 0.0)),
            (2, Vec3(-6.0, 0.0, 0.0)),
        )
    if scenario == "box":
        return ((0, Vec3(0.0, 0.0, 0.5)),)
    return ((0, Vec3()),)


def build(context: ModelContext) -> NativeController:
    scenario = _scenario(context.parameters)
    simulation = context.simulation(reserved_capacity=10_000)
    if scenario == "box":
        _add_box(simulation)

    targets: dict[str, JSONValue] = {}
    for cell_type, position in _founder_specs(scenario):
        founder = CellInit()
        founder.position = position
        founder.direction = Vec3(1.0, 0.0, 0.0)
        founder.length = 2.0 if scenario == "competition" else 3.5
        founder.radius = 0.5
        founder.growth_rate = _growth_rate(scenario, cell_type)
        founder.cell_type = cell_type
        founder_id = simulation.add_cell(founder)
        lower, upper = _target_range(scenario, cell_type, founder=True)
        targets[str(founder_id)] = context.rng.uniform(lower, upper)

    regulate, divided = _callbacks(scenario)
    mechanics = MechanicsConfig(gamma=20.0 if scenario == "box" else 10.0)
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=regulate,
        on_division=divided,
        mechanics=mechanics,
        state={"scenario": scenario, "division_targets": targets},
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> NativeController:
    scenario = _scenario(context.parameters)
    state = checkpoint.controller
    if not isinstance(state, dict):
        raise ValueError("checkpoint controller state is invalid")
    regulate, divided = _callbacks(scenario)
    controller = NativeController.from_checkpoint(
        checkpoint,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        regulate=regulate,
        on_division=divided,
    )
    if cast(object, controller.state.get("scenario")) != scenario:
        raise ValueError("checkpoint scenario does not match model parameters")
    return controller
