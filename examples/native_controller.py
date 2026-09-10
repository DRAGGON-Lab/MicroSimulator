"""Restartable regulated colony using the typed native controller."""

from __future__ import annotations

import math
from collections.abc import Mapping

from microsimulator import (
    CellInit,
    CellUpdate,
    ControllerStep,
    DivisionEvent,
    DivisionRequest,
    MechanicsConfig,
    ModelContext,
    NativeController,
    RegulationCallback,
    StepPlan,
    Vec3,
)
from microsimulator.checkpoint import CheckpointBundle, JSONValue

MODEL_ID = "examples.native-controller"
MODEL_VERSION = 1


def _number(parameters: Mapping[str, JSONValue], name: str, default: float) -> float:
    value = parameters.get(name, default)
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"parameter {name!r} must be finite")
    return float(value)


def _regulation(division_length: float) -> RegulationCallback:
    def regulate(step: ControllerStep) -> StepPlan:
        updates = tuple(CellUpdate(cell.id, growth_rate=0.2) for cell in step.cells)
        divisions = tuple(
            DivisionRequest(cell.id) for cell in step.cells if cell.length >= division_length
        )
        return StepPlan(updates=updates, divisions=divisions)

    return regulate


def _division(step: ControllerStep, event: DivisionEvent) -> None:
    count = step.state.get("division_count", 0)
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("division_count controller state is invalid")
    step.state["division_count"] = count + 1
    for daughter in (event.first, event.second):
        jitter = step.rng.uniform(-1.0e-3, 1.0e-3)
        direction = Vec3(daughter.direction.x, daughter.direction.y + jitter, daughter.direction.z)
        step.simulation.set_cell_geometry(
            daughter.id,
            daughter.position,
            direction,
            daughter.length,
        )


def build(context: ModelContext) -> NativeController:
    division_length = _number(context.parameters, "division_length", 4.0)
    simulation = context.simulation()
    founder = CellInit()
    founder.position = Vec3(context.rng.uniform(-0.05, 0.05), 0.0, 0.0)
    founder.length = _number(context.parameters, "initial_length", 3.0)
    founder.radius = 0.5
    simulation.add_cell(founder)
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=_regulation(division_length),
        on_division=_division,
        mechanics=MechanicsConfig(),
        state={"division_count": 0},
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> NativeController:
    division_length = _number(context.parameters, "division_length", 4.0)
    return NativeController.from_checkpoint(
        checkpoint,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        regulate=_regulation(division_length),
        on_division=_division,
    )
