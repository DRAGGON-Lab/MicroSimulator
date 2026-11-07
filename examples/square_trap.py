"""Live nutrient-rich square trap with mechanics-limited growth."""

from __future__ import annotations

import math

from microsimulator import (
    BoxConstraintInit,
    CellInit,
    ConstraintRegion,
    ControllerStep,
    DivisionEvent,
    DivisionRequest,
    GridShape,
    MechanicsConfig,
    ModelContext,
    NativeController,
    SignalGridSpec,
    Simulation,
    StepPlan,
    Vec3,
)
from microsimulator.checkpoint import CheckpointBundle

MODEL_ID = "examples.square-trap"
MODEL_VERSION = 1


def _regulate(step: ControllerStep) -> StepPlan:
    return StepPlan(
        divisions=tuple(DivisionRequest(cell.id) for cell in step.cells if cell.length > 2.0)
    )


def _divided(step: ControllerStep, event: DivisionEvent) -> None:
    handedness = 1.0 if step.rng.randrange(2) else -1.0
    tangent = Vec3(-event.parent.direction.y, event.parent.direction.x, 0.0)
    cosine, sine = math.cos(0.01), math.sin(0.01)
    for cell, sign in ((event.first, handedness), (event.second, -handedness)):
        direction = Vec3(
            cell.direction.x * cosine + tangent.x * sine * sign,
            cell.direction.y * cosine + tangent.y * sine * sign,
            cell.direction.z,
        )
        step.simulation.set_cell_geometry(cell.id, cell.position, direction, cell.length)


def build(context: ModelContext) -> NativeController:
    simulation: Simulation = context.simulation()
    shape = GridShape()
    shape.x, shape.y, shape.z = 9, 9, 3
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.origin = Vec3(-4.0, -4.0, -1.0)
    grid.spacing = Vec3(1.0, 1.0, 1.0)
    grid.diffusion = [0.0]
    grid.advection = [Vec3()]
    simulation.configure_signal_grid(grid, [10.0] * (9 * 9 * 3))

    chamber = BoxConstraintInit()
    chamber.center = Vec3(0.0, 0.0, 0.0)
    chamber.half_extents = Vec3(2.0, 2.0, 0.6)
    chamber.allowed_region = ConstraintRegion.INSIDE
    simulation.add_box_constraint(chamber)

    founder = CellInit()
    founder.length = 1.0
    founder.radius = 0.4
    founder.growth_rate = 1.0
    simulation.add_cell(founder)
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=_regulate,
        on_division=_divided,
        mechanics=MechanicsConfig(),
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> NativeController:
    del context
    return NativeController.from_checkpoint(
        checkpoint,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        regulate=_regulate,
        on_division=_divided,
    )
