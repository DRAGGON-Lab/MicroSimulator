"""Colony growth confined to a round culture dish.

The dish is a single inside-region cylinder constraint: its barrel is the circular
dish wall and its caps confine the colony to a monolayer.
"""

from __future__ import annotations

import math

from cellmodeller2 import (
    CellInit,
    ConstraintRegion,
    ControllerStep,
    CylinderConstraintInit,
    DivisionEvent,
    MechanicsConfig,
    ModelContext,
    NativeController,
    Simulation,
    StepPlan,
    UniformLengthDivision,
    Vec3,
)
from cellmodeller2.checkpoint import CheckpointBundle, JSONValue

MODEL_ID = "examples.culture-dish"
MODEL_VERSION = 1
DIVISION = UniformLengthDivision(3.2, 3.8, jitter_z=False)

DISH_RADIUS = 30.0
DISH_HALF_HEIGHT = 1.0
CELL_RADIUS = 0.5
FOUNDER_COUNT = 5


def _add_dish(simulation: Simulation) -> None:
    dish = CylinderConstraintInit()
    dish.center = Vec3(0.0, 0.0, 0.0)
    dish.radius = DISH_RADIUS
    dish.half_height = DISH_HALF_HEIGHT
    dish.coefficient = 1.0
    dish.allowed_region = ConstraintRegion.INSIDE
    simulation.add_cylinder_constraint(dish)


def _regulate(step: ControllerStep) -> StepPlan:
    return StepPlan(divisions=DIVISION.requests(step))


def _divided(step: ControllerStep, event: DivisionEvent) -> None:
    DIVISION.on_division(step, event)


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(reserved_capacity=20_000)
    _add_dish(simulation)

    founder_ids = []
    for index in range(FOUNDER_COUNT):
        placement = context.rng.uniform(0.0, 2.0 * math.pi)
        distance = context.rng.uniform(0.0, DISH_RADIUS * 0.5)
        angle = context.rng.uniform(0.0, 2.0 * math.pi)
        founder = CellInit()
        founder.position = Vec3(
            distance * math.cos(placement),
            distance * math.sin(placement),
            0.0,
        )
        founder.direction = Vec3(math.cos(angle), math.sin(angle), 0.0)
        founder.length = 3.0 + 0.2 * index
        founder.radius = CELL_RADIUS
        founder.growth_rate = 1.0
        founder_ids.append(simulation.add_cell(founder))

    state: dict[str, JSONValue] = {"scope": "culture-dish"}
    DIVISION.initialize(state, context.rng, tuple(founder_ids))
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=_regulate,
        on_division=_divided,
        mechanics=MechanicsConfig(),
        state=state,
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
