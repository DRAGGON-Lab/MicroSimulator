"""Typed MicroSimulator migration of Tutorial_2/Tutorial_2b.py."""

from __future__ import annotations

from microsimulator import (
    CellInit,
    CellUpdate,
    ControllerStep,
    MechanicsConfig,
    ModelContext,
    NativeController,
    RatePlanBuilder,
    SpeciesRatePlan,
    StepPlan,
    UniformLengthDivision,
)
from microsimulator.checkpoint import CheckpointBundle, JSONValue

MODEL_ID = "legacy.Tutorial_2.Tutorial_2b"
MODEL_VERSION = 1
DIVISION = UniformLengthDivision(3.0, 3.5, jitter_z=False)


def _regulate(step: ControllerStep) -> StepPlan:
    return StepPlan(
        updates=tuple(CellUpdate(cell.id, growth_rate=0.6) for cell in step.cells),
        divisions=DIVISION.requests(step),
    )


def _rate_plan() -> SpeciesRatePlan:
    rates = RatePlanBuilder()
    x = rates.species(0)
    y = rates.species(1)
    x_squared = x * x
    dx = 2.0 * (1.0 + x_squared) / (1.0 + x_squared + y * y) - x
    dy = 2.0 * (1.0 + x_squared) / (1.0 + x_squared) - y
    return rates.species_plan(2, (dx, dy))


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(species_count=2)
    simulation.set_species_rate_plan(_rate_plan())
    founder = CellInit()
    founder.length = 3.5
    founder.radius = 0.5
    founder.growth_rate = 0.6
    founder.species = [0.0, 0.0]
    founder_id = simulation.add_cell(founder)
    state: dict[str, JSONValue] = {}
    DIVISION.initialize(state, context.rng, (founder_id,))
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=_regulate,
        on_division=DIVISION.on_division,
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
        on_division=DIVISION.on_division,
    )
