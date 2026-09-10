"""Typed MicroSimulator migration of ex2b_diluteRepression.py."""

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

MODEL_ID = "legacy.ex2b_diluteRepression"
MODEL_VERSION = 1
DIVISION = UniformLengthDivision(2.5, 3.0, jitter_z=False)


def _regulate(step: ControllerStep) -> StepPlan:
    return StepPlan(
        updates=tuple(CellUpdate(cell.id, growth_rate=2.0) for cell in step.cells),
        divisions=DIVISION.requests(step),
    )


def _rate_plan() -> SpeciesRatePlan:
    rates = RatePlanBuilder()
    x0 = rates.species(0)
    repression = 4.0 / (4.0 + x0 * x0)
    return rates.species_plan(2, (rates.constant(0.0), repression))


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(species_count=2)
    simulation.set_species_rate_plan(_rate_plan())
    founder = CellInit()
    founder.length = 3.5
    founder.radius = 0.5
    founder.growth_rate = 2.0
    founder.species = [10.0, 0.0]
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
