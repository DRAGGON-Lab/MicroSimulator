"""Typed MicroSimulator migration of Tutorial_3/Tutorial_3.py."""

from __future__ import annotations

from microsimulator import (
    CellInit,
    CellUpdate,
    ControllerStep,
    CoupledRatePlan,
    GridShape,
    MechanicsConfig,
    ModelContext,
    NativeController,
    RatePlanBuilder,
    SignalGridSpec,
    SignalIntegrationKind,
    StepPlan,
    UniformLengthDivision,
    Vec3,
)
from microsimulator.checkpoint import CheckpointBundle, JSONValue

MODEL_ID = "legacy.Tutorial_3.Tutorial_3"
MODEL_VERSION = 1
DIVISION = UniformLengthDivision(3.5, 4.0, jitter_z=False)
VOXEL_VOLUME = 64.0


def _grid() -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 80, 80, 8
    grid = SignalGridSpec()
    grid.signal_count = 2
    grid.shape = shape
    grid.origin = Vec3(-160.0, -160.0, -16.0)
    grid.spacing = Vec3(4.0, 4.0, 4.0)
    grid.diffusion = [10.0, 10.0]
    grid.advection = [Vec3(), Vec3()]
    grid.integration = SignalIntegrationKind.CRANK_NICOLSON
    grid.solver.absolute_tolerance = 1.0e-12
    return grid


def _rates() -> CoupledRatePlan:
    rates = RatePlanBuilder()
    alpha_in = rates.species(0)
    beta_in = rates.species(1)
    alpha = rates.signal(0)
    beta = rates.signal(1)
    area = rates.cell_surface_area()
    alpha_exchange = (alpha - alpha_in) * area
    beta_exchange = (beta - beta_in) * area
    type_zero = rates.equal(rates.cell_type(), 0)
    return rates.coupled_plan(
        2,
        2,
        (
            rates.select(
                type_zero, 1.0 + alpha_exchange / VOXEL_VOLUME, alpha_exchange / VOXEL_VOLUME
            ),
            rates.select(
                type_zero, beta_exchange / VOXEL_VOLUME, 1.0 + beta_exchange / VOXEL_VOLUME
            ),
        ),
        (-alpha_exchange, -beta_exchange),
    )


def _regulate(step: ControllerStep) -> StepPlan:
    updates: list[CellUpdate] = []
    for cell in step.cells:
        partner = cell.species[1] if cell.cell_type == 0 else cell.species[0]
        growth_rate = 0.1 + 0.9 * partner / (0.1 + partner)
        updates.append(CellUpdate(cell.id, growth_rate=growth_rate))
    return StepPlan(updates=tuple(updates), divisions=DIVISION.requests(step))


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(species_count=2)
    simulation.configure_signal_grid(_grid())
    simulation.set_coupled_rate_plan(_rates())
    founders: list[int] = []
    for cell_type, x in ((0, -3.0), (1, 3.0)):
        founder = CellInit()
        founder.position = Vec3(x, 0.0, 0.0)
        founder.length = 3.5
        founder.radius = 0.5
        founder.growth_rate = 1.0
        founder.cell_type = cell_type
        founder.species = [0.0, 0.0]
        founders.append(simulation.add_cell(founder))
    state: dict[str, JSONValue] = {}
    DIVISION.initialize(state, context.rng, tuple(founders))
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
