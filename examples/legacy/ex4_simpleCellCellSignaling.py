"""Typed MicroSimulator migration of ex4_simpleCellCellSignaling.py."""

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
    PlaneConstraintInit,
    RatePlanBuilder,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    StepPlan,
    UniformLengthDivision,
    Vec3,
)
from microsimulator.checkpoint import CheckpointBundle, JSONValue

MODEL_ID = "legacy.ex4_simpleCellCellSignaling"
MODEL_VERSION = 1
DIVISION = UniformLengthDivision(2.5, 3.0, jitter_z=False)
VOXEL_VOLUME = 64.0


def _grid() -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 64, 8, 12
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.origin = Vec3(-128.0, -14.0, -8.0)
    grid.spacing = Vec3(4.0, 4.0, 4.0)
    grid.diffusion = [10.0]
    grid.advection = [Vec3()]
    grid.integration = SignalIntegrationKind.CRANK_NICOLSON
    grid.solver.absolute_tolerance = 1.0e-12
    return grid


def _rates() -> CoupledRatePlan:
    rates = RatePlanBuilder()
    x0 = rates.species(0)
    extracellular = rates.signal(0)
    exchange_amount = 0.1 * (extracellular - x0) * rates.cell_surface_area()
    exchange_concentration = exchange_amount / VOXEL_VOLUME
    type_zero = rates.equal(rates.cell_type(), 0)
    x0_squared = x0 * x0
    return rates.coupled_plan(
        3,
        1,
        (
            rates.select(type_zero, 1.0 + exchange_concentration, exchange_concentration),
            rates.select(type_zero, 1.0, 0.0),
            rates.select(type_zero, 0.0, x0_squared / (5.0e-5 + x0_squared)),
        ),
        (-exchange_amount,),
    )


def _regulate(step: ControllerStep) -> StepPlan:
    return StepPlan(
        updates=tuple(CellUpdate(cell.id, growth_rate=2.0) for cell in step.cells),
        divisions=DIVISION.requests(step),
    )


def _add_channel(simulation: Simulation) -> None:
    for y, normal_y in ((-16.0, 1.0), (16.0, -1.0)):
        plane = PlaneConstraintInit()
        plane.point = Vec3(0.0, y, 0.0)
        plane.inward_normal = Vec3(0.0, normal_y, 0.0)
        plane.coefficient = 1.0
        simulation.add_plane_constraint(plane)


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(species_count=3)
    simulation.configure_signal_grid(_grid())
    simulation.set_coupled_rate_plan(_rates())
    _add_channel(simulation)
    founders: list[int] = []
    for cell_type, x in ((0, -10.0), (1, 10.0)):
        founder = CellInit()
        founder.position = Vec3(x, 0.0, 0.0)
        founder.length = 3.5
        founder.radius = 0.5
        founder.growth_rate = 2.0
        founder.cell_type = cell_type
        founder.species = [0.0, 0.0, 0.0]
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
