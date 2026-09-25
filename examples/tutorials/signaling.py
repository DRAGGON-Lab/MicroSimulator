"""Native ports of the CellModeller grid-signaling tutorials.

The ``scenario`` parameter selects ``single_gene``, ``communication``, or
``mutualism``.
"""

from __future__ import annotations

from collections.abc import Mapping

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

MODEL_ID = "tutorials.signaling"
MODEL_VERSION = 1
_SCENARIOS = frozenset({"single_gene", "communication", "mutualism"})


def _scenario(parameters: Mapping[str, JSONValue]) -> str:
    value = parameters.get("scenario", "single_gene")
    if not isinstance(value, str) or value not in _SCENARIOS:
        raise ValueError(f"scenario must be one of {sorted(_SCENARIOS)}")
    return value


def _grid(scenario: str) -> SignalGridSpec:
    shape = GridShape()
    grid = SignalGridSpec()
    if scenario == "mutualism":
        shape.x, shape.y, shape.z = 80, 80, 8
        grid.signal_count = 2
        grid.origin = Vec3(-160.0, -160.0, -16.0)
        grid.diffusion = [10.0, 10.0]
        grid.advection = [Vec3(), Vec3()]
    else:
        shape.x, shape.y, shape.z = 64, 8, 12
        grid.signal_count = 1
        grid.origin = Vec3(-128.0, -14.0, -8.0)
        grid.diffusion = [10.0]
        grid.advection = [Vec3()]
    grid.shape = shape
    grid.spacing = Vec3(4.0, 4.0, 4.0)
    grid.integration = SignalIntegrationKind.CRANK_NICOLSON
    grid.solver.absolute_tolerance = 1.0e-12
    return grid


def _rate_plan(scenario: str) -> CoupledRatePlan:
    rates = RatePlanBuilder()
    voxel_volume = 64.0
    area = rates.cell_surface_area()
    if scenario == "single_gene":
        intracellular = rates.species(0)
        extracellular = rates.signal(0)
        exchange_amount = 0.1 * (extracellular - intracellular) * area
        return rates.coupled_plan(
            1,
            1,
            (1.0 + exchange_amount / voxel_volume,),
            (-exchange_amount,),
        )
    if scenario == "communication":
        x0 = rates.species(0)
        extracellular = rates.signal(0)
        exchange_amount = 0.1 * (extracellular - x0) * area
        exchange_concentration = exchange_amount / voxel_volume
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

    alpha_in = rates.species(0)
    beta_in = rates.species(1)
    alpha = rates.signal(0)
    beta = rates.signal(1)
    alpha_exchange = (alpha - alpha_in) * area
    beta_exchange = (beta - beta_in) * area
    type_zero = rates.equal(rates.cell_type(), 0)
    return rates.coupled_plan(
        2,
        2,
        (
            rates.select(
                type_zero,
                1.0 + alpha_exchange / voxel_volume,
                alpha_exchange / voxel_volume,
            ),
            rates.select(
                type_zero,
                beta_exchange / voxel_volume,
                1.0 + beta_exchange / voxel_volume,
            ),
        ),
        (-alpha_exchange, -beta_exchange),
    )


def _division(scenario: str) -> UniformLengthDivision:
    return (
        UniformLengthDivision(3.5, 4.0, jitter_z=False)
        if scenario == "mutualism"
        else UniformLengthDivision(2.5, 3.0, jitter_z=False)
    )


def _callbacks(scenario: str):
    division = _division(scenario)

    def regulate(step: ControllerStep) -> StepPlan:
        updates: list[CellUpdate] = []
        for cell in step.cells:
            if scenario == "mutualism":
                partner = cell.species[1] if cell.cell_type == 0 else cell.species[0]
                growth_rate = 0.1 + 0.9 * partner / (0.1 + partner)
            else:
                growth_rate = 2.0
            updates.append(CellUpdate(cell.id, growth_rate=growth_rate))
        return StepPlan(updates=tuple(updates), divisions=division.requests(step))

    return division, regulate


def _add_channel(simulation: Simulation) -> None:
    for y, normal_y in ((-16.0, 1.0), (16.0, -1.0)):
        plane = PlaneConstraintInit()
        plane.point = Vec3(0.0, y, 0.0)
        plane.inward_normal = Vec3(0.0, normal_y, 0.0)
        plane.coefficient = 1.0
        simulation.add_plane_constraint(plane)


def build(context: ModelContext) -> NativeController:
    scenario = _scenario(context.parameters)
    species_count = 3 if scenario == "communication" else 2 if scenario == "mutualism" else 1
    simulation = context.simulation(reserved_capacity=10_000, species_count=species_count)
    simulation.configure_signal_grid(_grid(scenario))
    simulation.set_coupled_rate_plan(_rate_plan(scenario))
    if scenario != "mutualism":
        _add_channel(simulation)

    founder_specs = (
        ((0, -3.0), (1, 3.0))
        if scenario == "mutualism"
        else ((0, -10.0), (1, 10.0))
        if scenario == "communication"
        else ((0, 0.0),)
    )
    founders: list[CellInit] = []
    for cell_type, x in founder_specs:
        founder = CellInit()
        founder.position = Vec3(x, 0.0, 0.0)
        founder.length = 3.5
        founder.radius = 0.5
        founder.growth_rate = 1.0 if scenario == "mutualism" else 2.0
        founder.cell_type = cell_type
        founder.species = [0.0] * species_count
        founders.append(founder)

    division, regulate = _callbacks(scenario)
    state: dict[str, JSONValue] = {"scenario": scenario}
    division.initialize_founders(simulation, state, context.rng, tuple(founders))
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=regulate,
        on_division=division.on_division,
        mechanics=MechanicsConfig(),
        state=state,
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> NativeController:
    scenario = _scenario(context.parameters)
    division, regulate = _callbacks(scenario)
    controller = NativeController.from_checkpoint(
        checkpoint,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        regulate=regulate,
        on_division=division.on_division,
    )
    if controller.state.get("scenario") != scenario:
        raise ValueError("checkpoint scenario does not match model parameters")
    return controller
