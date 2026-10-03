"""Mother-machine growth, mechanical retention, nutrient delivery, and washout.

Lengths are micrometers and time is seconds. The growth-channel dimensions
follow Wang et al. (2010); the six-channel array and perfusion channel are a
compact model, not a reconstruction of the complete experimental chip.
All cells are free to move. Walls and cell contacts retain the closed-end
lineage; growth pushes descendants into the perfusion channel.

The default maximum growth rate is ln(2)/1800 per second. For visible growth
during a short viewer session, explicitly use --parameter growth_rate=0.3.
This accelerates biology only, and is not an experimentally calibrated run.
"""

import math

from microsimulator import (
    CellInit,
    CellUpdate,
    ChannelMetadata,
    CheckpointBundle,
    ControllerStep,
    CoupledRatePlan,
    DivisionEvent,
    GridShape,
    MechanicsConfig,
    ModelContext,
    NativeController,
    RatePlanBuilder,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    StepPlan,
    UniformLengthDivision,
    Vec3,
)
from microsimulator.checkpoint import JSONValue
from microsimulator.microfluidics import MotherMachineDevice

MODEL_ID = "examples.mother-machine"
MODEL_VERSION = 1
DEVICE = MotherMachineDevice()
SHAPE = (40, 120, 7)
ORIGIN = (-12.5, -22.3125, -0.7)
SPACING = (1.0, 0.375, 1.4)
PULSE_PERIOD = 6.0
NUTRIENT_INLET = 10.0
NUTRIENT_YIELD = 0.5
CELL_RADIUS = 0.5
CELLS_PER_CHANNEL = 7
DIVISION = UniformLengthDivision(3.0, 3.6)


def _grid(simulation: Simulation) -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = SHAPE
    grid = SignalGridSpec()
    grid.signal_count = 2
    grid.shape = shape
    grid.origin = Vec3(*ORIGIN)
    grid.spacing = Vec3(*SPACING)
    grid.diffusion = [10.0, 40.0]
    grid.integration = SignalIntegrationKind.BACKWARD_EULER
    # Float32 diffusion on the narrow cross section cannot reliably reach the
    # generic 1e-7 absolute residual. This is 1e-6 of the supplied nutrient.
    solver = grid.solver
    solver.absolute_tolerance = 1.0e-5
    grid.solver = solver
    DEVICE.apply_to_grid(
        grid,
        inlet_values=[0.0, NUTRIENT_INLET],
        outlet_values=[0.0, NUTRIENT_INLET],
        simulation=simulation,
    )

    return grid


def _inlet_indices(grid: SignalGridSpec) -> list[int]:
    nx, ny, nz = SHAPE
    obstacles = grid.obstacles

    return [
        (x * ny + y) * nz + z
        for x in range(nx)
        for y in range(3)
        for z in range(nz)
        if not obstacles[(x * ny + y) * nz + z]
    ]


def _regulate(step: ControllerStep) -> StepPlan:
    # Only the inlet reservoir is prescribed. Transport supplies the interior
    # and the closed ends through the channel mouths.
    levels = step.simulation.signal_levels
    indices = step.state["inlet_indices"]
    assert isinstance(indices, list)
    concentration = float(step.time % PULSE_PERIOD < PULSE_PERIOD / 2)

    for index in indices:
        assert isinstance(index, int)
        levels[index] = concentration

    step.simulation.set_signal_levels(levels)
    maximum = step.state["growth_rate"]
    assert isinstance(maximum, float)
    divisions = DIVISION.requests(step)
    washed = tuple(cell.id for cell in step.cells if cell.position.x < 0 and cell.position.y > 19.0)

    if washed:
        DIVISION.forget(step, washed)
        divisions = tuple(request for request in divisions if request.parent_id not in washed)
        count = step.state["washed_out"]
        assert isinstance(count, int)
        step.state["washed_out"] = count + len(washed)
        mothers = step.state["mother_ids"]
        assert isinstance(mothers, list)
        step.state["mother_ids"] = [None if cell_id in washed else cell_id for cell_id in mothers]

    return StepPlan(
        updates=tuple(
            CellUpdate(cell.id, growth_rate=_growth(step.simulation, cell.position, maximum))
            for cell in step.cells
            if cell.id not in washed
        ),
        divisions=divisions,
        removals=washed,
    )


def _growth(simulation: Simulation, position: Vec3, maximum: float) -> float:
    nutrient = max(0.0, simulation.sample_signals(position)[1])

    return maximum * nutrient / (5.0 + nutrient)


def _rates() -> CoupledRatePlan:
    rates = RatePlanBuilder()
    uptake = -rates.cell_volume_change_rate() / NUTRIENT_YIELD

    return rates.coupled_plan(0, 2, (), (rates.constant(0.0), uptake))


def _divided(step: ControllerStep, event: DivisionEvent) -> None:
    DIVISION.on_division(step, event)
    count = step.state["division_count"]
    assert isinstance(count, int)
    step.state["division_count"] = count + 1
    mothers = step.state["mother_ids"]
    generations = step.state["mother_generations"]
    assert isinstance(mothers, list) and isinstance(generations, list)

    if event.parent.id in mothers:
        # Founder directions point toward the opening. The first daughter
        # inherits the pole facing the closed end. This is lineage bookkeeping:
        # it changes neither daughter's position, orientation, nor mobility.
        index = mothers.index(event.parent.id)
        mothers[index] = event.first.id
        generation = generations[index]
        assert isinstance(generation, int)
        generations[index] = generation + 1
        step.simulation.set_cell_attributes(event.second.id, event.second.growth_rate, 1)


def _seed_cells(simulation: Simulation, context: ModelContext, state: dict[str, JSONValue]) -> None:
    enabled = context.parameters.get("cells", True)

    if not isinstance(enabled, bool):
        raise ValueError("cells must be a Boolean")

    founders: list[CellInit] = []
    maximum = state["growth_rate"]
    assert isinstance(maximum, float)

    for center in DEVICE.growth_centers if enabled else ():
        pole = DEVICE.growth_length - 0.05

        for index in range(CELLS_PER_CHANNEL):
            cell = CellInit()
            cell.length = context.rng.uniform(2.0, 2.4)
            cell.radius = CELL_RADIUS
            cell.position = Vec3(
                pole - cell.length / 2 - cell.radius, center, DEVICE.growth_height / 2
            )
            cell.direction = Vec3(-1, 0, 0)
            cell.cell_type = 0 if index == 0 else 1
            cell.growth_rate = _growth(simulation, cell.position, maximum)
            founders.append(cell)
            pole -= cell.length + 2 * cell.radius + 0.03

    ids = DIVISION.initialize_founders(simulation, state, context.rng, tuple(founders))
    state["mother_ids"] = [*ids[::CELLS_PER_CHANNEL]]
    state["mother_generations"] = [0 for _ in ids[::CELLS_PER_CHANNEL]]


def build(context: ModelContext) -> NativeController:
    growth = context.parameters.get("growth_rate", math.log(2) / 1800)

    if (
        isinstance(growth, bool)
        or not isinstance(growth, int | float)
        or not math.isfinite(growth)
        or growth < 0
    ):
        raise ValueError("growth_rate must be finite and nonnegative (per second)")

    simulation = context.simulation()
    grid = _grid(simulation)
    inlet = _inlet_indices(grid)
    levels = [0.0] * grid.site_count + [
        0.0 if solid else NUTRIENT_INLET for solid in grid.obstacles
    ]

    for index in inlet:
        levels[index] = 1.0

    simulation.configure_signal_grid(grid, levels)
    simulation.set_coupled_rate_plan(_rates())
    DEVICE.add_constraints(simulation)
    state: dict[str, JSONValue] = {
        "inlet_indices": [*inlet],
        "growth_rate": float(growth),
        "washed_out": 0,
        "division_count": 0,
    }
    _seed_cells(simulation, context, state)

    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=_regulate,
        on_division=_divided,
        mechanics=MechanicsConfig(
            passes=20, gamma=100.0, residual_rms_tolerance=1.0e-4, flow_drift=True
        ),
        state=state,
        channel_metadata=ChannelMetadata(
            signals=("Flow tracer (relative concentration)", "Nutrient (relative concentration)")
        ),
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> NativeController:
    return NativeController.from_checkpoint(
        checkpoint,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        regulate=_regulate,
        on_division=_divided,
    )
