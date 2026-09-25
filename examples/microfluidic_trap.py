"""A cell trap fed by a flowing channel.

Fresh nutrient enters at the channel inlet, is carried past the trap mouth by
the numerically solved steady device flow, and reaches the colony by diffusion
through the open trap face. Only explicitly fixed cells contribute stationary
flow resistance; this freely moving trap population is not a porous matrix. Cell
growth follows Monod kinetics on the local nutrient level and consumes
nutrient at a fixed yield, so the colony's growth pattern reflects the balance
between flow supply, diffusion into the trap, and consumption by the cells
already there.
"""

from __future__ import annotations

from dataclasses import replace

from microsimulator import (
    CellInit,
    CellUpdate,
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
from microsimulator.checkpoint import CheckpointBundle, JSONValue
from microsimulator.flow import colony_mobility, gap_mobility, solve_flow_field
from microsimulator.microfluidics import TrapChannelDevice

MODEL_ID = "examples.microfluidic-trap"
MODEL_VERSION = 4
DIVISION = UniformLengthDivision(3.2, 3.8, jitter_z=False)

FLOW_SPEED = 20.0
DEVICE = TrapChannelDevice(mean_flow_speed=FLOW_SPEED)
CELL_RADIUS = 0.5
NUTRIENT_INLET = 10.0
BASE_GROWTH_RATE = 1.0
NUTRIENT_K = 5.0
# Nutrient uses arbitrary concentration units. Each accepted step consumes
# the actual increase of B = pi*r^2*(length + 2*r), divided by this yield.
# The value is illustrative; penetration and growth require refinement checks.
NUTRIENT_YIELD = 0.5
WASHOUT_Y = DEVICE.channel_half_length - 10.0

# Resistance feedback uses only fixed (attached) cells, with a physical
# smoothing radius independent of the grid. Free cells do not form a matrix.
RESOLVE_INTERVAL = 100
DRAG_COEFFICIENT = 100.0


def _grid(simulation: Simulation | None = None) -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 64, 72, 6
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    # Two z layers span the trap's six-micrometer depth exactly, so its fluid
    # volume is the device's rather than the half voxel of slack a coarser
    # lattice would leave on each side, and the lattice still reaches a voxel
    # past the walls: contact relaxation can press a crowded cell into a wall
    # and briefly out through it, and sampling outside the lattice is an error.
    grid.origin = Vec3(-140.0, -144.0, -7.5)
    grid.spacing = Vec3(4.0, 4.0, 3.0)
    grid.diffusion = [40.0]
    grid.advection = [Vec3()]
    grid.integration = SignalIntegrationKind.BACKWARD_EULER
    device = DEVICE if simulation is not None else replace(DEVICE, mean_flow_speed=0.0)
    device.apply_to_grid(
        grid,
        inlet_values=[NUTRIENT_INLET],
        outlet_values=[0.0],
        simulation=simulation,
    )
    return grid


GRID = _grid()
GAP_MOBILITY = gap_mobility(GRID)


def _rate_plan() -> CoupledRatePlan:
    rates = RatePlanBuilder()
    uptake = -rates.cell_volume_change_rate() / NUTRIENT_YIELD
    return rates.coupled_plan(0, 1, (), (uptake,))


def _primed_levels(grid: SignalGridSpec) -> list[float]:
    # The device is loaded flooded with fresh media before flow starts.
    return [
        NUTRIENT_INLET if solid == 0 else 0.0
        for solid in grid.obstacles
    ]


def _nutrient_growth(simulation: Simulation, position: Vec3) -> float:
    nutrient = max(0.0, simulation.sample_signals(position)[0])
    return BASE_GROWTH_RATE * nutrient / (NUTRIENT_K + nutrient)


def _regulate(step: ControllerStep) -> StepPlan:
    if step.completed_steps and step.completed_steps % RESOLVE_INTERVAL == 0:
        mobility = colony_mobility(
            GRID, (cell for cell in step.cells if cell.fixed),
            base=GAP_MOBILITY, drag_coefficient=DRAG_COEFFICIENT
        )
        field, _ = solve_flow_field(
            GRID,
            mean_inlet_speed=FLOW_SPEED,
            mobility=mobility,
            simulation=step.simulation,
        )
        step.simulation.set_velocity_field(field)
    divisions = DIVISION.requests(step)
    washed = tuple(cell.id for cell in step.cells if abs(cell.position.y) > WASHOUT_Y)
    if washed:
        DIVISION.forget(step, washed)
        divisions = tuple(request for request in divisions if request.parent_id not in washed)
    return StepPlan(
        updates=tuple(
            CellUpdate(cell.id, growth_rate=_nutrient_growth(step.simulation, cell.position))
            for cell in step.cells
            if cell.id not in washed
        ),
        divisions=divisions,
        removals=washed,
    )


def _divided(step: ControllerStep, event: DivisionEvent) -> None:
    DIVISION.on_division(step, event)


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(reserved_capacity=10_000)
    grid = _grid(simulation)
    simulation.configure_signal_grid(grid, _primed_levels(grid))
    simulation.set_coupled_rate_plan(_rate_plan())
    DEVICE.add_constraints(simulation)

    founder = CellInit()
    founder.position = Vec3(DEVICE.trap_back_x - 5.0, 0.0, 0.0)
    founder.length = 3.5
    founder.radius = CELL_RADIUS
    founder.growth_rate = 1.0
    state: dict[str, JSONValue] = {"scope": "microfluidic-trap"}
    DIVISION.initialize_founders(simulation, state, context.rng, (founder,))
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=_regulate,
        on_division=_divided,
        mechanics=MechanicsConfig(flow_drift=True),
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
