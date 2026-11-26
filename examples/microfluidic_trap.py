"""A cell trap fed by a flowing channel.

Fresh nutrient enters at the channel inlet, is carried past the trap mouth by
the numerically solved steady device flow, and reaches the colony by diffusion
through the open trap face. Cells exchange drag with the medium according to
their translation relative to it, including when contacts stall or push them. Cell
growth follows Monod kinetics on the local nutrient level and consumes
nutrient at a fixed yield, so the colony's growth pattern reflects the balance
between flow supply, diffusion into the trap, and consumption by the cells
already there.

The trap is a monolayer that a few hundred cells fill. Growth and contact are
solved together: cells push each other toward the trap mouth with contact
forces, and a cell elongates more slowly the larger the axial load those forces
put on it. The cells deep in the trap carry the load of everything between them
and the mouth, so they slow and stall, while the cells at the mouth keep
growing, spill into the channel, and are carried to the outlet. The population
settles at a steady turnover.
"""

from __future__ import annotations

from dataclasses import replace

from microsimulator import (
    CellInit,
    CellSnapshot,
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
from microsimulator.flow import colony_flow_coupling, gap_mobility, solve_flow_field
from microsimulator.microfluidics import TrapChannelDevice

MODEL_ID = "examples.microfluidic-trap"
MODEL_VERSION = 7
DIVISION = UniformLengthDivision(3.2, 3.8, jitter_z=False)

FLOW_SPEED = 20.0
CELL_RADIUS = 0.5
# Lengths are micrometers. The device is 1.5 deep, half again a cell's
# diameter, so the colony is a monolayer, and the trap is 40 long by 20 wide,
# which a few hundred cells fill. From then on growth pushes cells out of the
# trap mouth into the 20-wide channel, where the flow carries them away.
DEVICE = TrapChannelDevice(
    trap_open_x=-20.0,
    trap_back_x=20.0,
    trap_half_y=10.0,
    trap_half_z=0.75,
    channel_far_x=-40.0,
    channel_half_length=60.0,
    mean_flow_speed=FLOW_SPEED,
)
NUTRIENT_INLET = 10.0
BASE_GROWTH_RATE = 1.0
NUTRIENT_K = 5.0
# Nutrient uses arbitrary concentration units. Each accepted step consumes
# the actual increase of B = pi*r^2*(length + 2*r), divided by this yield.
# The value is illustrative; penetration and growth require refinement checks.
NUTRIENT_YIELD = 0.5
# Growth slows under load: elongation falls linearly with the axial load on a
# cell and stops at this load. Loads are forces in units of drag times length
# squared per time, and the load on a cell at the back of the full trap is a
# few hundred. The value is a model choice, not a measured mechanical response.
GROWTH_STALL_LOAD = 200.0
WASHOUT_Y = DEVICE.channel_half_length - 10.0

# At the default dt, medium at the mean channel speed travels one grid spacing
# between solves. The empirical drag and density cap set the strength of coupling.
RESOLVE_INTERVAL = 5
DRAG_COEFFICIENT = 2.0
MAX_VOLUME_FRACTION = 0.5


def _grid(simulation: Simulation | None = None) -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 39, 66, 6
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    # Every wall lies midway between two planes of site centers, so the fluid
    # sites span exactly the device: ten across the channel, twenty along the
    # trap, and two through the depth. The lattice still reaches past the
    # walls, because sampling outside the lattice is an error.
    grid.origin = Vec3(-47.0, -65.0, -1.875)
    grid.spacing = Vec3(2.0, 2.0, 0.75)
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


class TrapController(NativeController):
    def step(self, dt: float) -> None:
        super().step(dt)
        if dt > 0.0 and self.completed_steps % RESOLVE_INTERVAL == 0:
            # Save only the velocities needed for the next flow solve. Division
            # and washout have already happened, so IDs refer to the accepted cells.
            result = self.last_contact_dynamics
            assert result is not None
            velocities = result.center_velocities
            self.state["cell_velocities"] = {
                str(cell.id): [velocities[cell.slot].x, velocities[cell.slot].y]
                for cell in self.simulation.cells()
            }


def _regulate(step: ControllerStep) -> StepPlan:
    velocities = step.state.get("cell_velocities")
    if velocities is not None:
        if not isinstance(velocities, dict) or len(velocities) != len(step.cells):
            raise ValueError("trap flow needs one saved velocity per cell")
        moving_cells: list[tuple[CellSnapshot, Vec3]] = []
        for cell in step.cells:
            xy = velocities.get(str(cell.id))
            if not isinstance(xy, list) or len(xy) != 2:
                raise ValueError(f"invalid saved flow velocity for cell {cell.id}")
            x, y = xy
            if (not isinstance(x, int | float) or isinstance(x, bool)
                    or not isinstance(y, int | float) or isinstance(y, bool)):
                raise ValueError(f"invalid saved flow velocity for cell {cell.id}")
            moving_cells.append((cell, Vec3(x, y, 0.0)))
        mobility, force = colony_flow_coupling(
            GRID, moving_cells, base=GAP_MOBILITY, drag_coefficient=DRAG_COEFFICIENT,
            max_volume_fraction=MAX_VOLUME_FRACTION,
        )
        field, _ = solve_flow_field(
            GRID, mean_inlet_speed=FLOW_SPEED, mobility=mobility, body_force=force,
            simulation=step.simulation,
        )
        step.simulation.set_velocity_field(field)
        del step.state["cell_velocities"]
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
    simulation = context.simulation(reserved_capacity=2_000)
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
    return TrapController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=_regulate,
        on_division=_divided,
        mechanics=MechanicsConfig(
            flow_drift=True, growth_law="linear", growth_stall_load=GROWTH_STALL_LOAD
        ),
        state=state,
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> NativeController:
    del context
    return TrapController.from_checkpoint(
        checkpoint,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        regulate=_regulate,
        on_division=_divided,
    )
