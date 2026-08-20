"""SimBOL's Danino quorum-sensing clock in a flow-fed microfluidic trap.

Media flows along the channel with the numerically solved steady device flow:
it delivers nutrient, carries secreted AHL downstream, and washes out cells
that escape the trap. The colony feeds back on the flow: at a fixed cadence
the model rasterizes the packed cells into a Brinkman drag field, re-solves
the flow, and swaps the field into the running simulation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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
    SignalGridAffineReaction,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    StepPlan,
    UniformLengthDivision,
    Vec3,
)
from microsimulator.checkpoint import CheckpointBundle, JSONValue
from microsimulator.flow import (
    colony_mobility,
    colony_species_density,
    gap_mobility,
    solve_flow_field,
)
from microsimulator.microfluidics import TrapChannelDevice

MODEL_ID = "tutorials.danino-clock"
MODEL_VERSION = 7
DIVISION = UniformLengthDivision(3.2, 3.8, jitter_z=False)

FLOW_SPEED = 20.0
DEVICE = TrapChannelDevice(mean_flow_speed=FLOW_SPEED)
CELL_RADIUS = 0.5
# Cells leave the analysis region well before the channel's end. Flow here is
# compressed relative to growth, so a cell swept the full channel length would
# divide several times on the way out and the downstream population would grow
# without bound; the trap itself spans only the first fifteen micrometers.
WASHOUT_Y = 40.0

# The clock's rate constants share one scale. Growth sets the model's unit of
# time, so the scale is what places the clock's period relative to a doubling:
# this value gives about two doubling times, the order Danino et al. report.
CLOCK_RATE = 25.0
# AiiA's removal of AHL, per unit of each. This is a loss proportional to the
# AHL already there, so the model hands it to the grid as an affine reaction
# rather than scattering it from the cells: transport takes a loss field into
# its implicit diagonal, which stays positive while the loss times the step is
# under two, where an explicit cell source of the same strength needs half
# that step. The loss a synchronized pulse reaches in a packed trap is what
# sets the time step this model runs at.
AHL_REMOVAL = 1.0
# The AHL concentration at which the Hill response is half activated. LuxI and
# AiiA respond to the same activation, so their ratio - and with it the AHL
# where production balances removal - is fixed by their decay constants at
# 8 / AHL_REMOVAL times 0.3 / 1.2. A threshold above that is unreachable at any
# cell density, for any run length, and the clock never starts; this one sits
# where the response is steep enough to oscillate.
AHL_THRESHOLD = 4.0
# AHL crosses the trap in about the square of its width over this coefficient.
# Below roughly ten thousand that exchange is slower than the clock's period
# and the trap oscillates in independent patches instead of as one quorum.
AHL_DIFFUSION = 10_000.0

NUTRIENT_INLET = 10.0
BASE_GROWTH_RATE = 1.0
NUTRIENT_K = 5.0
# Nutrient is one limiting substrate in arbitrary concentration units, fed at
# NUTRIENT_INLET. Uptake is tied to realized growth: a cell consumes
# growth_rate * volume / NUTRIENT_YIELD per unit time, so Monod-limited growth
# and consumption stay consistent. The yield sets the coupling strength, and
# this value makes a packed trap's uptake comparable to the diffusive supply
# through its mouth, so nutrient penetrates a few tens of micrometers and the
# colony behind that front grows more slowly.
NUTRIENT_YIELD = 0.5

# Brinkman feedback: how often the colony's drag re-solves the device flow,
# and how strongly a packed voxel resists through-flow.
RESOLVE_INTERVAL = 400
# How often AiiA is rasterized into the grid's AHL loss. The field follows the
# clock, so refreshing it a few dozen times a period keeps it current while
# leaving the per-step cost of building it in the noise.
REMOVAL_INTERVAL = 10
DRAG_COEFFICIENT = 100.0


def _grid() -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 64, 72, 6
    grid = SignalGridSpec()
    grid.signal_count = 2
    grid.shape = shape
    # Two z layers span the trap's six-micrometer depth exactly, so its fluid
    # volume is the device's rather than the half voxel of slack a coarser
    # lattice would leave on each side, and the lattice still reaches a voxel
    # past the walls: contact relaxation can press a crowded cell into a wall
    # and briefly out through it, and sampling outside the lattice is an error.
    grid.origin = Vec3(-140.0, -144.0, -7.5)
    grid.spacing = Vec3(4.0, 4.0, 3.0)
    grid.diffusion = [AHL_DIFFUSION, 20.0]
    grid.advection = [Vec3(), Vec3()]
    grid.integration = SignalIntegrationKind.CRANK_NICOLSON
    DEVICE.apply_to_grid(
        grid,
        inlet_values=[0.0, NUTRIENT_INLET],
        outlet_values=[0.0, 0.0],
    )
    return grid


GRID = _grid()
GAP_MOBILITY = gap_mobility(GRID)


def _primed_levels(grid: SignalGridSpec) -> list[float]:
    # The device is loaded flooded with fresh media before flow starts; AHL
    # starts at zero everywhere.
    site_count = grid.shape.x * grid.shape.y * grid.shape.z
    levels = [0.0] * (2 * site_count)
    for site, solid in enumerate(grid.obstacles):
        if solid == 0:
            levels[site_count + site] = NUTRIENT_INLET
    return levels


def _rate_plan() -> CoupledRatePlan:
    rates = RatePlanBuilder()
    luxi = rates.maximum(rates.species(0), 0.0)
    aiia = rates.maximum(rates.species(1), 0.0)
    gfp = rates.maximum(rates.species(2), 0.0)
    ahl = rates.maximum(rates.signal(0), 0.0)
    ahl_cubed = ahl**3.0
    hill = ahl_cubed / (AHL_THRESHOLD**3.0 + ahl_cubed)
    activated = CLOCK_RATE * (0.02 + 8.0 * hill)
    return rates.coupled_plan(
        3,
        2,
        (
            activated - CLOCK_RATE * 1.2 * luxi,
            activated - CLOCK_RATE * 0.3 * aiia,
            activated - CLOCK_RATE * 0.5 * gfp,
        ),
        (
            CLOCK_RATE * 8.0 * luxi,
            -(rates.growth_rate() * rates.cell_volume()) / NUTRIENT_YIELD,
        ),
    )


_FLUID = np.asarray(GRID.obstacles, dtype=np.uint8) == 0
_NO_SOURCES = [0.0] * (2 * GRID.site_count)


def _ahl_removal_field(cells: Sequence[CellSnapshot]) -> SignalGridAffineReaction:
    """Rasterize AiiA into the grid's first-order AHL loss.

    Every cell removes AHL in proportion to its AiiA and to the AHL around it.
    Summed over the cells of a voxel and divided by its volume, that is a loss
    rate per unit time on the AHL field, which is what an affine reaction
    carries. Nutrient takes no field reaction; its uptake follows growth and
    stays a cell source.
    """

    aiia = np.asarray(colony_species_density(GRID, cells, species=1))
    loss = np.zeros(2 * GRID.site_count, dtype=np.float64)
    loss[: GRID.site_count] = np.where(_FLUID, CLOCK_RATE * AHL_REMOVAL * aiia, 0.0)
    reaction = SignalGridAffineReaction()
    reaction.source_rates = _NO_SOURCES
    reaction.loss_rates = loss.tolist()
    return reaction


def _nutrient_growth(simulation: Simulation, position: Vec3) -> float:
    nutrient = max(0.0, simulation.sample_signals(position)[1])
    return BASE_GROWTH_RATE * nutrient / (NUTRIENT_K + nutrient)


def _regulate(step: ControllerStep) -> StepPlan:
    if step.completed_steps % REMOVAL_INTERVAL == 0:
        step.simulation.set_signal_reaction(_ahl_removal_field(step.cells))
    if step.completed_steps and step.completed_steps % RESOLVE_INTERVAL == 0:
        mobility = colony_mobility(
            GRID, step.cells, base=GAP_MOBILITY, drag_coefficient=DRAG_COEFFICIENT
        )
        field, _ = solve_flow_field(GRID, mean_inlet_speed=FLOW_SPEED, mobility=mobility)
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
    for daughter in (event.first, event.second):
        step.simulation.set_species(
            daughter.id,
            [max(0.0, value * step.rng.uniform(0.9, 1.1)) for value in event.parent.species],
        )


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(reserved_capacity=5_000, species_count=3)
    simulation.configure_signal_grid(GRID, _primed_levels(GRID))
    simulation.set_coupled_rate_plan(_rate_plan())
    DEVICE.add_constraints(simulation)

    founder = CellInit()
    founder.position = Vec3(DEVICE.trap_back_x - 5.0, 0.0, 0.0)
    founder.length = 3.5
    founder.radius = CELL_RADIUS
    founder.growth_rate = 1.0
    founder.species = [context.rng.uniform(0.0, 0.2), context.rng.uniform(0.0, 0.2), 0.0]
    founder_id = simulation.add_cell(founder)
    state: dict[str, JSONValue] = {"scope": "clock-nutrient-field-and-trap"}
    DIVISION.initialize(state, context.rng, (founder_id,))
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
