"""One modeled biopixel from the Prindle sensing-array study.

The supplied CAD contains 496 matching Layer-2 outlines in a 16 by 31 layout;
the Nature article describes a nominal 500-biopixel device. The supplemental
methods, rather than an inferred CAD wall inset, supply this model's 100 by 85
by 1.65 micrometer trapping region. Loading the DXF validates its layout and a
source-specific unit conversion but does not determine the cavity walls.

The adjacent 100 by 10 by 300 micrometer channel, mean flow speed, numerical
wall thickness, transport parameters, and Brinkman feedback are explicit model
choices. The example simulates one trap under one chosen local boundary
condition; it does not model hydraulic variation or coupling across the array.
"""

from __future__ import annotations

from pathlib import Path

from cellmodeller2 import (
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
from cellmodeller2.checkpoint import CheckpointBundle, JSONValue
from cellmodeller2.flow import colony_mobility, gap_mobility, solve_flow_field
from cellmodeller2.masks import (
    MaskError,
    MaskRectangle,
    extract_rectangles,
    load_mask_polylines,
    match_rectangles,
)
from cellmodeller2.microfluidics import BiopixelTrapDevice

MODEL_ID = "tutorials.biopixel-trap"
MODEL_VERSION = 6
DIVISION = UniformLengthDivision(3.2, 3.8, jitter_z=False)

_MASK = Path(__file__).resolve().parents[2] / "docs" / "tutorials" / "devices" / "prindle.dxf"
_MASK_UNIT_SCALE = 1000.0
_MASK_OUTLINE = (110.0, 100.0)
FLOW_SPEED = 20.0


def _load_prindle_layout() -> tuple[MaskRectangle, ...]:
    polylines = load_mask_polylines(_MASK)
    rectangles = extract_rectangles(
        polylines,
        layer="Layer-2",
        unit_scale=_MASK_UNIT_SCALE,
    )
    traps = match_rectangles(rectangles, *_MASK_OUTLINE, tolerance=1.0)
    if len(traps) != 496:
        raise MaskError(f"expected 496 Prindle layout outlines, found {len(traps)}")
    return traps


TRAP_OUTLINES = _load_prindle_layout()
DEVICE = BiopixelTrapDevice(mean_flow_speed=FLOW_SPEED)
CELL_RADIUS = 0.5
WASHOUT_Y = DEVICE.channel_half_length - 10.0

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
RESOLVE_INTERVAL = 100
DRAG_COEFFICIENT = 100.0


def _grid() -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 42, 60, 14
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    # The lattice of site centers covers every position a cell can reach, with
    # a margin of one voxel past the floor and the far channel wall: contact
    # relaxation lets a crowded cell press slightly into a wall, and sampling
    # outside the lattice is an error. Two z layers span the cavity exactly;
    # the resulting gap-height mobility ratio belongs to this model geometry.
    grid.origin = Vec3(-100.0, -147.5, -0.4125)
    grid.spacing = Vec3(5.0, 5.0, 0.825)
    grid.diffusion = [40.0]
    grid.advection = [Vec3()]
    grid.integration = SignalIntegrationKind.CRANK_NICOLSON
    DEVICE.apply_to_grid(grid, inlet_values=[NUTRIENT_INLET], outlet_values=[0.0])
    return grid


GRID = _grid()
GAP_MOBILITY = gap_mobility(GRID)


def _rate_plan() -> CoupledRatePlan:
    rates = RatePlanBuilder()
    uptake = -(rates.growth_rate() * rates.cell_volume()) / NUTRIENT_YIELD
    return rates.coupled_plan(0, 1, (), (uptake,))


def _primed_levels(grid: SignalGridSpec) -> list[float]:
    # The device is loaded flooded with fresh media before flow starts.
    return [NUTRIENT_INLET if solid == 0 else 0.0 for solid in grid.obstacles]


def _nutrient_growth(simulation: Simulation, position: Vec3) -> float:
    nutrient = max(0.0, simulation.sample_signals(position)[0])
    return BASE_GROWTH_RATE * nutrient / (NUTRIENT_K + nutrient)


def _regulate(step: ControllerStep) -> StepPlan:
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


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(reserved_capacity=20_000)
    simulation.configure_signal_grid(GRID, _primed_levels(GRID))
    simulation.set_coupled_rate_plan(_rate_plan())
    DEVICE.add_constraints(simulation)

    founder = CellInit()
    founder.position = Vec3(DEVICE.trap_depth * 0.5, 0.0, DEVICE.trap_height * 0.5)
    founder.length = 3.5
    founder.radius = CELL_RADIUS
    founder.growth_rate = 1.0
    founder_id = simulation.add_cell(founder)
    state: dict[str, JSONValue] = {"scope": "biopixel-trap"}
    DIVISION.initialize(state, context.rng, (founder_id,))
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=_regulate,
        on_division=_divided,
        mechanics=MechanicsConfig(flow_drift=True, passes=2),
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
