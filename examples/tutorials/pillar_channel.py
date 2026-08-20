"""Colonies seeded on a pillar array in a flowing channel.

The device is a monolayer channel crossed by a staggered array of cylindrical
pillars - geometry with no analytic flow profile, so the field comes from the
numerical solve: `solve_flow_field` routes the media around every pillar with
per-voxel mass conservation, and the same solve re-runs at a fixed cadence
with the colony's Brinkman drag so growing colonies divert the flow. Founder
cells are adhered (fixed) in pillar wakes; each division keeps the mother
attached and releases the daughter into the stream, which carries it between
the pillars and washes it out at the end of the channel - a biofilm shedding
cells into flow.
"""

from __future__ import annotations

import math

from cellmodeller2 import (
    BoxConstraintInit,
    CellInit,
    CellUpdate,
    ConstraintRegion,
    ControllerStep,
    CoupledRatePlan,
    CylinderConstraintInit,
    DivisionEvent,
    GridBoundaryKind,
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

MODEL_ID = "tutorials.pillar-channel"
MODEL_VERSION = 2
DIVISION = UniformLengthDivision(3.2, 3.8, jitter_z=False)

CHANNEL_HALF_WIDTH = 40.0
CHANNEL_HALF_LENGTH = 120.0
CHANNEL_HALF_HEIGHT = 3.0
PILLAR_RADIUS = 10.0
PILLARS = ((-20.0, -60.0), (20.0, -60.0), (0.0, 0.0), (-20.0, 60.0), (20.0, 60.0))

FLOW_SPEED = 20.0
CELL_RADIUS = 0.5
WASHOUT_Y = CHANNEL_HALF_LENGTH - 10.0
# Adhesion sites in pillar wakes; the anchored cell of each lineage stays
# within a cell length of its site.
FOUNDER_SITES = ((-20.0, -46.0), (20.0, -46.0), (0.0, 14.0))

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


def _in_pillar_core(px: float, py: float, margin: float) -> bool:
    # A voxel is solid only when it lies entirely inside the pillar (its
    # center plus half the voxel diagonal stays within the radius). The
    # mechanics cylinders therefore enclose every solid voxel, so a cell
    # center can never sit inside the mask and signal sampling is always in
    # fluid; the stair-stepped flow blockage is conservative by the same
    # margin.
    core = PILLAR_RADIUS - margin
    if core <= 0.0:
        return False
    return any(
        (px - x) * (px - x) + (py - y) * (py - y) < core * core for x, y in PILLARS
    )


def _grid() -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 22, 60, 4
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.origin = Vec3(-42.0, -118.0, -8.0)
    grid.spacing = Vec3(4.0, 4.0, 4.0)
    grid.diffusion = [40.0]
    grid.advection = [Vec3()]
    grid.integration = SignalIntegrationKind.CRANK_NICOLSON
    margin = 0.5 * math.hypot(grid.spacing.x, grid.spacing.y)
    obstacles = [0] * grid.site_count
    for x in range(shape.x):
        px = grid.origin.x + grid.spacing.x * x
        for y in range(shape.y):
            py = grid.origin.y + grid.spacing.y * y
            for z in range(shape.z):
                pz = grid.origin.z + grid.spacing.z * z
                solid = (
                    abs(px) >= CHANNEL_HALF_WIDTH
                    or abs(pz) >= CHANNEL_HALF_HEIGHT
                    or _in_pillar_core(px, py, margin)
                )
                if solid:
                    obstacles[(x * shape.y + y) * shape.z + z] = 1
    grid.obstacles = obstacles
    for name in ("y_lower", "y_upper"):
        boundary = getattr(grid, name)
        boundary.kind = GridBoundaryKind.FIXED
        boundary.values = [NUTRIENT_INLET if name == "y_lower" else 0.0]
        setattr(grid, name, boundary)
    field, _ = solve_flow_field(
        grid, mean_inlet_speed=FLOW_SPEED, mobility=gap_mobility(grid)
    )
    grid.velocity_field = field
    return grid


GRID = _grid()
GAP_MOBILITY = gap_mobility(GRID)


def _add_walls(simulation: Simulation) -> None:
    chamber = BoxConstraintInit()
    chamber.center = Vec3(0.0, 0.0, 0.0)
    chamber.half_extents = Vec3(
        CHANNEL_HALF_WIDTH, CHANNEL_HALF_LENGTH, CHANNEL_HALF_HEIGHT
    )
    chamber.coefficient = 1.0
    chamber.allowed_region = ConstraintRegion.INSIDE
    simulation.add_box_constraint(chamber)
    for x, y in PILLARS:
        pillar = CylinderConstraintInit()
        pillar.center = Vec3(x, y, 0.0)
        pillar.radius = PILLAR_RADIUS
        pillar.half_height = CHANNEL_HALF_HEIGHT + 1.0
        pillar.coefficient = 1.0
        pillar.allowed_region = ConstraintRegion.OUTSIDE
        simulation.add_cylinder_constraint(pillar)


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


def _site_distance(position: Vec3) -> float:
    return min(math.hypot(position.x - x, position.y - y) for x, y in FOUNDER_SITES)


def _divided(step: ControllerStep, event: DivisionEvent) -> None:
    DIVISION.on_division(step, event)
    # Daughters inherit adhesion. The daughter nearer the adhesion site stays
    # attached and the other is released into the stream; anchoring by site,
    # not by daughter order, keeps the attached lineage at its wake instead of
    # random-walking with every division (fixed cells are never moved by
    # mechanics, so a walking anchor would end up inside a pillar).
    if event.parent.fixed:
        released = (
            event.second
            if _site_distance(event.first.position) <= _site_distance(event.second.position)
            else event.first
        )
        step.simulation.set_cell_fixed(released.id, False)


def build(context: ModelContext) -> NativeController:
    simulation = context.simulation(reserved_capacity=10_000)
    simulation.configure_signal_grid(GRID, _primed_levels(GRID))
    simulation.set_coupled_rate_plan(_rate_plan())
    _add_walls(simulation)

    founder_ids = []
    for x, y in FOUNDER_SITES:
        founder = CellInit()
        founder.position = Vec3(x, y, 0.0)
        founder.direction = Vec3(0.0, 1.0, 0.0)
        founder.length = 3.5
        founder.radius = CELL_RADIUS
        founder.growth_rate = 1.0
        founder.fixed = True
        founder_ids.append(simulation.add_cell(founder))
    state: dict[str, JSONValue] = {"scope": "pillar-channel"}
    DIVISION.initialize(state, context.rng, tuple(founder_ids))
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
