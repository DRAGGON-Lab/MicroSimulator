"""Cells growing, dividing, and shedding into flow through a pillar array.

Lengths are micrometers and time is seconds. A shallow, depth-integrated flow
solve routes medium around five pillars. Dye is supplied in a thin inlet
reservoir strip; the native advection-diffusion solver carries it downstream.
The dye is passive; a separate nutrient supports growth and is consumed by
cells. Growth is deliberately accelerated for an interactive demonstration,
not calibrated to an organism. Set the model parameter cells=false for dye only.
"""

import math

from microsimulator import (
    BoxConstraintInit,
    CellInit,
    CellUpdate,
    ChannelMetadata,
    CheckpointBundle,
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
from microsimulator.checkpoint import JSONValue
from microsimulator.flow import gap_mobility, solve_flow_field

MODEL_ID = "examples.microfluidic-flow"
MODEL_VERSION = 2
PILLARS = ((-20.0, -60.0), (20.0, -60.0), (0.0, 0.0), (-20.0, 60.0), (20.0, 60.0))
PILLAR_RADIUS = 10.0
SHAPE = (42, 120, 4)
ORIGIN = (-41.0, -119.0, -4.5)
SPACING = (2.0, 2.0, 3.0)
FLOW_SPEED = 40.0
PULSE_PERIOD = 3.0
NUTRIENT_INLET = 10.0
NUTRIENT_YIELD = 0.5
GROWTH_RATE = 0.4
DIVISION = UniformLengthDivision(3.8, 4.8, jitter_z=False)
ANCHORS = tuple((x + offset, y + 14) for x, y in PILLARS for offset in (-6, -2, 2, 6))


def _solid(x: float, y: float, z: float) -> bool:
    return (
        abs(x) >= 40.0
        or abs(z) >= 3.0
        or any((x - px) ** 2 + (y - py) ** 2 < PILLAR_RADIUS**2 for px, py in PILLARS)
    )


def _grid(simulation: Simulation) -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = SHAPE
    grid = SignalGridSpec()
    grid.signal_count = 2
    grid.shape = shape
    grid.origin = Vec3(*ORIGIN)
    grid.spacing = Vec3(*SPACING)
    grid.diffusion = [0.5, 40.0]
    grid.advection = [Vec3(), Vec3()]
    grid.integration = SignalIntegrationKind.BACKWARD_EULER

    for name in ("y_lower", "y_upper"):
        boundary = getattr(grid, name)
        boundary.kind = GridBoundaryKind.FIXED
        boundary.values = [0.0, NUTRIENT_INLET if name == "y_lower" else 0.0]
        setattr(grid, name, boundary)

    grid.obstacles = [
        int(
            _solid(
                ORIGIN[0] + x * SPACING[0], ORIGIN[1] + y * SPACING[1], ORIGIN[2] + z * SPACING[2]
            )
        )
        for x in range(shape.x)
        for y in range(shape.y)
        for z in range(shape.z)
    ]
    field, _ = solve_flow_field(
        grid,
        mean_inlet_speed=FLOW_SPEED,
        mobility=gap_mobility(grid),
        simulation=simulation,
    )
    grid.velocity_field = field

    return grid


def _walls(simulation: Simulation) -> None:
    channel = BoxConstraintInit()
    channel.half_extents = Vec3(40, 120, 3)
    channel.allowed_region = ConstraintRegion.INSIDE
    simulation.add_box_constraint(channel)

    for x, y in PILLARS:
        pillar = CylinderConstraintInit()
        pillar.center = Vec3(x, y, 0)
        pillar.radius = PILLAR_RADIUS
        pillar.half_height = 4
        pillar.allowed_region = ConstraintRegion.OUTSIDE
        simulation.add_cylinder_constraint(pillar)


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


def _feed(step: ControllerStep) -> StepPlan:
    # Prescribed inlet reservoir concentration alternates between dyed and
    # clear medium. Only this inlet strip is reset; interior transport is solved.
    concentration = float(step.time % PULSE_PERIOD < PULSE_PERIOD / 2)
    levels = step.simulation.signal_levels
    indices = step.state["inlet_indices"]
    assert isinstance(indices, list)

    for index in indices:
        assert isinstance(index, int)
        levels[index] = concentration

    step.simulation.set_signal_levels(levels)

    divisions = DIVISION.requests(step)
    washed = tuple(cell.id for cell in step.cells if abs(cell.position.y) > 108)

    if washed:
        DIVISION.forget(step, washed)
        divisions = tuple(request for request in divisions if request.parent_id not in washed)

    return StepPlan(
        updates=tuple(
            CellUpdate(cell.id, growth_rate=_growth(step.simulation, cell.position))
            for cell in step.cells
            if cell.id not in washed
        ),
        divisions=divisions,
        removals=washed,
    )


def _growth(simulation: Simulation, position: Vec3) -> float:
    nutrient = max(0.0, simulation.sample_signals(position)[1])

    return GROWTH_RATE * nutrient / (5.0 + nutrient)


def _rates() -> CoupledRatePlan:
    rates = RatePlanBuilder()
    uptake = -rates.cell_volume_change_rate() / NUTRIENT_YIELD

    return rates.coupled_plan(0, 2, (), (rates.constant(0.0), uptake))


def _anchor_distance(position: Vec3) -> float:
    return min(math.hypot(position.x - x, position.y - y) for x, y in ANCHORS)


def _divided(step: ControllerStep, event: DivisionEvent) -> None:
    DIVISION.on_division(step, event)

    if event.parent.fixed:
        released = (
            event.second
            if _anchor_distance(event.first.position) <= _anchor_distance(event.second.position)
            else event.first
        )
        step.simulation.set_cell_fixed(released.id, False)


def _seed_cells(simulation: Simulation, context: ModelContext, state: dict[str, JSONValue]) -> None:
    founders: list[CellInit] = []
    enabled = context.parameters.get("cells", True)

    if not isinstance(enabled, bool):
        raise ValueError("cells must be a Boolean")

    sites = (*ANCHORS, *((x, y) for y in (-96, -24, 36) for x in (-32, -8, 8, 32)))

    for index, (x, y) in enumerate(sites if enabled else ()):
        cell = CellInit()
        cell.position = Vec3(x, y, 0)
        cell.direction = Vec3(0, 1, 0)
        cell.radius = 0.65
        cell.length = context.rng.uniform(3.2, 4.6)
        cell.fixed = index < len(ANCHORS)
        cell.cell_type = index // 4 if cell.fixed else 5
        cell.growth_rate = _growth(simulation, cell.position)
        founders.append(cell)

    DIVISION.initialize_founders(simulation, state, context.rng, tuple(founders))


def build(context: ModelContext) -> NativeController:
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
    _walls(simulation)
    state: dict[str, JSONValue] = {"inlet_indices": [*inlet]}
    _seed_cells(simulation, context, state)

    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=_feed,
        on_division=_divided,
        mechanics=MechanicsConfig(flow_drift=True),
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
        regulate=_feed,
        on_division=_divided,
    )
