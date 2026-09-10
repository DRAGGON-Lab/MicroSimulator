"""SimBOL's Danino quorum-sensing clock, nutrient field, and trap geometry."""

from __future__ import annotations

import math

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
    PlaneConstraintInit,
    RatePlanBuilder,
    SignalGridAffineReaction,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    SphereConstraintInit,
    SphereRegion,
    StepPlan,
    UniformLengthDivision,
    Vec3,
)
from microsimulator.checkpoint import CheckpointBundle, JSONValue

MODEL_ID = "tutorials.danino-clock"
MODEL_VERSION = 2
DIVISION = UniformLengthDivision(3.2, 3.8, jitter_z=False)

TRAP_OPEN_X = -60.0
TRAP_BACK_X = 60.0
TRAP_HALF_Y = 15.0
TRAP_HALF_Z = 3.0
CHANNEL_FAR_X = -100.0
CHANNEL_HALF_LENGTH = 120.0
CELL_RADIUS = 0.5

AHL_SINK_RATE = 5.0
NUTRIENT_TARGET = 10.0
NUTRIENT_SUPPLY_RATE = 2.0
NUTRIENT_DECAY_RATE = 0.5
BASE_GROWTH_RATE = 1.0
NUTRIENT_K = 5.0


def _grid() -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 64, 72, 4
    grid = SignalGridSpec()
    grid.signal_count = 2
    grid.shape = shape
    grid.origin = Vec3(-140.0, -144.0, -8.0)
    grid.spacing = Vec3(4.0, 4.0, 4.0)
    grid.diffusion = [40.0, 20.0]
    grid.advection = [Vec3(), Vec3()]
    grid.integration = SignalIntegrationKind.CRANK_NICOLSON
    grid.solver.absolute_tolerance = 1.0e-12

    site_count = shape.x * shape.y * shape.z
    source_rates = [0.0] * (2 * site_count)
    loss_rates = [0.0] * (2 * site_count)
    for x in range(shape.x):
        outside = grid.origin.x + x * grid.spacing.x < TRAP_OPEN_X
        for y in range(shape.y):
            for z in range(shape.z):
                site = x * shape.y * shape.z + y * shape.z + z
                if outside:
                    loss_rates[site] = AHL_SINK_RATE
                    loss_rates[site_count + site] = NUTRIENT_DECAY_RATE
                else:
                    source_rates[site_count + site] = NUTRIENT_SUPPLY_RATE * NUTRIENT_TARGET
                    loss_rates[site_count + site] = NUTRIENT_SUPPLY_RATE
    reaction = SignalGridAffineReaction()
    reaction.source_rates = source_rates
    reaction.loss_rates = loss_rates
    grid.reaction = reaction
    return grid


def _rate_plan() -> CoupledRatePlan:
    rates = RatePlanBuilder()
    luxi = rates.maximum(rates.species(0), 0.0)
    aiia = rates.maximum(rates.species(1), 0.0)
    gfp = rates.maximum(rates.species(2), 0.0)
    ahl = rates.maximum(rates.signal(0), 0.0)
    ahl_cubed = ahl**3.0
    hill = ahl_cubed / (8.0 + ahl_cubed)
    activated = 0.02 + 8.0 * hill
    return rates.coupled_plan(
        3,
        2,
        (
            activated - 1.2 * luxi,
            activated - 0.3 * aiia,
            activated - 0.5 * gfp,
        ),
        (8.0 * luxi - 4.0 * aiia * ahl, rates.constant(0.0)),
    )


def _add_plane(
    simulation: Simulation,
    point: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> None:
    plane = PlaneConstraintInit()
    plane.point = Vec3(*point)
    plane.inward_normal = Vec3(*normal)
    plane.coefficient = 1.0
    simulation.add_plane_constraint(plane)


def _add_sphere(simulation: Simulation, center: tuple[float, float, float]) -> None:
    sphere = SphereConstraintInit()
    sphere.center = Vec3(*center)
    sphere.radius = CELL_RADIUS
    sphere.coefficient = 1.0
    sphere.allowed_region = SphereRegion.OUTSIDE
    simulation.add_sphere_constraint(sphere)


def _wall(
    simulation: Simulation,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> None:
    delta = tuple(right - left for left, right in zip(start, end, strict=True))
    length = math.sqrt(sum(value * value for value in delta))
    count = max(2, math.ceil(length / CELL_RADIUS) + 1)
    for index in range(count):
        fraction = index / (count - 1)
        center = (
            start[0] + fraction * delta[0],
            start[1] + fraction * delta[1],
            start[2] + fraction * delta[2],
        )
        _add_sphere(simulation, center)


def _add_trap(simulation: Simulation) -> None:
    setback = 3.0
    radius = CELL_RADIUS
    _wall(
        simulation,
        (TRAP_OPEN_X + setback, -TRAP_HALF_Y - radius, 0.0),
        (TRAP_BACK_X, -TRAP_HALF_Y - radius, 0.0),
    )
    _wall(
        simulation,
        (TRAP_OPEN_X + setback, TRAP_HALF_Y + radius, 0.0),
        (TRAP_BACK_X, TRAP_HALF_Y + radius, 0.0),
    )
    _wall(
        simulation,
        (TRAP_BACK_X + radius, -TRAP_HALF_Y, 0.0),
        (TRAP_BACK_X + radius, TRAP_HALF_Y, 0.0),
    )
    _add_plane(simulation, (CHANNEL_FAR_X, 0.0, 0.0), (1.0, 0.0, 0.0))
    _wall(
        simulation,
        (TRAP_OPEN_X, -CHANNEL_HALF_LENGTH + 3.0, 0.0),
        (TRAP_OPEN_X, -TRAP_HALF_Y, 0.0),
    )
    _wall(
        simulation,
        (TRAP_OPEN_X, TRAP_HALF_Y, 0.0),
        (TRAP_OPEN_X, CHANNEL_HALF_LENGTH - 3.0, 0.0),
    )
    for y in (-TRAP_HALF_Y, TRAP_HALF_Y):
        outer_y = y - radius if y < 0.0 else y + radius
        _wall(
            simulation,
            (TRAP_OPEN_X + setback, outer_y, 0.0),
            (TRAP_OPEN_X, outer_y, 0.0),
        )
        _wall(
            simulation,
            (TRAP_OPEN_X, outer_y, 0.0),
            (TRAP_OPEN_X, y, 0.0),
        )
    _add_plane(simulation, (0.0, 0.0, TRAP_HALF_Z), (0.0, 0.0, -1.0))
    _add_plane(simulation, (0.0, 0.0, -TRAP_HALF_Z), (0.0, 0.0, 1.0))


def _nutrient_growth(simulation: Simulation, position: Vec3) -> float:
    nutrient = max(0.0, simulation.sample_signals(position)[1])
    return BASE_GROWTH_RATE * nutrient / (NUTRIENT_K + nutrient)


def _regulate(step: ControllerStep) -> StepPlan:
    return StepPlan(
        updates=tuple(
            CellUpdate(cell.id, growth_rate=_nutrient_growth(step.simulation, cell.position))
            for cell in step.cells
        ),
        divisions=DIVISION.requests(step),
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
    simulation.configure_signal_grid(_grid())
    simulation.set_coupled_rate_plan(_rate_plan())
    _add_trap(simulation)

    founder = CellInit()
    founder.position = Vec3(TRAP_BACK_X - 5.0, 0.0, 0.0)
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
        on_division=_divided,
    )
