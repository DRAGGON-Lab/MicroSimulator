"""Growth that responds to compression, natively and through the controller."""

from __future__ import annotations

import math
import random

import pytest
from microsimulator import (
    BackendKind,
    BoxConstraintInit,
    CellInit,
    ConstraintRegion,
    ControllerStep,
    DivisionEvent,
    DivisionRequest,
    GridShape,
    GrowthLimitParameters,
    MechanicsConfig,
    NativeController,
    RatePlanBuilder,
    SignalGridSpec,
    Simulation,
    StepPlan,
    UniformLengthDivision,
    Vec3,
    backend_available,
)

_BACKENDS = [BackendKind.CPU, BackendKind.METAL, BackendKind.CUDA]
_RADIUS = 0.5


def _limit(free_overlap: float = 0.1, stall_overlap: float = 0.5) -> GrowthLimitParameters:
    limit = GrowthLimitParameters()
    limit.free_overlap = free_overlap
    limit.stall_overlap = stall_overlap
    return limit


def _pair(simulation: Simulation, overlap_radii: float) -> None:
    """Two parallel rods side by side, overlapping by a number of radii."""

    cell = CellInit()
    cell.length = 2.0
    cell.radius = _RADIUS
    cell.growth_rate = 1.0
    cell.position = Vec3(4.0, 4.0, 0.5)
    simulation.add_cell(cell)
    cell.position = Vec3(4.0, 4.0 + 2.0 * _RADIUS - overlap_radii * _RADIUS, 0.5)
    simulation.add_cell(cell)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    ("overlap_radii", "kept"),
    [(0.0, 1.0), (0.1, 1.0), (0.3, 0.5), (0.5, 0.0), (0.9, 0.0)],
)
def test_elongation_falls_linearly_between_the_free_and_stall_overlaps(
    backend: BackendKind, overlap_radii: float, kept: float
) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend.name} backend is unavailable")
    simulation = Simulation(backend)
    _pair(simulation, overlap_radii)
    simulation.step(0.1, _limit())
    for cell in simulation.cells():
        assert math.isclose(cell.length, 2.0 + kept * 0.2, abs_tol=1.0e-4)
    assert math.isclose(simulation.time, 0.1, rel_tol=1.0e-6)


def test_cells_out_of_contact_grow_as_in_an_unlimited_step() -> None:
    limited = Simulation()
    unlimited = Simulation()
    for simulation in (limited, unlimited):
        _pair(simulation, -2.0)
    limited.step(0.1, _limit())
    unlimited.step(0.1)
    assert [cell.length for cell in limited.cells()] == [cell.length for cell in unlimited.cells()]


def test_invalid_growth_limits_are_rejected_before_the_step() -> None:
    simulation = Simulation()
    _pair(simulation, 0.2)
    for free_overlap, stall_overlap in ((-0.1, 0.4), (0.4, 0.4), (0.5, 0.4), (0.1, math.inf)):
        with pytest.raises(ValueError, match="growth limit overlaps"):
            simulation.step(0.1, _limit(free_overlap, stall_overlap))
    assert simulation.time == 0.0
    assert [cell.length for cell in simulation.cells()] == [2.0, 2.0]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_nutrient_is_charged_only_for_growth_that_happened(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend.name} backend is unavailable")

    def consumed(overlap_radii: float) -> float:
        shape = GridShape()
        shape.x, shape.y, shape.z = 9, 9, 2
        spec = SignalGridSpec()
        spec.signal_count = 1
        spec.shape = shape
        spec.diffusion = [0.0]
        spec.advection = [Vec3()]
        simulation = Simulation(backend)
        levels = [10.0] * (9 * 9 * 2)
        simulation.configure_signal_grid(spec, levels)
        rates = RatePlanBuilder()
        simulation.set_coupled_rate_plan(
            rates.coupled_plan(0, 1, (), (-rates.cell_volume_change_rate(),))
        )
        _pair(simulation, overlap_radii)
        simulation.step(0.05, _limit())
        return sum(levels) - sum(simulation.signal_levels)

    free = consumed(0.0)
    assert free > 0.0
    assert math.isclose(consumed(0.3), 0.5 * free, rel_tol=1.0e-3)
    assert abs(consumed(0.6)) <= 1.0e-4 * free


def test_mechanics_config_carries_the_growth_limit_through_json() -> None:
    assert MechanicsConfig().growth_limit_parameters() is None

    configuration = MechanicsConfig(
        growth_limit=True, growth_free_overlap=0.05, growth_stall_overlap=0.3
    )
    assert MechanicsConfig.from_json(configuration.to_json()) == configuration
    limit = configuration.growth_limit_parameters()
    assert limit is not None
    assert math.isclose(limit.free_overlap, 0.05, rel_tol=1.0e-6)
    assert math.isclose(limit.stall_overlap, 0.3, rel_tol=1.0e-6)
    assert math.isclose(
        limit.contacts.activation_margin, configuration.contact_activation_margin, rel_tol=1.0e-6
    )

    # A configuration saved without the growth limit resumes with it off.
    legacy = configuration.to_json()
    for name in ("growth_limit", "growth_free_overlap", "growth_stall_overlap"):
        del legacy[name]
    assert MechanicsConfig.from_json(legacy) == MechanicsConfig()

    with pytest.raises(ValueError, match="non-negative and increasing"):
        MechanicsConfig(growth_free_overlap=0.4, growth_stall_overlap=0.4)
    invalid = configuration.to_json()
    invalid["growth_limit"] = 1
    with pytest.raises(ValueError, match=r"mechanics\.growth_limit must be Boolean"):
        MechanicsConfig.from_json(invalid)
    unknown = configuration.to_json()
    unknown["growth_pressure"] = 1.0
    with pytest.raises(ValueError, match="mechanics configuration is invalid"):
        MechanicsConfig.from_json(unknown)


def _chamber(mechanics: MechanicsConfig) -> NativeController:
    """One founder in a closed 4 by 4 chamber a cell and a half deep."""

    simulation = Simulation()
    chamber = BoxConstraintInit()
    chamber.half_extents = Vec3(2.0, 2.0, 0.6)
    chamber.allowed_region = ConstraintRegion.INSIDE
    simulation.add_box_constraint(chamber)
    founder = CellInit()
    founder.length = 1.0
    founder.radius = 0.4
    founder.growth_rate = 1.0
    simulation.add_cell(founder)
    rng = random.Random(0)

    def regulate(step: ControllerStep) -> StepPlan:
        return StepPlan(
            divisions=tuple(DivisionRequest(cell.id) for cell in step.cells if cell.length > 2.0)
        )

    def divided(step: ControllerStep, event: DivisionEvent) -> None:
        # Daughters leave the division slightly splayed so a pair spanning the
        # chamber can turn instead of staying end to end.
        sign = 1.0 if step.rng.randrange(2) else -1.0
        for cell, turn in ((event.first, sign), (event.second, -sign)):
            angle = math.atan2(cell.direction.y, cell.direction.x) + 0.01 * turn
            step.simulation.set_cell_geometry(
                cell.id, cell.position, Vec3(math.cos(angle), math.sin(angle), 0.0), cell.length
            )

    return NativeController(
        simulation,
        model_id="tests.closed-chamber",
        model_version=1,
        rng=rng,
        regulate=regulate,
        on_division=divided,
        mechanics=mechanics,
    )


def _floor_coverage(simulation: Simulation) -> float:
    return (
        sum(
            math.pi * cell.radius**2 + 2.0 * cell.radius * cell.length
            for cell in simulation.cells()
        )
        / 16.0
    )


def test_a_closed_chamber_fills_and_then_holds_its_population() -> None:
    limited = _chamber(MechanicsConfig(growth_limit=True))
    unlimited = _chamber(MechanicsConfig())
    for _ in range(200):
        limited.step(0.05)
        unlimited.step(0.05)
    # Growth that ignores contact has packed the chamber many times over.
    assert unlimited.simulation.cell_count >= 200
    assert _floor_coverage(unlimited.simulation) > 10.0

    # The limited colony used the floor before it stopped, and stays put.
    filled = limited.simulation.cell_count
    assert 8 <= filled <= 20
    assert _floor_coverage(limited.simulation) > 0.75
    for _ in range(400):
        limited.step(0.05)
    assert limited.simulation.cell_count <= filled + 4
    assert _floor_coverage(limited.simulation) < 1.5


def test_an_unconfined_colony_keeps_most_of_its_growth() -> None:
    division = UniformLengthDivision(3.2, 3.8, jitter_z=False)

    def total_length(mechanics: MechanicsConfig) -> float:
        simulation = Simulation(reserved_capacity=2_000)
        rng = random.Random(0)
        founder = CellInit()
        founder.length = 3.0
        founder.radius = _RADIUS
        founder.growth_rate = 1.0
        state: dict[str, object] = {}
        division.initialize_founders(simulation, state, rng, (founder,))  # type: ignore[arg-type]
        controller = NativeController(
            simulation,
            model_id="tests.unconfined-colony",
            model_version=1,
            rng=rng,
            regulate=lambda step: StepPlan(divisions=division.requests(step)),
            on_division=division.on_division,
            mechanics=mechanics,
            state=state,  # type: ignore[arg-type]
        )
        for _ in range(160):
            controller.step(0.05)
        return sum(cell.length + 2.0 * cell.radius for cell in simulation.cells())

    # Relaxation stiff enough to keep free neighbors apart leaves little
    # overlap for the limit to respond to.
    unlimited = total_length(MechanicsConfig(gamma=100.0))
    limited = total_length(MechanicsConfig(gamma=100.0, growth_limit=True))
    assert limited <= unlimited
    assert limited >= 0.85 * unlimited
