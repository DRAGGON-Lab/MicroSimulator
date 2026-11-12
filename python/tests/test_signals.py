from __future__ import annotations

import math
from pathlib import Path

import pytest
from microsimulator import (
    BackendFeature,
    BackendKind,
    CellInit,
    CoupledRatePlan,
    GridBoundary,
    GridBoundaryKind,
    GridFace,
    GridShape,
    RateInstruction,
    RateOp,
    RatePlanBuilder,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    Vec3,
    backend_available,
)
from microsimulator.checkpoint import load_checkpoint, save_checkpoint


def _line_spec(length: int = 3) -> SignalGridSpec:
    shape = GridShape()
    shape.x = length
    shape.y = 1
    shape.z = 1
    spec = SignalGridSpec()
    spec.signal_count = 1
    spec.shape = shape
    spec.diffusion = [1.0]
    spec.advection = [Vec3()]
    return spec


def _assert_levels(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert math.isclose(left, right, rel_tol=1.0e-6, abs_tol=1.0e-6)


def _instruction(operation: RateOp, *, first: int = 0, value: float = 0.0) -> RateInstruction:
    instruction = RateInstruction()
    instruction.operation = operation
    instruction.first = first
    instruction.value = value
    return instruction


def test_cpu_signal_transport_sampling_and_stability() -> None:
    simulation = Simulation()
    simulation.configure_signal_grid(_line_spec(), [0.0, 1.0, 0.0])

    assert simulation.has_signal_grid
    assert simulation.signal_count == 1
    assert simulation.supports(BackendFeature.SIGNALS)
    assert simulation.sample_signals(Vec3(0.5, 0.0, 0.0)) == [0.5]

    simulation.step(0.25)
    _assert_levels(simulation.signal_levels, [0.25, 0.5, 0.25])
    assert math.isclose(sum(simulation.signal_levels), 1.0)
    before = simulation.signal_levels
    before_time = simulation.time
    with pytest.raises(ValueError, match="stability"):
        simulation.step(0.51)
    assert simulation.signal_levels == before
    assert simulation.time == before_time
    with pytest.raises(IndexError, match="outside"):
        simulation.sample_signals(Vec3(3.0, 0.0, 0.0))


def test_cpu_crank_nicolson_accepts_a_step_beyond_the_euler_bound() -> None:
    spec = _line_spec()
    spec.integration = SignalIntegrationKind.CRANK_NICOLSON
    # The relative tolerance asks for a reduction of the residual the step
    # starts with, so it sets how closely the committed field approaches the
    # exact one: this solve is checked to a millionth, so it asks for rather
    # better than that.
    spec.solver.relative_tolerance = 1.0e-7
    simulation = Simulation()
    simulation.configure_signal_grid(spec, [0.0, 1.0, 0.0])

    simulation.step(1.0)

    _assert_levels(simulation.signal_levels, [0.4, 0.2, 0.4])
    report = simulation.last_signal_solve_report
    assert report is not None
    assert report.converged
    assert report.iterations > 0
    assert report.residual_rms <= 2.0e-5


def _uptake_removed(background: float, integration: SignalIntegrationKind) -> float:
    """Total signal a single growing cell removes from a uniform field in one step."""

    rates = RatePlanBuilder()
    uptake = -(rates.growth_rate() * rates.cell_volume())
    plan = rates.coupled_plan(0, 1, (), (uptake,))

    shape = GridShape()
    shape.x, shape.y, shape.z = 4, 4, 4
    spec = SignalGridSpec()
    spec.signal_count = 1
    spec.shape = shape
    spec.spacing = Vec3(4.0, 4.0, 4.0)
    spec.diffusion = [0.0]
    spec.advection = [Vec3()]
    spec.integration = integration

    simulation = Simulation()
    simulation.configure_signal_grid(spec, [background] * 64)
    simulation.set_coupled_rate_plan(plan)
    cell = CellInit()
    cell.position = Vec3(4.0, 4.0, 4.0)
    cell.length = 2.6
    cell.radius = 0.5
    cell.growth_rate = 0.667
    simulation.add_cell(cell)

    before = sum(simulation.signal_levels)
    simulation.step(0.02)
    return (before - sum(simulation.signal_levels)) * spec.voxel_volume


@pytest.mark.parametrize("background", [1.0, 10.0, 100.0])
def test_cell_sources_reach_the_grid_at_any_resolvable_background(background: float) -> None:
    """A cell's exchange with the field must survive the convergence test.

    An implicit step is accepted on a residual, and a cell's contribution is
    small next to a well-stocked field. Judging that residual against the field
    would let the contribution fall under the threshold and be dropped, leaving
    the model silently inert, so it is judged against the residual the step
    starts with instead. Forward Euler applies its sources unconditionally and
    is the reference here.
    """

    explicit = _uptake_removed(background, SignalIntegrationKind.FORWARD_EULER)
    implicit = _uptake_removed(background, SignalIntegrationKind.CRANK_NICOLSON)
    assert math.isclose(implicit, explicit, rel_tol=1.0e-3)
    assert implicit > 0.0


def test_a_source_below_the_field_noise_commits_a_converged_step() -> None:
    """The limit of an implicit solve: a source it cannot see is not an error.

    Convergence is decided on a residual computed in float32, so a source that
    moves the field by less than its own representable resolution leaves no
    residual to detect. The step still converges and commits rather than
    failing; a model whose exchange is that small next to its background needs
    forward Euler or a concentration scale that resolves it.
    """

    removed = _uptake_removed(1.0e4, SignalIntegrationKind.CRANK_NICOLSON)
    assert removed == 0.0


def test_cell_sources_match_their_declared_amount() -> None:
    cell_volume = math.pi * 0.5**2 * (2.6 + 2.0 * 0.5)
    expected = 0.02 * 0.667 * cell_volume
    for integration in (
        SignalIntegrationKind.FORWARD_EULER,
        SignalIntegrationKind.CRANK_NICOLSON,
    ):
        assert math.isclose(_uptake_removed(10.0, integration), expected, rel_tol=0.02)


def test_fixed_and_periodic_boundaries_are_explicit() -> None:
    fixed = _line_spec(2)
    lower = GridBoundary()
    lower.kind = GridBoundaryKind.FIXED
    lower.values = [2.0]
    fixed.x_lower = lower
    simulation = Simulation()
    simulation.configure_signal_grid(fixed, [0.0, 0.0])
    simulation.step(0.25)
    _assert_levels(simulation.signal_levels, [0.5, 0.0])

    periodic = _line_spec(4)
    periodic.diffusion = [0.0]
    periodic.advection = [Vec3(1.0, 0.0, 0.0)]
    lower = GridBoundary()
    lower.kind = GridBoundaryKind.PERIODIC
    upper = GridBoundary()
    upper.kind = GridBoundaryKind.PERIODIC
    periodic.x_lower = lower
    periodic.x_upper = upper
    simulation = Simulation()
    simulation.configure_signal_grid(periodic, [1.0, 0.0, 0.0, 0.0])
    simulation.step(0.5)
    _assert_levels(simulation.signal_levels, [0.5, 0.5, 0.0, 0.0])


def _inlet_line(inlet: float, backend: BackendKind = BackendKind.CPU) -> Simulation:
    """A line of fluid fed through a fixed lower x face and closed at the other end."""

    spec = _line_spec(4)
    lower = GridBoundary()
    lower.kind = GridBoundaryKind.FIXED
    lower.values = [inlet]
    spec.x_lower = lower
    simulation = Simulation(backend)
    simulation.configure_signal_grid(spec, [1.0, 0.5, 0.0, 0.0])
    return simulation


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.METAL, BackendKind.CUDA])
def test_fixed_boundary_values_set_during_a_run_act_like_configured_ones(
    backend: BackendKind,
) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend.name} backend is unavailable")
    switched = _inlet_line(0.0, backend)
    for _ in range(3):
        switched.step(0.1)
    before_switch = switched.signal_levels

    # From the switch on, the run is the one a grid configured with the new
    # inlet value would make from the same field.
    reference = _inlet_line(4.0, backend)
    reference.set_signal_levels(before_switch)
    switched.set_signal_boundary_values(GridFace.X_LOWER, [4.0])
    for _ in range(5):
        switched.step(0.1)
        reference.step(0.1)
    assert switched.signal_levels == reference.signal_levels
    assert switched.signal_levels[0] > before_switch[0]

    for _ in range(2000):
        switched.step(0.1)
    # The closed far end leaves the inlet value as the only steady state.
    assert all(math.isclose(level, 4.0, rel_tol=1.0e-4) for level in switched.signal_levels)


def test_only_a_fixed_boundary_accepts_valid_values() -> None:
    simulation = _inlet_line(2.0)
    with pytest.raises(RuntimeError, match="only a fixed signal grid boundary"):
        simulation.set_signal_boundary_values(GridFace.X_UPPER, [1.0])
    with pytest.raises(ValueError, match="must match signal count"):
        simulation.set_signal_boundary_values(GridFace.X_LOWER, [1.0, 1.0])
    with pytest.raises(ValueError, match="finite and non-negative"):
        simulation.set_signal_boundary_values(GridFace.X_LOWER, [-1.0])
    with pytest.raises(RuntimeError, match="does not have a signal grid"):
        Simulation().set_signal_boundary_values(GridFace.X_LOWER, [1.0])

    # A rejected call leaves the configured inlet in place.
    untouched = _inlet_line(2.0)
    simulation.step(0.1)
    untouched.step(0.1)
    assert simulation.signal_levels == untouched.signal_levels


def test_boundary_values_set_during_a_run_survive_a_checkpoint(tmp_path: Path) -> None:
    simulation = _inlet_line(0.0)
    simulation.step(0.1)
    simulation.set_signal_boundary_values(GridFace.X_LOWER, [3.0])
    simulation.step(0.1)
    path = tmp_path / "inlet.msim"
    save_checkpoint(simulation, path)
    restored = load_checkpoint(path)
    for _ in range(4):
        simulation.step(0.1)
        restored.step(0.1)
    assert restored.signal_levels == simulation.signal_levels
    assert restored.signal_levels[0] > 1.0


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.METAL, BackendKind.CUDA])
def test_an_inlet_pulse_is_carried_downstream_by_the_flow(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend.name} backend is unavailable")
    length, speed, dt = 40, 2.0, 0.05
    spec = _line_spec(length)
    spec.diffusion = [0.01]
    spec.advection = [Vec3(speed, 0.0, 0.0)]
    for name in ("x_lower", "x_upper"):
        boundary = GridBoundary()
        boundary.kind = GridBoundaryKind.FIXED
        boundary.values = [0.0]
        setattr(spec, name, boundary)
    simulation = Simulation(backend)
    simulation.configure_signal_grid(spec, [0.0] * length)

    probe = 20
    pulse_start, pulse_end = 1.0, 13.0
    arrival: float | None = None
    departure: float | None = None
    peak = 0.0
    for step in range(600):
        time = step * dt
        inlet = 1.0 if pulse_start <= time < pulse_end else 0.0
        simulation.set_signal_boundary_values(GridFace.X_LOWER, [inlet])
        simulation.step(dt)
        level = simulation.signal_levels[probe]
        peak = max(peak, level)
        if arrival is None and level >= 0.5:
            arrival = time + dt
        if arrival is not None and departure is None and level < 0.5:
            departure = time + dt

    # Site centers sit at integer x and the inlet face at -0.5, so the pulse
    # edges reach the probe a transit time after they pass the inlet.
    transit = (probe + 0.5) / speed
    assert arrival is not None and departure is not None
    assert abs(arrival - (pulse_start + transit)) < 0.1 * transit
    assert abs(departure - (pulse_end + transit)) < 0.1 * transit
    assert peak > 0.95
    assert simulation.signal_levels[probe] < 0.01


def test_gpu_signal_paths_are_native_or_fail_before_mutation() -> None:
    spec = _line_spec()
    reference = Simulation()
    reference.configure_signal_grid(spec, [0.0, 1.0, 0.0])
    reference.step(0.25)
    for backend in (BackendKind.METAL, BackendKind.CUDA):
        if not backend_available(backend):
            continue
        simulation = Simulation(backend)
        simulation.configure_signal_grid(spec, [0.0, 1.0, 0.0])
        if simulation.supports(BackendFeature.SIGNALS):
            simulation.step(0.25)
            _assert_levels(simulation.signal_levels, reference.signal_levels)
            assert simulation.time == reference.time
        else:
            with pytest.raises(RuntimeError, match="does not implement signal grid"):
                simulation.step(0.25)
            assert simulation.time == 0.0
            assert simulation.signal_levels == [0.0, 1.0, 0.0]


def test_gpu_coupling_is_native_or_fails_before_growth() -> None:
    for backend in BackendKind:
        if not backend_available(backend):
            continue
        simulation = Simulation(backend, species_count=1)
        simulation.configure_signal_grid(_line_spec(), [0.0, 1.0, 0.0])
        cell = CellInit()
        cell.position = Vec3(1.0, 0.0, 0.0)
        cell.growth_rate = 0.0
        cell.species = [1.0]
        cell_id = simulation.add_cell(cell)
        simulation.set_coupled_rate_plan(
            CoupledRatePlan(
                1,
                1,
                [
                    _instruction(RateOp.SIGNAL, first=0),
                    _instruction(RateOp.CONSTANT, value=0.0),
                ],
                [0],
                [1],
            )
        )
        if simulation.supports(BackendFeature.COUPLED_RATES):
            simulation.step(0.25)
            assert math.isclose(simulation.cell(cell_id).species[0], 1.25)
            _assert_levels(simulation.signal_levels, [0.25, 0.5, 0.25])
        else:
            with pytest.raises(RuntimeError, match="does not implement coupled rates"):
                simulation.step(0.25)
            assert simulation.cell(cell_id).species == [1.0]
            assert simulation.signal_levels == [0.0, 1.0, 0.0]


def test_native_coupling_handles_empty_grids_and_atomic_failure() -> None:
    for backend in BackendKind:
        if not backend_available(backend):
            continue
        empty = Simulation(backend)
        empty.configure_signal_grid(_line_spec(), [0.0, 1.0, 0.0])
        empty.set_coupled_rate_plan(
            CoupledRatePlan(0, 1, [_instruction(RateOp.CONSTANT, value=0.0)], [], [0])
        )
        if not empty.supports(BackendFeature.COUPLED_RATES):
            continue
        empty.step(0.25)
        _assert_levels(empty.signal_levels, [0.25, 0.5, 0.25])

        failing = Simulation(backend, species_count=1)
        spec = _line_spec()
        spec.diffusion = [0.0]
        failing.configure_signal_grid(spec, [0.0, 0.0, 0.0])
        cell = CellInit()
        cell.position = Vec3(1.0, 0.0, 0.0)
        cell.growth_rate = 0.0
        cell.species = [2.0]
        cell_id = failing.add_cell(cell)
        failing.set_coupled_rate_plan(
            CoupledRatePlan(
                1,
                1,
                [
                    _instruction(RateOp.CONSTANT, value=0.0),
                    _instruction(RateOp.CONSTANT, value=-1.0),
                ],
                [0],
                [1],
            )
        )
        with pytest.raises(ValueError, match=r"negative|invalid"):
            failing.step(0.25)
        assert failing.cell(cell_id).species == [2.0]
        assert failing.signal_levels == [0.0, 0.0, 0.0]
        assert failing.time == 0.0
