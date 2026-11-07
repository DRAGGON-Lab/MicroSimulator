"""Signed growth, retry reporting, and transient solve state."""

from __future__ import annotations

import math

import pytest
from microsimulator import (
    BackendKind,
    CellInit,
    ConstraintContactParameters,
    ContactParameters,
    GridShape,
    MechanicsIntegrationParameters,
    MechanicsParameters,
    PlaneConstraintInit,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    SolverStatus,
    Vec3,
    backend_available,
)


def _step(simulation: Simulation, dt: float, *, passes: int = 1):
    return simulation.step_with_mechanics(
        dt,
        MechanicsParameters(),
        ContactParameters(),
        MechanicsIntegrationParameters(),
        ConstraintContactParameters(),
        passes,
    )


def _walled_cell(
    backend: BackendKind, half_width: float, growth_rate: float
) -> tuple[Simulation, int]:
    simulation = Simulation(backend)
    for x, normal in ((-half_width, 1.0), (half_width, -1.0)):
        wall = PlaneConstraintInit()
        wall.point = Vec3(x, 0.0, 0.0)
        wall.inward_normal = Vec3(normal, 0.0, 0.0)
        simulation.add_plane_constraint(wall)
    cell = CellInit()
    cell.length = 1.0
    cell.radius = 0.2
    cell.growth_rate = growth_rate
    cell.fixed = True
    return simulation, simulation.add_cell(cell)


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.CUDA])
@pytest.mark.parametrize("axis_x", [0.0, 0.1])
def test_parallel_wall_contacts_move_cell_once(backend: BackendKind, axis_x: float) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    simulation = Simulation(backend)
    wall = PlaneConstraintInit()
    wall.inward_normal = Vec3(1.0, 0.0, 0.0)
    simulation.add_plane_constraint(wall)
    cell = CellInit()
    cell.position = Vec3(0.1, 0.0, 0.0)
    cell.direction = Vec3(axis_x, 1.0, 0.0)
    cell.length = 1.0
    cell.radius = 0.2
    cell.growth_rate = 0.0
    cell_id = simulation.add_cell(cell)
    assert len(simulation.find_external_contacts().contacts) == 2

    _step(simulation, 0.0)

    expected_x = 0.2 + 0.5 * axis_x / math.hypot(axis_x, 1.0) + 8e-5
    assert simulation.cell(cell_id).position.x == pytest.approx(expected_x, abs=2e-6)
    assert all(
        contact.signed_separation >= -1e-4
        for contact in simulation.find_external_contacts().contacts
    )


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.CUDA])
def test_corner_contacts_are_recomputed_after_each_wall_move(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    simulation = Simulation(backend)
    for normal in (Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)):
        wall = PlaneConstraintInit()
        wall.inward_normal = normal
        simulation.add_plane_constraint(wall)
    cell = CellInit()
    cell.position = Vec3(0.1, 0.1, 0.0)
    cell.direction = Vec3(0.0, 0.0, 1.0)
    cell.length = 1.0
    cell.radius = 0.2
    cell.growth_rate = 0.0
    cell_id = simulation.add_cell(cell)

    _step(simulation, 0.0)

    after = simulation.cell(cell_id)
    assert after.position.x == pytest.approx(0.20008, abs=2e-6)
    assert after.position.y == pytest.approx(0.20008, abs=2e-6)


def test_fixed_penetrating_cell_does_not_move() -> None:
    simulation = Simulation(BackendKind.CPU)
    wall = PlaneConstraintInit()
    wall.inward_normal = Vec3(1.0, 0.0, 0.0)
    simulation.add_plane_constraint(wall)
    cell = CellInit()
    cell.position = Vec3(0.1, 0.0, 0.0)
    cell.direction = Vec3(0.0, 1.0, 0.0)
    cell.radius = 0.2
    cell.fixed = True
    cell_id = simulation.add_cell(cell)
    with pytest.raises(RuntimeError, match="infeasible wall geometry"):
        _step(simulation, 0.0)
    assert simulation.cell(cell_id).position.x == pytest.approx(0.1)


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.CUDA])
def test_fixed_cell_grows_without_moving(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    simulation, cell_id = _walled_cell(backend, 2.0, 0.5)
    before = simulation.cell(cell_id)
    reports = _step(simulation, 1.0)
    after = simulation.cell(cell_id)
    assert len(reports) == 1
    assert after.length == pytest.approx(1.5)
    assert (after.position.x, after.position.y, after.position.z) == (
        before.position.x,
        before.position.y,
        before.position.z,
    )


@pytest.mark.parametrize("fixed", [False, True])
def test_negative_growth_shrinks_cell(fixed: bool) -> None:
    simulation = Simulation(BackendKind.CPU)
    cell = CellInit()
    cell.length = 1.0
    cell.growth_rate = -0.5
    cell.fixed = fixed
    cell_id = simulation.add_cell(cell)
    _step(simulation, 1.0)
    assert simulation.cell(cell_id).length == pytest.approx(0.5)


def test_growth_can_shrink_to_zero_and_rejects_negative_length() -> None:
    simulation = Simulation(BackendKind.CPU)
    cell = CellInit()
    cell.length = 1.0
    cell.growth_rate = -1.0
    cell_id = simulation.add_cell(cell)
    _step(simulation, 1.0)
    assert simulation.cell(cell_id).length == 0.0
    assert simulation.time == pytest.approx(1.0)

    invalid = Simulation(BackendKind.CPU)
    cell.growth_rate = -2.0
    invalid_id = invalid.add_cell(cell)
    with pytest.raises(OverflowError, match="invalid cell length"):
        _step(invalid, 1.0)
    assert invalid.cell(invalid_id).length == 1.0
    assert invalid.time == 0.0


def test_retry_applies_all_passes_to_fixed_growth() -> None:
    simulation, cell_id = _walled_cell(BackendKind.CPU, 0.8, 1.0)
    reports = _step(simulation, 1.0, passes=2)
    assert len(reports) == 2
    assert simulation.cell(cell_id).length == pytest.approx(1.125)
    assert all(
        edge.signed_separation >= -1e-4 for edge in simulation.find_external_contacts().contacts
    )


def test_infeasible_positive_growth_does_not_report_unused_solves() -> None:
    simulation, cell_id = _walled_cell(BackendKind.CPU, 0.7, 1.0)
    reports = _step(simulation, 1.0, passes=2)
    assert reports == []
    assert simulation.cell(cell_id).length == pytest.approx(1.0)


def test_infeasible_positive_growth_keeps_other_cell_shrinkage() -> None:
    simulation, growing_id = _walled_cell(BackendKind.CPU, 0.7, 1.0)
    shrinking = CellInit()
    shrinking.position = Vec3(0.0, 2.0, 0.0)
    shrinking.length = 1.0
    shrinking.radius = 0.2
    shrinking.growth_rate = -0.5
    shrinking_id = simulation.add_cell(shrinking)
    assert _step(simulation, 1.0) == []
    assert simulation.cell(growing_id).length == pytest.approx(1.0)
    assert simulation.cell(shrinking_id).length == pytest.approx(0.5)


def test_each_mechanics_pass_enforces_convergence() -> None:
    def crowded() -> Simulation:
        simulation = Simulation(BackendKind.CPU)
        for x in (-0.3, 0.0, 0.3):
            cell = CellInit()
            cell.position = Vec3(x, 0.0, 0.0)
            cell.length = 0.5
            cell.radius = 0.4
            simulation.add_cell(cell)
        return simulation

    parameters = MechanicsParameters()
    parameters.max_iterations = 1
    parameters.residual_rms_tolerance = 0.0
    integration = MechanicsIntegrationParameters()
    integration.require_convergence = False
    simulation = crowded()
    reports = simulation.step_with_mechanics(
        0.0, parameters, ContactParameters(), integration, ConstraintContactParameters(), 2
    )
    assert len(reports) == 2
    assert all(report.report.status == SolverStatus.ITERATION_LIMIT for report in reports)

    strict = crowded()
    integration.require_convergence = True
    before = strict._checkpoint()
    with pytest.raises(RuntimeError, match="converged solver result"):
        strict.step_with_mechanics(
            0.0, parameters, ContactParameters(), integration, ConstraintContactParameters(), 2
        )
    assert [cell.position.x for cell in strict.cells()] == [
        cell.position.x for cell in before.world.cells
    ]


def test_restore_clears_transient_signal_report() -> None:
    simulation = Simulation(BackendKind.CPU)
    shape = GridShape()
    shape.x, shape.y, shape.z = 3, 1, 1
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.diffusion = [1.0]
    grid.advection = [Vec3()]
    grid.integration = SignalIntegrationKind.BACKWARD_EULER
    simulation.configure_signal_grid(grid, [0.0, 1.0, 0.0])
    checkpoint = simulation._checkpoint()
    simulation.step(0.25)
    assert simulation.last_signal_solve_report is not None
    simulation.restore(checkpoint)
    assert simulation.last_signal_solve_report is None
