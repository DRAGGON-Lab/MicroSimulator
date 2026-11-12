# pyright: reportUnknownMemberType=false
"""Signed growth, retry reporting, and transient solve state."""

from __future__ import annotations

import math

import pytest
from microsimulator import (
    BackendKind,
    CellInit,
    ConstraintContactParameters,
    ContactParameters,
    GridBoundaryKind,
    GridShape,
    MechanicsIntegrationParameters,
    MechanicsParameters,
    PlaneConstraintInit,
    RatePlanBuilder,
    SignalGridSpec,
    SignalGridVelocityField,
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


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.CUDA])
def test_blocked_wall_growth_preserves_distant_growth_and_shrinkage(
    backend: BackendKind,
) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    simulation, blocked_id = _walled_cell(backend, 0.7, 1.0)
    growing = CellInit()
    growing.position = Vec3(0.0, 5.0, 0.0)
    growing.direction = Vec3(0.0, 1.0, 0.0)
    growing.length = 1.0
    growing.radius = 0.2
    growing.growth_rate = 0.5
    growing_id = simulation.add_cell(growing)
    shrinking = CellInit()
    shrinking.position = Vec3(0.0, 10.0, 0.0)
    shrinking.direction = Vec3(0.0, 1.0, 0.0)
    shrinking.length = 1.0
    shrinking.radius = 0.2
    shrinking.growth_rate = -0.5
    shrinking_id = simulation.add_cell(shrinking)

    assert _step(simulation, 1.0, passes=2) == []
    assert simulation.cell(blocked_id).length == pytest.approx(1.0)
    assert simulation.cell(growing_id).length == pytest.approx(1.5)
    assert simulation.cell(shrinking_id).length == pytest.approx(0.5)
    assert simulation.time == pytest.approx(1.0)
    assert all(
        edge.signed_separation >= -1e-4 for edge in simulation.find_external_contacts().contacts
    )
    assert all(edge.signed_separation >= -1e-4 for edge in simulation.find_cell_contacts().contacts)


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.CUDA])
def test_blocked_contact_cluster_preserves_independent_growth(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    simulation = Simulation(backend)
    blocked_ids: list[int] = []
    for y in (0.0, 1.4):
        cell = CellInit()
        cell.position = Vec3(0.0, y, 0.0)
        cell.direction = Vec3(0.0, 1.0, 0.0)
        cell.length = 1.0
        cell.radius = 0.2
        cell.growth_rate = 1.0
        cell.fixed = True
        blocked_ids.append(simulation.add_cell(cell))
    independent = CellInit()
    independent.position = Vec3(0.0, 6.0, 0.0)
    independent.length = 1.0
    independent.radius = 0.2
    independent.growth_rate = 0.5
    independent_id = simulation.add_cell(independent)

    assert _step(simulation, 1.0) == []
    assert [simulation.cell(cell_id).length for cell_id in blocked_ids] == pytest.approx([1.0, 1.0])
    assert simulation.cell(independent_id).length == pytest.approx(1.5)
    assert simulation.time == pytest.approx(1.0)
    assert all(edge.signed_separation >= -1e-4 for edge in simulation.find_cell_contacts().contacts)


def test_coupled_consumption_uses_growth_accepted_by_replay() -> None:
    simulation = Simulation(BackendKind.CPU)
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = GridShape()
    grid.shape.x = grid.shape.y = grid.shape.z = 1
    grid.spacing = Vec3(4.0, 4.0, 4.0)
    grid.diffusion = [0.0]
    grid.advection = [Vec3()]
    simulation.configure_signal_grid(grid, [10.0])
    for x, normal in ((-0.7, 1.0), (0.7, -1.0)):
        wall = PlaneConstraintInit()
        wall.point = Vec3(x, 0.0, 0.0)
        wall.inward_normal = Vec3(normal, 0.0, 0.0)
        simulation.add_plane_constraint(wall)
    blocked = CellInit()
    blocked.length = 1.0
    blocked.radius = 0.2
    blocked.growth_rate = 1.0
    blocked.fixed = True
    blocked_id = simulation.add_cell(blocked)
    rates = RatePlanBuilder()
    simulation.set_coupled_rate_plan(
        rates.coupled_plan(0, 1, (), (-rates.cell_volume_change_rate(),))
    )
    independent = CellInit()
    independent.position = Vec3(0.0, 2.0, 0.0)
    independent.direction = Vec3(0.0, 1.0, 0.0)
    independent.length = 1.0
    independent.radius = 0.2
    independent.growth_rate = 0.5
    independent_id = simulation.add_cell(independent)

    assert _step(simulation, 1.0) == []
    assert simulation.cell(blocked_id).length == pytest.approx(1.0)
    assert simulation.cell(independent_id).length == pytest.approx(1.5)
    consumed = (10.0 - simulation.signal_levels[0]) * grid.voxel_volume
    assert consumed == pytest.approx(math.pi * 0.2**2 * 0.5, abs=3e-5)


def _position_coupled_simulation(
    backend: BackendKind, *, flow_speed: float = 0.0, wall: bool = False
) -> tuple[Simulation, int]:
    simulation = Simulation(backend, species_count=1)
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = GridShape()
    grid.shape.x, grid.shape.y, grid.shape.z = 4, 1, 1
    grid.diffusion = [0.0]
    grid.advection = [Vec3()]
    if flow_speed:
        grid.x_lower.kind = GridBoundaryKind.FIXED
        grid.x_lower.values = [2.0]
        grid.x_upper.kind = GridBoundaryKind.FIXED
        grid.x_upper.values = [8.0]
        velocity = SignalGridVelocityField()
        velocity.x_faces = [flow_speed] * 5
        velocity.y_faces = [0.0] * 8
        velocity.z_faces = [0.0] * 8
        grid.velocity_field = velocity
    simulation.configure_signal_grid(grid, [2.0, 8.0, 8.0, 8.0])
    if wall:
        constraint = PlaneConstraintInit()
        constraint.point = Vec3(0.4, 0.0, 0.0)
        constraint.inward_normal = Vec3(1.0, 0.0, 0.0)
        simulation.add_plane_constraint(constraint)
    cell = CellInit()
    cell.position = Vec3(0.25, 0.0, 0.0)
    cell.direction = Vec3(0.0, 1.0, 0.0)
    cell.length = 1.0
    cell.radius = 0.2
    cell.growth_rate = 0.4
    cell.species = [1.0]
    cell_id = simulation.add_cell(cell)
    rates = RatePlanBuilder()
    simulation.set_coupled_rate_plan(
        rates.coupled_plan(
            1,
            1,
            (rates.signal(0) + rates.position_x(),),
            (rates.constant(1.0),),
        )
    )
    return simulation, cell_id


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.CUDA])
def test_coupled_rates_use_position_before_flow_drift(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    reference, reference_id = _position_coupled_simulation(backend, flow_speed=1.2)
    mechanical, mechanical_id = _position_coupled_simulation(backend, flow_speed=1.2)

    reference.step(0.5)
    mechanical.step_with_mechanics(
        0.5,
        MechanicsParameters(),
        ContactParameters(),
        MechanicsIntegrationParameters(),
        ConstraintContactParameters(),
        1,
        True,
    )

    assert mechanical.cell(mechanical_id).position.x == pytest.approx(0.85, abs=2e-5)
    assert mechanical.cell(mechanical_id).length == pytest.approx(1.2)
    assert mechanical.cell(mechanical_id).species == pytest.approx(
        reference.cell(reference_id).species
    )
    assert mechanical.signal_levels == pytest.approx(reference.signal_levels)
    assert mechanical.last_signal_solve_report is not None


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.CUDA])
def test_coupled_rates_use_position_before_wall_projection(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    reference, reference_id = _position_coupled_simulation(backend, wall=True)
    mechanical, mechanical_id = _position_coupled_simulation(backend, wall=True)

    reference.step(0.5)
    mechanical.step_with_mechanics(
        0.5,
        MechanicsParameters(),
        ContactParameters(),
        MechanicsIntegrationParameters(),
        ConstraintContactParameters(),
        1,
    )

    assert mechanical.cell(mechanical_id).position.x > 0.6
    assert mechanical.cell(mechanical_id).length == pytest.approx(1.2)
    assert mechanical.cell(mechanical_id).species == pytest.approx(
        reference.cell(reference_id).species
    )
    assert mechanical.signal_levels == pytest.approx(reference.signal_levels)


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.CUDA])
def test_failed_coupled_mechanics_restores_geometry_species_and_signal(
    backend: BackendKind,
) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    simulation, cell_id = _position_coupled_simulation(backend, wall=True)
    rates = RatePlanBuilder()
    simulation.set_coupled_rate_plan(
        rates.coupled_plan(1, 1, (rates.constant(1.0),), (rates.constant(-100.0),))
    )
    before_cell = simulation.cell(cell_id)
    before_levels = simulation.signal_levels

    with pytest.raises((ValueError, RuntimeError)):
        _step(simulation, 0.5)

    after_cell = simulation.cell(cell_id)
    assert after_cell.position.x == before_cell.position.x
    assert after_cell.length == before_cell.length
    assert after_cell.species == before_cell.species
    assert simulation.signal_levels == before_levels
    assert simulation.last_signal_solve_report is None
    assert simulation.time == 0.0


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


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.CUDA, BackendKind.METAL])
def test_step_mechanics_uses_selected_backend_solver(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    simulation = Simulation(backend)
    wall = PlaneConstraintInit()
    wall.inward_normal = Vec3(1.0, 0.0, 0.0)
    simulation.add_plane_constraint(wall)
    for y in (0.0, 1.405):
        cell = CellInit()
        cell.position = Vec3(0.205, y, 0.0)
        cell.direction = Vec3(0.0, 1.0, 0.0)
        cell.length = 1.0
        cell.radius = 0.2
        cell.growth_rate = 0.0
        simulation.add_cell(cell)
    contacts = ContactParameters()
    contacts.activation_margin = 0.01
    walls = ConstraintContactParameters()
    walls.activation_margin = 0.01
    parameters = MechanicsParameters()
    parameters.max_iterations = 1
    parameters.residual_rms_tolerance = 0.0
    integration = MechanicsIntegrationParameters()
    integration.require_convergence = False
    expected = simulation.solve_cell_mechanics(parameters, contacts, walls)
    assert expected.report.status == SolverStatus.ITERATION_LIMIT

    reports = simulation.step_with_mechanics(0.0, parameters, contacts, integration, walls, 2)

    assert len(reports) == 2
    actual = reports[0]
    assert actual.report.status == expected.report.status
    assert actual.report.iterations == expected.report.iterations
    assert actual.report.final_residual_rms == pytest.approx(
        expected.report.final_residual_rms, abs=1e-8
    )
    for left, right in zip(actual.corrections, expected.corrections, strict=True):
        assert left.translation.x == pytest.approx(right.translation.x, abs=1e-8)
        assert left.translation.y == pytest.approx(right.translation.y, abs=1e-8)
        assert left.length == pytest.approx(right.length, abs=1e-8)


@pytest.mark.parametrize("count", [5, 16, 32])
def test_cuda_dense_wall_mechanics_matches_cpu(count: int) -> None:
    if not backend_available(BackendKind.CUDA):
        pytest.skip("CUDA unavailable")
    cuda = Simulation(BackendKind.CUDA)
    wall = PlaneConstraintInit()
    wall.inward_normal = Vec3(1.0, 0.0, 0.0)
    cuda.add_plane_constraint(wall)
    for index in range(count):
        cell = CellInit()
        cell.position = Vec3(0.51 + 1.01 * (index % 4), 0.95 * (index // 4), 0.0)
        cell.direction = Vec3(0.0, 1.0, 0.0)
        cell.length = 0.5
        cell.radius = 0.5
        cell.growth_rate = 0.0
        cuda.add_cell(cell)
    cpu = Simulation(BackendKind.CPU)
    cpu.restore(cuda._checkpoint())
    cpu_result = cpu.solve_cell_mechanics()
    cuda_result = cuda.solve_cell_mechanics()
    assert cpu_result.report.status == cuda_result.report.status
    assert len(cpu_result.corrections) == len(cuda_result.corrections)
    for expected, actual in zip(cpu_result.corrections, cuda_result.corrections, strict=True):
        for field in ("translation", "rotation"):
            left, right = getattr(expected, field), getattr(actual, field)
            for axis in ("x", "y", "z"):
                reference = getattr(left, axis)
                assert getattr(right, axis) == pytest.approx(
                    reference, abs=3e-4 + 3e-4 * abs(reference)
                )
        assert actual.length == pytest.approx(
            expected.length, abs=3e-4 + 3e-4 * abs(expected.length)
        )


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
