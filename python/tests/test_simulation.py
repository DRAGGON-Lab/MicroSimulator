from __future__ import annotations

import math

import pytest
from microsimulator import (
    BackendFeature,
    BackendKind,
    BoxConstraintInit,
    CellInit,
    ConstraintContactParameters,
    ConstraintRegion,
    ContactParameters,
    ExternalConstraintKind,
    GridBoundaryKind,
    GridShape,
    MechanicsIntegrationParameters,
    MechanicsParameters,
    PlaneConstraintInit,
    RateInstruction,
    RateOp,
    RodContactLocation,
    RodEndpoint,
    SignalGridSpec,
    SignalGridVelocityField,
    Simulation,
    SolverBreakdown,
    SolverStatus,
    SpeciesRatePlan,
    Vec3,
    backend_available,
    backend_device_count,
)


def rate_instruction(
    operation: RateOp,
    *,
    first: int = 0,
    second: int = 0,
    third: int = 0,
    value: float = 0.0,
) -> RateInstruction:
    instruction = RateInstruction()
    instruction.operation = operation
    instruction.first = first
    instruction.second = second
    instruction.third = third
    instruction.value = value
    return instruction


@pytest.mark.parametrize("backend", list(BackendKind))
def test_growth_and_division_preserve_declared_semantics(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")

    simulation = Simulation(backend)
    initial = CellInit()
    initial.position = Vec3(2.0, 3.0, 0.0)
    initial.direction = Vec3(2.0, 0.0, 0.0)
    initial.length = 4.0
    initial.radius = 0.5
    initial.growth_rate = 0.25
    initial.cell_type = 7

    parent = simulation.add_cell(initial)
    simulation.step(0.5)
    assert math.isclose(simulation.cell(parent).length, 4.5)

    first, second = simulation.divide_equal(parent)
    assert simulation.cell_count == 2
    assert simulation.lineage_parent(first) == parent
    assert simulation.lineage_parent(second) == parent
    assert [cell.slot for cell in simulation.cells()] == [0, 1]
    assert all(math.isclose(cell.direction.x, 1.0) for cell in simulation.cells())
    simulation.validate()


@pytest.mark.parametrize("backend", list(BackendKind))
def test_asymmetric_division_preserves_capsule_extent(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")

    simulation = Simulation(backend)
    initial = CellInit()
    initial.position = Vec3(2.0, 3.0, 0.0)
    initial.direction = Vec3(2.0, 0.0, 0.0)
    initial.length = 6.0
    initial.radius = 0.5
    parent = simulation.add_cell(initial)

    first, second = simulation.divide(parent, 0.25)
    first_cell = simulation.cell(first)
    second_cell = simulation.cell(second)
    assert math.isclose(first_cell.length, 1.25)
    assert math.isclose(second_cell.length, 3.75)
    assert math.isclose(first_cell.position.x, -0.375)
    assert math.isclose(second_cell.position.x, 3.125)
    assert math.isclose(
        (second_cell.position.x - second_cell.length * 0.5)
        - (first_cell.position.x + first_cell.length * 0.5),
        2.0 * initial.radius,
    )
    simulation.validate()


def test_fixed_attribute_is_mutable_and_inherited() -> None:
    simulation = Simulation()
    initial = CellInit()
    initial.length = 4.0
    initial.fixed = True
    parent = simulation.add_cell(initial)
    assert simulation.cell(parent).fixed

    simulation.set_cell_fixed(parent, False)
    assert not simulation.cell(parent).fixed
    simulation.set_cell_fixed(parent, True)
    first, second = simulation.divide_equal(parent)
    assert simulation.cell(first).fixed
    assert simulation.cell(second).fixed


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1, math.nan, math.inf])
def test_asymmetric_division_rejects_invalid_fraction_atomically(fraction: float) -> None:
    simulation = Simulation()
    initial = CellInit()
    initial.length = 6.0
    parent = simulation.add_cell(initial)

    with pytest.raises(ValueError, match="fraction"):
        simulation.divide(parent, fraction)
    assert simulation.cell_count == 1
    assert simulation.cell(parent).length == 6.0


@pytest.mark.parametrize("backend", list(BackendKind))
def test_unavailable_backend_fails_instead_of_falling_back(backend: BackendKind) -> None:
    if backend_available(backend):
        Simulation(backend)
        return
    with pytest.raises(RuntimeError, match=r"not implemented|unavailable"):
        Simulation(backend)


def test_backend_device_selection_is_explicit() -> None:
    assert backend_device_count(BackendKind.CPU) == 1
    assert backend_available(BackendKind.CPU, 0)
    assert not backend_available(BackendKind.CPU, 1)
    assert Simulation(device_index=0).backend_info.device_index == 0
    with pytest.raises(IndexError, match="device index 0"):
        Simulation(device_index=1)

    for backend in BackendKind:
        count = backend_device_count(backend)
        assert backend_available(backend) == (count > 0)
        assert not backend_available(backend, count)
        if count > 0:
            assert Simulation(backend, device_index=0).backend_info.device_index == 0


def test_invalid_time_step_is_rejected() -> None:
    simulation = Simulation()
    with pytest.raises(ValueError, match="time step"):
        simulation.step(-0.1)


@pytest.mark.parametrize("backend", list(BackendKind))
def test_species_step_dilutes_then_evaluates_typed_rates(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    simulation = Simulation(backend, species_count=2)
    if not simulation.supports(BackendFeature.SPECIES):
        pytest.skip("backend does not implement species integration")
    cell = CellInit()
    cell.length = 2.0
    cell.radius = 0.5
    cell.growth_rate = 0.5
    cell.species = [4.0, 2.0]
    cell_id = simulation.add_cell(cell)

    plan = SpeciesRatePlan(
        2,
        [
            rate_instruction(RateOp.CONSTANT, value=2.0),
            rate_instruction(RateOp.SPECIES, first=0),
            rate_instruction(RateOp.CONSTANT, value=-0.5),
            rate_instruction(RateOp.MULTIPLY, first=1, second=2),
        ],
        [0, 3],
    )
    simulation.set_species_rate_plan(plan)
    simulation.step(0.25)

    result = simulation.cell(cell_id)
    dilution = 3.0 / 3.25
    assert simulation.supports(BackendFeature.SPECIES)
    assert simulation.species_count == 2
    assert math.isclose(result.species[0], 4.0 * dilution + 0.5, rel_tol=1.0e-6)
    assert math.isclose(
        result.species[1],
        2.0 * dilution - 0.5 * (4.0 * dilution) * 0.25,
        rel_tol=1.0e-6,
    )

    first, second = simulation.divide_equal(cell_id)
    assert simulation.cell(first).species == result.species
    assert simulation.cell(second).species == result.species
    simulation.set_species(first, [3.0, 4.0])
    assert simulation.cell(first).species == [3.0, 4.0]
    assert simulation.cell(second).species == result.species


def test_species_plan_rejects_forward_references() -> None:
    with pytest.raises(ValueError, match="earlier instruction"):
        SpeciesRatePlan(
            1,
            [rate_instruction(RateOp.NEGATE, first=0)],
            [0],
        )


@pytest.mark.parametrize("backend", list(BackendKind))
def test_contact_graph_is_available_through_the_public_api(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    simulation = Simulation(backend)
    if not simulation.supports(BackendFeature.CELL_CONTACTS):
        pytest.skip("backend does not implement cell contacts")
    first = CellInit()
    first.length = 4.0
    first.radius = 0.5
    second = CellInit()
    second.position = Vec3(0.0, 0.8, 0.0)
    second.length = 4.0
    second.radius = 0.5
    simulation.add_cell(first)
    simulation.add_cell(second)

    parameters = ContactParameters()
    graph = simulation.find_cell_contacts(parameters)
    assert graph.cell_count == 2
    assert not graph.empty
    assert len(graph) == 2
    assert graph.incident_contact_indices(0) == [0, 1]
    assert graph.neighbor_ids(0) == [2]
    assert graph.neighbor_ids(1) == [1]
    assert [contact.ordinal for contact in graph.contacts] == [0, 1]
    assert all(
        math.isclose(contact.signed_separation, -0.2, abs_tol=1.0e-6) for contact in graph.contacts
    )


def test_cpu_mechanics_reports_convergence() -> None:
    simulation = Simulation()
    first = CellInit()
    first.length = 4.0
    second = CellInit()
    second.position = Vec3(0.0, 0.8, 0.0)
    second.length = 4.0
    simulation.add_cell(first)
    simulation.add_cell(second)

    parameters = MechanicsParameters()
    parameters.residual_rms_tolerance = 1.0e-6
    result = simulation.solve_cell_mechanics(parameters)

    assert simulation.supports(BackendFeature.CELL_MECHANICS)
    assert result.report.status == SolverStatus.CONVERGED
    assert result.report.breakdown == SolverBreakdown.NONE
    assert result.report.final_residual_rms <= parameters.residual_rms_tolerance
    assert len(result.corrections) == 2
    assert result.corrections[0].translation.y < 0.0
    assert result.corrections[1].translation.y > 0.0


def test_cell_attributes_can_be_updated_by_stable_id() -> None:
    simulation = Simulation()
    cell_id = simulation.add_cell(CellInit())
    simulation.set_cell_attributes(cell_id, growth_rate=2.5, cell_type=7)

    cell = simulation.cell(cell_id)
    assert cell.id == cell_id
    assert cell.growth_rate == 2.5
    assert cell.cell_type == 7

    with pytest.raises(ValueError, match="finite"):
        simulation.set_cell_attributes(cell_id, growth_rate=math.nan, cell_type=8)
    assert simulation.cell(cell_id).cell_type == 7


@pytest.mark.parametrize("backend", list(BackendKind))
def test_plane_constraint_graph_is_typed_and_incident(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    simulation = Simulation(backend)
    if not simulation.supports(BackendFeature.EXTERNAL_CONSTRAINTS):
        pytest.skip("backend does not implement external constraints")
    cell = CellInit()
    cell.position = Vec3(0.0, 0.25, 0.0)
    cell.length = 2.0
    cell.radius = 0.5
    cell_id = simulation.add_cell(cell)

    plane = PlaneConstraintInit()
    plane.point = Vec3(0.0, 0.0, 0.0)
    plane.inward_normal = Vec3(0.0, 2.0, 0.0)
    plane.coefficient = 3.0
    constraint_id = simulation.add_plane_constraint(plane)

    parameters = ConstraintContactParameters()
    graph = simulation.find_external_contacts(parameters)

    assert simulation.supports(BackendFeature.EXTERNAL_CONSTRAINTS)
    assert graph.cell_count == 1
    assert len(graph) == 2
    assert graph.incident_contact_indices(0) == [0, 1]
    assert all(contact.cell_id == cell_id for contact in graph.contacts)
    assert all(contact.constraint_id == constraint_id for contact in graph.contacts)
    assert all(
        contact.constraint_kind == ExternalConstraintKind.PLANE for contact in graph.contacts
    )
    assert [contact.location for contact in graph.contacts] == [
        RodContactLocation.NEGATIVE,
        RodContactLocation.POSITIVE,
    ]
    assert RodEndpoint is RodContactLocation
    assert [contact.endpoint for contact in graph.contacts] == [
        RodEndpoint.NEGATIVE,
        RodEndpoint.POSITIVE,
    ]
    assert all(math.isclose(contact.signed_separation, -0.25) for contact in graph.contacts)
    assert all(math.isclose(contact.normal.y, -1.0) for contact in graph.contacts)
    assert all(math.isclose(contact.point_on_cell.y, -0.25) for contact in graph.contacts)
    assert all(
        math.isclose(contact.weight, 3.0 / math.sqrt(2.0), rel_tol=1.0e-6)
        for contact in graph.contacts
    )


@pytest.mark.parametrize("backend", list(BackendKind))
def test_constraints_participate_in_mechanical_relaxation(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    simulation = Simulation(backend)
    if not simulation.supports(BackendFeature.EXTERNAL_CONSTRAINTS):
        pytest.skip("backend does not implement external constraints")
    cell = CellInit()
    cell.position = Vec3(0.0, 0.4, 0.0)
    cell.length = 2.0
    cell.radius = 0.5
    cell_id = simulation.add_cell(cell)

    plane = PlaneConstraintInit()
    plane.inward_normal = Vec3(0.0, 1.0, 0.0)
    simulation.add_plane_constraint(plane)

    result = simulation.relax_cell_mechanics()

    assert result.report.status == SolverStatus.CONVERGED
    assert result.corrections[0].translation.y > 0.0
    assert simulation.cell(cell_id).position.y > 0.4


@pytest.mark.parametrize("backend", list(BackendKind))
def test_box_constraints_participate_in_mechanical_relaxation(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    simulation = Simulation(backend)
    if not simulation.supports(BackendFeature.EXTERNAL_CONSTRAINTS):
        pytest.skip("backend does not implement external constraints")
    cell = CellInit()
    cell.position = Vec3(1.4, 0.0, 0.0)
    cell.direction = Vec3(0.0, 1.0, 0.0)
    cell.length = 2.0
    cell.radius = 0.5
    cell_id = simulation.add_cell(cell)

    box = BoxConstraintInit()
    box.half_extents = Vec3(1.0, 1.0, 1.0)
    box.allowed_region = ConstraintRegion.OUTSIDE
    constraint_id = simulation.add_box_constraint(box)

    graph = simulation.find_external_contacts()
    assert len(graph) == 2
    assert all(contact.constraint_id == constraint_id for contact in graph.contacts)
    assert all(contact.constraint_kind == ExternalConstraintKind.BOX for contact in graph.contacts)
    assert all(
        math.isclose(contact.signed_separation, -0.1, abs_tol=1.0e-6)
        for contact in graph.contacts
    )

    result = simulation.relax_cell_mechanics()

    assert result.report.status == SolverStatus.CONVERGED
    assert result.corrections[0].translation.x > 0.0
    assert simulation.cell(cell_id).position.x > 1.4


@pytest.mark.parametrize("backend", list(BackendKind))
def test_finite_wall_detects_midspan_capsule_contact(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    simulation = Simulation(backend)
    if not simulation.supports(BackendFeature.EXTERNAL_CONSTRAINTS):
        pytest.skip("backend does not implement external constraints")
    cell = CellInit()
    cell.position = Vec3(0.0, 0.75, 0.0)
    cell.direction = Vec3(1.0, 0.0, 0.0)
    cell.length = 3.5
    cell.radius = 0.5
    cell_id = simulation.add_cell(cell)

    wall = BoxConstraintInit()
    wall.half_extents = Vec3(1.0, 1.0, 5.0)
    constraint_id = simulation.add_box_constraint(wall)

    graph = simulation.find_external_contacts()
    assert len(graph) == 1
    contact = graph.contacts[0]
    assert contact.constraint_id == constraint_id
    assert contact.constraint_kind == ExternalConstraintKind.BOX
    assert contact.location == RodContactLocation.INTERIOR
    assert math.isclose(contact.point_on_cell.x, 0.0, abs_tol=2.0e-5)
    assert math.isclose(contact.point_on_cell.y, 0.25, abs_tol=2.0e-5)
    assert math.isclose(contact.normal.y, -1.0, abs_tol=2.0e-5)
    assert math.isclose(contact.signed_separation, -0.75, abs_tol=2.0e-5)

    result = simulation.relax_cell_mechanics()
    assert result.report.status == SolverStatus.CONVERGED
    assert result.corrections[0].translation.y > 0.0
    assert simulation.cell(cell_id).position.y > 0.75


def test_cpu_mechanics_relaxation_updates_geometry() -> None:
    simulation = Simulation()
    first = CellInit()
    first.length = 4.0
    second = CellInit()
    second.position = Vec3(0.0, 0.8, 0.0)
    second.length = 4.0
    first_id = simulation.add_cell(first)
    second_id = simulation.add_cell(second)

    integration = MechanicsIntegrationParameters()
    result = simulation.relax_cell_mechanics(integration_parameters=integration)

    assert result.report.status == SolverStatus.CONVERGED
    assert simulation.cell(first_id).position.y < 0.0
    assert simulation.cell(second_id).position.y > 0.8
    assert math.isclose(simulation.cell(first_id).length, 4.0)


@pytest.mark.parametrize("backend", [BackendKind.METAL, BackendKind.CUDA])
def test_native_growth_matches_cpu(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")

    cpu = Simulation(BackendKind.CPU)
    native = Simulation(backend)
    for index in range(33):
        cell = CellInit()
        cell.length = 1.0 + index * 0.1
        cell.growth_rate = (index % 7) * 0.025
        assert cpu.add_cell(cell) == native.add_cell(cell)

    for dt in (0.01, 0.025, 0.1):
        cpu.step(dt)
        native.step(dt)

    for cpu_cell, native_cell in zip(cpu.cells(), native.cells(), strict=True):
        assert math.isclose(cpu_cell.length, native_cell.length, abs_tol=1.0e-6)


def uniform_flow_grid(
    *, origin: float, spacing: float, sites: int, speed: float
) -> SignalGridSpec:
    """A collapsed y/z lattice carrying a uniform x flow between fixed ends."""

    shape = GridShape()
    shape.x, shape.y, shape.z = sites, 1, 1
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.origin = Vec3(origin, 0.0, 0.0)
    grid.spacing = Vec3(spacing, 1.0, 1.0)
    grid.diffusion = [0.0]
    grid.advection = [Vec3()]
    grid.x_lower.kind = GridBoundaryKind.FIXED
    grid.x_lower.values = [0.0]
    grid.x_upper.kind = GridBoundaryKind.FIXED
    grid.x_upper.values = [0.0]
    field = SignalGridVelocityField()
    field.x_faces = [speed] * (sites + 1)
    field.y_faces = [0.0] * (2 * sites)
    field.z_faces = [0.0] * (2 * sites)
    grid.velocity_field = field
    return grid


@pytest.mark.parametrize(
    ("origin", "spacing"), [(0.0, 1.0), (0.1, 0.3), (-97.5, 1.65), (0.7, 5.0)]
)
def test_flow_drift_clamps_endpoints_on_any_lattice(origin: float, spacing: float) -> None:
    sites = 33
    simulation = Simulation()
    simulation.configure_signal_grid(uniform_flow_grid(
        origin=origin, spacing=spacing, sites=sites, speed=2.0
    ))
    cell = CellInit()
    cell.position = Vec3(origin + spacing * (sites - 1), 0.0, 0.0)
    cell.direction = Vec3(1.0, 0.0, 0.0)
    cell.length = 2.0 * spacing
    cell.radius = 0.3
    cell_id = simulation.add_cell(cell)

    simulation.apply_flow_drift(0.1)

    assert math.isclose(
        simulation.cell(cell_id).position.x,
        origin + spacing * (sites - 1) + 0.2,
        rel_tol=1.0e-5,
        abs_tol=1.0e-5,
    )


def test_flow_drift_substeps_instead_of_clipping_the_total_rotation() -> None:
    shape = GridShape()
    shape.x, shape.y, shape.z = 3, 3, 1
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.spacing = Vec3(1.0, 1.0, 1.0)
    grid.diffusion = [0.0]
    grid.advection = [Vec3()]
    grid.x_lower.kind = GridBoundaryKind.FIXED
    grid.x_lower.values = [0.0]
    grid.x_upper.kind = GridBoundaryKind.FIXED
    grid.x_upper.values = [0.0]
    field = SignalGridVelocityField()
    field.x_faces = [float(y) for _ in range(4) for y in range(3)]
    field.y_faces = [0.0] * 12
    field.z_faces = [0.0] * 18
    grid.velocity_field = field

    cell = CellInit()
    cell.position = Vec3(1.0, 1.0, 0.0)
    cell.direction = Vec3(0.0, 1.0, 0.0)
    cell.length = 2.0
    cell.radius = 0.3

    capped = Simulation()
    capped.configure_signal_grid(grid)
    capped_id = capped.add_cell(cell)
    capped.apply_flow_drift(1.0)
    limit = MechanicsIntegrationParameters().max_rotation_radians
    assert capped.cell(capped_id).direction.x > 3 * math.sin(limit)

    frozen = Simulation()
    frozen.configure_signal_grid(grid)
    frozen_id = frozen.add_cell(cell)
    integration = MechanicsIntegrationParameters()
    integration.max_rotation_radians = 0.0
    frozen.apply_flow_drift(1.0, integration)
    assert math.isclose(frozen.cell(frozen_id).direction.x, 0.0, abs_tol=1.0e-6)
    assert math.isclose(frozen.cell(frozen_id).direction.y, 1.0, abs_tol=1.0e-6)


def test_a_cell_inside_a_wall_samples_no_flow_and_does_not_drift() -> None:
    """Mechanics can press a crowded cell into a wall; drift must survive it.

    The velocity field is zero on every face of a solid site, so a stencil with
    no fluid in it samples exactly zero and the cell stays put. Concentration
    there has no such value, so sampling it is still an error.
    """

    sites = 5
    grid = uniform_flow_grid(origin=0.0, spacing=1.0, sites=sites, speed=2.0)
    # A solid block in the middle of the line, with the flow stopping at it.
    grid.obstacles = [0, 0, 1, 0, 0]
    field = SignalGridVelocityField()
    field.x_faces = [2.0, 2.0, 0.0, 0.0, 2.0, 2.0]
    field.y_faces = [0.0] * (2 * sites)
    field.z_faces = [0.0] * (2 * sites)
    grid.velocity_field = field

    simulation = Simulation()
    simulation.configure_signal_grid(grid, [1.0, 1.0, 0.0, 1.0, 1.0])
    buried = CellInit()
    buried.position = Vec3(2.0, 0.0, 0.0)
    buried.direction = Vec3(1.0, 0.0, 0.0)
    buried.length = 0.0
    buried.radius = 0.3
    buried_id = simulation.add_cell(buried)

    simulation.apply_flow_drift(0.5)

    assert simulation.cell(buried_id).position.x == 2.0
    with pytest.raises(ValueError, match="inside a grid obstacle"):
        simulation.sample_signals(Vec3(2.0, 0.0, 0.0))
