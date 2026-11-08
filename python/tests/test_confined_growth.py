"""Coupled growth must respect a closed, nutrient-rich chamber."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from microsimulator import (
    BackendKind,
    CellInit,
    CellUpdate,
    ControllerStep,
    DivisionRequest,
    GridShape,
    MechanicsConfig,
    ModelContext,
    NativeController,
    PlaneConstraintInit,
    SignalGridSpec,
    Simulation,
    StepPlan,
    Vec3,
    backend_available,
)
from microsimulator.microfluidics import TrapChannelDevice
from microsimulator.runner import build_model


def chamber(backend: BackendKind) -> NativeController:
    simulation = Simulation(backend)
    shape = GridShape()
    shape.x, shape.y, shape.z = 9, 9, 3
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.origin = Vec3(-4.0, -4.0, -1.0)
    grid.spacing = Vec3(1.0, 1.0, 1.0)
    grid.diffusion = [0.0]
    grid.advection = [Vec3()]
    simulation.configure_signal_grid(grid, [10.0] * (9 * 9 * 3))
    for point, normal in (
        (Vec3(-2.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)),
        (Vec3(2.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0)),
        (Vec3(0.0, -2.0, 0.0), Vec3(0.0, 1.0, 0.0)),
        (Vec3(0.0, 2.0, 0.0), Vec3(0.0, -1.0, 0.0)),
        (Vec3(0.0, 0.0, -0.6), Vec3(0.0, 0.0, 1.0)),
        (Vec3(0.0, 0.0, 0.6), Vec3(0.0, 0.0, -1.0)),
    ):
        wall = PlaneConstraintInit()
        wall.point = point
        wall.inward_normal = normal
        simulation.add_plane_constraint(wall)
    founder = CellInit()
    founder.length = 1.0
    founder.radius = 0.4
    founder.growth_rate = 1.0
    simulation.add_cell(founder)

    def regulate(step: ControllerStep) -> StepPlan:
        return StepPlan(
            divisions=tuple(DivisionRequest(cell.id) for cell in step.cells if cell.length > 2.0)
        )

    return NativeController(
        simulation,
        model_id="confined-growth",
        model_version=1,
        rng=random.Random(42),
        regulate=regulate,
        mechanics=MechanicsConfig(),
    )


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.CUDA])
def test_nutrient_rich_chamber_contains_cells_and_stalls_growth(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend} unavailable")
    model = chamber(backend)
    totals = []
    counts = []
    for step in range(120):
        previous = model.simulation.cells()
        model.step(0.05)
        cells = model.simulation.cells()
        assert min(model.simulation.signal_levels) == pytest.approx(10.0)
        for cell in cells:
            assert model.simulation.sample_signals(cell.position)[0] == pytest.approx(10.0)
            for coordinate, axis in (
                (cell.position.x, cell.direction.x),
                (cell.position.y, cell.direction.y),
            ):
                assert abs(coordinate) + abs(axis) * cell.length / 2 + cell.radius <= 2.0001
            assert (
                abs(cell.position.z) + abs(cell.direction.z) * cell.length / 2 + cell.radius
                <= 0.6001
            )
        assert all(
            contact.signed_separation >= -1e-4
            for contact in model.simulation.find_cell_contacts().contacts
        )
        if step == 0:
            assert cells[0].length - previous[0].length == pytest.approx(0.05, rel=1e-4)
        totals.append(sum(cell.length for cell in cells))
        counts.append(len(cells))
    assert max(counts) > 1
    assert max(totals[-20:]) - min(totals[-20:]) < 0.1
    assert len(set(counts[-20:])) == 1


def test_fixed_cell_outside_wall_rolls_back_controller_step() -> None:
    model = chamber(BackendKind.CPU)
    cell = model.simulation.cells()[0]
    model.simulation.set_cell_geometry(cell.id, Vec3(1.8, 0.0, 0.0), cell.direction, cell.length)
    model.simulation.set_cell_fixed(cell.id, True)
    before = model.simulation._checkpoint()
    before_nutrient = model.simulation.signal_levels
    with pytest.raises(RuntimeError, match="infeasible wall geometry"):
        model.step(0.05)
    assert model.completed_steps == 0
    assert model.simulation.time == before.time
    assert model.simulation.cell(cell.id).position.x == pytest.approx(1.8)
    assert model.simulation.signal_levels == before_nutrient


def test_cpu_cuda_contacts_corrections_and_growth_agree() -> None:
    if not backend_available(BackendKind.CUDA):
        pytest.skip("CUDA unavailable")
    cpu, cuda = chamber(BackendKind.CPU), chamber(BackendKind.CUDA)
    for model in (cpu, cuda):
        cell = model.simulation.cells()[0]
        model.simulation.set_cell_geometry(
            cell.id, Vec3(1.1, 0.0, 0.0), cell.direction, cell.length
        )
    cpu_contacts = cpu.simulation.find_external_contacts().contacts
    cuda_contacts = cuda.simulation.find_external_contacts().contacts
    assert [(contact.constraint_id, contact.location) for contact in cpu_contacts] == [
        (contact.constraint_id, contact.location) for contact in cuda_contacts
    ]
    assert [contact.signed_separation for contact in cpu_contacts] == pytest.approx(
        [contact.signed_separation for contact in cuda_contacts], abs=1e-5
    )
    cpu_result = cpu.simulation.solve_cell_mechanics()
    cuda_result = cuda.simulation.solve_cell_mechanics()
    assert cpu_result.corrections[0].length == pytest.approx(
        cuda_result.corrections[0].length, abs=1e-4
    )
    for _ in range(20):
        cpu.step(0.05)
        cuda.step(0.05)
    cpu_cells = cpu.simulation.cells()
    cuda_cells = cuda.simulation.cells()
    assert [cell.id for cell in cpu_cells] == [cell.id for cell in cuda_cells]
    for expected, actual in zip(cpu_cells, cuda_cells, strict=True):
        assert actual.length == pytest.approx(expected.length, abs=1e-3)
        assert actual.position.x == pytest.approx(expected.position.x, abs=1e-3)
        assert actual.position.y == pytest.approx(expected.position.y, abs=1e-3)


def test_cpu_cuda_crowded_chamber_growth_and_contacts_agree() -> None:
    if not backend_available(BackendKind.CUDA):
        pytest.skip("CUDA unavailable")
    cpu, cuda = chamber(BackendKind.CPU), chamber(BackendKind.CUDA)
    for _ in range(120):
        cpu.step(0.05)
        cuda.step(0.05)
    expected = cpu.simulation.cells()
    actual = cuda.simulation.cells()
    assert [cell.id for cell in actual] == [cell.id for cell in expected]
    assert sum(cell.length for cell in actual) == pytest.approx(
        sum(cell.length for cell in expected), rel=0.02
    )
    cpu_contacts = cpu.simulation.find_cell_contacts().contacts
    cuda_contacts = cuda.simulation.find_cell_contacts().contacts
    assert [(edge.first_id, edge.second_id) for edge in cpu_contacts] == [
        (edge.first_id, edge.second_id) for edge in cuda_contacts
    ]
    assert all(edge.signed_separation >= -1e-4 for edge in cuda_contacts)
    assert all(
        edge.signed_separation >= -1e-4
        for edge in cuda.simulation.find_external_contacts().contacts
    )


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.CUDA])
def test_trap_back_wall_contains_full_capsule(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend} unavailable")
    simulation = Simulation(backend)
    device = TrapChannelDevice()
    device.add_constraints(simulation)
    cell = CellInit()
    cell.position = Vec3(device.trap_back_x - 1.0, 0.0, 0.0)
    cell.direction = Vec3(1.0, 0.0, 0.0)
    cell.length = 1.0
    cell.radius = 0.5
    cell.growth_rate = 1.0
    simulation.add_cell(cell)
    model = NativeController(
        simulation,
        model_id="back-wall",
        model_version=1,
        rng=random.Random(1),
        mechanics=MechanicsConfig(),
    )
    for _ in range(50):
        model.step(0.05)
        current = simulation.cells()[0]
        assert (
            current.position.x + current.direction.x * current.length / 2 + current.radius
        ) <= device.trap_back_x + 1e-4
        assert all(
            contact.signed_separation >= -1e-4
            for contact in simulation.find_external_contacts().contacts
        )


def test_narrower_than_diameter_is_explicitly_infeasible() -> None:
    simulation = Simulation(BackendKind.CPU)
    for x, normal in ((-0.3, 1.0), (0.3, -1.0)):
        wall = PlaneConstraintInit()
        wall.point = Vec3(x, 0.0, 0.0)
        wall.inward_normal = Vec3(normal, 0.0, 0.0)
        simulation.add_plane_constraint(wall)
    cell = CellInit()
    cell.length = 0.1
    cell.radius = 0.4
    simulation.add_cell(cell)

    def mutate(step: ControllerStep) -> StepPlan:
        step.state["attempt"] = step.rng.random()
        return StepPlan(updates=(CellUpdate(step.cells[0].id, growth_rate=2.0),))

    model = NativeController(
        simulation,
        model_id="infeasible-width",
        model_version=1,
        rng=random.Random(1),
        regulate=mutate,
        mechanics=MechanicsConfig(),
    )
    before = model.controller_state()
    with pytest.raises(RuntimeError, match="infeasible wall geometry"):
        model.step(0.01)
    assert model.completed_steps == 0
    assert simulation.cells()[0].length == pytest.approx(0.1)
    assert simulation.cells()[0].growth_rate == pytest.approx(cell.growth_rate)
    assert model.controller_state() == before


def test_square_trap_daughters_slide_until_unsuppressible_jam() -> None:
    model, _ = build_model(
        Path("examples/square_trap.py"), ModelContext(BackendKind.CPU, 0, seed=42)
    )
    maximum_count = 0
    maximum_lateral_offset = 0.0
    totals = []
    jammed = False
    for _ in range(150):
        before = model.controller_state()
        before_time = model.simulation.time
        before_cells = [
            (
                cell.id,
                cell.length,
                cell.position.x,
                cell.position.y,
                cell.direction.x,
                cell.direction.y,
                cell.growth_rate,
            )
            for cell in model.simulation.cells()
        ]
        try:
            model.step(0.05)
        except RuntimeError as error:
            # Once the packed trap has no newly suppressible growth, the step rolls back.
            if "mechanics projection is infeasible without suppressible growth" not in str(error):
                raise
            assert model.controller_state() == before
            assert model.simulation.time == before_time
            assert [
                (
                    cell.id,
                    cell.length,
                    cell.position.x,
                    cell.position.y,
                    cell.direction.x,
                    cell.direction.y,
                    cell.growth_rate,
                )
                for cell in model.simulation.cells()
            ] == before_cells
            jammed = True
            break
        cells = model.simulation.cells()
        maximum_count = max(maximum_count, len(cells))
        maximum_lateral_offset = max(
            maximum_lateral_offset, *(abs(cell.position.y) for cell in cells)
        )
        totals.append(sum(cell.length for cell in cells))
        assert all(
            contact.signed_separation >= -1e-4
            for contact in model.simulation.find_cell_contacts().contacts
        )
        assert all(
            contact.signed_separation >= -1e-4
            for contact in model.simulation.find_external_contacts().contacts
        )
    assert maximum_count >= 8
    assert maximum_lateral_offset > 0.2
    assert totals[60] > totals[20] + 2.0
    assert jammed


@pytest.mark.parametrize("backend", [BackendKind.CPU, BackendKind.CUDA])
def test_axially_trapped_cells_slide_sideways(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend} unavailable")
    simulation = Simulation(backend)
    for x, normal in ((-1.1, 1.0), (1.1, -1.0)):
        wall = PlaneConstraintInit()
        wall.point = Vec3(x, 0.0, 0.0)
        wall.inward_normal = Vec3(normal, 0.0, 0.0)
        simulation.add_plane_constraint(wall)
    for x in (-0.45, 0.45):
        cell = CellInit()
        cell.position = Vec3(x, 0.0, 0.0)
        cell.direction = Vec3(1.0, 0.0, 0.0)
        cell.length = 0.5
        cell.radius = 0.3
        cell.growth_rate = 1.0
        simulation.add_cell(cell)
    model = NativeController(
        simulation,
        model_id="axial-slide",
        model_version=1,
        rng=random.Random(42),
        mechanics=MechanicsConfig(),
    )
    for _ in range(30):
        model.step(0.05)
        assert all(
            edge.signed_separation >= -1e-4 for edge in simulation.find_cell_contacts().contacts
        )
        assert all(
            edge.signed_separation >= -1e-4 for edge in simulation.find_external_contacts().contacts
        )
    cells = simulation.cells()
    assert abs(cells[0].position.y - cells[1].position.y) > 0.1
    assert sum(cell.length for cell in cells) > 1.0


def test_fixed_overlapping_cells_rollback() -> None:
    simulation = Simulation(BackendKind.CPU)
    for x in (-0.1, 0.1):
        cell = CellInit()
        cell.position = Vec3(x, 0.0, 0.0)
        cell.length = 0.5
        cell.radius = 0.3
        cell.fixed = True
        simulation.add_cell(cell)
    model = NativeController(
        simulation,
        model_id="fixed-overlap",
        model_version=1,
        rng=random.Random(1),
        mechanics=MechanicsConfig(),
    )
    with pytest.raises(RuntimeError, match=r"infeasible.*cell packing"):
        model.step(0.05)
    assert model.completed_steps == 0
    assert simulation.time == 0.0


@pytest.mark.parametrize(
    ("offset", "second_axis"),
    [
        (Vec3(0.6, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)),
        (Vec3(0.0, 0.5, 0.0), Vec3(1.0, 0.0, 0.0)),
        (Vec3(0.3, 0.3, 0.0), Vec3(0.0, 1.0, 0.0)),
    ],
)
@pytest.mark.parametrize("first_fixed", [False, True])
def test_projection_resolves_cell_contact_geometries(
    offset: Vec3, second_axis: Vec3, first_fixed: bool
) -> None:
    simulation = Simulation(BackendKind.CPU)
    first = CellInit()
    first.length = 0.5
    first.radius = 0.4
    first.fixed = first_fixed
    first_id = simulation.add_cell(first)
    second = CellInit()
    second.position = offset
    second.direction = second_axis
    second.length = 0.5
    second.radius = 0.4
    simulation.add_cell(second)
    model = NativeController(
        simulation,
        model_id="contact-projection",
        model_version=1,
        rng=random.Random(1),
        mechanics=MechanicsConfig(),
    )
    model.step(0.0)
    assert all(edge.signed_separation >= -1e-4 for edge in simulation.find_cell_contacts().contacts)
    if first_fixed:
        assert simulation.cell(first_id).position.x == pytest.approx(0.0)
        assert simulation.cell(first_id).position.y == pytest.approx(0.0)
