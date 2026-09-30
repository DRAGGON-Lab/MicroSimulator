# pyright: reportPrivateUsage=false

import math
from pathlib import Path

import pytest
from microsimulator import (
    BackendKind,
    ModelContext,
    NativeController,
    Vec3,
    build_model,
    capture_scene,
    load_checkpoint_bundle,
    run_simulation,
)
from microsimulator.checkpoint import JSONValue
from microsimulator.microfluidics import MotherMachineDevice
from microsimulator.runner import BatchError

MODEL = Path(__file__).resolve().parents[2] / "examples/mother_machine.py"
DEVICE = MotherMachineDevice()


def _assert_confined(model: NativeController) -> None:
    state = model.controller_state()["state"]
    assert isinstance(state, dict)
    mothers = state["mother_ids"]
    assert isinstance(mothers, list)
    assert len(mothers) == DEVICE.channel_count

    for mother_id, center in zip(mothers, DEVICE.growth_centers, strict=True):
        assert isinstance(mother_id, int)
        mother = model.simulation.cell(mother_id)
        assert mother.position.x > 21
        assert abs(mother.position.y - center) < 0.26
        assert mother.direction.x < 0
        assert mother.cell_type == 0

    for cell in model.simulation.cells():
        assert not cell.fixed
        half_x = abs(cell.direction.x) * cell.length / 2 + cell.radius
        assert cell.position.x + half_x <= DEVICE.growth_length + 0.03

        if cell.position.x - half_x > 0:
            center = min(DEVICE.growth_centers, key=lambda y: abs(y - cell.position.y))
            half_y = abs(cell.direction.y) * cell.length / 2 + cell.radius
            half_z = abs(cell.direction.z) * cell.length / 2 + cell.radius
            assert abs(cell.position.y - center) + half_y <= DEVICE.growth_width / 2 + 0.03
            assert cell.position.z - half_z >= -0.03
            assert cell.position.z + half_z <= DEVICE.growth_height + 0.03


@pytest.mark.parametrize("dt", [0.025, 0.0125])
def test_growth_retains_old_pole_lineages_and_expels_descendants(dt: float) -> None:
    model, _ = build_model(
        MODEL, ModelContext(BackendKind.CPU, 0, seed=42, parameters={"growth_rate": 0.3})
    )
    assert isinstance(model, NativeController)
    assert model.simulation.cell_count == 42
    initial_mothers = {cell.id for cell in model.simulation.cells() if cell.cell_type == 0}
    passed_into_flow = False

    for index in range(round(10 / dt)):
        model.step(dt)
        passed_into_flow |= any(cell.position.x < 0 for cell in model.simulation.cells())

        if index % 20 == 0:
            _assert_confined(model)

    _assert_confined(model)
    state = model.controller_state()["state"]
    assert isinstance(state, dict)
    generations = state["mother_generations"]
    assert isinstance(generations, list)
    assert all(isinstance(value, int) and value >= 2 for value in generations)
    assert passed_into_flow
    washed, divisions = state["washed_out"], state["division_count"]
    assert isinstance(washed, int) and isinstance(divisions, int)
    assert washed > 20
    assert model.simulation.cell_count == 42 + divisions - washed
    assert not initial_mothers.intersection(cell.id for cell in model.simulation.cells())


def test_nutrient_controls_growth_and_is_consumed() -> None:
    context = ModelContext(BackendKind.CPU, 0, seed=42, parameters={"growth_rate": 0.3})
    model, _ = build_model(MODEL, context)
    control, _ = build_model(
        MODEL, ModelContext(BackendKind.CPU, 0, seed=42, parameters={"cells": False})
    )
    assert isinstance(model, NativeController) and isinstance(control, NativeController)
    before = {cell.id: cell.length for cell in model.simulation.cells()}
    model.step(0.025)
    control.step(0.025)
    sites = len(model.simulation.signal_levels) // 2
    assert all(cell.length > before[cell.id] for cell in model.simulation.cells())
    assert sum(model.simulation.signal_levels[sites:]) < sum(
        control.simulation.signal_levels[sites:]
    )

    levels = model.simulation.signal_levels
    levels[sites:] = [0.0] * sites
    model.simulation.set_signal_levels(levels)
    before = {cell.id: cell.length for cell in model.simulation.cells()}
    model.step(0.025)
    assert all(cell.growth_rate == 0 for cell in model.simulation.cells())
    assert {cell.id: cell.length for cell in model.simulation.cells()} == before


def test_transport_supplies_closed_channels_through_their_mouths() -> None:
    model, _ = build_model(
        MODEL, ModelContext(BackendKind.CPU, 0, seed=42, parameters={"cells": False})
    )
    assert isinstance(model, NativeController)
    grid = model.simulation._checkpoint().signal_grid
    assert grid is not None
    field = grid.spec.velocity_field
    assert field is not None
    spec = grid.spec
    nz, ny = spec.shape.z, spec.shape.y
    main_flow = field.y_faces[6 * (ny + 1) * nz + 60 * nz + 1]
    deep_flow = field.y_faces[32 * (ny + 1) * nz + 50 * nz + 1]
    assert main_flow > 20
    assert abs(deep_flow) < 0.01

    for _ in range(40):
        model.step(0.025)

    mouth = model.simulation.sample_signals(Vec3(1, -11.25, 0.7))[0]
    end = model.simulation.sample_signals(Vec3(24, -11.25, 0.7))[0]
    assert mouth > 0.01
    assert 0 <= end < mouth * 0.01
    assert all(
        value == 0
        for value, solid in zip(
            model.simulation.signal_levels[: spec.site_count], spec.obstacles, strict=True
        )
        if solid
    )
    scene = capture_scene(model.simulation)
    assert len(scene.constraints.boxes) == 10
    assert not scene.constraints.cylinders


def test_checkpoint_restores_mothers_division_and_washout(tmp_path: Path) -> None:
    context = ModelContext(BackendKind.CPU, 0, seed=42, parameters={"growth_rate": 0.3})
    model, provenance = build_model(MODEL, context)
    assert isinstance(model, NativeController)
    checkpoint = tmp_path / "mother-machine.json"
    run_simulation(model, steps=100, dt=0.025, output=checkpoint, provenance=provenance)
    resumed, _ = build_model(MODEL, context, checkpoint=load_checkpoint_bundle(checkpoint))
    assert isinstance(resumed, NativeController)

    for _ in range(40):
        model.step(0.025)
        resumed.step(0.025)

    assert capture_scene(model.simulation) == capture_scene(resumed.simulation)
    assert model.controller_state() == resumed.controller_state()


def test_default_growth_is_unaccelerated_and_cells_remain_mobile() -> None:
    model, _ = build_model(MODEL, ModelContext(BackendKind.CPU, 0, seed=42))
    assert isinstance(model, NativeController)
    initial = {cell.id: cell.position.x for cell in model.simulation.cells()}

    for _ in range(10):
        model.step(0.025)

    assert all(0 < cell.growth_rate <= math.log(2) / 1800 for cell in model.simulation.cells())
    assert all(not cell.fixed for cell in model.simulation.cells())
    assert model.simulation.cell_count == len(initial)
    state = model.controller_state()["state"]
    assert isinstance(state, dict)
    assert state["washed_out"] == 0


@pytest.mark.parametrize(
    "parameters", [{"growth_rate": -1}, {"growth_rate": True}, {"cells": "yes"}]
)
def test_invalid_model_parameters_are_rejected(parameters: dict[str, JSONValue]) -> None:
    with pytest.raises(BatchError):
        build_model(MODEL, ModelContext(BackendKind.CPU, 0, seed=42, parameters=parameters))


def test_device_rejects_overlapping_or_out_of_bounds_growth_channels() -> None:
    with pytest.raises(ValueError):
        MotherMachineDevice(channel_pitch=1)

    with pytest.raises(ValueError):
        MotherMachineDevice(channel_count=20)

    with pytest.raises(ValueError):
        MotherMachineDevice(growth_height=8)
