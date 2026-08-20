# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from pathlib import Path

import pytest
from microsimulator import (
    BackendKind,
    GridShape,
    ModelContext,
    SignalGridSpec,
    SimulationController,
    Vec3,
)
from microsimulator.microfluidics import BiopixelTrapDevice, TrapChannelDevice
from microsimulator.runner import build_model

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _grid() -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 64, 72, 4
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.origin = Vec3(-140.0, -144.0, -8.0)
    grid.spacing = Vec3(4.0, 4.0, 4.0)
    grid.diffusion = [40.0]
    grid.advection = [Vec3()]
    return grid


def test_device_grid_projection_is_engine_valid() -> None:
    device = TrapChannelDevice(mean_flow_speed=20.0)
    grid = _grid()
    device.apply_to_grid(grid, inlet_values=[10.0], outlet_values=[0.0])
    grid.validate()

    assert grid.velocity_field is not None
    assert any(value != 0.0 for value in grid.velocity_field.y_faces)
    assert grid.y_lower.values == [10.0]

    solid = sum(grid.obstacles)
    assert 0 < solid < len(grid.obstacles)


def test_device_flow_runs_through_the_channel_and_rests_in_the_trap() -> None:
    device = TrapChannelDevice(mean_flow_speed=20.0)
    grid = _grid()
    device.apply_to_grid(grid, inlet_values=[10.0], outlet_values=[0.0])
    assert grid.velocity_field is not None

    def y_face(x: int, fy: int, z: int) -> float:
        assert grid.velocity_field is not None
        return grid.velocity_field.y_faces[
            x * (grid.shape.y + 1) * grid.shape.z + fy * grid.shape.z + z
        ]

    center_x = (device.channel_far_x + device.trap_open_x) * 0.5
    channel_column = int((center_x - grid.origin.x) / grid.spacing.x + 0.5)
    trap_column = int((0.0 - grid.origin.x) / grid.spacing.x + 0.5)
    mid_face = grid.shape.y // 2
    fluid_z = int((0.0 - grid.origin.z) / grid.spacing.z + 0.5)
    channel_speed = y_face(channel_column, mid_face, fluid_z)
    trap_speed = abs(y_face(trap_column, mid_face, fluid_z))
    assert channel_speed > 15.0
    assert trap_speed < channel_speed * 0.05

    half = (grid.spacing.x * 0.5, grid.spacing.y * 0.5, grid.spacing.z * 0.5)
    assert device._solid(device.trap_back_x + grid.spacing.x, 0.0, 0.0, half)
    assert not device._solid(device.trap_back_x, 0.0, 0.0, half)
    assert not device._solid(center_x, 0.0, 0.0, half)


def test_biopixel_defaults_separate_reported_trap_from_model_channel() -> None:
    device = BiopixelTrapDevice()

    assert device.trap_width == 100.0
    assert device.trap_depth == 85.0
    assert device.trap_height == 1.65
    assert device.channel_width == 100.0
    assert device.channel_height == 10.0
    assert device.wall_thickness == 10.0


def test_wall_surfaces_stay_inside_the_fluid_mask() -> None:
    cases = (
        (
            TrapChannelDevice(),
            (4.0, 4.0, 4.0),
            (
                (60.0, 0.0, 0.0),
                (-60.0, 15.0, 0.0),
                (0.0, -15.0, 0.0),
                (-100.0, 0.0, 0.0),
                (0.0, 0.0, 3.0),
                (0.0, 0.0, -3.0),
            ),
        ),
        (
            BiopixelTrapDevice(),
            (5.0, 5.0, 1.65),
            (
                (85.0, 0.0, 0.8),
                (0.0, 50.0, 0.8),
                (0.0, -50.0, 0.8),
                (-100.0, 0.0, 5.0),
                (50.0, 0.0, 1.65),
                (-50.0, 0.0, 10.0),
            ),
        ),
    )

    for device, spacing, surfaces in cases:
        half = (spacing[0] * 0.5, spacing[1] * 0.5, spacing[2] * 0.5)
        for surface in surfaces:
            assert not device._solid(surface[0], surface[1], surface[2], half)


def test_biopixel_cavity_is_shallow_beside_the_model_channel() -> None:
    device = BiopixelTrapDevice()
    half = (2.5, 2.5, 0.825)

    assert not device._solid(42.5, 0.0, 0.825, half)
    assert device._solid(42.5, 0.0, 2.475, half)
    assert not device._solid(-50.0, 0.0, 9.075, half)
    assert math.isclose(device.trap_height / device.channel_height, 0.165)


def test_planar_mask_wall_error_is_bounded_by_half_spacing() -> None:
    device = TrapChannelDevice()
    for h in (1.0, 2.0, 4.0):
        for shift in (0.1, 0.4, 0.9):
            spec = SignalGridSpec()
            spec.signal_count = 1
            shape = GridShape()
            shape.x, shape.y, shape.z = int(180 / h), 2, 1
            spec.shape, spec.spacing = shape, Vec3(h, h, h)
            spec.origin, spec.diffusion = Vec3(-110 + shift * h, 0, 0), [0]
            device.apply_to_grid(spec, inlet_values=[0], outlet_values=[0])
            columns = [x for x in range(shape.x) if not spec.obstacles[x * 2]]
            lower = spec.origin.x + (columns[0] - 0.5) * h
            upper = spec.origin.x + (columns[-1] + 0.5) * h
            assert abs(lower - device.channel_far_x) <= h / 2 + 1e-5
            assert abs(upper - device.trap_back_x) <= h / 2 + 1e-5


def test_trap_example_builds_steps_and_transports_nutrient() -> None:
    model, _ = build_model(
        _EXAMPLES / "microfluidic_trap.py",
        ModelContext(BackendKind.CPU, 0, seed=11),
    )
    assert isinstance(model, SimulationController)
    for _ in range(20):
        model.step(0.02)

    simulation = model.simulation
    device = TrapChannelDevice()
    channel_x = (device.channel_far_x + device.trap_open_x) * 0.5
    upstream = simulation.sample_signals(Vec3(channel_x, -100.0, 0.0))[0]
    trap_interior = simulation.sample_signals(Vec3(0.0, 0.0, 0.0))[0]
    assert upstream > 5.0
    assert trap_interior > 5.0
    assert upstream >= trap_interior - 1.0e-3
    with pytest.raises(ValueError, match="inside a grid obstacle"):
        simulation.sample_signals(Vec3(0.0, 100.0, 0.0))
    assert len(simulation.cells()) >= 1


def test_biopixel_model_uses_reported_cavity_dimensions() -> None:
    device = BiopixelTrapDevice(mean_flow_speed=20.0)

    assert device.trap_width == 100.0
    assert device.trap_depth == 85.0
    assert device.trap_height == 1.65
    assert device.channel_height == 10.0

    # The CAD layout is tested independently in test_masks.py. These checks
    # cover the published cavity size and the separately chosen model channel.
    half = (2.5, 2.5, 0.825)
    assert not device._solid(42.5, 0.0, 0.825, half)
    assert device._solid(42.5, 0.0, 2.475, half)
    assert not device._solid(-50.0, 0.0, 9.075, half)


def test_biopixel_example_confines_a_monolayer_under_flow() -> None:
    model, _ = build_model(
        _EXAMPLES / "tutorials" / "biopixel_trap.py",
        ModelContext(BackendKind.CPU, 0, seed=5),
    )
    assert isinstance(model, SimulationController)
    # 110 steps crosses the model's Brinkman re-solve cadence at step 100, so
    # the run exercises the colony-drag solve and the runtime field swap.
    for _ in range(110):
        model.step(0.02)

    cells = model.simulation.cells()
    assert len(cells) >= 2
    for cell in cells:
        assert 0.0 < cell.position.z < 1.65
        assert -50.0 < cell.position.y < 50.0
        assert cell.position.x < 95.0
    checkpoint = model.simulation._checkpoint()
    assert checkpoint.signal_grid is not None
    assert checkpoint.signal_grid.spec.velocity_field is not None
