# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
import runpy
from pathlib import Path
from typing import Any

import pytest
from microsimulator import (
    BackendKind,
    CellInit,
    GridFace,
    GridShape,
    ModelContext,
    NativeController,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    SimulationController,
    Vec3,
    backend_available,
)
from microsimulator.design import load_design_scene
from microsimulator.microfluidics import (
    BiopixelTrapDevice,
    InletSchedule,
    LayoutDevice,
    TrapChannelDevice,
)
from microsimulator.runner import build_model

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_TRAP_CHAMBER = Path(__file__).resolve().parent / "fixtures" / "design" / "trap_chamber.scene.json"


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


def _trap_example_device() -> TrapChannelDevice:
    device = runpy.run_path(str(_EXAMPLES / "microfluidic_trap.py"))["DEVICE"]
    assert isinstance(device, TrapChannelDevice)
    return device


def test_trap_example_builds_steps_and_transports_nutrient() -> None:
    model, _ = build_model(
        _EXAMPLES / "microfluidic_trap.py",
        ModelContext(BackendKind.CPU, 0, seed=11),
    )
    assert isinstance(model, SimulationController)
    for _ in range(20):
        model.step(0.02)

    simulation = model.simulation
    device = _trap_example_device()
    channel_x = (device.channel_far_x + device.trap_open_x) * 0.5
    upstream_y = -device.channel_half_length + 10.0
    upstream = simulation.sample_signals(Vec3(channel_x, upstream_y, 0.0))[0]
    trap_interior = simulation.sample_signals(Vec3(0.0, 0.0, 0.0))[0]
    assert upstream > 5.0
    assert trap_interior > 5.0
    assert upstream >= trap_interior - 1.0e-3
    with pytest.raises(ValueError, match="inside a grid obstacle"):
        simulation.sample_signals(Vec3(0.0, device.trap_half_y + 20.0, 0.0))
    assert len(simulation.cells()) >= 1


def test_trap_example_fills_and_settles_at_a_steady_turnover() -> None:
    model, _ = build_model(
        _EXAMPLES / "microfluidic_trap.py",
        ModelContext(BackendKind.CPU, 0, seed=42),
    )
    assert isinstance(model, NativeController)
    simulation = model.simulation
    device = _trap_example_device()
    populations: list[int] = []
    # Axial loads of the settled colony, by depth into the trap from its mouth.
    depth_bins = 8
    bin_width = (device.trap_back_x - device.trap_open_x) / depth_bins
    loads_by_depth: list[list[float]] = [[] for _ in range(depth_bins)]
    # Long enough for the colony to fill the trap, press on all three walls,
    # and shed cells into the channel for several doubling times.
    for step in range(1, 1501):
        model.step(0.02)
        if step % 100 == 0:
            populations.append(len(simulation.cells()))
        if step > 1100 and step % 20 == 0:
            result = model.last_contact_dynamics
            assert result is not None
            for cell in simulation.cells():
                depth = cell.position.x - device.trap_open_x
                if depth > 0.0:
                    index = min(depth_bins - 1, int(depth / bin_width))
                    loads_by_depth[index].append(result.axial_loads[cell.slot])

    cells = simulation.cells()
    in_trap = [cell for cell in cells if cell.position.x > device.trap_open_x]
    in_channel = [cell for cell in cells if cell.position.x <= device.trap_open_x]
    # A monolayer of this footprint holds a few hundred cells, and the trap
    # stays full while cells keep leaving through its mouth.
    assert 150 <= len(in_trap) <= 450
    assert in_channel
    assert max(cell.id for cell in cells) > 3 * len(cells)
    settled = populations[-5:]
    assert max(settled) - min(settled) <= 0.15 * max(settled)

    # Cells deeper in the trap carry the drag of everything between them and
    # the mouth. The load climbs steeply from the mouth, levels off where it
    # slows growth, and peaks against the back wall, where cells lie along the
    # wall with no open boundary ahead of them.
    profile = [sum(loads) / len(loads) for loads in loads_by_depth]
    assert profile[:-1] == sorted(profile[:-1])
    assert profile[3] - profile[0] > 1.8 * (profile[6] - profile[3])
    assert profile[-1] > 1.3 * profile[-2]
    assert profile[0] < 0.5 * profile[-2]

    walls = simulation.find_external_contacts().contacts
    assert walls
    # What the contact forces press into the walls, which are stiff.
    assert min(contact.signed_separation for contact in walls) >= -2.0e-3
    slack = 2.0e-3
    for cell in cells:
        reach = abs(cell.direction.x) * cell.length * 0.5 + cell.radius
        assert cell.position.x + reach <= device.trap_back_x + slack
        assert abs(cell.position.z) + cell.radius <= device.trap_half_z + slack
        # Every cell still has fluid to exchange nutrient with.
        assert simulation.sample_signals(cell.position)[0] >= 0.0


@pytest.mark.parametrize("backend", [BackendKind.METAL, BackendKind.CUDA])
def test_trap_example_builds_its_initial_flow_on_the_selected_backend(
    backend: BackendKind,
) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend.name} backend is unavailable")
    model, _ = build_model(
        _EXAMPLES / "microfluidic_trap.py",
        ModelContext(backend, 0, seed=13),
    )
    assert isinstance(model, SimulationController)
    assert model.simulation.backend_info.kind == backend
    checkpoint = model.simulation._checkpoint()
    assert checkpoint.signal_grid is not None
    assert checkpoint.signal_grid.spec.velocity_field is not None


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


def test_inlet_schedule_switches_ramps_and_pulses() -> None:
    switching = InletSchedule((10.0, 0.0), (5.0, 8.0), ((10.0, 2.0), (4.0, 0.0)))
    assert switching.values_at(0.0) == [10.0, 0.0]
    assert switching.values_at(4.999) == [10.0, 0.0]
    assert switching.values_at(5.0) == [10.0, 2.0]
    assert switching.values_at(7.0) == [10.0, 2.0]
    assert switching.values_at(100.0) == [4.0, 0.0]

    ramp = InletSchedule((1.0,), (2.0, 4.0), ((1.0,), (5.0,)), ramp=True)
    assert ramp.values_at(1.0) == [1.0]
    assert ramp.values_at(3.0) == [3.0]
    assert ramp.values_at(4.0) == [5.0]
    assert ramp.values_at(9.0) == [5.0]

    pulse = InletSchedule.pulse([0.0], [3.0], start=2.0, end=6.0)
    assert [pulse.values_at(time)[0] for time in (1.9, 2.0, 5.9, 6.0)] == [0.0, 3.0, 3.0, 0.0]
    assert InletSchedule((7.0,)).values_at(123.0) == [7.0]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (((1.0,), (1.0,), ()), "one composition per time"),
        (((1.0,), (2.0, 2.0), ((1.0,), (1.0,))), "strictly increasing"),
        (((1.0,), (math.inf,), ((1.0,),)), "strictly increasing"),
        (((), (), ()), "at least one signal"),
        (((1.0,), (1.0,), ((1.0, 2.0),)), "share one signal count"),
        (((1.0,), (1.0,), ((-1.0,),)), "finite and non-negative"),
    ],
)
def test_inlet_schedule_rejects_invalid_definitions(
    arguments: tuple[tuple[float, ...], tuple[float, ...], tuple[tuple[float, ...], ...]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        InletSchedule(*arguments)
    with pytest.raises(ValueError, match="time must be finite"):
        InletSchedule((1.0,)).values_at(math.nan)


def test_device_inlet_values_change_the_media_entering_the_channel() -> None:
    device = TrapChannelDevice()
    spec = _grid()
    device.apply_to_grid(spec, inlet_values=[0.0], outlet_values=[0.0])
    simulation = Simulation()
    simulation.configure_signal_grid(spec, [0.0] * (spec.shape.x * spec.shape.y * spec.shape.z))
    simulation.step(0.01)
    assert max(simulation.signal_levels) == 0.0

    schedule = InletSchedule.pulse([0.0], [10.0], start=0.0, end=0.05)
    for _ in range(5):
        device.set_inlet_values(simulation, schedule.values_at(simulation.time))
        simulation.step(0.01)
    levels = simulation.signal_levels
    shape = spec.shape
    inlet_row = [
        levels[x * shape.y * shape.z + z]
        for x in range(shape.x)
        for z in range(shape.z)
        if spec.obstacles[x * shape.y * shape.z + z] == 0
    ]
    assert inlet_row and min(inlet_row) > 0.0
    # Solid voxels never store signal, whatever the inlet carries.
    assert all(level == 0.0 for level, solid in zip(levels, spec.obstacles, strict=True) if solid)


def _trap_chamber_device(mean_flow_speed: float = 0.0) -> LayoutDevice:
    scene = load_design_scene(_TRAP_CHAMBER)
    trap = scene.instance("trap")
    assert trap.low is not None and trap.high is not None
    window = scene.window(trap.low, trap.high)
    return LayoutDevice(window.boxes, window.size, mean_flow_speed=mean_flow_speed)


def _biopixel_grid(origin: Vec3) -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 42, 60, 14
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.origin = origin
    grid.spacing = Vec3(5.0, 5.0, 0.825)
    grid.diffusion = [40.0]
    return grid


def test_a_design_window_projects_as_the_device_stated_by_hand() -> None:
    # The design draws the biopixel cavity and channel with its window's low corner where the
    # hand-stated device has (-100, -150): the same lattice, moved by that much, must resolve
    # the same walls, and the same walls the same flow. Sites lie on the trap mouth, on both
    # side walls and on the back wall.
    stated = _biopixel_grid(Vec3(-100.0, -147.5, -0.4125))
    BiopixelTrapDevice(mean_flow_speed=20.0).apply_to_grid(stated, [10.0], [0.0])
    drawn = _biopixel_grid(Vec3(0.0, 2.5, -0.4125))
    _trap_chamber_device(mean_flow_speed=20.0).apply_to_grid(drawn, [10.0], [0.0])
    drawn.validate()

    assert drawn.obstacles == stated.obstacles
    assert 0 < sum(drawn.obstacles) < len(drawn.obstacles)
    assert (drawn.y_lower.values, drawn.y_upper.values) == ([10.0], [0.0])
    assert drawn.velocity_field is not None and stated.velocity_field is not None
    for mine, theirs in (
        (drawn.velocity_field.x_faces, stated.velocity_field.x_faces),
        (drawn.velocity_field.y_faces, stated.velocity_field.y_faces),
    ):
        assert len(mine) == len(theirs)
        assert all(abs(a - b) <= 1e-4 for a, b in zip(mine, theirs, strict=True))
    assert max(drawn.velocity_field.y_faces) > 20.0


def test_layout_device_gap_follows_the_boxes_and_their_seams() -> None:
    device = _trap_chamber_device()

    assert device.height == 10.0
    assert device.gap_height(50.0, 150.0) == 10.0
    assert device.gap_height(140.0, 150.0) == 1.65
    # The seam where the cavity opens onto the channel is fluid as high as the lower of the two.
    assert device.gap_height(100.0, 150.0) == 1.65
    # A point on a wall plane, or on the window's side, is in the wall.
    assert device.gap_height(140.0, 100.0) == 0.0
    assert device.gap_height(140.0, 200.0) == 0.0
    assert device.gap_height(0.0, 150.0) == 0.0
    assert device.gap_height(185.0, 150.0) == 0.0
    # The channel continues past the inlet and outlet faces as it meets them.
    assert device.gap_height(50.0, -40.0) == 10.0
    assert device.gap_height(50.0, 340.0) == 10.0
    assert device.gap_height(140.0, 340.0) == 0.0

    half = (0.0, 0.0, 0.0)
    assert not device._solid(140.0, 150.0, 0.825, half)
    assert device._solid(140.0, 150.0, 1.65, half)
    assert device._solid(50.0, 150.0, 0.0, half)
    assert device._solid(50.0, 150.0, 10.0, half)


def test_layout_device_lattice_covers_the_window_with_a_margin_of_wall() -> None:
    device = _trap_chamber_device(mean_flow_speed=20.0)
    origin, shape = device.lattice((5.0, 5.0, 0.825))

    assert (origin.x, origin.y) == (-2.5, 2.5) and math.isclose(origin.z, -0.4125, abs_tol=1e-6)
    assert (shape.x, shape.y, shape.z) == (39, 60, 14)

    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape, grid.origin, grid.spacing = shape, origin, Vec3(5.0, 5.0, 0.825)
    grid.diffusion = [40.0]
    device.apply_to_grid(grid, [10.0], [0.0])
    grid.validate()

    def solid(x: int, y: int, z: int) -> bool:
        return grid.obstacles[x * shape.y * shape.z + y * shape.z + z] == 1

    # The outermost sites across the flow and in depth are all wall.
    assert all(solid(x, y, z) for x in (0, shape.x - 1) for y in range(shape.y) for z in range(14))
    assert all(solid(x, y, z) for x in range(shape.x) for y in range(shape.y) for z in (0, 13))
    # The channel is open from the inlet face to the outlet face.
    assert not solid(10, 0, 6) and not solid(10, shape.y - 1, 6)
    with pytest.raises(ValueError, match="spacing must be positive"):
        device.lattice((5.0, 0.0, 1.0))


@pytest.mark.parametrize(
    ("inlet", "sign"),
    [
        (GridFace.X_LOWER, 1.0),
        (GridFace.X_UPPER, -1.0),
        (GridFace.Y_LOWER, 1.0),
        (GridFace.Y_UPPER, -1.0),
    ],
)
def test_layout_device_runs_media_from_its_inlet_face(inlet: GridFace, sign: float) -> None:
    along_x = inlet in (GridFace.X_LOWER, GridFace.X_UPPER)
    # A channel 20 wide down the middle of a window 100 long and 40 across.
    boxes = (
        (((0.0, 10.0, 0.0), (100.0, 30.0, 5.0)),)
        if along_x
        else (((10.0, 0.0, 0.0), (30.0, 100.0, 5.0)),)
    )
    device = LayoutDevice(
        boxes, (100.0, 40.0) if along_x else (40.0, 100.0), inlet=inlet, mean_flow_speed=3.0
    )
    origin, shape = device.lattice((2.0, 2.0, 1.0))
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape, grid.origin, grid.spacing = shape, origin, Vec3(2.0, 2.0, 1.0)
    grid.diffusion = [40.0]
    grid.integration = SignalIntegrationKind.BACKWARD_EULER
    device.apply_to_grid(grid, inlet_values=[7.0], outlet_values=[1.0])
    grid.validate()

    lower, upper = (grid.x_lower, grid.x_upper) if along_x else (grid.y_lower, grid.y_upper)
    entering, leaving = (lower, upper) if sign > 0.0 else (upper, lower)
    assert (entering.values, leaving.values) == ([7.0], [1.0])
    assert grid.velocity_field is not None
    faces = grid.velocity_field.x_faces if along_x else grid.velocity_field.y_faces
    moving = [speed for speed in faces if speed != 0.0]
    assert moving and all(speed * sign > 0.0 for speed in moving)
    assert max(abs(speed) for speed in moving) > 3.0

    simulation = Simulation()
    simulation.configure_signal_grid(grid, [0.0] * (shape.x * shape.y * shape.z))
    device.set_inlet_values(simulation, [9.0])
    simulation.step(0.01)
    levels = simulation.signal_levels

    def level(along: int) -> float:
        # A site at mid-width and mid-depth of the channel, this far along the flow axis.
        x, y = (along, shape.y // 2) if along_x else (shape.x // 2, along)
        return levels[x * shape.y * shape.z + y * shape.z + shape.z // 2]

    count = shape.x if along_x else shape.y
    first, last = (0, count - 1) if sign > 0.0 else (count - 1, 0)
    assert level(first) > level(last)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (((), (10.0, 10.0)), "at least one fluid box"),
        (((((0.0, 0.0, 0.0), (10.0, 10.0, 1.0)),), (10.0, 0.0)), "two positive finite extents"),
        (((((0.0, 0.0, 0.5), (10.0, 10.0, 1.0)),), (10.0, 10.0)), "rise from the floor"),
        (((((0.0, 0.0, 0.0), (10.0, 0.0, 1.0)),), (10.0, 10.0)), "beyond its low corner"),
        (((((0.0, 0.0, 0.0), (10.0, 12.0, 1.0)),), (10.0, 10.0)), "inside the window"),
        (((((0.0, 0.0, 0.0), (10.0, 8.0, 1.0)),), (10.0, 10.0)), "must both cut fluid"),
    ],
)
def test_layout_device_rejects_invalid_definitions(
    arguments: tuple[Any, tuple[float, float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LayoutDevice(*arguments)
    box = (((0.0, 0.0, 0.0), (10.0, 10.0, 1.0)),)
    with pytest.raises(ValueError, match="lower or upper x or y face"):
        LayoutDevice(box, (10.0, 10.0), inlet=GridFace.Z_LOWER)
    with pytest.raises(ValueError, match="wall thickness"):
        LayoutDevice(box, (10.0, 10.0), wall_thickness=0.0)


def _deepest_wall_contact(device: LayoutDevice, position: tuple[float, float, float]) -> float:
    """How far a short cell stands clear of a device's walls; negative when inside one."""

    simulation = Simulation()
    device.add_constraints(simulation)
    cell = CellInit()
    cell.position = Vec3(*position)
    cell.direction = Vec3(0.0, 1.0, 0.0)
    cell.length = 1.0
    cell.radius = 0.5
    simulation.add_cell(cell)
    contacts = simulation.find_external_contacts().contacts
    return min((contact.signed_separation for contact in contacts), default=math.inf)


def test_layout_device_walls_confine_cells_to_the_fluid() -> None:
    device = _trap_chamber_device()

    # Clear in the channel and in the middle of the shallow cavity.
    assert _deepest_wall_contact(device, (50.0, 150.0, 5.0)) > 0.0
    assert _deepest_wall_contact(device, (140.0, 150.0, 0.825)) > 0.0
    # Above the cavity's roof, in the wall beside it, behind its back, and outside the channel.
    assert _deepest_wall_contact(device, (140.0, 150.0, 5.0)) < -1.0
    assert _deepest_wall_contact(device, (140.0, 50.0, 0.825)) < -1.0
    assert _deepest_wall_contact(device, (190.0, 150.0, 0.825)) < -1.0
    assert _deepest_wall_contact(device, (-3.0, 150.0, 5.0)) < -1.0
    # Pressed against the cavity's roof and the channel's floor by less than its radius.
    assert -0.5 < _deepest_wall_contact(device, (140.0, 150.0, 1.4)) < 0.0
    assert -0.5 < _deepest_wall_contact(device, (50.0, 150.0, 0.3)) < 0.0


def test_layout_device_walls_stand_between_the_trenches_of_a_comb() -> None:
    # A channel along y with three dead-end trenches off its right side, 2 wide at a pitch of 6.
    channel = ((0.0, 0.0, 0.0), (20.0, 40.0, 8.0))
    trenches = tuple(((20.0, y, 0.0), (45.0, y + 2.0, 1.2)) for y in (10.0, 16.0, 22.0))
    device = LayoutDevice((channel, *trenches), (45.0, 40.0), wall_thickness=4.0)

    for y in (11.0, 17.0, 23.0):
        assert device.gap_height(30.0, y) == 1.2
        assert _deepest_wall_contact(device, (30.0, y, 0.6)) > -0.5
    for y in (5.0, 14.0, 20.0, 30.0):
        assert device.gap_height(30.0, y) == 0.0
        assert _deepest_wall_contact(device, (30.0, y, 0.6)) < -0.5
