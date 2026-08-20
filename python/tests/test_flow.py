from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
from cellmodeller2 import (
    GridBoundaryKind,
    GridShape,
    SignalGridSpec,
    SignalGridVelocityField,
    SignalIntegrationKind,
    Simulation,
    Vec3,
)
from cellmodeller2.flow import FlowError, colony_mobility, gap_mobility, solve_flow_field
from cellmodeller2.microfluidics import TrapChannelDevice


def _duct(nx: int = 4, ny: int = 8, nz: int = 3) -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = nx, ny, nz
    spec = SignalGridSpec()
    spec.signal_count = 1
    spec.shape = shape
    spec.spacing = Vec3(1.0, 1.0, 1.0)
    spec.diffusion = [1.0]
    spec.advection = [Vec3()]
    for name in ("y_lower", "y_upper"):
        boundary = getattr(spec, name)
        boundary.kind = GridBoundaryKind.FIXED
        boundary.values = [0.0]
        setattr(spec, name, boundary)
    return spec


def _site(spec: SignalGridSpec, x: int, y: int, z: int) -> int:
    return (x * spec.shape.y + y) * spec.shape.z + z


def _y_face(spec: SignalGridSpec, x: int, fy: int, z: int) -> int:
    return (x * (spec.shape.y + 1) + fy) * spec.shape.z + z


def _cross_section_fluxes(spec: SignalGridSpec, field: SignalGridVelocityField) -> list[float]:
    return [
        sum(
            field.y_faces[_y_face(spec, x, fy, z)]
            for x in range(spec.shape.x)
            for z in range(spec.shape.z)
        )
        for fy in range(spec.shape.y + 1)
    ]


def test_uniform_duct_is_exact_plug_flow() -> None:
    spec = _duct()
    field, report = solve_flow_field(spec, mean_inlet_speed=5.0)
    assert all(math.isclose(value, 5.0, abs_tol=1.0e-8) for value in field.y_faces)
    assert all(abs(value) < 1.0e-8 for value in field.x_faces)
    assert all(abs(value) < 1.0e-8 for value in field.z_faces)
    assert math.isclose(report.max_speed, 5.0, rel_tol=1.0e-9)
    spec.velocity_field = field
    spec.validate()


def test_parallel_channels_split_flux_in_the_mobility_ratio() -> None:
    spec = _duct(nx=2, ny=6, nz=1)
    mobility = [1.0 if x == 0 else 3.0 for x in range(2) for _ in range(6)]
    field, _ = solve_flow_field(spec, mean_inlet_speed=4.0, mobility=mobility)
    slow = field.y_faces[_y_face(spec, 0, 3, 0)]
    fast = field.y_faces[_y_face(spec, 1, 3, 0)]
    assert math.isclose(fast / slow, 3.0, rel_tol=1.0e-6)
    assert math.isclose((slow + fast) / 2.0, 4.0, rel_tol=1.0e-9)
    assert all(abs(value) < 1.0e-8 for value in field.x_faces)


def test_a_pillar_routes_flow_around_itself_conservatively() -> None:
    spec = _duct(nx=5, ny=7, nz=1)
    obstacles = [0] * (5 * 7)
    for y in (2, 3, 4):
        obstacles[_site(spec, 2, y, 0)] = 1
    spec.obstacles = obstacles
    field, _ = solve_flow_field(spec, mean_inlet_speed=6.0)

    fluxes = _cross_section_fluxes(spec, field)
    for flux in fluxes[1:]:
        assert math.isclose(flux, fluxes[0], rel_tol=1.0e-6)
    # Faces of the pillar carry no flow; its flanks carry more than the inlet mean.
    assert field.y_faces[_y_face(spec, 2, 3, 0)] == 0.0
    assert field.y_faces[_y_face(spec, 1, 3, 0)] > 6.0
    assert any(value != 0.0 for value in field.x_faces)
    spec.velocity_field = field
    spec.validate()


def test_brinkman_drag_diverts_flux_from_a_porous_region() -> None:
    spec = _duct(nx=2, ny=6, nz=1)
    mobility = [1.0] * (2 * 6)
    for y in (2, 3):
        mobility[_site(spec, 1, y, 0)] = 0.05
    field, _ = solve_flow_field(spec, mean_inlet_speed=4.0, mobility=mobility)
    open_flux = field.y_faces[_y_face(spec, 0, 3, 0)]
    porous_flux = field.y_faces[_y_face(spec, 1, 3, 0)]
    assert porous_flux > 0.0
    assert open_flux > 4.0 > porous_flux
    fluxes = _cross_section_fluxes(spec, field)
    for flux in fluxes[1:]:
        assert math.isclose(flux, fluxes[0], rel_tol=1.0e-6)


def test_ill_posed_problems_are_rejected() -> None:
    spec = _duct()
    with pytest.raises(FlowError, match="one of x, y, z"):
        solve_flow_field(spec, mean_inlet_speed=1.0, axis="w")
    with pytest.raises(FlowError, match="finite and nonzero"):
        solve_flow_field(spec, mean_inlet_speed=0.0)
    with pytest.raises(FlowError, match="must be FIXED"):
        solve_flow_field(spec, mean_inlet_speed=1.0, axis="x")
    with pytest.raises(FlowError, match="one value per grid site"):
        solve_flow_field(spec, mean_inlet_speed=1.0, mobility=[1.0])

    periodic = _duct()
    boundary = periodic.x_lower
    boundary.kind = GridBoundaryKind.PERIODIC
    periodic.x_lower = boundary
    boundary = periodic.x_upper
    boundary.kind = GridBoundaryKind.PERIODIC
    periodic.x_upper = boundary
    with pytest.raises(FlowError, match="periodic"):
        solve_flow_field(periodic, mean_inlet_speed=1.0)

    blocked_inlet = _duct(nx=3, ny=4, nz=1)
    obstacles = [0] * (3 * 4)
    for x in range(3):
        obstacles[_site(blocked_inlet, x, 0, 0)] = 1
    blocked_inlet.obstacles = obstacles
    with pytest.raises(FlowError, match="entirely blocked"):
        solve_flow_field(blocked_inlet, mean_inlet_speed=1.0)

    dead_end = _duct(nx=3, ny=4, nz=1)
    obstacles = [0] * (3 * 4)
    for x in range(3):
        obstacles[_site(dead_end, x, 2, 0)] = 1
    dead_end.obstacles = obstacles
    with pytest.raises(FlowError, match="no through-flow"):
        solve_flow_field(dead_end, mean_inlet_speed=1.0)


@dataclass(frozen=True)
class _Rod:
    position: Vec3
    length: float = 3.0
    radius: float = 0.5


def test_colony_mobility_adds_drag_where_cells_pack() -> None:
    spec = _duct(nx=3, ny=3, nz=1)
    spec.spacing = Vec3(4.0, 4.0, 4.0)
    obstacles = [0] * 9
    obstacles[_site(spec, 2, 2, 0)] = 1
    spec.obstacles = obstacles
    # Site centers sit at multiples of the spacing: voxel (0,0,0) is centered
    # on the origin and voxel (1,1,0) on (4, 4, 0).
    crowd = [_Rod(Vec3(0.0, 0.0, 0.0)) for _ in range(40)]
    lone = [_Rod(Vec3(4.0, 4.0, 0.0))]
    outside = [_Rod(Vec3(-10.0, 0.0, 0.0))]
    mobility = colony_mobility(spec, crowd + lone + outside, base=1.0, drag_coefficient=100.0)
    packed = mobility[_site(spec, 0, 0, 0)]
    sparse = mobility[_site(spec, 1, 1, 0)]
    empty = mobility[_site(spec, 0, 2, 0)]
    assert mobility[_site(spec, 2, 2, 0)] == 0.0
    assert packed < sparse < empty == 1.0
    # Packed voxels hit the volume-fraction cap rather than shrinking without bound.
    capped = 1.0 / (1.0 + 100.0 * 0.9**2 / (1.0 - 0.9) ** 3)
    assert math.isclose(packed, capped, rel_tol=1.0e-9)

    # A per-site base composes with the colony drag; zero-drag recovery is exact.
    layered = colony_mobility(spec, [], base=[0.5] * 9, drag_coefficient=100.0)
    assert layered[_site(spec, 1, 1, 0)] == 0.5
    assert layered[_site(spec, 2, 2, 0)] == 0.0

    with pytest.raises(FlowError, match="finite and positive"):
        colony_mobility(spec, [], base=0.0)
    with pytest.raises(FlowError, match="one value per grid site"):
        colony_mobility(spec, [], base=[1.0])
    with pytest.raises(FlowError, match="strictly between"):
        colony_mobility(spec, [], max_volume_fraction=1.0)


def test_gap_mobility_scales_with_the_squared_gap_height() -> None:
    spec = _duct(nx=2, ny=4, nz=4)
    obstacles = [0] * (2 * 4 * 4)
    for y in range(4):
        for z in range(1, 4):
            obstacles[_site(spec, 1, y, z)] = 1
    spec.obstacles = obstacles
    mobility = gap_mobility(spec)
    assert mobility[_site(spec, 0, 0, 0)] == 1.0
    assert math.isclose(mobility[_site(spec, 1, 0, 0)], 0.0625)
    assert mobility[_site(spec, 1, 0, 2)] == 0.0

    blocked = _duct(nx=1, ny=1, nz=1)
    blocked.obstacles = [1]
    with pytest.raises(FlowError, match="no fluid sites"):
        gap_mobility(blocked)


def test_simulation_swaps_the_solved_field_at_runtime() -> None:
    spec = _duct(nx=2, ny=4, nz=1)
    field, _ = solve_flow_field(spec, mean_inlet_speed=3.0)
    simulation = Simulation()
    simulation.configure_signal_grid(spec, [0.0] * spec.site_count)
    simulation.set_velocity_field(field)

    invalid = SignalGridVelocityField()
    invalid.x_faces = [0.0]
    invalid.y_faces = [0.0]
    invalid.z_faces = [0.0]
    with pytest.raises(ValueError, match="every lattice face"):
        simulation.set_velocity_field(invalid)
    simulation.set_velocity_field(None)


def test_a_solved_field_advects_signals_through_the_engine() -> None:
    def _mid_level(with_flow: bool) -> float:
        spec = _duct(nx=1, ny=12, nz=1)
        spec.diffusion = [0.01]
        spec.integration = SignalIntegrationKind.CRANK_NICOLSON
        boundary = spec.y_lower
        boundary.values = [10.0]
        spec.y_lower = boundary
        if with_flow:
            field, _ = solve_flow_field(spec, mean_inlet_speed=2.0)
            spec.velocity_field = field
        simulation = Simulation()
        simulation.configure_signal_grid(spec, [0.0] * spec.site_count)
        for _ in range(10):
            simulation.step(0.5)
        return simulation.sample_signals(Vec3(0.5, 6.0, 0.5))[0]

    advected = _mid_level(with_flow=True)
    diffused = _mid_level(with_flow=False)
    assert advected > 5.0
    assert advected > 10.0 * diffused


def test_trap_channel_device_supports_a_numerical_field() -> None:
    device = TrapChannelDevice(mean_flow_speed=20.0)
    shape = GridShape()
    shape.x, shape.y, shape.z = 64, 72, 4
    spec = SignalGridSpec()
    spec.signal_count = 1
    spec.shape = shape
    spec.origin = Vec3(-140.0, -144.0, -8.0)
    spec.spacing = Vec3(4.0, 4.0, 4.0)
    spec.diffusion = [40.0]
    spec.advection = [Vec3()]
    device.apply_to_grid(spec, inlet_values=[10.0], outlet_values=[0.0])

    field, report = solve_flow_field(spec, mean_inlet_speed=20.0)
    spec.velocity_field = field
    spec.validate()

    # The straight channel carries plug flow near the requested mean; the
    # dead-end trap sees only the weak recirculation at its mouth.
    mid_face = shape.y // 2
    channel_speed = max(
        field.y_faces[_y_face(spec, x, mid_face, z)]
        for x in range(shape.x)
        for z in range(shape.z)
    )
    trap_column = int((0.0 - spec.origin.x) / spec.spacing.x)
    trap_speed = abs(field.y_faces[_y_face(spec, trap_column, mid_face, 1)])
    assert channel_speed > 15.0
    assert trap_speed < channel_speed * 0.05
    assert report.max_speed >= channel_speed


def test_anisotropic_spacing_scales_the_solved_speeds() -> None:
    """A duct's mean speed is set by the request, not by the lattice shape.

    The conductances divide by the squared spacing and the face fluxes
    multiply it back, so a grid whose voxels are not cubes must still carry
    exactly the requested speed, and a stretched cross-section must split
    flux evenly across it.
    """

    shape = GridShape()
    shape.x, shape.y, shape.z = 4, 6, 3
    spec = SignalGridSpec()
    spec.signal_count = 1
    spec.shape = shape
    spec.spacing = Vec3(5.0, 0.4, 1.65)
    spec.diffusion = [1.0]
    spec.advection = [Vec3()]
    for name in ("y_lower", "y_upper"):
        boundary = getattr(spec, name)
        boundary.kind = GridBoundaryKind.FIXED
        boundary.values = [0.0]
        setattr(spec, name, boundary)
    field, _ = solve_flow_field(spec, mean_inlet_speed=7.0)
    spec.velocity_field = field
    spec.validate()
    assert all(math.isclose(value, 7.0, rel_tol=1.0e-8) for value in field.y_faces)


def test_reversed_and_transverse_flow_axes_solve() -> None:
    spec = _duct()
    field, _ = solve_flow_field(spec, mean_inlet_speed=-3.0)
    spec.velocity_field = field
    spec.validate()
    assert all(math.isclose(value, -3.0, abs_tol=1.0e-8) for value in field.y_faces)

    across = _duct()
    for name in ("y_lower", "y_upper"):
        boundary = getattr(across, name)
        boundary.kind = GridBoundaryKind.NO_FLUX
        boundary.values = []
        setattr(across, name, boundary)
    for name in ("x_lower", "x_upper"):
        boundary = getattr(across, name)
        boundary.kind = GridBoundaryKind.FIXED
        boundary.values = [0.0]
        setattr(across, name, boundary)
    sideways, _ = solve_flow_field(across, mean_inlet_speed=2.0, axis="x")
    across.velocity_field = sideways
    across.validate()
    assert all(math.isclose(value, 2.0, abs_tol=1.0e-8) for value in sideways.x_faces)


def test_partly_blocked_inlets_and_walled_off_pockets_solve() -> None:
    spec = _duct(nx=4, ny=6, nz=1)
    obstacles = [0] * 24
    for y in range(6):
        obstacles[_site(spec, 0, y, 0)] = 1
    spec.obstacles = obstacles
    field, _ = solve_flow_field(spec, mean_inlet_speed=2.0)
    spec.velocity_field = field
    spec.validate()
    inlet = [field.y_faces[_y_face(spec, x, 0, 0)] for x in range(4)]
    assert inlet[0] == 0.0
    assert all(math.isclose(value, 2.0, rel_tol=1.0e-8) for value in inlet[1:])
    fluxes = _cross_section_fluxes(spec, field)
    assert all(math.isclose(flux, fluxes[0], rel_tol=1.0e-8) for flux in fluxes)

    pocket = _duct(nx=5, ny=6, nz=1)
    sealed = [0] * 30
    for y in (2, 4):
        for x in (3, 4):
            sealed[_site(pocket, x, y, 0)] = 1
    sealed[_site(pocket, 2, 3, 0)] = 1
    pocket.obstacles = sealed
    sealed_field, _ = solve_flow_field(pocket, mean_inlet_speed=1.0)
    pocket.velocity_field = sealed_field
    pocket.validate()
    # The pocket is cut off from the flow, so it carries none.
    assert sealed_field.y_faces[_y_face(pocket, 3, 3, 0)] == 0.0
    assert sealed_field.y_faces[_y_face(pocket, 4, 3, 0)] == 0.0
