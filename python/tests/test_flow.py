from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from microsimulator import (
    BackendFeature,
    BackendKind,
    FlowPort,
    FlowPortKind,
    GridBoundaryKind,
    GridBoundaryPatch,
    GridFace,
    GridFacePatch,
    GridShape,
    SignalGridAffineReaction,
    SignalGridSpec,
    SignalGridVelocityField,
    SignalIntegrationKind,
    Simulation,
    Vec3,
    backend_available,
)
from microsimulator.flow import (
    FlowError,
    colony_mobility,
    colony_species_density,
    gap_mobility,
    solve_flow_field,
    solve_port_flow_field,
)
from microsimulator.microfluidics import TrapChannelDevice


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
    assert all(math.isclose(value, 5.0, abs_tol=5.0e-5) for value in field.y_faces)
    assert all(abs(value) < 2.0e-5 for value in field.x_faces)
    assert all(abs(value) < 2.0e-5 for value in field.z_faces)
    assert math.isclose(report.max_speed, 5.0, rel_tol=1.0e-5)
    spec.velocity_field = field
    spec.validate()


@pytest.mark.parametrize("backend", list(BackendKind))
def test_depth_averaged_flow_uses_the_selected_native_backend(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend.name} backend is unavailable")
    spec = _duct(nx=3, ny=5, nz=1)
    expected, _ = solve_flow_field(spec, mean_inlet_speed=2.0)
    simulation = Simulation(backend)
    assert simulation.supports(BackendFeature.DEPTH_AVERAGED_FLOW)
    actual, report = solve_flow_field(spec, mean_inlet_speed=2.0, simulation=simulation)
    assert report.residual <= 1.0e-6
    assert all(
        math.isclose(observed, reference, abs_tol=5.0e-4, rel_tol=5.0e-4)
        for observed, reference in zip(actual.y_faces, expected.y_faces, strict=True)
    )


def test_parallel_channels_split_flux_in_the_mobility_ratio() -> None:
    spec = _duct(nx=2, ny=6, nz=1)
    mobility = [1.0 if x == 0 else 3.0 for x in range(2) for _ in range(6)]
    field, _ = solve_flow_field(spec, mean_inlet_speed=4.0, mobility=mobility)
    slow = field.y_faces[_y_face(spec, 0, 3, 0)]
    fast = field.y_faces[_y_face(spec, 1, 3, 0)]
    assert math.isclose(fast / slow, 3.0, rel_tol=1.0e-5)
    assert math.isclose((slow + fast) / 2.0, 4.0, rel_tol=1.0e-6)
    assert all(abs(value) < 2.0e-5 for value in field.x_faces)


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
    assert packed < sparse < empty <= 1.0
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
    assert all(math.isclose(value, 7.0, rel_tol=1.0e-5) for value in field.y_faces)


def test_reversed_and_transverse_flow_axes_solve() -> None:
    spec = _duct()
    field, _ = solve_flow_field(spec, mean_inlet_speed=-3.0)
    spec.velocity_field = field
    spec.validate()
    assert all(math.isclose(value, -3.0, abs_tol=2.0e-5) for value in field.y_faces)

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
    assert all(math.isclose(value, 2.0, abs_tol=2.0e-5) for value in sideways.x_faces)


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


def test_colony_species_density_rasterizes_one_channel() -> None:
    """Intracellular concentration times B is conserved by physical smoothing."""

    spec = _duct(nx=3, ny=3, nz=1)
    spec.spacing = Vec3(4.0, 4.0, 4.0)

    @dataclass
    class _Cell:
        position: Vec3
        species: list[float]
        length: float = 2.0
        radius: float = 0.5

    cells = [
        _Cell(Vec3(0.0, 0.0, 0.0), [1.0, 2.0]),
        _Cell(Vec3(0.0, 0.0, 0.0), [3.0, 4.0]),
        _Cell(Vec3(4.0, 4.0, 0.0), [0.0, 5.0]),
        _Cell(Vec3(400.0, 0.0, 0.0), [9.0, 9.0]),
    ]
    density = colony_species_density(spec, cells, species=1)
    voxel = spec.voxel_volume
    amount = 11 * math.pi * 0.5**2 * (2 + 2 * 0.5)
    assert math.isclose(sum(density) * voxel, amount, rel_tol=1e-7)
    assert density[_site(spec, 0, 0, 0)] > 0
    assert density[_site(spec, 1, 1, 0)] > 0
    with pytest.raises(FlowError, match="outside the cell"):
        colony_species_density(spec, cells, species=5)


def test_a_running_simulation_swaps_its_signal_reaction() -> None:
    """A loss that depends on cell state can be handed to transport each step."""

    spec = _duct(nx=3, ny=1, nz=1)
    spec.x_lower.kind = GridBoundaryKind.NO_FLUX
    spec.x_lower.values = []
    spec.x_upper.kind = GridBoundaryKind.NO_FLUX
    spec.x_upper.values = []
    for name in ("y_lower", "y_upper"):
        boundary = getattr(spec, name)
        boundary.kind = GridBoundaryKind.NO_FLUX
        boundary.values = []
        setattr(spec, name, boundary)
    spec.diffusion = [0.0]
    spec.integration = SignalIntegrationKind.CRANK_NICOLSON

    simulation = Simulation()
    simulation.configure_signal_grid(spec, [4.0, 4.0, 4.0])
    simulation.step(0.5)
    assert simulation.signal_levels == [4.0, 4.0, 4.0]

    reaction = SignalGridAffineReaction()
    reaction.source_rates = [0.0, 0.0, 0.0]
    reaction.loss_rates = [2.0, 0.0, 0.0]
    simulation.set_signal_reaction(reaction)
    simulation.step(0.5)
    levels = simulation.signal_levels
    assert levels[0] < 4.0
    assert levels[1] == 4.0

    # Transport takes the reaction into its implicit diagonal, so a loss that
    # would remove more than a site holds in one explicit step is still stable.
    # Crank-Nicolson stays positive while the loss times the step is under two.
    reaction.loss_rates = [3.0, 0.0, 0.0]
    simulation.set_signal_reaction(reaction)
    before_strong = simulation.signal_levels[0]
    simulation.step(0.5)
    assert 0.0 < simulation.signal_levels[0] < before_strong * 0.5

    simulation.set_signal_reaction(None)
    before = simulation.signal_levels
    simulation.step(0.5)
    assert simulation.signal_levels == before


def _open_box(nx: int, ny: int, nz: int = 1) -> SignalGridSpec:
    """A box of fluid whose four side faces are fixed boundaries a port may open."""

    spec = _duct(nx, ny, nz)
    for name in ("x_lower", "x_upper"):
        boundary = getattr(spec, name)
        boundary.kind = GridBoundaryKind.FIXED
        boundary.values = [0.0]
        setattr(spec, name, boundary)
    return spec


def _port(
    face: GridFace, begin: int, end: int, depth: int, kind: FlowPortKind, value: float
) -> FlowPort:
    return FlowPort(face, GridFacePatch((begin, 0), (end, depth)), kind, value)


@pytest.mark.parametrize("backend", list(BackendKind))
def test_ports_carry_their_stated_speeds_on_the_selected_backend(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend.name} backend is unavailable")
    spec = _open_box(8, 10, 2)
    ports = [
        _port(GridFace.Y_LOWER, 0, 3, 2, FlowPortKind.SPEED, 3.0),
        _port(GridFace.Y_LOWER, 5, 8, 2, FlowPortKind.SPEED, 1.0),
        _port(GridFace.X_UPPER, 6, 9, 2, FlowPortKind.SPEED, -0.5),
        _port(GridFace.Y_UPPER, 0, 8, 2, FlowPortKind.PRESSURE, 0.0),
    ]
    expected = solve_port_flow_field(spec, ports)
    simulation = Simulation(backend)
    result = solve_port_flow_field(spec, ports, simulation=simulation)

    assert result.report.solves == 3 and result.report.relative_residual <= 1.1e-6
    for port, state in zip(ports[:3], result.ports[:3], strict=True):
        assert math.isclose(state.mean_speed, port.value, rel_tol=1.0e-4)
    # What enters by the three driven ports leaves by the outlet.
    assert math.isclose(sum(state.flux for state in result.ports), 0.0, abs_tol=1.0e-4)
    assert result.ports[3].pressure == 0.0 and result.ports[0].pressure > result.ports[1].pressure
    for name in ("x_faces", "y_faces", "z_faces"):
        assert all(
            math.isclose(observed, reference, abs_tol=5.0e-4, rel_tol=5.0e-4)
            for observed, reference in zip(
                getattr(result.field, name), getattr(expected.field, name), strict=True
            )
        )
    # The lower face is closed between the two inlets.
    assert result.field.y_faces[_y_face(spec, 3, 0, 0)] == 0.0
    assert result.field.y_faces[_y_face(spec, 4, 0, 1)] == 0.0


@pytest.mark.parametrize("backend", list(BackendKind))
def test_two_inlets_on_one_face_meet_in_their_flow_shares(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend.name} backend is unavailable")
    # Two inlets on the lower face carry different media at different speeds
    # into one channel. Signal 0 marks the first medium and signal 1 the second.
    # A fixed face is a reservoir wherever fluid touches it, port or not, so a
    # wall stands on the lower face between the inlets.
    spec = _duct(8, 24, 1)
    spec.signal_count = 2
    spec.diffusion = [0.1, 0.1]
    spec.advection = [Vec3(), Vec3()]
    spec.integration = SignalIntegrationKind.BACKWARD_EULER
    obstacles = [0] * (8 * 24)
    obstacles[_site(spec, 3, 0, 0)] = obstacles[_site(spec, 4, 0, 0)] = 1
    spec.obstacles = obstacles
    upper = spec.y_upper
    upper.values = [0.0, 0.0]
    spec.y_upper = upper
    lower = spec.y_lower
    lower.values = [0.0, 0.0]
    lower.patches = [
        GridBoundaryPatch(GridFacePatch((0, 0), (3, 1)), [1.0, 0.0]),
        GridBoundaryPatch(GridFacePatch((5, 0), (8, 1)), [0.0, 1.0]),
    ]
    spec.y_lower = lower
    ports = [
        _port(GridFace.Y_LOWER, 0, 3, 1, FlowPortKind.SPEED, 3.0),
        _port(GridFace.Y_LOWER, 5, 8, 1, FlowPortKind.SPEED, 1.0),
        _port(GridFace.Y_UPPER, 0, 8, 1, FlowPortKind.PRESSURE, 0.0),
    ]
    simulation = Simulation(backend)
    spec.velocity_field = solve_port_flow_field(spec, ports, simulation=simulation).field
    simulation.configure_signal_grid(spec, [0.0] * (2 * 8 * 24))
    for _ in range(400):
        simulation.step(0.25)
    levels = np.asarray(simulation.signal_levels).reshape(2, 8, 24)

    # Each inlet's first sites hold that inlet's medium.
    assert levels[0, 1, 0] > 0.9 and levels[1, 1, 0] < 0.05
    assert levels[1, 6, 0] > 0.9 and levels[0, 6, 0] < 0.05
    # The media are carried, not made: everywhere they sum to no more than one.
    assert np.all(levels.sum(axis=0) <= 1.0 + 1.0e-4)
    # At the outlet the flow carries the media in the shares of the inlets'
    # fluxes, three to one, up to what diffuses back in from the outlet's
    # empty reservoir.
    field = np.asarray(spec.velocity_field.y_faces).reshape(8, 25)
    outflow = field[:, 23]
    carried = [float((levels[signal, :, 22] * outflow).sum() / outflow.sum()) for signal in (0, 1)]
    assert math.isclose(carried[0], 0.75, abs_tol=0.02)
    assert math.isclose(carried[1], 0.25, abs_tol=0.02)


def test_ill_posed_port_problems_are_rejected() -> None:
    spec = _open_box(4, 6, 2)
    inlet = _port(GridFace.Y_LOWER, 0, 2, 2, FlowPortKind.SPEED, 1.0)
    outlet = _port(GridFace.Y_UPPER, 0, 4, 2, FlowPortKind.PRESSURE, 0.0)
    solve_port_flow_field(spec, [inlet, outlet])

    with pytest.raises(FlowError, match="at least one port"):
        solve_port_flow_field(spec, [])
    with pytest.raises(FlowError, match="at least one pressure port"):
        solve_port_flow_field(spec, [inlet])
    with pytest.raises(FlowError, match="nothing flows"):
        solve_port_flow_field(spec, [outlet])
    with pytest.raises(FlowError, match="must not overlap"):
        overlapping = _port(GridFace.Y_LOWER, 1, 3, 2, FlowPortKind.SPEED, 1.0)
        solve_port_flow_field(spec, [inlet, overlapping, outlet])
    with pytest.raises(FlowError, match="at least one site of its face"):
        solve_port_flow_field(
            spec, [_port(GridFace.Y_LOWER, 0, 5, 2, FlowPortKind.SPEED, 1.0), outlet]
        )
    with pytest.raises(FlowError, match="full depth of its columns"):
        solve_port_flow_field(
            spec, [_port(GridFace.Y_LOWER, 0, 2, 1, FlowPortKind.SPEED, 1.0), outlet]
        )
    with pytest.raises(FlowError, match="must open a fixed boundary"):
        solve_port_flow_field(
            _duct(4, 6, 2), [_port(GridFace.X_LOWER, 0, 2, 2, FlowPortKind.SPEED, 1.0), outlet]
        )
    with pytest.raises(FlowError, match="finite"):
        solve_port_flow_field(
            spec, [_port(GridFace.Y_LOWER, 0, 2, 2, FlowPortKind.SPEED, math.inf), outlet]
        )
    walled = _open_box(4, 6, 2)
    obstacles = [0] * (4 * 6 * 2)
    for x in range(4):
        for z in range(2):
            obstacles[_site(walled, x, 3, z)] = 1
    walled.obstacles = obstacles
    with pytest.raises(FlowError, match="no pressure port is reachable"):
        solve_port_flow_field(walled, [inlet, outlet])
