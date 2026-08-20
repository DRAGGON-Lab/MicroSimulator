from __future__ import annotations

import math

import numpy as np
import pytest
from cellmodeller2 import GridBoundaryKind, Vec3
from cellmodeller2.flow import FlowError, gap_mobility, solve_flow_field
from cellmodeller2.flow_reference import (
    SQUARE_DUCT_PEAK_TO_MEAN,
    centerline_value,
    duct_grid,
    plane_poiseuille,
    site_index,
    two_layer_brinkman,
)
from cellmodeller2.stokes import colony_drag, solve_stokes_field


def _plane_poiseuille_error(nx: int) -> float:
    spec = duct_grid(nx, 6, 1, (1.0 / nx, 0.25, 1.0))
    field, _ = solve_stokes_field(spec, mean_inlet_speed=1.0, tolerance=1.0e-10)
    profile = np.asarray(field.y_faces).reshape(nx, 7, 1)[:, 3, 0]
    positions = (np.arange(nx) + 0.5) / nx
    exact = plane_poiseuille(positions)
    return float(np.max(np.abs(profile - exact)) / np.max(exact))


def test_plane_poiseuille_profile_converges_at_second_order() -> None:
    coarse = _plane_poiseuille_error(8)
    fine = _plane_poiseuille_error(16)
    assert coarse < 0.02
    assert fine < 0.005
    assert 3.0 < coarse / fine < 5.0


def test_square_duct_peak_to_mean_matches_shah_and_london() -> None:
    # u_max / u_mean = 2.0962 for a square duct (Shah & London 1978).
    n = 16
    spec = duct_grid(n, 6, n, (1.0 / n, 0.25, 1.0 / n))
    field, report = solve_stokes_field(spec, mean_inlet_speed=1.0, tolerance=1.0e-9)
    cross = np.asarray(field.y_faces).reshape(n, 7, n)[:, 3, :]
    ratio = centerline_value(cross) / float(cross.mean())
    assert abs(ratio - SQUARE_DUCT_PEAK_TO_MEAN) / SQUARE_DUCT_PEAK_TO_MEAN < 0.015
    assert report.divergence_rms < 1.0e-6


def test_two_layer_brinkman_channel_matches_the_exact_solution() -> None:
    drag_value = 200.0
    nz = 32
    spec = duct_grid(1, 6, nz, (1.0, 0.25, 1.0 / nz))
    drag = [
        0.0 if (z + 0.5) / nz < 0.5 else drag_value
        for _ in range(1)
        for _ in range(6)
        for z in range(nz)
    ]
    field, _ = solve_stokes_field(spec, mean_inlet_speed=1.0, drag=drag, tolerance=1.0e-9)
    profile = np.asarray(field.y_faces).reshape(1, 7, nz)[0, 3, :]
    positions = (np.arange(nz) + 0.5) / nz
    # The solve rescales to the requested mean speed, so both profiles are
    # compared at unit mean.
    exact = two_layer_brinkman(drag_value, positions)
    exact = exact / exact.mean()
    error = float(np.max(np.abs(profile / profile.mean() - exact)) / np.max(np.abs(exact)))
    assert error < 0.01
    # The open layer carries several times the porous layer's flux.
    open_flux = float(profile[positions < 0.5].sum())
    porous_flux = float(profile[positions >= 0.5].sum())
    assert open_flux / porous_flux > 3.0


def test_zero_drag_recovers_pure_stokes() -> None:
    spec = duct_grid(8, 6, 1, (0.125, 0.25, 1.0))
    plain, _ = solve_stokes_field(spec, mean_inlet_speed=2.0)
    dragged, _ = solve_stokes_field(spec, mean_inlet_speed=2.0, drag=[0.0] * (8 * 6))
    assert plain.y_faces == dragged.y_faces


def test_stokes_field_is_engine_valid_and_conservative_around_a_pillar() -> None:
    spec = duct_grid(9, 12, 1, (1.0, 1.0, 1.0))
    obstacles = [0] * (9 * 12)
    for y in (5, 6):
        for x in (4, 5):
            obstacles[site_index(spec, x, y, 0)] = 1
    spec.obstacles = obstacles
    field, report = solve_stokes_field(spec, mean_inlet_speed=6.0, tolerance=1.0e-9)
    spec.velocity_field = field
    spec.validate()
    assert report.divergence_rms < 1.0e-6

    def y_face(x: int, fy: int) -> float:
        return field.y_faces[x * 13 + fy]

    fluxes = [sum(y_face(x, fy) for x in range(9)) for fy in range(13)]
    for flux in fluxes[1:]:
        assert math.isclose(flux, fluxes[0], rel_tol=1.0e-5)
    assert y_face(4, 6) == 0.0
    assert y_face(1, 6) > 6.0


def test_thin_gap_stokes_depth_averages_to_the_hele_shaw_solution() -> None:
    # A shallow channel with a pillar: depth-averaging the resolved MAC field
    # must reproduce the Hele-Shaw flux split around the pillar.
    nx, ny, nz = 6, 10, 6
    spec = duct_grid(nx, ny, nz, (1.0, 1.0, 0.05))
    obstacles = [0] * (nx * ny * nz)
    for y in (4, 5):
        for x in (1, 2):
            for z in range(nz):
                obstacles[site_index(spec, x, y, z)] = 1
    spec.obstacles = obstacles

    stokes_field, _ = solve_stokes_field(spec, mean_inlet_speed=1.0, tolerance=1.0e-9)
    hele_shaw_field, _ = solve_flow_field(
        spec, mean_inlet_speed=1.0, mobility=gap_mobility(spec)
    )

    def column_flux(field_values: list[float], x: int, fy: int) -> float:
        return sum(field_values[(x * (ny + 1) + fy) * nz + z] for z in range(nz))

    mid = ny // 2
    stokes_split = [column_flux(stokes_field.y_faces, x, mid) for x in range(nx)]
    hele_shaw_split = [column_flux(hele_shaw_field.y_faces, x, mid) for x in range(nx)]
    stokes_total = sum(stokes_split)
    hele_shaw_total = sum(hele_shaw_split)
    for x in range(nx):
        assert math.isclose(
            stokes_split[x] / stokes_total,
            hele_shaw_split[x] / hele_shaw_total,
            abs_tol=0.01,
        )


def test_ill_posed_stokes_problems_are_rejected() -> None:
    spec = duct_grid(4, 6, 1, (1.0, 1.0, 1.0))
    with pytest.raises(FlowError, match="one of x, y, z"):
        solve_stokes_field(spec, mean_inlet_speed=1.0, axis="w")
    with pytest.raises(FlowError, match="finite and nonzero"):
        solve_stokes_field(spec, mean_inlet_speed=0.0)
    with pytest.raises(FlowError, match="must be FIXED"):
        solve_stokes_field(spec, mean_inlet_speed=1.0, axis="x")
    with pytest.raises(FlowError, match="one value per grid site"):
        solve_stokes_field(spec, mean_inlet_speed=1.0, drag=[1.0])

    blocked = duct_grid(3, 4, 1, (1.0, 1.0, 1.0))
    obstacles = [0] * 12
    for x in range(3):
        obstacles[site_index(blocked, x, 2, 0)] = 1
    blocked.obstacles = obstacles
    with pytest.raises(FlowError, match="no through-flow"):
        solve_stokes_field(blocked, mean_inlet_speed=1.0)


def test_colony_drag_rasterizes_the_colony() -> None:
    spec = duct_grid(3, 3, 1, (4.0, 4.0, 4.0))
    obstacles = [0] * 9
    obstacles[site_index(spec, 2, 2, 0)] = 1
    spec.obstacles = obstacles

    class _Rod:
        def __init__(self, x: float, y: float) -> None:
            self.position = Vec3(x, y, 0.0)
            self.length = 3.0
            self.radius = 0.5

    crowd = [_Rod(0.0, 0.0) for _ in range(40)]
    drag = colony_drag(spec, crowd, drag_coefficient=50.0)
    packed = drag[site_index(spec, 0, 0, 0)]
    empty = drag[site_index(spec, 1, 1, 0)]
    solid = drag[site_index(spec, 2, 2, 0)]
    assert packed > 0.0
    assert empty == 0.0
    assert solid == 0.0
    assert math.isclose(packed, 50.0 * 0.9**2 / (1.0 - 0.9) ** 3, rel_tol=1.0e-9)
    with pytest.raises(FlowError, match="finite and non-negative"):
        colony_drag(spec, [], drag_coefficient=-1.0)


def test_thin_gaps_over_predict_flux_until_they_are_resolved() -> None:
    """The MAC solve needs several voxels across a channel to resolve no-slip.

    Two stacked channels of gap ratio four carry flux in the ratio of their
    cubed gaps, so their mean velocities differ by the squared ratio. A gap
    one voxel across cannot hold a parabola and carries far too much; the
    ratio approaches the lubrication limit as the gap resolves, and the report
    names the resolution so a caller can judge the error.
    """

    lubrication = 1.0 / 16.0
    errors: list[float] = []
    for thin in (1, 2, 4, 8):
        nz = thin + 1 + 4 * thin
        spec = duct_grid(1, 8, nz, (1.0, 1.0, 1.0))
        obstacles = [0] * (8 * nz)
        for y in range(8):
            obstacles[site_index(spec, 0, y, thin)] = 1
        spec.obstacles = obstacles
        field, report = solve_stokes_field(spec, mean_inlet_speed=1.0, tolerance=1.0e-9)
        profile = np.asarray(field.y_faces).reshape(1, 9, nz)[0, 4, :]
        ratio = float(profile[:thin].mean() / profile[thin + 1 :].mean())
        errors.append(ratio / lubrication)
        assert report.min_gap_voxels == thin
    assert errors[0] > 2.0
    assert errors[1] < errors[0]
    assert errors[2] < 1.2
    assert errors[3] < 1.05


def test_reversed_and_transverse_flow_axes_solve() -> None:
    forward = duct_grid(8, 6, 1, (0.125, 0.25, 1.0))
    field, report = solve_stokes_field(forward, mean_inlet_speed=-2.0)
    forward.velocity_field = field
    forward.validate()
    inlet = np.asarray(field.y_faces).reshape(8, 7, 1)[:, 0, 0]
    assert math.isclose(float(inlet.mean()), -2.0, rel_tol=1.0e-6)
    assert report.max_speed > 0.0

    # The same channel across x reproduces the same profile.
    across = duct_grid(6, 8, 1, (0.25, 0.125, 1.0))
    across.y_lower.kind = GridBoundaryKind.NO_FLUX
    across.y_lower.values = []
    across.y_upper.kind = GridBoundaryKind.NO_FLUX
    across.y_upper.values = []
    for name in ("x_lower", "x_upper"):
        boundary = getattr(across, name)
        boundary.kind = GridBoundaryKind.FIXED
        boundary.values = [0.0]
        setattr(across, name, boundary)
    sideways, _ = solve_stokes_field(across, mean_inlet_speed=1.0, axis="x")
    across.velocity_field = sideways
    across.validate()
    profile = np.asarray(sideways.x_faces).reshape(7, 8, 1)[3, :, 0]
    positions = (np.arange(8) + 0.5) / 8
    error = float(np.max(np.abs(profile - plane_poiseuille(positions))) / 1.5)
    assert error < 0.02


def test_partly_blocked_inlets_and_walled_off_pockets_solve() -> None:
    spec = duct_grid(4, 6, 1, (1.0, 1.0, 1.0))
    obstacles = [0] * 24
    for y in range(6):
        obstacles[site_index(spec, 0, y, 0)] = 1
    spec.obstacles = obstacles
    field, report = solve_stokes_field(spec, mean_inlet_speed=2.0, tolerance=1.0e-9)
    spec.velocity_field = field
    spec.validate()
    inlet = np.asarray(field.y_faces).reshape(4, 7, 1)[:, 0, 0]
    # The mean is taken over open inlet faces, and the blocked column is still.
    assert math.isclose(float(inlet[1:].mean()), 2.0, rel_tol=1.0e-6)
    assert inlet[0] == 0.0
    assert report.divergence_rms < 1.0e-6

    # A fluid site sealed off from the flow leaves the solve well posed.
    pocket = duct_grid(5, 6, 1, (1.0, 1.0, 1.0))
    sealed = [0] * 30
    for y in (2, 4):
        for x in (3, 4):
            sealed[site_index(pocket, x, y, 0)] = 1
    for x in (3, 4):
        sealed[site_index(pocket, x, 3, 0)] = 0
    sealed[site_index(pocket, 2, 3, 0)] = 1
    pocket.obstacles = sealed
    sealed_field, sealed_report = solve_stokes_field(pocket, mean_inlet_speed=1.0)
    pocket.velocity_field = sealed_field
    pocket.validate()
    assert sealed_report.divergence_rms < 1.0e-6
    assert sealed_field.y_faces[(3 * 7) + 3] == 0.0
