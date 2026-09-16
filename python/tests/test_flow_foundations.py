from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from microsimulator import BackendKind, Simulation, Vec3, backend_available
from microsimulator.biomass import biomass_volume
from microsimulator.flow import FlowError, colony_volume_fraction, solve_flow_field
from microsimulator.flow_reference import duct_grid
from microsimulator.stokes import solve_stokes_field
from numpy.typing import NDArray


@dataclass
class Rod:
    position: Vec3
    length: float = 2.0
    radius: float = 0.5


@pytest.mark.parametrize("backend", list(BackendKind))
def test_parallel_depths_have_cubic_conductance(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    spec = duct_grid(2, 6, 4, (1, 1, 1))
    solid = np.zeros((2, 6, 4), dtype=np.uint8)
    solid[1, :, 1:] = 1
    spec.obstacles = solid.ravel().tolist()
    field, _ = solve_flow_field(spec, mean_inlet_speed=1, backend=backend)
    faces = np.asarray(field.y_faces, dtype=np.float64).reshape(2, 7, 4)
    assert math.isclose(float(faces[0, 3].sum() / faces[1, 3].sum()), 64, rel_tol=3e-5)
    assert float(faces[0, 3].max() - faces[0, 3].min()) < 1e-6


@pytest.mark.parametrize("backend", list(BackendKind))
def test_variable_depth_lift_conserves_uniform_tracer(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    spec = duct_grid(2, 8, 4, (1, 1, 1))
    solid = np.zeros((2, 8, 4), dtype=np.uint8)
    solid[:, 4:, 2:] = 1
    spec.obstacles = solid.ravel().tolist()
    field, _ = solve_flow_field(spec, mean_inlet_speed=1, backend=backend)
    fluxes = np.asarray(field.y_faces, dtype=np.float64).reshape(2, 9, 4).sum(axis=(0, 2))
    # Binary32 pressure differencing amplifies the pressure tolerance in fluxes.
    assert np.max(np.abs(fluxes / fluxes[0] - 1)) < 2e-5
    spec.velocity_field, spec.diffusion = field, [0]
    spec.y_lower.values = spec.y_upper.values = [1]
    simulation = Simulation(backend)
    initial = (solid == 0).astype(float).ravel().tolist()
    simulation.configure_signal_grid(spec, initial)
    simulation.step(0.01)
    assert max(abs(a - b) for a, b in zip(initial, simulation.signal_levels, strict=True)) < 2e-6


def test_shallow_model_rejects_vertical_variation_and_overhangs() -> None:
    spec = duct_grid(2, 6, 3, (1, 1, 1))
    with pytest.raises(FlowError, match="constant through"):
        solve_flow_field(spec, mean_inlet_speed=1, mobility=[1, 2, 3] * 12)
    solid = [0] * 36
    solid[1] = 1
    spec.obstacles = solid
    with pytest.raises(FlowError, match="contiguous"):
        solve_flow_field(spec, mean_inlet_speed=1)


def test_zero_mobility_barrier_is_rejected() -> None:
    spec = duct_grid(2, 6, 1, (1, 1, 1))
    with pytest.raises(FlowError, match="unreachable"):
        solve_flow_field(spec, mean_inlet_speed=1, mobility=[1, 1, 0, 1, 1, 1] * 2)


def test_biomass_deposition_conserves_amount_across_refinement() -> None:
    densities: list[NDArray[np.float64]] = []
    rod = Rod(Vec3(0.3, 0.1, -0.2))
    expected = 100 * biomass_volume(rod.length, rod.radius)
    for n in (4, 8, 16):
        h = 16 / n
        spec = duct_grid(n, n, n, (h, h, h))
        spec.origin = Vec3(*([-8 + h / 2] * 3))
        density = colony_volume_fraction(spec, [rod] * 100, averaging_radius=4)
        assert math.isclose(float(density.sum()) * spec.voxel_volume, expected, rel_tol=2e-7)
        densities.append(density)
    # Exact cell integrals agree after aggregating fine voxels, even above density 0.9.
    fine_to_coarse = densities[2].reshape(4, 4, 4, 4, 4, 4).mean(axis=(1, 3, 5))
    assert float(np.max(np.abs(fine_to_coarse - densities[0]))) < 1e-10
    assert float(densities[2].max()) > 0.9


@pytest.mark.parametrize("backend", list(BackendKind))
def test_inexact_preconditioner_preserves_true_stokes_solution(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    spec = duct_grid(5, 8, 4, (0.7, 1.1, 0.6))
    drag = [0.2 + 0.01 * i for i in range(spec.site_count)]
    expected, _ = solve_stokes_field(spec, mean_inlet_speed=1, drag=drag)
    actual, report = solve_stokes_field(
        spec, mean_inlet_speed=1, drag=drag, backend=backend, inner_tolerance=0.05
    )
    assert report.relative_residual <= 1e-6
    assert report.momentum_relative_residual <= 1e-6
    assert max(abs(a - b) for a, b in zip(actual.y_faces, expected.y_faces, strict=True)) < 1e-4
