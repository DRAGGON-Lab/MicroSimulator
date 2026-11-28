from __future__ import annotations

import math
from dataclasses import dataclass, field

import deposition_reference as reference
import numpy as np
import pytest
from microsimulator import BackendKind, SignalGridSpec, Vec3, backend_available
from microsimulator.flow import (
    colony_flow_coupling,
    colony_mobility,
    colony_species_density,
    colony_volume_fraction,
)
from microsimulator.flow_reference import duct_grid
from microsimulator.stokes import colony_drag
from numpy.typing import NDArray


@dataclass
class Rod:
    position: Vec3
    length: float
    radius: float = 0.5
    species: list[float] = field(default_factory=lambda: [1.0])


def _device() -> tuple[SignalGridSpec, NDArray[np.uint8]]:
    """A shallow channel with a pillar and a partial wall."""
    spec = duct_grid(24, 32, 3, (1, 1, 1))
    spec.origin = Vec3(-11.5, -15.5, 0.0)
    solid = np.zeros((24, 32, 3), dtype=np.uint8)
    solid[10:14, 12:16, :] = 1
    solid[4, :20, :] = 1
    spec.obstacles = solid.ravel().tolist()
    return spec, solid


def _colony(solid: NDArray[np.uint8], count: int, seed: int) -> list[tuple[Rod, Vec3]]:
    rng = np.random.default_rng(seed)
    cells: list[tuple[Rod, Vec3]] = []
    while len(cells) < count:
        x, y = rng.uniform(-11.5, 11.5), rng.uniform(-15.5, 15.5)
        z = rng.uniform(0.0, 2.0)
        if solid[round(x + 11.5), round(y + 15.5), round(z)]:
            continue
        rod = Rod(Vec3(x, y, z), float(rng.uniform(1.0, 4.0)), species=[float(rng.uniform())])
        cells.append((rod, Vec3(float(rng.normal()), float(rng.normal()), 0.0)))
    return cells


def _relative(native: object, expected: object) -> float:
    a = np.asarray(native, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


def test_native_deposition_matches_the_float64_reference() -> None:
    spec, solid = _device()
    pairs = _colony(solid, 400, seed=7)
    rods = [rod for rod, _ in pairs]
    assert _relative(
        colony_volume_fraction(spec, rods), reference.colony_volume_fraction(spec, rods)
    ) < 1e-5
    assert _relative(
        colony_species_density(spec, rods, species=0),
        reference.colony_species_density(spec, rods, species=0),
    ) < 1e-5
    base = (1.0 - solid.astype(np.float64)).ravel().tolist()
    mobility, force = colony_flow_coupling(
        spec, pairs, base=base, drag_coefficient=2.0, max_volume_fraction=0.5
    )
    expected_mobility, expected_force = reference.colony_flow_coupling(
        spec, pairs, base=base, drag_coefficient=2.0, max_volume_fraction=0.5
    )
    assert _relative(mobility, expected_mobility) < 1e-5
    assert _relative(force, expected_force) < 1e-5
    assert _relative(
        colony_mobility(spec, rods, base=0.5, drag_coefficient=2.0),
        reference.colony_mobility(spec, rods, base=0.5, drag_coefficient=2.0),
    ) < 1e-5


def test_native_brinkman_drag_matches_its_formula() -> None:
    spec, solid = _device()
    rods = [rod for rod, _ in _colony(solid, 300, seed=3)]
    drag = np.asarray(colony_drag(spec, rods, drag_coefficient=5.0, max_volume_fraction=0.6))
    fraction = np.minimum(reference.colony_volume_fraction(spec, rods), 0.6).ravel()
    expected = 5.0 * fraction**2 / (1.0 - fraction) ** 3
    expected[solid.ravel() != 0] = 0.0
    assert _relative(drag, expected) < 1e-5


@pytest.mark.parametrize("backend", [BackendKind.METAL, BackendKind.CUDA])
def test_device_deposition_matches_the_cpu(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend.name} backend is unavailable")
    spec, solid = _device()
    pairs = _colony(solid, 500, seed=11)
    rods = [rod for rod, _ in pairs]
    base = (0.5 * (1.0 - solid.astype(np.float64))).ravel().tolist()

    def run(selected: BackendKind) -> tuple[object, ...]:
        mobility, force = colony_flow_coupling(
            spec, pairs, base=base, drag_coefficient=2.0, max_volume_fraction=0.5,
            backend=selected,
        )
        return (
            colony_volume_fraction(spec, rods, backend=selected).tolist(),
            colony_species_density(spec, rods, species=0, backend=selected),
            mobility,
            force,
            colony_drag(spec, rods, drag_coefficient=5.0, backend=selected),
        )

    device = run(backend)
    assert run(backend) == device, "a device repeats its deposit exactly"
    for actual, expected in zip(device, run(BackendKind.CPU), strict=True):
        expected_array = np.asarray(expected, dtype=np.float64)
        scale = max(float(np.abs(expected_array).max(initial=0.0)), 1e-30)
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64), expected_array, rtol=0, atol=1e-5 * scale
        )


def test_deposition_of_no_cells_is_empty() -> None:
    spec, _ = _device()
    density = colony_volume_fraction(spec, [])
    assert density.shape == (24, 32, 3)
    assert float(density.sum()) == 0.0
    mobility, force = colony_flow_coupling(spec, [], base=1.0)
    assert all(value in (0.0, 1.0) for value in mobility)
    assert all(fx == 0.0 and fy == 0.0 for fx, fy in force)
    assert math.isclose(sum(mobility), float(24 * 32 * 3 - 4 * 4 * 3 - 20 * 3))
