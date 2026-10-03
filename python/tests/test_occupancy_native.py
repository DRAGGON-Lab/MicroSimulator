"""The same independent float64 oracle is used for every native GPU device."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from microsimulator import BackendKind, backend_device_count
from microsimulator import occupancy_reference as reference
from microsimulator.occupancy import Capsule, Face, OccupancySolver, ReservoirFace
from numpy.typing import NDArray

DEVICES = [
    (backend, index)
    for backend in (BackendKind.METAL, BackendKind.CUDA)
    for index in range(backend_device_count(backend))
]


@pytest.fixture(scope="module", params=DEVICES or [None])
def solver(request: pytest.FixtureRequest) -> OccupancySolver:
    device: tuple[BackendKind, int] | None = request.param

    if device is None:
        pytest.skip("no native Metal or CUDA device")

    return OccupancySolver(device[0], device_index=device[1])


def assert_parity(actual: NDArray[np.float32], expected: NDArray[np.float64]) -> None:
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)
    assert actual.dtype == np.float32


def test_geometry_empty_overlap_walls_rotation_translation(solver: OccupancySolver) -> None:
    centers = np.array([(x, y, z) for x in range(-2, 3) for y in range(-1, 2) for z in (-1, 0)])
    cells = [Capsule((0.13, -0.17, 0.09), (3, 1, -2), 1.6, 0.43)]
    walls = [i in (4, 12) for i in range(len(centers))]

    for m in (4, 8, 16):
        expected = reference.geometric_porosity(
            centers, (1, 0.8, 0.7), cells, subdivisions=m, walls=walls
        )
        actual = solver.geometric_porosity(
            centers, (1, 0.8, 0.7), cells, subdivisions=m, walls=walls
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            actual,
            solver.geometric_porosity(
                centers, (1, 0.8, 0.7), cells * 2, subdivisions=m, walls=walls
            ),
        )

    np.testing.assert_array_equal(solver.geometric_porosity(centers, (1, 1, 1), []), 1)
    np.testing.assert_array_equal(
        solver.geometric_porosity([[0, 0, 0]], (1, 1, 1), [Capsule((0, 0, 0), (1, 0, 0), 0, 2)]),
        [0],
    )
    assert solver.geometric_porosity(np.empty((0, 3)), (1, 1, 1), []).size == 0


def test_geometry_refinement_and_division(solver: OccupancySolver) -> None:
    centers = np.array([[0, 0, 0]], dtype=np.float64)
    sphere = Capsule((0, 0, 0), (1, 0, 0), 0, 0.4)
    exact = 4 / 3 * math.pi * 0.4**3
    errors: list[float] = []

    for m in (8, 16, 32):
        epsilon = solver.geometric_porosity(centers, (1, 1, 1), [sphere], subdivisions=m)
        errors.append(abs(1 - float(epsilon[0]) - exact))

    assert errors[-1] < errors[0] and errors[-1] / exact < 0.02
    centers = np.array([[x, 0, 0] for x in range(-3, 4)], dtype=np.float64)
    parent = Capsule((0, 0, 0), (1, 0, 0), 4, 0.4)
    daughters = [Capsule((x, 0, 0), (1, 0, 0), 1.6, 0.4) for x in (-1.2, 1.2)]
    old = solver.geometric_porosity(centers, (1, 1, 1), [parent], subdivisions=16)
    divided = solver.geometric_porosity(centers, (1, 1, 1), daughters, subdivisions=16)
    assert divided.sum() > old.sum()
    remapped = solver.remap_amounts(old * 2, old, divided, [])
    removed = solver.remap_amounts(remapped, divided, np.ones_like(divided), [])
    np.testing.assert_array_equal(removed, old * 2)


def test_cutoff_and_face_aperture_once(solver: OccupancySolver) -> None:
    epsilon = np.array([0, 1e-12, 1e-8, 1e-7, 0.25, 0.75, 1], dtype=np.float32)
    expected = np.where(epsilon < np.float32(1e-8), 0, epsilon).astype(np.float64) * 2
    assert_parity(solver.accessible_volumes(epsilon, 2), expected)

    for velocity in (-5.0, 0.0, 5.0):
        expected_face = reference.porosity_face(
            0, 1, 0.25, 0.75, diffusion=2, area=3, distance=4, intrinsic_velocity=velocity
        )
        actual = solver.porosity_face(
            0, 1, 0.25, 0.75, diffusion=2, area=3, distance=4, intrinsic_velocity=velocity
        )
        assert actual == expected_face

    for cutoff in (1e-9, 1e-8, 1e-7):
        configured = OccupancySolver(
            solver.backend, device_index=solver.device_index, epsilon_cutoff=cutoff
        )
        np.testing.assert_array_equal(
            configured.accessible_volumes([cutoff / 2, cutoff * 2], 1), [0, np.float32(cutoff * 2)]
        )
        face = configured.porosity_face(0, 1, cutoff / 2, 1, diffusion=1, area=1, distance=1)
        assert face.conductance == face.volume_flux == 0


def test_remap_components_periodic_edges_reopening_and_rejection(solver: OccupancySolver) -> None:
    amount = np.array([2.0, 3.0, 1.0])
    old = np.ones(3)
    new = np.array([0, 0.25, 0.75])
    neighbors = [(0, 1), (1, 2)]
    result = solver.remap_amounts(amount, old, new, neighbors)
    np.testing.assert_array_equal(result, [0, 3.5, 2.5])
    np.testing.assert_array_equal(solver.remap_amounts(result, new, old, neighbors), result)

    # A persistent wall separates components; an explicit periodic edge joins ends.
    with pytest.raises(ValueError, match="no accessible recipient"):
        solver.remap_amounts([2, 0, 1], [1, 0, 1], [0, 0, 1], neighbors)

    np.testing.assert_array_equal(
        solver.remap_amounts([2, 0, 1], [1, 0, 1], [0, 0, 1], [*neighbors, (2, 0)]), [0, 0, 3]
    )

    with pytest.raises(ValueError, match="no accessible recipient"):
        solver.remap_amounts(amount, old, [0, 0, 0], neighbors)

    np.testing.assert_array_equal(amount, [2, 3, 1])
    np.testing.assert_array_equal(old, 1)
    np.testing.assert_array_equal(solver.remap_amounts([0], [1], [0], []), [0])
    assert solver.remap_amounts([], [], [], []).size == 0


def test_exchange_sampling_scatter_adjoint(solver: OccupancySolver) -> None:
    volume = np.array([0, 0.25, 0.75, 1.5])
    kernel = np.array([1, 0.5, 0.5, 0.2])
    weights = solver.exchange_weights(kernel, volume)
    assert_parity(weights, reference.exchange_weights(kernel, volume.astype(np.float64)))
    concentration = np.array([0, 2, 4, 3])
    rate = 1.25
    assert math.isclose(float(weights.sum()), 1, abs_tol=1e-7)
    assert math.isclose(
        float((weights * rate) @ concentration), float(weights @ concentration) * rate, rel_tol=1e-6
    )
    assert weights[0] == 0

    with pytest.raises(ValueError, match="accessible exchange"):
        solver.exchange_weights([1], [0])


def test_transport_empty_limit_periodic_and_unequal_storage(solver: OccupancySolver) -> None:
    for volume, faces in [
        ([1, 1, 1], [Face(0, 1, 1), Face(1, 2, 1)]),
        ([0.25, 0.75, 1.0], [Face(0, 1, 0.2, 0.1), Face(1, 2, 0.3, 0.1), Face(2, 0, 0.4, 0.1)]),
        ([1, 0, 0.5], [Face(0, 1, 0), Face(1, 2, 0)]),
    ]:
        amount = np.array([1, 0, 0.5])
        result, ledger = solver.backward_euler(amount, volume, faces, 0.1)
        expected, _ = reference.backward_euler(amount, volume, faces, 0.1)
        assert_parity(result, expected)
        assert abs(ledger.residual) <= 5e-6
        np.testing.assert_array_equal(amount, [1, 0, 0.5])

    empty, ledger = solver.backward_euler([], [], [], 1)
    assert empty.size == 0 and ledger.residual == 0
    np.testing.assert_array_equal(solver.backward_euler([0], [0], [], 1)[0], [0])


def test_closed_1000_step_drift(solver: OccupancySolver) -> None:
    volume = np.array([0.5, 1.5])
    face = solver.porosity_face(0, 1, 0.25, 0.75, diffusion=1, area=1, distance=1)
    amount = np.array([4, 0], dtype=np.float32)

    for _ in range(1000):
        amount, ledger = solver.backward_euler(amount, volume, [face], 1)
        assert abs(ledger.residual) / 4 <= 5e-6

    assert abs(float(amount.sum(dtype=np.float64)) - 4) / 4 <= 5e-5
    assert_parity(solver.concentration(amount, volume), np.array([2, 2]))


def test_boundary_reaction_exchange_and_signed_flux(solver: OccupancySolver) -> None:
    for flux in (-0.05, 0.05):
        volume = [0.25, 0.75]
        faces = [Face(0, 1, 0.1, flux)]
        reservoirs = [ReservoirFace(0, 5, 0.2, -0.1), ReservoirFace(1, 0, 0, 0.1)]
        source = [0.75, 2.25]
        loss = [0.3, 0.7]
        actual, ledger = solver.backward_euler(
            [0.5, 3], volume, faces, 0.2, source=source, loss=loss, reservoirs=reservoirs
        )
        expected, balance = reference.backward_euler(
            [0.5, 3], volume, faces, 0.2, source=source, loss=loss, reservoirs=reservoirs
        )
        assert_parity(actual, expected)
        assert abs(ledger.residual) < 5e-6

        for name in ("before", "after", "source", "reaction", "boundary"):
            assert math.isclose(
                getattr(ledger, name), getattr(balance, name), rel_tol=2e-4, abs_tol=2e-6
            )

    # Prescribed divergent intrinsic flow conserves amount but need not preserve c.
    actual, ledger = solver.backward_euler([1, 1], [1, 1], [Face(0, 1, 0, 0.5)], 0.2)
    assert actual[0] != actual[1] and abs(ledger.residual) < 5e-6


def test_timestep_and_spatial_refinement(solver: OccupancySolver) -> None:
    errors: list[float] = []

    for steps in (10, 20, 40):
        amount = np.array([1, 0], dtype=np.float32)

        for _ in range(steps):
            amount, _ = solver.backward_euler(amount, [1, 1], [Face(0, 1, 1)], 1 / steps)

        exact = np.array([0.5 * (1 + math.exp(-2)), 0.5 * (1 - math.exp(-2))])
        errors.append(float(np.max(np.abs(amount - exact))))

    assert errors[2] < 0.55 * errors[1] < 0.31 * errors[0]
    # Diffusion eigenmode on a fixed physical no-flux interval, fixed dt.
    # At h=1/40, even the float64 solution rounded to float32 has an amount
    # equation residual near 1.1e-6. Use an explicit 2e-6 solve tolerance while
    # keeping the independent 5e-6 ledger and spatial convergence gates.
    errors = []

    for count in (10, 20, 40):
        h = 1 / count
        x = (np.arange(count) + 0.5) * h
        initial = (1 + 0.5 * np.cos(math.pi * x)) * h
        faces = [Face(i, i + 1, 1 / h) for i in range(count - 1)]
        amount, _ = solver.backward_euler(
            initial, np.full(count, h), faces, 0.01, relative_tolerance=2e-6
        )
        exact = 1 + 0.5 * np.cos(math.pi * x) / (1 + 0.01 * math.pi**2)
        errors.append(float(np.max(np.abs(amount / h - exact))))

    assert errors[2] < 0.32 * errors[1] < 0.1 * errors[0]


def test_invalid_inputs_and_failed_candidates_are_atomic(solver: OccupancySolver) -> None:
    invalid: list[Callable[[], object]] = [
        lambda: solver.concentration([1], [0]),
        lambda: solver.concentration([-1], [1]),
        lambda: solver.accessible_volumes([1.1], 1),
        lambda: solver.accessible_volumes([1], -1),
        lambda: solver.remap_amounts([1], [1], [1], [(0, 1)]),
        lambda: solver.geometric_porosity([[0, 0, 0]], (1, 1, 1), [], subdivisions=0),
        lambda: solver.geometric_porosity([[0, 0, 0]], (1, 1, 1), [], subdivisions=True),
        lambda: solver.backward_euler([1, 0], [1, 0], [Face(0, 1, 1)], 1),
        lambda: solver.backward_euler([1], [1], [], 1, source=[]),
        lambda: solver.backward_euler([0], [0], [], 1, source=[1]),
        lambda: solver.backward_euler([1], [1], [], 1, loss=[-1]),
        lambda: solver.backward_euler([1], [1], [], -1),
        lambda: solver.backward_euler([1], [1], [], 1, source=[-2]),
        lambda: solver.backward_euler([float("nan")], [1], [], 1),
        lambda: solver.backward_euler([1e-60], [1], [], 1),
    ]

    for operation in invalid:
        with pytest.raises((ValueError, TypeError)):
            operation()

    amount = np.array([1.0, 0.0])
    previous_report = solver.last_report

    with pytest.raises(RuntimeError, match="did not converge"):
        solver.backward_euler(amount, [1, 1], [Face(0, 1, 1)], 100, max_iterations=1)

    np.testing.assert_array_equal(amount, [1, 0])
    assert solver.last_report is previous_report
    np.testing.assert_array_equal(solver.backward_euler(amount, [1, 1], [], 0)[0], amount)


def test_no_implicit_cpu_fallback() -> None:
    with pytest.raises(ValueError, match="Metal or CUDA"):
        OccupancySolver(BackendKind.CPU)

    with pytest.raises(ValueError):
        OccupancySolver("auto")

    for backend in (BackendKind.METAL, BackendKind.CUDA):
        with pytest.raises((RuntimeError, IndexError)):
            OccupancySolver(backend, device_index=backend_device_count(backend))


def test_multiple_components_and_cross_block_transport(solver: OccupancySolver) -> None:
    # Cross both CUDA's 128-thread block and Metal's 64-thread dispatch width.
    count = 257
    rng = np.random.default_rng(31)
    old = rng.uniform(0.2, 1, count)
    new = rng.uniform(0.2, 1, count)
    amount = rng.uniform(0, 2, count)
    old[128] = new[128] = amount[128] = 0
    new[::7] = 0
    neighbors = [(i, i + 1) for i in range(count - 1)]
    expected = reference.remap_amounts(amount, old, new, neighbors)
    actual = solver.remap_amounts(amount, old, new, neighbors)
    assert_parity(actual, expected)
    np.testing.assert_array_equal(actual, solver.remap_amounts(amount, old, new, neighbors))
    faces = [
        Face(i, j, 0.1, float(rng.uniform(-0.03, 0.03)))
        for i, j in neighbors
        if new[i] > 0 and new[j] > 0
    ]
    sources = np.where(new > 0, 0.02, 0)
    losses = rng.uniform(0, 0.1, count)
    updated, balance = solver.backward_euler(actual, new, faces, 0.03, source=sources, loss=losses)
    oracle, _ = reference.backward_euler(
        actual.astype(np.float64), new, faces, 0.03, source=sources, loss=losses
    )
    assert_parity(updated, oracle)
    assert abs(balance.residual) / balance.before < 5e-6


def test_fixed_physical_geometry_and_exchange_support_refinement(solver: OccupancySolver) -> None:
    sphere = Capsule((0.03, -0.05, 0.04), (1, 0, 0), 0, 0.37)
    exact = 4 / 3 * math.pi * sphere.radius**3
    errors: list[float] = []

    for count in (4, 8, 16):
        h = 2 / count
        axis = -1 + (np.arange(count) + 0.5) * h
        centers = (
            np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
            .reshape(-1, 3)
            .astype(np.float64)
        )
        epsilon = solver.geometric_porosity(centers, (h, h, h), [sphere], subdivisions=8)
        oracle = reference.geometric_porosity(centers, (h, h, h), [sphere], subdivisions=8)
        assert_parity(epsilon, oracle)
        errors.append(abs(float((1 - epsilon).sum(dtype=np.float64)) * h**3 - exact))
        volume = solver.accessible_volumes(epsilon, h**3)
        # Same physical Gaussian support at every grid resolution.
        distance_squared = np.sum((centers - np.array(sphere.center)) ** 2, axis=1)
        kernel = np.exp(-distance_squared / (2 * 0.5**2))
        weights = solver.exchange_weights(kernel, volume)
        assert_parity(weights, reference.exchange_weights(kernel, volume.astype(np.float64)))
        assert abs(float(weights.sum(dtype=np.float64)) - 1) < 1e-6

    assert errors[-1] < errors[0] and errors[-1] / exact < 0.01
