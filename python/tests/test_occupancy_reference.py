from __future__ import annotations

import math

import numpy as np
import pytest
from microsimulator.occupancy_reference import (
    Capsule,
    Face,
    ReservoirFace,
    accessible_volumes,
    backward_euler,
    concentration,
    exchange_weights,
    geometric_porosity,
    porosity_face,
    remap_amounts,
)


def test_empty_grid_matches_existing_native_backward_euler() -> None:
    from microsimulator import GridShape, SignalGridSpec, SignalIntegrationKind, Simulation, Vec3

    shape = GridShape()
    shape.x, shape.y, shape.z = 3, 1, 1
    spec = SignalGridSpec()
    spec.shape, spec.signal_count = shape, 1
    spec.diffusion, spec.advection = [1.0], [Vec3()]
    spec.integration = SignalIntegrationKind.BACKWARD_EULER
    simulation = Simulation()
    simulation.configure_signal_grid(spec, [0.0, 1.0, 0.0])
    simulation.step(0.1)
    result, balance = backward_euler([0, 1, 0], [1, 1, 1], [Face(0, 1, 1), Face(1, 2, 1)], 0.1)
    np.testing.assert_allclose(result, simulation.signal_levels, rtol=2e-6, atol=2e-7)
    assert abs(balance.residual) < 1e-12
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(geometric_porosity(centers, (1, 1, 1), []), [1, 1])


def test_partial_closed_grid_uses_amount_and_unequal_storage() -> None:
    volume = accessible_volumes([0.25, 0.75], 2.0)
    amount = volume * np.array([8.0, 0.0])
    face = porosity_face(0, 1, 0.25, 0.75, diffusion=1, area=1, distance=1)
    initial = amount.sum()
    for _ in range(100):
        amount, ledger = backward_euler(amount, volume, [face], 1.0)
        assert abs(ledger.residual) < 1e-12
    np.testing.assert_allclose(concentration(amount, volume), [2, 2], atol=1e-12)
    assert abs(amount.sum() - initial) < 1e-12
    assert not math.isclose(float(amount[0]), float(amount[1]))


def test_changing_occupancy_and_full_closure_conserve_without_division_by_zero() -> None:
    amount = np.array([2.0, 3.0, 1.0])
    old = np.ones(3)
    new = accessible_volumes([0, 0.25, 0.75], 1)
    result = remap_amounts(amount, old, new, [(0, 1), (1, 2)])
    np.testing.assert_allclose(result, [0, 3.5, 2.5], atol=1e-15)
    assert result.sum() == amount.sum()
    assert np.isfinite(concentration(result, new)).all()
    reopened = remap_amounts(result, new, [1, 1, 1], [(0, 1), (1, 2)])
    assert reopened[0] == 0  # no invented solute in newly exposed storage
    assert reopened.sum() == amount.sum()
    with pytest.raises(ValueError, match="no accessible recipient"):
        remap_amounts(amount, old, [0, 0, 0], [(0, 1), (1, 2)])
    np.testing.assert_array_equal(amount, [2, 3, 1])  # rejection is atomic
    # A persistent wall separates the only potential recipient.
    with pytest.raises(ValueError, match="no accessible recipient"):
        remap_amounts([1, 0, 0], [1, 0, 1], [0, 0, 1], [(0, 1), (1, 2)])
    np.testing.assert_array_equal(accessible_volumes([1e-12, 0], 1), [0, 0])
    np.testing.assert_array_equal(remap_amounts([0], [1], [0], []), [0])


def test_geometry_union_wall_clipping_division_and_removal() -> None:
    centers = np.array(
        [(x, y, z) for x in range(-3, 4) for y in range(-1, 2) for z in range(-1, 2)],
        dtype=np.float64,
    )
    parent = Capsule((0, 0, 0), (1, 0, 0), 4, 0.4)
    daughters = [
        Capsule((-1.2, 0, 0), (1, 0, 0), 1.6, 0.4),
        Capsule((1.2, 0, 0), (1, 0, 0), 1.6, 0.4),
    ]
    old = geometric_porosity(centers, (1, 1, 1), [parent], subdivisions=16)
    np.testing.assert_array_equal(
        old, geometric_porosity(centers, (1, 1, 1), [parent, parent], subdivisions=16)
    )
    divided = geometric_porosity(centers, (1, 1, 1), daughters, subdivisions=16)
    assert divided.sum() > old.sum()  # native division reduces geometric solid volume
    b_parent = math.pi * 0.4**2 * (4 + 0.8)
    b_daughters = 2 * math.pi * 0.4**2 * (1.6 + 0.8)
    assert math.isclose(b_parent, b_daughters)
    assert math.isclose(
        (math.pi * 0.4**2 * 4 + 4 / 3 * math.pi * 0.4**3)
        - 2 * (math.pi * 0.4**2 * 1.6 + 4 / 3 * math.pi * 0.4**3),
        2 / 3 * math.pi * 0.4**3,
    )
    # No voxel closes in this geometry transition; amount remains voxel-local.
    amount = old * 2
    updated = remap_amounts(amount, old, divided, [])
    removed = remap_amounts(updated, divided, np.ones_like(old), [])
    assert abs(removed.sum() - amount.sum()) < 1e-12
    wall_mask = [bool(index == 31) for index in range(len(centers))]
    with_wall = geometric_porosity(centers, (1, 1, 1), [parent], walls=wall_mask)
    assert with_wall[31] == 0
    assert np.all((with_wall >= 0) & (with_wall <= 1))


def test_boundary_reaction_and_cell_exchange_have_explicit_amount_ledgers() -> None:
    volume = np.array([0.25, 0.75])
    weights = exchange_weights([0.5, 0.5], volume)
    c = np.array([2.0, 4.0])
    source = weights * 3.0
    assert float(weights @ c) == 3.5
    assert float(source.sum()) == 3.0
    updated, ledger = backward_euler(
        c * volume,
        volume,
        [Face(0, 1, 0.1, 0.05)],
        0.2,
        source=source,
        loss=[0.3, 0.7],
        reservoirs=[ReservoirFace(0, 5, 0.2, -0.1), ReservoirFace(1, 0, 0, 0.1)],
    )
    assert np.all(updated >= 0)
    assert ledger.boundary > 0 and ledger.reaction < 0 and ledger.source > 0
    assert abs(ledger.residual) < 1e-12
    face = porosity_face(0, 1, 0.25, 0.75, diffusion=2, area=3, distance=4, intrinsic_velocity=5)
    assert face.volume_flux == 0.375 * 3 * 5  # porosity appears exactly once
    with pytest.raises(ValueError, match="accessible exchange"):
        exchange_weights([1, 0], [0, 1])


def test_diffusion_timestep_refinement_and_geometric_quadrature_refinement() -> None:
    # Two-cell antisymmetric diffusion mode has eigenvalue -2 for epsilon=1.
    errors: list[float] = []
    for steps in (10, 20, 40):
        amount = np.array([1.5, 0.5])
        for _ in range(steps):
            amount, _ = backward_euler(amount, [1, 1], [Face(0, 1, 1)], 1 / steps)
        errors.append(abs(float(amount[0]) - (1 + 0.5 * math.exp(-2))))
    assert errors[2] < 0.55 * errors[1] < 0.31 * errors[0]
    sphere = Capsule((0, 0, 0), (1, 0, 0), 0, 0.5)
    exact = 4 / 3 * math.pi * 0.5**3
    geometric_errors: list[float] = []
    for resolution in (8, 16, 32):
        epsilon = geometric_porosity(np.zeros((1, 3)), (2, 2, 2), [sphere], subdivisions=resolution)
        geometric_errors.append(abs(float((1 - epsilon[0]) * 8) - exact))
    assert geometric_errors[-1] < geometric_errors[0]
    assert geometric_errors[-1] < 0.02 * exact


def test_invalid_reference_inputs_fail_without_silent_clipping() -> None:
    with pytest.raises(ValueError):
        accessible_volumes([1.1], 1)
    with pytest.raises(ValueError):
        concentration([1], [0])
    with pytest.raises(ValueError):
        backward_euler([0], [1], [], 1, source=[-1])
    with pytest.raises(ValueError):
        backward_euler([0, 1], [0, 1], [Face(0, 1, 1)], 0.1)


def test_empty_limit_spatial_operator_is_second_order() -> None:
    errors: list[float] = []
    for count in (10, 20, 40):
        h = 1 / count
        centers = (np.arange(count, dtype=np.float64) + 0.5) * h
        values = 2 + np.cos(math.pi * centers)
        rate = np.zeros(count)
        for i in range(count - 1):
            face = porosity_face(i, i + 1, 1, 1, diffusion=1, area=1, distance=h)
            amount_flux = face.conductance * (values[i + 1] - values[i])
            rate[i] += amount_flux / h
            rate[i + 1] -= amount_flux / h
        exact = -(math.pi**2) * np.cos(math.pi * centers)
        errors.append(float(np.max(np.abs(rate - exact))))
    assert errors[2] < 0.26 * errors[1] < 0.07 * errors[0]
