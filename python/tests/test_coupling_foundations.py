from __future__ import annotations

import math
from pathlib import Path

import pytest
from microsimulator import (
    BackendKind,
    CellInit,
    GridBoundaryKind,
    GridShape,
    MechanicsIntegrationParameters,
    RatePlanBuilder,
    SignalGridAffineReaction,
    SignalGridSpec,
    SignalGridVelocityField,
    SignalIntegrationKind,
    Simulation,
    Vec3,
    backend_available,
    load_checkpoint,
    save_checkpoint,
)


def linear_flow(*, rotation: bool = False) -> SignalGridSpec:
    spec = SignalGridSpec()
    shape = GridShape()
    shape.x, shape.y, shape.z = 9, 9, 1
    spec.shape, spec.origin = shape, Vec3(-4, -4, 0)
    spec.spacing, spec.signal_count = Vec3(1, 1, 1), 1
    spec.diffusion, spec.advection = [0], [Vec3()]
    for boundary in (spec.x_lower, spec.x_upper, spec.y_lower, spec.y_upper):
        boundary.kind, boundary.values = GridBoundaryKind.FIXED, [0]
    field = SignalGridVelocityField()
    field.x_faces = [float((4 - y) if rotation else (y - 4)) for _ in range(10) for y in range(9)]
    field.y_faces = [float(x - 4) if rotation else 0 for x in range(9) for _ in range(10)]
    field.z_faces = [0] * 162
    spec.velocity_field = field
    return spec


@pytest.mark.parametrize("length", [0.0, 2.0, 8.0])
def test_finite_aspect_jeffery_shear(length: float) -> None:
    sim = Simulation()
    sim.configure_signal_grid(linear_flow())
    cell = CellInit()
    cell.position, cell.direction = Vec3(), Vec3(0, 1, 0)
    cell.length, cell.radius = length, 0.5
    cid = sim.add_cell(cell)
    fixed = CellInit()
    fixed.position, fixed.direction, fixed.fixed = Vec3(), Vec3(0, 1, 0), True
    fid = sim.add_cell(fixed)
    sim.apply_flow_drift(2.0)
    aspect = length + 1
    phase = 2 * aspect / (aspect * aspect + 1)
    x, y = aspect * math.sin(phase), math.cos(phase)
    scale = math.hypot(x, y)
    direction = sim.cell(cid).direction
    assert abs(direction.x - x / scale) < 0.002
    assert abs(direction.y - y / scale) < 0.002
    assert sim.cell(fid).direction.y == 1
    assert sim.cell(cid).length == length


def test_rigid_rotation_has_second_order_drift_convergence() -> None:
    errors: list[float] = []
    for n in (4, 8, 16):
        sim = Simulation()
        sim.configure_signal_grid(linear_flow(rotation=True))
        cell = CellInit()
        cell.position, cell.direction = Vec3(1, 0, 0), Vec3(1, 0, 0)
        cid = sim.add_cell(cell)
        parameters = MechanicsIntegrationParameters()
        parameters.max_rotation_radians = 0.5
        for _ in range(n):
            sim.apply_flow_drift(0.8 / n, parameters)
        final = sim.cell(cid)
        errors.append(
            math.hypot(final.position.x - math.cos(0.8), final.position.y - math.sin(0.8))
        )
        assert (
            math.hypot(final.direction.x - math.cos(0.8), final.direction.y - math.sin(0.8)) < 0.002
        )
    assert errors[0] / errors[1] > 3.5
    assert errors[1] / errors[2] > 3.5


def reaction_grid(integration: SignalIntegrationKind) -> SignalGridSpec:
    spec = SignalGridSpec()
    shape = GridShape()
    shape.x = shape.y = shape.z = 1
    spec.shape, spec.signal_count = shape, 1
    spec.diffusion, spec.advection = [0], [Vec3()]
    spec.integration = integration
    return spec


@pytest.mark.parametrize("backend", list(BackendKind))
def test_backward_euler_stiff_reaction_and_restart(backend: BackendKind, tmp_path: Path) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    spec = reaction_grid(SignalIntegrationKind.BACKWARD_EULER)
    reaction = SignalGridAffineReaction()
    reaction.loss_rates, reaction.source_rates = [100], [3]
    spec.reaction = reaction
    sim = Simulation(backend)
    sim.configure_signal_grid(spec, [2])
    sim.step(0.1)
    assert math.isclose(sim.signal_levels[0], 2.3 / 11, rel_tol=2e-6)
    save_checkpoint(sim, tmp_path / "implicit.json")
    restored = load_checkpoint(tmp_path / "implicit.json", backend=backend)
    restored.step(0.1)
    sim.step(0.1)
    assert restored.signal_levels == sim.signal_levels


@pytest.mark.parametrize("backend", list(BackendKind))
@pytest.mark.parametrize("integration", list(SignalIntegrationKind))
def test_rejected_uptake_rolls_back_growth_species_and_time(
    backend: BackendKind, integration: SignalIntegrationKind
) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    sim = Simulation(backend, species_count=1)
    sim.configure_signal_grid(reaction_grid(integration), [0.001])
    cell = CellInit()
    cell.growth_rate, cell.species = 1, [2]
    cid = sim.add_cell(cell)
    rates = RatePlanBuilder()
    sim.set_coupled_rate_plan(
        rates.coupled_plan(1, 1, (rates.constant(1),), (-rates.cell_volume_change_rate(),))
    )
    with pytest.raises((ValueError, RuntimeError)):
        sim.step(1)
    assert sim.time == 0
    assert sim.cell(cid).length == cell.length
    assert sim.cell(cid).species == [2]
    assert math.isclose(sim.signal_levels[0], 0.001, rel_tol=1e-7)
    # Retrying also replaces any failed backend work buffers.
    sim.set_signal_levels([10])
    sim.step(0.01)
    assert sim.time > 0


@pytest.mark.parametrize("backend", list(BackendKind))
def test_backward_euler_diffusion_is_positive_and_conservative(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    spec = reaction_grid(SignalIntegrationKind.BACKWARD_EULER)
    shape = GridShape()
    shape.x, shape.y, shape.z = 2, 1, 1
    spec.shape, spec.diffusion = shape, [1]
    spec.solver.absolute_tolerance = spec.solver.relative_tolerance = 1e-6
    sim = Simulation(backend)
    sim.configure_signal_grid(spec, [1, 0])
    sim.step(10)
    assert math.isclose(sum(sim.signal_levels), 1, abs_tol=3e-6)
    assert math.isclose(sim.signal_levels[0], 11 / 21, abs_tol=3e-6)
    assert math.isclose(sim.signal_levels[1], 10 / 21, abs_tol=3e-6)
