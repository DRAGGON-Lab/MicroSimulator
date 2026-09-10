from __future__ import annotations

import math

import pytest
from microsimulator import (
    BackendKind,
    CellInit,
    GridBoundaryKind,
    GridShape,
    RatePlanBuilder,
    SignalGridSpec,
    Simulation,
    Vec3,
    backend_available,
)


@pytest.mark.parametrize("backend", list(BackendKind))
def test_sampling_and_scatter_do_not_bridge_disconnected_corners(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    spec = SignalGridSpec()
    shape = GridShape()
    shape.x = shape.y = 2
    shape.z = 1
    spec.shape, spec.signal_count = shape, 1
    spec.diffusion, spec.advection = [0], [Vec3()]
    spec.obstacles = [0, 1, 1, 0]
    simulation = Simulation(backend)
    simulation.configure_signal_grid(spec, [2, 0, 0, 50])
    position = Vec3(0.4, 0.4, 0)
    assert simulation.sample_signals(position) == [2]
    assert simulation.sample_signals(Vec3(0.6, 0.6, 0)) == [50]
    rates = RatePlanBuilder()
    simulation.set_coupled_rate_plan(rates.coupled_plan(0, 1, (), (rates.constant(1),)))
    cell = CellInit()
    cell.position, cell.growth_rate = position, 0
    simulation.add_cell(cell)
    simulation.step(0.25)
    assert simulation.signal_levels == [2.25, 0, 0, 50]


def _periodic_error(backend: BackendKind, n: int) -> float:
    diffusion, speed, duration = 0.03, 0.2, 0.1
    h = 1 / n
    spec = SignalGridSpec()
    shape = GridShape()
    shape.x, shape.y, shape.z = n, 1, 1
    spec.shape, spec.signal_count = shape, 1
    spec.spacing = Vec3(h, 1, 1)
    spec.diffusion, spec.advection = [diffusion], [Vec3(speed, 0, 0)]
    for name in ["x_lower", "x_upper"]:
        boundary = getattr(spec, name)
        boundary.kind = GridBoundaryKind.PERIODIC
        setattr(spec, name, boundary)
    initial = [1 + 0.25 * math.sin(2 * math.pi * i * h) for i in range(n)]
    simulation = Simulation(backend)
    simulation.configure_signal_grid(spec, initial)
    steps = math.ceil(duration / (0.12 * h * h / diffusion))
    for _ in range(steps):
        simulation.step(duration / steps)
    values = simulation.signal_levels
    assert min(values) >= 0
    assert math.isclose(sum(values) * h, sum(initial) * h, abs_tol=2e-6)
    amplitude = 0.25 * math.exp(-diffusion * (2 * math.pi) ** 2 * duration)
    exact = [1 + amplitude * math.sin(2 * math.pi * (i * h - speed * duration)) for i in range(n)]
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(values, exact, strict=True)) / n)


@pytest.mark.parametrize("backend", list(BackendKind))
def test_periodic_advection_diffusion_converges_and_conserves(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    errors = [_periodic_error(backend, n) for n in (16, 32, 64)]
    assert errors[0] / errors[1] > 1.7
    assert errors[1] / errors[2] > 1.7
