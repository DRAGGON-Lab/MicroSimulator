from __future__ import annotations

import math

import pytest
from microsimulator import (
    BackendKind,
    CellInit,
    GridShape,
    RatePlanBuilder,
    RatePlanError,
    SignalGridSpec,
    Simulation,
    Vec3,
    backend_available,
)


@pytest.mark.parametrize("backend", list(BackendKind))
def test_symbolic_species_plan_runs_on_every_available_backend(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    builder = RatePlanBuilder()
    x = builder.species(0)
    y = builder.species(1)
    induced = builder.select(builder.equal(builder.cell_type(), 1), 2.0, 0.0)
    plan = builder.species_plan(
        2,
        (
            induced - x,
            3.0 * x * x / (1.0 + x * x + y * y) - y,
        ),
    )
    simulation = Simulation(backend, species_count=2)
    simulation.set_species_rate_plan(plan)
    cell = CellInit()
    cell.growth_rate = 0.0
    cell.cell_type = 1
    cell.species = [1.0, 0.5]
    cell_id = simulation.add_cell(cell)

    simulation.step(0.25)

    result = simulation.cell(cell_id)
    assert math.isclose(result.species[0], 1.25, rel_tol=1.0e-6)
    expected_y = 0.5 + 0.25 * (3.0 / 2.25 - 0.5)
    assert math.isclose(result.species[1], expected_y, rel_tol=1.0e-6)


@pytest.mark.parametrize("backend", list(BackendKind))
def test_symbolic_coupled_plan_uses_geometry_and_signal_sources(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    builder = RatePlanBuilder()
    intracellular = builder.species(0)
    extracellular = builder.signal(0)
    exchange = 0.1 * (extracellular - intracellular) * builder.cell_surface_area()
    plan = builder.coupled_plan(1, 1, (1.0 + exchange / builder.cell_volume(),), (-exchange,))
    simulation = Simulation(backend, species_count=1)
    shape = GridShape()
    shape.x = 1
    shape.y = 1
    shape.z = 1
    spec = SignalGridSpec()
    spec.signal_count = 1
    spec.shape = shape
    spec.diffusion = [0.0]
    spec.advection = [Vec3()]
    simulation.configure_signal_grid(spec, [0.0])
    simulation.set_coupled_rate_plan(plan)
    cell = CellInit()
    cell.growth_rate = 0.0
    cell.species = [2.0]
    cell_id = simulation.add_cell(cell)

    simulation.step(0.1)

    assert simulation.cell(cell_id).species[0] < 2.1
    assert simulation.signal_levels[0] > 0.0


def test_rate_builder_rejects_nonfinite_constants_and_mixed_graphs() -> None:
    first = RatePlanBuilder()
    second = RatePlanBuilder()
    with pytest.raises(RatePlanError, match="finite"):
        first.constant(float("nan"))
    with pytest.raises(RatePlanError, match="different builders"):
        _ = first.species(0) + second.species(0)
    with pytest.raises(TypeError, match="Booleans"):
        bool(first.species(0))
