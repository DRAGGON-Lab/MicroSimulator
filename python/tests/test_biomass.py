from __future__ import annotations

import math
from pathlib import Path

import pytest
from microsimulator import (
    BackendKind,
    CellInit,
    GridShape,
    RatePlanBuilder,
    SignalGridSpec,
    Simulation,
    Vec3,
    backend_available,
    load_checkpoint,
    save_checkpoint,
)
from microsimulator.biomass import biomass_volume, capsule_volume


@pytest.mark.parametrize("fraction", [0.2, 0.5, 0.8])
def test_division_preserves_biomass_and_intracellular_amount(fraction: float) -> None:
    simulation = Simulation(species_count=1)
    cell = CellInit()
    cell.length, cell.radius, cell.species = 6, 0.5, [3]
    parent = simulation.add_cell(cell)
    initial = biomass_volume(cell.length, cell.radius)
    daughters = [simulation.cell(i) for i in simulation.divide(parent, fraction)]
    assert math.isclose(
        sum(biomass_volume(c.length, c.radius) for c in daughters), initial, rel_tol=1e-6
    )
    assert math.isclose(
        sum(c.species[0] * biomass_volume(c.length, c.radius) for c in daughters),
        3 * initial,
        rel_tol=1e-6,
    )
    assert sum(capsule_volume(c.length, c.radius) for c in daughters) < capsule_volume(
        cell.length, cell.radius
    )


@pytest.mark.parametrize("backend", list(BackendKind))
def test_realized_growth_consumes_exactly_its_yield_on_every_backend(
    backend: BackendKind, tmp_path: Path
) -> None:
    if not backend_available(backend):
        pytest.skip("backend unavailable")
    simulation = Simulation(backend, species_count=1)
    spec = SignalGridSpec()
    shape = GridShape()
    shape.x = shape.y = shape.z = 1
    spec.shape = shape
    spec.spacing = Vec3(4, 4, 4)
    spec.signal_count, spec.diffusion, spec.advection = 1, [0], [Vec3()]
    simulation.configure_signal_grid(spec, [10])
    rates = RatePlanBuilder()
    simulation.set_coupled_rate_plan(
        rates.coupled_plan(1, 1, (rates.constant(0),), (-rates.cell_volume_change_rate() / 0.4,))
    )
    cell = CellInit()
    cell.length, cell.radius, cell.growth_rate, cell.species = 2, 0.5, 0.7, [3]
    cid = simulation.add_cell(cell)
    initial = biomass_volume(cell.length, cell.radius)
    for dt in [0, 0.03, 0.1, 0.2]:
        simulation.step(dt)
        current = simulation.cell(cid)
        volume = biomass_volume(current.length, current.radius)
        consumed = (10 - simulation.signal_levels[0]) * spec.voxel_volume
        assert math.isclose(0.4 * consumed, volume - initial, abs_tol=3e-5)
        assert math.isclose(current.species[0] * volume, 3 * initial, rel_tol=2e-6)
    save_checkpoint(simulation, tmp_path / "biomass.json")
    restored = load_checkpoint(tmp_path / "biomass.json", backend=backend)
    simulation.step(0.02)
    restored.step(0.02)
    assert restored.signal_levels == simulation.signal_levels
