"""Independent contracts for authored perfusion, nutrient growth, and restart."""

from __future__ import annotations

import json
import math
from pathlib import Path

import microsimulator as cm
import pytest
from microsimulator.flow import FluidDomain, FluidProperties, Pressure, VolumeFlow
from microsimulator.growth import (
    BiomassConversion,
    CellGrowth,
    EssentialNutrientGrowth,
    MonodGrowth,
    NutrientRequirement,
)
from microsimulator.media import Concentration, Medium, PiecewiseConstant
from microsimulator.stokes import StokesFlow
from microsimulator.transport import AdvectiveFeed, ConvectiveOutflow, Solute, SoluteTransport


def fluid(flow: float = 1e-6, size: int = 4) -> StokesFlow:
    result = StokesFlow(
        FluidDomain.rectangular_channel(size_um=(size, size, size), spacing_um=1),
        FluidProperties(),
        {"inlet": VolumeFlow(flow), "outlet": Pressure(0)},
    )
    result.geometry.surface_resolution = 32

    return result


def test_recipe_units_immutable_and_unknown_solute() -> None:
    source = {"glucose": Concentration.mM(1)}
    recipe = Medium("feed", source)
    source["glucose"] = Concentration.mM(2)
    assert math.isclose(recipe.concentrations["glucose"].canonical("mol", 1e-6), 1e-18)
    assert math.isclose(Concentration(1, "g/L").canonical("mol", 1e-6, 180), 1000 / 180 * 1e-18)

    with pytest.raises(ValueError, match="molecular weight"):
        Concentration(1, "g/L").canonical("mol", 1e-6)

    sim = cm.Simulation()

    with pytest.raises(ValueError, match="unknown solutes"):
        sim.configure_culture(
            fluid=fluid(),
            transport=SoluteTransport(
                (Solute("other", 1),),
                {"inlet": AdvectiveFeed(recipe), "outlet": ConvectiveOutflow()},
                recipe,
            ),
        )

    assert not sim.has_culture


def test_schedule_split_restart_and_zero_flow(tmp_path: Path) -> None:
    feed = Medium.millimolar("feed", {"tracer": 1})
    empty = Medium.millimolar("empty", {})
    sim = cm.Simulation()
    sim.configure_culture(
        fluid=fluid(),
        transport=SoluteTransport(
            (Solute("tracer", 1),),
            {
                "inlet": AdvectiveFeed(PiecewiseConstant(feed, ((0.5, empty),))),
                "outlet": ConvectiveOutflow(),
            },
            empty,
        ),
    )
    sim.step(0.5)
    assert sim.culture_checkpoint is not None
    assert sim.culture_checkpoint.event_index == 1
    assert sim.culture_checkpoint.configuration.reservoirs[0].concentrations == [0]
    path = tmp_path / "culture.json"
    cm.save_checkpoint(sim, path)
    restarted = cm.load_checkpoint(path)
    sim.step(0.5)
    restarted.step(0.5)
    assert sim.culture_checkpoint is not None and restarted.culture_checkpoint is not None
    assert (
        sim.culture_checkpoint.extracellular_amounts
        == restarted.culture_checkpoint.extracellular_amounts
    )
    no_flow = cm.Simulation()
    no_flow.configure_culture(
        fluid=fluid(0),
        transport=SoluteTransport(
            (Solute("tracer", 100),),
            {"inlet": AdvectiveFeed(feed), "outlet": ConvectiveOutflow()},
            empty,
        ),
    )
    no_flow.step(1)
    assert no_flow.culture_state is not None
    assert no_flow.culture_state.extracellular_amounts["tracer"] == 0
    document = json.loads(path.read_text())
    assert document["version"] == 11


def seeded(
    backend: cm.BackendKind, *, essential: bool = False, empty: bool = False
) -> tuple[cm.Simulation, int]:
    sim = cm.Simulation(backend)
    cell = cm.CellInit()
    cell.position = cm.Vec3(5.5, 5.5, 5.5)
    cell.radius, cell.length, cell.growth_rate, cell.fixed = 2, 1, 0, True
    cid = sim.add_cell(cell)
    medium = Medium.millimolar("initial", {} if empty else {"glucose": 1, "nitrogen": 1})
    model = (
        EssentialNutrientGrowth.molar(
            mu_max_per_hour=0.8,
            requirements={
                "glucose": NutrientRequirement(0.1, 90),
                "nitrogen": NutrientRequirement(0.1, 100),
            },
        )
        if essential
        else MonodGrowth.molar(
            substrate="glucose", mu_max_per_hour=0.8, half_saturation_mM=0.1, yield_g_per_mol=90
        )
    )
    sim.configure_culture(
        fluid=fluid(0, 12),
        transport=SoluteTransport(
            (Solute("glucose", 600), Solute("nitrogen", 1500)),
            {
                "inlet": AdvectiveFeed(medium),
                "outlet": ConvectiveOutflow(backflow=Medium.millimolar("external", {})),
            },
            medium,
        ),
        cell_growth={cid: CellGrowth(model, BiomassConversion(3e-13))},
    )

    return sim, cid


@pytest.mark.parametrize("backend", [cm.BackendKind.CPU, cm.BackendKind.METAL])
@pytest.mark.parametrize("essential", [False, True])
def test_uptake_produces_biomass(backend: cm.BackendKind, essential: bool, tmp_path: Path) -> None:
    if cm.backend_device_count(backend) == 0:
        pytest.skip("native device unavailable")

    sim, cid = seeded(backend, essential=essential)
    before = sim.culture_state
    assert before is not None
    sim.step(0.1)
    after = sim.culture_state
    assert after is not None
    a, b = before.cells[cid], after.cells[cid]
    assert a.dry_biomass_g is not None and b.dry_biomass_g is not None
    gain = b.dry_biomass_g - a.dry_biomass_g
    assert gain > 0
    assert math.isclose(gain, 90 * b.uptake["glucose"], rel_tol=1e-8)

    if essential:
        assert math.isclose(gain, 100 * b.uptake["nitrogen"], rel_tol=1e-8)

    for name in ("glucose", "nitrogen"):
        supplied = sum(t[name] for t in after.boundary_transfer.values())
        error = (
            after.extracellular_amounts[name]
            - before.extracellular_amounts[name]
            - supplied
            + b.uptake[name]
        )
        assert abs(error) < 1e-5 * before.extracellular_amounts[name]

    path = tmp_path / "growth.json"
    cm.save_checkpoint(sim, path)
    restored = cm.load_checkpoint(path).culture_state
    assert restored is not None
    assert restored.cells[cid] == b


def test_no_substrate_produces_no_biomass() -> None:
    sim, cid = seeded(cm.BackendKind.CPU, empty=True)
    before = sim.culture_state
    sim.step(0.1)
    after = sim.culture_state
    assert before is not None and after is not None
    assert after.cells[cid].dry_biomass_g == before.cells[cid].dry_biomass_g
    assert after.cells[cid].uptake["glucose"] == 0
