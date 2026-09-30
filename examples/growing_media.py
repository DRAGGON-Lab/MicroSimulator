"""Perfuse an attached cell and measure nutrient consumption and biomass growth.

Parameters are illustrative, not organism calibration. Geometry and timestep
must be refined before quantitative interpretation. The default run starts
without extracellular nutrients or intracellular nutrient reserves.
"""

# ruff: noqa: N803 -- unit symbols retain the distinction between mM and mm.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import microsimulator as cm
from microsimulator.flow import FluidDomain, FluidProperties, Pressure, VolumeFlow
from microsimulator.growth import (
    BiomassConversion,
    CellGrowth,
    EssentialNutrientGrowth,
    MonodGrowth,
    NutrientRequirement,
)
from microsimulator.media import Medium, PiecewiseConstant
from microsimulator.scene import capture_scene, save_scene
from microsimulator.stokes import StokesFlow
from microsimulator.transport import AdvectiveFeed, ConvectiveOutflow, Solute, SoluteTransport


def build_culture(
    backend: cm.BackendKind = cm.BackendKind.CPU,
    *,
    flow_ul_per_min: float = 1e-6,
    glucose_mM: float = 0.2,
    ammonium_mM: float = 0.5,
    essential: bool = False,
    spacing_um: float = 1.0,
    switch_at_seconds: float | None = None,
) -> cm.Simulation:
    simulation = cm.Simulation(backend)
    founder = cm.CellInit()
    founder.position = cm.Vec3(6, 6, 6)
    founder.radius, founder.length = 2, 1
    founder.fixed, founder.growth_rate = True, 0
    cell_id = simulation.add_cell(founder)
    empty = Medium.millimolar("nutrient-free", {})
    medium = Medium.millimolar(
        "feed",
        {"glucose": glucose_mM, "ammonium": ammonium_mM},
        provenance="Illustrative effective composition; not calibrated.",
    )
    feed = (
        PiecewiseConstant(medium, ((switch_at_seconds, empty),))
        if switch_at_seconds is not None
        else medium
    )
    model = (
        EssentialNutrientGrowth.molar(
            mu_max_per_hour=0.8,
            requirements={
                "glucose": NutrientRequirement(0.1, 90),
                "ammonium": NutrientRequirement(0.02, 100),
            },
        )
        if essential
        else MonodGrowth.molar(
            substrate="glucose", mu_max_per_hour=0.8, half_saturation_mM=0.1, yield_g_per_mol=90
        )
    )
    simulation.configure_culture(
        fluid=StokesFlow(
            domain=FluidDomain.rectangular_channel(size_um=(12, 12, 12), spacing_um=spacing_um),
            properties=FluidProperties(),
            boundaries={"inlet": VolumeFlow(flow_ul_per_min), "outlet": Pressure(0)},
        ),
        transport=SoluteTransport(
            solutes=(Solute("glucose", 600), Solute("ammonium", 1500)),
            initial_medium=empty,
            boundaries={"inlet": AdvectiveFeed(feed), "outlet": ConvectiveOutflow(backflow=empty)},
        ),
        cell_growth={cell_id: CellGrowth(model, BiomassConversion(density_g_per_um3=3e-13))},
    )

    return simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--flow-ul-per-min", type=float, default=1e-6)
    parser.add_argument("--glucose-mm", type=float, default=0.2)
    parser.add_argument("--ammonium-mm", type=float, default=0.5)
    parser.add_argument("--essential", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=1)
    parser.add_argument("--spacing-um", type=float, default=1)
    parser.add_argument("--switch-at-seconds", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sim = build_culture(
        {"cpu": cm.BackendKind.CPU, "metal": cm.BackendKind.METAL, "cuda": cm.BackendKind.CUDA}[
            args.backend
        ],
        flow_ul_per_min=args.flow_ul_per_min,
        glucose_mM=args.glucose_mm,
        ammonium_mM=args.ammonium_mm,
        essential=args.essential,
        spacing_um=args.spacing_um,
        switch_at_seconds=args.switch_at_seconds,
    )

    for _ in range(args.steps):
        sim.step(args.dt)
        state = sim.culture_state
        assert state is not None
        cell = next(iter(state.cells.values()))
        print(
            json.dumps(
                {
                    "time_s": state.time_seconds,
                    "dry_biomass_g": cell.dry_biomass_g,
                    "biomass_produced_g": cell.biomass_produced_g,
                    "growth_per_hour": cell.realized_specific_growth_rate_per_hour,
                    "surface_mM": dict(cell.surface_concentrations_mM),
                    "consumed_mol": dict(cell.uptake),
                    "boundary_mol": {
                        port: dict(values) for port, values in state.boundary_transfer.items()
                    },
                    "budget_residual_mol": dict(state.budget_residual),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        cm.save_checkpoint(sim, args.output / "culture.checkpoint.json")
        save_scene(capture_scene(sim), args.output / "culture.scene.json")


if __name__ == "__main__":
    main()
