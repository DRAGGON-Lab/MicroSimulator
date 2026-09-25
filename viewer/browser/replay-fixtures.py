"""Generate browser fixtures using the public model, checkpoint and exporter APIs."""

from __future__ import annotations

import argparse
from pathlib import Path

from microsimulator import (
    BackendKind,
    CellInit,
    ChannelMetadata,
    GridShape,
    ModelContext,
    SignalGridSpec,
    Simulation,
    Vec3,
    build_model,
    run_simulation,
    save_checkpoint,
)
from microsimulator.replay import export_replay

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
root = parser.parse_args().output
root.mkdir(parents=True, exist_ok=True)
model, provenance = build_model("examples/replay_demo.py", ModelContext(BackendKind.CPU, 0, 17))
summary = run_simulation(
    model,
    steps=5,
    dt=0.2,
    output=root / "lifecycle.json",
    checkpoint_every=1,
    provenance=provenance,
)
export_replay(summary.periodic_checkpoints, root / "lifecycle")
paths = []
# Data-only compatible scene states exercise unavailable grids and clamping.
# Distinct CPU simulations produce these grid fixtures; no biology claim is made.
for ordinal, size in enumerate((3, None, 1, 3)):
    simulation = Simulation(species_count=2)
    cell = CellInit()
    cell.species = [0.25, 0.75]
    if size is not None:
        shape = GridShape()
        shape.x, shape.y, shape.z = size, size, size
        spec = SignalGridSpec()
        spec.signal_count = 2
        spec.shape = shape
        spec.spacing = Vec3(1, 1, 1)
        spec.diffusion = [0, 0]
        spec.advection = [Vec3(), Vec3()]
        simulation.configure_signal_grid(spec, [0.25] * size**3 + [0.75] * size**3)
    simulation.add_cell(cell)
    simulation.step(ordinal * 0.2)
    path = root / f"grid-{ordinal}.json"
    save_checkpoint(
        simulation,
        path,
        channel_metadata=ChannelMetadata(
            species=("Green", "Red"), signals=("Nutrient", "Cue") if size else ()
        ),
    )
    paths.append(path)
export_replay(paths, root / "grids")
print(root)
