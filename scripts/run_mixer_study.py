#!/usr/bin/env python3
"""Measure how far the liquids of a mixer chip have mixed, by scheme and by grid.

The chip is the design tool's example mixer as `examples/mixer_device.py` builds
it from a design scene. The study carries one liquid, the one the design meters
to half of the joined stream, to a steady state, and reports how unmixed it is
at three sections along the mixing channel: where it has just joined, after the
serpentine's first bend, and at the serpentine's end.

The measure is the intensity of segregation across the channel, weighted by the
flow: the variance of the liquid's share over `mean * (1 - mean)`. It is 1 for
two streams side by side and 0 for a blend.

Each section also has an expected value. The shallow flow carries the stream as
a plug, bends in such a flow turn it without mixing it, and across the channel
only diffusion acts, so a step at mid-width decays as its cosine modes do:

    segregation = sum over odd n of 8 / (pi n)^2 * exp(-2 n^2 pi^2 D t / W^2)

for diffusivity D, channel width W and the time t the stream has spent since it
joined. A grid and a scheme are good enough for a mixing question when they
reproduce that; the table shows which do.

Usage: uv run python scripts/run_mixer_study.py SCENE [--backend cpu|metal|cuda]
       [--spacing 50 25 ...] [--seconds 45]
The scene is the design's, written by the design tool as the example describes.
"""

from __future__ import annotations

import argparse
import math
import runpy
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from microsimulator import (
    BackendKind,
    SignalAdvectionKind,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    Vec3,
    backend_available,
)
from microsimulator.design import load_design_scene

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "mixer_device.py"
# How many voxels the fastest fluid crosses in one step. The limited scheme's
# implicit solve settles while this is about six or fewer.
COURANT = 5.0
CHANNEL_WIDTH = 200.0
# Sections of the mixing channel, as a point of the design inside each and the
# length of channel the stream has run to reach it since the last liquid joined.
SECTIONS = {
    "just joined": ((15100.0, 12100.0), 2100.0),
    "after one bend": ((17000.0, 12700.0), 8500.0),
    "serpentine end": ((18900.0, 15700.0), 32400.0),
}


def expected_segregation(length: float, speed: float, diffusivity: float) -> float:
    """The segregation of a step at mid-width after running a length of channel as a plug."""

    decay = math.pi**2 * diffusivity * (length / speed) / CHANNEL_WIDTH**2
    return sum(
        8.0 / (math.pi * mode) ** 2 * math.exp(-2.0 * mode**2 * decay) for mode in range(1, 60, 2)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("scene", type=Path)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--spacing", type=float, nargs="+", default=[50.0, 25.0])
    parser.add_argument("--seconds", type=float, default=45.0)
    arguments = parser.parse_args()
    backend = BackendKind[arguments.backend.upper()]
    if not backend_available(backend):
        print(f"backend {arguments.backend} is unavailable", file=sys.stderr)
        return 2

    example = runpy.run_path(str(_EXAMPLE))
    scene = load_design_scene(arguments.scene)
    window = example["window_of"](scene)
    chip = example["device"](scene)
    diffusivity = float(example["DIFFUSIVITY"])
    # One signal: the liquid entering the inlet nearest the collector's outlet.
    liquid = len(chip.ports) - 2
    chip = replace(
        chip,
        ports=tuple(
            port
            if port.values is None
            else replace(port, values=(1.0 if index == liquid else 0.0,))
            for index, port in enumerate(chip.ports)
        ),
    )

    print("spacing  scheme    section          voxels  share   segregation  expected")
    for spacing in arguments.spacing:
        for scheme in (SignalAdvectionKind.UPWIND, SignalAdvectionKind.VAN_LEER):
            steps = (spacing, spacing, float(example["DEPTH_SPACING"]))
            origin, shape = chip.lattice(steps)
            grid = SignalGridSpec()
            grid.signal_count = 1
            grid.shape, grid.origin, grid.spacing = shape, origin, Vec3(*steps)
            grid.diffusion = [diffusivity]
            grid.integration = SignalIntegrationKind.BACKWARD_EULER
            grid.advection_scheme = scheme
            grid.solver.absolute_tolerance = 1.0e-5
            simulation = Simulation(backend)
            started = time.perf_counter()
            chip.apply_to_grid(
                grid,
                [0.0],
                [0.0],
                simulation=simulation,
                tolerance=float(example["FLOW_TOLERANCE"]),
            )
            assert grid.velocity_field is not None
            fastest = max(
                float(np.abs(np.asarray(grid.velocity_field.x_faces)).max()),
                float(np.abs(np.asarray(grid.velocity_field.y_faces)).max()),
            )
            simulation.configure_signal_grid(grid, [0.0] * (shape.x * shape.y * shape.z))
            dt = COURANT * spacing / fastest
            for _ in range(math.ceil(arguments.seconds / dt)):
                simulation.step(dt)
            elapsed = time.perf_counter() - started

            layer = shape.z // 2
            dimensions = (shape.x, shape.y, shape.z)
            levels = np.asarray(simulation.signal_levels, dtype=np.float64).reshape(dimensions)
            solid = np.asarray(grid.obstacles).reshape(dimensions)[:, :, layer] != 0
            faces = np.asarray(grid.velocity_field.x_faces).reshape(
                shape.x + 1, shape.y, shape.z
            )[:, :, layer]
            for name, (point, length) in SECTIONS.items():
                local = window.local(*point)
                column = round((local[0] - origin.x) / spacing)
                row = round((local[1] - origin.y) / spacing)
                rows = [row]
                while not solid[column, rows[0] - 1]:
                    rows.insert(0, rows[0] - 1)
                while not solid[column, rows[-1] + 1]:
                    rows.append(rows[-1] + 1)
                held = levels[column, rows, layer]
                weights = np.abs(faces[column, rows])
                weights = weights / weights.sum()
                mean = float((weights * held).sum())
                segregation = float((weights * (held - mean) ** 2).sum() / (mean * (1.0 - mean)))
                speed = float(np.abs(faces[column, rows]).mean())
                print(
                    f"{spacing:7.1f}  {scheme.name.lower():8s}  {name:15s}  {len(rows):6d}  "
                    f"{mean:5.3f}  {segregation:11.3f}  "
                    f"{expected_segregation(length, speed, diffusivity):8.3f}",
                    flush=True,
                )
            print(f"         ({shape.x} x {shape.y} sites, {elapsed:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
