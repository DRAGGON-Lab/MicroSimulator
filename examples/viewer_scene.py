"""Generate a deterministic colony scene for the standalone viewer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from microsimulator import (
    BoxConstraintInit,
    CellInit,
    ConstraintRegion,
    GridShape,
    SignalGridSpec,
    Simulation,
    Vec3,
    capture_scene,
    save_scene,
)


def build_scene() -> Simulation:
    simulation = Simulation(species_count=2)

    shape = GridShape()
    shape.x = 32
    shape.y = 24
    shape.z = 1
    grid = SignalGridSpec()
    grid.signal_count = 2
    grid.shape = shape
    grid.origin = Vec3(-12.4, -9.2, 0.0)
    grid.spacing = Vec3(0.8, 0.8, 1.0)
    grid.diffusion = [0.0, 0.0]
    grid.advection = [Vec3(), Vec3()]
    first_signal: list[float] = []
    second_signal: list[float] = []
    for x_index in range(shape.x):
        x = grid.origin.x + x_index * grid.spacing.x
        for y_index in range(shape.y):
            y = grid.origin.y + y_index * grid.spacing.y
            first_signal.append(math.exp(-((x + 3.0) ** 2 + (y - 1.5) ** 2) / 28.0))
            second_signal.append(math.exp(-((x - 4.0) ** 2 + (y + 2.0) ** 2) / 20.0))
    simulation.configure_signal_grid(grid, first_signal + second_signal)

    chamber = BoxConstraintInit()
    chamber.center = Vec3(0.0, 0.0, 1.5)
    chamber.half_extents = Vec3(11.0, 8.5, 2.5)
    chamber.allowed_region = ConstraintRegion.INSIDE
    simulation.add_box_constraint(chamber)

    pillar = BoxConstraintInit()
    pillar.center = Vec3(8.2, 5.8, 1.0)
    pillar.half_extents = Vec3(0.8, 0.8, 2.0)
    pillar.allowed_region = ConstraintRegion.OUTSIDE
    simulation.add_box_constraint(pillar)

    for row in range(-4, 5):
        for column in range(-5, 6):
            if (column / 5.5) ** 2 + (row / 4.5) ** 2 > 1.0:
                continue
            x = column * 1.62 + (0.81 if row % 2 else 0.0)
            y = row * 1.43
            angle = 0.16 * x - 0.11 * y
            cell = CellInit()
            cell.position = Vec3(x, y, 0.18 * math.sin(column + row))
            cell.direction = Vec3(math.cos(angle), math.sin(angle), 0.08 * math.sin(row))
            cell.length = 1.7 + 0.5 * (0.5 + 0.5 * math.sin(column * 0.8 - row))
            cell.radius = 0.38
            cell.growth_rate = 0.12 + 0.03 * math.cos(column + row)
            cell.cell_type = (column - row) % 4
            cell.fixed = row == -4 and abs(column) <= 1
            cell.species = [
                0.5 + 0.5 * math.sin(x * 0.32),
                0.5 + 0.5 * math.cos(y * 0.41),
            ]
            simulation.add_cell(cell)
    return simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("viewer-demo.scene.json"),
    )
    arguments = parser.parse_args()
    save_scene(capture_scene(build_scene()), arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()
