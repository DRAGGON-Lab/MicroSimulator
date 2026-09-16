#!/usr/bin/env python3
"""Quantify nutrient penetration, attached growth, and numerical sensitivity.

A fixed population in a perfused 20 x 40 x 4 channel uses a physical averaging
kernel for both nutrient uptake and resistance. Cells remain attached; this
controlled experiment excludes division, mechanics, and detachment. All units
are micrometers and model time, with nutrient in an arbitrary amount/volume
scale. See docs/tutorials/nutrient-validation.md for equations and scope.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from microsimulator import (
    BackendKind,
    CellInit,
    SignalGridAffineReaction,
    SignalIntegrationKind,
    Simulation,
    Vec3,
)
from microsimulator.biomass import biomass_volume
from microsimulator.flow import colony_mobility, colony_volume_fraction, solve_flow_field
from microsimulator.flow_reference import duct_grid

WIDTH, LENGTH, HEIGHT = 20.0, 40.0, 4.0
DIFFUSION, SPEED = 4.0, 0.3
MU, MONOD_K, YIELD = 0.2, 0.2, 0.5
RADIUS, AVERAGING_RADIUS, DRAG = 0.5, 4.0, 40.0


@dataclass
class Result:
    spacing: float
    dt: float
    refresh: float
    duration: float
    biomass_gain: float
    penetration_half: float
    upstream_growth: float
    downstream_growth: float
    nutrient_amount: float
    boundary_supply: float
    balance_relative_error: float
    y: list[float]
    nutrient_profile: list[float]
    growth_y: list[float]
    growth_rate: list[float]


def run(h: float, dt: float, refresh: float, duration: float, backend: BackendKind) -> Result:
    nx, ny = round(WIDTH / h), round(LENGTH / h)
    spec = duct_grid(nx, ny, 1, (h, h, HEIGHT))
    spec.origin = Vec3(h / 2, h / 2, HEIGHT / 2)
    spec.diffusion, spec.integration = [DIFFUSION], SignalIntegrationKind.BACKWARD_EULER
    spec.y_lower.values, spec.y_upper.values = [1], [0]
    spec.solver.absolute_tolerance = 1e-7
    spec.solver.relative_tolerance = 1e-6
    sim = Simulation(backend)
    field, _ = solve_flow_field(spec, mean_inlet_speed=SPEED, simulation=sim)
    spec.velocity_field = field
    sim.configure_signal_grid(spec, [0] * spec.site_count)
    ids: list[int] = []
    for x in (3, 6.5, 10, 13.5, 17):
        for y in (3, 7, 11, 15, 19, 23, 27, 31, 35):
            cell = CellInit()
            cell.position = Vec3(x, y, HEIGHT / 2)
            cell.length, cell.radius, cell.growth_rate, cell.fixed = 2, RADIUS, 0, True
            ids.append(sim.add_cell(cell))
    cells = sim.cells()
    initial = np.array([biomass_volume(c.length, c.radius) for c in cells], dtype=np.float64)
    # K_i integrates to one. The exact voxel-integrated physical kernel is
    # independent of rod growth and spacing; only its attached amount changes.
    kernels = np.stack(
        [
            colony_volume_fraction(spec, [c], averaging_radius=AVERAGING_RADIUS).ravel() / amount
            for c, amount in zip(cells, initial, strict=True)
        ]
    )
    volume = spec.voxel_volume
    weights = kernels * volume
    assert np.max(np.abs(weights.sum(axis=1) - 1)) < 1e-6
    initial_y = [float(c.position.y) for c in cells]
    boundary_supply = 0.0
    next_refresh = 0.0
    steps = round(duration / dt)
    if not math.isclose(steps * dt, duration, abs_tol=1e-7):
        raise ValueError("duration must be an integer number of steps")
    for step in range(steps):
        time = step * dt
        cells = sim.cells()
        if time + 1e-9 >= next_refresh:
            mobility = colony_mobility(
                spec, cells, drag_coefficient=DRAG, averaging_radius=AVERAGING_RADIUS
            )
            field, _ = solve_flow_field(
                spec, mean_inlet_speed=SPEED, mobility=mobility, simulation=sim
            )
            sim.set_velocity_field(field)
            next_refresh += refresh
        old = np.asarray(sim.signal_levels, dtype=np.float64)
        mean = weights @ old
        cylinder = np.array([math.pi * c.radius**2 * c.length for c in cells])
        coefficient = MU * cylinder / (YIELD * (MONOD_K + mean))
        reaction = SignalGridAffineReaction()
        reaction.source_rates = [0] * spec.site_count
        reaction.loss_rates = (coefficient @ kernels).tolist()
        sim.set_signal_reaction(reaction)
        sim.step(dt)
        new = np.asarray(sim.signal_levels, dtype=np.float64)
        consumed = dt * coefficient * (weights @ new)
        for cid, cell, amount in zip(ids, cells, consumed, strict=True):
            length = cell.length + YIELD * float(amount) / (math.pi * cell.radius**2)
            sim.set_cell_geometry(cid, cell.position, cell.direction, length)
        # Exact discrete BE boundary flux convention used by engine transport:
        # boundary values are ghost-center concentrations, one h from a site.
        concentration = new.reshape(nx, ny)
        y_faces = np.asarray(field.y_faces, dtype=np.float64).reshape(nx, ny + 1)
        influx = DIFFUSION / h * (1 - concentration[:, 0]) + y_faces[:, 0]
        outflux = DIFFUSION / h * concentration[:, -1] + y_faces[:, -1] * concentration[:, -1]
        boundary_supply += dt * h * HEIGHT * float((influx - outflux).sum())
    final_cells = sim.cells()
    final = np.array([biomass_volume(c.length, c.radius) for c in final_cells])
    nutrient = np.asarray(sim.signal_levels, dtype=np.float64).reshape(nx, ny)
    profile = nutrient.mean(axis=0)
    y = (np.arange(ny) + 0.5) * h
    # Boundary sample follows the ghost-center discretization above.
    # First downstream crossing, with no assumption of monotonicity farther on.
    previous_y, previous_c = -h / 2, 1.0
    penetration = LENGTH
    for yy, cc in zip(y, profile, strict=True):
        coordinate, concentration = float(yy), float(cc)
        if concentration <= 0.5:
            penetration = previous_y + (coordinate - previous_y) * (previous_c - 0.5) / (
                previous_c - concentration
            )
            break
        previous_y, previous_c = coordinate, concentration
    growth = (final - initial) / (duration * initial)
    gain = float((final - initial).sum())
    nutrient_amount = float(nutrient.sum()) * volume
    balance = abs(nutrient_amount + gain / YIELD - boundary_supply) / max(boundary_supply, 1e-12)
    return Result(
        h,
        dt,
        refresh,
        duration,
        gain,
        penetration,
        float(growth[np.array(initial_y) <= 11].mean()),
        float(growth[np.array(initial_y) >= 27].mean()),
        nutrient_amount,
        boundary_supply,
        balance,
        y.tolist(),
        profile.tolist(),
        initial_y,
        growth.tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--duration", type=float, default=8)
    args = parser.parse_args()
    backend = {"cpu": BackendKind.CPU, "metal": BackendKind.METAL, "cuda": BackendKind.CUDA}[
        args.backend
    ]
    cases = [
        (2.0, 0.04, 0.4),
        (1.0, 0.04, 0.4),
        (0.5, 0.04, 0.4),
        (1.0, 0.02, 0.4),
        (1.0, 0.04, 0.2),
    ]
    results: list[Result] = []
    for case in cases:
        result = run(*case, args.duration, backend)
        print(
            f"h={result.spacing:g} dt={result.dt:g} refresh={result.refresh:g}: "
            f"gain={result.biomass_gain:.6g}, penetration={result.penetration_half:.6g}, "
            f"balance error={result.balance_relative_error:.3g}",
            flush=True,
        )
        results.append(result)
    base = results[1]
    comparisons: dict[str, dict[str, float]] = {}
    for label, other in [
        ("grid_2_to_1", results[0]),
        ("grid_1_to_half", results[2]),
        ("dt_halved", results[3]),
        ("refresh_halved", results[4]),
    ]:
        comparisons[label] = {
            metric: abs(getattr(other, metric) - getattr(base, metric))
            / max(abs(getattr(base, metric)), 1e-12)
            for metric in ("biomass_gain", "penetration_half", "upstream_growth")
        }
    checks: dict[str, bool] = {
        "mass_balance": max(r.balance_relative_error for r in results) < 2e-4,
        "spatial_growth": base.upstream_growth > 3 * base.downstream_growth,
        "grid_gain": comparisons["grid_1_to_half"]["biomass_gain"] < 0.1,
        "time_gain": comparisons["dt_halved"]["biomass_gain"] < 0.02,
        "refresh_gain": comparisons["refresh_halved"]["biomass_gain"] < 0.02,
    }
    payload = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "backend": args.backend,
        "parameters": {
            "diffusion": DIFFUSION,
            "speed": SPEED,
            "growth_rate": MU,
            "monod_k": MONOD_K,
            "yield": YIELD,
            "averaging_radius": AVERAGING_RADIUS,
            "drag": DRAG,
        },
        "runs": [asdict(r) for r in results],
        "relative_changes": comparisons,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if not all(checks.values()):
        raise SystemExit(f"Nutrient validation failed: {checks}")


if __name__ == "__main__":
    main()
