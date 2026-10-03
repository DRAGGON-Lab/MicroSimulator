#!/usr/bin/env python3
"""Run machine-readable checks of MicroSimulator's biological numerics.

The CPU backend is the scientific reference. Other backends run the same
checks as conformance tests; agreement with CPU does not establish physical
validity or experimental calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from microsimulator import (
    BackendKind,
    CellInit,
    GridBoundary,
    GridBoundaryKind,
    GridShape,
    MechanicsParameters,
    RatePlanBuilder,
    SignalGridSpec,
    SignalGridVelocityField,
    Simulation,
    Vec3,
    backend_available,
)
from microsimulator.biomass import biomass_volume


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    category: str
    passed: bool
    seconds: float
    metrics: dict[str, float | int | bool | str]
    error: str | None = None


def _result(
    name: str,
    category: str,
    check: Callable[[], tuple[bool, dict[str, float | int | bool | str]]],
) -> BenchmarkResult:
    started = time.perf_counter()
    try:
        passed, metrics = check()
        return BenchmarkResult(name, category, passed, time.perf_counter() - started, metrics)
    except Exception as exc:  # A benchmark runner must preserve all failure evidence.
        return BenchmarkResult(
            name,
            category,
            False,
            time.perf_counter() - started,
            {},
            f"{type(exc).__name__}: {exc}",
        )


def _simulation(backend: BackendKind) -> Simulation:
    return Simulation(backend)


def growth_recurrence(backend: BackendKind) -> tuple[bool, dict[str, float]]:
    mu, duration = 0.7, 1.2
    errors: list[float] = []
    for dt in (0.2, 0.1, 0.05):
        sim = _simulation(backend)
        cell = CellInit()
        cell.length, cell.growth_rate = 2.0, mu
        cid = sim.add_cell(cell)
        for _ in range(round(duration / dt)):
            sim.step(dt)
        recurrence = 2.0 * (1.0 + mu * dt) ** round(duration / dt)
        actual = sim.cell(cid).length
        if not math.isclose(actual, recurrence, rel_tol=2e-6):
            return False, {"actual": actual, "recurrence": recurrence}
        errors.append(abs(actual - 2.0 * math.exp(mu * duration)))
    return errors[0] > errors[1] > errors[2], {
        "dt_0.2_error_vs_exponential": errors[0],
        "dt_0.1_error_vs_exponential": errors[1],
        "dt_0.05_error_vs_exponential": errors[2],
        "observed_order_fine": math.log2(errors[1] / errors[2]),
    }


def biomass_and_dilution(backend: BackendKind) -> tuple[bool, dict[str, float]]:
    sim = Simulation(backend, species_count=1)
    cell = CellInit()
    cell.length, cell.radius, cell.growth_rate, cell.species = 2.0, 0.5, 0.4, [3.0]
    cid = sim.add_cell(cell)
    initial_volume = biomass_volume(cell.length, cell.radius)
    initial_amount = initial_volume * cell.species[0]
    sim.step(0.25)
    final = sim.cell(cid)
    final_volume = biomass_volume(final.length, final.radius)
    final_amount = final_volume * final.species[0]
    relative_error = abs(final_amount - initial_amount) / initial_amount
    expected_volume = math.pi * cell.radius**2 * (final.length + 2 * cell.radius)
    return relative_error < 2e-6 and math.isclose(final_volume, expected_volume), {
        "initial_volume": initial_volume,
        "final_volume": final_volume,
        "intracellular_amount_relative_error": relative_error,
    }


def division_conservation(backend: BackendKind) -> tuple[bool, dict[str, float | int | bool]]:
    sim = Simulation(backend, species_count=1)
    cell = CellInit()
    cell.position, cell.direction = Vec3(2, 3, 0), Vec3(1, 0, 0)
    cell.length, cell.radius, cell.species = 6.0, 0.5, [2.5]
    parent = sim.add_cell(cell)
    initial_volume = biomass_volume(cell.length, cell.radius)
    daughters = sim.divide(parent, 0.3)
    snapshots = [sim.cell(cid) for cid in daughters]
    final_volume = sum(biomass_volume(c.length, c.radius) for c in snapshots)
    final_amount = sum(c.species[0] * biomass_volume(c.length, c.radius) for c in snapshots)
    lineage = all(sim.lineage_parent(cid) == parent for cid in daughters)
    gap = (
        snapshots[1].position.x
        - snapshots[1].length / 2
        - (snapshots[0].position.x + snapshots[0].length / 2)
    )
    volume_error = abs(final_volume - initial_volume) / initial_volume
    amount_error = abs(final_amount - 2.5 * initial_volume) / (2.5 * initial_volume)
    return volume_error < 2e-6 and amount_error < 2e-6 and lineage, {
        "biomass_relative_error": volume_error,
        "species_amount_relative_error": amount_error,
        "daughter_surface_gap": gap,
        "expected_surface_gap": 2 * cell.radius,
        "lineage_preserved": lineage,
    }


def contacts_and_mechanics(backend: BackendKind) -> tuple[bool, dict[str, float | bool | int]]:
    sim = _simulation(backend)
    ids: list[int] = []
    for y, fixed in ((0.0, False), (0.8, False), (3.0, True)):
        cell = CellInit()
        cell.position, cell.length, cell.radius, cell.fixed = Vec3(0, y, 0), 4.0, 0.5, fixed
        ids.append(sim.add_cell(cell))
    graph = sim.find_cell_contacts()
    pair = [c for c in graph.contacts if {c.first_id, c.second_id} == set(ids[:2])]
    initial_com = sum(sim.cell(cid).position.y for cid in ids[:2]) / 2
    initial_overlap = max(0.0, -min(c.signed_separation for c in pair))
    fixed_before = sim.cell(ids[2]).position.y
    report = sim.relax_cell_mechanics()
    final_graph = sim.find_cell_contacts()
    final_pair = [c for c in final_graph.contacts if {c.first_id, c.second_id} == set(ids[:2])]
    final_overlap = max(0.0, -min((c.signed_separation for c in final_pair), default=0.0))
    final_com = sum(sim.cell(cid).position.y for cid in ids[:2]) / 2
    fixed_unchanged = sim.cell(ids[2]).position.y == fixed_before
    symmetric = len(pair) == 2 and math.isclose(
        pair[0].signed_separation, pair[1].signed_separation
    )
    return (
        symmetric
        and fixed_unchanged
        and abs(final_com - initial_com) < 1e-6
        and final_overlap < initial_overlap
    ), {
        "directed_contacts": len(pair),
        "contact_symmetric": symmetric,
        "fixed_cell_unchanged": fixed_unchanged,
        "center_of_mass_drift": abs(final_com - initial_com),
        "initial_overlap": initial_overlap,
        "final_overlap": final_overlap,
        "solver_iterations": report.report.iterations,
    }


def _flow_grid(*, shear: bool = False) -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 9, 9, 1
    spec = SignalGridSpec()
    spec.shape, spec.origin = shape, Vec3(-4, -4, 0)
    spec.spacing, spec.signal_count = Vec3(1, 1, 1), 1
    spec.diffusion, spec.advection = [0], [Vec3()]
    for name in ("x_lower", "x_upper", "y_lower", "y_upper"):
        boundary = getattr(spec, name)
        boundary.kind, boundary.values = GridBoundaryKind.FIXED, [0]
        setattr(spec, name, boundary)
    field = SignalGridVelocityField()
    field.x_faces = [float(y - 4) if shear else 1.25 for _ in range(10) for y in range(9)]
    field.y_faces = [0.0] * 90
    field.z_faces = [0.0] * 162
    spec.velocity_field = field
    return spec


def flow_translation_and_rotation(
    backend: BackendKind,
) -> tuple[bool, dict[str, float]]:
    uniform = _simulation(backend)
    uniform.configure_signal_grid(_flow_grid())
    point = CellInit()
    point.position, point.length, point.radius = Vec3(), 0.0, 0.5
    pid = uniform.add_cell(point)
    uniform.apply_flow_drift(0.4)
    translation_error = abs(uniform.cell(pid).position.x - 0.5)

    shear = _simulation(backend)
    shear.configure_signal_grid(_flow_grid(shear=True))
    rod = CellInit()
    rod.position, rod.direction = Vec3(), Vec3(0, 1, 0)
    rod.length, rod.radius = 2.0, 0.5
    rid = shear.add_cell(rod)
    duration = 0.5
    shear.apply_flow_drift(duration)
    aspect = (rod.length + 2 * rod.radius) / (2 * rod.radius)
    phase = duration * aspect / (aspect * aspect + 1)
    expected_x, expected_y = aspect * math.sin(phase), math.cos(phase)
    scale = math.hypot(expected_x, expected_y)
    actual = shear.cell(rid).direction
    rotation_error = math.hypot(actual.x - expected_x / scale, actual.y - expected_y / scale)
    return translation_error < 2e-6 and rotation_error < 0.002, {
        "uniform_translation_error": translation_error,
        "jeffery_direction_error": rotation_error,
        "aspect_ratio": aspect,
    }


def transport(backend: BackendKind) -> tuple[bool, dict[str, float | bool]]:
    shape = GridShape()
    shape.x, shape.y, shape.z = 32, 1, 1
    spec = SignalGridSpec()
    spec.shape, spec.signal_count, spec.spacing = shape, 1, Vec3(1 / 32, 1, 1)
    spec.diffusion, spec.advection = [0.03], [Vec3(0.2, 0, 0)]
    for name in ("x_lower", "x_upper"):
        boundary = GridBoundary()
        boundary.kind = GridBoundaryKind.PERIODIC
        setattr(spec, name, boundary)
    initial = [1 + 0.25 * math.sin(2 * math.pi * i / 32) for i in range(32)]
    sim = _simulation(backend)
    sim.configure_signal_grid(spec, initial)
    initial_mass = sum(initial) / 32
    duration, dt = 0.1, 0.0025
    for _ in range(round(duration / dt)):
        sim.step(dt)
    values = sim.signal_levels
    mass_error = abs(sum(values) / 32 - initial_mass)
    initial_amplitude = (max(initial) - min(initial)) / 2
    final_amplitude = (max(values) - min(values)) / 2

    fixed = SignalGridSpec()
    fixed_shape = GridShape()
    fixed_shape.x, fixed_shape.y, fixed_shape.z = 2, 1, 1
    fixed.shape, fixed.signal_count = fixed_shape, 1
    fixed.diffusion, fixed.advection = [1], [Vec3()]
    fixed.x_lower.kind, fixed.x_lower.values = GridBoundaryKind.FIXED, [2]
    boundary_sim = _simulation(backend)
    boundary_sim.configure_signal_grid(fixed, [0, 0])
    boundary_sim.step(0.25)
    boundary_error = abs(boundary_sim.signal_levels[0] - 0.5)
    return mass_error < 3e-6 and min(
        values
    ) >= 0 and final_amplitude < initial_amplitude and boundary_error < 2e-6, {
        "periodic_mass_error": mass_error,
        "minimum_concentration": min(values),
        "amplitude_ratio": final_amplitude / initial_amplitude,
        "fixed_boundary_update_error": boundary_error,
    }


def membrane_exchange(backend: BackendKind) -> tuple[bool, dict[str, float]]:
    shape = GridShape()
    shape.x = shape.y = shape.z = 1
    spec = SignalGridSpec()
    spec.shape, spec.signal_count, spec.spacing = shape, 1, Vec3(4, 4, 4)
    spec.diffusion, spec.advection = [0], [Vec3()]
    sim = Simulation(backend, species_count=1)
    sim.configure_signal_grid(spec, [2.0])
    cell = CellInit()
    cell.length, cell.radius, cell.growth_rate, cell.species = 2.0, 0.5, 0.0, [0.5]
    cid = sim.add_cell(cell)
    rates = RatePlanBuilder()
    flux = rates.constant(0.4) * (rates.signal(0) - rates.species(0))
    sim.set_coupled_rate_plan(rates.coupled_plan(1, 1, (flux,), (-flux * rates.cell_volume(),)))
    volume = biomass_volume(cell.length, cell.radius)
    voxel = spec.voxel_volume
    before = cell.species[0] * volume + 2.0 * voxel
    difference_before = 2.0 - cell.species[0]
    sim.step(0.1)
    after_cell = sim.cell(cid)
    after = after_cell.species[0] * volume + sim.signal_levels[0] * voxel
    difference_after = sim.signal_levels[0] - after_cell.species[0]

    equilibrium = Simulation(backend, species_count=1)
    equilibrium.configure_signal_grid(spec, [1.25])
    equal_cell = CellInit()
    equal_cell.length, equal_cell.radius, equal_cell.growth_rate = 2.0, 0.5, 0.0
    equal_cell.species = [1.25]
    equal_id = equilibrium.add_cell(equal_cell)
    equilibrium.set_coupled_rate_plan(
        rates.coupled_plan(1, 1, (flux,), (-flux * rates.cell_volume(),))
    )
    equilibrium.step(0.1)
    equilibrium_error = max(
        abs(equilibrium.signal_levels[0] - 1.25),
        abs(equilibrium.cell(equal_id).species[0] - 1.25),
    )
    conservation_error = abs(after - before) / before
    return conservation_error < 2e-6 and abs(difference_after) < abs(
        difference_before
    ) and equilibrium_error < 2e-6, {
        "amount_relative_error": conservation_error,
        "concentration_gap_ratio": abs(difference_after / difference_before),
        "equilibrium_error": equilibrium_error,
    }


def monod_and_yield(backend: BackendKind) -> tuple[bool, dict[str, float]]:
    mu_max, half_saturation, yield_coefficient = 0.8, 0.2, 0.4
    low = mu_max * 1e-9 / (half_saturation + 1e-9)
    at_half = mu_max * half_saturation / (half_saturation + half_saturation)
    high = mu_max * 1e9 / (half_saturation + 1e9)

    shape = GridShape()
    shape.x = shape.y = shape.z = 1
    spec = SignalGridSpec()
    spec.shape, spec.signal_count, spec.spacing = shape, 1, Vec3(4, 4, 4)
    spec.diffusion, spec.advection = [0], [Vec3()]
    sim = _simulation(backend)
    sim.configure_signal_grid(spec, [10])
    rates = RatePlanBuilder()
    sim.set_coupled_rate_plan(
        rates.coupled_plan(
            0,
            1,
            (),
            (-rates.cell_volume_change_rate() / yield_coefficient,),
        )
    )
    cell = CellInit()
    cell.length, cell.radius, cell.growth_rate = 2, 0.5, 0.7
    cid = sim.add_cell(cell)
    initial_volume = biomass_volume(cell.length, cell.radius)
    sim.step(0.1)
    biomass_gain = biomass_volume(sim.cell(cid).length, cell.radius) - initial_volume
    consumed = (10 - sim.signal_levels[0]) * spec.voxel_volume
    yield_error = abs(yield_coefficient * consumed - biomass_gain)
    return low < 1e-7 and math.isclose(
        at_half, mu_max / 2
    ) and high > 0.999999 * mu_max and yield_error < 3e-5, {
        "monod_zero_limit": low,
        "monod_half_saturation_rate": at_half,
        "monod_high_limit": high,
        "yield_absolute_error": yield_error,
    }


def mechanics_sensitivity(backend: BackendKind) -> tuple[bool, dict[str, float]]:
    overlaps: dict[str, float] = {}
    for gamma in (2.0, 10.0, 50.0):
        sim = _simulation(backend)
        for y in (0.0, 0.75, 1.5):
            cell = CellInit()
            cell.position, cell.length, cell.radius = Vec3(0, y, 0), 3.0, 0.5
            sim.add_cell(cell)
        parameters = MechanicsParameters()
        parameters.gamma = gamma
        for _ in range(3):
            sim.relax_cell_mechanics(parameters)
        contacts = sim.find_cell_contacts().contacts
        overlaps[f"gamma_{gamma:g}"] = max(
            (max(0.0, -contact.signed_separation) for contact in contacts), default=0.0
        )
    finite = all(math.isfinite(value) for value in overlaps.values())
    improved = max(overlaps.values()) < 0.25

    pass_sim = _simulation(backend)
    for y in (0.0, 0.7, 1.4, 2.1):
        cell = CellInit()
        cell.position, cell.length, cell.radius = Vec3(0, y, 0), 3.0, 0.5
        pass_sim.add_cell(cell)
    pass_sim.relax_cell_mechanics()
    after_one = max(
        (
            max(0.0, -contact.signed_separation)
            for contact in pass_sim.find_cell_contacts().contacts
        ),
        default=0.0,
    )
    for _ in range(3):
        pass_sim.relax_cell_mechanics()
    after_four = max(
        (
            max(0.0, -contact.signed_separation)
            for contact in pass_sim.find_cell_contacts().contacts
        ),
        default=0.0,
    )
    overlaps["dense_overlap_after_one_pass"] = after_one
    overlaps["dense_overlap_after_four_passes"] = after_four

    separations: list[float] = []
    for dt in (0.1, 0.05):
        sim = _simulation(backend)
        ids: list[int] = []
        for y in (0.0, 0.9):
            cell = CellInit()
            cell.position, cell.length, cell.radius = Vec3(0, y, 0), 2.0, 0.5
            cell.growth_rate = 0.5
            ids.append(sim.add_cell(cell))
        for _ in range(round(0.2 / dt)):
            sim.step(dt)
            sim.relax_cell_mechanics()
        separations.append(abs(sim.cell(ids[1]).position.y - sim.cell(ids[0]).position.y))
    overlaps["split_dt_separation_difference"] = abs(separations[0] - separations[1])
    return finite and improved and after_four < after_one, overlaps


def exchange_refinement(backend: BackendKind) -> tuple[bool, dict[str, float]]:
    cell_volume = biomass_volume(2.0, 0.5)
    voxel_volume = 64.0
    rate, duration = 0.4, 0.5
    initial_gap = 1.5
    exact_gap = initial_gap * math.exp(-rate * (1 + cell_volume / voxel_volume) * duration)
    errors: list[float] = []
    balance_errors: list[float] = []
    for dt in (0.1, 0.05, 0.025):
        shape = GridShape()
        shape.x = shape.y = shape.z = 1
        spec = SignalGridSpec()
        spec.shape, spec.signal_count, spec.spacing = shape, 1, Vec3(4, 4, 4)
        spec.diffusion, spec.advection = [0], [Vec3()]
        sim = Simulation(backend, species_count=1)
        sim.configure_signal_grid(spec, [2.0])
        cell = CellInit()
        cell.length, cell.radius, cell.growth_rate, cell.species = 2.0, 0.5, 0.0, [0.5]
        cid = sim.add_cell(cell)
        rates = RatePlanBuilder()
        flux = rates.constant(rate) * (rates.signal(0) - rates.species(0))
        sim.set_coupled_rate_plan(rates.coupled_plan(1, 1, (flux,), (-flux * rates.cell_volume(),)))
        initial_amount = 0.5 * cell_volume + 2.0 * voxel_volume
        for _ in range(round(duration / dt)):
            sim.step(dt)
        gap = sim.signal_levels[0] - sim.cell(cid).species[0]
        errors.append(abs(gap - exact_gap))
        final_amount = sim.cell(cid).species[0] * cell_volume + sim.signal_levels[0] * voxel_volume
        balance_errors.append(abs(final_amount - initial_amount) / initial_amount)
    order_fine = math.log2(errors[1] / errors[2])
    return errors[0] > errors[1] > errors[2] and order_fine > 0.9 and max(balance_errors) < 2e-6, {
        "gap_error_dt_0.1": errors[0],
        "gap_error_dt_0.05": errors[1],
        "gap_error_dt_0.025": errors[2],
        "observed_order_fine": order_fine,
        "maximum_amount_relative_error": max(balance_errors),
    }


def transport_refinement(backend: BackendKind) -> tuple[bool, dict[str, float]]:
    errors: list[float] = []
    diffusion, speed, duration = 0.03, 0.2, 0.1
    for n in (16, 32, 64):
        shape = GridShape()
        shape.x, shape.y, shape.z = n, 1, 1
        spec = SignalGridSpec()
        spec.shape, spec.signal_count, spec.spacing = shape, 1, Vec3(1 / n, 1, 1)
        spec.diffusion, spec.advection = [diffusion], [Vec3(speed, 0, 0)]
        for name in ("x_lower", "x_upper"):
            boundary = GridBoundary()
            boundary.kind = GridBoundaryKind.PERIODIC
            setattr(spec, name, boundary)
        initial = [1 + 0.25 * math.sin(2 * math.pi * i / n) for i in range(n)]
        sim = _simulation(backend)
        sim.configure_signal_grid(spec, initial)
        steps = math.ceil(duration / (0.12 * (1 / n) ** 2 / diffusion))
        for _ in range(steps):
            sim.step(duration / steps)
        amplitude = 0.25 * math.exp(-diffusion * (2 * math.pi) ** 2 * duration)
        exact = [
            1 + amplitude * math.sin(2 * math.pi * (i / n - speed * duration)) for i in range(n)
        ]
        errors.append(
            math.sqrt(sum((a - b) ** 2 for a, b in zip(sim.signal_levels, exact, strict=True)) / n)
        )
    order_coarse = math.log2(errors[0] / errors[1])
    order_fine = math.log2(errors[1] / errors[2])
    return order_coarse > 0.75 and order_fine > 0.75, {
        "error_n16": errors[0],
        "error_n32": errors[1],
        "error_n64": errors[2],
        "order_coarse": order_coarse,
        "order_fine": order_fine,
    }


def _external_gate(name: str, command: list[str]) -> BenchmarkResult:
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-12:])
    return BenchmarkResult(
        name,
        "integrated",
        completed.returncode == 0,
        time.perf_counter() - started,
        {"return_code": completed.returncode, "output_tail": tail},
        None if completed.returncode == 0 else f"command exited {completed.returncode}",
    )


def _source_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
        )
    except FileNotFoundError:
        return None
    value = completed.stdout.strip()
    return value or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quick", "extended"), default="quick")
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend = {"cpu": BackendKind.CPU, "metal": BackendKind.METAL, "cuda": BackendKind.CUDA}[
        args.backend
    ]
    if not backend_available(backend):
        parser.error(f"backend {args.backend!r} is unavailable in this build")

    checks = [
        ("growth_recurrence", "growth", growth_recurrence),
        ("biomass_and_dilution", "biomass", biomass_and_dilution),
        ("division_conservation", "division", division_conservation),
        ("contacts_and_mechanics", "mechanics", contacts_and_mechanics),
        ("flow_translation_and_rotation", "flow_drift", flow_translation_and_rotation),
        ("transport", "transport", transport),
        ("membrane_exchange", "exchange", membrane_exchange),
        ("monod_and_yield", "growth_coupling", monod_and_yield),
    ]
    results = [
        _result(name, category, lambda check=check: check(backend))
        for name, category, check in checks
    ]

    if args.suite == "extended":
        results.extend(
            [
                _result(
                    "mechanics_sensitivity", "mechanics", lambda: mechanics_sensitivity(backend)
                ),
                _result("transport_refinement", "transport", lambda: transport_refinement(backend)),
                _result("exchange_refinement", "exchange", lambda: exchange_refinement(backend)),
            ]
        )
        nutrient_output = args.output.with_name(f"{args.output.stem}-nutrient.json")
        results.extend(
            [
                _external_gate(
                    "analytic_flow_solvers",
                    [sys.executable, "scripts/run_flow_benchmarks.py", "--backend", args.backend],
                ),
                _external_gate(
                    "nutrient_penetration_and_balance",
                    [
                        sys.executable,
                        "scripts/run_nutrient_benchmarks.py",
                        "--backend",
                        args.backend,
                        "--output",
                        str(nutrient_output),
                    ],
                ),
                _external_gate(
                    "growing_dividing_confined_colony",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "python/tests/test_controller.py",
                        "-k",
                        "native_controller_composes_regulation_division_and_mechanics",
                    ],
                ),
                _external_gate(
                    "integrated_microfluidic_device",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "python/tests/test_microfluidics.py",
                        "-k",
                        "device_flow_runs_through_the_channel_and_rests_in_the_trap or "
                        "biopixel_example_confines_a_monolayer_under_flow",
                    ],
                ),
            ]
        )

    payload = {
        "schema_version": 1,
        "suite": args.suite,
        "backend": args.backend,
        "backend_device": _simulation(backend).backend_info.name,
        "source_commit": _source_commit(),
        "scientific_role": (
            "CPU reference validation" if args.backend == "cpu" else "backend conformance only"
        ),
        "calibration_claim": False,
        "passed": all(result.passed for result in results),
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for result in results:
        print(f"{'PASS' if result.passed else 'FAIL'} {result.name} ({result.seconds:.3f}s)")
        if result.error:
            print(f"  {result.error}")
    print(f"Evidence: {args.output}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
