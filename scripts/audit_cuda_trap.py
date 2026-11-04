"""Record and compare CPU/CUDA snapshots of the microfluidic trap."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from microsimulator import BackendKind, ModelContext
from microsimulator.flow import gap_mobility, solve_flow_field
from microsimulator.runner import build_model

MARKS = {0, 1, 50, 100, 500, 1000}


def snapshot(model: object, step: int) -> dict:
    simulation = model.simulation
    checkpoint = simulation._checkpoint()
    grid = checkpoint.signal_grid
    field = None if grid is None else grid.spec.velocity_field
    report = simulation.last_signal_solve_report
    return {
        "step": step,
        "cells": [
            {
                "id": cell.id,
                "length": cell.length,
                "position": [cell.position.x, cell.position.y, cell.position.z],
            }
            for cell in simulation.cells()
        ],
        "nutrient": simulation.signal_levels,
        "flow": None
        if field is None
        else {
            "x": field.x_faces,
            "y": field.y_faces,
            "z": field.z_faces,
        },
        "signal_report": None
        if report is None
        else {
            "converged": report.converged,
            "iterations": report.iterations,
            "residual_rms": report.residual_rms,
        },
    }


def run(backend: str, output: Path, steps: int) -> None:
    kind = BackendKind.CPU if backend == "cpu" else BackendKind.CUDA
    start = time.perf_counter()
    model, provenance = build_model(
        Path("examples/microfluidic_trap.py"), ModelContext(kind, 0, seed=42)
    )
    built = time.perf_counter()
    simulation = model.simulation
    info = simulation.backend_info
    if backend == "cuda" and (info.kind != BackendKind.CUDA or not info.native):
        raise RuntimeError("CUDA audit requires the native CUDA backend")
    grid = simulation._checkpoint().signal_grid
    if grid is None or grid.spec.velocity_field is None:
        raise RuntimeError("trap construction did not configure a velocity field")
    _, flow_report = solve_flow_field(
        grid.spec,
        mean_inlet_speed=20.0,
        mobility=gap_mobility(grid.spec),
        simulation=simulation,
    )
    snapshots = {"0": snapshot(model, 0)}
    for step in range(1, steps + 1):
        model.step(0.001)
        if step in MARKS:
            snapshots[str(step)] = snapshot(model, step)
            print(f"{backend}: step {step}, cells {simulation.cell_count}", flush=True)
    ended = time.perf_counter()
    result = {
        "backend": backend,
        "device": info.device,
        "native": info.native,
        "model_sha256": provenance["model"]["sha256"],
        "seed": 42,
        "dt": 0.001,
        "steps": steps,
        "build_s": built - start,
        "steps_s": ended - built,
        "flow_report": {
            "iterations": flow_report.iterations,
            "residual": flow_report.residual,
            "relative_residual": flow_report.relative_residual,
            "max_speed": flow_report.max_speed,
        },
        "snapshots": snapshots,
    }
    output.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {output}", flush=True)


def compare(cpu_path: Path, cuda_path: Path, output: Path) -> None:
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    cuda = json.loads(cuda_path.read_text(encoding="utf-8"))
    if (cpu["model_sha256"], cpu["seed"], cpu["dt"], cpu["steps"]) != (
        cuda["model_sha256"],
        cuda["seed"],
        cuda["dt"],
        cuda["steps"],
    ):
        raise ValueError("CPU/CUDA run inputs differ")
    comparisons = []
    passed = True
    pressure_residual_error = abs(
        cpu["flow_report"]["relative_residual"] - cuda["flow_report"]["relative_residual"]
    )
    passed &= (
        cpu["flow_report"]["relative_residual"] <= 1.1e-6
        and cuda["flow_report"]["relative_residual"] <= 1.1e-6
        and pressure_residual_error <= 1e-6
    )
    for step, expected in cpu["snapshots"].items():
        actual = cuda["snapshots"][step]
        expected_cells = {cell["id"]: cell for cell in expected["cells"]}
        actual_cells = {cell["id"]: cell for cell in actual["cells"]}
        ids_match = expected_cells.keys() == actual_cells.keys()
        length_error = 0.0
        position_error = 0.0
        if ids_match:
            for cell_id, cell in expected_cells.items():
                other = actual_cells[cell_id]
                length_error = max(length_error, abs(cell["length"] - other["length"]))
                position_error = max(
                    position_error,
                    *(abs(a - b) for a, b in zip(cell["position"], other["position"], strict=True)),
                )
        nutrient_pairs = list(zip(expected["nutrient"], actual["nutrient"], strict=True))
        nutrient_error = max((abs(a - b) for a, b in nutrient_pairs), default=0.0)
        first_bad = next((i for i, (a, b) in enumerate(nutrient_pairs) if abs(a - b) > 0.01), None)
        total_cpu = sum(expected["nutrient"])
        total_cuda = sum(actual["nutrient"])
        total_error_percent = 100 * abs(total_cuda - total_cpu) / total_cpu
        max_nutrient = max(actual["nutrient"])
        flow_errors = {}
        for component in ("x", "y", "z"):
            left = expected["flow"][component]
            right = actual["flow"][component]
            flow_errors[component] = max(
                (abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0
            )
        ok = (
            ids_match
            and length_error <= 0.001
            and position_error <= 0.01
            and nutrient_error <= 0.01
            and total_error_percent <= 0.1
            and max_nutrient <= 10.001
        )
        passed &= ok
        comparisons.append(
            {
                "step": int(step),
                "pass": ok,
                "cpu_cells": len(expected_cells),
                "cuda_cells": len(actual_cells),
                "ids_match": ids_match,
                "max_length_error": length_error,
                "max_position_error": position_error,
                "max_nutrient_error": nutrient_error,
                "first_nutrient_index_over_0.01": first_bad,
                "total_nutrient_error_percent": total_error_percent,
                "max_cuda_nutrient": max_nutrient,
                "max_flow_face_errors": flow_errors,
                "cpu_signal_report": expected["signal_report"],
                "cuda_signal_report": actual["signal_report"],
            }
        )
    result = {
        "pass": passed,
        "cpu_device": cpu["device"],
        "cuda_device": cuda["device"],
        "cpu_flow_report": cpu["flow_report"],
        "cuda_flow_report": cuda["flow_report"],
        "pressure_relative_residual_error": pressure_residual_error,
        "comparisons": comparisons,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    run_parser = subcommands.add_parser("run")
    run_parser.add_argument("backend", choices=("cpu", "cuda"))
    run_parser.add_argument("output", type=Path)
    run_parser.add_argument("--steps", type=int, default=1000)
    compare_parser = subcommands.add_parser("compare")
    compare_parser.add_argument("cpu", type=Path)
    compare_parser.add_argument("cuda", type=Path)
    compare_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.backend, args.output, args.steps)
    else:
        compare(args.cpu, args.cuda, args.output)


if __name__ == "__main__":
    main()
