"""Record and compare seeded mechanics trajectories before and after a refactor.

Run from the repository root with the environment containing the native extension.
Each scenario writes a per-step JSONL file, a compact manifest, and geometry
snapshots. Step failures and safety-gate violations are recorded; the command
returns an unsuccessful status after writing every attempted run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

SCENARIOS = {
    "square": (Path("examples/square_trap.py"), 150, 0.05, (0, 1, 16, 20, 60, 100, 150)),
    "flow": (
        Path("examples/microfluidic_trap.py"),
        12_310,
        0.001,
        (0, 1, 37, 100, 1_000, 3_000, 6_000, 9_000, 12_000, 12_310),
    ),
}
GAP_TOLERANCE = 1.0e-4
_DLL_HANDLES: list[object] = []


def _enable_cuda_runtime() -> None:
    """Let Windows resolve CUDA DLLs when importing the native extension."""
    if not hasattr(os, "add_dll_directory"):
        return
    cuda_path = os.environ.get("CUDA_PATH", "")
    candidates = [str(Path(cuda_path) / "bin")] if cuda_path else []
    candidates.extend(
        str(path)
        for path in Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").glob("*/bin")
    )
    for path in candidates:
        if path and Path(path).is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(path))


def _git_revision() -> str:
    for executable in ("git", "C:/Program Files/Git/cmd/git.exe"):
        try:
            return subprocess.check_output(
                [executable, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return "unavailable"


def _contacts(simulation: object) -> tuple[float, float, list[list[int]], list[list[int]]]:
    pairs = simulation.find_cell_contacts().contacts
    walls = simulation.find_external_contacts().contacts
    return (
        min((edge.signed_separation for edge in pairs), default=0.0),
        min((edge.signed_separation for edge in walls), default=0.0),
        sorted([[edge.first_id, edge.second_id] for edge in pairs]),
        sorted([[edge.cell_id, edge.constraint_id] for edge in walls]),
    )


def _added_edges(current: list[list[int]], previous: list[list[int]]) -> list[list[int]]:
    """Preserve repeated contact rows between the same cells or wall."""
    difference = Counter(map(tuple, current)) - Counter(map(tuple, previous))
    return sorted([list(edge) for edge in difference.elements()])


def _measure(simulation: object, step: int, *, snapshot: bool) -> tuple[dict, dict | None]:
    cells = simulation.cells()
    cell_gap, wall_gap, pairs, walls = _contacts(simulation)
    samples = []
    sample_errors = []
    for cell in cells:
        try:
            sample = simulation.sample_signals(cell.position)[0]
            if not math.isfinite(sample):
                raise ValueError(f"non-finite nutrient: {sample}")
            samples.append(sample)
        except (IndexError, ValueError) as error:
            sample_errors.append({"cell_id": cell.id, "error": str(error)})
    row = {
        "step": step,
        "cells": len(cells),
        "biomass": sum(
            math.pi * cell.radius**2 * (cell.length + 2 * cell.radius) for cell in cells
        ),
        "total_length": sum(cell.length for cell in cells),
        "lateral_spread": max((abs(cell.position.y) for cell in cells), default=0.0),
        "cell_gap": cell_gap,
        "wall_gap": wall_gap,
        "nutrient_min": min(samples, default=None),
        "nutrient_max": max(samples, default=None),
        "nutrient_mean": sum(samples) / len(samples) if samples else None,
        "invalid_nutrient_samples": len(sample_errors),
    }
    detail = None
    if snapshot:
        detail = {
            "step": step,
            "cells": [
                {
                    "id": cell.id,
                    "position": [cell.position.x, cell.position.y, cell.position.z],
                    "direction": [cell.direction.x, cell.direction.y, cell.direction.z],
                    "length": cell.length,
                    "radius": cell.radius,
                    "fixed": cell.fixed,
                }
                for cell in cells
            ],
            "cell_edges": pairs,
            "wall_edges": walls,
            "sample_errors": sample_errors,
        }
    return row, detail


def run(output: Path, scenarios: list[str], backends: list[str], steps_override: int | None) -> int:
    _enable_cuda_runtime()
    from microsimulator import BackendKind, ModelContext, _core, backend_available
    from microsimulator.runner import build_model

    output.mkdir(parents=True, exist_ok=True)
    failed = False
    for scenario in scenarios:
        model_path, steps, dt, marks = SCENARIOS[scenario]
        if steps_override is not None:
            steps = min(steps, steps_override)
        for backend in backends:
            kind = BackendKind.CPU if backend == "cpu" else BackendKind.CUDA
            if not backend_available(kind):
                raise RuntimeError(f"native {backend} backend unavailable; no CPU substitution")
            name = f"{scenario}_{backend}"
            model, provenance = build_model(model_path, ModelContext(kind, 0, seed=42))
            simulation = model.simulation
            info = simulation.backend_info
            if info.kind != kind or not info.native:
                raise RuntimeError(f"{name} did not select the requested native backend")
            manifest = {
                "scenario": scenario,
                "backend": backend,
                "device": info.device,
                "native": info.native,
                "extension_path": str(Path(_core.__file__).resolve()),
                "extension_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
                "git_revision": _git_revision(),
                "model_path": str(model_path),
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "provenance_sha256": provenance["model"]["sha256"],
                "seed": 42,
                "dt": dt,
                "planned_steps": steps,
                "failure": None,
                "first_penetration": None,
                "penetration_steps": 0,
                "first_invalid_nutrient": None,
                "invalid_nutrient_samples": 0,
            }
            snapshots = {}
            worst_cell = 0.0
            worst_wall = 0.0
            start = time.perf_counter()
            with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
                for step in range(steps + 1):
                    if step:
                        try:
                            model.step(dt)
                        except Exception as error:
                            manifest["failure"] = {"step": step, "error": repr(error)}
                            failed = True
                            break
                    try:
                        row, detail = _measure(simulation, step, snapshot=step in marks)
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    except Exception as error:
                        manifest["failure"] = {"step": step, "error": repr(error)}
                        failed = True
                        break
                    if detail is not None:
                        snapshots[str(step)] = detail
                    worst_cell = min(worst_cell, row["cell_gap"])
                    worst_wall = min(worst_wall, row["wall_gap"])
                    if row["cell_gap"] < -GAP_TOLERANCE or row["wall_gap"] < -GAP_TOLERANCE:
                        failed = True
                        if manifest["first_penetration"] is None:
                            manifest["first_penetration"] = {
                                "step": step,
                                "cell_gap": row["cell_gap"],
                                "wall_gap": row["wall_gap"],
                            }
                        manifest["penetration_steps"] += 1
                    if row["invalid_nutrient_samples"]:
                        failed = True
                        if manifest["first_invalid_nutrient"] is None:
                            manifest["first_invalid_nutrient"] = {
                                "step": step,
                                "count": row["invalid_nutrient_samples"],
                                "errors": detail["sample_errors"] if detail is not None else None,
                            }
                        manifest["invalid_nutrient_samples"] += row["invalid_nutrient_samples"]
                    if step % 1000 == 0:
                        print(f"{name}: {step}/{steps}, cells={row['cells']}", flush=True)
            manifest["elapsed_seconds"] = time.perf_counter() - start
            manifest["completed_steps"] = step
            manifest["worst_cell_gap"] = worst_cell
            manifest["worst_wall_gap"] = worst_wall
            (output / f"{name}.manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            (output / f"{name}.snapshots.json").write_text(
                json.dumps(snapshots, indent=2) + "\n", encoding="utf-8"
            )
            outcome = manifest["failure"] or (
                "safety gates failed"
                if manifest["penetration_steps"] or manifest["invalid_nutrient_samples"]
                else "complete"
            )
            print(
                f"{name}: {outcome}, "
                f"penetration steps={manifest['penetration_steps']}, "
                f"invalid samples={manifest['invalid_nutrient_samples']}",
                flush=True,
            )
    return int(failed)


def compare(baseline: Path, candidate: Path, output: Path) -> int:
    result = {"baseline": str(baseline), "candidate": str(candidate), "runs": {}}
    passed = True
    for scenario, (_, steps, _, marks) in SCENARIOS.items():
        for backend in ("cpu", "cuda"):
            name = f"{scenario}_{backend}"
            old = json.loads((baseline / f"{name}.manifest.json").read_text(encoding="utf-8"))
            new = json.loads((candidate / f"{name}.manifest.json").read_text(encoding="utf-8"))
            old_rows = [
                json.loads(line)
                for line in (baseline / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            new_rows = [
                json.loads(line)
                for line in (candidate / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            old_snaps = json.loads(
                (baseline / f"{name}.snapshots.json").read_text(encoding="utf-8")
            )
            new_snaps = json.loads(
                (candidate / f"{name}.snapshots.json").read_text(encoding="utf-8")
            )
            if (
                old["model_sha256"] != new["model_sha256"]
                or old["seed"] != new["seed"]
                or old["dt"] != new["dt"]
                or old["planned_steps"] != new["planned_steps"]
            ):
                raise ValueError(f"{name}: model or run inputs differ")
            snapshots = {}
            for mark in marks:
                key = str(mark)
                if key not in old_snaps or key not in new_snaps:
                    continue
                a, b = old_snaps[key], new_snaps[key]
                a_ids = {cell["id"]: cell for cell in a["cells"]}
                b_ids = {cell["id"]: cell for cell in b["cells"]}
                common = a_ids.keys() & b_ids.keys()
                snapshots[key] = {
                    "baseline_cells": len(a_ids),
                    "candidate_cells": len(b_ids),
                    "shared_ids": len(common),
                    "max_shared_position_delta": max(
                        (
                            max(
                                abs(x - y)
                                for x, y in zip(
                                    a_ids[i]["position"], b_ids[i]["position"], strict=True
                                )
                            )
                            for i in common
                        ),
                        default=None,
                    ),
                    "cell_edges_added": _added_edges(b["cell_edges"], a["cell_edges"]),
                    "cell_edges_removed": _added_edges(a["cell_edges"], b["cell_edges"]),
                    "wall_edges_added": _added_edges(b["wall_edges"], a["wall_edges"]),
                    "wall_edges_removed": _added_edges(a["wall_edges"], b["wall_edges"]),
                }
            square = scenario == "square"
            growth_ok = True
            if square and len(new_rows) > 60 and len(old_rows) > 60:
                baseline_gain = old_rows[60]["biomass"] - old_rows[20]["biomass"]
                candidate_gain = new_rows[60]["biomass"] - new_rows[20]["biomass"]
                growth_ok = (
                    new_rows[60]["biomass"] >= old_rows[60]["biomass"] * 0.9
                    and candidate_gain >= baseline_gain * 0.8
                )
            if not square and len(new_rows) > 1_000 and len(old_rows) > 1_000:
                baseline_gain = old_rows[1_000]["biomass"] - old_rows[100]["biomass"]
                candidate_gain = new_rows[1_000]["biomass"] - new_rows[100]["biomass"]
                growth_ok = candidate_gain >= baseline_gain * 0.8
            square_ok = not square or (
                max(row["cells"] for row in new_rows) >= 8
                and max(row["lateral_spread"] for row in new_rows) > 0.2
            )
            ok = (
                new["failure"] is None
                and new["completed_steps"] == steps
                and new["worst_cell_gap"] >= -GAP_TOLERANCE
                and new["worst_wall_gap"] >= -GAP_TOLERANCE
                and new["invalid_nutrient_samples"] == 0
                and growth_ok
                and square_ok
            )
            passed &= ok
            result["runs"][name] = {
                "pass": ok,
                "baseline": old,
                "candidate": new,
                "growth_progression_ok": growth_ok,
                "square_geometry_ok": square_ok,
                "biomass_by_snapshot": {
                    str(mark): {
                        "baseline": old_rows[mark]["biomass"],
                        "candidate": new_rows[mark]["biomass"],
                    }
                    for mark in marks
                    if mark < len(old_rows) and mark < len(new_rows)
                },
                "snapshots": snapshots,
                "runtime_ratio": new["elapsed_seconds"] / old["elapsed_seconds"],
            }
    result["pass"] = passed
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"comparison: {'PASS' if passed else 'FAIL'}; wrote {output}")
    return int(not passed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("output", type=Path)
    run_parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    run_parser.add_argument("--backend", choices=("cpu", "cuda", "all"), default="all")
    run_parser.add_argument("--steps", type=int, help="limit each scenario for a smoke run")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        if args.steps is not None and args.steps < 1:
            parser.error("--steps must be positive")
        scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
        backends = ["cpu", "cuda"] if args.backend == "all" else [args.backend]
        return run(args.output, scenarios, backends, args.steps)
    return compare(args.baseline, args.candidate, args.output)


if __name__ == "__main__":
    sys.exit(main())
