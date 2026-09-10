#!/usr/bin/env python3
"""Record numerical trajectories from the pinned CellModeller OpenCL runtime."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import platform
import random
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import scipy


@dataclass(frozen=True, slots=True)
class Scenario:
    identifier: str
    role: str
    model: str
    seed: int
    dt: float
    sample_steps: tuple[int, ...]


_SCENARIOS = (
    Scenario(
        "growing_2d_colony",
        "growing 2D colony",
        "ex1_simpleGrowth2D.py",
        12345,
        0.05,
        (0, 1, 2, 5, 10, 20),
    ),
    Scenario(
        "constrained_3d_colony",
        "constrained 3D colony",
        "Tutorial_1/Tutorial_1c.py",
        23456,
        0.05,
        (0, 1, 2, 5, 10, 20),
    ),
    Scenario(
        "neighbor_dependent_conjugation",
        "neighbor-dependent model",
        "Conjugation.py",
        34567,
        0.05,
        (0, 20, 40, 60, 80, 100),
    ),
    Scenario(
        "constitutive_species",
        "species model",
        "ex2_constGene.py",
        45678,
        0.01,
        (0, 1, 2, 5, 10, 20),
    ),
    Scenario(
        "mutualistic_signaling",
        "coupled signaling model",
        "Tutorial_3/Tutorial_3.py",
        56789,
        0.01,
        (0, 1, 2, 3, 5, 10),
    ),
)


class _HeadlessRenderer:
    def __init__(self, *_: object, **__: object) -> None:
        pass


def _install_runtime_shims() -> None:
    original_set = cl_array.Array.set

    def shape_compatible_set(array: Any, source: Any, *args: Any, **kwargs: Any) -> Any:
        host = np.asarray(source)
        if host.size == array.size and host.shape != array.shape:
            host = host.reshape(array.shape)
        return original_set(array, host, *args, **kwargs)

    cl_array.Array.set = shape_compatible_set
    from CellModeller.GUI import Renderers

    Renderers.GLBacteriumRenderer = _HeadlessRenderer
    Renderers.GLGridRenderer = _HeadlessRenderer


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_digests(matrix_path: Path) -> tuple[str, dict[str, str]]:
    document = json.loads(matrix_path.read_text(encoding="utf-8"))
    return document["legacy_commit"], {row["path"]: row["sha256"] for row in document["examples"]}


def _signal_statistics(simulator: Any) -> list[dict[str, float]]:
    integrator = simulator.integ
    if integrator is None or not hasattr(integrator, "signalLevel"):
        return []
    shape = tuple(int(value) for value in integrator.gridDim)
    levels = np.asarray(integrator.signalLevel, dtype=np.float64).reshape(shape)
    result: list[dict[str, float]] = []
    for channel in levels:
        result.append(
            {
                "sum": float(channel.sum()),
                "maximum": float(channel.max()),
                "l2_norm": float(np.linalg.norm(channel)),
            }
        )
    return result


def _frame(simulator: Any, step: int) -> dict[str, object]:
    cells = sorted(simulator.cellStates.values(), key=lambda cell: cell.id)
    positions = np.asarray(
        [[float(value) for value in cell.pos] for cell in cells], dtype=np.float64
    )
    lengths = np.asarray([float(cell.length) for cell in cells], dtype=np.float64)
    centroid = positions.mean(axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    species_count = max((len(getattr(cell, "species", ())) for cell in cells), default=0)
    species = np.asarray(
        [[float(value) for value in getattr(cell, "species", ())] for cell in cells],
        dtype=np.float64,
    ).reshape((len(cells), species_count))
    type_counts: dict[str, int] = {}
    neighbor_pairs: set[tuple[int, int]] = set()
    for cell in cells:
        key = str(int(cell.cellType))
        type_counts[key] = type_counts.get(key, 0) + 1
        for neighbor in getattr(cell, "neighbours", []):
            neighbor_pairs.add(tuple(sorted((int(cell.id), int(neighbor)))))
    result: dict[str, object] = {
        "step": step,
        "cell_count": len(cells),
        "cell_type_counts": type_counts,
        "total_length": float(lengths.sum()),
        "minimum_length": float(lengths.min()),
        "maximum_length": float(lengths.max()),
        "centroid": [float(value) for value in centroid],
        "coordinate_minimum": [float(value) for value in positions.min(axis=0)],
        "coordinate_maximum": [float(value) for value in positions.max(axis=0)],
        "maximum_centroid_distance": float(distances.max()),
        "neighbor_pair_count": len(neighbor_pairs),
        "mechanics_substeps": int(getattr(simulator.phys, "sub_tick_i", 0)),
        "species_sum": [float(value) for value in species.sum(axis=0)],
        "species_minimum": [float(value) for value in species.min(axis=0)],
        "species_maximum": [float(value) for value in species.max(axis=0)],
        "signals": _signal_statistics(simulator),
    }
    for value in _numbers(result):
        if not math.isfinite(value):
            raise RuntimeError(f"scenario produced a non-finite value at step {step}")
    return result


def _numbers(value: object):
    if isinstance(value, float):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)


def _record_scenario(
    scenario: Scenario,
    *,
    legacy_root: Path,
    platform_index: int,
    device_index: int,
    expected_digest: str,
) -> dict[str, object]:
    model = legacy_root / "Examples" / scenario.model
    if _digest(model) != expected_digest:
        raise RuntimeError(f"legacy source digest mismatch for {scenario.model}")
    random.seed(scenario.seed)
    np.random.seed(scenario.seed)
    from CellModeller.Simulator import Simulator

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        simulator = Simulator(
            str(model),
            scenario.dt,
            saveOutput=False,
            clPlatformNum=platform_index,
            clDeviceNum=device_index,
            is_gui=False,
        )
    simulator.saveOutput = False
    frames: list[dict[str, object]] = []
    final_step = scenario.sample_steps[-1]
    samples = set(scenario.sample_steps)
    for step in range(final_step + 1):
        if step in samples:
            frames.append(_frame(simulator, step))
        if step != final_step:
            with contextlib.redirect_stdout(quiet):
                simulator.step()
    return {
        "id": scenario.identifier,
        "role": scenario.role,
        "model": scenario.model,
        "model_sha256": expected_digest,
        "seed": scenario.seed,
        "dt": scenario.dt,
        "sample_steps": list(scenario.sample_steps),
        "frames": frames,
    }


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--platform-index", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=project_root / "compatibility" / "legacy-examples-v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
    warnings.filterwarnings("ignore", message="Kernel .* has been retrieved more than once.*")
    legacy_root = arguments.legacy_root.resolve()
    expected_commit, digests = _source_digests(arguments.matrix)
    actual_commit = _legacy_commit(legacy_root)
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"legacy commit mismatch: expected {expected_commit}, found {actual_commit}"
        )
    platforms = cl.get_platforms()
    opencl_platform = platforms[arguments.platform_index]
    opencl_device = opencl_platform.get_devices()[arguments.device_index]
    sys.path.insert(0, str(legacy_root))
    _install_runtime_shims()
    scenarios = [
        _record_scenario(
            scenario,
            legacy_root=legacy_root,
            platform_index=arguments.platform_index,
            device_index=arguments.device_index,
            expected_digest=digests[scenario.model],
        )
        for scenario in _SCENARIOS
    ]
    document = {
        "format": "microsimulator-recorded-legacy-trajectories",
        "version": 1,
        "legacy_repository": "https://github.com/CellModeller/CellModeller",
        "legacy_commit": actual_commit,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyopencl": cl.VERSION_TEXT,
            "opencl_platform": opencl_platform.name,
            "opencl_platform_version": opencl_platform.version,
            "opencl_device": opencl_device.name,
            "compatibility_shims": [
                "headless no-op renderer constructors",
                "reshape equal-sized host arrays for current PyOpenCL Array.set",
            ],
        },
        "scenarios": scenarios,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.output)
    print(f"recorded {len(scenarios)} legacy trajectories: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
