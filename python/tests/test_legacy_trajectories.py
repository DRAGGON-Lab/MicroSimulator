from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from microsimulator import (
    BackendKind,
    ModelContext,
    RunnableModel,
    backend_available,
    build_legacy_model,
    build_model,
)
from microsimulator.runner import native_simulation

_ROOT = Path(__file__).resolve().parents[2]
_REFERENCE = _ROOT / "compatibility" / "legacy-trajectories-v1.json"
_LEGACY_ROOT_VALUE = os.environ.get("CM_LEGACY_ROOT")


@dataclass(frozen=True, slots=True)
class Tolerance:
    cell_count_absolute: int
    total_length_relative: float
    centroid_absolute: float
    radius_relative: float
    neighbor_absolute: int = 0
    neighbor_relative: float = 0.0
    species_relative: float = 0.0
    signal_sum_relative: float = 0.0
    signal_maximum_relative: float = 0.0


_TOLERANCES = {
    "growing_2d_colony": Tolerance(1, 0.25, 0.7, 0.25),
    "constrained_3d_colony": Tolerance(0, 0.12, 0.02, 0.12),
    "neighbor_dependent_conjugation": Tolerance(
        4,
        0.15,
        0.25,
        0.10,
        neighbor_absolute=2,
        neighbor_relative=0.20,
    ),
    "constitutive_species": Tolerance(
        0,
        0.12,
        0.02,
        0.12,
        species_relative=0.03,
    ),
    "mutualistic_signaling": Tolerance(
        0,
        0.002,
        0.001,
        0.001,
        species_relative=0.001,
        signal_sum_relative=0.003,
        signal_maximum_relative=0.12,
    ),
}

_MODELS = {
    "growing_2d_colony": ("legacy", "ex1_simpleGrowth2D.py"),
    "constrained_3d_colony": ("legacy", "Tutorial_1/Tutorial_1c.py"),
    "neighbor_dependent_conjugation": ("legacy", "Conjugation.py"),
    "constitutive_species": ("native", "examples/legacy/ex2_constGene.py"),
    "mutualistic_signaling": ("native", "examples/legacy/Tutorial_3/Tutorial_3.py"),
}


def _document() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_REFERENCE.read_text(encoding="utf-8")))


def _scenarios() -> list[dict[str, Any]]:
    document = _document()
    assert document["format"] in (
        "microsimulator-recorded-legacy-trajectories",
        "cellmodeller2-recorded-legacy-trajectories",
    )
    assert document["version"] == 1
    assert document["legacy_commit"] == "4896f543c6250f053eea2312e628cc3a96bf7408"
    scenarios = cast(list[dict[str, Any]], document["scenarios"])
    assert {scenario["id"] for scenario in scenarios} == set(_TOLERANCES)
    return scenarios


def _capture(model: RunnableModel, step: int) -> dict[str, Any]:
    simulation = native_simulation(model)
    cells = simulation.cells()
    positions = np.asarray(
        [[cell.position.x, cell.position.y, cell.position.z] for cell in cells],
        dtype=np.float64,
    )
    lengths = np.asarray([cell.length for cell in cells], dtype=np.float64)
    centroid = positions.mean(axis=0)
    species = np.asarray([cell.species for cell in cells], dtype=np.float64).reshape(
        (len(cells), simulation.species_count)
    )
    neighbor_pairs: set[tuple[int, int]] = set()
    legacy_cells = getattr(model, "cells", None)
    if legacy_cells is not None:
        for cell in legacy_cells.values():
            for neighbor in cell.neighbours:
                neighbor_pairs.add(tuple(sorted((cell.id, neighbor))))
    signals: list[dict[str, float]] = []
    if simulation.has_signal_grid:
        levels = np.asarray(simulation.signal_levels, dtype=np.float64).reshape(
            (simulation.signal_count, -1)
        )
        signals = [
            {
                "sum": float(channel.sum()),
                "maximum": float(channel.max()),
                "l2_norm": float(np.linalg.norm(channel)),
            }
            for channel in levels
        ]
    return {
        "step": step,
        "cell_count": len(cells),
        "cell_type_counts": dict(Counter(str(cell.cell_type) for cell in cells)),
        "total_length": float(lengths.sum()),
        "centroid": [float(value) for value in centroid],
        "maximum_centroid_distance": float(np.linalg.norm(positions - centroid, axis=1).max()),
        "neighbor_pair_count": len(neighbor_pairs),
        "species_sum": [float(value) for value in species.sum(axis=0)],
        "signals": signals,
    }


def _close(actual: float, expected: float, *, relative: float, absolute: float) -> None:
    assert math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute), (
        f"{actual} is outside rel={relative} abs={absolute} of legacy {expected}"
    )


def _compare(
    identifier: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    tolerance = _TOLERANCES[identifier]
    assert actual["step"] == expected["step"]
    assert abs(actual["cell_count"] - expected["cell_count"]) <= tolerance.cell_count_absolute
    _close(
        actual["total_length"],
        expected["total_length"],
        relative=tolerance.total_length_relative,
        absolute=1.0e-6,
    )
    for value, reference in zip(actual["centroid"], expected["centroid"], strict=True):
        _close(value, reference, relative=0.0, absolute=tolerance.centroid_absolute)
    _close(
        actual["maximum_centroid_distance"],
        expected["maximum_centroid_distance"],
        relative=tolerance.radius_relative,
        absolute=1.0e-6,
    )
    assert abs(actual["neighbor_pair_count"] - expected["neighbor_pair_count"]) <= (
        tolerance.neighbor_absolute
        + math.ceil(tolerance.neighbor_relative * expected["neighbor_pair_count"])
    )
    assert len(actual["species_sum"]) == len(expected["species_sum"])
    for value, reference in zip(actual["species_sum"], expected["species_sum"], strict=True):
        _close(value, reference, relative=tolerance.species_relative, absolute=1.0e-8)
    assert len(actual["signals"]) == len(expected["signals"])
    for values, references in zip(actual["signals"], expected["signals"], strict=True):
        _close(
            values["sum"],
            references["sum"],
            relative=tolerance.signal_sum_relative,
            absolute=1.0e-8,
        )
        _close(
            values["maximum"],
            references["maximum"],
            relative=tolerance.signal_maximum_relative,
            absolute=1.0e-8,
        )
    if identifier != "neighbor_dependent_conjugation":
        assert set(actual["cell_type_counts"]) == set(expected["cell_type_counts"])
        for cell_type, count in actual["cell_type_counts"].items():
            assert (
                abs(count - expected["cell_type_counts"][cell_type])
                <= tolerance.cell_count_absolute
            )


@pytest.mark.parametrize("backend", list(BackendKind))
@pytest.mark.parametrize("scenario", _scenarios(), ids=lambda value: value["id"])
def test_recorded_legacy_trajectory_contract(
    backend: BackendKind,
    scenario: dict[str, Any],
) -> None:
    if _LEGACY_ROOT_VALUE is None:
        pytest.skip("CM_LEGACY_ROOT is required for recorded legacy trajectory tests")
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    legacy_root = Path(_LEGACY_ROOT_VALUE)
    identifier = cast(str, scenario["id"])
    kind, relative_path = _MODELS[identifier]
    legacy_source = legacy_root / "Examples" / cast(str, scenario["model"])
    assert hashlib.sha256(legacy_source.read_bytes()).hexdigest() == scenario["model_sha256"]
    context = ModelContext(backend, 0, seed=cast(int, scenario["seed"]))
    model, _ = (
        build_legacy_model(legacy_source, context)
        if kind == "legacy"
        else build_model(_ROOT / relative_path, context)
    )
    frames = {
        cast(int, frame["step"]): frame for frame in cast(list[dict[str, Any]], scenario["frames"])
    }
    final_step = max(frames)
    dt = cast(float, scenario["dt"])
    for step in range(final_step + 1):
        if step in frames:
            _compare(identifier, _capture(model, step), frames[step])
        if step != final_step:
            model.step(dt)
    if identifier == "neighbor_dependent_conjugation":
        final_types = _capture(model, final_step)["cell_type_counts"]
        assert {"0", "1", "2"}.issubset(final_types)
