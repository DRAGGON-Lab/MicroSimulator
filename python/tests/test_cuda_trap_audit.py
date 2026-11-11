"""Regression checks for the CPU-referenced trap audit's velocity gate."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_AUDIT = Path(__file__).resolve().parents[2] / "scripts" / "audit_cuda_trap.py"
_STEPS = ("0", "1", "50", "100", "500", "1000")


def _run_document(device: str) -> dict[str, Any]:
    snapshot = {
        "cells": [{"id": 7, "length": 2.0, "position": [1.0, 2.0, 3.0]}],
        "nutrient": [1.0, 2.0],
        "flow": {component: [0.0, 10.0] for component in ("x", "y", "z")},
        "signal_report": None,
    }
    return {
        "device": device,
        "model_sha256": "same-model",
        "seed": 42,
        "dt": 0.001,
        "steps": 1000,
        "flow_report": {"relative_residual": 5.0e-7},
        "snapshots": {step: copy.deepcopy(snapshot) for step in _STEPS},
    }


def _compare(
    tmp_path: Path, cpu: dict[str, Any], cuda: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    cpu_path = tmp_path / "cpu.json"
    cuda_path = tmp_path / "cuda.json"
    report_path = tmp_path / "comparison.json"
    cpu_path.write_text(json.dumps(cpu), encoding="utf-8")
    cuda_path.write_text(json.dumps(cuda), encoding="utf-8")
    process = subprocess.run(
        [sys.executable, str(_AUDIT), "compare", str(cpu_path), str(cuda_path), str(report_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert report_path.exists(), process.stderr
    return process.returncode, json.loads(report_path.read_text(encoding="utf-8"))


def test_matching_and_within_tolerance_velocity_faces_pass(tmp_path: Path) -> None:
    cpu = _run_document("cpu")
    cuda = _run_document("cuda")
    cuda["snapshots"]["50"]["flow"]["x"][0] = 0.00079
    cuda["snapshots"]["100"]["flow"]["y"][1] = 10.0087

    exit_code, report = _compare(tmp_path, cpu, cuda)

    assert exit_code == 0
    assert report["pass"] is True
    assert all(
        item["flow_pass"] and item["first_flow_face_failure"] is None
        for item in report["comparisons"]
    )


@pytest.mark.parametrize("component", ("x", "y", "z"))
@pytest.mark.parametrize("step", _STEPS)
def test_velocity_mismatch_alone_fails_at_each_recorded_stage(
    tmp_path: Path, component: str, step: str
) -> None:
    cpu = _run_document("cpu")
    cuda = _run_document("cuda")
    cuda["snapshots"][step]["flow"][component][0] = 0.00081

    exit_code, report = _compare(tmp_path, cpu, cuda)

    assert exit_code == 1
    assert report["pass"] is False
    failed = [item for item in report["comparisons"] if not item["pass"]]
    assert len(failed) == 1
    assert failed[0]["step"] == int(step)
    assert failed[0]["flow_pass"] is False
    assert failed[0]["first_flow_face_failure"]["component"] == component
    assert failed[0]["first_flow_face_failure"]["index"] == 0
    assert failed[0]["first_flow_face_failure"]["reason"] == "over_tolerance"
    assert failed[0]["max_nutrient_error"] == 0.0
    assert failed[0]["max_length_error"] == 0.0
    assert failed[0]["max_position_error"] == 0.0


def test_relative_velocity_tolerance_boundary(tmp_path: Path) -> None:
    cpu = _run_document("cpu")
    cuda = _run_document("cuda")
    cuda["snapshots"]["1"]["flow"]["y"][1] = 10.0089

    exit_code, report = _compare(tmp_path, cpu, cuda)

    assert exit_code == 1
    failed = next(item for item in report["comparisons"] if item["step"] == 1)
    assert failed["first_flow_face_failure"]["component"] == "y"
    assert failed["first_flow_face_failure"]["index"] == 1
    assert math.isclose(failed["first_flow_face_failure"]["tolerance"], 0.0088)


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_nonfinite_velocity_face_fails(tmp_path: Path, value: float) -> None:
    cpu = _run_document("cpu")
    cuda = _run_document("cuda")
    cuda["snapshots"]["500"]["flow"]["z"][1] = value

    exit_code, report = _compare(tmp_path, cpu, cuda)

    assert exit_code == 1
    failed = next(item for item in report["comparisons"] if item["step"] == 500)
    assert failed["first_flow_face_failure"] == {
        "component": "z",
        "index": 1,
        "reason": "nonfinite",
    }
    assert failed["max_flow_face_errors"]["z"] is None


def test_velocity_face_count_mismatch_fails(tmp_path: Path) -> None:
    cpu = _run_document("cpu")
    cuda = _run_document("cuda")
    cuda["snapshots"]["100"]["flow"]["x"].append(0.0)

    exit_code, report = _compare(tmp_path, cpu, cuda)

    assert exit_code == 1
    failed = next(item for item in report["comparisons"] if item["step"] == 100)
    assert failed["first_flow_face_failure"] == {
        "component": "x",
        "index": 2,
        "reason": "length_mismatch",
        "cpu_faces": 2,
        "cuda_faces": 3,
    }
