from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from microsimulator import (
    RUN_MANIFEST_FORMAT,
    RUN_MANIFEST_VERSION,
    RunManifestError,
    load_checkpoint_bundle,
    load_run_manifest,
)
from microsimulator.cli import main


def _write_model(path: Path, *, side_effect: Path | None = None) -> str:
    effect = f"Path({str(side_effect)!r}).write_text('executed')\n" if side_effect else ""
    path.write_text(
        "from pathlib import Path\n"
        "from microsimulator import CellInit\n"
        f"{effect}"
        "def build(context):\n"
        "    simulation = context.simulation()\n"
        "    cell = CellInit()\n"
        "    cell.length = float(context.parameters.get('length', 1.0))\n"
        "    simulation.add_cell(cell)\n"
        "    return simulation\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_controller_model(path: Path) -> str:
    path.write_text(
        """from microsimulator import CellInit

class Controller:
    def __init__(self, simulation):
        self.simulation = simulation
        self.completed_steps = 0

    def step(self, dt):
        self.simulation.step(dt)
        self.completed_steps += 1

    def controller_state(self):
        return {
            "kind": "manifest-test-controller",
            "version": 1,
            "completed_steps": self.completed_steps,
        }

def build(context):
    simulation = context.simulation()
    simulation.add_cell(CellInit())
    return Controller(simulation)
""",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job(
    *,
    job_id: str,
    model_sha256: str,
    output: str,
    checkpoint_every: int = 0,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "model": {"path": "model.py", "sha256": model_sha256},
        "backend": "cpu",
        "device_index": 0,
        "seed": 42,
        "parameters": {"length": 3.5},
        "stopping": {"maximum_steps": 100, "dt": 0.25, "cell_count": 1},
        "checkpoint_every": checkpoint_every,
        "output": output,
    }


def _write_manifest(path: Path, jobs: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(
            {
                "format": RUN_MANIFEST_FORMAT,
                "version": RUN_MANIFEST_VERSION,
                "jobs": jobs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _invalid_backend(job: dict[str, Any]) -> None:
    job["backend"] = "opencl"


def _negative_dt(job: dict[str, Any]) -> None:
    cast(dict[str, Any], job["stopping"])["dt"] = -0.1


def _zero_cell_count(job: dict[str, Any]) -> None:
    cast(dict[str, Any], job["stopping"])["cell_count"] = 0


def _invalid_id(job: dict[str, Any]) -> None:
    job["id"] = "spaces are not allowed"


@pytest.mark.parametrize("format_name", [RUN_MANIFEST_FORMAT, "cellmodeller2-run-manifest"])
def test_manifest_parsing_is_data_only_and_cli_executes_one_named_job(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], format_name: str
) -> None:
    model = tmp_path / "model.py"
    marker = tmp_path / "executed.txt"
    model_digest = _write_model(model, side_effect=marker)
    manifest_path = tmp_path / "experiment.cm2.runs.json"
    _write_manifest(
        manifest_path,
        [
            _job(job_id="replicate-001", model_sha256=model_digest, output="runs/001.cm2.json"),
            _job(job_id="replicate-002", model_sha256=model_digest, output="runs/002.cm2.json"),
        ],
    )

    document = json.loads(manifest_path.read_text())
    document["format"] = format_name
    manifest_path.write_text(json.dumps(document))
    manifest = load_run_manifest(manifest_path)

    assert not marker.exists()
    assert manifest.source == manifest_path.resolve()
    assert manifest.sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert [job.id for job in manifest.jobs] == ["replicate-001", "replicate-002"]
    first = manifest.job("replicate-001")
    assert first.model == model.resolve()
    assert first.output == (tmp_path / "runs/001.cm2.json").resolve()
    assert first.maximum_steps == 100
    assert first.stop_cell_count == 1

    status = main(
        [
            "run-manifest",
            str(manifest_path),
            "--job",
            "replicate-001",
            "--quiet",
        ]
    )

    assert status == 0
    assert marker.read_text() == "executed"
    assert not (tmp_path / "runs/002.cm2.json").exists()
    output = tmp_path / "runs/001.cm2.json"
    bundle = load_checkpoint_bundle(output)
    experiment = cast(dict[str, Any], bundle.provenance["experiment"])
    assert experiment == {
        "job_id": "replicate-001",
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    }
    run = cast(dict[str, Any], bundle.provenance["run"])
    assert run["completed_steps"] == 0
    assert run["stop_reason"] == "cell_count"
    assert "steps=0" in capsys.readouterr().out


def test_model_digest_is_checked_before_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "model.py"
    marker = tmp_path / "must-not-exist.txt"
    _write_model(model, side_effect=marker)
    manifest_path = tmp_path / "bad-digest.json"
    _write_manifest(
        manifest_path,
        [_job(job_id="bad-digest", model_sha256="0" * 64, output="output.cm2.json")],
    )

    assert (
        main(
            [
                "run-manifest",
                str(manifest_path),
                "--job",
                "bad-digest",
                "--quiet",
            ]
        )
        == 2
    )
    assert "digest does not match manifest" in capsys.readouterr().err
    assert not marker.exists()
    assert not (tmp_path / "output.cm2.json").exists()


def test_manifest_job_accepts_a_native_controller(tmp_path: Path) -> None:
    model = tmp_path / "controller.py"
    digest = _write_controller_model(model)
    job = _job(job_id="controller", model_sha256=digest, output="controller.cm2.json")
    cast(dict[str, Any], job["model"])["path"] = "controller.py"
    cast(dict[str, Any], job["stopping"])["maximum_steps"] = 2
    cast(dict[str, Any], job["stopping"])["cell_count"] = None
    manifest_path = tmp_path / "controller-runs.json"
    _write_manifest(manifest_path, [job])

    assert (
        main(
            [
                "run-manifest",
                str(manifest_path),
                "--job",
                "controller",
                "--quiet",
            ]
        )
        == 0
    )
    bundle = load_checkpoint_bundle(tmp_path / "controller.cm2.json")
    assert bundle.controller == {
        "kind": "manifest-test-controller",
        "version": 1,
        "completed_steps": 2,
    }


def test_manifest_rejects_duplicate_ids_and_output_collisions(tmp_path: Path) -> None:
    digest = _write_model(tmp_path / "model.py")
    manifest_path = tmp_path / "invalid.json"
    duplicate = _job(job_id="same", model_sha256=digest, output="first.json")
    _write_manifest(manifest_path, [duplicate, {**duplicate, "output": "second.json"}])
    with pytest.raises(RunManifestError, match="IDs must be unique"):
        load_run_manifest(manifest_path)

    first = _job(
        job_id="first",
        model_sha256=digest,
        output="runs/colony.json",
        checkpoint_every=2,
    )
    second = _job(
        job_id="second",
        model_sha256=digest,
        output="runs/colony",
        checkpoint_every=3,
    )
    _write_manifest(manifest_path, [first, second])
    with pytest.raises(RunManifestError, match="colliding periodic outputs"):
        load_run_manifest(manifest_path)

    second["checkpoint_every"] = 0
    second["output"] = "runs/colony.step-00000002.json"
    _write_manifest(manifest_path, [first, second])
    with pytest.raises(RunManifestError, match="colliding final/periodic outputs"):
        load_run_manifest(manifest_path)

    first["output"] = "runs/legacy.cm2.json"
    second["output"] = "runs/legacy.step-00000002.cm2.json"
    _write_manifest(manifest_path, [first, second])
    with pytest.raises(RunManifestError, match="colliding final/periodic outputs"):
        load_run_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_invalid_backend, "unknown backend"),
        (_negative_dt, "non-negative"),
        (_zero_cell_count, "outside"),
        (_invalid_id, "ASCII letters"),
    ],
)
def test_manifest_rejects_invalid_job_fields(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    digest = _write_model(tmp_path / "model.py")
    job = _job(job_id="valid", model_sha256=digest, output="output.cm2.json")
    mutation(job)
    path = tmp_path / "invalid.json"
    _write_manifest(path, [job])

    with pytest.raises(RunManifestError, match=message):
        load_run_manifest(path)


def test_manifest_rejects_duplicate_keys_and_nonfinite_parameters(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"format":"first","format":"second"}')
    with pytest.raises(RunManifestError, match="duplicate key"):
        load_run_manifest(path)

    digest = _write_model(tmp_path / "model.py")
    job = _job(job_id="nonfinite", model_sha256=digest, output="output.cm2.json")
    job["parameters"] = {"rate": float("inf")}
    _write_manifest(path, [job])
    with pytest.raises(RunManifestError, match="non-finite"):
        load_run_manifest(path)
