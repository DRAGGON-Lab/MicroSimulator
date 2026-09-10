from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
from microsimulator import (
    BackendKind,
    BatchError,
    ModelContext,
    Simulation,
    SimulationController,
    build_model,
    load_checkpoint,
    load_checkpoint_bundle,
    run_simulation,
)
from microsimulator.cli import main


def _write_model(path: Path) -> None:
    path.write_text(
        """from microsimulator import CellInit, Vec3

def build(context):
    simulation = context.simulation()
    cell = CellInit()
    cell.position = Vec3(context.rng.random(), 0.0, 0.0)
    cell.length = float(context.parameters.get("length", 2.0))
    cell.growth_rate = 0.25
    cell.cell_type = int(context.parameters.get("cell_type", 0))
    simulation.add_cell(cell)
    return simulation
""",
        encoding="utf-8",
    )


def _write_controller_model(path: Path) -> None:
    path.write_text(
        """from microsimulator import (
    CellInit,
    capture_random_state,
    restore_random_state,
)

class Controller:
    def __init__(self, simulation, rng, completed_steps=0):
        self.simulation = simulation
        self.rng = rng
        self.completed_steps = completed_steps

    def step(self, dt):
        cell = self.simulation.cell(1)
        growth_rate = 0.1 + self.rng.random() * 0.2
        self.simulation.set_cell_attributes(cell.id, growth_rate, cell.cell_type)
        self.simulation.step(dt)
        self.completed_steps += 1

    def controller_state(self):
        return {
            "kind": "runner-test-controller",
            "version": 1,
            "completed_steps": self.completed_steps,
            "random": capture_random_state(self.rng),
        }

def build(context):
    simulation = context.simulation()
    cell = CellInit()
    cell.length = 2.0 + context.rng.random()
    simulation.add_cell(cell)
    return Controller(simulation, context.rng)

def resume(context, checkpoint):
    state = checkpoint.controller
    if not isinstance(state, dict) or state.get("kind") != "runner-test-controller":
        raise ValueError("unsupported controller state")
    return Controller(
        checkpoint.simulation,
        restore_random_state(state["random"]),
        int(state["completed_steps"]),
    )
""",
        encoding="utf-8",
    )


def _document(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_batch_library_is_deterministic_and_preflights_outputs(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    _write_model(model)
    context = ModelContext(
        backend=BackendKind.CPU,
        device_index=0,
        seed=1234,
        parameters={"length": 4.0, "cell_type": 7},
    )
    simulation, provenance = build_model(model, context)
    assert isinstance(simulation, Simulation)
    output = tmp_path / "run.json"
    progress_steps: list[int] = []
    summary = run_simulation(
        simulation,
        steps=3,
        dt=0.1,
        output=output,
        checkpoint_every=2,
        provenance=provenance,
        progress=lambda progress: progress_steps.append(progress.completed_steps),
    )

    assert summary.output == output
    assert summary.stop_reason == "step_limit"
    assert summary.cell_count_threshold is None
    assert summary.periodic_checkpoints == (tmp_path / "run.step-00000002.json",)
    assert summary.periodic_checkpoints[0].exists()
    assert progress_steps == [1, 2, 3]
    restored = load_checkpoint(output)
    assert math.isclose(restored.time, simulation.time)
    assert restored.cell(1).cell_type == 7
    assert restored.cell(1).position.x == simulation.cell(1).position.x
    assert restored.cell(1).length == simulation.cell(1).length

    document = _document(output)
    assert document["provenance"]["model"]["seed"] == 1234
    assert document["provenance"]["model"]["sha256"] == hashlib.sha256(
        model.read_bytes()
    ).hexdigest()
    assert document["provenance"]["run"] == {
        "completed_steps": 3,
        "dt": 0.1,
        "requested_steps": 3,
        "status": "complete",
        "stop_reason": "step_limit",
        "stopping": {"cell_count": None, "maximum_steps": 3},
    }

    original_bytes = output.read_bytes()
    with pytest.raises(BatchError, match="already exists"):
        run_simulation(simulation, steps=0, dt=0.1, output=output)
    assert output.read_bytes() == original_bytes

    second, second_provenance = build_model(
        model,
        ModelContext(BackendKind.CPU, 0, 1234, {"length": 4.0, "cell_type": 7}),
    )
    assert isinstance(second, Simulation)
    second_output = tmp_path / "second.cm2.json"
    run_simulation(
        second,
        steps=3,
        dt=0.1,
        output=second_output,
        provenance=second_provenance,
    )
    assert _document(second_output)["simulation"] == document["simulation"]


def test_periodic_checkpoint_preserves_legacy_suffix(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    _write_model(model)
    simulation, provenance = build_model(model, ModelContext(BackendKind.CPU, 0, seed=7))
    assert isinstance(simulation, Simulation)

    summary = run_simulation(
        simulation,
        steps=1,
        dt=0.1,
        output=tmp_path / "legacy.cm2.json",
        checkpoint_every=1,
        provenance=provenance,
    )

    assert summary.periodic_checkpoints == (tmp_path / "legacy.step-00000001.cm2.json",)


def test_native_controller_resumes_exact_runtime_state(tmp_path: Path) -> None:
    model = tmp_path / "controller_model.py"
    _write_controller_model(model)
    context = ModelContext(BackendKind.CPU, 0, seed=8128)
    controller, provenance = build_model(model, context)
    assert isinstance(controller, SimulationController)

    uninterrupted = tmp_path / "uninterrupted.cm2.json"
    run_simulation(
        controller,
        steps=6,
        dt=0.125,
        output=uninterrupted,
        provenance=provenance,
    )

    first = tmp_path / "first.cm2.json"
    controller, provenance = build_model(
        model,
        ModelContext(BackendKind.CPU, 0, seed=8128),
    )
    run_simulation(controller, steps=3, dt=0.125, output=first, provenance=provenance)
    resumed = tmp_path / "resumed.cm2.json"
    assert (
        main(
            [
                "run",
                "--model",
                str(model),
                "--resume",
                str(first),
                "--steps",
                "3",
                "--dt",
                "0.125",
                "--output",
                str(resumed),
                "--quiet",
            ]
        )
        == 0
    )

    expected = load_checkpoint_bundle(uninterrupted)
    actual = load_checkpoint_bundle(resumed)
    assert _document(resumed)["simulation"] == _document(uninterrupted)["simulation"]
    assert actual.controller == expected.controller
    assert actual.simulation.time == expected.simulation.time


def test_native_controller_resume_checks_source_before_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "controller_model.py"
    _write_controller_model(model)
    controller, provenance = build_model(model, ModelContext(BackendKind.CPU, 0, seed=3))
    checkpoint = tmp_path / "controller.cm2.json"
    run_simulation(controller, steps=1, dt=0.1, output=checkpoint, provenance=provenance)

    marker = tmp_path / "executed.txt"
    model.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    output = tmp_path / "must-not-exist.cm2.json"
    assert (
        main(
            [
                "run",
                "--model",
                str(model),
                "--resume",
                str(checkpoint),
                "--steps",
                "1",
                "--dt",
                "0.1",
                "--output",
                str(output),
                "--quiet",
            ]
        )
        == 2
    )
    assert "digest does not match checkpoint" in capsys.readouterr().err
    assert not marker.exists()
    assert not output.exists()


def test_controller_cannot_silently_checkpoint_null_state(tmp_path: Path) -> None:
    class InvalidController:
        def __init__(self) -> None:
            self.simulation = Simulation(BackendKind.CPU)

        def step(self, dt: float) -> None:
            self.simulation.step(dt)

        def controller_state(self) -> None:
            return None

    with pytest.raises(BatchError, match="non-null"):
        run_simulation(
            InvalidController(),
            steps=0,
            dt=0.1,
            output=tmp_path / "invalid.cm2.json",
        )


def test_cell_count_threshold_can_finish_before_the_first_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "model.py"
    _write_model(model)
    simulation, provenance = build_model(
        model,
        ModelContext(BackendKind.CPU, 0, seed=5),
    )
    output = tmp_path / "already-large-enough.cm2.json"
    progress: list[int] = []

    summary = run_simulation(
        simulation,
        steps=100,
        dt=0.1,
        output=output,
        checkpoint_every=10,
        stop_cell_count=1,
        provenance=provenance,
        progress=lambda value: progress.append(value.completed_steps),
    )

    assert summary.completed_steps == 0
    assert summary.stop_reason == "cell_count"
    assert summary.cell_count_threshold == 1
    assert summary.periodic_checkpoints == ()
    assert progress == []
    assert load_checkpoint(output).time == 0.0
    assert _document(output)["provenance"]["run"] == {
        "completed_steps": 0,
        "dt": 0.1,
        "requested_steps": 100,
        "status": "complete",
        "stop_reason": "cell_count",
        "stopping": {"cell_count": 1, "maximum_steps": 100},
    }

    cli_output = tmp_path / "cli-threshold.cm2.json"
    assert (
        main(
            [
                "run",
                "--model",
                str(model),
                "--steps",
                "100",
                "--dt",
                "0.1",
                "--stop-cell-count",
                "1",
                "--output",
                str(cli_output),
                "--quiet",
            ]
        )
        == 0
    )
    assert "steps=0" in capsys.readouterr().out


@pytest.mark.parametrize("threshold", [0, -1, True, 1 << 64])
def test_cell_count_threshold_must_be_positive_uint64(tmp_path: Path, threshold: int) -> None:
    simulation = ModelContext(BackendKind.CPU, 0, seed=0).simulation()
    with pytest.raises(BatchError, match="positive uint64"):
        run_simulation(
            simulation,
            steps=1,
            dt=0.1,
            output=tmp_path / "invalid.cm2.json",
            stop_cell_count=threshold,
        )


def test_cli_runs_models_resumes_and_lists_devices(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "model.py"
    _write_model(model)
    first = tmp_path / "first.cm2.json"
    status = main(
        [
            "run",
            "--model",
            str(model),
            "--backend",
            "cpu",
            "--device-index",
            "0",
            "--seed",
            "99",
            "--parameter",
            "length=3.5",
            "--steps",
            "2",
            "--dt",
            "0.25",
            "--output",
            str(first),
            "--quiet",
        ]
    )
    assert status == 0
    assert first.exists()
    assert "wrote" in capsys.readouterr().out

    resumed = tmp_path / "resumed.cm2.json"
    status = main(
        [
            "run",
            "--resume",
            str(first),
            "--steps",
            "2",
            "--dt",
            "0.25",
            "--output",
            str(resumed),
            "--quiet",
        ]
    )
    assert status == 0
    assert math.isclose(load_checkpoint(resumed).time, 1.0)
    resume_document = _document(resumed)
    assert resume_document["provenance"]["resume"]["sha256"] == hashlib.sha256(
        first.read_bytes()
    ).hexdigest()
    capsys.readouterr()

    status = main(["devices", "--json"])
    assert status == 0
    devices = cast(list[dict[str, Any]], json.loads(capsys.readouterr().out))
    cpu = next(record for record in devices if record["backend"] == "cpu")
    assert cpu == {
        "available": True,
        "backend": "cpu",
        "devices": [{"index": 0, "name": "host"}],
    }


def test_cli_rejects_bad_parameters_without_creating_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "model.py"
    _write_model(model)
    output = tmp_path / "run.cm2.json"
    status = main(
        [
            "run",
            "--model",
            str(model),
            "--parameter",
            "length=NaN",
            "--steps",
            "1",
            "--dt",
            "0.1",
            "--output",
            str(output),
            "--quiet",
        ]
    )
    assert status == 2
    assert "contains NaN" in capsys.readouterr().err
    assert not output.exists()
