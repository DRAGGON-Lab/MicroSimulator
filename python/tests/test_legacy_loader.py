from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import cast

import pytest
from microsimulator import (
    BackendKind,
    LegacyCompatibilityError,
    ModelContext,
    backend_available,
    build_legacy_model,
    load_checkpoint_bundle,
    resume_legacy_model,
    run_simulation,
)
from microsimulator.checkpoint import JSONValue
from microsimulator.cli import main

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("backend", list(BackendKind))
def test_unchanged_growth_model_loads_through_setup_facade(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    context = ModelContext(backend, 0, seed=42)
    model, provenance = build_legacy_model(_FIXTURES / "legacy_growth.py", context)

    assert list(model.cells) == [1]
    founder = model.cells[1]
    assert founder.cellType == 2
    assert founder.color.tolist() == [0.1, 0.2, 0.3]
    assert 4.1 <= founder.targetVol <= 4.2
    model_provenance = cast(dict[str, JSONValue], provenance["model"])
    assert model_provenance["compatibility"] == "legacy-python-callbacks-v1"

    model.step(0.2)
    model.step(0.0)
    assert list(model.cells) == [2, 3]
    assert all(cell.dir[2] == 0.0 for cell in model.cells.values())
    assert model.simulation.lineage_parent(2) == 1
    assert model.simulation.lineage_parent(3) == 1


def test_legacy_loader_rejects_opencl_integrators_explicitly() -> None:
    context = ModelContext(BackendKind.CPU, 0, seed=0)
    with pytest.raises(LegacyCompatibilityError, match="OpenCL integrators"):
        build_legacy_model(_FIXTURES / "legacy_opencl_integrator.py", context)


def test_legacy_batch_checkpoint_resumes_exactly_and_checks_source(tmp_path: Path) -> None:
    model_path = _FIXTURES / "legacy_growth.py"
    context = ModelContext(BackendKind.CPU, 0, seed=42)
    uninterrupted, uninterrupted_provenance = build_legacy_model(model_path, context)
    uninterrupted_path = tmp_path / "uninterrupted.cm2.json"
    run_simulation(
        uninterrupted,
        steps=3,
        dt=0.2,
        output=uninterrupted_path,
        provenance=uninterrupted_provenance,
    )

    first_path = tmp_path / "first.cm2.json"
    assert (
        main(
            [
                "run",
                "--legacy-model",
                str(model_path),
                "--seed",
                "42",
                "--steps",
                "2",
                "--dt",
                "0.2",
                "--output",
                str(first_path),
                "--quiet",
            ]
        )
        == 0
    )
    resumed_path = tmp_path / "resumed.cm2.json"
    assert (
        main(
            [
                "run",
                "--legacy-model",
                str(model_path),
                "--resume",
                str(first_path),
                "--steps",
                "1",
                "--dt",
                "0.2",
                "--output",
                str(resumed_path),
                "--quiet",
            ]
        )
        == 0
    )

    expected = load_checkpoint_bundle(uninterrupted_path)
    actual = load_checkpoint_bundle(resumed_path)
    assert actual.controller == expected.controller
    assert cast(dict[str, JSONValue], actual.controller)["version"] == 4
    for left, right in zip(
        actual.simulation.cells(), expected.simulation.cells(), strict=True
    ):
        assert left.id == right.id
        assert left.slot == right.slot
        assert left.position.x == right.position.x
        assert left.position.y == right.position.y
        assert left.position.z == right.position.z
        assert left.direction.x == right.direction.x
        assert left.direction.y == right.direction.y
        assert left.direction.z == right.direction.z
        assert left.length == right.length

    changed_model = tmp_path / "changed.py"
    copyfile(model_path, changed_model)
    changed_model.write_text(
        changed_model.read_text(encoding="utf-8") + "\nCHANGED = True\n",
        encoding="utf-8",
    )
    first_bundle = load_checkpoint_bundle(first_path)
    with pytest.raises(LegacyCompatibilityError, match="digest"):
        resume_legacy_model(changed_model, context, first_bundle)


def test_legacy_batch_stops_when_division_reaches_cell_threshold(tmp_path: Path) -> None:
    model, provenance = build_legacy_model(
        _FIXTURES / "legacy_growth.py",
        ModelContext(BackendKind.CPU, 0, seed=42),
    )
    output = tmp_path / "cell-threshold.cm2.json"

    summary = run_simulation(
        model,
        steps=100,
        dt=0.2,
        output=output,
        checkpoint_every=1,
        stop_cell_count=2,
        provenance=provenance,
    )

    assert summary.completed_steps == 2
    assert summary.stop_reason == "cell_count"
    assert summary.cell_count == 2
    assert summary.periodic_checkpoints == (
        tmp_path / "cell-threshold.step-00000001.cm2.json",
        tmp_path / "cell-threshold.step-00000002.cm2.json",
    )
    run = cast(dict[str, JSONValue], load_checkpoint_bundle(output).provenance["run"])
    assert run["completed_steps"] == 2
    assert run["requested_steps"] == 100
    assert run["stop_reason"] == "cell_count"


def test_legacy_loader_accepts_alternating_division_axes() -> None:
    context = ModelContext(BackendKind.CPU, 0, seed=7)
    model, _ = build_legacy_model(_FIXTURES / "legacy_alternating.py", context)

    controller = model.controller_state()
    options = cast(dict[str, JSONValue], controller["options"])
    assert controller["version"] == 4
    assert options["alternate_divisions"] is True
    assert options["division_jitter_z"] is None
    assert options["max_substeps"] == 8
