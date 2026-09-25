from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import rfc8785
from microsimulator import (
    CellInit,
    ChannelMetadata,
    GridShape,
    SignalGridSpec,
    Simulation,
    Vec3,
    load_scene,
    save_checkpoint,
)
from microsimulator.cli import main
from microsimulator.replay import ReplayExportError, export_replay


def lifecycle_checkpoints(directory: Path) -> list[Path]:
    """Native growth/division/removal; names deliberately oppose lexical ordering."""
    simulation = Simulation(species_count=1)
    cell = CellInit()
    cell.length = 2.0
    cell.growth_rate = 0.5
    cell.species = [0.25]
    parent = simulation.add_cell(cell)
    paths: list[Path] = []
    for ordinal, name in enumerate(
        ("z-start.json", "a-growth.json", "m-division.json", "b-removal.json")
    ):
        if ordinal == 1:
            simulation.step(0.2)
        elif ordinal == 2:
            simulation.divide_equal(parent)
        elif ordinal == 3:
            simulation.remove_cell(2)
            simulation.step(0.1)
        path = directory / name
        save_checkpoint(
            simulation,
            path,
            channel_metadata=ChannelMetadata(species=("Reporter",)),
            provenance={"model": {"path": "/missing/model-that-must-not-be-imported.py"}},
        )
        paths.append(path)
    return paths


def test_ordered_export_preserves_topology_labels_equal_times_and_input_files(
    tmp_path: Path,
) -> None:
    paths = lifecycle_checkpoints(tmp_path)
    before = [path.read_bytes() for path in paths]
    summary = export_replay(paths, tmp_path / "bundle")
    assert summary.frame_count == 4
    manifest = json.loads((summary.output / "manifest.json").read_text())
    assert manifest["format"] == "microsimulator-replay"
    assert manifest["version"] == 1
    recording = manifest["recording"]
    assert (
        hashlib.sha256(rfc8785.dumps(recording)).hexdigest() == manifest["integrity"]["recording"]
    )
    frames = [load_scene(summary.output / entry["file"]) for entry in recording["frames"]]
    assert [len(frame.cells) for frame in frames] == [1, 1, 2, 1]
    assert frames[1].cells[0].length > frames[0].cells[0].length
    assert [cell.id for cell in frames[2].cells] == [2, 3]
    assert frames[3].cells[0].id == 3
    assert frames[2].time == frames[1].time
    for ordinal, (entry, source) in enumerate(zip(recording["frames"], paths, strict=True)):
        assert entry["ordinal"] == ordinal
        assert entry["checkpoint_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        encoded = (summary.output / entry["file"]).read_bytes()
        assert entry["bytes"] == len(encoded)
        assert entry["sha256"] == hashlib.sha256(encoded).hexdigest()
        assert frames[ordinal].channel_metadata.species == ("Reporter",)
    assert [path.read_bytes() for path in paths] == before


def test_source_backend_preserved_while_exporting_without_original_device(tmp_path: Path) -> None:
    paths = lifecycle_checkpoints(tmp_path)
    # This is a provenance fixture, not evidence of a CUDA simulation run.
    source = cast(dict[str, Any], json.loads(paths[0].read_text()))
    source["source_backend"] = {
        "kind": "cuda",
        "name": "Recorded GPU",
        "device": "Unavailable GPU",
        "device_index": 7,
        "native": True,
    }
    paths[0].write_text(json.dumps(source))
    result = export_replay(paths[:1], tmp_path / "cpu-export")
    manifest = json.loads((result.output / "manifest.json").read_text())
    assert manifest["recording"]["export_backend"]["kind"] == "cpu"
    entry = manifest["recording"]["frames"][0]
    assert entry["source_backend"] == source["source_backend"]
    scene = load_scene(result.output / entry["file"])
    assert scene.backend.kind == "cuda"
    assert scene.backend.device_index == 7


def test_failure_is_attributed_to_ordinal_and_leaves_no_partial_bundle(tmp_path: Path) -> None:
    paths = lifecycle_checkpoints(tmp_path)
    output = tmp_path / "bad-order"
    with pytest.raises(ReplayExportError, match=r"frame 1.*precedes previous"):
        export_replay([paths[1], paths[0]], output)
    assert not output.exists()
    assert not list(tmp_path.glob(".bad-order.*"))
    paths[2].write_text("invalid")
    with pytest.raises(ReplayExportError, match=r"frame 2.*not valid"):
        export_replay(paths, output)
    assert not output.exists()
    with pytest.raises(ReplayExportError, match=r"frame 0.*missing.json"):
        export_replay([tmp_path / "missing.json"], output)
    output.mkdir()
    with pytest.raises(ReplayExportError, match="output already exists"):
        export_replay(paths[:1], output)
    with pytest.raises(ReplayExportError, match="ordered checkpoints"):
        export_replay([], tmp_path / "empty")


def test_export_rejects_checkpoint_digest_tampering(tmp_path: Path) -> None:
    paths = lifecycle_checkpoints(tmp_path)
    document = json.loads(paths[0].read_text())
    document["simulation"]["time"] = 100
    paths[0].write_text(json.dumps(document))
    with pytest.raises(ReplayExportError, match="state digest does not match"):
        export_replay(paths, tmp_path / "bad")


def test_cli_exports_exact_argument_order_and_reports_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = lifecycle_checkpoints(tmp_path)
    output = tmp_path / "cli"
    assert main(["export-replay", *(str(path) for path in paths), "--output", str(output)]) == 0
    assert "frames=4" in capsys.readouterr().out
    assert (
        main(
            ["export-replay", str(paths[1]), str(paths[0]), "--output", str(tmp_path / "backwards")]
        )
        == 2
    )
    assert "frame 1" in capsys.readouterr().err


def test_signal_grid_changes_are_independent_frames(tmp_path: Path) -> None:
    paths: list[Path] = []
    for ordinal, size in enumerate((None, 1, 3)):
        simulation = Simulation(species_count=0)
        if size is not None:
            shape = GridShape()
            shape.x, shape.y, shape.z = size, size, size
            spec = SignalGridSpec()
            spec.signal_count = 2
            spec.shape = shape
            spec.spacing = Vec3(1, 1, 1)
            spec.diffusion = [0, 0]
            spec.advection = [Vec3(), Vec3()]
            simulation.configure_signal_grid(spec, [1.0] * (2 * size**3))
        path = tmp_path / f"{ordinal}.json"
        save_checkpoint(simulation, path)
        paths.append(path)
    result = export_replay(paths, tmp_path / "grid")
    frames = [
        load_scene(result.output / f"frames/{ordinal:08d}.scene.json") for ordinal in range(3)
    ]
    assert frames[0].signal_grid is None
    assert frames[1].signal_grid is not None and frames[1].signal_grid.shape == (1, 1, 1)
    assert frames[2].signal_grid is not None and frames[2].signal_grid.shape == (3, 3, 3)


def test_export_bounds_and_parent_io_failures_are_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import microsimulator.replay as replay

    paths = lifecycle_checkpoints(tmp_path)
    parent = tmp_path / "file-parent"
    parent.write_text("preserve")
    with pytest.raises(ReplayExportError, match="could not prepare replay destination"):
        export_replay(paths, parent / "bundle")
    assert parent.read_text() == "preserve"
    monkeypatch.setattr(replay, "MAX_REPLAY_MANIFEST_BYTES", 10)
    with pytest.raises(ReplayExportError, match="manifest exceeds"):
        export_replay(paths, tmp_path / "bounded")
    assert not (tmp_path / "bounded").exists()
    assert not list(tmp_path.glob(".bounded.*"))
    monkeypatch.setattr(replay, "MAX_CHECKPOINT_BYTES", 10)
    with pytest.raises(ReplayExportError, match=r"frame 0.*checkpoint exceeds"):
        export_replay(paths, tmp_path / "input-limit")
