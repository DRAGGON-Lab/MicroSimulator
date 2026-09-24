from __future__ import annotations

# ruff: noqa: RUF001 -- explicit Unicode-label coverage.
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import rfc8785
from microsimulator import (
    BackendKind,
    ChannelMetadata,
    ChannelMetadataError,
    CheckpointError,
    ModelContext,
    NativeController,
    SceneError,
    Simulation,
    build_model,
    capture_scene,
    dumps_scene,
    load_checkpoint,
    load_checkpoint_bundle,
    load_scene,
    parse_scene,
    run_simulation,
    save_checkpoint,
    save_scene,
)
from microsimulator.runner import BatchError, model_channel_metadata
from microsimulator.viewer_server import LiveSession

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/named_channels.py"
LABELS = ChannelMetadata(
    species=("Green reporter", "Red reporter"), signals=("Nutrient", "Extracellular cue")
)


def _build() -> tuple[NativeController, dict[str, Any]]:
    model, provenance = build_model(EXAMPLE, ModelContext(BackendKind.CPU, 0, 17))
    assert isinstance(model, NativeController)
    return model, provenance


def test_named_model_periodic_checkpoint_resume_and_standalone_export(tmp_path: Path) -> None:
    model, provenance = _build()
    output = tmp_path / "named.json"
    summary = run_simulation(
        model, steps=2, dt=0.01, output=output, checkpoint_every=1, provenance=provenance
    )
    for path in (*summary.periodic_checkpoints, output):
        assert load_checkpoint_bundle(path).channel_metadata == LABELS
    bundle = load_checkpoint_bundle(output)
    resumed, provenance = build_model(
        EXAMPLE, ModelContext(BackendKind.CPU, 0, 17), checkpoint=bundle
    )
    assert model_channel_metadata(resumed) == LABELS
    resumed_output = tmp_path / "continued.json"
    run_simulation(resumed, steps=1, dt=0.01, output=resumed_output, provenance=provenance)
    continued = load_checkpoint_bundle(resumed_output)
    assert continued.channel_metadata == LABELS
    model.step(0.01)
    assert continued.simulation.signal_levels == model.simulation.signal_levels
    assert continued.simulation.cell(1).species == model.simulation.cell(1).species
    # Only data and the bundle API are needed to export; no model import/execute.
    destination = tmp_path / "named.scene.json"
    save_scene(
        capture_scene(bundle.simulation, channel_metadata=bundle.channel_metadata), destination
    )
    assert load_scene(destination).channel_metadata == LABELS


def test_live_labels_survive_step_reset_and_checkpoint(tmp_path: Path) -> None:
    session = LiveSession(_build, dt=0.01, checkpoint_output=tmp_path / "live.json")
    for operation in (lambda: None, session.step, session.reset):
        operation()
        message = session.frame_message(playing=False)
        frame = parse_scene(json.dumps(message["scene"]))
        assert frame.channel_metadata == LABELS
    assert load_checkpoint_bundle(session.checkpoint()).channel_metadata == LABELS


def test_metadata_counts_fail_before_stepping_or_writing(tmp_path: Path) -> None:
    model, _ = _build()
    model.channel_metadata = ChannelMetadata(species=("only one",))
    with pytest.raises(BatchError, match=r"species: expected 2 labels, got 1"):
        run_simulation(model, steps=1, dt=0.1, output=tmp_path / "bad.json")
    assert model.simulation.time == 0
    assert not (tmp_path / "bad.json").exists()
    with pytest.raises(ChannelMetadataError, match=r"species: expected 2 labels"):
        capture_scene(model.simulation, channel_metadata=model.channel_metadata)
    with pytest.raises(CheckpointError, match=r"species: expected 2 labels"):
        save_checkpoint(
            model.simulation, tmp_path / "bad.json", channel_metadata=model.channel_metadata
        )
    with pytest.raises(ChannelMetadataError, match=r"signals: expected 2 labels"):
        ChannelMetadata(signals=()).resolved(2, 2)


def test_missing_duplicate_unicode_empty_and_markup_labels_roundtrip(tmp_path: Path) -> None:
    model, _ = _build()
    labels = ChannelMetadata(species=("<b>α 🧪</b>", "<b>α 🧪</b>"), signals=(None, " "))
    save_checkpoint(model.simulation, tmp_path / "labels.json", channel_metadata=labels)
    bundle = load_checkpoint_bundle(tmp_path / "labels.json")
    assert bundle.channel_metadata == labels
    frame = capture_scene(bundle.simulation, channel_metadata=bundle.channel_metadata)
    assert parse_scene(dumps_scene(frame)).channel_metadata == labels
    with pytest.raises(CheckpointError, match="contains channel metadata"):
        load_checkpoint(tmp_path / "labels.json")
    with pytest.raises(ChannelMetadataError, match="invalid Unicode"):
        ChannelMetadata(species=("\ud800",))
    with pytest.raises(ChannelMetadataError, match="expected a string"):
        ChannelMetadata.from_json({"species": [7], "signals": []}, 1, 0)


def test_metadata_tampering_rejected_and_v8_migrates_only_after_verification(
    tmp_path: Path,
) -> None:
    model, _ = _build()
    path = tmp_path / "labels.json"
    save_checkpoint(model.simulation, path, channel_metadata=LABELS)
    document = json.loads(path.read_text())
    document["channel_metadata"]["species"][0] = "tampered"
    path.write_text(json.dumps(document))
    with pytest.raises(CheckpointError, match="channel metadata digest does not match"):
        load_checkpoint_bundle(path)
    document["version"] = 8
    del document["channel_metadata"]
    del document["integrity"]["channel_metadata"]
    path.write_text(json.dumps(document))
    assert load_checkpoint_bundle(path).channel_metadata == ChannelMetadata().resolved(2, 2)
    document["simulation"]["time"] = 999
    path.write_text(json.dumps(document))
    with pytest.raises(CheckpointError, match="state digest does not match"):
        load_checkpoint_bundle(path)


def test_scene_v2_verifies_original_payload_and_v3_rejects_invalid_labels() -> None:
    model, _ = _build()
    document = json.loads(dumps_scene(capture_scene(model.simulation, channel_metadata=LABELS)))
    document["version"] = 2
    del document["frame"]["channel_metadata"]
    document["integrity"]["frame"] = hashlib.sha256(rfc8785.dumps(document["frame"])).hexdigest()
    assert parse_scene(json.dumps(document)).channel_metadata == ChannelMetadata().resolved(2, 2)
    document["frame"]["time"] = 999
    with pytest.raises(SceneError, match="frame digest does not match"):
        parse_scene(json.dumps(document))
    document["version"] = 3
    document["frame"]["channel_metadata"] = {"species": [], "signals": [None, None]}
    document["integrity"]["frame"] = hashlib.sha256(rfc8785.dumps(document["frame"])).hexdigest()
    with pytest.raises(SceneError, match="species: expected 2 labels"):
        parse_scene(json.dumps(document))


def test_unnamed_native_model_and_closed_channel_schema() -> None:
    simulation = Simulation(species_count=2)
    assert model_channel_metadata(simulation) == ChannelMetadata(species=(None, None), signals=())
    with pytest.raises(ChannelMetadataError, match="exactly species and signals"):
        ChannelMetadata.from_json({"species": [], "signals": [], "extra": []}, 0, 0)


def test_resumed_model_cannot_silently_replace_persisted_labels(tmp_path: Path) -> None:
    model, provenance = _build()
    path = tmp_path / "labels.json"
    run_simulation(model, steps=0, dt=0.1, output=path, provenance=provenance)
    bundle = load_checkpoint_bundle(path)
    changed = replace(
        bundle,
        channel_metadata=ChannelMetadata(
            species=("Changed", "Red reporter"), signals=LABELS.signals
        ),
    )
    # Standard native restore treats the checkpoint's labels as authoritative.
    resumed, _ = build_model(EXAMPLE, ModelContext(BackendKind.CPU, 0, 17), checkpoint=changed)
    assert model_channel_metadata(resumed) == changed.channel_metadata


def test_shared_python_typescript_v3_fixture() -> None:
    frame = load_scene(ROOT / "viewer/tests/fixtures/channels-v3.scene.json")
    assert frame.channel_metadata.species == ("<b>α 🧪</b>", "<b>α 🧪</b>")
    assert frame.channel_metadata.signals == (None, " ")
