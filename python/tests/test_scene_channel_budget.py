from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import rfc8785
from microsimulator import (
    MAX_SCENE_CHANNELS,
    ChannelMetadata,
    SceneBackend,
    SceneConstraints,
    SceneError,
    SceneFrame,
    SceneGridBoundary,
    SceneSignalGrid,
    Simulation,
    capture_scene,
    dumps_scene,
    load_scene,
    parse_scene,
    save_scene,
)


def _frame(species_count: int = 0, signal_count: int = 1) -> SceneFrame:
    boundary = SceneGridBoundary("no_flux", ())
    grid = SceneSignalGrid(
        signal_count,
        (1, 1, 1),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        boundary,
        boundary,
        boundary,
        boundary,
        boundary,
        boundary,
        (0.0,) * signal_count,
    )
    return SceneFrame(
        0.0,
        SceneBackend("cpu", "CPU reference", "host", 0, False),
        species_count,
        (),
        SceneConstraints((), (), (), ()),
        grid,
    )


def _forbid_expansion(
    self: ChannelMetadata, species_count: int, signal_count: int
) -> ChannelMetadata:
    raise AssertionError("scene count must be bounded before label expansion")


@pytest.mark.parametrize("version", [2, 3])
def test_scene_channel_budget_boundary_roundtrips(version: int, tmp_path: Path) -> None:
    # Each group independently accepts the inclusive limit, including a scene
    # without cells that still needs species selector labels.
    frame = _frame(MAX_SCENE_CHANNELS, MAX_SCENE_CHANNELS)
    document = json.loads(dumps_scene(frame))
    document["version"] = version
    if version == 2:
        del document["frame"]["channel_metadata"]
    document["integrity"]["frame"] = hashlib.sha256(rfc8785.dumps(document["frame"])).hexdigest()
    encoded = json.dumps(document)
    assert parse_scene(encoded) == frame
    path = tmp_path / "boundary.scene.json"
    path.write_text(encoded)
    assert load_scene(path) == frame
    save_scene(frame, path)
    assert load_scene(path) == frame
    assert frame.channel_metadata == ChannelMetadata(
        species=(None,) * MAX_SCENE_CHANNELS, signals=(None,) * MAX_SCENE_CHANNELS
    )


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("group", ["species", "signals"])
@pytest.mark.parametrize("count", [MAX_SCENE_CHANNELS + 1, (1 << 32) - 1])
def test_scene_reader_rejects_tiny_oversized_claim_before_expansion(
    version: int, group: str, count: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = json.loads(dumps_scene(_frame()))
    document["version"] = version
    if version == 2:
        del document["frame"]["channel_metadata"]
    if group == "species":
        document["frame"]["species_count"] = count
    else:
        document["frame"]["signal_grid"]["signal_count"] = count
    document["integrity"]["frame"] = hashlib.sha256(rfc8785.dumps(document["frame"])).hexdigest()
    encoded = json.dumps(document)
    assert len(encoded) < 2000
    monkeypatch.setattr(ChannelMetadata, "resolved", _forbid_expansion)
    with pytest.raises(SceneError, match="scene presentation channel budget of 4096"):
        parse_scene(encoded)
    path = tmp_path / "oversized.scene.json"
    path.write_text(encoded)
    with pytest.raises(SceneError, match="scene presentation channel budget of 4096"):
        load_scene(path)


@pytest.mark.parametrize("group", ["species", "signals"])
@pytest.mark.parametrize("count", [MAX_SCENE_CHANNELS + 1, (1 << 32) - 1])
def test_construct_capture_and_write_bound_counts_before_copy_or_expansion(
    group: str, count: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frame()
    assert frame.signal_grid is not None
    grid = (
        replace(frame.signal_grid, signal_count=count) if group == "signals" else frame.signal_grid
    )
    species_count = count if group == "species" else 0
    monkeypatch.setattr(ChannelMetadata, "resolved", _forbid_expansion)
    with pytest.raises(SceneError, match="scene presentation channel budget of 4096"):
        replace(frame, species_count=species_count, signal_grid=grid)

    # No _checkpoint method: rejection must precede any native-state copy.
    simulation = cast(
        Simulation, SimpleNamespace(species_count=species_count, signal_count=grid.signal_count)
    )
    with pytest.raises(SceneError, match="scene presentation channel budget of 4096"):
        capture_scene(simulation)

    # Encoders independently validate even objects restored without __init__.
    object.__setattr__(frame, "species_count", species_count)
    object.__setattr__(frame, "signal_grid", grid)
    with pytest.raises(SceneError, match="scene presentation channel budget of 4096"):
        dumps_scene(frame)
    path = tmp_path / "rejected.scene.json"
    with pytest.raises(SceneError, match="scene presentation channel budget of 4096"):
        save_scene(frame, path)
    assert not path.exists()
