from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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
    SignalGridSpec,
    Simulation,
    Vec3,
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


def _resigned(document: dict[str, Any]) -> str:
    document["integrity"]["frame"] = hashlib.sha256(rfc8785.dumps(document["frame"])).hexdigest()
    return json.dumps(document)


def test_scene_channel_budget_boundary_roundtrips(tmp_path: Path) -> None:
    # Each group independently accepts the inclusive limit, including a scene
    # without cells that still needs species selector labels.
    frame = _frame(MAX_SCENE_CHANNELS, MAX_SCENE_CHANNELS)
    encoded = _resigned(json.loads(dumps_scene(frame)))
    assert parse_scene(encoded) == frame
    path = tmp_path / "boundary.scene.json"
    path.write_text(encoded)
    assert load_scene(path) == frame
    save_scene(frame, path)
    assert load_scene(path) == frame
    assert frame.channel_metadata == ChannelMetadata(
        species=(None,) * MAX_SCENE_CHANNELS, signals=(None,) * MAX_SCENE_CHANNELS
    )


@pytest.mark.parametrize("group", ["species", "signals"])
@pytest.mark.parametrize("count", [MAX_SCENE_CHANNELS + 1, (1 << 32) - 1])
def test_scene_reader_rejects_tiny_oversized_claim(group: str, count: int, tmp_path: Path) -> None:
    # The declared count alone is rejected: the document stays small and never
    # supplies the labels or levels the claim would imply.
    document = json.loads(dumps_scene(_frame()))
    if group == "species":
        document["frame"]["species_count"] = count
    else:
        document["frame"]["signal_grid"]["signal_count"] = count
    encoded = _resigned(document)
    assert len(encoded) < 2000
    with pytest.raises(SceneError, match="scene presentation channel budget of 4096"):
        parse_scene(encoded)
    path = tmp_path / "oversized.scene.json"
    path.write_text(encoded)
    with pytest.raises(SceneError, match="scene presentation channel budget of 4096"):
        load_scene(path)


@pytest.mark.parametrize("group", ["species", "signals"])
@pytest.mark.parametrize("count", [MAX_SCENE_CHANNELS + 1, (1 << 32) - 1])
def test_constructing_a_frame_bounds_channel_counts(group: str, count: int) -> None:
    frame = _frame()
    assert frame.signal_grid is not None
    grid = frame.signal_grid
    if group == "signals":
        grid = grid.replace(signal_count=count)
    species_count = count if group == "species" else 0
    with pytest.raises(SceneError, match="scene presentation channel budget of 4096"):
        frame.replace(species_count=species_count, signal_grid=grid)


@pytest.mark.parametrize("group", ["species", "signals"])
def test_capture_bounds_channel_counts(group: str) -> None:
    count = MAX_SCENE_CHANNELS + 1
    if group == "species":
        simulation = Simulation(species_count=count)
    else:
        simulation = Simulation()
        spec = SignalGridSpec()
        spec.signal_count = count
        spec.diffusion = [0.0] * count
        spec.advection = [Vec3() for _ in range(count)]
        simulation.configure_signal_grid(spec, [0.0] * count)
    with pytest.raises(SceneError, match="scene presentation channel budget of 4096"):
        capture_scene(simulation)
