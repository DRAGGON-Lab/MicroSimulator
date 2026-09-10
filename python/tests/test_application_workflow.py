from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from microsimulator import (
    BackendKind,
    CheckpointBundle,
    ModelContext,
    NativeController,
    PlaneConstraintInit,
    Vec3,
    backend_device_count,
    build_model,
    load_checkpoint_bundle,
    run_simulation,
)
from microsimulator.analysis import export_dataset, open_dataset
from microsimulator.checkpoint import JSONValue
from microsimulator.viewer_server import LiveSession

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL = _PROJECT_ROOT / "examples" / "native_controller.py"
_PARAMETERS: dict[str, JSONValue] = {
    "initial_length": 3.95,
    "division_length": 4.0,
}


def _targets() -> tuple[tuple[BackendKind, int], ...]:
    return tuple(
        (backend, device_index)
        for backend in BackendKind
        for device_index in range(backend_device_count(backend))
    )


def _build(
    backend: BackendKind,
    device_index: int,
    *,
    checkpoint: CheckpointBundle | None = None,
) -> tuple[NativeController, dict[str, JSONValue]]:
    model, provenance = build_model(
        _MODEL,
        ModelContext(
            backend,
            device_index,
            seed=1729,
            parameters=_PARAMETERS,
        ),
        checkpoint=checkpoint,
    )
    assert isinstance(model, NativeController)
    if checkpoint is None:
        plane = PlaneConstraintInit()
        plane.point = Vec3(0.0, -0.25, 0.0)
        plane.inward_normal = Vec3(0.0, 1.0, 0.0)
        model.simulation.add_plane_constraint(plane)
    return model, provenance


def _simulation_state(path: Path) -> object:
    document = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    return document["simulation"]


_APPLICATION_TARGETS = _targets()


@pytest.mark.parametrize(
    ("backend", "device_index"),
    _APPLICATION_TARGETS,
    ids=[
        f"{backend.name.lower()}-{device_index}" for backend, device_index in _APPLICATION_TARGETS
    ],
)
def test_application_workflow(
    backend: BackendKind,
    device_index: int,
    tmp_path: Path,
) -> None:
    uninterrupted_model, uninterrupted_provenance = _build(backend, device_index)
    uninterrupted = tmp_path / "uninterrupted.cm2.json"
    run_simulation(
        uninterrupted_model,
        steps=4,
        dt=0.25,
        output=uninterrupted,
        provenance=uninterrupted_provenance,
    )

    split_model, split_provenance = _build(backend, device_index)
    midpoint = tmp_path / "midpoint.cm2.json"
    run_simulation(
        split_model,
        steps=2,
        dt=0.25,
        output=midpoint,
        provenance=split_provenance,
    )
    midpoint_bundle = load_checkpoint_bundle(
        midpoint,
        backend=backend,
        device_index=device_index,
    )
    resumed_model, resumed_provenance = _build(
        backend,
        device_index,
        checkpoint=midpoint_bundle,
    )
    resumed = tmp_path / "resumed.cm2.json"
    run_simulation(
        resumed_model,
        steps=2,
        dt=0.25,
        output=resumed,
        provenance=resumed_provenance,
    )

    expected = load_checkpoint_bundle(uninterrupted)
    actual = load_checkpoint_bundle(resumed)
    assert _simulation_state(resumed) == _simulation_state(uninterrupted)
    assert actual.controller == expected.controller
    assert actual.source_backend.kind == backend.name.lower()
    assert actual.source_backend.device_index == device_index

    live_checkpoint = tmp_path / "live.cm2.json"

    def factory() -> tuple[NativeController, dict[str, JSONValue]]:
        return _build(backend, device_index)

    session = LiveSession(factory, dt=0.25, checkpoint_output=live_checkpoint)
    initial = session.frame_message(playing=False)
    session.step(2)
    stepped = session.frame_message(playing=False)
    session.checkpoint()
    session.reset()
    reset = session.frame_message(playing=False)
    initial_scene = cast(dict[str, Any], initial["scene"])
    stepped_scene = cast(dict[str, Any], stepped["scene"])
    reset_scene = cast(dict[str, Any], reset["scene"])
    assert initial_scene["frame"]["backend"]["kind"] == backend.name.lower()
    assert stepped_scene["frame"]["time"] == 0.5
    assert stepped_scene["frame"]["cells"] != initial_scene["frame"]["cells"]
    assert reset_scene["frame"] == initial_scene["frame"]
    assert load_checkpoint_bundle(live_checkpoint).controller is not None

    dataset_path = tmp_path / "application.cm2.dataset"
    summary = export_dataset(
        [midpoint, resumed],
        dataset_path,
        backend=backend,
        device_index=device_index,
        include_contacts=True,
        include_external_contacts=True,
    )
    dataset = open_dataset(dataset_path)
    options = cast(dict[str, object], dataset.manifest["options"])
    assert dataset.verified
    assert options["backend"] == backend.name.lower()
    assert options["device_index"] == device_index
    assert options["contact_conformance"] == (
        "cpu_reference" if backend == BackendKind.CPU else "hardware_conformant"
    )
    assert summary.frame_count == 2
    assert summary.contact_rows > 0
    assert summary.external_contact_rows > 0
