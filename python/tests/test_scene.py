from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import rfc8785
from microsimulator import (
    SCENE_FORMAT,
    SCENE_VERSION,
    BackendFeature,
    BackendKind,
    BoxConstraintInit,
    CellInit,
    ConstraintRegion,
    CylinderConstraintInit,
    GridBoundaryKind,
    GridShape,
    PlaneConstraintInit,
    SceneBackend,
    SceneCell,
    SceneConstraints,
    SceneError,
    SceneFrame,
    SignalGridSpec,
    Simulation,
    Vec3,
    backend_available,
    capture_scene,
    dumps_scene,
    load_scene,
    parse_scene,
    save_scene,
)


def _simulation(backend: BackendKind = BackendKind.CPU) -> Simulation:
    simulation = Simulation(backend, species_count=2)

    shape = GridShape()
    shape.x = 2
    shape.y = 2
    shape.z = 1
    grid = SignalGridSpec()
    grid.signal_count = 2
    grid.shape = shape
    grid.origin = Vec3(-1.0, -2.0, 0.5)
    grid.spacing = Vec3(0.5, 0.75, 1.25)
    grid.diffusion = [0.0, 0.0]
    grid.advection = [Vec3(), Vec3()]
    grid.x_lower.kind = GridBoundaryKind.FIXED
    grid.x_lower.values = [0.25, 0.5]
    simulation.configure_signal_grid(grid, [float(index) for index in range(8)])

    first = CellInit()
    first.position = Vec3(1.25, -2.5, 0.75)
    first.direction = Vec3(2.0, 1.0, 0.0)
    first.length = 4.5
    first.radius = 0.4
    first.growth_rate = 0.0
    first.cell_type = 7
    first.fixed = True
    first.species = [3.0, -0.25]
    parent = simulation.add_cell(first)

    second = CellInit()
    second.position = Vec3(-1.0, 2.0, 3.0)
    second.length = 2.0
    second.radius = 0.25
    second.growth_rate = 0.0
    second.cell_type = -4
    second.species = [1.0, 2.0]
    simulation.add_cell(second)

    plane = PlaneConstraintInit()
    plane.point = Vec3(0.0, 0.0, -1.0)
    plane.inward_normal = Vec3(0.0, 0.0, 2.0)
    plane.coefficient = 1.25
    assert simulation.add_plane_constraint(plane) == 1

    box = BoxConstraintInit()
    box.center = Vec3(4.0, -1.0, 0.5)
    box.half_extents = Vec3(1.5, 0.5, 2.0)
    box.coefficient = 0.75
    box.allowed_region = ConstraintRegion.OUTSIDE
    assert simulation.add_box_constraint(box) == 2

    dish = CylinderConstraintInit()
    dish.center = Vec3(0.0, 0.0, 1.0)
    dish.radius = 6.0
    dish.half_height = 2.0
    dish.coefficient = 1.0
    dish.allowed_region = ConstraintRegion.INSIDE
    assert simulation.add_cylinder_constraint(dish) == 3

    simulation.divide_equal(parent)
    simulation.step(0.125)
    return simulation


def _semantic_frame(frame: SceneFrame) -> SceneFrame:
    return replace(
        frame,
        backend=SceneBackend(
            kind="cpu",
            name="normalized",
            device="normalized",
            device_index=0,
            native=False,
        ),
    )


def _resign(document: dict[str, Any]) -> str:
    canonical = rfc8785.dumps(document["frame"])
    document["integrity"]["frame"] = hashlib.sha256(canonical).hexdigest()
    return json.dumps(document)


def test_capture_scene_is_backend_neutral_and_complete() -> None:
    reference = capture_scene(_simulation())
    assert reference.time == 0.125
    assert reference.species_count == 2
    assert [cell.slot for cell in reference.cells] == [0, 1, 2]
    assert [cell.id for cell in reference.cells] == [3, 2, 4]
    assert reference.cells[0].parent_id == 1
    assert reference.cells[2].parent_id == 1
    assert reference.cells[0].fixed
    assert reference.cells[0].species == (3.0, -0.25)
    assert math.isclose(
        sum(value * value for value in reference.cells[0].direction),
        1.0,
        abs_tol=1.0e-6,
    )

    constraints = reference.constraints
    assert [plane.id for plane in constraints.planes] == [1]
    assert constraints.planes[0].point == (0.0, 0.0, -1.0)
    assert math.isclose(constraints.planes[0].inward_normal[2], 1.0, abs_tol=1.0e-6)
    assert constraints.spheres == ()
    assert [box.id for box in constraints.boxes] == [2]
    assert constraints.boxes[0].center == (4.0, -1.0, 0.5)
    assert constraints.boxes[0].half_extents == (1.5, 0.5, 2.0)
    assert constraints.boxes[0].allowed_region == "outside"
    assert [cylinder.id for cylinder in constraints.cylinders] == [3]
    assert constraints.cylinders[0].radius == 6.0
    assert constraints.cylinders[0].allowed_region == "inside"

    grid = reference.signal_grid
    assert grid is not None
    assert grid.signal_count == 2
    assert grid.shape == (2, 2, 1)
    assert grid.origin == (-1.0, -2.0, 0.5)
    assert grid.spacing == (0.5, 0.75, 1.25)
    assert grid.x_lower.kind == "fixed"
    assert grid.x_lower.values == (0.25, 0.5)
    assert grid.levels == tuple(float(index) for index in range(8))

    for backend in BackendKind:
        if backend is BackendKind.CPU or not backend_available(backend):
            continue
        probe = Simulation(backend)
        if not probe.supports(BackendFeature.SIGNALS):
            continue
        assert _semantic_frame(capture_scene(_simulation(backend))) == _semantic_frame(reference)


@pytest.mark.parametrize("format_name", [SCENE_FORMAT, "cellmodeller2-scene"])
def test_scene_round_trip_is_exact_and_uses_decimal_identifiers(
    tmp_path: Path, format_name: str
) -> None:
    frame = capture_scene(_simulation())
    encoded = dumps_scene(frame)
    document = cast(dict[str, Any], json.loads(encoded))
    assert document["format"] == SCENE_FORMAT
    assert document["version"] == SCENE_VERSION
    assert [cell["id"] for cell in document["frame"]["cells"]] == ["3", "2", "4"]
    assert document["frame"]["cells"][0]["parent_id"] == "1"
    document["format"] = format_name
    assert parse_scene(json.dumps(document)) == frame
    document["frame"]["time"] += 1.0
    with pytest.raises(SceneError, match="digest"):
        parse_scene(json.dumps(document))

    path = tmp_path / "colony.cm2.scene.json"
    save_scene(frame, path)
    assert load_scene(path) == frame
    assert list(tmp_path.glob(".*.tmp")) == []


def test_scene_preserves_identifiers_outside_javascript_integer_range() -> None:
    identifier = (1 << 64) - 1
    frame = SceneFrame(
        time=0.0,
        backend=SceneBackend("cpu", "CPU reference", "host", 0, False),
        species_count=0,
        cells=(
            SceneCell(
                id=identifier,
                parent_id=identifier - 1,
                slot=0,
                position=(0.0, 0.0, 0.0),
                direction=(1.0, 0.0, 0.0),
                length=1.0,
                radius=0.5,
                growth_rate=0.0,
                cell_type=0,
                fixed=False,
                species=(),
            ),
        ),
        constraints=SceneConstraints(planes=(), spheres=(), boxes=(), cylinders=()),
        signal_grid=None,
    )
    encoded = dumps_scene(frame)
    assert f'"id": "{identifier}"' in encoded
    assert parse_scene(encoded) == frame


def test_scene_rejects_tampering_unknown_fields_and_duplicate_keys() -> None:
    document = cast(dict[str, Any], json.loads(dumps_scene(capture_scene(_simulation()))))
    document["frame"]["time"] = 3.0
    with pytest.raises(SceneError, match="digest"):
        parse_scene(json.dumps(document))

    document = cast(dict[str, Any], json.loads(dumps_scene(capture_scene(_simulation()))))
    document["frame"]["cells"][0]["color"] = [1.0, 0.0, 0.0]
    with pytest.raises(SceneError, match=r"unknown keys.*color"):
        parse_scene(_resign(document))

    with pytest.raises(SceneError, match="duplicate key"):
        parse_scene('{"format":"first","format":"second"}')


def test_scene_rejects_invalid_geometry_and_grid_shape() -> None:
    frame = capture_scene(_simulation())
    bad_direction = replace(
        frame,
        cells=(replace(frame.cells[0], direction=(2.0, 0.0, 0.0)), *frame.cells[1:]),
    )
    with pytest.raises(SceneError, match="normalized"):
        dumps_scene(bad_direction)

    assert frame.signal_grid is not None
    bad_levels = replace(frame, signal_grid=replace(frame.signal_grid, levels=(1.0,)))
    with pytest.raises(SceneError, match="expected 8 values"):
        dumps_scene(bad_levels)


@pytest.mark.parametrize("source", ["", b"\xff", "[1, 2, 3]"])
def test_scene_rejects_malformed_input(source: str | bytes) -> None:
    with pytest.raises(SceneError):
        parse_scene(source)
