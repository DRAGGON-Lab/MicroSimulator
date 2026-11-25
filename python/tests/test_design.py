from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from microsimulator.design import (
    DesignSceneError,
    load_design_scene,
    parse_design_scene,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "design"
_TRAP_CHAMBER = _FIXTURES / "trap_chamber.scene.json"


def _scene(**changes: Any) -> str:
    """A small scene: one channel along y with a row of four trenches off its right side."""

    scene: dict[str, Any] = {
        "version": "0.6.0",
        "design": "comb",
        "passes": True,
        "layers": [
            {"name": "trench", "height_um": 1.0, "gds": 1, "colour": "#e0a030"},
            {"name": "flow", "height_um": 20.0, "gds": 2, "colour": "#4c9be8"},
        ],
        "outlines": {"tooth": [{"path": "tooth", "layer": "trench", "rect": [0, 0, 25, 2]}]},
        "shapes": [{"path": "lane.channel", "layer": "flow", "rect": [1000, 500, 1100, 900]}],
        "translates": [
            {
                "path": "lane.right",
                "outline": "tooth",
                "origin": [1100, 600],
                "step": [0, 10],
                "count": 4,
                "quarter_turns": 0,
            }
        ],
        "ports": [
            {
                "path": "lane.inlet",
                "at": [1050, 500],
                "width_um": 100,
                "layer": "flow",
                "facing": [0, -1],
            }
        ],
        "flows": [
            {
                "element": "lane",
                "start": "a",
                "end": "b",
                "flow_ul_min": 1.2,
                "velocity_mm_s": 10.0,
                "shear_pa": 0.5,
                "reynolds": 0.1,
            }
        ],
        "pressures_mbar": {"a": 10.0, "b": 0.0},
        "viscosity_mpa_s": 0.7,
        "fractions": {"inducer": {"a": 1.0, "b": 0.25}},
        "instances": [
            {"path": "lane", "region": "trench_lane", "rect": [1000, 500, 1125, 900]},
            {"path": "label", "region": "text", "rect": None},
        ],
        "a_field_of_a_later_patch": {"anything": [1, 2, 3]},
    }
    scene.update(changes)
    return json.dumps(scene)


def test_a_scene_is_read_as_plain_data() -> None:
    scene = parse_design_scene(_scene())

    assert (scene.version, scene.design, scene.passes) == ("0.6.0", "comb", True)
    assert {layer.name: layer.height for layer in scene.layers} == {"trench": 1.0, "flow": 20.0}
    channel = scene.rectangles[0]
    assert (channel.path, channel.layer) == ("lane.channel", "flow")
    assert (channel.low, channel.high) == ((1000.0, 500.0), (1100.0, 900.0))
    assert (scene.rows[0].outline, scene.rows[0].count) == ("tooth", 4)
    assert scene.ports[0].facing == (0.0, -1.0)
    assert scene.viscosity_mpa_s == 0.7
    assert scene.pressures_mbar == {"a": 10.0, "b": 0.0}

    lane = scene.instance("lane")
    assert (lane.region, lane.low, lane.high) == ("trench_lane", (1000.0, 500.0), (1125.0, 900.0))
    assert scene.instance("label").low is None
    assert [flow.velocity_mm_s for flow in scene.flows_of("lane")] == [10.0]
    assert scene.flows_of("elsewhere") == ()
    assert scene.fraction("inducer", "b") == 0.25
    with pytest.raises(DesignSceneError, match="places no region"):
        scene.instance("missing")
    with pytest.raises(DesignSceneError, match="feeds no species"):
        scene.fraction("nutrient", "a")
    with pytest.raises(DesignSceneError, match="no vertex"):
        scene.fraction("inducer", "c")


@pytest.mark.parametrize("version", ["0.6.0", "0.6.3", "0.6.12"])
def test_a_scene_of_the_format_read_or_a_later_patch_is_taken(version: str) -> None:
    assert parse_design_scene(_scene(version=version)).version == version


@pytest.mark.parametrize(
    "version", ["0.5.0", "0.7.0", "1.0.0", "1.6.0", "0.6", "6.0", "0.6.0-rc1", "v0.6.0", 5, None]
)
def test_a_scene_of_another_format_is_refused(version: object) -> None:
    with pytest.raises(DesignSceneError, match=r"this reader takes 0\.6\.0 and later patches"):
        parse_design_scene(_scene(version=version))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{", "not valid JSON"),
        ("[]", "the scene must be a JSON object"),
        (_scene(viscosity_mpa_s="thin"), "viscosity_mpa_s must be a number"),
        (_scene(passes=1), "passes must be true or false"),
        (_scene(shapes={}), "shapes must be a JSON array"),
        (_scene(shapes=[{"path": "a", "layer": "flow", "rect": [0, 0, 1]}]), "four numbers"),
        (_scene(shapes=[{"path": "a", "layer": "flow", "rect": [5, 0, 1, 1]}]), "low corner"),
        (_scene(layers=[{"name": "flow", "height_um": 0}]), "height_um must be positive"),
        (_scene(pressures_mbar={"a": True}), "must be a number"),
        (_scene().replace('"viscosity_mpa_s": 0.7', '"viscosity_mpa_s": NaN'), "non-finite"),
        (_scene().replace('"flows"', '"streams"'), "has no 'flows'"),
    ],
)
def test_a_malformed_scene_is_refused(text: str, message: str) -> None:
    with pytest.raises(DesignSceneError, match=message):
        parse_design_scene(text)


def test_a_scene_file_is_read_within_a_size_bound(tmp_path: Path) -> None:
    path = tmp_path / "comb.scene.json"
    path.write_text(_scene(), encoding="utf-8")
    assert load_design_scene(path).design == "comb"
    with pytest.raises(DesignSceneError, match="exceeds the 64-byte limit"):
        load_design_scene(path, max_bytes=64)
    with pytest.raises(DesignSceneError, match="could not read"):
        load_design_scene(tmp_path / "absent.scene.json")


def test_a_window_holds_clipped_fluid_in_its_own_coordinates() -> None:
    scene = parse_design_scene(_scene())
    # Trenches stand at y = 600, 610, 620 and 630, each 2 high. The window takes the second
    # whole, half of the third, and 15 of their 25 length.
    window = scene.window((1090.0, 605.0), (1115.0, 621.0))

    assert (window.origin, window.size) == ((1090.0, 605.0), (25.0, 16.0))
    assert window.local(1100.0, 610.0) == (10.0, 5.0)
    assert set(window.boxes) == {
        ((0.0, 0.0, 0.0), (10.0, 16.0, 20.0)),
        ((10.0, 5.0, 0.0), (25.0, 7.0, 1.0)),
        ((10.0, 15.0, 0.0), (25.0, 16.0, 1.0)),
    }

    assert scene.window((0.0, 0.0), (10.0, 10.0)).boxes == ()
    with pytest.raises(DesignSceneError, match="beyond its low corner"):
        scene.window((10.0, 10.0), (10.0, 20.0))


@pytest.mark.parametrize(
    ("quarter_turns", "first"),
    [
        (0, ((0.0, 0.0), (25.0, 2.0))),
        (1, ((-2.0, 0.0), (0.0, 25.0))),
        (2, ((-25.0, -2.0), (0.0, 0.0))),
        (3, ((0.0, -25.0), (2.0, 0.0))),
    ],
)
def test_a_row_is_turned_about_its_outline_origin_then_stepped(
    quarter_turns: int, first: tuple[tuple[float, float], tuple[float, float]]
) -> None:
    row = {
        "path": "row",
        "outline": "tooth",
        "origin": [500, 500],
        "step": [-40, 30],
        "count": 3,
        "quarter_turns": quarter_turns,
    }
    scene = parse_design_scene(_scene(shapes=[], translates=[row]))
    window = scene.window((0.0, 0.0), (1000.0, 1000.0))

    (low_x, low_y), (high_x, high_y) = first
    assert set(window.boxes) == {
        (
            (500.0 + low_x - 40.0 * at, 500.0 + low_y + 30.0 * at, 0.0),
            (500.0 + high_x - 40.0 * at, 500.0 + high_y + 30.0 * at, 1.0),
        )
        for at in range(3)
    }


def test_a_window_on_a_long_row_places_only_what_it_holds() -> None:
    row = {
        "path": "row",
        "outline": "tooth",
        "origin": [0, 0],
        "step": [0, 2.6],
        "count": 10_000_000,
        "quarter_turns": 0,
    }
    scene = parse_design_scene(_scene(shapes=[], translates=[row]))
    window = scene.window((0.0, 13_000.0), (25.0, 13_010.0))

    # Placements 5000 to 5003 begin inside the window; the last is cut by its high edge.
    begun = sorted(box[0][1] for box in window.boxes)
    assert len(begun) == 4
    assert all(math.isclose(low, 2.6 * at, abs_tol=1e-9) for at, low in enumerate(begun))
    assert all(box[1][1] <= 10.0 for box in window.boxes)


def test_a_window_refuses_what_the_scene_does_not_define() -> None:
    row = {"path": "r", "outline": "none", "origin": [0, 0], "step": [0, 1], "count": 1}
    row["quarter_turns"] = 0
    with pytest.raises(DesignSceneError, match="unknown outline"):
        parse_design_scene(_scene(translates=[row])).window((0.0, 0.0), (1.0, 1.0))
    shapes = [{"path": "a", "layer": "metal", "rect": [0, 0, 1, 1]}]
    with pytest.raises(DesignSceneError, match="unknown layer"):
        parse_design_scene(_scene(shapes=shapes)).window((0.0, 0.0), (1.0, 1.0))


def test_the_design_tool_scene_gives_the_trap_chamber_window() -> None:
    scene = load_design_scene(_TRAP_CHAMBER)

    assert (scene.version, scene.design) == ("0.6.0", "trap_chamber_chip")
    assert {layer.name: layer.height for layer in scene.layers} == {"trench": 1.65, "flow": 10.0}
    trap = scene.instance("trap")
    assert trap.region == "trap_chamber" and trap.low is not None and trap.high is not None
    window = scene.window(trap.low, trap.high)
    assert window.size == (185.0, 300.0)
    assert set(window.boxes) == {
        ((0.0, 0.0, 0.0), (100.0, 300.0, 10.0)),
        ((100.0, 100.0, 0.0), (185.0, 200.0, 1.65)),
    }

    # What the design says of how the chip is run comes with the geometry.
    assert scene.viscosity_mpa_s is not None and math.isclose(scene.viscosity_mpa_s, 0.7)
    (through,) = scene.flows_of("trap")
    assert math.isclose(through.flow_ul_min, 0.01)
    # 0.01 uL/min through the 100 by 10 um channel.
    assert math.isclose(through.velocity_mm_s, 0.01e-9 / 60.0 / (100e-6 * 10e-6) * 1e3)
    assert math.isclose(scene.fraction("nutrient", through.start), 1.0)
