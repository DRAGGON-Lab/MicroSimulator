from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

import pytest
from microsimulator.masks import (
    MaskError,
    MaskPolyline,
    extract_rectangles,
    load_mask_polylines,
    match_rectangles,
)

_PRINDLE = Path(__file__).resolve().parents[2] / "docs" / "tutorials" / "devices" / "prindle.dxf"


def test_rectangle_extraction_is_selective_and_explicitly_scaled() -> None:
    polylines = (
        MaskPolyline("A", True, ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0))),
        MaskPolyline("A", False, ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0))),
        MaskPolyline("A", True, ((0.0, 0.0), (2.0, 0.5), (2.0, 1.0), (0.0, 1.0))),
        MaskPolyline("B", True, ((0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (0.0, 2.0), (0.0, 0.0))),
    )
    rectangles = extract_rectangles(polylines, unit_scale=10.0)
    assert len(rectangles) == 2
    assert math.isclose(rectangles[0].width, 20.0)
    assert rectangles[0].center == (10.0, 5.0)

    layered = extract_rectangles(polylines, layer="B", unit_scale=10.0)
    assert len(layered) == 1
    assert math.isclose(layered[0].height, 20.0)

    rotated = match_rectangles(rectangles, 10.0, 20.0, tolerance=0.01)
    assert len(rotated) == 2
    strict = match_rectangles(rectangles, 10.0, 20.0, tolerance=0.01, allow_rotated=False)
    assert len(strict) == 1


def test_mask_reader_rejects_unusable_input(tmp_path: Path) -> None:
    empty = tmp_path / "empty.dxf"
    empty.write_text("")
    with pytest.raises(MaskError, match="empty"):
        load_mask_polylines(empty)

    no_entities = tmp_path / "no-entities.dxf"
    no_entities.write_text("  0\nSECTION\n  2\nHEADER\n  0\nENDSEC\n  0\nEOF\n")
    with pytest.raises(MaskError, match="no model-space polylines"):
        load_mask_polylines(no_entities)
    with pytest.raises(MaskError, match="byte limit"):
        load_mask_polylines(no_entities, max_bytes=8)


def test_block_definitions_are_opt_in_and_retain_their_name(tmp_path: Path) -> None:
    source = tmp_path / "blocks.dxf"
    source.write_text(
        "  0\nSECTION\n  2\nBLOCKS\n"
        "  0\nBLOCK\n  2\nFEATURE\n"
        "  0\nLWPOLYLINE\n  8\nBLOCK-LAYER\n 70\n1\n"
        " 10\n10\n 20\n20\n 10\n11\n 20\n20\n"
        " 10\n11\n 20\n21\n 10\n10\n 20\n21\n"
        "  0\nENDBLK\n  0\nENDSEC\n"
        "  0\nSECTION\n  2\nENTITIES\n"
        "  0\nLWPOLYLINE\n  8\nMODEL-LAYER\n 70\n1\n"
        " 10\n0\n 20\n0\n 10\n1\n 20\n0\n"
        " 10\n1\n 20\n1\n 10\n0\n 20\n1\n"
        "  0\nENDSEC\n  0\nEOF\n"
    )

    model_space = load_mask_polylines(source)
    assert len(model_space) == 1
    assert model_space[0].block is None

    with_blocks = load_mask_polylines(source, include_blocks=True)
    assert len(with_blocks) == 2
    block = next(polyline for polyline in with_blocks if polyline.block is not None)
    assert block.block == "FEATURE"
    assert block.vertices[0] == (10.0, 20.0)


def test_prindle_mask_yields_the_documented_layout() -> None:
    polylines = load_mask_polylines(_PRINDLE)
    raw_rectangles = extract_rectangles(polylines, layer="Layer-2")
    raw_traps = match_rectangles(raw_rectangles, 0.110, 0.100, tolerance=0.001)

    # The supplemental methods report a 100-micrometer trap dimension with
    # 25-micrometer spacing. In this particular drawing, its raw 0.100 outline
    # dimension and 0.125 row pitch therefore corroborate a conversion from
    # one drawing unit to one millimeter; this is evidence about this file,
    # not a convention imposed on other DXF inputs.
    assert len(raw_traps) == 496
    assert all(
        math.isclose(trap.width, 0.110, abs_tol=0.001)
        and math.isclose(trap.height, 0.100, abs_tol=0.001)
        for trap in raw_traps
    )

    rectangles = extract_rectangles(polylines, layer="Layer-2", unit_scale=1000.0)
    traps = match_rectangles(rectangles, 110.0, 100.0, tolerance=1.0)
    xs = sorted({round(trap.center[0], 1) for trap in traps})
    ys = sorted({round(trap.center[1], 1) for trap in traps})

    assert len(traps) == 496
    assert len(xs) == 16
    assert len(ys) == 31
    assert math.isclose(ys[1] - ys[0], 125.0, abs_tol=0.1)
    assert math.isclose(ys[-1] - ys[0], 3750.0, abs_tol=1.0)
    column_pitches = sorted({round(right - left, 1) for left, right in pairwise(xs)})
    assert column_pitches == [135.0, 160.0, 172.5]
    assert math.isclose(xs[-1] - xs[0], 2400.0, abs_tol=1.0)


def test_prindle_block_traversal_exposes_unplaced_layer_geometry() -> None:
    polylines = load_mask_polylines(_PRINDLE, include_blocks=True)
    blocks = {polyline.block for polyline in polylines if polyline.block is not None}
    assert len(blocks) >= 2

    layer5 = [
        polyline
        for polyline in polylines
        if polyline.layer == "Layer-5" and polyline.block is not None
    ]
    assert len(layer5) > 3000
    large_outlines = [
        polyline
        for polyline in layer5
        if min(
            max(vertex[0] for vertex in polyline.vertices)
            - min(vertex[0] for vertex in polyline.vertices),
            max(vertex[1] for vertex in polyline.vertices)
            - min(vertex[1] for vertex in polyline.vertices),
        )
        >= 0.9
    ]
    assert len(large_outlines) >= 40

    # These are unplaced block definitions. Without an accompanying process
    # map, the test deliberately makes no claim about their fabrication role.
    model_space = load_mask_polylines(_PRINDLE)
    assert all(polyline.block is None for polyline in model_space)
