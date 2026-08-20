from __future__ import annotations

import math
from pathlib import Path

import pytest
from cellmodeller2.masks import (
    MaskError,
    MaskPolyline,
    extract_rectangles,
    load_mask_polylines,
    match_rectangles,
)


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
