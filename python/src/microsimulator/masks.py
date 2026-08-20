"""Extraction of device geometry from photomask CAD.

A mask DXF is authoring input, like every other predicate in the microfluidics
helpers: geometry is read once into plain data and the runtime never touches
CAD. The reader is deliberately minimal and closed: it parses axis-aligned
closed `LWPOLYLINE` outlines from the model-space `ENTITIES` section of an
ASCII DXF, ignoring block definitions (orphaned array remnants in mask files),
paper space, and every other entity kind. Files are size-bounded and nothing is
executed. Coordinates remain in drawing units unless the caller supplies a
source-specific ``unit_scale`` to ``extract_rectangles``; the reader does not
infer physical units from a DXF header or file convention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MAX_MASK_BYTES = 64 << 20


class MaskError(ValueError):
    """Raised when a mask file cannot be safely read or interpreted."""


@dataclass(frozen=True, slots=True)
class MaskPolyline:
    layer: str
    closed: bool
    vertices: tuple[tuple[float, float], ...]
    block: str | None = None


@dataclass(frozen=True, slots=True)
class MaskRectangle:
    layer: str
    center: tuple[float, float]
    width: float
    height: float


def load_mask_polylines(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = MAX_MASK_BYTES,
    include_blocks: bool = False,
) -> tuple[MaskPolyline, ...]:
    """Read the polylines of an ASCII DXF mask drawing.

    Model-space entities always load. With ``include_blocks``, polylines inside
    block definitions load as authored and are tagged with their block name.
    INSERT transforms are not applied, so callers may interpret block geometry
    only when the relevant definitions are known to use the desired coordinate
    system directly.
    """

    source = Path(path)
    try:
        with source.open("rb") as stream:
            encoded = stream.read(max_bytes + 1)
    except OSError as error:
        raise MaskError(f"could not read mask {source}") from error
    if not encoded:
        raise MaskError("mask file is empty")
    if len(encoded) > max_bytes:
        raise MaskError(f"mask exceeds the {max_bytes}-byte limit")
    try:
        text = encoded.decode("ascii", errors="replace")
    except UnicodeDecodeError as error:  # pragma: no cover - replace never raises
        raise MaskError("mask is not ASCII DXF") from error

    lines = text.splitlines()
    if len(lines) < 2:
        raise MaskError("mask is not a group-coded DXF document")

    polylines: list[MaskPolyline] = []
    in_entities = False
    in_blocks = False
    block_name: str | None = None
    pending_block_name = False
    layer = ""
    closed = False
    xs: list[float] = []
    ys: list[float] = []
    collecting = False

    def finish() -> None:
        nonlocal collecting
        if collecting and len(xs) == len(ys) and len(xs) >= 2:
            polylines.append(
                MaskPolyline(
                    layer=layer,
                    closed=closed,
                    vertices=tuple(zip(xs, ys, strict=True)),
                    block=block_name,
                )
            )
        collecting = False

    index = 0
    while index + 1 < len(lines):
        code = lines[index].strip()
        value = lines[index + 1].strip()
        index += 2
        if code != "0":
            if pending_block_name and code == "2":
                block_name = value
                pending_block_name = False
            elif collecting:
                try:
                    if code == "8":
                        layer = value
                    elif code == "70":
                        closed = bool(int(value) & 1)
                    elif code == "10":
                        xs.append(float(value))
                    elif code == "20":
                        ys.append(float(value))
                except ValueError as error:
                    raise MaskError(f"mask contains a malformed {code} group") from error
            continue
        finish()
        pending_block_name = False
        if value == "SECTION":
            # A section names itself in the group pair that follows: code 2,
            # then the name. Anything else leaves the section unnamed rather
            # than silently reading the next value as its name.
            named = index + 1 < len(lines) and lines[index].strip() == "2"
            section = lines[index + 1].strip() if named else ""
            in_entities = section == "ENTITIES"
            in_blocks = section == "BLOCKS"
        elif value == "ENDSEC":
            in_entities = False
            in_blocks = False
        elif value == "BLOCK":
            pending_block_name = True
        elif value == "ENDBLK":
            block_name = None
        elif value == "LWPOLYLINE" and (
            in_entities or (include_blocks and in_blocks and block_name is not None)
        ):
            collecting = True
            layer = ""
            closed = False
            xs = []
            ys = []
    finish()
    if not polylines:
        raise MaskError("mask contains no model-space polylines")
    return tuple(polylines)


def extract_rectangles(
    polylines: tuple[MaskPolyline, ...],
    *,
    layer: str | None = None,
    unit_scale: float = 1.0,
    alignment_tolerance: float = 1.0e-9,
) -> tuple[MaskRectangle, ...]:
    """Return the closed axis-aligned rectangles among the polylines.

    A rectangle is a closed outline of four or five vertices (the fifth may
    repeat the first) whose distinct vertices are exactly the four corners of
    its bounding box. Coordinates and sizes are multiplied by ``unit_scale``.
    """

    if unit_scale <= 0.0:
        raise MaskError("unit scale must be positive")
    rectangles: list[MaskRectangle] = []
    for polyline in polylines:
        if layer is not None and polyline.layer != layer:
            continue
        if not polyline.closed or not (4 <= len(polyline.vertices) <= 5):
            continue
        xs = [vertex[0] for vertex in polyline.vertices]
        ys = [vertex[1] for vertex in polyline.vertices]
        low_x, high_x = min(xs), max(xs)
        low_y, high_y = min(ys), max(ys)
        if high_x - low_x <= 0.0 or high_y - low_y <= 0.0:
            continue
        corners = {
            (low_x, low_y),
            (low_x, high_y),
            (high_x, low_y),
            (high_x, high_y),
        }
        matched: set[tuple[float, float]] = set()
        aligned = True
        for x, y in polyline.vertices:
            corner = next(
                (
                    candidate
                    for candidate in corners
                    if abs(x - candidate[0]) <= alignment_tolerance
                    and abs(y - candidate[1]) <= alignment_tolerance
                ),
                None,
            )
            if corner is None:
                aligned = False
                break
            matched.add(corner)
        if not aligned or matched != corners:
            continue
        rectangles.append(
            MaskRectangle(
                layer=polyline.layer,
                center=(
                    (low_x + high_x) * 0.5 * unit_scale,
                    (low_y + high_y) * 0.5 * unit_scale,
                ),
                width=(high_x - low_x) * unit_scale,
                height=(high_y - low_y) * unit_scale,
            )
        )
    return tuple(rectangles)


def match_rectangles(
    rectangles: tuple[MaskRectangle, ...],
    width: float,
    height: float,
    *,
    tolerance: float = 1.0,
    allow_rotated: bool = True,
) -> tuple[MaskRectangle, ...]:
    """Return the rectangles whose size matches ``width x height``.

    Sizes compare within ``tolerance`` in the rectangles' own units; with
    ``allow_rotated`` the swapped orientation also matches.
    """

    if width <= 0.0 or height <= 0.0 or tolerance < 0.0:
        raise MaskError("match dimensions must be positive and tolerance non-negative")

    def matches(rectangle: MaskRectangle) -> bool:
        direct = (
            abs(rectangle.width - width) <= tolerance
            and abs(rectangle.height - height) <= tolerance
        )
        rotated = (
            abs(rectangle.width - height) <= tolerance
            and abs(rectangle.height - width) <= tolerance
        )
        return direct or (allow_rotated and rotated)

    return tuple(rectangle for rectangle in rectangles if matches(rectangle))
