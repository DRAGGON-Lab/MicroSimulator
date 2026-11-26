"""Reading a chip design's scene into device geometry.

The `microfluidics` design tool describes a chip as rectangles drawn on layers
of stated height, solves the flow through it, and writes the result as one JSON
document, its scene. This module reads that document as plain data and cuts a
window out of it: the fluid boxes of one part of the chip, in coordinates local
to the window. `microsimulator.microfluidics.LayoutDevice` projects those boxes
into the engine's inputs.

A scene is authoring input, like a photomask. The reader is closed: it parses
JSON within a size bound, reads the fields named here, passes over every other,
and executes nothing. It reads scenes of one format version and refuses any
other, since a field of another version may mean something else.

Scene lengths are micrometers, which are also model lengths. Flow is reported
in physical units (millimeters per second, microliters per minute); a model
converts them with its own choice of seconds per model time unit.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

from ._core import GridFace  # pyright: ignore[reportMissingModuleSource]

MAX_DESIGN_SCENE_BYTES = 64 << 20

# The scene format read here. While the format's major version is 0, its minor
# is raised when a field changes meaning and its patch when fields are added,
# so a scene of this major and minor with at least this patch holds every field
# the reader uses.
DESIGN_SCENE_VERSION = (0, 6)
DESIGN_SCENE_MINIMUM_PATCH = 0

_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")

# A fluid box as (low corner, high corner) in micrometers.
Box = tuple[tuple[float, float, float], tuple[float, float, float]]


class DesignSceneError(ValueError):
    """Raised when a design scene cannot be safely read or interpreted."""


@dataclass(frozen=True, slots=True)
class DesignLayer:
    """A layer of the chip: every rectangle drawn on it is a cavity this high above the glass."""

    name: str
    height: float


@dataclass(frozen=True, slots=True)
class DesignRectangle:
    """One rectangle drawn on a layer, by the path of what drew it."""

    path: str
    layer: str
    low: tuple[float, float]
    high: tuple[float, float]


@dataclass(frozen=True, slots=True)
class DesignRow:
    """``count`` placements of one outline, each ``step`` from the last.

    The outline's rectangles are turned counterclockwise about its own origin
    by ``quarter_turns`` quarter turns, then moved to ``origin`` plus the
    placement's index times ``step``.
    """

    path: str
    outline: str
    origin: tuple[float, float]
    step: tuple[float, float]
    count: int
    quarter_turns: int


@dataclass(frozen=True, slots=True)
class DesignPort:
    """Where a channel leaves a region, and the direction it leaves in."""

    path: str
    at: tuple[float, float]
    width: float
    layer: str
    facing: tuple[float, float]


@dataclass(frozen=True, slots=True)
class DesignFlow:
    """What flows through one edge of the design's flow graph.

    ``flow_ul_min`` is signed from ``start`` to ``end``. ``velocity_mm_s`` is
    the mean speed in the edge's narrowest duct.
    """

    element: str
    start: str
    end: str
    flow_ul_min: float
    velocity_mm_s: float
    shear_pa: float
    reynolds: float


@dataclass(frozen=True, slots=True)
class DesignInstance:
    """One placed region: its path, its kind, and the box about what it drew."""

    path: str
    region: str
    low: tuple[float, float] | None
    high: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class DesignCrossing:
    """Where a channel passes through one side of a window.

    ``side`` is the window's side as a face of the device's grid. ``low`` and
    ``high`` bound the channel along that side in window coordinates: along y
    on an x side and along x on a y side. ``height`` is the fluid's height just
    inside the window, and ``path`` names what was drawn there.
    """

    side: GridFace
    low: float
    high: float
    height: float
    path: str


@dataclass(frozen=True, slots=True)
class DesignWindow:
    """The fluid of one rectangular part of a chip.

    ``boxes`` are in window coordinates: the window's low corner is the origin,
    the glass is ``z = 0``, and each box rises from the glass to the height of
    its layer. ``origin`` is the window's low corner in design coordinates and
    ``size`` its extent in x and y.

    ``crossings`` are where fluid inside the window meets fluid outside it
    across one of the window's sides: the places a model opens as ports. Fluid
    that only touches a side, with a wall beyond, is not a crossing.
    """

    origin: tuple[float, float]
    size: tuple[float, float]
    boxes: tuple[Box, ...]
    crossings: tuple[DesignCrossing, ...] = ()

    def local(self, x: float, y: float) -> tuple[float, float]:
        """A point of the design in window coordinates."""

        return (x - self.origin[0], y - self.origin[1])


@dataclass(frozen=True, slots=True)
class DesignScene:
    """A chip design as its scene states it.

    ``fractions`` holds, for each fed species, the share from 0 to 1 of what
    arrives at each vertex of the flow graph. ``pressures_mbar``, ``fractions``
    and the ``start`` and ``end`` of each flow name the same vertices.
    ``passes`` is whether the design passed every check of the design tool.
    """

    version: str
    design: str
    passes: bool
    layers: tuple[DesignLayer, ...]
    rectangles: tuple[DesignRectangle, ...]
    rows: tuple[DesignRow, ...]
    outlines: Mapping[str, tuple[DesignRectangle, ...]]
    ports: tuple[DesignPort, ...]
    flows: tuple[DesignFlow, ...]
    pressures_mbar: Mapping[str, float]
    viscosity_mpa_s: float | None
    fractions: Mapping[str, Mapping[str, float]]
    instances: tuple[DesignInstance, ...]

    def instance(self, path: str) -> DesignInstance:
        """The placed region at ``path``."""

        for instance in self.instances:
            if instance.path == path:
                return instance
        raise DesignSceneError(f"the design places no region at {path!r}")

    def flows_of(self, element: str) -> tuple[DesignFlow, ...]:
        """The flow through every edge that belongs to one region or channel."""

        return tuple(flow for flow in self.flows if flow.element == element)

    def fraction(self, species: str, vertex: str) -> float:
        """The share of what arrives at a vertex that is ``species``, from 0 to 1."""

        shares = self.fractions.get(species)
        if shares is None:
            raise DesignSceneError(f"the design feeds no species {species!r}")
        share = shares.get(vertex)
        if share is None:
            raise DesignSceneError(f"the design's flow graph has no vertex {vertex!r}")
        return share

    def window(self, low: tuple[float, float], high: tuple[float, float]) -> DesignWindow:
        """The fluid inside the rectangle from ``low`` to ``high``, in design coordinates.

        Every rectangle is clipped to the window. Of a row of placements only
        those that reach the window are placed, so a window on a chip with many
        thousands of repeated features costs what the window holds.
        """

        if not all(math.isfinite(value) for value in (*low, *high)):
            raise DesignSceneError("a window's corners must be finite")
        if high[0] <= low[0] or high[1] <= low[1]:
            raise DesignSceneError("a window's high corner must lie beyond its low corner")
        heights = {layer.name: layer.height for layer in self.layers}
        boxes: list[Box] = []
        near: list[_Drawn] = []

        def clip(
            path: str, layer: str, corner: tuple[float, float], far: tuple[float, float]
        ) -> None:
            height = heights.get(layer)
            if height is None:
                raise DesignSceneError(f"a rectangle is drawn on unknown layer {layer!r}")
            if far[0] > corner[0] and far[1] > corner[1]:
                near.append((path, height, corner, far))
            x0, y0 = max(corner[0], low[0]), max(corner[1], low[1])
            x1, y1 = min(far[0], high[0]), min(far[1], high[1])
            if x1 > x0 and y1 > y0:
                boxes.append(
                    ((x0 - low[0], y0 - low[1], 0.0), (x1 - low[0], y1 - low[1], height))
                )

        for rectangle in self.rectangles:
            clip(rectangle.path, rectangle.layer, rectangle.low, rectangle.high)
        for row in self.rows:
            outline = self.outlines.get(row.outline)
            if outline is None:
                raise DesignSceneError(f"row {row.path!r} places unknown outline {row.outline!r}")
            for rectangle in outline:
                corner, far = _turned(rectangle, row.quarter_turns)
                for index in _reaching(row, corner, far, low, high):
                    dx = row.origin[0] + index * row.step[0]
                    dy = row.origin[1] + index * row.step[1]
                    clip(
                        row.path,
                        rectangle.layer,
                        (corner[0] + dx, corner[1] + dy),
                        (far[0] + dx, far[1] + dy),
                    )
        return DesignWindow(
            low, (high[0] - low[0], high[1] - low[1]), tuple(boxes), _crossings(near, low, high)
        )


# A drawn rectangle near a window: its path, its height, and its corners.
_Drawn = tuple[str, float, tuple[float, float], tuple[float, float]]


def _holds(drawn: _Drawn, axis: int, at: float, *, above: bool) -> bool:
    """Whether a rectangle holds the points just above, or just below, a coordinate."""

    start, end = drawn[2][axis], drawn[3][axis]
    return start <= at < end if above else start < at <= end


def _tallest(drawn: list[_Drawn], along: int, at: float) -> tuple[float, str]:
    """The height and path of the tallest rectangle that spans a coordinate."""

    covering = [each for each in drawn if each[2][along] < at < each[3][along]]
    return max(((each[1], each[0]) for each in covering), default=(0.0, ""))


def _crossings(
    near: list[_Drawn], low: tuple[float, float], high: tuple[float, float]
) -> tuple[DesignCrossing, ...]:
    """Where fluid inside a window meets fluid outside it across a side."""

    sides = (
        (GridFace.X_LOWER, 0, low[0], False),
        (GridFace.X_UPPER, 0, high[0], True),
        (GridFace.Y_LOWER, 1, low[1], False),
        (GridFace.Y_UPPER, 1, high[1], True),
    )
    crossings: list[DesignCrossing] = []
    for side, axis, at, upper in sides:
        along = 1 - axis
        # A rectangle lies inside the side when it holds the points just within
        # the window there, and outside it when it holds those just beyond.
        inside = [drawn for drawn in near if _holds(drawn, axis, at, above=not upper)]
        outside = [drawn for drawn in near if _holds(drawn, axis, at, above=upper)]
        edges = sorted(
            {
                min(max(edge[along], low[along]), high[along])
                for drawn in (*inside, *outside)
                for edge in (drawn[2], drawn[3])
            }
        )
        open_from: float | None = None
        height, path = 0.0, ""
        for start, end in pairwise(edges):
            middle = (start + end) * 0.5
            within, beyond = _tallest(inside, along, middle), _tallest(outside, along, middle)
            crosses = within[0] > 0.0 and beyond[0] > 0.0
            if open_from is not None and (not crosses or within[0] != height):
                crossings.append(
                    DesignCrossing(side, open_from - low[along], start - low[along], height, path)
                )
                open_from = None
            if crosses and open_from is None:
                open_from, (height, path) = start, within
        if open_from is not None:
            crossings.append(
                DesignCrossing(side, open_from - low[along], edges[-1] - low[along], height, path)
            )
    return tuple(crossings)


def _turned(
    rectangle: DesignRectangle, quarter_turns: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    """A rectangle's corners after whole counterclockwise quarter turns about the origin."""

    (x0, y0), (x1, y1) = rectangle.low, rectangle.high
    match quarter_turns % 4:
        case 0:
            return (x0, y0), (x1, y1)
        case 1:
            return (-y1, x0), (-y0, x1)
        case 2:
            return (-x1, -y1), (-x0, -y0)
        case _:
            return (y0, -x1), (y1, -x0)


def _reaching(
    row: DesignRow,
    corner: tuple[float, float],
    far: tuple[float, float],
    low: tuple[float, float],
    high: tuple[float, float],
) -> range:
    """The placements of a row whose rectangle may reach or touch the window.

    The range is the interval of indices that overlap the window on both axes,
    widened by one against rounding and so that a placement lying against the
    window's side from outside is among them; clipping discards any that do
    not reach inside.
    """

    first, last = 0, row.count - 1
    for axis in (0, 1):
        step = row.step[axis]
        start, end = corner[axis] + row.origin[axis], far[axis] + row.origin[axis]
        if step == 0.0:
            if end < low[axis] or start > high[axis]:
                return range(0)
            continue
        # A placement overlaps when its far edge is past the window's low edge
        # and its near edge is short of the window's high edge.
        bounds = sorted(((low[axis] - end) / step, (high[axis] - start) / step))
        first = max(first, math.floor(bounds[0]) - 1)
        last = min(last, math.ceil(bounds[1]) + 1)
    if row.step == (0.0, 0.0):
        # Every placement is the same rectangle: one stands for them all.
        last = min(last, first)
    return range(first, last + 1)


def load_design_scene(
    path: str | os.PathLike[str], *, max_bytes: int = MAX_DESIGN_SCENE_BYTES
) -> DesignScene:
    """Read the scene file a chip design was written to."""

    source = Path(path)
    try:
        with source.open("rb") as stream:
            encoded = stream.read(max_bytes + 1)
    except OSError as error:
        raise DesignSceneError(f"could not read design scene {source}") from error
    if len(encoded) > max_bytes:
        raise DesignSceneError(f"design scene exceeds the {max_bytes}-byte limit")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DesignSceneError("design scene is not UTF-8 JSON") from error
    return parse_design_scene(text)


def parse_design_scene(text: str) -> DesignScene:
    """Read a chip design's scene from its JSON text."""

    def refuse(constant: str) -> float:
        raise DesignSceneError(f"design scene holds the non-finite number {constant}")

    try:
        document: object = json.loads(text, parse_constant=refuse)
    except (ValueError, RecursionError) as error:
        if isinstance(error, DesignSceneError):
            raise
        raise DesignSceneError("design scene is not valid JSON") from error
    held = _mapping(document, "the scene")
    version = _checked_version(held.get("version"))
    def each(name: str) -> list[object]:
        return _sequence(_field(held, name), name)

    outlines = {
        name: tuple(_rectangle(shape) for shape in _sequence(shapes, f"outline {name!r}"))
        for name, shapes in _mapping(_field(held, "outlines"), "outlines").items()
    }
    viscosity = held.get("viscosity_mpa_s")
    return DesignScene(
        version=version,
        design=_text(_field(held, "design"), "design"),
        passes=_flag(_field(held, "passes"), "passes"),
        layers=tuple(_layer(layer) for layer in each("layers")),
        rectangles=tuple(_rectangle(shape) for shape in each("shapes")),
        rows=tuple(_row(row) for row in each("translates")),
        outlines=outlines,
        ports=tuple(_port(port) for port in each("ports")),
        flows=tuple(_flow(flow) for flow in each("flows")),
        pressures_mbar=_numbers(_field(held, "pressures_mbar"), "pressures_mbar"),
        viscosity_mpa_s=None if viscosity is None else _number(viscosity, "viscosity_mpa_s"),
        fractions={
            species: _numbers(shares, f"fractions of {species!r}")
            for species, shares in _mapping(_field(held, "fractions"), "fractions").items()
        },
        instances=tuple(_instance(instance) for instance in each("instances")),
    )


def _checked_version(version: object) -> str:
    major, minor = DESIGN_SCENE_VERSION
    read = f"{major}.{minor}.{DESIGN_SCENE_MINIMUM_PATCH} and later patches of {major}.{minor}"
    if not isinstance(version, str):
        raise DesignSceneError(
            f"design scene states no semantic version; this reader takes {read}"
        )
    matched = _VERSION.fullmatch(version)
    if matched is None:
        raise DesignSceneError(
            f"design scene version {version!r} is not a semantic version; this reader takes {read}"
        )
    found = (int(matched[1]), int(matched[2]))
    if found != DESIGN_SCENE_VERSION or int(matched[3]) < DESIGN_SCENE_MINIMUM_PATCH:
        raise DesignSceneError(f"design scene is version {version}; this reader takes {read}")
    return version


def _field(held: Mapping[str, object], name: str) -> object:
    if name not in held:
        raise DesignSceneError(f"design scene has no {name!r}")
    return held[name]


def _mapping(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise DesignSceneError(f"{what} must be a JSON object")
    return cast(dict[str, object], value)


def _sequence(value: object, what: str) -> list[object]:
    if not isinstance(value, list):
        raise DesignSceneError(f"{what} must be a JSON array")
    return cast(list[object], value)


def _text(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise DesignSceneError(f"{what} must be a string")
    return value


def _flag(value: object, what: str) -> bool:
    if not isinstance(value, bool):
        raise DesignSceneError(f"{what} must be true or false")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DesignSceneError(f"{what} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise DesignSceneError(f"{what} must be finite")
    return number


def _count(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DesignSceneError(f"{what} must be a non-negative whole number")
    return value


def _numbers(value: object, what: str) -> dict[str, float]:
    held = _mapping(value, what)
    return {name: _number(each, f"{what}[{name!r}]") for name, each in held.items()}


def _pair(value: object, what: str) -> tuple[float, float]:
    held = _sequence(value, what)
    if len(held) != 2:
        raise DesignSceneError(f"{what} must hold two numbers")
    return (_number(held[0], what), _number(held[1], what))


def _corners(value: object, what: str) -> tuple[tuple[float, float], tuple[float, float]]:
    held = _sequence(value, what)
    if len(held) != 4:
        raise DesignSceneError(f"{what} must hold four numbers")
    x0, y0, x1, y1 = (_number(each, what) for each in held)
    if x1 < x0 or y1 < y0:
        raise DesignSceneError(f"{what} must run from its low corner to its high corner")
    return (x0, y0), (x1, y1)


def _layer(value: object) -> DesignLayer:
    held = _mapping(value, "a layer")
    height = _number(_field(held, "height_um"), "a layer's height_um")
    if height <= 0.0:
        raise DesignSceneError("a layer's height_um must be positive")
    return DesignLayer(_text(_field(held, "name"), "a layer's name"), height)


def _rectangle(value: object) -> DesignRectangle:
    held = _mapping(value, "a shape")
    low, high = _corners(_field(held, "rect"), "a shape's rect")
    return DesignRectangle(
        _text(_field(held, "path"), "a shape's path"),
        _text(_field(held, "layer"), "a shape's layer"),
        low,
        high,
    )


def _row(value: object) -> DesignRow:
    held = _mapping(value, "a translates row")
    return DesignRow(
        path=_text(_field(held, "path"), "a row's path"),
        outline=_text(_field(held, "outline"), "a row's outline"),
        origin=_pair(_field(held, "origin"), "a row's origin"),
        step=_pair(_field(held, "step"), "a row's step"),
        count=_count(_field(held, "count"), "a row's count"),
        quarter_turns=_count(_field(held, "quarter_turns"), "a row's quarter_turns"),
    )


def _port(value: object) -> DesignPort:
    held = _mapping(value, "a port")
    return DesignPort(
        path=_text(_field(held, "path"), "a port's path"),
        at=_pair(_field(held, "at"), "a port's at"),
        width=_number(_field(held, "width_um"), "a port's width_um"),
        layer=_text(_field(held, "layer"), "a port's layer"),
        facing=_pair(_field(held, "facing"), "a port's facing"),
    )


def _flow(value: object) -> DesignFlow:
    held = _mapping(value, "a flow")
    return DesignFlow(
        element=_text(_field(held, "element"), "a flow's element"),
        start=_text(_field(held, "start"), "a flow's start"),
        end=_text(_field(held, "end"), "a flow's end"),
        flow_ul_min=_number(_field(held, "flow_ul_min"), "a flow's flow_ul_min"),
        velocity_mm_s=_number(_field(held, "velocity_mm_s"), "a flow's velocity_mm_s"),
        shear_pa=_number(_field(held, "shear_pa"), "a flow's shear_pa"),
        reynolds=_number(_field(held, "reynolds"), "a flow's reynolds"),
    )


def _instance(value: object) -> DesignInstance:
    held = _mapping(value, "an instance")
    rect = held.get("rect")
    low, high = (None, None) if rect is None else _corners(rect, "an instance's rect")
    return DesignInstance(
        _text(_field(held, "path"), "an instance's path"),
        _text(_field(held, "region"), "an instance's region"),
        low,
        high,
    )
