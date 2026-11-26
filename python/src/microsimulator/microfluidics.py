"""Authoring helpers for microfluidic device models.

A device is described once in physical coordinates and then projected into the
engine's typed inputs: box wall constraints for mechanics, a solid mask for the
signal grid, and a numerically solved face-staggered flow field for advection
(the shallow, depth-integrated solve of `microsimulator.flow`, with conservative
fluxes through supported column geometries). The runtime receives only materialized data;
every predicate here is authoring-time.

`TrapChannelDevice` and `BiopixelTrapDevice` state one geometry each through a
few dimensions. `LayoutDevice` takes any fluid drawn as boxes standing on a
common floor, such as a window of a chip design read by `microsimulator.design`.

The media entering at the inlet carries one concentration per signal. A model
changes that composition during a run with `set_inlet_values`, and
`InletSchedule` describes a composition that switches or ramps over model time.

Voxel centers are classified against the same physical wall predicates used by
mechanics. For axis-aligned planes the staircase wall offset is at most half a
voxel spacing. Curved geometry requires convergence checks; unresolved narrow
features cannot be recovered by renaming or enlarging the fluid region.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    BoxConstraintInit,
    ConstraintRegion,
    DepthAveragedPortFlowResult,
    FlowPort,
    FlowPortKind,
    GridBoundary,
    GridBoundaryKind,
    GridBoundaryPatch,
    GridFace,
    GridFacePatch,
    GridShape,
    PlaneConstraintInit,
    ResolvedPortFlowResult,
    SignalGridSpec,
    Simulation,
    Vec3,
)
from .flow import gap_mobility, solve_flow_field, solve_port_flow_field
from .stokes import solve_port_stokes_field

# A voxel edge that lands on a wall plane belongs to the wall, so the voxel
# tests admit a rounding margin: without it a wall drawn exactly on a lattice
# face resolves as solid or fluid according to float rounding.
_EDGE_TOLERANCE = 1.0e-6

# Wall blocks as (low corner, high corner) pairs in device coordinates.
_Blocks = tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]

# The flow axis each inlet face sets, and whether media runs toward its lower end.
_INLET_FLOW = {
    GridFace.X_LOWER: ("x", False),
    GridFace.X_UPPER: ("x", True),
    GridFace.Y_LOWER: ("y", False),
    GridFace.Y_UPPER: ("y", True),
}


def _reaches(edge: float, wall: float, half: float) -> bool:
    """Whether a voxel's lower edge has reached a wall lying above it."""

    return edge >= wall - _EDGE_TOLERANCE * half


def _recedes(edge: float, wall: float, half: float) -> bool:
    """Whether a voxel's upper edge has reached a wall lying below it."""

    return edge <= wall + _EDGE_TOLERANCE * half


@dataclass(frozen=True, slots=True)
class InletSchedule:
    """The composition of the inlet media as a function of model time.

    A composition holds one concentration per signal. The inlet carries
    ``initial`` until ``times[0]`` and ``values[i]`` from ``times[i]``. With
    ``ramp=False`` the composition switches at each time, as when a valve
    selects another reservoir. With ``ramp=True`` it varies linearly from each
    listed composition to the next and holds the last one afterwards.

    The schedule is a pure function of time, so a model that applies it from
    its regulation callback resumes from a checkpoint on the same composition.
    """

    initial: tuple[float, ...]
    times: tuple[float, ...] = ()
    values: tuple[tuple[float, ...], ...] = ()
    ramp: bool = False

    def __post_init__(self) -> None:
        if len(self.times) != len(self.values):
            raise ValueError("inlet schedule needs one composition per time")
        if any(not math.isfinite(time) for time in self.times) or any(
            later <= earlier for earlier, later in zip(self.times, self.times[1:], strict=False)
        ):
            raise ValueError("inlet schedule times must be finite and strictly increasing")
        if not self.initial:
            raise ValueError("inlet schedule compositions need at least one signal")
        for composition in (self.initial, *self.values):
            if len(composition) != len(self.initial):
                raise ValueError("inlet schedule compositions must share one signal count")
            if any(not math.isfinite(value) or value < 0.0 for value in composition):
                raise ValueError("inlet schedule concentrations must be finite and non-negative")

    @classmethod
    def pulse(
        cls,
        baseline: Sequence[float],
        pulse: Sequence[float],
        *,
        start: float,
        end: float,
    ) -> InletSchedule:
        """Baseline media, replaced by the pulse composition from ``start`` until ``end``."""

        return cls(tuple(baseline), (start, end), (tuple(pulse), tuple(baseline)))

    def values_at(self, time: float) -> list[float]:
        """The inlet concentration of every signal at a model time."""

        if not math.isfinite(time):
            raise ValueError("inlet schedule time must be finite")
        passed = sum(1 for change in self.times if change <= time)
        if passed == 0:
            return list(self.initial)
        current = self.values[passed - 1]
        if not self.ramp or passed == len(self.times):
            return list(current)
        following = self.values[passed]
        begin, finish = self.times[passed - 1], self.times[passed]
        fraction = (time - begin) / (finish - begin)
        return [
            before + (after - before) * fraction
            for before, after in zip(current, following, strict=True)
        ]


class _ChannelDevice:
    """Fluid fed by a straight channel, projected onto engine inputs.

    Subclasses describe their geometry with `_solid`, a predicate over a
    voxel's center and half extents, and inherit the projection of that
    geometry into a solid mask, a solved flow field, and inlet and outlet
    boundaries. Media enters through the grid face `_inlet_face` names, the
    lower y face unless a subclass says otherwise, and leaves through the face
    opposite.
    """

    __slots__ = ()

    mean_flow_speed: float

    def _inlet_face(self) -> GridFace:
        """The grid face media enters through."""

        return GridFace.Y_LOWER

    def _solid(self, px: float, py: float, pz: float, half: tuple[float, float, float]) -> bool:
        """Whether a voxel lies entirely inside a wall."""

        raise NotImplementedError

    def set_inlet_values(self, simulation: Simulation, values: Sequence[float]) -> None:
        """Set the concentration of every signal in the media entering at the inlet.

        The simulation's grid must come from `apply_to_grid`, which makes the
        device's inlet face a fixed boundary.
        """

        simulation.set_signal_boundary_values(self._inlet_face(), list(values))

    def _wall_boxes(
        self, simulation: Simulation, blocks: _Blocks, region: ConstraintRegion
    ) -> None:
        for low, high in blocks:
            box = BoxConstraintInit()
            box.center = Vec3(
                *((left + right) * 0.5 for left, right in zip(low, high, strict=True))
            )
            box.half_extents = Vec3(
                *((right - left) * 0.5 for left, right in zip(low, high, strict=True))
            )
            box.coefficient = 1.0
            box.allowed_region = region
            simulation.add_box_constraint(box)

    def _rasterize(self, spec: SignalGridSpec) -> None:
        """Write the device's solid mask to a grid and clear its constant advection."""

        shape = spec.shape
        origin = spec.origin
        spacing = spec.spacing
        # Evaluate the continuous geometry at voxel centers. Do not erode walls
        # to make interpolation easier: sampling handles fluid support separately.
        half = (0.0, 0.0, 0.0)
        obstacles = [0] * (shape.x * shape.y * shape.z)
        for x in range(shape.x):
            px = origin.x + spacing.x * x
            for y in range(shape.y):
                py = origin.y + spacing.y * y
                for z in range(shape.z):
                    pz = origin.z + spacing.z * z
                    if self._solid(px, py, pz, half):
                        obstacles[x * shape.y * shape.z + y * shape.z + z] = 1
        spec.obstacles = obstacles
        spec.advection = [Vec3() for _ in range(spec.signal_count)]

    def apply_to_grid(
        self,
        spec: SignalGridSpec,
        inlet_values: list[float],
        outlet_values: list[float],
        *,
        simulation: Simulation | None = None,
        backend: BackendKind = BackendKind.CPU,
        device_index: int = 0,
    ) -> None:
        """Materialize the device's mask, boundaries, and backend-solved flow field.

        If a simulation is supplied, its native backend executes the flow solve.
        Otherwise a temporary simulation uses ``backend`` and ``device_index``.
        """

        self._rasterize(spec)
        axis, toward_lower = _INLET_FLOW[self._inlet_face()]
        lower, upper = (spec.x_lower, spec.x_upper) if axis == "x" else (spec.y_lower, spec.y_upper)
        entering, leaving = (upper, lower) if toward_lower else (lower, upper)
        entering.kind = GridBoundaryKind.FIXED
        entering.values = list(inlet_values)
        leaving.kind = GridBoundaryKind.FIXED
        leaving.values = list(outlet_values)

        if self.mean_flow_speed != 0.0:
            # The solver runs flow from the lower to the upper face of its axis
            # and reverses it for a negative speed.
            field, _ = solve_flow_field(
                spec,
                mean_inlet_speed=-self.mean_flow_speed if toward_lower else self.mean_flow_speed,
                axis=axis,
                mobility=gap_mobility(spec),
                simulation=simulation,
                backend=backend,
                device_index=device_index,
            )
            spec.velocity_field = field


@dataclass(frozen=True, slots=True)
class TrapChannelDevice(_ChannelDevice):
    """An open-sided cell trap fed by a straight flow channel.

    The channel runs along the y axis between ``channel_far_x`` and
    ``trap_open_x``. The trap cavity spans ``trap_open_x`` to ``trap_back_x``
    in x and ``-trap_half_y`` to ``trap_half_y`` in y, open toward the channel
    and sealed on its other three sides by solid blocks. Media flows along +y
    through the channel with the solved steady device flow; the dead-end trap
    interior carries only the weak circulation at its mouth and exchanges with
    the channel by diffusion through the open face.
    """

    trap_open_x: float = -60.0
    trap_back_x: float = 60.0
    trap_half_y: float = 15.0
    trap_half_z: float = 3.0
    channel_far_x: float = -100.0
    channel_half_length: float = 120.0
    wall_thickness: float = 2.0
    mean_flow_speed: float = 0.0

    def add_constraints(self, simulation: Simulation) -> None:
        """Add the device's wall constraints to a simulation."""

        # The walls reach past the chamber in z. A box pushes a cell that has
        # entered it toward its nearest face, so a wall ending level with the
        # ceiling offers a cell pressed against that ceiling an escape of zero
        # length in z, which the chamber then blocks: the cell never leaves in
        # y and a crowded trap drives it further in. Standing the walls proud
        # keeps the way out of a wall the way the cell came in.
        wall_top = self.trap_half_z + self.wall_thickness
        blocks = (
            (
                (self.trap_open_x, self.trap_half_y, -wall_top),
                (
                    self.trap_back_x + self.wall_thickness,
                    self.channel_half_length,
                    wall_top,
                ),
            ),
            (
                (self.trap_open_x, -self.channel_half_length, -wall_top),
                (
                    self.trap_back_x + self.wall_thickness,
                    -self.trap_half_y,
                    wall_top,
                ),
            ),
            (
                (
                    self.trap_back_x,
                    -self.trap_half_y - self.wall_thickness,
                    -wall_top,
                ),
                (
                    self.trap_back_x + self.wall_thickness,
                    self.trap_half_y + self.wall_thickness,
                    wall_top,
                ),
            ),
        )
        self._wall_boxes(simulation, blocks, ConstraintRegion.OUTSIDE)

        chamber = BoxConstraintInit()
        chamber.center = Vec3(
            (self.channel_far_x + self.trap_back_x + self.wall_thickness) * 0.5,
            0.0,
            0.0,
        )
        chamber.half_extents = Vec3(
            (self.trap_back_x + self.wall_thickness - self.channel_far_x) * 0.5,
            self.channel_half_length,
            self.trap_half_z,
        )
        chamber.coefficient = 1.0
        chamber.allowed_region = ConstraintRegion.INSIDE
        simulation.add_box_constraint(chamber)

    def _solid(self, px: float, py: float, pz: float, half: tuple[float, float, float]) -> bool:
        hx, hy, hz = half
        if _recedes(px + hx, self.channel_far_x, hx):
            return True
        if _reaches(px - hx, self.trap_back_x + self.wall_thickness, hx):
            return True
        if _reaches(abs(pz) - hz, self.trap_half_z, hz):
            return True
        if _reaches(px - hx, self.trap_open_x, hx) and _reaches(abs(py) - hy, self.trap_half_y, hy):
            return True
        return _reaches(px - hx, self.trap_back_x, hx)


@dataclass(frozen=True, slots=True)
class BiopixelTrapDevice(_ChannelDevice):
    """One trap of a biopixel array: a shallow monolayer cavity beside a tall channel.

    The cavity spans ``0`` to ``trap_depth`` in x with its open face at ``x = 0``
    toward the channel, ``trap_width`` in y, and only ``trap_height`` in z, so
    the colony grows as a monolayer under the cavity ceiling. The flow channel
    runs along y between ``-channel_width`` and ``0`` at the full
    ``channel_height``. The device floor is ``z = 0``.

    The default trap dimensions are the 100 by 85 by 1.65 micrometer trapping
    region reported by Prindle et al. (Nature 481, 39-44, 2012;
    doi:10.1038/nature10722). Channel dimensions, numerical wall thickness, and
    flow speed are modeling inputs rather than measurements from that study. A
    single instance makes no claim that every trap in an array has identical
    local flow or concentration boundary conditions.
    """

    trap_depth: float = 85.0
    trap_width: float = 100.0
    trap_height: float = 1.65
    channel_width: float = 100.0
    channel_height: float = 10.0
    channel_half_length: float = 150.0
    wall_thickness: float = 10.0
    mean_flow_speed: float = 0.0

    def add_constraints(self, simulation: Simulation) -> None:
        """Add the device's wall constraints to a simulation."""

        half_y = self.trap_width * 0.5
        thickness = self.wall_thickness

        floor = PlaneConstraintInit()
        floor.point = Vec3(0.0, 0.0, 0.0)
        floor.inward_normal = Vec3(0.0, 0.0, 1.0)
        simulation.add_plane_constraint(floor)

        ceiling = PlaneConstraintInit()
        ceiling.point = Vec3(0.0, 0.0, self.channel_height)
        ceiling.inward_normal = Vec3(0.0, 0.0, -1.0)
        simulation.add_plane_constraint(ceiling)

        boxes = (
            (
                (0.0, -half_y - thickness, self.trap_height),
                (
                    self.trap_depth + thickness,
                    half_y + thickness,
                    self.channel_height + thickness,
                ),
            ),
            (
                (0.0, half_y, -thickness),
                (
                    self.trap_depth + thickness,
                    self.channel_half_length,
                    self.channel_height + thickness,
                ),
            ),
            (
                (0.0, -self.channel_half_length, -thickness),
                (
                    self.trap_depth + thickness,
                    -half_y,
                    self.channel_height + thickness,
                ),
            ),
            (
                (self.trap_depth, -half_y - thickness, -thickness),
                (
                    self.trap_depth + thickness,
                    half_y + thickness,
                    self.channel_height + thickness,
                ),
            ),
        )
        self._wall_boxes(simulation, boxes, ConstraintRegion.OUTSIDE)

        chamber = BoxConstraintInit()
        chamber.center = Vec3(
            (self.trap_depth + self.wall_thickness - self.channel_width) * 0.5,
            0.0,
            self.channel_height * 0.5,
        )
        chamber.half_extents = Vec3(
            (self.channel_width + self.trap_depth + self.wall_thickness) * 0.5,
            self.channel_half_length,
            self.channel_height * 0.5,
        )
        chamber.coefficient = 1.0
        chamber.allowed_region = ConstraintRegion.INSIDE
        simulation.add_box_constraint(chamber)

    def _solid(self, px: float, py: float, pz: float, half: tuple[float, float, float]) -> bool:
        hx, hy, hz = half
        if _recedes(px + hx, -self.channel_width, hx):
            return True
        if _reaches(px - hx, self.trap_depth + self.wall_thickness, hx):
            return True
        if _recedes(pz + hz, 0.0, hz) or _reaches(pz - hz, self.channel_height, hz):
            return True
        if _reaches(px - hx, 0.0, hx) and _reaches(abs(py) - hy, self.trap_width * 0.5, hy):
            return True
        if _reaches(px - hx, self.trap_depth, hx):
            return True
        return _reaches(px - hx, 0.0, hx) and _reaches(pz - hz, self.trap_height, hz)


@dataclass(frozen=True, slots=True)
class _HeightMap:
    """The gap height over a window, as cells between the edges of its boxes.

    Cell ``(i, j)`` spans ``xs[i]`` to ``xs[i + 1]`` and ``ys[j]`` to
    ``ys[j + 1]``; its height is that of the tallest box covering it, and zero
    where no box does.
    """

    xs: tuple[float, ...]
    ys: tuple[float, ...]
    heights: tuple[tuple[float, ...], ...]
    tallest: float


@lru_cache(maxsize=16)
def _height_map(boxes: _Blocks, size: tuple[float, float]) -> _HeightMap:
    xs = tuple(sorted({0.0, size[0], *(edge[0] for box in boxes for edge in box)}))
    ys = tuple(sorted({0.0, size[1], *(edge[1] for box in boxes for edge in box)}))
    heights = [[0.0] * (len(ys) - 1) for _ in range(len(xs) - 1)]
    for low, high in boxes:
        for i in range(xs.index(low[0]), xs.index(high[0])):
            column = heights[i]
            for j in range(ys.index(low[1]), ys.index(high[1])):
                column[j] = max(column[j], high[2])
    return _HeightMap(
        xs, ys, tuple(tuple(column) for column in heights), max(box[1][2] for box in boxes)
    )


# Spans along a side of a window: (low, high) pairs.
_Spans = tuple[tuple[float, float], ...]


def _within(spans: _Spans, position: float) -> bool:
    return any(low < position < high for low, high in spans)


def _cells_at(
    edges: tuple[float, ...], position: float, below: bool, above: bool
) -> tuple[int, ...]:
    """The cells along one axis that a position lies in or between.

    A position on an edge two cells share lies between both. A position on or
    past the window's lower edge lies in the first cell when the window is
    open there (``below``), so that the geometry continues outward through an
    opening, and in no cell otherwise; likewise past its upper edge.
    """

    last = len(edges) - 2
    if position <= edges[0]:
        return (0,) if below else ()
    if position >= edges[-1]:
        return (last,) if above else ()
    cell = bisect_right(edges, position) - 1
    return (cell - 1, cell) if position == edges[cell] else (cell,)


_SIDES = (GridFace.X_LOWER, GridFace.X_UPPER, GridFace.Y_LOWER, GridFace.Y_UPPER)
_OPPOSITE = {
    GridFace.X_LOWER: GridFace.X_UPPER,
    GridFace.X_UPPER: GridFace.X_LOWER,
    GridFace.Y_LOWER: GridFace.Y_UPPER,
    GridFace.Y_UPPER: GridFace.Y_LOWER,
}
_EVERYWHERE: _Spans = ((-math.inf, math.inf),)


def _boundary(spec: SignalGridSpec, face: GridFace) -> GridBoundary:
    if face == GridFace.X_LOWER:
        return spec.x_lower
    if face == GridFace.X_UPPER:
        return spec.x_upper
    return spec.y_lower if face == GridFace.Y_LOWER else spec.y_upper


@dataclass(frozen=True, slots=True)
class LayoutPort:
    """An opening in one side of a layout device, through which media enters or leaves.

    ``side`` is the window's side as a face of the grid, and ``low`` and
    ``high`` bound the opening along it in device coordinates: along y on an x
    side and along x on a y side. `microsimulator.design.DesignCrossing` gives
    all three for each channel a design window cuts.

    A port with a ``speed`` carries that mean speed over its open area, in
    model units, positive into the device and negative out of it. A port
    without one holds ``pressure`` and carries whatever flow that draws; a
    device needs at least one, usually its outlet at the default of zero.
    ``values`` is the concentration of every signal in the media just outside
    the port, which is what enters where the port's flow runs inward. Left
    unstated, an inward port takes the inlet values, and any other the outlet
    values, that `LayoutDevice.apply_to_grid` is given.
    """

    side: GridFace
    low: float
    high: float
    speed: float | None = None
    pressure: float = 0.0
    values: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class LayoutDevice(_ChannelDevice):
    """Fluid drawn as boxes on a common floor, such as a window of a chip design.

    ``boxes`` are (low corner, high corner) pairs in device coordinates. The
    window spans ``0`` to ``size[0]`` in x and ``0`` to ``size[1]`` in y, the
    floor is ``z = 0``, and every box rises from the floor: a chip cast on a
    mold and bonded to glass has all its cavities, whatever their depth, open
    onto the glass. Where boxes overlap the fluid is as high as the tallest.
    `microsimulator.design.DesignWindow` supplies ``boxes`` and ``size`` for a
    part of a design.

    Flow is stated one of two ways. With ``ports``, media enters and leaves
    through those openings in the window's sides, each at its own speed and
    with its own media, as where several channels meet. The geometry continues
    past the window through each port, and the rest of every side is wall.
    Without ports, media enters through the whole ``inlet`` face at
    ``mean_flow_speed`` and leaves through the face opposite; both must cut
    fluid, the geometry at each continues past the window unchanged, and the
    window's other two sides are walls. Either way, fluid the window cuts at a
    wall ends there. Speeds are in model units, as for every device: a speed a
    design reports in physical units is converted with the model's own choice
    of seconds per model time unit.

    Walls, floor and ceiling follow the box edges exactly. A voxel is fluid
    when its center lies inside the fluid, so a center on a wall plane is solid
    and a center on the seam where two boxes meet is fluid where both are.
    """

    boxes: _Blocks
    size: tuple[float, float]
    inlet: GridFace = GridFace.Y_LOWER
    wall_thickness: float = 10.0
    mean_flow_speed: float = 0.0
    ports: tuple[LayoutPort, ...] = ()

    def __post_init__(self) -> None:
        if len(self.size) != 2 or not all(
            math.isfinite(extent) and extent > 0.0 for extent in self.size
        ):
            raise ValueError("a layout device's size must be two positive finite extents")
        if self.inlet not in _INLET_FLOW:
            raise ValueError("a layout device's inlet must be a lower or upper x or y face")
        if not math.isfinite(self.wall_thickness) or self.wall_thickness <= 0.0:
            raise ValueError("a layout device's wall thickness must be positive and finite")
        if not math.isfinite(self.mean_flow_speed):
            raise ValueError("a layout device's mean flow speed must be finite")
        if not self.boxes:
            raise ValueError("a layout device needs at least one fluid box")
        for low, high in self.boxes:
            if not all(math.isfinite(value) for value in (*low, *high)):
                raise ValueError("a fluid box's corners must be finite")
            if low[2] != 0.0:
                raise ValueError("a fluid box must rise from the floor at z = 0")
            if any(far <= near for near, far in zip(low, high, strict=True)):
                raise ValueError("a fluid box's high corner must lie beyond its low corner")
            if low[0] < 0.0 or low[1] < 0.0 or high[0] > self.size[0] or high[1] > self.size[1]:
                raise ValueError("a fluid box must lie inside the window")
        held = _height_map(self.boxes, self.size)
        if self.ports:
            self._check_ports()
            return
        along_x = _INLET_FLOW[self.inlet][0] == "x"
        if along_x:
            ends = (held.heights[0], held.heights[-1])
        else:
            ends = tuple(tuple(column[at] for column in held.heights) for at in (0, -1))
        if any(max(end) == 0.0 for end in ends):
            raise ValueError("a layout device's inlet and outlet faces must both cut fluid")

    def _check_ports(self) -> None:
        if self.mean_flow_speed != 0.0:
            raise ValueError("a layout device states its flow through ports or along one axis")
        for index, port in enumerate(self.ports):
            if port.side not in _SIDES:
                raise ValueError("a layout port's side must be a lower or upper x or y face")
            on_x = port.side in (GridFace.X_LOWER, GridFace.X_UPPER)
            extent = self.size[1] if on_x else self.size[0]
            if not (math.isfinite(port.low) and math.isfinite(port.high)):
                raise ValueError("a layout port's span must be finite")
            if port.high <= port.low or port.low < 0.0 or port.high > extent:
                raise ValueError("a layout port's span must lie along its side of the window")
            if port.speed is not None and not math.isfinite(port.speed):
                raise ValueError("a layout port's speed must be finite")
            if not math.isfinite(port.pressure):
                raise ValueError("a layout port's pressure must be finite")
            if port.values is not None and any(
                not math.isfinite(value) or value < 0.0 for value in port.values
            ):
                raise ValueError("a layout port's values must be finite and non-negative")
            if any(
                other.side == port.side and other.low < port.high and port.low < other.high
                for other in self.ports[:index]
            ):
                raise ValueError("layout ports on one side must not overlap")
            if not self._opens_fluid(port):
                raise ValueError("a layout port must open onto fluid")
        if all(port.speed is not None for port in self.ports):
            raise ValueError("a layout device needs a port that holds a pressure")

    def _opens_fluid(self, port: LayoutPort) -> bool:
        """Whether fluid lies against the window's side somewhere within a port's span."""

        held = _height_map(self.boxes, self.size)
        on_x = port.side in (GridFace.X_LOWER, GridFace.X_UPPER)
        at = 0 if port.side in (GridFace.X_LOWER, GridFace.Y_LOWER) else -1
        edges = held.ys if on_x else held.xs
        beside = held.heights[at] if on_x else tuple(column[at] for column in held.heights)
        return any(
            height > 0.0 and edges[cell] < port.high and port.low < edges[cell + 1]
            for cell, height in enumerate(beside)
        )

    def _inlet_face(self) -> GridFace:
        return self.inlet

    def _openings(self) -> dict[GridFace, _Spans]:
        """Along each side of the window, where the geometry continues past it."""

        if self.ports:
            return {
                side: tuple((port.low, port.high) for port in self.ports if port.side == side)
                for side in _SIDES
            }
        open_sides = (self.inlet, _OPPOSITE[self.inlet])
        return {side: _EVERYWHERE if side in open_sides else () for side in _SIDES}

    @property
    def height(self) -> float:
        """The height of the tallest fluid, where the device's ceiling stands."""

        return _height_map(self.boxes, self.size).tallest

    def gap_height(self, x: float, y: float) -> float:
        """The height of the fluid above a point of the floor, zero inside a wall."""

        held = _height_map(self.boxes, self.size)
        openings = self._openings()
        columns = _cells_at(
            held.xs,
            x,
            _within(openings[GridFace.X_LOWER], y),
            _within(openings[GridFace.X_UPPER], y),
        )
        rows = _cells_at(
            held.ys,
            y,
            _within(openings[GridFace.Y_LOWER], x),
            _within(openings[GridFace.Y_UPPER], x),
        )
        if not columns or not rows:
            return 0.0
        return min(held.heights[i][j] for i in columns for j in rows)

    def lattice(self, spacing: tuple[float, float, float]) -> tuple[Vec3, GridShape]:
        """The origin and shape of a signal grid that covers the device.

        In z, and along x and y for a device with ports, site centers start
        half a voxel outside the window and continue to the first center past
        its far side, so every position a cell can reach has solid sites beyond
        it to sample against, and each port opens through that outer layer of
        sites. For a device without ports, site centers along the flow axis run
        from the inlet face to the outlet face instead.
        """

        if not all(math.isfinite(step) and step > 0.0 for step in spacing):
            raise ValueError("lattice spacing must be positive and finite")
        along = None if self.ports else _INLET_FLOW[self.inlet][0]

        def through(extent: float, step: float) -> tuple[float, int]:
            return step * 0.5, max(1, round(extent / step))

        def across(extent: float, step: float) -> tuple[float, int]:
            return -step * 0.5, math.floor(extent / step + 0.5) + 2

        x = (through if along == "x" else across)(self.size[0], spacing[0])
        y = (through if along == "y" else across)(self.size[1], spacing[1])
        z = across(self.height, spacing[2])
        shape = GridShape()
        shape.x, shape.y, shape.z = x[1], y[1], z[1]
        return Vec3(x[0], y[0], z[0]), shape

    def flow_ports(self, spec: SignalGridSpec) -> list[FlowPort]:
        """The device's ports as openings in a grid's faces, for the flow solvers."""

        ports: list[FlowPort] = []
        for port in self.ports:
            on_x = port.side in (GridFace.X_LOWER, GridFace.X_UPPER)
            origin = spec.origin.y if on_x else spec.origin.x
            step = spec.spacing.y if on_x else spec.spacing.x
            count = spec.shape.y if on_x else spec.shape.x
            # The sites whose centers lie strictly within the port's span.
            begin = max(0, math.floor((port.low - origin) / step) + 1)
            end = min(count, math.ceil((port.high - origin) / step))
            if end <= begin:
                raise ValueError("a layout port spans no site of the grid; refine the grid")
            ports.append(
                FlowPort(
                    port.side,
                    GridFacePatch((begin, 0), (end, spec.shape.z)),
                    FlowPortKind.PRESSURE if port.speed is None else FlowPortKind.SPEED,
                    port.pressure if port.speed is None else port.speed,
                )
            )
        return ports

    def apply_to_grid(
        self,
        spec: SignalGridSpec,
        inlet_values: list[float],
        outlet_values: list[float],
        *,
        simulation: Simulation | None = None,
        backend: BackendKind = BackendKind.CPU,
        device_index: int = 0,
        resolved: bool = False,
        tolerance: float = 1.0e-6,
    ) -> None:
        """Materialize the device's mask, boundaries, and backend-solved flow field.

        For a device with ports, every side that holds a port becomes a fixed
        boundary carrying ``outlet_values``, with a patch at each port carrying
        that port's media, and `solve_flow` solves the flow through the ports.
        A grid from `lattice` places the ports on its faces. A device without
        ports is projected as every channel device is.
        """

        if not self.ports:
            if resolved:
                raise ValueError("a layout device without ports solves the shallow flow")
            _ChannelDevice.apply_to_grid(
                self,
                spec,
                inlet_values,
                outlet_values,
                simulation=simulation,
                backend=backend,
                device_index=device_index,
            )
            return

        self._rasterize(spec)
        flow_ports = self.flow_ports(spec)
        for side in _SIDES:
            patches = [
                GridBoundaryPatch(
                    opening.patch,
                    list(
                        port.values
                        if port.values is not None
                        else inlet_values
                        if port.speed is not None and port.speed > 0.0
                        else outlet_values
                    ),
                )
                for port, opening in zip(self.ports, flow_ports, strict=True)
                if port.side == side
            ]
            if not patches:
                continue
            boundary = _boundary(spec, side)
            boundary.kind = GridBoundaryKind.FIXED
            boundary.values = list(outlet_values)
            boundary.patches = patches
        if any(port.speed not in (None, 0.0) or port.pressure != 0.0 for port in self.ports):
            self.solve_flow(
                spec,
                simulation=simulation,
                backend=backend,
                device_index=device_index,
                resolved=resolved,
                tolerance=tolerance,
            )

    def solve_flow(
        self,
        spec: SignalGridSpec,
        *,
        simulation: Simulation | None = None,
        backend: BackendKind = BackendKind.CPU,
        device_index: int = 0,
        resolved: bool = False,
        tolerance: float = 1.0e-6,
    ) -> DepthAveragedPortFlowResult | ResolvedPortFlowResult:
        """Solve the flow through the device's ports and give the grid its velocity field.

        The shallow depth-averaged solver is used, with the gap-height mobility
        of the grid's mask, or the resolved Stokes solver when ``resolved`` is
        set. The result holds the pressure, flux and mean speed of every port,
        in the order of the device's ``ports``. The grid's mask and boundaries
        must come from `apply_to_grid`. ``tolerance`` is the solver's relative
        residual; it bounds the pressure equations' residual and not the flux,
        so a long device whose port fluxes do not close asks for a smaller one.
        """

        flow_ports = self.flow_ports(spec)
        result: DepthAveragedPortFlowResult | ResolvedPortFlowResult
        if resolved:
            result = solve_port_stokes_field(
                spec,
                flow_ports,
                tolerance=tolerance,
                simulation=simulation,
                backend=backend,
                device_index=device_index,
            )
        else:
            result = solve_port_flow_field(
                spec,
                flow_ports,
                mobility=gap_mobility(spec),
                tolerance=tolerance,
                simulation=simulation,
                backend=backend,
                device_index=device_index,
            )
        spec.velocity_field = result.field
        return result

    def set_port_values(self, simulation: Simulation, port: int, values: Sequence[float]) -> None:
        """Set the concentration of every signal in the media just outside one port.

        The simulation's grid must come from `apply_to_grid`. ``port`` indexes
        the device's ``ports``.
        """

        side = self.ports[port].side
        patch = sum(1 for other in self.ports[:port] if other.side == side)
        simulation.set_signal_boundary_patch_values(side, patch, list(values))

    def add_constraints(self, simulation: Simulation) -> None:
        """Add the device's wall constraints to a simulation."""

        held = _height_map(self.boxes, self.size)
        thickness = self.wall_thickness

        floor = PlaneConstraintInit()
        floor.point = Vec3(0.0, 0.0, 0.0)
        floor.inward_normal = Vec3(0.0, 0.0, 1.0)
        simulation.add_plane_constraint(floor)

        ceiling = PlaneConstraintInit()
        ceiling.point = Vec3(0.0, 0.0, held.tallest)
        ceiling.inward_normal = Vec3(0.0, 0.0, -1.0)
        simulation.add_plane_constraint(ceiling)

        # Everything lower than the ceiling is solid above its own height: a
        # wall where that height is zero, the roof of a shallow cavity
        # otherwise. Runs of cells of one height along x that repeat along y
        # merge into one block. Blocks stand proud of the ceiling and of the
        # window's sides, and walls of the floor, so that a cell pressed into
        # one leaves by the face it entered through.
        def edge(edges: tuple[float, ...], at: int) -> float:
            if at == 0:
                return edges[0] - thickness
            return edges[at] + thickness if at == len(edges) - 1 else edges[at]

        open_runs: dict[tuple[int, int, float], int] = {}
        blocks: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []

        def close(run: tuple[int, int, float], begun: int, ended: int) -> None:
            first, past, height = run
            blocks.append(
                (
                    (edge(held.xs, first), edge(held.ys, begun), height if height else -thickness),
                    (edge(held.xs, past), edge(held.ys, ended), held.tallest + thickness),
                )
            )

        for j in range(len(held.ys) - 1):
            runs: list[tuple[int, int, float]] = []
            first = 0
            for i in range(1, len(held.xs)):
                height = held.heights[first][j]
                if i == len(held.xs) - 1 or held.heights[i][j] != height:
                    if height < held.tallest:
                        runs.append((first, i, height))
                    first = i
            for run in [run for run in open_runs if run not in runs]:
                close(run, open_runs.pop(run), j)
            for run in runs:
                open_runs.setdefault(run, j)
        for run, begun in open_runs.items():
            close(run, begun, len(held.ys) - 1)
        self._wall_boxes(simulation, tuple(blocks), ConstraintRegion.OUTSIDE)

        chamber = BoxConstraintInit()
        chamber.center = Vec3(self.size[0] * 0.5, self.size[1] * 0.5, held.tallest * 0.5)
        chamber.half_extents = Vec3(self.size[0] * 0.5, self.size[1] * 0.5, held.tallest * 0.5)
        chamber.coefficient = 1.0
        chamber.allowed_region = ConstraintRegion.INSIDE
        simulation.add_box_constraint(chamber)

    def _solid(self, px: float, py: float, pz: float, half: tuple[float, float, float]) -> bool:
        del half
        return pz <= 0.0 or pz >= self.gap_height(px, py)
