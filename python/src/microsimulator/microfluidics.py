"""Authoring helpers for microfluidic device models.

A device is described once in physical coordinates and then projected into the
engine's typed inputs: box wall constraints for mechanics, a solid mask for the
signal grid, and a numerically solved face-staggered flow field for advection
(the shallow, depth-integrated solve of `microsimulator.flow`, with conservative
fluxes through supported column geometries). The runtime receives only materialized data;
every predicate here is authoring-time.

Voxel centers are classified against the same physical wall predicates used by
mechanics. For axis-aligned planes the staircase wall offset is at most half a
voxel spacing. Curved geometry requires convergence checks; unresolved narrow
features cannot be recovered by renaming or enlarging the fluid region.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    BoxConstraintInit,
    ConstraintRegion,
    GridBoundaryKind,
    PlaneConstraintInit,
    SignalGridSpec,
    Simulation,
    Vec3,
)
from .flow import gap_mobility, solve_flow_field

# A voxel edge that lands on a wall plane belongs to the wall, so the voxel
# tests admit a rounding margin: without it a wall drawn exactly on a lattice
# face resolves as solid or fluid according to float rounding.
_EDGE_TOLERANCE = 1.0e-6

# Wall blocks as (low corner, high corner) pairs in device coordinates.
_Blocks = tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]


def _reaches(edge: float, wall: float, half: float) -> bool:
    """Whether a voxel's lower edge has reached a wall lying above it."""

    return edge >= wall - _EDGE_TOLERANCE * half


def _recedes(edge: float, wall: float, half: float) -> bool:
    """Whether a voxel's upper edge has reached a wall lying below it."""

    return edge <= wall + _EDGE_TOLERANCE * half


class _ChannelDevice:
    """A trap fed by a straight channel along y, projected onto engine inputs.

    Subclasses describe their geometry with `_solid`, a predicate over a
    voxel's center and half extents, and inherit the projection of that
    geometry into a solid mask, a solved flow field, and inlet and outlet
    boundaries.
    """

    __slots__ = ()

    mean_flow_speed: float

    def _solid(self, px: float, py: float, pz: float, half: tuple[float, float, float]) -> bool:
        """Whether a voxel lies entirely inside a wall."""

        raise NotImplementedError

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
        spec.y_lower.kind = GridBoundaryKind.FIXED
        spec.y_lower.values = list(inlet_values)
        spec.y_upper.kind = GridBoundaryKind.FIXED
        spec.y_upper.values = list(outlet_values)

        if self.mean_flow_speed != 0.0:
            field, _ = solve_flow_field(
                spec,
                mean_inlet_speed=self.mean_flow_speed,
                axis="y",
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
