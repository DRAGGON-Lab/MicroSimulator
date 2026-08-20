"""Staggered-grid Stokes-Brinkman flow solve for device grids.

This is the high-fidelity companion to the Hele-Shaw solver in
`cellmodeller2.flow`: it resolves the full velocity field, including viscous
boundary layers on every wall, instead of depth-averaging them into a mobility
closure. The momentum balance is inertia-free Stokes with an optional Brinkman
drag,

```text
mu * lap(v) - mu * d(x) * v - grad(p) = 0        div(v) = 0
```

discretized on the marker-and-cell staggering the engine already uses:
velocity components on faces, pressure at cell centers. No-slip walls are the
obstacle voxel boundaries and every non-flow domain edge; wall planes sit half
a spacing beyond the outermost site centers, which is exactly where the device
helpers author their floors and ceilings. Normal velocities on fluid-solid
faces are eliminated at zero, and tangential components see the wall through
reflected ghosts. Pressure is fixed beyond the fluid boundary faces of the
flow axis (inlet one, outlet zero) with a zero-gradient outflow condition on
the normal velocity, so a fully developed channel reproduces its exact
profile shape. Because the problem is linear, the solution is rescaled to a
requested mean inlet speed and viscosity drops out; the drag field ``d`` is an
inverse permeability with units of one over length squared.

The saddle-point system is solved by the pressure Schur complement: an outer
conjugate gradient on `S = D A^-1 D^T` (symmetric positive definite), with
each application solving three independent component Laplacians by inner
conjugate gradient. Everything is matrix-free NumPy. This costs far more than
the Hele-Shaw solve - it is the build-time and benchmark solver, not the
per-hundred-steps re-solve inside a running model.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from ._core import SignalGridSpec, SignalGridVelocityField
from .flow import (
    FlowError,
    _BoolGrid,
    _conjugate_gradient,
    _FloatGrid,
    _flow_axis_index,
    _kozeny_carman_drag,
    _RodLike,
    colony_volume_fraction,
)

# Either a float field or a mask: shifting and reflecting treat them alike.
_Grid = TypeVar("_Grid", _FloatGrid, _BoolGrid)


@dataclass(frozen=True, slots=True)
class StokesSolveReport:
    outer_iterations: int
    inner_iterations: int
    divergence_rms: float
    mean_inlet_speed: float
    max_speed: float
    min_gap_voxels: int
    """Fluid voxels across the narrowest channel, transverse to the flow.

    A no-slip profile needs several voxels to resolve, so this number bounds
    the solve's accuracy: a channel one voxel across carries roughly two and a
    half times the flux its true parabolic profile would, four voxels bring
    that within about ten percent, and eight within a few percent. Below four,
    the depth-averaged Hele-Shaw closure of `cellmodeller2.flow` is the more
    accurate model of a shallow channel.
    """


def colony_drag(
    spec: SignalGridSpec,
    cells: Iterable[_RodLike],
    *,
    drag_coefficient: float,
    max_volume_fraction: float = 0.9,
) -> list[float]:
    """Build the Brinkman drag field (inverse permeability) from the colony.

    The colony's per-voxel volume fraction ``phi`` sets a Kozeny-Carman style
    drag ``drag_coefficient * phi^2 / (1 - phi)^3``. The coefficient carries
    units of one over length squared and is a modeling choice. Solid voxels
    stay at zero (they are walls, not porous media).
    """

    fraction = colony_volume_fraction(spec, cells, max_volume_fraction=max_volume_fraction)
    drag = _kozeny_carman_drag(fraction, drag_coefficient)
    obstacles = spec.obstacles
    if obstacles:
        dims = (spec.shape.x, spec.shape.y, spec.shape.z)
        solid = np.asarray(obstacles, dtype=np.uint8).reshape(dims) != 0
        drag[solid] = 0.0
    return [float(value) for value in drag.ravel()]


def _minimum_gap_voxels(fluid: _BoolGrid, dims: tuple[int, int, int], flow_axis: int) -> int:
    """The shortest run of fluid voxels across any axis transverse to the flow."""

    shortest = 0
    for axis in range(3):
        if axis == flow_axis or dims[axis] <= 1:
            continue
        lines = np.moveaxis(fluid, axis, -1)
        padded = np.zeros((*lines.shape[:-1], lines.shape[-1] + 2), dtype=np.int8)
        padded[..., 1:-1] = lines
        edges = np.diff(padded, axis=-1)
        starts = np.argwhere(edges == 1)
        ends = np.argwhere(edges == -1)
        if starts.size == 0:
            continue
        runs = ends[:, -1] - starts[:, -1]
        axis_shortest = int(runs.min())
        shortest = axis_shortest if shortest == 0 else min(shortest, axis_shortest)
    return shortest


class _StokesOperator:
    """The masked component Laplacians, divergence, and gradient of one problem."""

    def __init__(
        self,
        dims: tuple[int, int, int],
        spacing: tuple[float, float, float],
        fluid: _BoolGrid,
        drag: _FloatGrid,
        flow_axis: int,
    ) -> None:
        self.dims = dims
        self.spacing = spacing
        self.fluid = fluid
        # Collapsed axes (a single site) are invariant directions, matching
        # the engine's transport semantics: no wall reflection across them.
        self.live_axes = tuple(a for a in range(3) if dims[a] > 1)

        self.active: list[_BoolGrid] = []
        self.exists: list[_BoolGrid] = []
        self.face_drag: list[_FloatGrid] = []
        for c in range(3):
            lower = self._cell_beside(fluid, c, -1)
            upper = self._cell_beside(fluid, c, +1)
            active = lower & upper
            exists = lower | upper
            if c == flow_axis:
                # The flow-axis boundary faces carry prescribed ghost pressures
                # rather than a wall, so they stay active beside one fluid cell.
                active[self._edge_slice(c, 0)] = fluid[self._edge_slice(c, 0)]
                active[self._edge_slice(c, -1)] = fluid[self._edge_slice(c, -1)]
            self.active.append(active)
            self.exists.append(exists)
            drag_low = self._cell_beside_values(drag, c, -1)
            drag_high = self._cell_beside_values(drag, c, +1)
            counts = lower.astype(np.float64) + upper.astype(np.float64)
            face_drag = np.divide(
                drag_low + drag_high,
                counts,
                out=np.zeros_like(drag_low),
                where=counts > 0.0,
            )
            self.face_drag.append(face_drag)

    def _face_dims(self, c: int) -> tuple[int, int, int]:
        dims = list(self.dims)
        dims[c] += 1
        return dims[0], dims[1], dims[2]

    def _edge_slice(self, axis: int, edge: int) -> tuple[slice, slice, slice]:
        index: list[slice] = [slice(None)] * 3
        index[axis] = slice(0, 1) if edge == 0 else slice(-1, None)
        return index[0], index[1], index[2]

    def _cell_beside(self, cells: _BoolGrid, c: int, side: int) -> _BoolGrid:
        """Whether the cell on ``side`` of each c-face exists and is fluid."""

        face_dims = self._face_dims(c)
        result = np.zeros(face_dims, dtype=bool)
        target: list[slice] = [slice(None)] * 3
        target[c] = slice(1, None) if side < 0 else slice(0, -1)
        result[target[0], target[1], target[2]] = cells
        return result

    def _cell_beside_values(self, values: _FloatGrid, c: int, side: int) -> _FloatGrid:
        face_dims = self._face_dims(c)
        result = np.zeros(face_dims, dtype=np.float64)
        target: list[slice] = [slice(None)] * 3
        target[c] = slice(1, None) if side < 0 else slice(0, -1)
        result[target[0], target[1], target[2]] = values
        return result

    def _shift(self, field: _Grid, axis: int, offset: int) -> _Grid:
        """The field sampled at ``index + offset`` along ``axis``, zero beyond.

        Values past the array edge read as zero, and a mask shifted this way
        reads as false, which is the wall the reflection tests look for.
        """

        result = np.zeros_like(field)
        source: list[slice] = [slice(None)] * 3
        target: list[slice] = [slice(None)] * 3
        if offset > 0:
            source[axis] = slice(1, None)
            target[axis] = slice(0, -1)
        else:
            source[axis] = slice(0, -1)
            target[axis] = slice(1, None)
        result[target[0], target[1], target[2]] = field[source[0], source[1], source[2]]
        return result

    def apply_momentum(self, c: int, u: _FloatGrid) -> _FloatGrid:
        """``A u = -lap(u) + d u`` on active c-faces, zero elsewhere."""

        active = self.active[c]
        result = self.face_drag[c] * u
        for a in self.live_axes:
            inv_h2 = 1.0 / (self.spacing[a] * self.spacing[a])
            for offset in (-1, +1):
                neighbor = self._shift(u, a, offset)
                if a == c:
                    # Along the component axis neighbor faces hold genuine
                    # velocities (zero on walls). Active faces on the array
                    # edge - the flow-axis inlet and outlet - use a
                    # zero-gradient ghost equal to the face value.
                    edge = self._edge_slice(a, 0 if offset < 0 else -1)
                    ghost = np.zeros_like(u)
                    ghost[edge] = u[edge]
                    neighbor = neighbor + ghost
                else:
                    # Across the component axis a neighbor location with no
                    # adjacent fluid cell lies inside a wall whose plane sits
                    # half a spacing away: reflect for no-slip.
                    reflect = ~self._shift(self.exists[c], a, offset)
                    neighbor = np.where(reflect, -u, neighbor)
                result -= (neighbor - u) * inv_h2
        result[~active] = 0.0
        return result

    def divergence(self, velocity: list[_FloatGrid]) -> _FloatGrid:
        result = np.zeros(self.dims, dtype=np.float64)
        for c in range(3):
            faces = velocity[c]
            inv_h = 1.0 / self.spacing[c]
            upper: list[slice] = [slice(None)] * 3
            lower: list[slice] = [slice(None)] * 3
            upper[c] = slice(1, None)
            lower[c] = slice(0, -1)
            result += (
                faces[upper[0], upper[1], upper[2]] - faces[lower[0], lower[1], lower[2]]
            ) * inv_h
        result[~self.fluid] = 0.0
        return result

    def gradient(self, pressure: _FloatGrid) -> list[_FloatGrid]:
        """``-D^T p`` per component: the pressure gradient on active faces."""

        fields: list[_FloatGrid] = []
        clean = np.where(self.fluid, pressure, 0.0)
        for c in range(3):
            face = np.zeros(self._face_dims(c), dtype=np.float64)
            face += self._cell_beside_values(clean, c, +1)
            face -= self._cell_beside_values(clean, c, -1)
            face *= 1.0 / self.spacing[c]
            face[~self.active[c]] = 0.0
            fields.append(face)
        return fields


def solve_stokes_field(
    spec: SignalGridSpec,
    *,
    mean_inlet_speed: float,
    axis: str = "y",
    drag: Sequence[float] | None = None,
    tolerance: float = 1.0e-8,
    max_outer_iterations: int = 500,
    inner_tolerance: float = 1.0e-10,
    max_inner_iterations: int = 50_000,
) -> tuple[SignalGridVelocityField, StokesSolveReport]:
    """Solve the staggered Stokes-Brinkman flow and return the velocity field.

    Flow runs from the lower to the upper boundary of ``axis``; a negative
    ``mean_inlet_speed`` reverses it. ``drag`` optionally gives one Brinkman
    drag value (inverse permeability, units 1/length^2) per site; omitted or
    zero drag is pure Stokes.
    """

    if not math.isfinite(mean_inlet_speed) or mean_inlet_speed == 0.0:
        raise FlowError("mean inlet speed must be finite and nonzero")
    flow_axis = _flow_axis_index(spec, axis)

    dims = (spec.shape.x, spec.shape.y, spec.shape.z)
    spacing = (spec.spacing.x, spec.spacing.y, spec.spacing.z)
    sites = dims[0] * dims[1] * dims[2]
    obstacles = spec.obstacles
    if obstacles:
        if len(obstacles) != sites:
            raise FlowError("obstacles must hold one flag per grid site")
        fluid = np.asarray(obstacles, dtype=np.uint8).reshape(dims) == 0
    else:
        fluid = np.ones(dims, dtype=bool)
    if drag is None:
        drag_grid = np.zeros(dims, dtype=np.float64)
    else:
        if len(drag) != sites:
            raise FlowError("drag must hold one value per grid site")
        drag_grid = np.asarray(drag, dtype=np.float64).reshape(dims)
        if not bool(np.all(np.isfinite(drag_grid))) or bool(np.any(drag_grid < 0.0)):
            raise FlowError("drag values must be finite and non-negative")
        drag_grid = np.where(fluid, drag_grid, 0.0)

    operator = _StokesOperator(dims, spacing, fluid, drag_grid, flow_axis)
    if not bool(np.any(operator.active[flow_axis][operator._edge_slice(flow_axis, 0)])):
        raise FlowError("the inlet boundary is entirely blocked")

    # Momentum right-hand side from the prescribed inlet and outlet ghost
    # pressures (one and zero).
    force: list[_FloatGrid] = [
        np.zeros(operator._face_dims(c), dtype=np.float64) for c in range(3)
    ]
    inlet_slice = operator._edge_slice(flow_axis, 0)
    inlet_active = operator.active[flow_axis][inlet_slice]
    force[flow_axis][inlet_slice] = np.where(
        inlet_active, 1.0 / spacing[flow_axis], 0.0
    )

    # Momentum diagonals for the inner Jacobi preconditioner.
    diagonals: list[_FloatGrid] = []
    for c in range(3):
        diagonal = operator.face_drag[c].copy()
        for a in operator.live_axes:
            diagonal += 2.0 / (spacing[a] * spacing[a])
        diagonals.append(diagonal)

    inner_total = 0

    def solve_momentum(rhs: list[_FloatGrid]) -> list[_FloatGrid]:
        nonlocal inner_total
        solution: list[_FloatGrid] = []
        for c in range(3):
            component, iterations, _ = _conjugate_gradient(
                lambda u, c=c: operator.apply_momentum(c, u),
                rhs[c],
                diagonals[c],
                inner_tolerance,
                max_inner_iterations,
                mask=operator.active[c],
                label="stokes momentum",
            )
            inner_total += iterations
            solution.append(component)
        return solution

    particular = solve_momentum(force)
    schur_rhs = -operator.divergence(particular)

    def apply_schur(q: _FloatGrid) -> _FloatGrid:
        # gradient() is -D^T, so negating the divergence gives S = D A^-1 D^T.
        return -operator.divergence(solve_momentum(operator.gradient(q)))

    pressure, outer_iterations, _ = _conjugate_gradient(
        apply_schur,
        schur_rhs,
        np.ones(dims, dtype=np.float64),
        tolerance,
        max_outer_iterations,
        mask=fluid,
        label="stokes pressure",
    )

    correction = solve_momentum(operator.gradient(pressure))
    velocity = [particular[c] - correction[c] for c in range(3)]
    divergence = operator.divergence(velocity)
    divergence_rms = float(np.sqrt(np.mean(divergence[fluid] ** 2))) if bool(
        np.any(fluid)
    ) else 0.0

    inlet_values = velocity[flow_axis][inlet_slice]
    open_inlet = operator.active[flow_axis][inlet_slice]
    solved_mean = float(np.mean(inlet_values[open_inlet]))
    # The inlet must carry a real share of whatever the solve moved anywhere,
    # which makes the test independent of drag, spacing, and grid size.
    peak = max(float(np.max(np.abs(component))) for component in velocity)
    if peak == 0.0 or solved_mean <= 1.0e-9 * peak:
        raise FlowError("the device carries no through-flow: the outlet is unreachable")
    factor = mean_inlet_speed / solved_mean
    scaled = [component * factor for component in velocity]

    field = SignalGridVelocityField()
    field.x_faces = [float(value) for value in scaled[0].ravel()]
    field.y_faces = [float(value) for value in scaled[1].ravel()]
    field.z_faces = [float(value) for value in scaled[2].ravel()]
    max_speed = max(float(np.max(np.abs(component))) for component in scaled)
    report = StokesSolveReport(
        outer_iterations=outer_iterations,
        inner_iterations=inner_total,
        divergence_rms=divergence_rms * abs(factor),
        mean_inlet_speed=mean_inlet_speed,
        max_speed=max_speed,
        min_gap_voxels=_minimum_gap_voxels(fluid, dims, flow_axis),
    )
    return field, report
