"""Depth-integrated shallow flow on a Cartesian device grid.

The pressure has one unknown per x/y fluid column. The native solver solves
``div_xy(q) = 0`` with ``q = -H*m*grad_xy(p)`` and gap height H. Default relative
mobility is proportional to H**2, giving the H**3 Hele-Shaw conductance.
User-supplied mobility must be constant through each column. Colony resistance
is a calibrated phenomenological closure, not resolved cell hydrodynamics.

Columns must be contiguous and share a planar floor; use resolved Stokes flow
for overhangs, multilayer channels, or flow along z. Harmonic conductance and
half-cell pressure boundaries define the finite-volume operator. CPU, Metal,
and CUDA solve it natively. A conservative lift distributes each integrated
face flux over its open depth and reconstructs vertical flux by continuity for
3D transport. This lift does not resolve a wall-normal velocity profile.

`solve_flow_field` drives flow along one axis, from the whole of its lower face
to the whole of its upper, and normalizes velocities to the requested mean
inlet speed. `solve_port_flow_field` drives it through ports instead: openings
in the side faces of the grid that each carry a stated mean speed or hold a
pressure, so that several inlets and outlets on any side meet in one field.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    DepthAveragedFlowParameters,
    DepthAveragedFlowReport,
    DepthAveragedPortFlowParameters,
    DepthAveragedPortFlowResult,
    FlowAxis,
    FlowPort,
    GridBoundaryKind,
    SignalGridSpec,
    SignalGridVelocityField,
    Simulation,
    Vec3,
)
from .biomass import biomass_volume

_FloatGrid = NDArray[np.float64]
_BoolGrid = NDArray[np.bool_]

_AXES = {"x": 0, "y": 1, "z": 2}
_NATIVE_AXES = {"x": FlowAxis.X, "y": FlowAxis.Y, "z": FlowAxis.Z}


class FlowError(ValueError):
    """Raised when a flow problem is ill-posed or its solve fails."""


FlowSolveReport = DepthAveragedFlowReport


class _RodLike(Protocol):
    @property
    def position(self) -> Vec3: ...
    @property
    def length(self) -> float: ...
    @property
    def radius(self) -> float: ...


def _flow_axis_index(spec: SignalGridSpec, axis: str) -> int:
    """Validate a flow problem's axis and boundary kinds, and return the axis."""

    if axis not in _AXES:
        raise FlowError("flow axis must be one of x, y, z")
    flow_axis = _AXES[axis]
    boundaries = (
        (spec.x_lower, spec.x_upper),
        (spec.y_lower, spec.y_upper),
        (spec.z_lower, spec.z_upper),
    )
    for lower, upper in boundaries:
        if lower.kind == GridBoundaryKind.PERIODIC or upper.kind == GridBoundaryKind.PERIODIC:
            raise FlowError("the flow solver does not support periodic boundaries")
    for boundary in boundaries[flow_axis]:
        if boundary.kind != GridBoundaryKind.FIXED:
            raise FlowError("the flow axis boundaries must be FIXED to act as inlet and outlet")
    return flow_axis


def _kozeny_carman_drag(fraction: _FloatGrid, drag_coefficient: float) -> _FloatGrid:
    """Kozeny-Carman style drag of a packed volume fraction."""

    if not math.isfinite(drag_coefficient) or drag_coefficient < 0.0:
        raise FlowError("drag coefficient must be finite and non-negative")
    return drag_coefficient * fraction * fraction / (1.0 - fraction) ** 3


def solve_flow_field(
    spec: SignalGridSpec,
    *,
    mean_inlet_speed: float,
    axis: str = "y",
    mobility: Sequence[float] | None = None,
    body_force: Sequence[tuple[float, float]] | None = None,
    tolerance: float = 1.0e-6,
    max_iterations: int = 50_000,
    simulation: Simulation | None = None,
    backend: BackendKind = BackendKind.CPU,
    device_index: int = 0,
) -> tuple[SignalGridVelocityField, FlowSolveReport]:
    """Solve the device flow and return the face-staggered velocity field.

    Flow runs from the lower to the upper boundary of ``axis``; a negative
    ``mean_inlet_speed`` reverses it. The grid's shape, spacing, obstacles,
    and boundary kinds are read from ``spec``; ``mobility`` optionally gives
    one relative gap-mean mobility per site (default proportional to gap height squared).
    Optional ``body_force`` gives (fx, fy) in pressure-gradient units per x/y
    column, ordered x-major, y-minor. The solver projects that forcing and
    adjusts the pressure drop to maintain the requested inlet speed. A simulation
    selects its native backend; otherwise ``backend`` and ``device_index`` select it.
    """

    _flow_axis_index(spec, axis)
    parameters = DepthAveragedFlowParameters()
    parameters.mean_inlet_speed = mean_inlet_speed
    parameters.axis = _NATIVE_AXES[axis]
    if body_force is not None:
        parameters.body_force = list(body_force)
    parameters.relative_tolerance = tolerance
    parameters.max_iterations = max_iterations
    selected = (
        simulation if simulation is not None else Simulation(backend, device_index=device_index)
    )
    try:
        result = selected.solve_depth_averaged_flow(
            spec,
            [] if mobility is None else [float(value) for value in mobility],
            parameters,
        )
    except (OverflowError, RuntimeError, ValueError) as error:
        raise FlowError(str(error)) from error
    return result.field, result.report


def solve_port_flow_field(
    spec: SignalGridSpec,
    ports: Sequence[FlowPort],
    *,
    mobility: Sequence[float] | None = None,
    tolerance: float = 1.0e-6,
    max_iterations: int = 50_000,
    simulation: Simulation | None = None,
    backend: BackendKind = BackendKind.CPU,
    device_index: int = 0,
) -> DepthAveragedPortFlowResult:
    """Solve the device flow driven through ports.

    Each port is a rectangle of boundary sites on an x or y face of the grid,
    spanning the full depth of the fluid behind it. A ``SPEED`` port carries
    its value as the mean speed over its open area, positive into the grid and
    negative out of it; a ``PRESSURE`` port holds its value just outside the
    face. At least one port must hold a pressure, and every port's face must
    be a ``FIXED`` boundary. Faces of the grid outside every port are closed.

    The result holds the face-staggered velocity field, the solve report, and
    for each port its pressure, flux and mean speed. Pressures are in the
    solver's units: with relative mobility they are not physical pressures,
    and only their differences matter. ``mobility`` is as for
    `solve_flow_field`. If a simulation is supplied, its native backend
    executes the solves; one is made for each speed port.
    """

    parameters = DepthAveragedPortFlowParameters()
    parameters.relative_tolerance = tolerance
    parameters.max_iterations = max_iterations
    selected = (
        simulation if simulation is not None else Simulation(backend, device_index=device_index)
    )
    try:
        return selected.solve_depth_averaged_port_flow(
            spec,
            list(ports),
            [] if mobility is None else [float(value) for value in mobility],
            parameters,
        )
    except (OverflowError, RuntimeError, ValueError) as error:
        raise FlowError(str(error)) from error


def gap_mobility(spec: SignalGridSpec) -> list[float]:
    """Build the Hele-Shaw gap-height mobility field of a device grid.

    In the depth-averaged closure a channel's mobility scales with the square
    of its gap height, so a shallow cavity resists through-flow far more than
    the tall channel beside it. Each z column's gap is its fluid-voxel count
    times the z spacing; every fluid voxel in the column gets the relative
    mobility ``(gap / max_gap)^2`` and solid voxels get zero.
    """

    dims = (spec.shape.x, spec.shape.y, spec.shape.z)
    obstacles = spec.obstacles
    if obstacles:
        if len(obstacles) != dims[0] * dims[1] * dims[2]:
            raise FlowError("obstacles must hold one flag per grid site")
        fluid = (np.asarray(obstacles, dtype=np.uint8).reshape(dims) == 0).astype(np.float64)
    else:
        fluid = np.ones(dims, dtype=np.float64)
    gaps = fluid.sum(axis=2, keepdims=True)
    max_gap = float(np.max(gaps))
    if max_gap == 0.0:
        raise FlowError("the grid contains no fluid sites")
    mobility = fluid * (gaps / max_gap) ** 2
    return [float(value) for value in mobility.ravel()]


def _solid_mask(spec: SignalGridSpec) -> _BoolGrid | None:
    obstacles = spec.obstacles
    if not obstacles:
        return None
    return np.asarray(obstacles, dtype=bool).reshape(spec.shape.x, spec.shape.y, spec.shape.z)


def _deposition_weights(
    spec: SignalGridSpec, position: Vec3, averaging_radius: float, solid: _BoolGrid | None
) -> tuple[tuple[slice, ...], _FloatGrid] | None:
    """Voxel integrals of the normalized tent kernel on one connected fluid region."""
    dims = (spec.shape.x, spec.shape.y, spec.shape.z)
    origin = (spec.origin.x, spec.origin.y, spec.origin.z)
    spacing = (spec.spacing.x, spec.spacing.y, spec.spacing.z)
    centers = (position.x, position.y, position.z)
    if any(not math.isfinite(v) for v in centers):
        raise FlowError("deposited positions must be finite")
    if any(
        p < o - h / 2 or p >= o + (n - 0.5) * h
        for p, o, h, n in zip(centers, origin, spacing, dims, strict=True)
    ):
        return None  # Outside the modeled volume: removal/washout is the caller's responsibility.
    slices: list[slice] = []
    weights: list[_FloatGrid] = []
    for p, o, h, n in zip(centers, origin, spacing, dims, strict=True):
        lo = max(0, math.floor((p - averaging_radius - o) / h + 0.5))
        hi = min(n, math.ceil((p + averaging_radius - o) / h + 0.5))
        edges = (o + (np.arange(lo, hi + 1) - 0.5) * h - p) / averaging_radius
        cdf = np.where(
            edges <= -1,
            0,
            np.where(
                edges < 0,
                0.5 * (edges + 1) ** 2,
                np.where(edges < 1, 1 - 0.5 * (1 - edges) ** 2, 1),
            ),
        )
        slices.append(slice(lo, hi))
        weights.append(np.diff(cdf))
    kernel = weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None, :]
    region = tuple(slices)
    if solid is not None:
        kernel[solid[region]] = 0
        # Restrict to one face-connected fluid component of the kernel support.
        connected = np.zeros(kernel.shape, dtype=bool)
        seed = tuple(int(i) for i in np.unravel_index(int(np.argmax(kernel)), kernel.shape))
        pending = [seed]
        while pending:
            index = pending.pop()
            if connected[index] or kernel[index] <= 0:
                continue
            connected[index] = True
            for axis in range(3):
                for offset in (-1, 1):
                    adjacent = list(index)
                    adjacent[axis] += offset
                    if 0 <= adjacent[axis] < kernel.shape[axis]:
                        pending.append(tuple(adjacent))
        kernel[~connected] = 0
    total = float(kernel.sum())
    if total <= 0:
        raise FlowError("biomass deposition has no connected fluid support")
    return region, kernel / total


def colony_volume_fraction(
    spec: SignalGridSpec,
    cells: Iterable[_RodLike],
    *,
    averaging_radius: float = 4.0,
) -> _FloatGrid:
    """Conservative biomass density B/voxel_volume, without density clipping.

    B is the effective biochemical volume, not geometric capsule volume.
    The tent kernel radius is in physical length units and remains fixed under
    mesh refinement. Boundary-truncated kernels are renormalized within one
    connected fluid region. Cells outside the grid's physical extent are omitted.
    """
    if not math.isfinite(averaging_radius) or averaging_radius <= 0:
        raise FlowError("averaging radius must be finite and positive")
    dims = (spec.shape.x, spec.shape.y, spec.shape.z)
    volume = np.zeros(dims, dtype=np.float64)
    solid = _solid_mask(spec)
    for cell in cells:
        stencil = _deposition_weights(spec, cell.position, averaging_radius, solid)
        if stencil is not None:
            region, weights = stencil
            volume[region] += biomass_volume(cell.length, cell.radius) * weights
    return volume / spec.voxel_volume


class _SpeciesRodLike(_RodLike, Protocol):
    @property
    def position(self) -> Vec3: ...
    @property
    def species(self) -> list[float]: ...


def colony_species_density(
    spec: SignalGridSpec,
    cells: Iterable[_SpeciesRodLike],
    *,
    species: int,
    averaging_radius: float = 4.0,
) -> list[float]:
    """Conservatively deposit intracellular amount concentration * biomass volume."""
    if species < 0:
        raise FlowError("species index must be non-negative")
    if not math.isfinite(averaging_radius) or averaging_radius <= 0:
        raise FlowError("averaging radius must be finite and positive")
    totals = np.zeros((spec.shape.x, spec.shape.y, spec.shape.z), dtype=np.float64)
    solid = _solid_mask(spec)
    for cell in cells:
        if species >= len(cell.species):
            raise FlowError("species index is outside the cell's species")
        amount = cell.species[species] * biomass_volume(cell.length, cell.radius)
        if not math.isfinite(amount) or amount < 0:
            raise FlowError("deposited amounts must be finite and nonnegative")
        stencil = _deposition_weights(spec, cell.position, averaging_radius, solid)
        if stencil is not None:
            region, weights = stencil
            totals[region] += amount * weights
    return [float(value) for value in (totals / spec.voxel_volume).ravel()]


def colony_mobility(
    spec: SignalGridSpec,
    cells: Iterable[_RodLike],
    *,
    base: float | Sequence[float] = 1.0,
    drag_coefficient: float = 100.0,
    max_volume_fraction: float = 0.9,
    averaging_radius: float = 4.0,
) -> list[float]:
    """Column-mean phenomenological mobility from conserved biomass density.

    The density cap regularizes only the resistance law; deposited biomass is
    never discarded. Both base mobility and returned mobility are gap means.
    """
    density = colony_volume_fraction(spec, cells, averaging_radius=averaging_radius)
    mobility, _ = _darcy_coefficients(spec, density, base, drag_coefficient, max_volume_fraction)
    return mobility.ravel().tolist()


def _darcy_coefficients(
    spec: SignalGridSpec, density: _FloatGrid, base: float | Sequence[float],
    drag_coefficient: float, max_volume_fraction: float,
) -> tuple[_FloatGrid, _FloatGrid]:
    if not 0 < max_volume_fraction < 1:
        raise FlowError("maximum volume fraction must lie strictly between zero and one")
    dims = (spec.shape.x, spec.shape.y, spec.shape.z)
    if isinstance(base, (int, float)):
        if not math.isfinite(base) or base <= 0.0:
            raise FlowError("base mobility must be finite and positive")
        base_grid = np.full(dims, float(base), dtype=np.float64)
    else:
        if len(base) != dims[0] * dims[1] * dims[2]:
            raise FlowError("base mobility must hold one value per grid site")
        base_grid = np.asarray(base, dtype=np.float64).reshape(dims)
        if not bool(np.all(np.isfinite(base_grid))) or bool(np.any(base_grid < 0.0)):
            raise FlowError("base mobility values must be finite and non-negative")
    solid = _solid_mask(spec)
    fluid = np.ones(dims, dtype=bool) if solid is None else ~solid
    depth_count = np.maximum(fluid.sum(axis=2), 1)
    column_fraction = np.minimum(density.sum(axis=2) / depth_count, max_volume_fraction)
    drag = _kozeny_carman_drag(column_fraction, drag_coefficient)
    mobility = base_grid / (1.0 + base_grid * drag[:, :, None])
    return np.where(fluid, mobility, 0.0), drag


def colony_flow_coupling(
    spec: SignalGridSpec,
    moving_cells: Iterable[tuple[_RodLike, Vec3]],
    *,
    base: float | Sequence[float],
    drag_coefficient: float = 100.0,
    max_volume_fraction: float = 0.9,
    averaging_radius: float = 4.0,
) -> tuple[list[float], list[tuple[float, float]]]:
    """Mobility and body force for drag proportional to fluid/biomass relative velocity.

    Deposit biomass and its in-plane velocity with the same conservative kernel.
    With column biomass velocity V, drag d and empty-device mobility b, the
    shallow momentum balance is (1/b + d) u = -grad(p) + d V. Pass both return
    values to ``solve_flow_field`` to enforce continuity and the inlet flux.
    """
    if not math.isfinite(averaging_radius) or averaging_radius <= 0:
        raise FlowError("averaging radius must be finite and positive")
    dims = (spec.shape.x, spec.shape.y, spec.shape.z)
    volume = np.zeros(dims, dtype=np.float64)
    momentum = np.zeros((*dims[:2], 2), dtype=np.float64)
    solid = _solid_mask(spec)
    for cell, velocity in moving_cells:
        if not math.isfinite(velocity.x) or not math.isfinite(velocity.y):
            raise FlowError("in-plane cell velocity must be finite")
        stencil = _deposition_weights(spec, cell.position, averaging_radius, solid)
        if stencil is None:
            continue
        region, weights = stencil
        mass = biomass_volume(cell.length, cell.radius) * weights
        volume[region] += mass
        momentum[region[:2]] += mass.sum(axis=2)[:, :, None] * (velocity.x, velocity.y)
    mobility, drag = _darcy_coefficients(
        spec, volume / spec.voxel_volume, base, drag_coefficient, max_volume_fraction
    )
    mass = volume.sum(axis=2)
    force = momentum * np.divide(drag, mass, out=np.zeros_like(drag), where=mass > 0)[:, :, None]
    return mobility.ravel().tolist(), [(float(x), float(y)) for x, y in force.reshape(-1, 2)]
