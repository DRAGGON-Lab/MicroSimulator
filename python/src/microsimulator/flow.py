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
Velocities are normalized to the requested mean inlet speed.
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
    FlowAxis,
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
    one relative gap-mean mobility per site (default proportional to gap height squared). If a
    simulation is supplied, its native backend executes the solve. Otherwise
    a temporary simulation uses ``backend`` and ``device_index``.
    """

    _flow_axis_index(spec, axis)
    parameters = DepthAveragedFlowParameters()
    parameters.mean_inlet_speed = mean_inlet_speed
    parameters.axis = _NATIVE_AXES[axis]
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


def _deposit_amount(
    spec: SignalGridSpec,
    position: Vec3,
    amount: float,
    averaging_radius: float,
    target: _FloatGrid,
) -> None:
    """Integrate a separable tent kernel over voxels, then conserve its amount."""
    dims = (spec.shape.x, spec.shape.y, spec.shape.z)
    origin = (spec.origin.x, spec.origin.y, spec.origin.z)
    spacing = (spec.spacing.x, spec.spacing.y, spec.spacing.z)
    centers = (position.x, position.y, position.z)
    if any(not math.isfinite(v) for v in centers) or not math.isfinite(amount) or amount < 0:
        raise FlowError("deposited positions and nonnegative amounts must be finite")
    if any(
        p < o - h / 2 or p >= o + (n - 0.5) * h
        for p, o, h, n in zip(centers, origin, spacing, dims, strict=True)
    ):
        return  # Outside the modeled volume: removal/washout is the caller's responsibility.
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
    if spec.obstacles:
        solid = np.asarray(spec.obstacles, dtype=np.uint8).reshape(dims)[region] != 0
        kernel[solid] = 0
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
    target[region] += (amount / total) * kernel


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
    for cell in cells:
        _deposit_amount(
            spec, cell.position, biomass_volume(cell.length, cell.radius), averaging_radius, volume
        )
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
    for cell in cells:
        if species >= len(cell.species):
            raise FlowError("species index is outside the cell's species")
        amount = cell.species[species] * biomass_volume(cell.length, cell.radius)
        _deposit_amount(spec, cell.position, amount, averaging_radius, totals)
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
    fraction = colony_volume_fraction(spec, cells, averaging_radius=averaging_radius)
    fluid = (
        np.ones(dims, dtype=np.float64)
        if not spec.obstacles
        else (np.asarray(spec.obstacles).reshape(dims) == 0).astype(np.float64)
    )
    count = fluid.sum(axis=2, keepdims=True)
    column_density = fraction.sum(axis=2, keepdims=True) / np.maximum(count, 1)
    fraction = np.minimum(column_density, max_volume_fraction) * fluid
    drag = _kozeny_carman_drag(fraction, drag_coefficient)
    # m = b / (1 + b * drag) is 1 / (1/b + drag) extended continuously to b = 0.
    mobility = base_grid / (1.0 + base_grid * drag)
    obstacles = spec.obstacles
    if obstacles:
        solid = np.asarray(obstacles, dtype=np.uint8).reshape(dims) != 0
        mobility[solid] = 0.0
    return [float(value) for value in mobility.ravel()]
