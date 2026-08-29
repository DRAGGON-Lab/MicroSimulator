"""Steady Hele-Shaw-Brinkman flow solve for device grids.

The solver computes the depth-averaged Darcy-Brinkman pressure problem
``div(m grad p) = 0`` over the fluid voxels of a signal grid and returns the
face fluxes ``v = -m_face * dp/dn`` as a face-staggered velocity field. The
per-voxel mobility ``m`` carries the physics: uniform mobility is the Stokes
limit of the closure and resolves flow through arbitrary mask geometry, while
reduced mobility inside a colony (`colony_mobility`) adds Brinkman drag so a
packed trap diverts flow. Mobility is relative - the linear solution is
rescaled to a requested mean inlet speed - so callers never handle pressure or
viscosity units. Discrete conservation and zero velocity on closed faces hold
by construction, and the returned field passes the engine's grid validation
unchanged.

Pressure is fixed on the fluid boundary faces of the flow axis (inlet one,
outlet zero) and every other exterior face carries no flux; the flow axis
boundaries must therefore be `FIXED` and no axis may be periodic. The discrete
operator is symmetric positive definite and is solved matrix-free with
Jacobi-preconditioned conjugate gradient by the CPU, Metal, or CUDA backend
selected for the simulation. Side-wall boundary layers, whose thickness is on
the order of the gap height, are outside the closure.
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
    one relative mobility per site (default uniform, the Stokes limit). If a
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


def colony_volume_fraction(
    spec: SignalGridSpec,
    cells: Iterable[_RodLike],
    *,
    max_volume_fraction: float = 0.9,
) -> _FloatGrid:
    """Rasterize the colony into a per-voxel volume fraction grid.

    Each cell's capsule volume accumulates into the voxel holding its center
    (the grid origin is the center of site zero, so voxel ``i`` spans the
    half-open interval centered on ``origin + i * spacing``); fractions are
    capped at ``max_volume_fraction``.
    """

    if not 0.0 < max_volume_fraction < 1.0:
        raise FlowError("maximum volume fraction must lie strictly between zero and one")
    dims = (spec.shape.x, spec.shape.y, spec.shape.z)
    origin = (spec.origin.x, spec.origin.y, spec.origin.z)
    spacing = (spec.spacing.x, spec.spacing.y, spec.spacing.z)
    volume = np.zeros(dims, dtype=np.float64)
    for cell in cells:
        position = (cell.position.x, cell.position.y, cell.position.z)
        indices: list[int] = []
        inside = True
        for component in range(3):
            index = math.floor(
                (position[component] - origin[component]) / spacing[component] + 0.5
            )
            if not 0 <= index < dims[component]:
                inside = False
                break
            indices.append(index)
        if not inside:
            continue
        radius = cell.radius
        capsule = math.pi * radius * radius * cell.length + (4.0 / 3.0) * math.pi * radius**3
        volume[indices[0], indices[1], indices[2]] += capsule
    return np.minimum(volume / spec.voxel_volume, max_volume_fraction)



class _SpeciesRodLike(Protocol):
    @property
    def position(self) -> Vec3: ...
    @property
    def species(self) -> list[float]: ...


def colony_species_density(
    spec: SignalGridSpec,
    cells: Iterable[_SpeciesRodLike],
    *,
    species: int,
) -> list[float]:
    """Rasterize one intracellular species into a per-voxel density.

    Each cell's level accumulates into the voxel holding its center, on the
    same nearest-voxel convention as `colony_volume_fraction`, and the total is
    divided by the voxel volume. A rate written per cell and per unit of that
    species becomes a rate per unit volume of field, which is what an affine
    grid reaction carries.
    """

    if species < 0:
        raise FlowError("species index must be non-negative")
    dims = (spec.shape.x, spec.shape.y, spec.shape.z)
    origin = (spec.origin.x, spec.origin.y, spec.origin.z)
    spacing = (spec.spacing.x, spec.spacing.y, spec.spacing.z)
    totals = np.zeros(dims, dtype=np.float64)
    for cell in cells:
        levels = cell.species
        if species >= len(levels):
            raise FlowError("species index is outside the cell's species")
        position = (cell.position.x, cell.position.y, cell.position.z)
        indices: list[int] = []
        for component in range(3):
            index = math.floor(
                (position[component] - origin[component]) / spacing[component] + 0.5
            )
            if not 0 <= index < dims[component]:
                break
            indices.append(index)
        if len(indices) != 3:
            continue
        totals[indices[0], indices[1], indices[2]] += max(0.0, levels[species])
    return [float(value) for value in (totals / spec.voxel_volume).ravel()]


def colony_mobility(
    spec: SignalGridSpec,
    cells: Iterable[_RodLike],
    *,
    base: float | Sequence[float] = 1.0,
    drag_coefficient: float = 100.0,
    max_volume_fraction: float = 0.9,
) -> list[float]:
    """Build the Brinkman mobility field from the current colony.

    Each cell's capsule volume accumulates into the voxel holding its center
    (the grid origin is the center of site zero, so voxel ``i`` spans the
    half-open interval centered on ``origin + i * spacing``); the resulting
    volume fraction ``phi`` adds Kozeny-Carman style drag
    ``drag_coefficient * phi^2 / (1 - phi)^3`` to the base resistance, so
    ``1/m = 1/base + drag``. ``base`` is a uniform value or a per-site field
    such as `gap_mobility`. The drag coefficient is a modeling choice: it
    sets how strongly a packed colony resists through-flow relative to the
    open channel. Solid voxels stay at zero mobility.
    """

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
    fraction = colony_volume_fraction(spec, cells, max_volume_fraction=max_volume_fraction)
    drag = _kozeny_carman_drag(fraction, drag_coefficient)
    # m = b / (1 + b * drag) is 1 / (1/b + drag) extended continuously to b = 0.
    mobility = base_grid / (1.0 + base_grid * drag)
    obstacles = spec.obstacles
    if obstacles:
        solid = np.asarray(obstacles, dtype=np.uint8).reshape(dims) != 0
        mobility[solid] = 0.0
    return [float(value) for value in mobility.ravel()]
