"""Staggered-grid Stokes-Brinkman flow solve for device grids.

This is the high-fidelity companion to the Hele-Shaw solver in
`microsimulator.flow`: it resolves the full velocity field, including viscous
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
conjugate gradient. The selected CPU, Metal, or CUDA backend executes the
matrix-free operator and Krylov iterations. This costs far more than the
Hele-Shaw solve.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    FlowAxis,
    ResolvedFlowParameters,
    ResolvedFlowReport,
    SignalGridSpec,
    SignalGridVelocityField,
    Simulation,
)
from .flow import (
    FlowError,
    _flow_axis_index,
    _kozeny_carman_drag,
    _RodLike,
    colony_volume_fraction,
)

_NATIVE_AXES = {"x": FlowAxis.X, "y": FlowAxis.Y, "z": FlowAxis.Z}

StokesSolveReport = ResolvedFlowReport


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


def solve_stokes_field(
    spec: SignalGridSpec,
    *,
    mean_inlet_speed: float,
    axis: str = "y",
    drag: Sequence[float] | None = None,
    tolerance: float = 1.0e-6,
    max_outer_iterations: int = 500,
    inner_tolerance: float = 1.0e-6,
    max_inner_iterations: int = 50_000,
    simulation: Simulation | None = None,
    backend: BackendKind = BackendKind.CPU,
    device_index: int = 0,
) -> tuple[SignalGridVelocityField, StokesSolveReport]:
    """Solve the staggered Stokes-Brinkman flow and return the velocity field.

    Flow runs from the lower to the upper boundary of ``axis``; a negative
    ``mean_inlet_speed`` reverses it. ``drag`` optionally gives one Brinkman
    drag value (inverse permeability, units 1/length^2) per site; omitted or
    zero drag is pure Stokes. If a simulation is supplied, its native backend
    executes the solve. Otherwise a temporary simulation uses ``backend`` and
    ``device_index``.
    """

    _flow_axis_index(spec, axis)
    parameters = ResolvedFlowParameters()
    parameters.mean_inlet_speed = mean_inlet_speed
    parameters.axis = _NATIVE_AXES[axis]
    parameters.relative_tolerance = tolerance
    parameters.max_outer_iterations = max_outer_iterations
    parameters.inner_relative_tolerance = inner_tolerance
    parameters.max_inner_iterations = max_inner_iterations
    selected = (
        simulation if simulation is not None else Simulation(backend, device_index=device_index)
    )
    try:
        result = selected.solve_resolved_flow(
            spec,
            [] if drag is None else [float(value) for value in drag],
            parameters,
        )
    except (OverflowError, RuntimeError, ValueError) as error:
        raise FlowError(str(error)) from error
    return result.field, result.report
