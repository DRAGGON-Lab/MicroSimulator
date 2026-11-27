"""Staggered-grid Stokes-Brinkman flow solve for device grids.

This is the spatially resolved companion to the Hele-Shaw solver in
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

Flexible GMRES solves the fixed velocity-pressure block system. Approximate
momentum CG solves and a diagonal pressure approximation act only as a variable
preconditioner. Convergence uses a freshly evaluated block residual, with
continuity scaled by inverse minimum spacing; the report also gives the
momentum residual and physical divergence RMS. The selected CPU, Metal, or
CUDA backend executes the matrix-free operators and vector operations.
Resolved gaps and mesh convergence are needed for quantitative wall shear;
a one-voxel gap does not resolve a parabolic velocity profile.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    FlowAxis,
    FlowPort,
    ResolvedFlowParameters,
    ResolvedFlowReport,
    ResolvedPortFlowParameters,
    ResolvedPortFlowResult,
    SignalGridSpec,
    SignalGridVelocityField,
    Simulation,
)
from .flow import (
    FlowError,
    _deposit,
    _drag_parameters,
    _flow_axis_index,
    _native,
    _RodLike,
)

_NATIVE_AXES = {"x": FlowAxis.X, "y": FlowAxis.Y, "z": FlowAxis.Z}

StokesSolveReport = ResolvedFlowReport


def colony_drag(
    spec: SignalGridSpec,
    cells: Iterable[_RodLike],
    *,
    drag_coefficient: float,
    max_volume_fraction: float = 0.9,
    averaging_radius: float = 4.0,
    simulation: Simulation | None = None,
    backend: BackendKind = BackendKind.CPU,
    device_index: int = 0,
) -> list[float]:
    """Build the Brinkman drag field (inverse permeability) from the colony.

    The colony's per-voxel volume fraction ``phi`` sets a Kozeny-Carman style
    drag ``drag_coefficient * phi^2 / (1 - phi)^3``. The coefficient carries
    units of one over length squared and is a modeling choice. Solid voxels
    stay at zero (they are walls, not porous media). The deposit and drag run
    on the simulation's backend, or on ``backend``.
    """

    selected = _native(simulation, backend, device_index)
    deposit = _deposit(spec, list(cells), [], averaging_radius, selected)
    try:
        return selected.brinkman_drag(
            spec, deposit, _drag_parameters(drag_coefficient, max_volume_fraction)
        )
    except (OverflowError, RuntimeError, ValueError, TypeError) as error:
        raise FlowError(str(error)) from error


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


def solve_port_stokes_field(
    spec: SignalGridSpec,
    ports: Sequence[FlowPort],
    *,
    drag: Sequence[float] | None = None,
    tolerance: float = 1.0e-6,
    max_outer_iterations: int = 5000,
    inner_tolerance: float = 1.0e-6,
    max_inner_iterations: int = 50_000,
    simulation: Simulation | None = None,
    backend: BackendKind = BackendKind.CPU,
    device_index: int = 0,
) -> ResolvedPortFlowResult:
    """Solve the staggered Stokes-Brinkman flow driven through ports.

    Each port is a rectangle of boundary sites on any face of the grid. A
    ``SPEED`` port carries its value as the mean speed over its open area,
    positive into the grid and negative out of it, with the profile the
    geometry behind it gives; a ``PRESSURE`` port holds its value one spacing
    outside the face. At least one port must hold a pressure, and every port's
    face must be a ``FIXED`` boundary. Faces outside every port are no-slip
    walls. The field is the combination of one block solve per speed port.
    Where the geometry is one shallow flow describes, each solves for the
    correction to the depth-averaged pressure; otherwise it solves for the
    whole pressure field. Either way a long channel takes more outer
    iterations than `solve_stokes_field` does, and the report's residuals and
    the ports' fluxes show how closely the field conserves volume.

    The result holds the velocity field, the solve report, and for each port
    its pressure, flux and mean speed. Pressures are divided by viscosity.
    ``drag`` is as for `solve_stokes_field`.
    """

    parameters = ResolvedPortFlowParameters()
    parameters.relative_tolerance = tolerance
    parameters.max_outer_iterations = max_outer_iterations
    parameters.inner_relative_tolerance = inner_tolerance
    parameters.max_inner_iterations = max_inner_iterations
    selected = (
        simulation if simulation is not None else Simulation(backend, device_index=device_index)
    )
    try:
        return selected.solve_resolved_port_flow(
            spec,
            list(ports),
            [] if drag is None else [float(value) for value in drag],
            parameters,
        )
    except (OverflowError, RuntimeError, ValueError) as error:
        raise FlowError(str(error)) from error
