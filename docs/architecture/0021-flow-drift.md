# ADR 0021: finite-aspect flow drift

- Status: accepted
- Date: 2026-08-16
- Amended: 2026-09-10

## Decision

`Simulation.apply_flow_drift(dt, integration)` translates each non-fixed cell at the interpolated velocity of its center. Its direction p follows the equivalent-spheroid Jeffery equation `dp/dt = Omega p + lambda (E p - (p.Ep)p)`, where `E = (grad(u) + grad(u)^T)/2`, `Omega = (grad(u) - grad(u)^T)/2`, `a = (length + 2 radius)/(2 radius)`, and `lambda = (a² - 1)/(a² + 1)`. This retains rigid-body rotation for a sphere and finite-aspect tumbling in shear. A capsule is approximated by a spheroid with the same aspect ratio; the equation is not an exact capsule hydrodynamic solution. See [Jeffery (1922)](https://doi.org/10.1098/rspa.1922.0078) and the zero-inertia equation in [Einarsson et al. (2015)](https://arxiv.org/abs/1504.02849).

The field is sampled with the face-connected trilinear stencil, after converting opposing face velocities to site-centered velocities. Velocity gradients use centered differences of that interpolant over one grid spacing. Sampling is clamped at the center lattice; a fully solid stencil has zero velocity. Near-wall rotation is therefore an interpolation approximation and requires spatial refinement; wall hydrodynamic torques, lubrication forces, adhesion, and detachment are not included.

A normalized explicit midpoint step advances position and direction. Internal substeps limit translation to one quarter of the minimum grid spacing and angular displacement to `max_rotation_radians`. The latter bounds each substep, not the total rotation, so tightening it improves integration instead of changing the motion law. Zero retains the explicit option to freeze orientation. Excessive substep counts or invalid geometry reject the operation before any cell is changed. Length and radius remain unchanged, preserving biochemical biomass.

The controller composes drift with contact relaxation and growth by first-order splitting. Drift integration is second order in a smooth fixed velocity field, but this does not make the complete coupled simulation second order. Outer timestep and mechanics convergence still need checking. Fixed cells are the explicit attached population; free-cell washout is kinematic, not a prediction of attachment or detachment thresholds.

The operation runs on the host over committed state identically for every backend. `flow_drift` remains part of the controller checkpoint configuration.

## Validation

Tests cover uniform translation, finite-aspect Jeffery shear including the spherical limit, rigid-rotation convergence, fixed cells, zero orientation limit, clamped and fully solid sampling, length conservation, and controller checkpoint continuation.
