# ADR 0006: grid signaling and cell coupling

- Status: accepted
- Date: 2026-08-15

## Context

Diffusible signals connect the intracellular rate model to a spatial transport model. This is one numerical system: independently updating a host grid and a device cell model would change sampling time, conservation, and reproducibility. The CellModeller comparison documents ambiguous historical coefficient scaling and unit conventions.

MicroSimulator needs a portable state model and one stage contract that can be implemented independently with native CPU, Metal, and CUDA APIs.

## Grid state

A signal grid is a rectangular lattice with positive dimensions, a finite origin, and positive finite spacing in each axis. Dimensions of one are valid for reduced-dimensional models. Levels are finite, non-negative `float32` concentrations in signal-major, then x/y/z-major order:

```text
index = signal * (nx * ny * nz) + x * (ny * nz) + y * nz + z
position(x, y, z) = origin + spacing * (x, y, z)
```

Each signal declares a non-negative diffusion coefficient and a finite 3D advection velocity. Every grid face declares one of:

- `no_flux`: diffusive and advective face fluxes are zero;
- `periodic`: the stencil wraps to the opposite face; or
- `fixed`: the exterior stencil value is the declared non-negative reservoir concentration for that signal.

Periodic faces must occur in opposing pairs on an axis. For diffusion, a no-flux exterior stencil value equals the boundary lattice value. Fixed values are defined at the exterior stencil location one grid spacing beyond the boundary, which removes half-cell ambiguity.

The grid may also carry an immutable affine reaction field with non-negative finite source `b[i]` and first-order loss `lambda[i]` for every flattened level:

```text
R(c)[i] = b[i] - lambda[i] * c[i].
```

Both arrays use the level ordering above and must match the complete level count. Region and coordinate masks are authoring conveniences that materialize these arrays before validation; no runtime predicate is part of the grid.

## Transport discretization

Diffusion uses the conventional centered second difference independently in each non-degenerate axis:

```text
D * sum_axis((c[i-1] - 2*c[i] + c[i+1]) / spacing_axis^2)
```

Advection uses conservative face fluxes with first-order upwinding for `-velocity dot grad(c)`. A fixed reservoir supplies the exterior upwind value, a periodic face wraps, and a no-flux face has exactly zero advective flux. This is less accurate in smooth fields than a higher-order scheme, but its monotonicity and compact native implementation make it the reference stage. Higher-order transport can be added as a separately named integrator.

Forward Euler rejects a step before mutation unless, for every signal,

```text
dt * (2 * D * sum_axis(1 / spacing_axis^2)
      + sum_axis(abs(velocity_axis) / spacing_axis)
      + max_site(lambda)) <= 1.
```

Degenerate axes contribute zero. Candidate levels must remain finite and non-negative; the engine reports an invalid step rather than silently clamping.

## Sampling and scatter

Cells must lie inside the closed lattice bounds on every non-degenerate axis. An out-of-domain cell is a model error detected before mutation. Trilinear weights form a partition of unity, including reduced-dimensional axes. Sampling and scatter use the same weights, making the discrete operations adjoint.

The grid stores concentration. A cell signal output is an amount-per-time source, so scatter contributes `weight * source / voxel_volume` to the grid concentration rate. This gives an explicit conservation rule. Intracellular species outputs remain concentration-per-time rates under the existing effective-volume dilution convention.

## Coupled Euler stage

A typed coupled rate plan extends the existing expression operations with a sampled-signal input. It declares both intracellular species outputs and cell signal amount-rate outputs. A step executes in this order:

1. validate the complete step and preserve the old grid field;
2. advance cell growth and dilute species by old/new effective volume;
3. compute grid transport and affine reaction from the old field;
4. sample the old field at current cell positions;
5. evaluate species and signal outputs from the same diluted species, sampled signals, and post-growth cell geometry;
6. scatter cell signal amount rates into the grid concentration rate;
7. commit species and grid Euler candidates simultaneously; and
8. advance simulation time.

The native backend operation encompasses transport, sample, rate evaluation, scatter, and update. Intermediate arrays stay on the selected device during that operation. The host receives committed state for the current host-owned engine architecture; no stage may call the CPU reference as a fallback.

## Checkpoint and lifecycle behavior

Grid specification, affine reaction coefficients, levels, and the complete coupled rate plan are exact checkpoint state. Derived weights, indices, rates, and device buffers are reconstructed caches. Division inherits intracellular species as before and does not copy a sampled signal cache; the next coupled stage samples both daughters at their new positions.

Changing grid geometry after cells exist is outside the current API. Any remeshing operation must define its interpolation and mass-conservation behavior explicitly.

## Consequences

- Diffusion and advection coefficients have standard physical meanings.
- Boundary and edge behavior is inspectable and backend-conformable.
- Cell/grid exchange has a stated conservation unit rather than model-source convention.
- Explicit Euler failures are early and deterministic.
- Legacy signaling models require a reviewed coefficient and unit migration.
