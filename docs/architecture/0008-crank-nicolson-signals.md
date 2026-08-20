# ADR 0008: Crank-Nicolson signal transport

- Status: accepted
- Date: 2026-08-15

## Context

The legacy `CLCrankNicIntegrator` was intended to remove the explicit diffusion time-step restriction. Its comments and construction define a Crank-Nicolson transport stage, but the result of the final SciPy convolution was discarded. Observable legacy execution therefore stopped at the explicit right-hand side and did not apply the inverse operator.

MicroSimulator needs a named, checkpointed integration choice with the same equation on CPU, Metal, and CUDA. It must also diagnose convergence rather than hide a fixed iteration count.

## Equation

Let `T(c)` be the transport rate from ADR 0006, including the declared boundary reservoirs, let `R(c) = b - lambda*c` be the optional affine field reaction, and let `s(cells, c_n)` be the cell-scattered concentration rate evaluated from the old signal field. One step solves

```text
c_(n+1) - dt/2 (T + R)(c_(n+1))
  = c_n + dt/2 (T + R)(c_n) + dt s(cells, c_n).
```

This deliberately implements the intended legacy equation instead of its ignored-return bug. The standard diffusion coefficient, three-dimensional upwind advection, boundary semantics, and source units remain those of ADR 0006; the historical extra factor of one sixth is not restored.

Fixed reservoir boundaries make `T` affine, and a nonzero field source makes `R` affine. Applying both complete operators on both sides includes their constant contributions for the full time step. Spatial loss `-lambda*c` enters the implicit diagonal. Periodic and no-flux boundaries remain homogeneous.

## Solver contract

The linear system is solved by matrix-free Jacobi iteration over the local transport stencil. Diffusion and first-order upwind advection give a positive diagonal for `I - dt/2 T`; the update divides by that diagonal and retains the affine boundary contribution in the remainder.

Convergence uses the RMS residual of the declared equation:

```text
residual_rms <= max(absolute_tolerance, epsilon * rms(right_hand_side))
                  + relative_tolerance * initial_residual_rms
```

The relative term scales the residual the step begins with, not the field it begins from. A field's magnitude says nothing about how much of it one step has to change, so scaling by the field makes the threshold grow with the background: a cell's exchange with a well-stocked field then falls under it and the solve returns the old field, discarding the source. Scaling by the initial residual instead asks for a fixed reduction of whatever this step actually has to resolve, so the accuracy a model gets is the accuracy it asked for, independent of concentration scale.

The absolute term is raised to the residual floor of the field, `epsilon * rms(right_hand_side)` in float32. The right-hand side carries both the field and the step's operator terms, so this tracks stiffness as well as magnitude. A tolerance below the floor is unreachable, and clamping it up converges rather than iterating to the limit on rounding noise. The floor is also the resolution limit of an implicit step: a source that moves the field by less than its own float32 resolution leaves no residual to detect, and the step converges without it. A model whose exchange is that small next to its background belongs on forward Euler, which applies sources unconditionally, or on a concentration scale that resolves it.

The grid specification records the integration kind, maximum iteration count, and both tolerances. These fields are exact checkpoint state, so a checkpoint written before this rule resumes with the tolerances it recorded and the current interpretation of them. A solver that reaches the iteration limit or produces a non-finite residual fails the step; it never commits an unconverged field. A successful step exposes its iteration count and final residual. Final concentrations retain the engine-wide finite, non-negative invariant. Crank-Nicolson is stable for large diffusion steps but is not positivity preserving, so an oscillatory negative result is rejected.

## Backend staging

CPU is the numerical reference. Metal and CUDA use native Jacobi stencil kernels and compare their committed fields and reports against it. Metal forms the right-hand side, iterates, computes residual terms, and reduces them through native MSL pipelines. CUDA mirrors that sequence with native kernels ordered on one stream. In both cases the host reads only the scalar convergence result and the committed field. Calling the CPU reference from a GPU backend is not an allowed fallback.

## Consequences

- Forward Euler remains the default and retains its explicit stability check.
- Checkpoint version 5 records the integration and solver configuration; versions 1 through 4 migrate signal grids to Forward Euler defaults.
- Coupled rates use the same semi-implicit transport solve with explicit cell sources, preserving old-field sampling and simultaneous commit semantics.
- The legacy Green's-function truncation and ignored convolution are not compatibility targets.
