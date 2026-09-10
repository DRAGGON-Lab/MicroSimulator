# ADR 0015: focused affine grid reactions

- Status: accepted
- Date: 2026-08-15

## Context

Some spatial models apply a reaction throughout a physical region rather than at cell locations. The SimBOL Danino tutorial, for example, removes AHL in a flow channel, replenishes nutrient toward a target inside a trap, and decays nutrient outside it. Legacy CellModeller implements these terms by overriding a Python grid class and mutating its rate array.

MicroSimulator cannot admit arbitrary host callbacks or injected device source without losing its typed, data-only checkpoint contract and independent CPU, Metal, and CUDA implementations. A biological special case named for AHL, nutrient, or trap geometry would instead put model vocabulary into the numerical engine.

## Decision

`SignalGridSpec` may contain one optional `SignalGridAffineReaction`. It stores two immutable `float32` arrays in the grid's ordinary signal/x/y/z level order:

```text
reaction_rate[i] = source_rates[i] - loss_rates[i] * levels[i].
```

Both arrays must match the complete grid level count and contain only finite, non-negative values. A target relaxation `k(target - c)` compiles to source `k*target` and loss `k`. Model code may use coordinates, boxes, half-spaces, or other predicates while constructing the arrays, but the runtime receives only the materialized coefficients.

The affine reaction is part of the signal operator. Forward Euler evaluates it from the old field and adds the largest per-signal loss coefficient to the preflight stability bound. Crank-Nicolson includes loss in the matrix diagonal and the constant source on both trapezoidal halves. Backward Euler includes the complete affine reaction implicitly and preserves positivity for nonnegative explicit input at arbitrary reaction stiffness. Cell-scattered sources stay explicit and retain their amount-per-time convention.

CPU, Metal, and CUDA implement the same fixed operation. Device implementations receive source and loss buffers; they do not compile a model callback or branch on biological region names. Coefficients are exact checkpoint state. Version 7 checkpoints record an object or `null`; versions 1 through 6 migrate to no affine field reaction.

## Deliberate limits

This is a generic data representation for one focused numerical operation, not a general voxel-program extension point. It does not support arbitrary functions of position or time, cross-signal reactions, nonlinear local kinetics, mutable masks, or runtime coefficient callbacks. A demonstrated need for one of those behaviors requires another named numerical contract with an explicit integration and checkpoint design.

## Runtime replacement

A reaction field is data, so a model may replace it while a simulation runs:
`Simulation.set_signal_reaction` validates a candidate against the full grid
specification and swaps it atomically, exactly as a velocity field is swapped.
A cell-state-dependent first-order loss can therefore use an implicit field coefficient rather than an explicit cell sink. Backward Euler is the positivity-preserving baseline for stiff loss; Crank-Nicolson can oscillate even though it is linearly stable. Coefficients are frozen during each biological step, so their refresh interval contributes splitting error. Model code chooses and convergence-tests that interval, and the field remains exact checkpoint state.

## Consequences

- Spatial reservoirs and first-order sinks remain inspectable and portable.
- Region construction stays in ordinary model-authoring code and has no device execution semantics.
- A single branch-free native operator covers the Danino masks and equivalent source/loss fields without hard-coding a tutorial.
- Dense coefficient arrays increase checkpoint size in proportion to the signal field; compression or a sparse authoring format can be added later without changing the runtime equation.
