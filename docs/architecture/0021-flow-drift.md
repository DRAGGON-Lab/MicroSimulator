# ADR 0021: advective flow drift on cells

- Status: accepted
- Date: 2026-08-16

## Context

The velocity field advects grid signals but exerts nothing on cells. In a flow-fed trap,
cells outside the trap should be carried downstream; in the overdamped regime a cell's
velocity relaxes to the local fluid velocity on a timescale far below one time step, so the
cell translates with the flow and a rod in shear rotates.

## Decision

`Simulation.apply_flow_drift(dt, integration)` advects every non-fixed cell through the grid's velocity
field by one explicit step, as an operation between growth and contact relaxation. The fluid
velocity is sampled at both capsule centerline endpoints: each stencil site's cell-centered
velocity is the mean of its two face velocities per axis, and the trilinear weights are the
signal-sampling weights, including the obstacle renormalization near walls. Endpoints are
clamped to the lattice of site centers before sampling: mechanics walls, not the lattice
edge, bound cells, so a rod whose tip pokes past the outermost site samples the nearest
in-grid point rather than erroring. Clamping happens in the lattice coordinate the bound is
tested in rather than in position space, so a clamped endpoint lies inside the lattice for
every origin and spacing. From endpoint velocities `v1` and `v2` with cylinder length `l`
and axis `a`:

- translation is `dt * (v1 + v2) / 2`;
- the rotation vector is `dt * (a x (v2 - v1)) / l`, the least-squares rigid rotation for the
  endpoint velocity difference, taken as zero when `l` is degenerate, applied as an
  axis-angle rotation capped by the caller's mechanics integration rotation limit, through
  the same axis-angle helper mechanical integration uses.

Every update is validated before any world-state mutation, matching mechanical integration.
The operation requires a signal grid with a velocity field. A cell whose sampling stencil
holds no fluid samples zero rather than erroring: the field is validated zero on every face
of a solid site, so that is the field's own value there, and a cell that contact relaxation
has pressed into a wall simply does not drift. Concentration has no such value, so sampling
it inside an obstacle remains a model error. Fixed cells do not move.

Drift composes with contact relaxation by operator splitting: drift first, then the ordinary
relaxation resolves any overlap the drift produced against walls or neighbors. The controller
applies drift when its mechanics configuration enables `flow_drift`, before the relaxation
passes. A formulation that couples drag into the relaxation right-hand side, so wall contact
forces balance fluid forces within one solve, is a candidate refinement with its own contract;
the explicit split is the reference behavior.

The operation is host-side over committed state and identical on every backend; no kernel or
checkpoint change is involved. `flow_drift` is part of the controller's mechanics
configuration payload.

## Validation sequence

1. A free cell in uniform flow translates by exactly `velocity * dt` per drift call.
2. A rod spanning a shear gradient rotates toward alignment; a fixed cell does not move.
3. Drift against a wall followed by relaxation leaves the cell outside the wall.

## Consequences

- Washout becomes dynamic: flow carries cells to the removal predicate rather than the model
  teleporting them.
- Splitting error is first order in `dt`; models choose steps so per-step drift stays small
  relative to cell size, as they already do for growth.
- Cells in zero-velocity regions, including trap interiors, are unaffected.
