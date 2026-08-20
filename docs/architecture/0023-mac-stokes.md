# ADR 0023: staggered MAC Stokes-Brinkman solve and flow benchmarks

- Status: accepted
- Date: 2026-08-18

## Context

The Hele-Shaw solve (ADR 0022) depth-averages viscous drag into a mobility
field. That closure is the right cost point for device authoring and in-loop
colony feedback, but it cannot resolve viscous boundary layers on side walls
or the true cross-channel profile, and its accuracy claims need an anchor: a
solver whose only approximation is the mesh.

## Decision

`cellmodeller2.stokes` solves the inertia-free Stokes-Brinkman momentum
balance with incompressibility,

```text
mu lap(v) - mu d(x) v - grad p = 0        div v = 0
```

on the marker-and-cell staggering the engine already uses: velocities on
faces, pressure at cell centers, so the solved field is the engine's transport
input with no interpolation. Walls are the obstacle voxel boundaries and every
non-flow domain edge; wall planes sit half a spacing beyond the outermost site
centers, matching where the device helpers author floors and ceilings. Normal
velocities on fluid-solid faces are eliminated at zero and tangential
components see walls through reflected ghosts, the standard second-order
voxel-grid treatment. The flow-axis boundaries carry prescribed ghost
pressures (inlet one, outlet zero) with zero-gradient normal outflow, and the
linear solution is rescaled to a requested mean inlet speed, so viscosity
drops out; the Brinkman drag field is an inverse permeability
(`colony_drag` builds it from the colony's volume fraction). Collapsed axes
are invariant directions, matching engine transport semantics.

The saddle-point system is solved through the pressure Schur complement
`S = D A^-1 D^T`, symmetric positive definite, by outer conjugate gradient
with three independent inner component-Laplacian conjugate gradient solves per
application - matrix-free NumPy throughout, no new dependency. The cost sits
well above the Hele-Shaw solve, which remains the default for device authoring
and the in-model re-solve cadence; the MAC solver is for resolved studies and
for anchoring the closure.

## Validation

`scripts/run_flow_benchmarks.py` runs both solvers against literature and
exact references and fails nonzero on any tolerance miss; `test_stokes.py`
enforces the same physics at test sizes.

- Plane Poiseuille: exact parabola, observed convergence order 2. The duct
  peak is interpolated to the centerline, since cell centers straddle the axis
  of an evenly divided duct.
- Square duct: peak-to-mean velocity ratio 2.0962 (Shah & London 1978;
  White, Viscous Fluid Flow), within 0.5% at 32 voxels per side.
- Two-layer Brinkman channel: exact ODE solution (Brinkman 1949) matched in
  value and slope across the fluid-porous interface, with observed
  second-order convergence. Both profiles are compared at unit mean, since the
  solve rescales to the requested speed and amplitude carries no information.
- Cross-solver consistency: in a thin gap the depth-averaged MAC solution
  reproduces the Hele-Shaw flux split around a pillar to under one percent -
  each solver validates the other in the regime where both apply.
- Gap resolution: a channel one voxel across carries about two and a half
  times the flux its parabolic profile would, converging toward the
  lubrication limit as the gap resolves - within about ten percent at four
  voxels and a few percent at eight.
- The zero-drag path is bit-identical to omitting the drag field, and solved
  fields pass engine validation and discrete conservation checks unchanged.

## Consequences

- Resolved wall shear and cross-channel profiles are available where a study
  needs them, at build-time cost.
- The Hele-Shaw closure's domain of validity is now measured, not asserted.
- Resolution bounds the MAC solve as the closure bounds the depth-averaged
  one. Every solve reports `min_gap_voxels`, the fluid voxels across its
  narrowest transverse channel, so a caller can tell which of the two solvers
  is the better model of a given grid: below four voxels across a gap the
  closure is, because it carries the gap-height physics analytically.
- Inlet and outlet impose fully developed flow; strongly developing flow at a
  device inlet needs upstream padding voxels.
