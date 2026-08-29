# ADR 0023: staggered MAC Stokes-Brinkman solve and flow benchmarks

- Status: accepted
- Date: 2026-08-18
- Amended: 2026-08-29

## Context

The Hele-Shaw solve (ADR 0022) depth-averages viscous drag into a mobility
field. That closure is the right cost point for device authoring and in-loop
colony feedback, but it cannot resolve viscous boundary layers on side walls
or the true cross-channel profile, and its accuracy claims need an anchor: a
solver whose only approximation is the mesh.

## Decision

`microsimulator.stokes` solves the inertia-free Stokes-Brinkman momentum
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

Write the discrete momentum equation as `A v + G p = f` and incompressibility as `D v = 0`, where `G = -D^T` under the declared face and cell inner products. Eliminating velocity gives the positive pressure Schur system `(-D A^-1 G) p = -D A^-1 f`. An outer Jacobi-preconditioned conjugate-gradient solve applies this operator matrix-free; each application invokes an inner Jacobi-preconditioned conjugate-gradient solve for the block-diagonal face momentum operator. The three component blocks are stored in one concatenated face vector, which preserves their mathematical independence while allowing one backend-native Krylov operation.

Resolved flow is a `ComputeBackend` domain operation. The CPU implementation evaluates the same operators in C++, while Metal and CUDA keep pressure, velocity, Krylov work vectors, gradients, and divergences in device memory and execute independent MSL and CUDA kernels. The host controls the iteration from reduced scalar data and downloads the final velocity and divergence report. Neither accelerator implementation calls the CPU solver. The portable field contract is binary32 and both outer and inner relative tolerances default to `1e-6`.

The cost remains above the depth-averaged solve, but it is no longer restricted to a Python build-time calculation. Models can execute either solver on their selected backend, including a resolved re-solve when that cost is justified.

## Validation

`scripts/run_flow_benchmarks.py` runs both solvers through an explicitly selected backend against literature and exact references and fails nonzero on any tolerance miss; `test_stokes.py` enforces the same physics at test sizes. The shared C++ `flow_conformance` scenario separately compares every available native backend with the CPU reference for heterogeneous mobility and Brinkman drag.

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

- Resolved wall shear and cross-channel profiles are available where a study needs them on CPU, Metal, and CUDA.
- The Hele-Shaw closure's domain of validity is now measured, not asserted.
- Resolution bounds the MAC solve as the closure bounds the depth-averaged
  one. Every solve reports `min_gap_voxels`, the fluid voxels across its
  narrowest transverse channel, so a caller can tell which of the two solvers
  is the better model of a given grid: below four voxels across a gap the
  closure is, because it carries the gap-height physics analytically.
- Inlet and outlet impose fully developed flow; strongly developing flow at a
  device inlet needs upstream padding voxels.
