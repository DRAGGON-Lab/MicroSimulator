# ADR 0022: steady Hele-Shaw-Brinkman flow solve

- Status: accepted
- Date: 2026-08-16

## Context

Device flow fields are authored analytically, which is exact only for straight channels. A
junction, bend, pillar array, or partially blocking colony needs a numerical solve. At
microfluidic scale the Reynolds number is around `1e-4`, so the governing momentum balance is
inertia-free and linear, and for a fixed geometry the flow is steady: it can be computed once
in the authoring layer and handed to the engine as the existing face-staggered velocity
field, with no fluid solver in the simulation loop.

## Decision

`cellmodeller2.flow` solves the steady depth-averaged Darcy-Brinkman problem

```text
div(m(x) grad p) = 0        v_face = -m_face * dp/dn
```

over the fluid voxels of a signal grid, where `m` is a per-voxel mobility field and `m_face`
is the harmonic mean of the two adjacent voxel mobilities, zero when either voxel is solid.
This is the Hele-Shaw closure: for shallow channels the depth-averaged Stokes equations
reduce exactly to this form with mobility proportional to the squared local gap height, and a
porous colony enters as additional drag, with resistances adding as
`1/m = 1/m_channel + 1/m_colony`. The uniform-mobility configuration is the Stokes limit of
the model and its validation gate. The in-plane viscous term is deliberately dropped: side
wall boundary layers, whose thickness is on the order of the gap height, are not resolved. A
full staggered-grid Stokes solve is the named refinement if a study needs them.

Pressure is fixed on the fluid boundary faces of one axis - inlet one, outlet zero - and
every other exterior face carries no flux. The discrete operator is symmetric positive
definite and is solved matrix-free by Jacobi-preconditioned conjugate gradient in NumPy; no
new dependency is added. The face velocities are the discrete fluxes of the solved pressure,
so per-voxel mass conservation and zero velocity on closed faces hold by construction, and
the result passes the engine's velocity-field validation unchanged. Because the problem is
linear, the solved field is rescaled to a requested mean inlet speed, so callers never handle
pressure or viscosity units. A grid whose inlet is entirely blocked, or which declares
periodic boundaries, is an error.

`colony_mobility` builds the Brinkman drag field from cell state: each cell's volume
accumulates into its center voxel, the resulting volume fraction sets a Kozeny-Carman style
drag `phi^2 / (1 - phi)^3` scaled by a model-chosen coefficient, and resistances add to the
base mobility. The closure coefficient is a modeling choice, not a measured constant, and is
documented as such. Binning a whole capsule into its center voxel is a nearest-voxel
rasterization: a cell longer than a voxel contributes entirely to one of the voxels it
spans, so the volume fraction, and the drag field with it, is noisier than the colony at
spacings comparable to a cell.

For colony feedback the field must change mid-run, so the engine adds one mutation:
`Simulation.set_velocity_field` validates a replacement field against the full grid
specification and swaps it atomically; everything downstream - transport, drift, checkpoints -
uses whichever field is current. Model code chooses the re-solve cadence.

## Validation sequence

1. Uniform duct: solved field is uniform along the flow axis at exactly the requested mean
   speed, transverse faces zero, per-voxel divergence at solver tolerance.
2. Parallel channels of unequal mobility split flux in the mobility ratio.
3. A blocking pillar routes flow around itself with equal flux through every cross section.
4. A half-blocked Brinkman region carries reduced flux consistent with added drag.
5. Fully blocked inlets and periodic boundaries are rejected.
6. A runtime field swap is validated, applied, and checkpointed.

## Consequences

- Arbitrary mask geometry, including CAD-derived layouts, gets a conservative flow field
  from one build-time solve.
- Colony blockage feeds back on flow at a model-chosen cadence without any native fluid
  solver.
- In-plane boundary layers are the stated accuracy limit of the closure.
- The solved field is a depth-averaged velocity: every voxel in a column carries the
  column's mean. Advection of signals stays conservative, but a cell drifting near a floor
  or ceiling moves at the mean rather than at the slower speed its true profile would give
  it, and a rod sees no shear across the gap. A study that needs the profile within a
  resolved gap belongs on the staggered MAC solve.
