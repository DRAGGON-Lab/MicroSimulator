# ADR 0022: steady Hele-Shaw-Brinkman flow solve

- Status: accepted
- Date: 2026-08-16
- Amended: 2026-08-29

## Context

Analytic velocity profiles are exact only for simple channels. A junction, bend, pillar array, imported mask, or partially blocking colony requires a numerical solve over the authored device geometry. At microfluidic scales the governing momentum balance is commonly inertia-free and linear, so the flow for a fixed geometry is steady. The simulation nevertheless needs to select the implementation: initial device assembly and any later colony-coupled re-solve must execute through the same CPU, Metal, or CUDA backend chosen for the rest of the model.

## Decision

`microsimulator.flow` solves the steady depth-averaged Darcy-Brinkman problem

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

Pressure is fixed on the fluid boundary faces of one axis, with inlet pressure one and outlet pressure zero, while every other exterior face carries no flux. For neighboring fluid sites `i` and `j`, the face coefficient is the harmonic mean `m_ij = 2 m_i m_j / (m_i + m_j)` divided by the squared center spacing. Fixed pressure boundaries use the corresponding half-cell coefficient `2 m_i / h^2`. Summing these coefficients gives the Jacobi diagonal and the inlet contribution gives the right-hand side. The resulting symmetric positive-definite system is solved matrix-free by Jacobi-preconditioned conjugate gradient.

The solve is a domain operation on `ComputeBackend` and `Simulation`. C++ implements the readable CPU reference, MSL implements the Metal operator, Krylov vector updates, and reductions, and CUDA C++ implements the CUDA equivalents. Device vectors remain on the selected accelerator throughout each solve; the host receives reduction partials needed for convergence control and the final face field. No accelerator backend calls the CPU reference. The portable field and solver contract is binary32, with a default relative residual tolerance of `1e-6`.

The reconstructed face velocities are the discrete fluxes of the solved pressure, so per-voxel mass conservation and zero velocity on closed faces hold by construction. Because the problem is linear, the field is rescaled to a requested mean inlet speed. A grid whose inlet is entirely blocked, whose outlet is unreachable, or which declares periodic boundaries is rejected.

Python remains the device-authoring surface. `gap_mobility` converts a solid mask into a gap-height mobility and `colony_mobility` rasterizes cell volume into a Kozeny-Carman-style resistance field. These helpers construct backend-neutral dense input arrays; they do not solve the pressure system. The closure coefficient is a modeling choice, not a measured constant. Binning a whole capsule into its center voxel is a nearest-voxel approximation, so mobility becomes noisy when the grid spacing approaches a cell length.

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

- Arbitrary mask geometry, including CAD-derived layouts, gets a conservative flow field from a native backend solve.
- Colony blockage feeds back on flow at a model-chosen cadence through the already selected backend.
- The solver, not only downstream transport, can therefore use CUDA or Metal acceleration.
- In-plane boundary layers are the stated accuracy limit of the closure.
- The solved field is a depth-averaged velocity: every voxel in a column carries the
  column's mean. Advection of signals stays conservative, but a cell drifting near a floor
  or ceiling moves at the mean rather than at the slower speed its true profile would give
  it, and a rod sees no shear across the gap. A study that needs the profile within a
  resolved gap belongs on the staggered MAC solve.
