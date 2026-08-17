# ADR 0019: face-staggered signal velocity fields

- Status: accepted
- Date: 2026-08-16

## Context

Grid advection is a single constant vector per signal. A microfluidic device needs a
spatially varying flow: a Poiseuille profile across a channel, zero velocity inside walls, and
no flow through a trap mouth. A constant vector can express none of these. The existing
transport discretization computes conservative face fluxes but reads the same cell-centered
velocity on both faces of a voxel, which is exactly conservative only because that velocity is
constant; a varying field must live on faces to keep the discrete fluxes conservative.

## Decision

`SignalGridSpec` may declare one optional velocity field, shared by every signal, holding the
face-normal velocity component on each lattice face in three dense `float32` arrays:

```text
x_faces[fx, y, z]   fx in [0, nx]   index = fx * (ny * nz) + y * nz + z
y_faces[x, fy, z]   fy in [0, ny]   index = x * ((ny + 1) * nz) + fy * nz + z
z_faces[x, y, fz]   fz in [0, nz]   index = x * (ny * (nz + 1)) + y * (nz + 1) + fz
```

Voxel `(x, y, z)` reads its lower x face at `fx = x` and its upper x face at `fx = x + 1`, and
likewise on the other axes. A grid with a velocity field must declare a zero constant
`advection` vector for every signal; the field replaces the constant convention rather than
adding to it.

Every value must be finite. A face that transport treats as closed must carry zero velocity:
faces on `no_flux` exterior boundaries, and faces between a fluid site and a solid obstacle
site or between two solid sites. On a periodic axis the two exterior face layers are the same
physical face and must hold equal values. Faces on `fixed` boundaries may be nonzero; inflow
there upwinds from the declared reservoir and outflow upwinds from the interior, so a fixed
face pair is a physical inlet or outlet.

Transport keeps first-order upwinding but reads each face's own velocity: the lower face flux
upwinds from the lower neighbor when its velocity is non-negative and from the current site
otherwise, and symmetrically for the upper face. The Crank-Nicolson operator diagonal gains
`-(max(v_upper, 0) - min(v_lower, 0)) / spacing` per axis, which is non-positive and keeps the
Jacobi iteration's diagonal positive. The forward Euler preflight replaces the constant
Courant term with the per-site outflow sum `(max(v_upper, 0) - min(v_lower, 0)) / spacing`,
maximized over sites.

The runtime does not check discrete divergence. A divergence-free field conserves mass; model
code that authors a field with local divergence gets the corresponding local sources and
sinks, and remains subject to the finite, non-negative level invariant. The
`cellmodeller2.microfluidics` authoring helpers construct divergence-free channel profiles.

Velocity fields are exact checkpoint state in the version 8 signal grid specification as a
`velocity_field` object or `null`; versions 1 through 7 migrate to no field. CPU, Metal, and
CUDA implement the same face-flux operator natively over the three face buffers.

## Validation sequence

1. CPU face-flux transport against hand-computed stencils, including a shear profile, inlet
   and outlet fixed faces, and zero-velocity closed faces beside obstacles.
2. Conservation of total signal mass for a divergence-free field on a closed grid.
3. Crank-Nicolson convergence with the face-velocity diagonal.
4. Native Metal conformance on shared velocity-field fixtures.
5. Native CUDA conformance on shared velocity-field fixtures.

## Consequences

- Channel flow profiles, dead zones inside walls, and inlets and outlets are expressible on
  the signal lattice with standard physical units.
- Face storage triples the memory of the constant convention but makes varying-field fluxes
  discretely conservative.
- The field advects signals only; fluid forces on cells are a separate mechanics contract.
- Higher-order advection remains a separately named integrator, as ADR 0006 states.
