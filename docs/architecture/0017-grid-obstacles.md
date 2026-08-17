# ADR 0017: signal grid obstacles

- Status: accepted
- Date: 2026-08-16

## Context

Mechanical wall constraints are invisible to the signal grid: signals diffuse and advect
straight through the material that blocks cells. A microfluidic model therefore cannot
represent a trap wall that cells and chemistry both respect. The existing boundary kinds apply
only to the six outer faces of the lattice and cannot describe an interior solid.

## Decision

`SignalGridSpec` may declare a per-voxel obstacle mask: one `uint8` per lattice site in the
grid's ordinary x/y/z-major site order, shared by every signal. An empty mask means the grid
has no obstacles. A present mask must cover every site and contain only the values 0 (fluid)
and 1 (solid). Model code may rasterize wall geometry with any predicates while constructing
the mask; the runtime receives only the materialized array, following the affine reaction
precedent.

Solid sites carry no concentration and no dynamics: their levels must be zero in every
checkpoint and committed field, their affine reaction coefficients must be zero, their
transport rate is zero, and their operator diagonal is zero.

Every lattice face between a fluid site and a solid site is closed, with exactly the exterior
`no_flux` semantics: the diffusive stencil substitutes the fluid site's own value, the
advective face flux is zero, and the Crank-Nicolson diagonal drops the corresponding
contributions. A periodic face that wraps onto a solid site is also closed. The forward Euler
stability bound is unchanged; closing faces only removes flux terms, so the declared bound
remains conservative.

Trilinear sampling and scatter renormalize over fluid sites: stencil weights at solid sites
are zeroed and the remaining weights are divided by their sum, so sampling and scatter stay
adjoint and every scattered amount lands in fluid volume. A cell whose entire stencil is solid
is a model error detected before mutation, like an out-of-domain cell. This admits the common
case of a cell pressed against a wall, whose interpolation cube straddles the wall surface.

Obstacle masks are exact checkpoint state in schema version 8; versions 1 through 7 migrate to
no obstacles. CPU, Metal, and CUDA implement the same closed-face operator natively; a backend
without obstacle support must reject a masked grid explicitly rather than transport through
walls.

## Validation sequence

1. CPU closed-face transport against hand-computed stencils, for interior walls, wall-adjacent
   upwind advection, and periodic wrap onto a solid site.
2. Conservation of total signal mass in a closed masked region under diffusion and advection.
3. Renormalized sampling and scatter adjointness beside a wall, and the all-solid stencil
   error.
4. Crank-Nicolson convergence on masked grids with the diagonal matching the closed-face
   operator.
5. Native Metal conformance on shared masked fixtures.
6. Native CUDA conformance on shared masked fixtures.

## Consequences

- Wall geometry can be rasterized once into the mask, so mechanics and chemistry share one
  source of truth in model code.
- A trap's interior chemistry becomes physical: signal escapes only through the trap opening
  rather than through its walls.
- Voxelized walls are staircase approximations of the box constraints they rasterize; the
  mask resolution is the grid resolution.
- Renormalized weights make near-wall sampling exact for conservation but slightly reweight
  the interpolation toward fluid sites; models comparing against unmasked runs should expect
  differences only within one voxel of a wall.
