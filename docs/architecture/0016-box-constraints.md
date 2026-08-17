# ADR 0016: axis-aligned box constraints

- Status: accepted
- Date: 2026-08-16

## Context

Microfluidic devices confine cells with finite rectilinear features: trap walls with an open
side, pillars, and channel ceilings. The existing external constraints cannot express finite
extent. A plane is an infinite half-space, so an open-sided trap built from planes seals the
channel it should feed. Models today approximate finite walls with dense rows of sphere
constraints, which produces a bumpy surface, scales the constraint count with wall length
divided by cell radius, and leaks cells between spheres.

The native constraint pipelines pack each external constraint into two `float4` payloads on
Metal and CUDA. An oriented box does not fit in that budget; an axis-aligned box does.

## Decision

CellModeller2 adds a third external constraint kind: the axis-aligned box. A box declares a
finite center, positive finite half-extents on each axis, a positive finite coefficient, and an
allowed region that states whether cells are permitted outside or inside the box. A solid wall
or pillar is an outside box; a closed rectangular chamber is a single inside box. The allowed
region is the shared `ConstraintRegion` enumeration; `SphereRegion` remains as an alias for
existing models.

Contact generation follows the plane and sphere convention: both capsule centerline endpoints
are examined, each active endpoint emits a typed row, and two active rows share the
`1/sqrt(2)` weighting. For an endpoint at position `p`, the signed distance to the box surface
is derived from the per-axis clamp of `p - center` to the half-extents:

- Outside the box, the surface delta is `(p - center) - clamp(p - center)`, the signed
  distance is its magnitude, and the outward direction is its normalization.
- Inside the box or within the degeneracy epsilon of its surface, the signed distance is the
  negated smallest per-axis face clearance, and the outward direction is the signed axis of
  that smallest clearance. Ties select the first axis in x, y, z order, and a zero axis offset
  selects the positive direction. This resolves face, edge, corner, and interior cases with one
  deterministic rule.

An outside box has separation `signed_distance - radius` and contact normal opposite the
outward direction; an inside box has separation `-signed_distance - radius` and contact normal
along the outward direction. In both cases the normal points from the permitted region toward
the constraint boundary and negative separation means penetration, exactly as for planes and
spheres. `point_on_cell` is the capsule surface point reached from the centerline endpoint
along the normal.

Box rows use the existing one-sided seven-DOF external Jacobian and flow through the mechanics
operator unchanged. Backends advertise boxes through the existing `external_constraints`
feature; the CPU, Metal, and CUDA constraint pipelines all evaluate the box kind natively, and
the GPU payload packs the center into the geometry vector and the half-extents plus
coefficient into the parameter vector without widening the constraint record.

Boxes are stored in the simulation-owned constraint set with the same shared stable identifier
space as planes and spheres. Checkpoint version 8 records a `boxes` array in the constraint
envelope; versions 1 through 7 migrate to an empty box list.

## Validation sequence

1. CPU box signed-distance geometry across face, edge, corner, interior, and center-degenerate
   endpoints, for outside and inside regions.
2. Shared box contact fixtures alongside the existing plane and sphere conformance scenarios.
3. CPU external-constraint mechanics rows for box fixtures.
4. Native Metal box conformance.
5. Native CUDA box conformance.

## Consequences

- Finite rectilinear device geometry uses one constraint per wall instead of a sphere row per
  cell radius, and the wall surface is exact rather than scalloped.
- The interior nearest-face rule gives penetrating cells a deterministic escape direction, so
  deep penetration resolves toward the closest face rather than oscillating.
- A thin wall is a box with a small half-extent; cells approaching from both sides are pushed
  to their respective nearest faces by the same rule.
- Oriented boxes remain out of scope; expressing one would require widening the packed GPU
  constraint record and a separate constraint kind.
