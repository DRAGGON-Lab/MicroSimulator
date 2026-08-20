# ADR 0016: axis-aligned box constraints

- Status: accepted
- Date: 2026-08-16

## Context

Microfluidic devices confine cells with finite rectilinear features: trap walls with an open side, pillars, and channel ceilings. The existing external constraints cannot express finite extent. A plane is an infinite half-space, so an open-sided trap built from planes seals the channel it should feed. Models today approximate finite walls with dense rows of sphere constraints, which produces a bumpy surface, scales the constraint count with wall length divided by cell radius, and leaks cells between spheres.

The native constraint pipelines pack each external constraint into two `float4` payloads on Metal and CUDA. An oriented box does not fit in that budget; an axis-aligned box does.

## Decision

MicroSimulator adds a third external constraint kind: the axis-aligned box. A box declares a finite center, positive finite half-extents on each axis, a positive finite coefficient, and an allowed region that states whether cells are permitted outside or inside the box. A solid wall or pillar is an outside box; a closed rectangular chamber is a single inside box. The allowed region is the shared `ConstraintRegion` enumeration; `SphereRegion` remains as an alias for existing models.

For a centerline point `p`, the signed distance to the box surface is derived from the per-axis clamp of `p - center` to the half-extents:

- Outside the box, the surface delta is `(p - center) - clamp(p - center)`, the signed distance is its magnitude, and the outward direction is its normalization.
- Inside the box or within the degeneracy epsilon of its surface, the signed distance is the negated smallest per-axis face clearance, and the outward direction is the signed axis of that smallest clearance. Ties select the first axis in x, y, z order, and a zero axis offset selects the positive direction. This resolves face, edge, corner, and interior cases with one deterministic rule.

For an outside box, the implementation minimizes this convex signed-distance function over the complete capsule centerline, then computes `signed_distance - radius`. A conservative segment-versus-box test expands the box by the capsule radius and activation margin before the narrow phase. The narrow phase uses a deterministic 40-iteration ternary bracket; equal sampled distances retain the middle third so a flat minimum produces a stable central representative. If both rod endpoints attain the global minimum, both endpoint rows are retained with `1/sqrt(2)` weighting. Otherwise one row is emitted at the minimizing centerline position and tagged `interior`. A rod can therefore intersect a finite wall at mid-span even when both endpoints are clear without escaping contact generation.

For an inside box, both endpoints remain sufficient because the box is convex and containing the two endpoint spheres contains their capsule hull. Its separation is `-signed_distance - radius`. An outside box has contact normal opposite the outward direction; an inside box has contact normal along the outward direction. In both cases the normal points from the permitted region toward the constraint boundary, negative separation means penetration, and `point_on_cell` is reached from the selected centerline position along the normal.

Box rows use the existing one-sided seven-DOF external Jacobian and flow through the mechanics operator unchanged. Backends advertise boxes through the existing `external_constraints` feature; the CPU, Metal, and CUDA constraint pipelines all evaluate the box kind natively, and the GPU payload packs the center into the geometry vector and the half-extents plus coefficient into the parameter vector without widening the constraint record.

Boxes are stored in the simulation-owned constraint set with the same shared stable identifier space as planes and spheres. Checkpoint version 8 records a `boxes` array in the constraint envelope; versions 1 through 7 migrate to an empty box list.

## Validation sequence

1. CPU box signed-distance geometry across face, edge, corner, interior, and center-degenerate endpoints, for outside and inside regions.
2. Shared box contact fixtures alongside the existing plane and sphere conformance scenarios.
3. Mid-span wall fixtures whose capsule intersects while both centerline endpoints remain clear.
4. CPU external-constraint mechanics rows for box fixtures.
5. Native Metal box conformance.
6. Native CUDA box conformance.

## Consequences

- Finite rectilinear device geometry uses one constraint per wall instead of a sphere row per cell radius, and the wall surface is exact rather than scalloped.
- The interior nearest-face rule gives penetrating cells a deterministic escape direction, so deep penetration resolves toward the closest face rather than oscillating.
- A thin wall is a box with a small half-extent; cells approaching from both sides and cells crossing its finite span are detected by the same full-capsule rule.
- Oriented boxes remain out of scope; expressing one would require widening the packed GPU constraint record and a separate constraint kind.
