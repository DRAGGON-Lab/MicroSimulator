# ADR 0018: axis-aligned cylinder constraints

- Status: accepted
- Date: 2026-08-16

## Context

Circular confinement is a basic experimental geometry: culture dishes, circular micro-chambers, and round pillars. The existing constraint kinds cannot express it. A sphere's inside region confines a monolayer to a circle only at its equator and renders as a ball; a polygonal ring of boxes scallops the boundary. The constraint pipelines pack each external constraint into two `float4` payloads on Metal and CUDA; a z-aligned finite cylinder fits that budget, an arbitrarily oriented one does not.

## Decision

MicroSimulator adds a fourth external constraint kind: the z-aligned cylinder. A cylinder declares a finite center, a positive finite radius, a positive finite half-height, a positive finite coefficient, and an allowed `ConstraintRegion`. A round dish or circular chamber is an inside cylinder; a round pillar is an outside cylinder.

For a centerline point `p`, let `d` be its radial distance from the cylinder axis in the xy plane, `dz` its signed z offset from the center, `R` the radius, and `H` the half-height. The signed distance to the cylinder surface and the outward direction are:

- Outside both bounds (`d > R` and `|dz| > H`): the rim corner case. The distance is the hypotenuse of the radial and axial excesses, and the outward direction blends the radial unit vector and the signed z axis in proportion to those excesses.
- Radially outside only: the distance is `d - R` and the outward direction is the radial unit vector.
- Axially outside only: the distance is `|dz| - H` and the outward direction is the signed z axis.
- Inside: the distance is the negated smaller of the radial clearance `R - d` and the axial clearance `H - |dz|`, and the outward direction is the corresponding unit vector. A tie selects the radial direction, and a point within the degeneracy epsilon of the axis uses the positive x axis as its radial direction, matching the sphere rule.

For an outside cylinder, the implementation minimizes this convex signed-distance function over the complete capsule centerline, then computes `signed_distance - radius`. A conservative segment-versus-bounds test expands the cylinder's axis-aligned bounds by the capsule radius and activation margin before the narrow phase. The narrow phase uses the same deterministic 40-iteration ternary bracket as outside boxes and also evaluates the analytic radial-axis and axial-center projections. If both endpoints attain the global minimum, both endpoint rows are retained with `1/sqrt(2)` weighting; otherwise one row is emitted at the minimizing centerline position and tagged `interior`. This detects rods that cross a finite pillar at mid-span with both endpoints clear.

For an inside cylinder, both endpoints remain sufficient because the cylinder is convex. An outside cylinder has contact normal opposite the outward direction; an inside cylinder has contact normal along the outward direction. The normal points from the permitted region toward the constraint boundary, negative separation means penetration, and `point_on_cell` is reached from the selected centerline position along the normal.

Cylinder rows use the existing one-sided seven-DOF external Jacobian. The GPU payload packs the center into the geometry vector with the radius in its fourth component, and the half-height plus coefficient into the parameter vector. Cylinders share the constraint set's stable identifier space, ride the `external_constraints` backend feature, and are recorded in the checkpoint's version 8 constraint envelope as a `cylinders` array; versions 1 through 7 migrate to an empty list. Scene format v2 carries cylinders alongside the other constraint kinds for device rendering.

## Validation sequence

1. CPU cylinder signed-distance geometry across barrel, cap, rim-corner, interior, and axis-degenerate points, for outside and inside regions.
2. Shared cylinder contact fixtures alongside the existing conformance scenarios.
3. Mid-span pillar fixtures whose capsule intersects while both centerline endpoints remain clear.
4. CPU external-constraint mechanics rows for cylinder fixtures.
5. Native Metal cylinder conformance.
6. Native CUDA cylinder conformance.

## Consequences

- A culture dish is one inside cylinder plus a floor plane, with an exact circular boundary at every z rather than a sphere's equator.
- The interior nearest-wall rule gives penetrating cells a deterministic escape direction, with radial escape preferred on exact ties.
- Outside-cylinder contact covers the complete capsule centerline rather than only its endpoints.
- Tilted cylinders remain out of scope; expressing one would require widening the packed GPU constraint record.
