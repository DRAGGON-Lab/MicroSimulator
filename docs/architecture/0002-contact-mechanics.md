# ADR 0002: typed dynamic contact mechanics

- Status: accepted
- Date: 2026-08-14

## Context

Rod contact and its mechanics require irregular GPU work, dynamic topology, reductions, and iterative solves. A fixed-stride contact table is unsafe at high density and mixes cell pairs, planes, and spheres through integer conventions. A generic portable GPU abstraction would also prevent Metal and CUDA from using their native scan, sort, reduction, and synchronization facilities.

## Decision

MicroSimulator represents a contact as a typed record with stable cell IDs, current compact slots, contact ordinal or centerline location, surface point, unit normal, signed separation, and row weight. External constraints use tagged plane, sphere, box, or cylinder references rather than sentinel cell IDs.

The CPU reference performs an exhaustive pair search. It is intentionally simple and is the geometry oracle for small conformance scenarios.

Contact discovery uses a deterministic sweep-and-prune broad phase over capsule axis-aligned bounds. Each bound expands the centerline by its radius and half the activation margin, so overlap is a conservative prerequisite for the narrow-phase separation test. Bounds are sorted by minimum x coordinate and stable cell ID; candidates are emitted in stable-ID order. Sparse staging therefore scales with the number of overlapping bounds rather than allocating an `N x N` pair matrix.

Metal and CUDA consume that candidate list through a native two-pass dynamic pipeline:

1. upload compact candidate slot pairs with the current cell arrays;
2. count zero, one, or two narrow-phase contacts per candidate;
3. scan the counts into offsets with native kernels;
4. allocate or grow the contact arrays;
5. fill the typed contact records with native kernels.

There is no scientific contact cap. Allocation failure is explicit. In deterministic mode, contacts are ordered by stable endpoint IDs and contact ordinal. Fast mode may use backend-native ordering, but consumers must not infer identity from contact-array position.

The narrow phase distinguishes one-row from two-row contacts and weights each row of a two-row contact by `1/sqrt(2)`. Degenerate normals are resolved deterministically by projecting center displacement perpendicular to the rod axis; if that also vanishes, the least-aligned Cartesian basis axis defines a perpendicular. Every emitted normal must be finite and unit length.

Contacts are the input to the [contact dynamics](0026-contact-dynamics.md) solve, which turns each one into a row with a non-negative force and solves contact together with growth. That record defines the cell drag, the solve, and how cells and walls are kept apart.

Planes, spheres, boxes, and cylinders are stored in a simulation-owned constraint set with stable IDs shared across constraint kinds. A plane declares a point and a normalized inward normal; its permitted half-space lies in the inward direction. Each finite constraint explicitly declares whether cells are permitted inside or outside. Negative separation means penetration in every case.

Plane half-spaces and inside regions of convex finite constraints are endpoint-extremal, so their contact generation examines both capsule centerline endpoints. Outside regions of finite convex constraints instead minimize the obstacle signed-distance function over the complete centerline and subtract the capsule radius. Sphere minimization uses exact projection onto the segment. Box and cylinder minimization use a deterministic 40-iteration ternary bracket over their convex signed-distance functions; equal sampled distances retain the middle third, selecting a stable central representative on a flat minimum. Cylinder evaluation also includes the analytic radial-axis and axial-center projections, avoiding iterative location error in the barrel and cap cases. Expanded axis-aligned obstacle bounds provide a conservative early rejection for boxes and cylinders.

When a finite outside constraint is closest at the rod mid-span, contact generation emits one row tagged `interior` at that centerline location. When the minimum lies at an endpoint, every endpoint within the activation margin emits its own row: a rod lying against a flat face at a slight angle has both ends inside the surface, and a single row at the deeper end lets the solve answer with a rotation that drives the other end further in. Two endpoint rows each have weight `coefficient / sqrt(2)`; a single row has the full coefficient. An external contact normal points from the permitted region toward the constraint boundary. `point_on_cell` is the capsule surface point reached from the selected centerline location along that normal. Sphere-center, box-face-tie, and cylinder-axis degeneracies use documented deterministic directions before applying the inside/outside orientation. A contact with a constraint has no second cell. A simulation with constraints must find those contacts on a backend that advertises external-constraint support; unsupported native backends fail explicitly instead of dropping the boundary contacts.

The public location type is `RodContactLocation`, and `ExternalContact.location` distinguishes `negative`, `positive`, and `interior`. `RodEndpoint` remains a C++ and Python type alias, and the Python `ExternalContact.endpoint` accessor remains a compatibility alias for existing plane and sphere consumers. The C++ record field is intentionally named `location` because not every contact lies at an endpoint.

The Metal and CUDA implementations upload stable-ID-sorted tagged constraints and run native count/scan/fill pipelines over cell-constraint pairs. Each pair emits zero, one, or two location-tagged rows into dynamic storage.

CPU, Metal, and CUDA own separate implementations of the contact geometry. They share contact fixtures, tolerances, and exact location-tag invariants. Metal uses Metal compute pipelines and MSL; CUDA uses CUDA C++ and the CUDA Runtime API.

### Neighbor reporting

`ContactGraph::neighbor_ids(slot)` derives a neighbor view from the current graph. It returns ascending, unique stable cell IDs. Multiple geometric rows for the same capsule pair therefore contribute one neighbor, and external constraint contacts never appear. The view has the same lifetime as the graph: after growth, division, or mechanical integration, callers compute a new graph rather than treating neighbors as persistent cell state.

## Validation sequence

1. Robust capsule geometry and an exhaustive CPU geometry oracle.
2. Shared contact fixtures and a dynamic contact-graph API.
3. Deterministic sweep-and-prune candidate staging.
4. Native Metal and CUDA count/scan/fill pipelines.
5. Typed plane, sphere, box, and cylinder constraint geometry.
6. Full-centerline finite-obstacle contact fixtures, including clear endpoints with penetrating mid-spans.
7. Native Metal external-constraint conformance.
8. Native CUDA external-constraint conformance.

This sequence isolates geometry disagreements before solver behavior can hide them. Backend support requires the corresponding shared scenario to pass on real hardware.

## Consequences

- Pair staging scales with conservative spatial candidates and contact storage scales with actual topology rather than a per-cell maximum.
- Stable cell identity survives compaction while kernels continue to use dense slots.
- GPU implementations duplicate orchestration where native primitives differ.
- Deterministic mode may require an ordering pass that fast mode can skip.
