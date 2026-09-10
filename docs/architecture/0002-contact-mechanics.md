# ADR 0002: typed dynamic contact mechanics

- Status: accepted
- Date: 2026-08-14

## Context

Rod contact and mechanical relaxation are the first CellModeller feature that requires irregular GPU work, dynamic topology, reductions, and iterative solves. The legacy fixed-stride contact table is unsafe at high density and mixes cell pairs, planes, and spheres through integer conventions. A generic portable GPU abstraction would also prevent Metal and CUDA from using their native scan, sort, reduction, and synchronization facilities.

## Decision

MicroSimulator represents a contact as a typed record with stable cell IDs, current compact slots, contact ordinal, surface point, unit normal, signed separation, and row weight. External constraints use tagged plane or sphere references rather than sentinel cell IDs.

The CPU reference performs an exhaustive pair search. It is intentionally simple and is the geometry oracle for small conformance scenarios.

Contact discovery uses a deterministic sweep-and-prune broad phase over capsule axis-aligned bounds. Each bound expands the centerline by its radius and half the activation margin, so overlap is a conservative prerequisite for the narrow-phase separation test. Bounds are sorted by minimum x coordinate and stable cell ID; candidates are emitted in stable-ID order. Sparse staging therefore scales with the number of overlapping bounds rather than allocating an `N x N` pair matrix.

Metal and CUDA consume that candidate list through a native two-pass dynamic pipeline:

1. upload compact candidate slot pairs with the current cell arrays;
2. count zero, one, or two narrow-phase contacts per candidate;
3. scan the counts into offsets with native kernels;
4. allocate or grow the contact arrays;
5. fill the typed contact records with native kernels.

There is no scientific contact cap. Allocation failure is explicit. In deterministic mode, contacts are ordered by stable endpoint IDs and contact ordinal. Fast mode may use backend-native ordering, but consumers must not infer identity from contact-array position.

The narrow phase preserves the legacy one-row/two-row distinction and the `1/sqrt(2)` two-row weighting. Degenerate normals are resolved deterministically by projecting center displacement perpendicular to the rod axis; if that also vanishes, the least-aligned Cartesian basis axis defines a perpendicular. Every emitted normal must be finite and unit length.

The solver exposes a matrix-free operator rather than a materialized sparse matrix. The compatibility operator is the regularized seven-DOF system described in the CellModeller mechanics comparison. The public result includes convergence status, iteration count, initial and final residuals, and a breakdown reason. A backend may use native reductions and preconditioning, but it must apply the same declared operator and convergence criterion.

For a cell with full capsule length `L = l + 2r`, the reference regularizer uses mass `m = mu_a L`, axial inertia `m r^2 / 2`, transverse inertia `m (L^2 + 3r^2) / 12`, and length weight `gamma`. The operator adds `gamma^-1 M`. This finite-radius tensor intentionally replaces the legacy slender-rod tensor's zero-energy axial rotation, making the declared seven-DOF operator positive definite for valid cells. Legacy trajectory comparisons must identify this as an intentional numerical-model difference.

The reference solver starts from zero and uses unpreconditioned conjugate gradient. Residual RMS is `sqrt(dot(r, r) / cell_count)`, preserving the legacy normalization, and the default iteration limit is `7 * cell_count`. Apparent convergence is checked by recomputing `B^T b - A delta_q`; non-finite residuals, non-finite curvature, and non-positive curvature are reported as typed breakdowns rather than allowed to contaminate the simulation.

Integration is a separate backend-neutral operation over a solver result. It requires convergence by default, applies translation directly, interprets the rotation vector as axis-angle and caps it at five degrees, and computes the length increment as `max(0, desired_increment + delta_l)`. This preserves the legacy rule that mechanics may suppress requested elongation but may not directly shorten a cell. All updates are validated before any world-state array is mutated. `Simulation.step` remains the growth-only primitive; `relax_cell_mechanics` is the explicit contact-relaxation operation.

Planes and spheres are stored in a simulation-owned constraint set with stable IDs shared across constraint kinds. A plane declares a point and a normalized inward normal; its permitted half-space lies in the inward direction. A sphere explicitly declares whether cells are permitted inside or outside. Constraint contact generation examines both capsule centerline endpoints and emits a typed row when its signed surface separation is below the activation margin. Negative separation means penetration in every case.

An external contact normal points from the permitted region toward the constraint boundary: opposite a plane's inward normal, toward the center of an outside sphere, and away from the center of an inside sphere. `point_on_cell` is the capsule surface point reached from the centerline endpoint along that normal, not the endpoint itself. When both rod endpoints are active, each row has weight `coefficient / sqrt(2)`; otherwise the single row has the full coefficient. A sphere-center degeneracy uses the positive x-axis as its deterministic radial direction before applying the inside/outside orientation. External rows use the same seven-DOF cell Jacobian as pair contacts, but have no second cell term. They therefore enter the declared system as `B_external^T B_external` and `B_external^T b_external` alongside the pair rows. A simulation with constraints must execute both geometry and mechanics on a backend that advertises external-constraint support; unsupported native backends fail explicitly instead of dropping the boundary rows.

The Metal and CUDA implementations upload stable-ID-sorted tagged constraints and run native count/scan/fill pipelines over cell-constraint pairs. Each pair emits zero, one, or two endpoint rows into dynamic storage. Pair and external rows are then combined in the native solver buffers; a reserved invalid second slot marks the one-sided row inside each private operator without reintroducing a sentinel into the public contact model.

CPU, Metal, and CUDA own separate implementations. They share contact fixtures, operator probes, tolerances, and exact endpoint invariants. Metal uses Metal compute pipelines and MSL; CUDA uses CUDA C++ and the CUDA Runtime API.

### Neighbor reporting

`ContactGraph::neighbor_ids(slot)` derives the legacy neighbor view from the current graph. It returns ascending, unique stable cell IDs. Multiple geometric rows for the same capsule pair therefore contribute one neighbor, and external constraint contacts never appear. The view has the same lifetime as the graph: after growth, division, or mechanical integration, callers compute a new graph rather than treating neighbors as persistent cell state.

## Validation sequence

1. Robust capsule geometry and an exhaustive CPU geometry oracle.
2. Shared contact fixtures and a dynamic contact-graph API.
3. Deterministic sweep-and-prune candidate staging.
4. Native Metal and CUDA count/scan/fill pipelines.
5. CPU matrix-free operator and diagnosed conjugate-gradient solver.
6. Native Metal and CUDA operator, reductions, and solver loops.
7. Typed plane and sphere constraint geometry.
8. CPU external-constraint mechanics rows.
9. Native Metal external-constraint conformance.
10. Native CUDA external-constraint conformance.

This sequence isolates geometry disagreements before solver behavior can hide them. Backend support requires the corresponding shared scenario to pass on real hardware.

## Consequences

- Pair staging scales with conservative spatial candidates and contact storage scales with actual topology rather than a per-cell maximum.
- Stable cell identity survives compaction while kernels continue to use dense slots.
- GPU implementations duplicate orchestration where native primitives differ.
- Deterministic mode may require an ordering pass that fast mode can skip.
- Legacy trajectories with undefined normals or overflowed contact rows are not reproducibility targets.
