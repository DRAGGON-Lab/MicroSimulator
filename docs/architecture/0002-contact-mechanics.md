# ADR 0002: typed dynamic contact mechanics

- Status: accepted
- Date: 2026-08-14

## Context

Rod contact and mechanical relaxation require irregular GPU work, dynamic topology, reductions, and iterative solves. A fixed-stride contact table is unsafe at high density and mixes cell pairs, planes, and spheres through integer conventions. A generic portable GPU abstraction would also prevent Metal and CUDA from using their native scan, sort, reduction, and synchronization facilities.

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

The solver exposes a matrix-free operator rather than a materialized sparse matrix. The operator is a regularized seven-DOF system. The public result includes convergence status, iteration count, initial and final residuals, and a breakdown reason. A backend may use native reductions and preconditioning, but it must apply the same declared operator and convergence criterion.

For a cell with full capsule length `L = l + 2r`, the reference regularizer uses mass `m = mu_a L`, axial inertia `m r^2 / 2`, transverse inertia `m (L^2 + 3r^2) / 12`, and length weight `gamma`. The operator adds `gamma^-1 M`. A slender-rod tensor gives axial rotation zero energy; this finite-radius tensor does not, which makes the declared seven-DOF operator positive definite for valid cells.

The reference solver starts from zero and uses unpreconditioned conjugate gradient. Residual RMS is `sqrt(dot(r, r) / cell_count)`, and the default iteration limit is `7 * cell_count`. Apparent convergence is checked by recomputing `B^T b - A delta_q`; non-finite residuals, non-finite curvature, and non-positive curvature are reported as typed breakdowns rather than allowed to contaminate the simulation.

Integration is a separate backend-neutral operation over a solver result. It requires convergence by default, applies translation directly, interprets the rotation vector as axis-angle and caps it at five degrees, and computes the length increment as `max(0, desired_increment + delta_l)`. Mechanics may suppress requested elongation but may not directly shorten a cell. All updates are validated before any world-state array is mutated. `Simulation.step` remains the growth-only primitive; `relax_cell_mechanics` is the explicit contact-relaxation operation.

### Growth under compression

`Simulation.step` commits growth before any relaxation runs, and `relax_cell_mechanics` integrates without length increments, so on their own growth and contact do not interact: a colony with nowhere to go keeps elongating into itself. `Simulation.step(dt, GrowthLimitParameters)` makes growth respond to compression. Before growth it finds the cell contacts of the geometry the step starts from, which is the outcome of the previous step's relaxation, and takes each cell's deepest overlap with a neighbor in units of that cell's radius. A cell's elongation over the step is multiplied by a factor that is one up to `free_overlap`, falls linearly, and is zero from `stall_overlap`. The defaults are 0.1 and 0.4. The scaling is applied before species, coupled, and signal updates, so uptake tied to volume change is charged for the growth that happened. Shrinkage is never scaled, and wall contacts do not enter the measure, because the wall projection leaves no wall overlap to read.

The free band exists because relaxation leaves a small overlap between growing neighbors that are free to move, which should not slow them. The rule is a response to measured overlap, not a contact force law, and it inherits what the relaxation can and cannot do:

- It needs relaxation that keeps free neighbors apart. With `gamma = 10` an unconfined colony of a few hundred cells carries overlaps of a quarter radius and more, and the limit then slows it; with `gamma = 100` a 256-cell colony keeps about 94% of its unlimited length.
- In a closed space it stops growth once the space is full, and the population then holds. The regularized solve minimizes the sum of squared overlaps, which does not bound the largest one, so a full space can still hold pairs overlapping by more than a radius. A pair pushed past one radius has inverted and the solve no longer separates it.
- With stiff relaxation (`gamma = 100`) a full closed space keeps trading overlap between pairs, cells that come out shallow resume growing, and the population creeps upward slowly instead of holding.

Walls are impenetrable. The regularized solve weighs each wall row against the cells pressing on it, so on its own it leaves a crowded cell part-way through a wall. After integrating a pass, `relax_cell_mechanics` therefore moves every mobile cell that is more than `1e-4` inside a constraint out of it. Each end of the cell's centerline moves back along the normals of the contacts it is inside, to a depth of `5e-5` so the resting contact stays active for the next solve and visible to contact analysis; a mid-span contact moves both ends; and the cell is rebuilt at its own length between the moved ends. A rod tilted against a wall comes to lie along it, and a rod wedged upright between a floor and a ceiling is turned flat. Up to eight sweeps settle the ends against each other. Length is unchanged, fixed cells are not moved, and the overlap this opens between cells is left to the next contact solve. A cell deeper than a wall's mid-plane would resolve toward the far face, so when any correction exceeds its cell's radius the pass is applied in up to sixteen equal parts that each move a cell at most one radius, projecting after each part. The projection uses the host constraint geometry on every backend. A wall thinner than a cell's diameter, or a correction longer than sixteen radii, is outside this guarantee.

Planes, spheres, boxes, and cylinders are stored in a simulation-owned constraint set with stable IDs shared across constraint kinds. A plane declares a point and a normalized inward normal; its permitted half-space lies in the inward direction. Each finite constraint explicitly declares whether cells are permitted inside or outside. Negative separation means penetration in every case.

Plane half-spaces and inside regions of convex finite constraints are endpoint-extremal, so their contact generation examines both capsule centerline endpoints. Outside regions of finite convex constraints instead minimize the obstacle signed-distance function over the complete centerline and subtract the capsule radius. Sphere minimization uses exact projection onto the segment. Box and cylinder minimization use a deterministic 40-iteration ternary bracket over their convex signed-distance functions; equal sampled distances retain the middle third, selecting a stable central representative on a flat minimum. Cylinder evaluation also includes the analytic radial-axis and axial-center projections, avoiding iterative location error in the barrel and cap cases. Expanded axis-aligned obstacle bounds provide a conservative early rejection for boxes and cylinders.

When a finite outside constraint is closest at the rod mid-span, contact generation emits one row tagged `interior` at that centerline location. When the minimum lies at an endpoint, every endpoint within the activation margin emits its own row: a rod lying against a flat face at a slight angle has both ends inside the surface, and a single row at the deeper end lets the solve answer with a rotation that drives the other end further in. Two endpoint rows each have weight `coefficient / sqrt(2)`; a single row has the full coefficient. An external contact normal points from the permitted region toward the constraint boundary. `point_on_cell` is the capsule surface point reached from the selected centerline location along that normal. Sphere-center, box-face-tie, and cylinder-axis degeneracies use documented deterministic directions before applying the inside/outside orientation. External rows use the same seven-DOF cell Jacobian as pair contacts, but have no second cell term. They therefore enter the declared system as `B_external^T B_external` and `B_external^T b_external` alongside the pair rows. A simulation with constraints must execute both geometry and mechanics on a backend that advertises external-constraint support; unsupported native backends fail explicitly instead of dropping the boundary rows.

The public location type is `RodContactLocation`, and `ExternalContact.location` distinguishes `negative`, `positive`, and `interior`. `RodEndpoint` remains a C++ and Python type alias, and the Python `ExternalContact.endpoint` accessor remains a compatibility alias for existing plane and sphere consumers. The C++ record field is intentionally named `location` because not every contact lies at an endpoint.

The Metal and CUDA implementations upload stable-ID-sorted tagged constraints and run native count/scan/fill pipelines over cell-constraint pairs. Each pair emits zero, one, or two location-tagged rows into dynamic storage. Pair and external rows are then combined in the native solver buffers; a reserved invalid second slot marks the one-sided row inside each private operator without reintroducing a sentinel into the public contact model.

CPU, Metal, and CUDA own separate implementations of the contact geometry and the system operator. They share the conjugate-gradient iteration in `cpp/core/mechanics_solver.hpp`, contact fixtures, operator probes, tolerances, and exact location-tag invariants. Metal uses Metal compute pipelines and MSL; CUDA uses CUDA C++ and the CUDA Runtime API.

### Neighbor reporting

`ContactGraph::neighbor_ids(slot)` derives a neighbor view from the current graph. It returns ascending, unique stable cell IDs. Multiple geometric rows for the same capsule pair therefore contribute one neighbor, and external constraint contacts never appear. The view has the same lifetime as the graph: after growth, division, or mechanical integration, callers compute a new graph rather than treating neighbors as persistent cell state.

## Validation sequence

1. Robust capsule geometry and an exhaustive CPU geometry oracle.
2. Shared contact fixtures and a dynamic contact-graph API.
3. Deterministic sweep-and-prune candidate staging.
4. Native Metal and CUDA count/scan/fill pipelines.
5. CPU matrix-free operator and diagnosed conjugate-gradient solver.
6. Native Metal and CUDA operator, reductions, and solver loops.
7. Typed plane, sphere, box, and cylinder constraint geometry.
8. Full-centerline finite-obstacle contact fixtures, including clear endpoints with penetrating mid-spans.
9. CPU external-constraint mechanics rows.
10. Native Metal external-constraint conformance.
11. Native CUDA external-constraint conformance.

This sequence isolates geometry disagreements before solver behavior can hide them. Backend support requires the corresponding shared scenario to pass on real hardware.

## Consequences

- Pair staging scales with conservative spatial candidates and contact storage scales with actual topology rather than a per-cell maximum.
- Stable cell identity survives compaction while kernels continue to use dense slots.
- GPU implementations duplicate orchestration where native primitives differ.
- Deterministic mode may require an ordering pass that fast mode can skip.
