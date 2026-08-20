# Backend conformance scenarios

The shared numerical scenarios run against every enumerated device in a test-enabled build. This reference describes their inputs, observables, and tolerances. The growth scenario deliberately uses 513 heterogeneous cells so native launches cross common threadgroup and block boundaries.

## Growth

The scenario checks the declared explicit Euler recurrence independently:

```text
length[n + 1] = length[n] + growth_rate * length[n] * dt
```

Lengths use an absolute tolerance of `1e-6` and a relative tolerance of `1e-6`. Stable identifiers, compact slots, growth rates, cell types, backend identity, and structural validation are exact checks.

A backend only earns conformance when this executable passes with that native backend enabled on its real hardware. A CPU-only run validates the scenario and reference implementation, but says nothing about GPU conformance.

## Backend contract

The backend contract scenario constructs every enumerated device and requires it to advertise growth, species, cell contacts, cell mechanics, external constraints, signals, and coupled rates. Individual scientific fixtures may retain capability guards to diagnose partially implemented development builds, but a feature-complete Metal or CUDA build cannot pass the conformance suite by opting out of one of those fixtures.

## Species

The species scenario uses 513 cells, three concentrations, heterogeneous geometry and cell types, and a typed plan that reads concentrations and cell attributes. It checks growth dilution, post-dilution rate evaluation, simultaneous explicit Euler updates, every declared instruction operation, zero-length time steps, stable identity, and cell-major schema preservation. Levels and lengths use absolute and relative tolerances of `2e-5`; identities and shapes are exact.

## Lifecycle

The lifecycle scenario interleaves growth with two generations of deterministic equal division. Stable identifiers, slot reuse, active ordering, lineage, daughter geometry, inherited cell attributes, and simulation time are checked. Floating-point length and position checks use the same `1e-6` absolute and relative tolerances as the growth scenario; lifecycle identities are exact.

## Composed trajectory

The trajectory scenario composes coupled intracellular/extracellular rates, growth, cell and plane contacts, fixed-cell projection, mechanics relaxation, and equal division over three heterogeneous time steps. It compares every intermediate cell and signal state with a fresh CPU trajectory, requires each mechanics solve to converge, checks exact lineage identities, and confirms the final contact count. Geometry uses a `2e-3` absolute and relative tolerance, species use `2e-4`, and signal levels use `5e-4`; the wider bounds account for native solver differences feeding subsequent trajectory stages.

## Signal grid

The signal-grid scenario uses two fields on a 9-by-7-by-5 anisotropic lattice, mixed diffusion and vector advection, periodic x boundaries, no-flux y boundaries, and distinct fixed reservoirs on the z faces. One native transport step and an interior trilinear sample are compared with the CPU reference using absolute and relative tolerances of `5e-6`. An available backend is skipped until it advertises native signal-grid support; after that advertisement this executable is its hardware conformance gate.

## Coupled rates

The coupled scenario uses 513 heterogeneous cells, three intracellular species, and two signals on a 9-by-7-by-5 anisotropic lattice. It combines post-growth dilution, fractional old-field sampling, typed species and signal outputs, transport, and repeated trilinear scatter destinations in one step. Cell state uses an absolute tolerance of `2e-5` and grid state uses `1e-4` to allow backend-specific floating-point accumulation order. An available backend is skipped until it advertises the complete native coupled operation; after advertisement this executable becomes its hardware gate.

## Checkpoint resume

The checkpoint scenario captures a divided colony with a typed species plan, mixed constraints, inactive ancestry, and nontrivial identity frontiers. Every available backend must restore all persisted host fields exactly, then execute the next growth/species step within `1e-5` of a fresh CPU restore. The Python suite separately verifies the JSON envelope, atomic write behavior, provenance, integrity digest, and malformed-input rejection.

## Cell contacts

The contact scenario compares each contact-capable backend with the CPU geometry oracle for mixed end-on, parallel, anti-parallel, skew, point-like, empty, and single-cell geometry. Both paths consume the deterministic sweep-and-prune candidate list; focused CPU tests additionally prove that 2,048 widely separated capsules stage no candidate pairs and that dense bounds retain every pair. The shared scenario also checks stable-ID ordering after slot reuse and a 31-cell coincident case with 60 incident contacts per cell. Contact fields use absolute and relative tolerances of `2e-5`; IDs, slots, ordinals, graph sizes, and incidence indices are exact.

## External constraints

The external-constraint scenario compares every typed field and per-cell incidence index with the CPU oracle for mixed planes, spheres, boxes, and cylinders; inside and outside regions; empty inputs; stable IDs across constraint kinds; deterministic degeneracies; and finite obstacles intersected only by a capsule mid-span. Geometry fields use absolute and relative tolerances of `2e-5`; identities, kinds, centerline-location tags, graph sizes, and incidence are exact.

## Mechanics

The CPU mechanics fixtures probe the declared regularizer, operator symmetry and positive definiteness, solver convergence, exact residual recomputation, iteration-limit reporting, and non-finite-curvature breakdown. The shared mechanics scenario then compares native correction vectors and diagnostics with the CPU reference for a mixed colony, a one-iteration limit, empty systems, and buffer growth. It also applies a converged mixed-colony result and compares the resulting position, direction, and length state. Corrections, geometry, and initial residuals use absolute and relative tolerances of `3e-4`; convergence status and breakdown type are exact.

## Constraint mechanics

The external-constraint mechanics scenario compares correction vectors and integrated cell geometry with the CPU reference for plane, sphere, box, and cylinder boundaries, including outside finite obstacles whose only contact lies at the rod mid-span. It uses the mechanics tolerances above and requires an available backend without external-constraint support to reject a nonempty constraint set explicitly. The same executable is the native Metal and CUDA hardware acceptance gate.
