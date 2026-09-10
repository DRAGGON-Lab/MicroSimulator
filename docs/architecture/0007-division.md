# ADR 0007: explicit daughter fractions

- Status: accepted
- Date: 2026-08-14

## Context

The legacy simulator passes `CellState.asymm` to the biophysics layer as two relative weights. The shipped `CLBacterium.divide` accepts those keyword arguments but drops them before `divide_cell`, so every maintained release still constructs equal daughters. MicroSimulator needs a defined asymmetric operation rather than preserving that accidental no-op.

Cell geometry stores the capsule centerline length `l` and radius `r`. The legacy equal split creates two centerline lengths `l/2 - r`, places the daughter outer endpoints at the parent's outer endpoints, and leaves a `2r` centerline gap where the daughter caps touch.

## Decision

`Simulation.divide(parent_id, first_fraction)` accepts a finite fraction strictly between zero and one. It divides the available daughter centerline span `l - 2r` as

```
l1 = (l - 2r) first_fraction
l2 = (l - 2r) (1 - first_fraction)
```

The first daughter occupies the parent's negative-axis end and the second occupies its positive-axis end. Their outer endpoints match the parent and the gap between their centerline endpoints remains `2r`. Growth rate, cell type, radius, direction, and species concentrations are copied to both daughters; stable IDs and lineage follow the existing lifecycle contract.

`divide_equal(parent_id)` remains as the readable `first_fraction = 0.5` operation. The legacy adapter normalizes the two positive `asymm` weights and calls the same primitive. Invalid fractions or weights fail before topology is mutated.

Division is a backend-neutral topology event. Metal and CUDA do not duplicate it in a kernel: their next native compute operation consumes the updated dense world arrays. The shared lifecycle conformance scenario runs both equal and asymmetric events on every available backend.

## Consequences

- Asymmetric division now implements the dormant legacy intent rather than the historical `CLBacterium` no-op.
- The fraction describes axial geometry, not biochemical partitioning.
- Species values remain concentrations and are copied, matching the existing equal-division integrator behavior.
- Parent rods shorter than `2r` and endpoint fractions are rejected.
