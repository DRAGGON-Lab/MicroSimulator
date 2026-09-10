# Numerical contract

MicroSimulator separates exact state semantics from floating-point agreement.

## Exact across backends

- Stable cell identifiers and parent-child lineage.
- The number and type of lifecycle events for deterministic scenarios.
- Array shapes, active-cell counts, and checkpoint schema versions.
- Failure on capacity, invalid state, or unsupported capability.
- Contact ordinals and centerline-location tags when the fixture does not contain geometric degeneracy.

## Compared by tolerance

- Position, orientation, length, radius, area, and volume.
- Contact point, normal, overlap, and stiffness.
- Operator applications and solver residuals.
- Species and signal levels and rates.

Every conformance scenario records absolute and relative tolerances per field. Tolerances may not be widened solely to make a backend pass.

## Reproducibility modes

`deterministic` uses counter-based random numbers, stable event ordering, and reproducible reduction policies where the backend permits them. It promises repeatability on the same device and toolchain, not bitwise identity across different GPU architectures.

`fast` permits backend-specific ordering and algebraic fusion. Long-running chaotic simulations are compared through declared ensemble statistics.

## Precision

The portable production contract is IEEE 754 binary32. The CPU backend also supports a binary64 diagnostic implementation where useful. Backend-specific binary64 support is not part of the portable contract.
