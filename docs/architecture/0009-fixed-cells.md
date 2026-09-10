# ADR 0009: persistent fixed rod cells

- Status: accepted
- Date: 2026-08-15

## Context

The legacy `CLFixedPosition` module is an alternate point-volume biophysics model. It stores cell centers and scalar volumes, advances volume with explicit Euler growth, and never changes position. It does not expose a per-cell fixed flag and does not use the capsule geometry or mechanics operator of the main rod model.

MicroSimulator needs the useful behavior without introducing a second, incompatible geometry representation. Fixedness also needs to survive division and checkpoint restore and to behave identically on every backend.

## Decision

Fixedness is a persistent Boolean cell attribute. A fixed cell remains a rod: growth, species rates, signal sampling and secretion, and division continue to use the ordinary cell lifecycle. Daughters inherit the parent's fixedness. Callers may change the attribute by stable cell identifier.

Mechanics constrains all seven correction degrees of freedom for fixed cells. With `P` denoting the diagonal projection that zeros fixed-cell degrees of freedom, the constrained system is

```text
(P A P + I - P) x = P b.
```

This preserves a symmetric positive-definite operator for conjugate gradient, makes the fixed solution entries exactly zero, and lets movable cells react to contacts with immovable cells. Mechanics integration independently ignores fixed-cell translation, rotation, and contact-induced length correction. An explicit desired growth increment still increases rod length.

Checkpoint schema v6 records `fixed` for every active cell. Readers migrate v1 through v5 cells to `fixed = false`, matching the behavior of files written before the attribute existed.

## Consequences

- One world-state and lifecycle model serves movable and fixed cells.
- Fixed cells remain active biological participants and immovable mechanical obstacles.
- Direct geometry mutation remains an explicit caller operation; fixedness is a mechanics constraint, not an access-control mechanism.
- The legacy point-volume state is not exactly restart-compatible with rod length. Import requires a declared model-specific volume-to-rod conversion and is outside the generic checkpoint migration.
