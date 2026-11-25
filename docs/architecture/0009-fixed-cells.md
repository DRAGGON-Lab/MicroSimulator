# ADR 0009: persistent fixed rod cells

- Status: accepted
- Date: 2026-08-15

## Context

Holding cells in place with a separate point-volume biophysics model, one that stores cell centers and scalar volumes, advances volume with explicit Euler growth, and never changes position, provides no per-cell fixed flag and bypasses the capsule geometry and mechanics operator of the rod model.

MicroSimulator needs the useful behavior without introducing a second, incompatible geometry representation. Fixedness also needs to survive division and checkpoint restore and to behave identically on every backend.

## Decision

Fixedness is a persistent Boolean cell attribute. A fixed cell remains a rod: growth, species rates, signal sampling and secretion, and division continue to use the ordinary cell lifecycle. Daughters inherit the parent's fixedness. Callers may change the attribute by stable cell identifier.

A fixed cell has no mobility in the [contact dynamics](0026-contact-dynamics.md) solve: contact forces do not move or turn it, and the flow does not carry it. Movable cells react to contacts with it as they do with a wall. It still elongates, at its growth rate scaled by the growth load response to the axial load its contacts put on it, and that elongation closes the gaps of its contacts.

Checkpoints record `fixed` for every active cell.

## Consequences

- One world-state and lifecycle model serves movable and fixed cells.
- Fixed cells remain active biological participants and immovable mechanical obstacles.
- Direct geometry mutation remains an explicit caller operation; fixedness is a mechanics constraint, not an access-control mechanism.
