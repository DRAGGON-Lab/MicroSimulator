# ADR 0020: cell removal

- Status: accepted
- Date: 2026-08-16

## Context

Cells leave real systems: washout past a trap mouth, lysis, and ablation. The world state
only ever grows; a model cannot express a colony that loses members, so a flow-fed trap
accumulates cells that physically would be carried away.

## Decision

`WorldState::remove_cell` removes one active cell by stable identifier. The removed cell's
slot is backfilled by moving the last slot's arrays into it, so cell storage stays compact and
kernels remain dense; the moved cell keeps its stable identifier and only its slot changes.
Removing an unknown identifier is an error. Identifiers are never reused: `next_id` is
untouched and the lineage map keeps every recorded child-parent entry, so ancestry queries
about removed cells still resolve and checkpoints of their descendants stay traceable.
`Simulation.remove_cell` exposes the operation, and contact graphs computed before a removal
follow the existing rule that graphs are recomputed after any topology change.

The controller's `StepPlan` gains a `removals` tuple of cell identifiers, applied after
updates and divisions and before the native step. Plan validation rejects unknown
identifiers, duplicates, and identifiers that also appear as division parents. Snapshots seen
by `regulate` predate the removals, so a model computes removals from the same view it uses
for everything else. `UniformLengthDivision.forget` drops division-target state for removed
cells; a model calls it before returning the plan.

Checkpoints need no schema change: cells are recorded by identifier with compact slots, and
lineage validation already admits entries whose cells are no longer active.

## Validation sequence

1. World-state unit tests for slot backfill, identifier stability of the moved cell, species
   block movement, lineage retention, and unknown-identifier rejection.
2. Checkpoint round trip after removals.
3. Controller plan validation and application ordering.

## Consequences

- Washout, death, and ablation are expressible; a trap model can shed cells its flow would
  carry away.
- Slot indices are unstable across removals, as they already are across division; stable
  identifiers remain the only persistent handle.
- Backends see removals as ordinary state changes; no kernel is aware of them.
