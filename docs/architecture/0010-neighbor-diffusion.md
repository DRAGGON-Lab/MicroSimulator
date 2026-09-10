# ADR 0010: retire legacy neighbor diffusion

- Status: accepted
- Date: 2026-08-15

## Context

CellModeller contains a single `NeighbourDiffusion.py` module that suggests cell-to-cell diffusion over a neighbor graph. It is not referenced by the bundled models and cannot execute in the maintained code path. The source behavior is described in `docs/compatibility/legacy-neighbor-diffusion-audit.md`.

The visible loop resembles an unweighted graph Laplacian, but its comment says the flux should be scaled by wall or contact area. The code therefore does not establish the physical quantity represented by a level, the units of its coefficient, the contact measure, or the intended behavior as a contact appears and disappears.

## Decision

MicroSimulator does not provide `NeighbourDiffusion` as a compatibility feature. No CPU, Metal, or CUDA backend advertises it, and its absence is not a gap in native backend parity.

A cell-contact transport model must independently define:

- whether state is amount, concentration, surface density, or another typed quantity;
- a symmetric pairwise flux and the contact area or conductance used to weight it;
- conservation and non-negativity requirements;
- topology timing relative to growth, mechanics, division, and reactions;
- stability limits or an implicit solver with convergence diagnostics; and
- inheritance and checkpoint semantics when the cell graph changes.

The implementation must use the engine's typed `ContactGraph` and the same fixture on CPU, Metal, and CUDA. It must not reproduce the legacy dependence on mutable Python neighbor lists.

## Consequences

- Dead code does not create an invented compatibility promise.
- Grid signaling remains the supported extracellular transport model.
- Intracellular species remain cell-local unless a separately specified contact-transport stage is added.
- A scientifically defined graph transport model can still be added without preserving the legacy module's names or storage layout.
