# ADR 0014: restartable native model controllers

- Status: accepted
- Date: 2026-08-15

## Context

The first native model contract returned a bare `Simulation`. That was enough for fixed growth, species, signal, and coupled-rate plans, but it could not own per-step regulation, division policy, mechanics scheduling, runtime randomness, or other evolving Python state.

Checkpoint files must remain non-executable data. Restoring Python behavior therefore requires an explicitly selected model source, a verified source digest, and a clear division between native state and controller state.

## Decision

`build(context)` may return either a native `Simulation` or any object that structurally implements `SimulationController`:

- `simulation` returns the native state owner;
- `step(dt)` advances exactly one biological step; and
- `controller_state()` returns complete finite, non-null JSON data for exact restart.

The batch runner, data-only run manifests, and live viewer all operate on that same structural contract. They checkpoint the controller payload alongside the native state without serializing code or live Python objects. Any object that satisfies the protocol conforms; no inheritance is required.

A controller-backed model resumes through `resume(context, checkpoint)`. The operator must explicitly provide both `--model` and `--resume`; MicroSimulator does not execute a path found in checkpoint provenance. Before compiling the model, the runner verifies its SHA-256 against the checkpoint, reconstructs the original seed and immutable parameters, and rejects a mismatched resume context. The returned controller must own the exact `checkpoint.simulation` instance so native state cannot be silently rebuilt or discarded. A bare native checkpoint with a null controller remains resumable with `--resume` alone.

`capture_random_state` and `restore_random_state` provide a closed, versioned JSON representation of a dedicated Python `random.Random` MT19937 stream, including its Gaussian cache. Native controllers use these helpers rather than ambient module randomness. Model-specific state remains model-defined JSON and must be validated by the model's resume entry point.

`NativeController` is the standard implementation of the protocol. A regulation callback receives immutable cell snapshots plus the controller's explicit RNG and mutable JSON model state, then returns a `StepPlan`. The plan contains typed `CellUpdate` and `DivisionRequest` records and is validated in full before native state changes. Optional division callbacks receive the parent and both native daughter snapshots. The fixed step order is:

1. compute and validate host regulation;
2. apply cell attributes, species, and fixed-state updates;
3. apply division requests and division callbacks;
4. apply requested cell removals;
5. execute `Simulation.step(dt, ...)` with the mechanics configuration, which advances growth and contact together with the [contact dynamics](0026-contact-dynamics.md) solve, carrying the cells with the flow when flow drift is enabled, and then advances the typed rate plans. Without a mechanics configuration the step is `Simulation.step(dt)`: growth and rate plans alone.

`NativeController.last_contact_dynamics` holds that step's report, the axial load on each cell, and the elongation each cell was allowed.

`UniformLengthDivision.jitter_z` controls only the random perturbation added to daughter orientation: `None` disables jitter, `False` adds XY-only jitter, and `True` adds XYZ jitter. Native normalization can change an inherited nonzero Z component even when the added Z perturbation is zero. Division places daughters along the parent's three-dimensional axis; the contact solve and flow drift remain three-dimensional. See the [planarity diagnostics and tutorial audit](../tutorials/planarity.md) for reproducible examples and the limits of finite-height confinement.

The standard payload records a stable model ID and version, completed-step counter, model JSON state, random stream, and every mechanics parameter. `NativeController.from_checkpoint` validates and restores that payload while the checkpoint's native state retains rate plans, signal grids, geometry, and lineage. Every mechanics parameter is an explicit part of the native-controller contract.

## Consequences

- Native models can compose runtime policy without coupling the engine to one modeling DSL or callback base class.
- Exact controller resume has one path through batch execution, manifests, and the live viewer.
- Checkpoints remain data-only and never grant authority to execute recorded source paths.
- A model source change invalidates existing controller resumes unless an explicit migration is provided.
- Additional typed orchestration helpers can be built on this protocol without changing the runner or checkpoint format.
