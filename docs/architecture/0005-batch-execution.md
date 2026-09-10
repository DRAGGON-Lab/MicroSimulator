# ADR 0005: deterministic batch execution

- Status: accepted
- Date: 2026-08-15

## Context

MicroSimulator needs one reproducible, non-interactive path for local runs, cluster jobs, and backend tests. Legacy batch scripts assemble simulator state through ambient module globals and mix model construction, output policy, and execution. They do not give every run an explicit backend, device, seed, or collision policy.

Checkpoint loading must also remain distinct from Python model execution. A checkpoint is portable data and safe to parse; a model file is trusted Python code chosen explicitly by the operator.

## Decision

The batch-facing `microsimulator` operations are:

- `microsimulator devices` enumerates native CPU, Metal, and CUDA devices; and
- `microsimulator run` either executes a Python file supplied with `--model` or restores a data-only checkpoint supplied with `--resume`; and
- `microsimulator run-manifest` executes exactly one named job from a strict data-only experiment manifest.

A Python model exports `build(context)` and returns either a `Simulation` created by `context.simulation()` or a structural `SimulationController` that owns one. The context owns the selected backend and device, an unsigned 64-bit seed, a dedicated Python pseudorandom generator, and immutable JSON parameters. The runner rejects native state on another backend or device. Controllers own per-step policy and return complete data-only state for checkpoints. They resume through an explicit `resume(context, checkpoint)` entry point after model-digest verification and must reuse the restored native simulation. Models remain ordinary Python and may define typed helper layers without coupling the engine to a particular modeling DSL. See ADR 0014.

The run loop uses an integer maximum step count and an explicit finite non-negative time step. An optional positive unsigned cell-count threshold is checked before the first step and after every completed step. The first reached condition ends the run; a maximum step count remains mandatory so a non-growing model terminates deterministically. Cell-count stopping depends on the active-cell count, never reserved capacity.

Before simulation begins, the runner checks the final and every potentially written periodic checkpoint path for collisions. It does not overwrite existing data unless `--overwrite` is present. Every write uses the checkpoint layer's atomic-replace behavior. Periodic filenames contain an eight-digit completed step number, and the final checkpoint is written even when the initial cell count causes a zero-step run. A run summary reports only periodic checkpoints actually written before termination.

Run provenance records the model or resume input's absolute path and SHA-256 digest. Model provenance also records the seed and JSON parameters. Each checkpoint records the maximum and completed steps, time step, complete stopping rule, and current or final stop reason. Randomness during model construction is reproducible when the model uses `context.rng`. Runtime Python controllers persist dedicated streams with the versioned random-state helpers.

A run manifest contains an explicit ID, model path and SHA-256, backend, device, seed, JSON parameter map, maximum steps, time step, optional cell-count threshold, checkpoint interval, and output path for every job. Relative paths resolve from the manifest directory. IDs and all potential final/periodic outputs are disjoint. Parsing is closed-schema JSON and does not load model code. When a selected job executes, its exact model bytes are checked against the declared digest before compilation. The manifest path, file digest, and job ID enter checkpoint provenance.

One invocation executes one job. Parallelism, retries, resource requests, and job placement remain responsibilities of a caller-chosen scheduler rather than a second scheduler embedded in MicroSimulator.

## Consequences

- Batch jobs have one inspectable entry point and stable exit behavior.
- Backend and device selection cannot be silently overridden by a model.
- Identical source, seed, parameters, backend, and numeric environment provide a reproducible construction and execution contract.
- Model files are trusted executable code and must not be confused with safe checkpoint input.
- Scheduling systems can treat periodic checkpoints as independent restart points without parsing process logs.
- Experiment plans are reviewable data, while model execution remains an explicit trusted-code action with a pre-execution digest check.
