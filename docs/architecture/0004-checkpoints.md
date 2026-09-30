# ADR 0004: versioned data-only checkpoints

- Status: accepted
- Date: 2026-08-15

## Context

Legacy CellModeller writes Python pickle files containing live `CellState` objects, model source text, lineage, selected integrator arrays, and a step number. Loading a pickle can execute arbitrary Python. The legacy resume path also reconstructs the next cell identifier from active cells, which loses the true allocation frontier when the largest allocated cells are no longer active. The saved fields vary with the configured integrator and signaling model.

MicroSimulator needs reproducible restart files that can cross CPU, Metal, and CUDA hosts without serializing device resources or executable model code.

## Decision

The public checkpoint is UTF-8 JSON with the format identifier `microsimulator-checkpoint` and an integer schema version. Writers emit version 11, which records:

- simulation time;
- every active cell in compact slot order, including its stable identifier, geometry, attributes, and fixed-schema species levels;
- the exact next cell identifier and complete parent map, including inactive ancestors;
- typed plane, sphere, box, and cylinder constraints and the exact next constraint identifier;
- the complete typed species-rate instruction plan and outputs;
- producer, source-backend, and caller-supplied provenance; and
- a SHA-256 digest of the canonical simulation payload.

The payload also includes an optional signal grid with concentration fields, boundaries, affine reactions, integration kind, and solver parameters; the typed coupled-rate plan; optional controller state; and channel metadata. A non-null controller cannot be silently discarded by `load_checkpoint`; callers use `load_checkpoint_bundle` and restore it with the matching controller. Channel metadata contains ordered `species` and `signals` arrays of strings or null, with lengths matching the native channel counts. Controller state and channel metadata carry separate SHA-256 digests over canonical JSON. `load_checkpoint_bundle` exposes labels without executing model code; `load_checkpoint` refuses named metadata it would otherwise discard. See the [channel authoring guide](../models/channel-labels.md).

The optional `culture` field stores fluid, chemical, and biological configuration, persistent binary64 cell poses, biochemical volumes, intracellular amounts, extracellular fragment amounts, cumulative reservoir transfers, and diagnostics. Its closed data schema names only predefined native records, arrays, and enums. Restore reconstructs geometry and validates the fragment count and agreement between authoritative amounts and the float32 cell projection. The selected backend recreates numerical resources. Culture solutes determine the signal-channel count even though this mode does not own a legacy signal grid.

Readers accept versions 1 through 11 and authenticate each original payload before applying explicit defaults for missing fields. Version 10 is migrated from `media_flow` to `culture` with explicit reserve-growth semantics; earlier formats restore without culture state. For versions predating channel metadata, absent groups remain compact as `ChannelMetadata(species=None, signals=None)` until native restoration validates counts. Scene export resolves labels only after enforcing its separate 4096-channel presentation budget per group. Native counts are not capped by that scene budget.

Files are written to a temporary sibling, flushed, and atomically replaced. Loading rejects duplicate JSON keys, non-finite numbers, unknown fields for the declared version, unsupported versions, oversized files, digest mismatches, and any state that fails native domain validation. No module is imported and no source text, callback, pickle opcode, or other executable representation is accepted.

Device buffers, command queues, streams, compiled pipelines, contact graphs, and mechanics workspaces are deliberately excluded. They are derived caches and are reconstructed lazily by the selected backend. A checkpoint written by one backend may therefore be restored on another backend and device. The source backend name and device index remain provenance rather than restore instructions; callers choose the target explicitly.

## Exact-resume meaning

Immediately after restore, all persisted integers and IEEE-754 cell and time values match the saved host state exactly. Subsequent CPU execution is tested for exact continuation. Native GPU continuation is compared to the same CPU reference under each operation's declared numerical tolerance; cross-device bitwise equality is not promised.

The digest detects accidental corruption but is not an authenticity signature. Untrusted provenance remains untrusted data and must be escaped by downstream renderers. Schema migrations are explicit version-to-version data transforms; a reader never guesses the meaning of an unknown schema.

## Consequences

- Checkpoints are portable, inspectable, and safe to parse as data.
- Allocation counters are first-class state rather than inferred metadata.
- Exact restart includes lineage, model equations, and authenticated controller data, not only visible cells.
- Schema evolution requires a new version and a tested migration path.
- Legacy pickle ingestion, if added, must be a separate one-way conversion tool run under an explicitly untrusted-code policy.
