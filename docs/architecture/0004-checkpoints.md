# ADR 0004: versioned data-only checkpoints

- Status: accepted
- Date: 2026-08-15

## Context

A checkpoint built from Python pickle files of live objects and model source text executes arbitrary Python on load. Reconstructing the next cell identifier from the active cells loses the true allocation frontier when the largest allocated cells are no longer active, and a format whose saved fields vary with the configured integrator and signaling model cannot be validated against one schema.

MicroSimulator needs reproducible restart files that can cross CPU, Metal, and CUDA hosts without serializing device resources or executable model code.

## Decision

The public checkpoint is UTF-8 JSON with the format identifier `microsimulator-checkpoint` and format version 10. It records:

- simulation time;
- every active cell in compact slot order, including its stable identifier, geometry, attributes, whether it is fixed in mechanics, and fixed-schema species levels;
- the exact next cell identifier and complete parent map, including inactive ancestors;
- typed plane, sphere, box, and cylinder constraints and the exact next constraint identifier;
- the complete typed species-rate instruction plan and outputs;
- an optional signal grid: its specification, integration kind and iterative-solver parameters, optional spatial affine source/loss field, obstacle mask, optional face velocity field, and complete signal-major concentration field;
- an optional typed coupled cell/grid rate plan;
- an optional data-only controller payload, JSON `null` for a bare native simulation;
- `channel_metadata` with ordered `species` and `signals` arrays whose entries are strings or null and whose lengths match the native channel counts; and
- producer, source-backend, and caller-supplied provenance.

The simulation payload, the controller payload, and the channel metadata each carry a SHA-256 digest under `integrity`. A digest covers the payload's canonical JSON: RFC 8785 member order, string escaping, and number formatting, with an integer written as its exact decimal digits. Integers span the signed and unsigned 64-bit ranges. The digest is computed from the parsed values, so reformatting a file does not invalidate it.

The C++ core implements the format in `microsimulator/checkpoint_file.hpp`: encoding, schema and domain validation, digests, and file replacement. JSON text is parsed and written by JSON for Modern C++, in its indented form with one value per line, and digests are computed by PicoSHA2, two single-header libraries vendored under `cpp/third_party` and compiled into the core, so a program that links the core needs no additional header or library. The controller and provenance are opaque JSON that the core stores and authenticates without interpreting; Python supplies and consumes them. A non-null controller cannot be silently discarded by `load_checkpoint`; callers use `load_checkpoint_bundle` and restore it with the matching controller. `load_checkpoint_bundle` exposes channel labels without executing model code, and `load_checkpoint` refuses named metadata it would otherwise discard. See the [channel authoring guide](../models/channel-labels.md).

Native channel counts are not capped by the scene format's 4096-channel presentation budget; checkpoint metadata always holds explicit count-matched arrays.

Files are written to a temporary sibling, flushed, and atomically replaced. Loading rejects text that is not strict JSON, duplicate keys, non-finite numbers, unknown or missing fields, any format version other than 10, files over 1 GiB, digest mismatches, and any state that fails native domain validation. No module is imported and no source text, callback, pickle opcode, or other executable representation is accepted.

Device buffers, command queues, streams, compiled pipelines, contact graphs, and mechanics workspaces are deliberately excluded. They are derived caches and are reconstructed lazily by the selected backend. A checkpoint written by one backend may therefore be restored on another backend and device. The source backend name and device index remain provenance rather than restore instructions; callers choose the target explicitly.

## Exact-resume meaning

Immediately after restore, all persisted integers and IEEE-754 cell and time values match the saved host state exactly. Subsequent CPU execution is tested for exact continuation. Native GPU continuation is compared to the same CPU reference under each operation's declared numerical tolerance; cross-device bitwise equality is not promised.

The digest detects accidental corruption but is not an authenticity signature. Untrusted provenance remains untrusted data and must be escaped by downstream renderers. A reader never guesses the meaning of another schema: it accepts exactly the format version it implements.

## Consequences

- Checkpoints are portable, inspectable, and safe to parse as data.
- Allocation counters are first-class state rather than inferred metadata.
- Exact restart includes lineage, model equations, and authenticated controller data, not only visible cells.
- A schema change requires a new format version, and files of any other version are rejected.
- C++ programs save and restore checkpoints without Python.
