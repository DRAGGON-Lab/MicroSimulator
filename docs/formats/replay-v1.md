# Replay bundle format v1

A replay bundle is a directory containing `manifest.json` and independent scene files. It contains presentation data; it cannot restart a model or execute simulation steps. `microsimulator export-replay CHECKPOINT... --output DIRECTORY` exports an explicitly ordered sequence using CPU checkpoint deserialization. It never imports the original model, invokes controller callbacks, or requires the source run's GPU.

## Manifest

The manifest envelope has exactly `format`, `version`, `integrity`, and `recording` fields. Format is `microsimulator-replay`, version is `1`, and `integrity` has `algorithm: "sha256"` and `recording`, the lowercase SHA-256 digest of the RFC 8785 canonical representation of the complete recording object. This detects corruption; it is not proof of publisher authenticity.

`recording` contains exactly:

- `export_backend`: backend identity of the CPU used to deserialize the portable checkpoints. This has the same `kind`, `name`, `device`, `device_index`, and `native` fields as a scene backend.
- `frames`: a nonempty ordered array of frame entries.

Each entry contains exactly:

| Field | Meaning |
| --- | --- |
| `ordinal` | Zero-based contiguous ordinal equal to the entry's array index |
| `time` | Finite nonnegative recorded simulation time |
| `file` | Safe relative path to a `.scene.json` document |
| `bytes` | Exact positive UTF-8 file byte length |
| `sha256` | Lowercase SHA-256 of the exact scene file bytes, including whitespace |
| `checkpoint_sha256` | Lowercase SHA-256 of the exact source checkpoint bytes consumed |
| `source_backend` | Backend identity recorded by that checkpoint's producer |

The entry order is authoritative. Paths and checkpoint names are never sorted. Times must be nondecreasing. Equal times remain distinct frames and receive distinct ordinals; they are useful for topology events or separate observations at the same physical time. The exporter does not interpolate, merge, or drop snapshots. Supply checkpoints from the same run when stable cell identity across frames is required; the exporter cannot infer common ancestry from arbitrary checkpoint provenance.

Paths may use ASCII letters, digits, underscores, hyphens and dots within nonempty segments. Segments start with a letter, digit, underscore or hyphen. Absolute paths, dot segments, backslashes, URI schemes and percent-encoded escapes are rejected. References must be unique. References identify files selected from the bundle folder; the browser never fetches URLs from a manifest.

Readers verify the recording digest, strict schema, ordinals, timestamps and references before opening a frame. On demand, they verify the file size and exact-file digest, then use the shared scene reader to verify the scene's own digest/schema. The scene time and source backend must match the entry. Scene version 2 or 3 is supported through that reader; current exports use version 3 and preserve channel metadata.

The scene's backend describes the source run, so the viewer does not present the exporter's CPU as the simulation device. The exporter identity is retained separately. Checkpoint source-backend values are provenance, not a request to allocate that device. Source paths are omitted; `checkpoint_sha256` identifies the source bytes without leaking machine-local paths. Each checkpoint is copied into a temporary snapshot before parsing, so a producer replacing its original path cannot make the parsed bytes differ from the recorded digest.

## Limits and failure behavior

Manifests are limited to 16 MiB and 100,000 entries. Each scene is bounded by the scene format's 1 GiB encoded limit. The exporter requires a new destination; it refuses existing files, folders and symlinks. It builds a temporary sibling and publishes the completed bundle only after all frames pass validation. Failures identify the zero-based ordinal and source checkpoint, remove the temporary export, and leave source checkpoints untouched.

The viewer retains file handles and manifest metadata, but reads scene payloads only on demand. Its LRU cache holds at most three decoded frames and at most 64 MiB of conservative decoded-size accounting units. Objects too large for that budget are displayed without entering the cache. This is not a 64 MiB process-heap limit: the current displayed frame, renderer/GPU buffers, manifest, file handles and one active load/parse can exist outside the cache. Cache accounting includes cell arrays, species, signal levels, boundary values, constraints and labels.

A single worker performs frame loading/decoding; repeated seeks replace one pending ordinal rather than creating a queue of decodes. Superseded reads are canceled where possible. Both successful and failed stale requests are ignored. Opening a different dataset cancels the previous reader and prevents its completion from changing the view. Failures leave the last successfully displayed frame in place, identify the affected ordinal/file, pause playback and allow seeking to another frame.

## Playback and presentation

Playback displays every recorded frame without interpolation. Configurable 1–120 frames/s defines a maximum presentation cadence; slow loading reduces achieved speed instead of skipping frames. Recorded simulation time remains visible independently of playback speed, including in the transport bar at narrow supported window widths. Playback stops at the last frame; pressing Play there restarts from the first. Manual seeking and previous/next stepping pause automatic playback.

Opening a recording begins one viewer dataset. Subsequent frames, backward steps and seeks use the shared presentation-update path. Camera pose, reference-grid geometry and index-based channel preferences persist. Selected cells follow stable IDs across slot changes; selection clears if the ID is absent. Missing signal grids temporarily hide the controls, and smaller grids clamp displayed indices while retaining preferences for later compatible frames. Opening another recording or a static scene starts a new dataset.
