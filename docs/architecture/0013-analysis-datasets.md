# ADR 0013: versioned columnar analysis datasets

- Status: accepted
- Date: 2026-08-15

## Context

Legacy analysis reads executable pickle snapshots directly into ad hoc NumPy, NetworkX, Matplotlib, or ReportLab scripts. Geometry, species, contacts, simulation time, filesystem time, compact slots, and stable identities are mixed without a schema or provenance boundary. Signal grids are flattened into private implementation layouts.

MicroSimulator checkpoints are safe, exact restart artifacts, but JSON is not an efficient scan format for large ensembles or time series. Scene files are presentation projections and intentionally omit restart and equation state.

## Decision

MicroSimulator analysis exports are immutable dataset directories with this logical layout:

```text
run.dataset/
  manifest.json
  frames.parquet
  cells.parquet
  species.parquet
  contacts.parquet
  external_contacts.parquet
  signals.zarr/
```

Only tables present in the source and requested derivations are written. An export builds in a sibling temporary directory and publishes by rename. It fails if the destination exists unless replacement is explicitly requested. Input checkpoints are an explicit ordered list. A time-series export requires nondecreasing physical time; independent replicates are separate datasets with separate run IDs.

`manifest.json` is strict, versioned JSON. It records ordered source paths and SHA-256 values, checkpoint schema versions, model provenance, export options, contact parameters, backend/device identity used for reconstruction, table schemas, row counts, signal epochs, and dataset file digests. It contains no wall-clock timestamp, executable content, or environment-dependent absolute path unless the caller explicitly requests path provenance.

Parquet tables use an explicit Arrow schema and deterministic row ordering:

- `frames`: frame index, physical time, source digest, backend identity, cell, species, signal, and contact counts;
- `cells`: frame index, stable and parent IDs, slot, position, unit direction, cylinder length, radius, derived full capsule length, growth rate, signed cell type, and fixed state;
- `species`: frame index, stable cell ID, channel index, and level; and
- `contacts`: frame index plus every typed `CellContact` field. Multiple rows for parallel capsules are preserved. Unique neighbor edges are a dataframe grouping, not a lossy storage replacement.

External constraint contacts use a separate typed table because their endpoint and constraint-kind semantics differ from cell pairs.

Stable cell and constraint IDs are Arrow `uint64`; slots and channel indices are `uint32`; cell types are `int32`; engine state values are `float32`; and physical time is `float64`. Nullable parent IDs stay typed `uint64` rather than sentinel values. Derived quantities are named as derivations and never replace their source columns.

Signal grids use Zarr arrays with logical dimension order `(frame, channel, x, y, z)`. Coordinates, origin, spacing, boundary conditions, frame indices, and physical times are explicit metadata or coordinate arrays. A change in grid shape or geometry starts a new signal epoch rather than padding or silently resampling values. Chunking favors one frame and one channel per access unit; compression and physical chunk sizes are recorded in the manifest.

Parquet and Zarr are optional Python analysis dependencies. The native engine, batch runner, checkpoint reader, scene writer, and live viewer do not import them. Dataframe recipes use Polars lazy scans; Arrow remains the schema and Parquet interchange boundary. Zarr is used directly for multidimensional signals rather than storing opaque arrays inside table cells.

## Derived analyses

Documented recipes, not engine methods, provide:

- radial XY position and explicit-edge radial counts or species means;
- cylinder-length and full-capsule-length histograms;
- length-weighted XY line-density histograms;
- unique neighbor graphs and sister-neighbor counts from stable lineage IDs; and
- signal slices and time courses using named dimensions.

Recipes state weighting, bin edges, null handling, and units. They return data before plotting so statistical checks do not depend on a rendering library.

## Backend and conformance policy

Checkpoint state exports are backend-neutral because loading reconstructs the same authenticated state. Contact exports are derived computations: their manifest records backend, device, `ContactParameters`, and conformance status. CPU is the default geometry oracle, and Metal provides the feature-complete Apple GPU path. CUDA selection executes the native CUDA geometry implementation and participates in the shared C++ fixture and application-level derived-contact export used by the CUDA development gate. No exporter silently falls back to CPU.

## Experiment manifests

Replicate and parameter-sweep planning uses data-only run manifests containing one explicit run ID, seed, parameter map, source model digest, backend/device, stopping rule, and output path per job. Execution remains ordinary `microsimulator run` invocations under a caller-chosen local, CI, or cluster scheduler. The engine does not embed a second job scheduler.

Cell-count stopping is a first-class run condition rather than the legacy scripts' implicit relationship to preallocated capacity. A maximum step count is still required so a non-growing model terminates deterministically.

## Consequences

- Analysis is inspectable without importing model code or private engine objects.
- Large cell tables and signal volumes have formats suited to their access patterns.
- Stable identities, scientific meanings, and derivation provenance survive export.
- A schema migration is required for incompatible column or dimension changes.
- Presentation export can evolve independently of analysis storage.
- Parquet and Zarr dependencies remain optional and do not enter native backend builds.
