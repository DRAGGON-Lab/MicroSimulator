# MicroSimulator documentation

These guides explain how to build, run, inspect, and analyze MicroSimulator models. Design decisions and historical CellModeller comparisons are available as reference material, but the main documentation is organized around the work researchers and developers perform.

## Start here

| If you want to… | Read… |
| --- | --- |
| Run a first simulation | [Getting started](tutorials/getting-started.md) |
| Learn the modeling interface | [Tutorials](tutorials/README.md) |
| Understand numerical conventions | [Numerical contract](architecture/numerical-contract.md) |
| Analyze simulation output | [Analysis recipes](analysis/recipes.md) |
| Configure an accelerator | [Execution environments](#execution-environments) |
| Migrate a CellModeller model or snapshot | [Compatibility and migration](compatibility/README.md) |
| Test a contribution or backend | [Testing and validation](development/validation.md) |

## Tutorials

The tutorials form an ordered introduction, but each runnable model is self-contained:

1. [Run, inspect, and resume a model](tutorials/getting-started.md)
2. [Growth, division, cell types, and constraints](tutorials/biophysics-and-growth.md)
3. [Intracellular species and gene circuits](tutorials/intracellular-dynamics.md)
4. [Diffusible signals and cell-cell communication](tutorials/signaling.md)
5. [Plasmid segregation, contacts, and conjugation](tutorials/discrete-state-and-contacts.md)
6. [Checkpoints, contact graphs, and quantitative analysis](tutorials/analysis.md)
7. [SimBOL circuit examples](tutorials/simbol.md)

Executable teaching models are under [`examples/tutorials`](../examples/tutorials). Smaller focused examples are available in [`examples`](../examples).

## Architecture and numerics

The [architecture guide](architecture/README.md) introduces the engine design and groups the architecture decision records by subject. The [numerical contract](architecture/numerical-contract.md) defines precision, tolerances, time integration, determinism, ordering, and failure behavior across backends.

Start with these documents when extending the engine:

- [Independent native backends](architecture/0001-native-backends.md)
- [Contact mechanics](architecture/0002-contact-mechanics.md)
- [Grid signaling and cell coupling](architecture/0006-grid-signaling.md)
- [Data-only checkpoints](architecture/0004-checkpoints.md)
- [Restartable model controllers](architecture/0014-native-controllers.md)

## Analysis and visualization

- [Analysis recipes](analysis/recipes.md) covers lazy Polars workflows for colony geometry, species, lineage, contact graphs, and signal fields.
- [Viewer guide](../viewer/README.md) covers static scenes, interactive sessions, controls, development, and tests.
- [Scene format v1](formats/scene-v1.md) defines the data exchanged with visualization clients.
- [Live viewer protocol v1](protocols/live-viewer-v1.md) defines the authenticated loopback protocol for interactive sessions.

## Execution environments

The CPU backend uses the base C++23/Python toolchain described in the top-level [quick start](../README.md#quick-start). Accelerator-specific setup is documented separately:

- [Apple Metal](../environments/metal/README.md)
- [NVIDIA CUDA](../environments/cuda/README.md)
- [Legacy OpenCL trajectory environment](../environments/legacy-opencl/README.md)

The [testing and validation guide](development/validation.md) distinguishes compile checks, native numerical tests, and full application tests. Accelerator support requires execution on corresponding hardware with fallback disabled.

## Formats and protocols

- [Run manifest v1](formats/run-manifest-v1.md) defines reproducible batch jobs and parameter sweeps.
- [Scene format v1](formats/scene-v1.md) defines data-only visualization frames.
- [Live viewer protocol v1](protocols/live-viewer-v1.md) defines interactive viewer messages and authority boundaries.
- [Checkpoint design](architecture/0004-checkpoints.md) defines restart state and schema migration.
- [Analysis dataset design](architecture/0013-analysis-datasets.md) defines Parquet/Zarr schemas and provenance.

## CellModeller compatibility

The [compatibility and migration guide](compatibility/README.md) explains which CellModeller models and artifacts can be used directly, which require a typed translation, and where behavior intentionally differs. Source-pinned matrices and subsystem comparisons are kept there as supporting reference material rather than mixed into the main tutorials.

## Development and validation

The [testing and validation guide](development/validation.md) describes the test layers, backend requirements, hardware runners, and release checks. The [backend conformance reference](../tests/conformance/README.md) lists the shared numerical scenarios and tolerances.

For a standard CPU development build:

```console
uv sync --group dev
uv run pytest
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```
