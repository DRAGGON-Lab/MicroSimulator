# MicroSimulator documentation

MicroSimulator models microbial populations in microfluidic environments, connecting device geometry and flow to solute transport, cell growth, and signaling. These guides take you from a device description and biological rules to runnable experiments, visualization, and quantitative analysis. The modeling interface also supports colonies and multicellular systems without a device.

## Start here

| If you want to… | Read… |
| --- | --- |
| Understand how devices, flow, and cells fit together | [Microfluidics modeling guide](microfluidics.md) |
| Run a first simulation | [Getting started](tutorials/getting-started.md) |
| Build a trap or channel with growth and washout | [Microfluidic devices](tutorials/microfluidics.md) |
| Choose a flow solver and assess its numerical behavior | [Flow models](microfluidics.md#choosing-a-flow-model) and [flow benchmarks](tutorials/flow-solvers.md#numerical-evidence) |
| Measure nutrient penetration and growth | [Controlled nutrient study](tutorials/nutrient-validation.md) |
| Learn the modeling interface | [Tutorials](tutorials/README.md) |
| Understand numerical conventions | [Numerical contract](architecture/numerical-contract.md) |
| Analyze simulation output | [Analysis recipes](analysis/recipes.md) |
| Configure an accelerator | [Execution environments](#execution-environments) |
| Migrate a CellModeller model or snapshot | [Compatibility and migration](compatibility/README.md) |
| Test a contribution or backend | [Testing and validation](development/validation.md) |

## Tutorials

Start with [getting started](tutorials/getting-started.md) to install the tools, run a model, inspect it, and resume a checkpoint. Then follow the [tutorial index](tutorials/README.md) by topic. Each runnable model is self-contained.

| Topic | Guides |
| --- | --- |
| Devices, flow, and transport | [Walls, flow, and washout](tutorials/microfluidics.md); [pillar channels and flow solvers](tutorials/flow-solvers.md); [nutrient penetration and growth](tutorials/nutrient-validation.md) |
| Cell biology | [Growth and mechanics](tutorials/biophysics-and-growth.md); [gene circuits](tutorials/intracellular-dynamics.md); [signaling](tutorials/signaling.md); [plasmids and contacts](tutorials/discrete-state-and-contacts.md) |
| Circuits in populations | [SimBOL examples](tutorials/simbol.md); [Danino clock in a device](../examples/tutorials/danino_clock.py) |
| Quantitative output | [Checkpoints, contact graphs, and analysis](tutorials/analysis.md) |

Executable teaching models are under [`examples/tutorials`](../examples/tutorials). Smaller focused examples are available in [`examples`](../examples).

## Architecture and numerics

The [architecture guide](architecture/README.md) introduces the engine design and groups the architecture decision records by subject. The [numerical contract](architecture/numerical-contract.md) defines precision, tolerances, time integration, determinism, ordering, and failure behavior across backends.

Start with these documents when extending the engine:

- [Shallow device flow](architecture/0022-brinkman-flow.md) and [resolved Stokes-Brinkman flow](architecture/0023-mac-stokes.md)
- [Flow-driven cell motion](architecture/0021-flow-drift.md)
- [Biomass, growth, and uptake](architecture/0024-biomass-accounting.md)
- [Independent native backends](architecture/0001-native-backends.md)
- [Contact mechanics](architecture/0002-contact-mechanics.md)
- [Grid signaling and cell coupling](architecture/0006-grid-signaling.md)
- [Data-only checkpoints](architecture/0004-checkpoints.md)
- [Restartable model controllers](architecture/0014-native-controllers.md)

## Analysis and visualization

- [Analysis recipes](analysis/recipes.md) covers lazy Polars workflows for colony geometry, species, lineage, contact graphs, and signal fields.
- [Species and signal labels](models/channel-labels.md) describes native and SBML channel metadata.
- [Viewer guide](../viewer/README.md) covers static scenes, interactive sessions, controls, development, and tests.
- [Scene format v3](formats/scene-v3.md) defines the data exchanged with visualization clients.
- [Live viewer protocol v1](protocols/live-viewer-v1.md) defines the authenticated loopback protocol for interactive sessions.

## Execution environments

The CPU backend uses the base C++23/Python toolchain described in the top-level [quick start](../README.md#quick-start). Accelerator-specific setup is documented separately:

- [Apple Metal](../environments/metal/README.md)
- [NVIDIA CUDA](../environments/cuda/README.md)
- [Legacy OpenCL trajectory environment](../environments/legacy-opencl/README.md)

The [testing and validation guide](development/validation.md) distinguishes compile checks, native numerical tests, and full application tests. Accelerator support requires execution on corresponding hardware with fallback disabled.

## Formats and protocols

- [Run manifest v1](formats/run-manifest-v1.md) defines reproducible batch jobs and parameter sweeps.
- [Scene format v3](formats/scene-v3.md) defines data-only visualization frames.
- [Live viewer protocol v1](protocols/live-viewer-v1.md) defines interactive viewer messages and authority boundaries.
- [Checkpoint design](architecture/0004-checkpoints.md) defines restart state and schema migration.
- [Analysis dataset design](architecture/0013-analysis-datasets.md) defines Parquet/Zarr schemas and provenance.

## CellModeller compatibility

MicroSimulator originated as a CellModeller rewrite and now has an independent device, flow, and transport modeling workflow. The [compatibility and migration guide](compatibility/README.md) explains which CellModeller models and artifacts can be used directly, which require a typed translation, and where behavior intentionally differs. Source-pinned matrices and subsystem comparisons preserve the evidence behind those migration decisions.

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

Formatting, linting, and the Python type check run as commit hooks:

```console
uv run --with pre-commit pre-commit install
uv run --with pre-commit pre-commit run --all-files
```

The hooks cover the fast local gates only. Tests, native builds, and backend conformance need a configured build and hardware, and run through CTest and the [conformance scripts](development/validation.md).
