# Testing and validation

MicroSimulator tests observable scientific behavior at several levels, from individual numerical operations to complete application workflows. CPU is the readable numerical reference, Metal is the supported Apple GPU backend, and CUDA is the native NVIDIA backend under active development. A backend is supported only when its own implementation passes the same contracts without silently falling back to another backend.

## Validation layers

Backend-affecting work is checked in the following order:

1. State the governing equation and discrete algorithm.
2. Classify legacy behavior as authoritative, informative, or defective.
3. Add focused CPU unit tests.
4. Add a shared scenario with declared tolerances.
5. Implement the Metal and CUDA paths independently.
6. Run the shared scenario on real backend hardware.
7. Compose the feature into application and legacy-trajectory tests.
8. Add scaling, sanitizer, and long-run tests after correctness is established.

Passing compilation is necessary but not sufficient. GPU conformance always means execution on the corresponding hardware, and a complete application claim additionally requires the Python, checkpoint, legacy-model, viewer, and analysis workflows.

## Backend contract

Every test-enabled build runs the shared scenarios against every enumerated device compiled into that build. `backend_contract_conformance` requires each constructed device to advertise growth, species, contacts, mechanics, constraints, signals, and coupled rates. Capability guards in individual tests may help diagnose partial development builds, but they cannot turn a missing capability into a green complete-backend result.

`trajectory_conformance` composes coupled rates and transport, contact and constraint geometry, fixed-cell mechanics, integration, and division over three steps. It catches cross-feature errors that isolated one-step tests cannot expose. The exact scenarios, problem sizes, and numerical tolerances are maintained in the [conformance test reference](../../tests/conformance/README.md).

## CPU and Metal

CPU and Metal use both a native C++ test run and an application-level test run.

```console
scripts/run_metal_conformance.sh
CM_LEGACY_ROOT=/path/to/pinned/CellModeller scripts/run_metal_application_conformance.sh
```

The native gate performs a fresh Metal-enabled build, constructs every enumerated device, compiles every embedded MSL library, and runs the complete CTest suite. The application gate builds the Python extension with Metal enabled, runs the full Python and recorded-trajectory suites, and executes the complete legacy-example matrix on CPU and every Metal device.

Together, these gates cover:

- growth, equal and asymmetric division, stable identity, and lineage;
- cell contacts, plane and sphere constraints, fixed cells, and mechanics relaxation;
- species, signal transport, Forward Euler, Crank-Nicolson, and coupled rates;
- checkpoint migration, exact controller resume, and deterministic runtime random state;
- batch execution, stopping rules, output collision handling, and run manifests;
- scene capture, live-viewer reset and checkpoint behavior, and protocol validation;
- Parquet/Zarr analysis export, native contact derivation, and verified Polars recipes; and
- all 24 runnable rows of the pinned 25-example CellModeller compatibility matrix.

Each runner requires a clean source tree and writes a checksummed test record containing the exact commit, device and toolchain inventory, configure/build logs, JUnit results, and final status. The manual `Metal conformance` workflow runs the same commands on a self-hosted macOS runner carrying the `metal` label.

## Comparing with CellModeller

The application gate pins the original example sources at CellModeller commit `4896f543c6250f053eea2312e628cc3a96bf7408`. It authenticates 15 unchanged callback models and 9 typed equation migrations, advances all 24 runnable scenarios, and records `load.py` as the one migration-only row. Set `CM_LEGACY_ROOT` to that checkout to enable the suite.

Recorded trajectories independently compare five representative workflows with values produced by the original Apple OpenCL runtime:

- a growing 2D colony;
- a constrained 3D colony;
- a neighbor-dependent model;
- a species model; and
- a coupled signaling model.

These comparisons can reveal a semantic change shared by CPU and Metal that ordinary cross-backend tests would miss. The [legacy example matrix](../compatibility/legacy-example-matrix.md) and [recorded trajectories](../compatibility/legacy-trajectory-evidence.md) describe the coverage.

## CUDA

CUDA uses three distinct checks:

```console
scripts/run_cuda_compile_check.sh
scripts/run_cuda_conformance.sh
CM_LEGACY_ROOT=/path/to/pinned/CellModeller scripts/run_cuda_application_conformance.sh
```

The compile check mounts the source read-only in an ephemeral CUDA 12.8.1 development container, builds every native target, lists the registered tests, and records the image and compiler versions. It is a source/link portability check and does not require an NVIDIA driver or device.

The native conformance runner requires an NVIDIA GPU. `cuda_runtime_gate` prevents a CUDA-enabled build from succeeding through CPU-only execution, and the runner records the GPU inventory, compute capability, driver, toolkit, clean-build logs, JUnit output, result, and checksums.

The application runner builds the Python extension with CUDA enabled, requires a runtime device, runs the Python and recorded-trajectory suites, and executes the legacy-example matrix on CPU and every enumerated CUDA device. It also covers exact controller resume, live scene semantics, and analysis export with native cell and constraint contact derivation.

The hosted `CUDA compile check` workflow runs for pull requests and `main`. Hardware and application checks are manually dispatched to a self-hosted Linux x64 runner carrying the `gpu` label so untrusted pull-request code is not executed automatically on a persistent GPU host. CUDA support across an NVIDIA architecture requires all three checks on the release commit.

## What the tests cover

The Python and C++ suites exercise more than numerical parity:

- Species fixtures cover schema validation, initialization, inheritance, plan topology, effective-volume dilution, simultaneous updates, and all typed instruction operations.
- SBML fixtures cover supported Level 3 Version 2 Core symbols and operators, ordered metadata, local and global parameters, stoichiometry, and explicit rejection of unsupported semantics.
- Signal fixtures cover mass conservation, fixed reservoirs, periodic upwind advection, reduced dimensions, 3D interpolation, nonuniform affine sources and losses, stability rejection, checkpoint migration, and diagnosed solver convergence.
- Coupled fixtures cover old-field sampling, post-growth intracellular rates, amount-to-voxel conversion, repeated scatter destinations, empty colonies, and failure atomicity.
- Contact and mechanics fixtures cover sparse and dense candidate staging, coincident cells, stable ordering after slot reuse, operator properties, convergence and breakdown diagnostics, buffer growth, fixed-cell projection, and integrated geometry.
- Checkpoint fixtures compare every persisted field before continuing execution and reject corrupt, malformed, non-finite, duplicate, unknown, or unsupported data.
- Batch and manifest fixtures cover seed reproducibility, model digests, periodic output names, stopping rules, path resolution, collision preflight, and import-before-authentication hazards.
- Scene and viewer fixtures cover strict cross-language decoding, RFC 8785 digests, browser-safe identities, coloring, signal slicing, token and origin checks, WebSocket messages, reset, and checkpointing.
- Analysis fixtures read generated Parquet and Zarr artifacts back, verify schemas and provenance, and test geometry, species, contact, lineage, and signal recipes at boundary cases.

## Backend support policy

Pull requests must not claim backend support when an implementation invokes the CPU reference or transfers full state to the host to complete a device operation. Every hardware runner must use a clean build and retain its machine-readable results. A backend-affecting change invalidates evidence from an earlier commit and requires the corresponding gate to run again.
