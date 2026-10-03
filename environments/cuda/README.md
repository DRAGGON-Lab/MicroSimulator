# CUDA environment

CUDA is the NVIDIA backend under active development. It is implemented directly in CUDA C++ with the CUDA Runtime API: no portability layer, translated Metal source, or CPU computational fallback is used.

For application commands, use the [shared CPU, Metal, and CUDA tutorial examples](../../docs/tutorials/commands.md#run-the-same-trap-on-cpu-metal-or-cuda), [device discovery](../../docs/tutorials/commands.md#prepare-and-discover-devices), and [shell quoting guide](../../docs/tutorials/commands.md#choose-a-shell). Select this backend with `--backend cuda` and an enumerated `--device-index`; the model syntax is unchanged.

`CM_ENABLE_CUDA` is off by default so ordinary CPU builds do not acquire a CUDA toolchain dependency.

## Compile check

Run the driverless source and link gate in NVIDIA's CUDA 12.8.1 development container:

```console
scripts/run_cuda_compile_check.sh
```

The script mounts the source read-only, builds all CUDA-enabled native targets, lists the registered tests without executing them, and writes checksummed compiler, image, configure, and build evidence. Override `CM_CUDA_ARCHITECTURES` or `CM_CUDA_CONTAINER_IMAGE` to test another target or toolkit image.

The `CUDA compile check` workflow runs this gate for pull requests, pushes to `main`, and manual dispatches on GitHub's hosted Ubuntu runner. Compilation establishes toolchain compatibility, not numerical or application behavior.

## Native hardware conformance

On a Linux host with the CUDA Toolkit and an NVIDIA GPU, run:

```console
scripts/run_cuda_conformance.sh
```

Every CUDA-enabled test build includes `cuda_runtime_gate`, which fails if the runtime cannot enumerate and construct a native CUDA backend. The conformance runner requires a clean worktree, performs a fresh configure and clean rebuild, runs the complete CTest suite on every enumerated device, and writes a timestamped evidence directory under `build/`. The evidence records the exact commit, device inventory, compute capability, driver and toolkit, logs, JUnit results, final status, and `SHA256SUMS`.

Pass a path as the script's sole argument to select a different new evidence directory.

## Application conformance

Run the Python, compatibility, and application gate with a checkout of the pinned original CellModeller source:

```console
CM_LEGACY_ROOT=/path/to/pinned/CellModeller scripts/run_cuda_application_conformance.sh
```

This gate builds the Python extension with CUDA enabled, runs the full Python and recorded-trajectory suites, executes all 24 runnable legacy examples on CPU and every CUDA device, and exercises controller resume, viewer scene semantics, and analysis export with native derived contacts.

The manually dispatched `CUDA conformance` workflow runs both hardware gates on a self-hosted runner with the `self-hosted`, `linux`, `x64`, and `gpu` labels. It always uploads evidence, including failed results, and is intentionally not triggered by pull requests. Keep the runner compatible with the workflow's pinned action versions.

See the [validation workflow](../../docs/development/validation.md) for the complete acceptance policy.
