# CUDA environment

CUDA is the NVIDIA backend under active development. It is implemented directly in CUDA C++ with the CUDA Runtime API: no portability layer, translated Metal source, or CPU computational fallback is used.

For application commands, use the [shared CPU, Metal, and CUDA tutorial examples](../../docs/tutorials/commands.md#run-the-same-trap-on-cpu-metal-or-cuda), [device discovery](../../docs/tutorials/commands.md#prepare-and-discover-devices), and [shell quoting guide](../../docs/tutorials/commands.md#choose-a-shell). Select this backend with `--backend cuda` and an enumerated `--device-index`; the model syntax is unchanged.

`CM_ENABLE_CUDA` is off by default so ordinary CPU builds do not acquire a CUDA toolchain dependency.

## Windows development build

Install Visual Studio 2022 C++ Build Tools and the CUDA Toolkit, then open a fresh PowerShell session so `CUDA_PATH` is available. Set `CMAKE_CUDA_ARCHITECTURES` to the GPU's compute capability without the decimal point (for example, `120` for a compute capability 12.0 device):

```powershell
cmake -S . -B build/cuda-windows -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=120 -DCM_ENABLE_CUDA=ON -DCM_ENABLE_METAL=OFF -DCM_BUILD_TESTS=ON -DCM_BUILD_PYTHON=OFF
cmake --build build/cuda-windows --config Release --parallel 2
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
ctest --test-dir build/cuda-windows -C Release --output-on-failure
```

For the Python CLI, rebuild the package with `CMAKE_ARGS` containing `-DCM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`, then run `microsimulator devices`. Python loads the CUDA runtime from `CUDA_PATH\bin` when importing the extension. The Linux hardware and application gates below remain the release acceptance checks.

## Compile check

Run the driverless source and link gate in NVIDIA's CUDA 12.8.1 development container:

```console
scripts/run_cuda_compile_check.sh
```

The script mounts the source read-only, builds all CUDA-enabled native targets, lists the registered tests without executing them, and writes checksummed compiler, image, configure, and build evidence. Override `CM_CUDA_ARCHITECTURES` or `CM_CUDA_CONTAINER_IMAGE` to test another target or toolkit image.

The `CUDA compile check` workflow runs this gate for pull requests, pushes to `master`, and manual dispatches on GitHub's hosted Ubuntu runner. Compilation establishes toolchain compatibility, not numerical or application behavior.

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
