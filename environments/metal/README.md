# Metal environment

Metal is the feature-complete Apple GPU backend. It is implemented directly with the Metal API and independent Metal Shading Language kernels, and it is validated against both the CPU reference and recorded behavior from the original CellModeller OpenCL runtime.

For application commands, use the [shared CPU, Metal, and CUDA tutorial examples](../../docs/tutorials/commands.md#run-the-same-trap-on-cpu-metal-or-cuda), [device discovery](../../docs/tutorials/commands.md#prepare-and-discover-devices), and [shell quoting guide](../../docs/tutorials/commands.md#choose-a-shell). Select this backend with `--backend metal` and an enumerated `--device-index`; the model syntax is unchanged.

Metal is enabled by default on Apple platforms. It compiles embedded MSL source at runtime through `MTLDevice`, making kernel compilation part of device construction and validation.

## Native conformance

Run the complete C++ backend gate on Apple GPU hardware:

```console
scripts/run_metal_conformance.sh
```

Every Metal-enabled test build includes `metal_runtime_gate`, which constructs each enumerated device and compiles every embedded MSL library. The runner requires a clean worktree, performs a fresh configure and clean rebuild, executes the complete CTest suite, and writes a timestamped evidence directory under `build/`. The evidence records the source commit, display-device inventory, macOS, Xcode, Clang, Metal framework, logs, JUnit results, final status, and `SHA256SUMS`.

## Application conformance

Run the Python, compatibility, and application gate with a checkout of the pinned original CellModeller source:

```console
CM_LEGACY_ROOT=/path/to/pinned/CellModeller scripts/run_metal_application_conformance.sh
```

This gate builds the Python extension with Metal enabled, runs the full Python and recorded-trajectory suites, executes all 24 runnable legacy examples on CPU and every Metal device, and exercises controller resume, viewer scene semantics, and analysis export with native derived contacts.

The manually dispatched `Metal conformance` workflow runs both gates on a self-hosted macOS runner carrying the `metal` label and always uploads its evidence. It is intentionally not triggered by pull requests.

## Implementation notes

Contact geometry uses deterministic sweep-and-prune capsule candidates followed by native MSL count, inclusive-scan, and fill pipelines.

Rod mechanics uses native MSL Jacobian assembly, matrix-free `B` and `B^T` applications, vector updates, pairwise reductions, and a host-orchestrated conjugate-gradient loop. Only scalar reduction results cross back to the host during iteration; per-cell vectors remain in Metal buffers until the final correction result.

See the [validation workflow](../../docs/development/validation.md) for the complete acceptance policy.
