# Legacy OpenCL trajectory environment

This optional environment exists only to reproduce the recorded CellModeller reference trajectories. It is not a MicroSimulator runtime dependency.

On an Apple machine that still exposes an OpenCL device:

```console
uv venv --python 3.13 /tmp/cm-legacy-opencl
uv pip install \
  --python /tmp/cm-legacy-opencl/bin/python \
  --requirement environments/legacy-opencl/requirements.txt
/tmp/cm-legacy-opencl/bin/python scripts/record_legacy_trajectories.py \
  --legacy-root /path/to/CellModeller \
  --output /tmp/legacy-trajectories-v1.json
cmp compatibility/legacy-trajectories-v1.json /tmp/legacy-trajectories-v1.json
```

The checkout supplied with `--legacy-root` must be at the commit pinned by `compatibility/legacy-examples-v1.json`. The recorder verifies the commit and all five source digests before executing model code.
