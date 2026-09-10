# Comparing with recorded CellModeller trajectories

MicroSimulator keeps a raw, source-controlled trajectory set from the original OpenCL implementation at `compatibility/legacy-trajectories-v1.json`. The recording uses CellModeller commit `4896f543c6250f053eea2312e628cc3a96bf7408` and authenticates every model source against the 25-example matrix before execution.

## Recording environment

The current reference was recorded on the Apple M4 Max through Apple OpenCL 1.2 with Python 3.13.5, PyOpenCL 2026.1.2, NumPy 2.5.2, and SciPy 1.18.0. The reproduction script is `scripts/record_legacy_trajectories.py`. It applies two non-scientific compatibility shims:

- renderer constructors are replaced with no-op objects so batch recording never enters an OpenGL context;
- equal-sized flat host arrays are reshaped to the device-array shape because current PyOpenCL rejects the storage-order mismatch accepted by the legacy dependency stack.

Neither shim changes simulation values or kernel source. Python's `random` and NumPy's random generator are both seeded before each scenario. The fixture records aggregate geometry, population, cell type, neighbor, species, signal, and mechanics-substep observables at named steps rather than serializing an opaque legacy pickle.

## Representative scenarios

| Contract role            | Legacy model                | Final recorded step |
| ------------------------ | --------------------------- | ------------------: |
| growing 2D colony        | `ex1_simpleGrowth2D.py`     |                  20 |
| constrained 3D colony    | `Tutorial_1/Tutorial_1c.py` |                  20 |
| neighbor-dependent model | `Conjugation.py`            |                 100 |
| species model            | `ex2_constGene.py`          |                  20 |
| coupled signaling model  | `Tutorial_3/Tutorial_3.py`  |                  10 |

`Tutorial_3` is used for coupled signaling because the bundled `ex3` and `ex4` files exceed their own default `max_planes=1` capacity during setup. Its Crank-Nicolson implementation is recorded as it runs, including the known discarded-convolution defect; this makes the fixture evidence, not a request to reintroduce that defect.

## Comparison method

`python/tests/test_legacy_trajectories.py` executes the exact callback models or their source-pinned typed migrations on every available backend. It compares all recorded frames, not just the final state. The tolerances are declared by observable family in the test:

- population, total length, centroid, and colony radius bound the deliberate mechanics and RNG-policy differences;
- neighbor count and the appearance of acceptor, donor, and transconjugant types establish the contact-dependent behavior without requiring the same stochastic infection step;
- species totals remain within 3 percent for the constitutive model;
- coupled species totals remain within 0.1 percent, signal mass within 0.3 percent, and peak signal within 12 percent. The peak allowance covers the intended Crank-Nicolson and conventional-diffusion correction.

The CPU and Metal implementations pass these five contracts. This is not bitwise legacy parity: the raw values visibly retain the numerical-model departures documented in the mechanics and signaling audits. It is independent evidence that shared CPU/Metal behavior still preserves the representative CellModeller phenomena.

## Reproduction

The recorder intentionally runs under an environment containing the original OpenCL dependencies, not the MicroSimulator environment:

```console
python scripts/record_legacy_trajectories.py \
  --legacy-root /path/to/CellModeller \
  --output compatibility/legacy-trajectories-v1.json
```

To execute the comparison suite against a pinned legacy checkout:

```console
CM_LEGACY_ROOT=/path/to/CellModeller \
  pytest -q python/tests/test_legacy_trajectories.py
```
