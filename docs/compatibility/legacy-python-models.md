# Running CellModeller Python models

Maintained CellModeller models commonly define four host-Python callbacks: `setup(sim)`, `init(cell)`, `update(cells)`, and `divide(parent, d1, d2)`. MicroSimulator can run models that use this callback lifecycle for growth, mechanics, regulation, constraints, neighbors, division, and host-side species state.

```console
uv run microsimulator run \
  --legacy-model ../CellModeller/Examples/ex1a_simpleGrowth2D.py \
  --backend cpu \
  --seed 42 \
  --steps 100 \
  --dt 0.05 \
  --output results/legacy.json
```

## Supported callback behavior

`LegacyModelAdapter` presents mutable `LegacyCell` objects keyed by stable cell ID. It retains ordinary JSON-like Python attributes across equal division, propagates callback changes to growth rate, cell type, and host species, and refreshes geometry, effective growth, sampled signals, and optional neighbor IDs from native state.

The step order is regulation, division, native integration, mechanics, then cell-state refresh. Geometry remains engine-owned: callbacks may read position, direction, length, and radius, but attempts to mutate them fail explicitly.

`build_legacy_model` provides an opt-in `setup(sim)` facade and temporary import shims for `CLBacterium`, `ModuleRegulator`, and renderer declarations. The shims are removed after model construction and do not install a shadow `CellModeller` package.

## State and resume

Periodic and final checkpoints store adapter options, JSON-like model attributes, tuples, numeric NumPy arrays, and the exact runtime random stream in the authenticated controller payload. Resume verifies the model source digest, reuses the recorded seed and parameters, reconstructs founder state from the original seed, and restores the checkpointed random stream for subsequent callbacks.

Mutable module globals are not checkpointed. Evolving model state should live on cells or in supported native arrays. When no native species plan exists, `cell.species` may hold host-side model state and is checkpointed as such. Once a native species vector exists, its declared size and values are authoritative.

## Division and mechanics

Equal and asymmetric division are supported. The two positive `cell.asymm` weights are normalized into a native daughter fraction, and daughter-direction jitter uses the explicit seeded random stream. `alternate_divisions=True` preserves the 90-degree xy-plane axis rotation used by the CellModeller callback API.

The adapter also maps `max_substeps` to bounded new-contact-frontier relaxation. This limit is checkpointed with the controller so resume preserves the same mechanics schedule.

## Models that require translation

OpenCL strings returned by `specRateCL()` and `sigRateCL()` are not executed or translated automatically. Their equations must be expressed as a `SpeciesRatePlan` or `CoupledRatePlan`, allowing CPU, Metal, and CUDA to use independent native implementations. The bundled translations are listed in [typed translations of legacy equation models](legacy-example-migrations.md).

GUI renderers are also outside the adapter. The independent viewer consumes scene snapshots and does not own model execution or backend state.

The [example compatibility matrix](legacy-example-matrix.md) lists the pinned models exercised by the adapter and typed translations.
