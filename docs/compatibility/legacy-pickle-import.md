# Importing CellModeller pickle snapshots

CellModeller 1 writes Python pickle snapshots in two families:

- current mapping snapshots contain `cellStates`, `lineage`, `stepNum`, the model name and source, and optional species or signal arrays;
- older tuple snapshots contain cell states plus lineage, with an optional signaling tuple, but no step number or model source.

Pickle is an executable serialization format. `microsimulator import-legacy-pickle` therefore refuses to read a file unless `--trust-legacy-pickle` is present. A restricted unpickler admits the historical `CellState` representation, numeric NumPy arrays, and the small set of reconstruction globals those values need; all other globals fail before they are resolved. This reduces accidental exposure but is not a security boundary for adversarial files. Import snapshots only from a trusted simulation run.

## What the importer preserves

The importer requires `--native-state-only` because the old format cannot support exact continuation. It migrates:

- stable cell IDs and compact slots;
- position, normalized direction, centerline length, and radius;
- growth rate and cell type;
- per-cell species concentrations, when all cells agree on the species count;
- the complete monotonic lineage map;
- physical time supplied directly with `--time`, or derived from `stepNum` and an explicit legacy `--dt`.

The resulting file is an ordinary authenticated MicroSimulator JSON checkpoint with a zero species-rate plan. It contains no executable model source. Its provenance records the input digest, legacy format, time basis, model-source digest when present, and the names of all dropped cell attributes.

The old snapshots do not contain plane or sphere constraints, callback random state, or enough information to reconstruct a typed rate plan. Signal grid geometry lacks complete transport semantics. Those omissions are recorded in provenance and are never guessed. User-defined callback attributes such as division thresholds and display colors are reported as dropped fields; use the legacy callback loader and a fresh model start when those fields are required.

```console
uv run microsimulator import-legacy-pickle data/step-00100.pickle \
  --output results/step-00100.json \
  --dt 0.05 \
  --trust-legacy-pickle \
  --native-state-only
```

Tuple snapshots have no step number, so they require `--time` instead of `--dt`. Existing outputs are not replaced without `--overwrite`.
