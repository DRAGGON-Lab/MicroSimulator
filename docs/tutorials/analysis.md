# Checkpoints, contact graphs, and quantitative analysis

MicroSimulator separates simulation output into three artifacts:

- a checkpoint is an exact, integrity-checked restart artifact;
- a scene is an immutable presentation snapshot; and
- an analysis dataset is an immutable Parquet/Zarr projection with schemas and provenance.

Use checkpoints for resuming, scenes for viewing, and datasets for statistics.

## 1. Generate a time series

```console
uv run microsimulator run \
  --model examples/tutorials/biophysics.py \
  --parameter scenario='"basics"' \
  --seed 42 \
  --steps 200 \
  --dt 0.02 \
  --checkpoint-every 20 \
  --output results/growth.json
```

Periodic filenames contain their run-local completed step. Physical time is stored independently and is the appropriate axis when `dt` differs.

## 2. Export typed tables

```console
uv run microsimulator export-analysis \
  results/growth.step-*.json \
  --output results/growth.dataset \
  --contacts
```

The dataset contains:

- `frames.parquet`: physical time, backend provenance, and row counts;
- `cells.parquet`: stable IDs, parent IDs, slots, geometry, growth, and type;
- `species.parquet`: stable cell ID, channel, and level;
- `contacts.parquet` when requested; and
- `signals.zarr` when the source has a signal grid.

The manifest authenticates every source and output. Existing datasets are not replaced unless replacement is explicitly requested.

## 3. Cell geometry distributions

Use the typed recipes to compute radial XY position and a full-capsule length histogram:

```python
from microsimulator.analysis import open_dataset
from microsimulator.analysis_recipes import (
    cells_with_radial_position,
    length_histogram,
)

dataset = open_dataset("results/growth.dataset")
cells = cells_with_radial_position(dataset).collect()
histogram = length_histogram(
    dataset,
    edges=[0, 1, 2, 3, 4, 5, 6],
).collect()
```

The source table keeps both `cylinder_length` and derived `capsule_length`. The recipe defaults to full capsule length, `length + 2 * radius`, without renaming engine state. Radial position is `sqrt(x^2 + y^2)` and is retained as a derived column.

## 4. Unique contact edges

```python
from microsimulator.analysis import open_dataset
from microsimulator.analysis_recipes import unique_neighbor_edges

dataset = open_dataset("results/growth.dataset")
edges = unique_neighbor_edges(dataset).collect()
```

The raw contact table preserves every mechanics row. The recipe groups an unordered stable-ID pair and reports geometric row count, minimum signed separation, maximum derived overlap, and total contact weight. Negative signed separation means penetration; `overlap = max(0, -signed_separation)` is a derivation and does not replace the signed quantity.

This edge table can be passed to NetworkX or another graph library, but graph construction is a downstream choice. MicroSimulator does not serialize live Python graph objects.

## 5. Species and signals

Use `radial_species_mean` for explicit radial bins. Empty bins have zero count and a null mean, not an invented concentration of zero. For a signal field, use `signal_slice` or `signal_time_course`; both require named axes and indices and materialize only the selected data.

Signal arrays use dimension order `(frame, channel, x, y, z)`. Geometry changes start a new epoch instead of silently padding or resampling.

## 6. Analysis claims checklist

- Compare runs at the same physical time, cell count, or other declared endpoint.
- Use stable IDs for lineage and longitudinal tracking; never persist slots.
- State whether “length” means cylinder or full capsule length.
- State bin edges, inclusion convention, and handling of empty bins.
- Distinguish configured growth rate from realized mechanical elongation.
- Distinguish raw contact rows from unique biological neighbors.
- Record the seed set and parameter map for stochastic comparisons.
- Inspect numerical convergence and time-step sensitivity before interpreting an apparent biological pattern.

The complete recipe API, schemas, and signal examples are documented in [`docs/analysis/recipes.md`](../analysis/recipes.md).
