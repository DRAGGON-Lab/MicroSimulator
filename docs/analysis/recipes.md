# Analysis dataset recipes

MicroSimulator analysis starts from an immutable dataset, not executable model code or a simulation object's private buffers. Install the optional dependency set before using these APIs:

```console
uv sync --extra analysis
```

Open and verify the dataset once, then reuse the resulting handle. The reader accepts the original v1 identity contract and the current v2 contract. Verification checks the dataset identity and every Parquet and Zarr digest before a recipe can scan data:

```python
from microsimulator.analysis import open_dataset
from microsimulator.analysis_recipes import radial_counts

dataset = open_dataset("results/run.dataset")
counts = radial_counts(dataset, edges=[0.0, 5.0, 10.0, 20.0]).collect()
```

Table recipes return `polars.LazyFrame`. A caller decides when to collect, filter, aggregate across frames, or write another table. Every binned result retains `frame_index`, physical `time`, the bin index, its left and right edges, and its actual center. Bins are left-closed and right-open; only the last bin includes its right edge. Values outside the declared edges are excluded. Empty bins are retained.

## Geometry and species

`cells_with_radial_position(dataset)` adds `radial_xy = sqrt(position_x^2 + position_y^2)` to the complete cell scan. It does not replace the source coordinates.

`radial_counts(dataset, edges)` returns `cell_count` per frame and radial bin. `radial_species_mean(dataset, channel, edges)` returns `cell_count` and `species_mean`. An empty bin or unavailable channel has a zero count and a null mean, not an invented zero concentration.

`length_histogram(dataset, edges)` bins the exported `capsule_length` by default. Pass `length="cylinder_length"` to select the engine's cylinder length explicitly. This distinction preserves the legacy script's `length + 2 * radius` quantity without renaming the underlying state.

```python
from microsimulator.analysis_recipes import length_histogram, radial_species_mean

species = radial_species_mean(dataset, channel=0, edges=[0, 10, 20, 40])
capsules = length_histogram(dataset, edges=[0, 1, 2, 3, 4, 5, 6])
```

`line_density_xy(dataset, x_edges, y_edges)` sums full capsule length in each explicit XY bin. Its output is deliberately named `line_density_proxy`. It is the old notebook's length-weighted histogram and is not cell area, volume, biomass, packing fraction, or a value normalized by bin area.

## Contacts and lineage

Contact recipes require an export created with `--contacts`. `unique_neighbor_edges(dataset)` groups all parallel capsule-contact rows for an unordered stable-ID pair. It returns the number of geometric rows, minimum signed separation, maximum derived overlap, and total contact weight. The raw contact table remains authoritative when contact-point or ordinal detail is needed.

`sister_neighbor_counts(dataset)` counts unique active neighbors with the same non-null parent ID. Each edge contributes once to each endpoint. Founders with null parents are never classified as sisters merely because both parents are null.

```python
from microsimulator.analysis_recipes import sister_neighbor_counts, unique_neighbor_edges

edges = unique_neighbor_edges(dataset).collect()
sister_counts = sister_neighbor_counts(dataset).collect()
```

## Signals

Signal data keeps the named dimension order `(frame, channel, x, y, z)`. Geometry changes create separate epochs. `local_frame` indexes within one epoch, while `SignalSlice.frame_index` reports the dataset-wide frame index.

```python
from microsimulator.analysis_recipes import signal_slice, signal_time_course

plane = signal_slice(
    dataset,
    epoch=0,
    local_frame=3,
    channel=0,
    axis="z",
    index=0,
)
# plane.dimensions == ("x", "y"); plane.values is a float32 NumPy array.

course = signal_time_course(dataset, epoch=0, channel=0, x=10, y=12, z=0)
```

These functions materialize only the selected plane or voxel time course and return numerical data before plotting. Axis names and indices are explicit; the API does not silently transpose, pad, or resample epochs.
