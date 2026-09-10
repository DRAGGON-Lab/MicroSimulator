# Mapping legacy analysis workflows

The legacy `Scripts/` directory and `Scripts/Analysis.ipynb` illustrate how CellModeller results were batched, inspected, plotted, and rendered. They do not form a single compatibility API: several are Python 2-era, execute pickle content, reach into OpenCL solver buffers, or encode workstation-specific paths.

## Workflow map

| Source | CellModeller workflow | MicroSimulator approach |
| --- | --- | --- |
| `batch.py` | selects an OpenCL platform/device interactively, then runs until a fixed cell-capacity margin | `microsimulator devices` lists devices, while `microsimulator run` uses an explicit cell-count stopping criterion. |
| `multi_batch.py` | repeats the same unseeded run three times | Run manifests declare replicate IDs, seeds, parameters, and outputs as reviewable data. |
| `batch_iter.py` | hard-coded gamma sweep over eight values | A run manifest expresses the sweep; scheduler configuration remains outside the simulator. |
| `batchFile.py` | cluster-specific model defaults, timestamped pickle directories, and private physics setup calls | Explicit output paths, checkpoints, and provenance cover the portable parts of this workflow. |
| `CellModellerGUI.py` | starts the PyQt/OpenGL GUI | The independent viewer and live controller provide interactive inspection. |
| `LengthHistogram.py` | writes radial position and full capsule length to a global CSV | Typed cell tables expose both quantities, with `length + 2 * radius` named `capsule_length`. |
| `spatial_analyze.py` | computes counts and mean species channel 0 in 20 radial XY bins | The dataframe recipe accepts explicit bin edges, radius, and species channel. |
| `contactGraph.py` | reloads executable model text, reconstructs OpenCL contacts, collapses geometric rows into an undirected graph, and draws a PDF | Analysis export provides typed contact rows; a query derives unique neighbor edges without re-executing model source. |
| `Draw2DPDF.py` | renders 2D capsules and one signal slice from pickle fields | Scene v1 provides the required geometry and signals for a viewer or publication renderer. |
| `video.sh`, `contactVideo.sh` | shell loops through pickles, PDF conversion, and ffmpeg | Scene frames can feed an external rendering and video workflow without making ImageMagick conventions part of the engine. |
| `printTiming.py` | prints cell count beside checkpoint filesystem modification time | Checkpoints record simulation time; file modification time is not treated as physical time or measured runtime. |
| `gitPublish.sh` | merges a historical private repository into a public repository | Repository publication is outside the simulation and analysis interfaces. |
| `Analysis.ipynb` | demonstrates position/orientation plots, length and radial histograms, length-weighted 2D density, raw overlap inspection, lineage, and sister-neighbor counting | Documented recipes operate on versioned datasets rather than private solver buffers or local paths. |

## Data retained for analysis

The legacy scripts imply a compact, useful analysis vocabulary:

- simulation time, source checkpoint, backend identity, and cell count per frame;
- stable cell ID, active parent ID, compact slot, position, direction, cylinder length, radius, full capsule length, growth rate, cell type, fixed state, and every species channel;
- every geometric contact row, including its two stable IDs, ordinal, surface point, normal, signed separation, and weight;
- unique neighbor edges derived by grouping contact rows by stable ID pair;
- signal-grid values with explicit channel and x-y-z coordinates; and
- experiment identity, seed, parameters, model digest, and checkpoint digest.

The old notebook's density example weights XY histogram bins by full capsule length. That is a declared line-density proxy, not cell area, volume, biomass, or packing fraction. Any replacement recipe must use that name. Likewise, negative signed separation is penetration; an `overlap` column may be derived as `max(0, -signed_separation)` but must not replace the signed value.

The radial species script contains no trustworthy bin contract. Its plotted coordinates start at zero, while its text output uses half of each upper edge rather than the actual bin center. MicroSimulator defines bins through explicit edges and reports their left edge, right edge, and center.

## Behavior not carried forward

- Python pickle remains confined to the explicit trusted one-way importer.
- Analysis never imports model source from a checkpoint or invokes callbacks.
- Private OpenCL arrays and solver methods are not an analysis API.
- Stable IDs are never replaced by compact slots in exported relationships.
- File modification time is not accepted as a physical or performance clock.
- Hard-coded paths, implicit working-directory outputs, and unsorted glob loops are not reproducibility contracts.
- Legacy per-cell RGB callback attributes are not scientific state. Color maps remain declarative presentation choices.

## Supported workflows

MicroSimulator converts an explicit, ordered checkpoint series into versioned Parquet cell/contact tables and Zarr signal arrays. The documented dataframe recipes reproduce the meaningful quantities above. CPU, Metal, and CUDA checkpoints use the same storage schema, and derived contact rows record the backend and contact parameters used.

Offline publication rendering and video export are separately useful, but are not prerequisites for scientific analysis-table compatibility.
