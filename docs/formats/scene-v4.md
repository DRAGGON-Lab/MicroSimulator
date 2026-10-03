# MicroSimulator scene format v4

A scene is an immutable presentation snapshot with the `microsimulator-scene` format identifier. The root fields are `format`, `version` (4), `producer`, `integrity`, and `frame`. `integrity.frame` is the SHA-256 digest of the frame encoded with RFC 8785 canonical JSON. Readers reject unknown, duplicate, or missing fields, nonfinite numbers, invalid domain values, and documents larger than 1 GiB. A digest detects corruption; it does not authenticate a publisher.

The frame contains `time`, `backend`, `species_count`, `cells`, `constraints`, `signal_grid`, `channel_metadata`, and `media`. The [geometry and grid definitions](scene-v2.md) specify the cell, constraint, and scalar-grid fields. [Channel metadata](scene-v3.md) supplies ordered species and signal labels. Each group is limited to 4096 channels independently, before labels or viewer controls are allocated. Cell IDs are canonical positive decimal strings representing unsigned 64-bit values.

## Physical medium

`media` is null for simulations without physical media flow. Otherwise it contains these fields:

| Field                             | Meaning                                                                                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `length_unit_m`, `time_unit_s`    | Positive SI conversion factors for model lengths and times                                                                                        |
| `viscosity_pa_s`, `density_kg_m3` | Positive medium properties in SI                                                                                                                  |
| `shape`, `origin`, `spacing`      | Three grid dimensions, voxel-zero center, and positive isotropic model spacing                                                                    |
| `obstacles`                       | Empty array or one 0/1 wall flag per voxel, with z varying fastest                                                                                |
| `solutes`                         | Unique nonempty names in signal-channel order                                                                                                     |
| `cells`                           | Stable `id`, unit `orientation` quaternion `(w,x,y,z)`, positive `biochemical_volume`, and nonnegative `species_amounts` in species-channel order |
| `fragments`                       | Voxel `site`, connected `component`, positive fluid `volume`, `centroid`, and nonnegative `amounts` in solute order                               |
| `reservoirs`                      | Port `name` and signed cumulative `amounts` transferred into the fluid                                                                            |
| `max_speed_m_s`                   | Maximum speed from the last fluid solve                                                                                                           |
| `flow_relative_residual`          | Relative residual from that fluid solve                                                                                                           |
| `maximum_volume_residual`         | Largest local geometric-conservation residual, in model volume                                                                                    |

Media numbers retain binary64 precision. Fragment order defines the presentation fragment index. Multiple disconnected fragments may share one voxel; a reader must preserve them separately. Media cell IDs match the frame's cell IDs in order. Scene cell geometry remains the float32 rendering projection; quantitative binary64 poses are available in checkpoints and analysis datasets.

For a media scene with solutes, `signal_grid` contains fluid-volume-weighted voxel concentration means: the sum of fragment amounts in that voxel divided by the sum of their fluid volumes. Entirely solid voxels display zero. Its no-flux boundary fields are presentation placeholders; actual ports and boundary equations belong to checkpoint configuration. The viewer labels these concentrations as extracellular means. These arrays do not replace the fragment amounts for conservation calculations.

The last fluid solve describes the initial geometry of its accepted substep. A scene is not a restart artifact and does not contain solver configuration, rate plans, or controller authority. See the [media modeling guide](../models/fluid-culture.md) for state ownership and units.

## Readers and writers

Current writers emit [version 5](scene-v5.md). Readers accept versions 2–5, verify each original frame against its declared schema and digest, and supply absent channel labels and media state in memory. A version-2 or version-3 frame containing `media` is invalid. Labels are rendered as text and never interpreted as HTML or executable content. The Python-generated media fixture in `viewer/tests/fixtures/media-v4.scene.json` is shared interchange evidence for the browser reader.
