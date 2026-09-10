# Typed translations of legacy equation models

Nine bundled CellModeller examples define their species or signaling equations as OpenCL source and therefore cannot run through the Python callback adapter. Each has a self-contained MicroSimulator translation with typed rate equations and restartable orchestration. The table maps source models to those translations; numerical comparisons are documented separately in the recorded trajectory reference.

| CellModeller example | Equation family | MicroSimulator model |
| --- | --- | --- |
| `ACS2012/EdgeDetectorChamber.py` | five species, one diffusive signal | `examples/legacy/ACS2012/EdgeDetectorChamber.py` |
| `Tutorial_2/Tutorial_2a.py` | one constitutively produced species | `examples/legacy/Tutorial_2/Tutorial_2a.py` |
| `Tutorial_2/Tutorial_2b.py` | two-species nonlinear feedback | `examples/legacy/Tutorial_2/Tutorial_2b.py` |
| `Tutorial_3/Tutorial_3.py` | two species coupled to two diffusive signals | `examples/legacy/Tutorial_3/Tutorial_3.py` |
| `ex2_constGene.py` | one constitutively produced species | `examples/legacy/ex2_constGene.py` |
| `ex2a_dilution.py` | one species with growth dilution only | `examples/legacy/ex2a_dilution.py` |
| `ex2b_diluteRepression.py` | dilution plus Hill repression | `examples/legacy/ex2b_diluteRepression.py` |
| `ex3_simpleSignal.py` | one species coupled to one diffusive signal | `examples/legacy/ex3_simpleSignal.py` |
| `ex4_simpleCellCellSignaling.py` | three species coupled to one diffusive signal | `examples/legacy/ex4_simpleCellCellSignaling.py` |

## Common modeling choices

The species-only ports preserve the OpenCL equations as `RatePlanBuilder` graphs and keep the declared initial concentrations and growth rates. The legacy biophysics assigns `cell.volume = cell.length`, so its `targetVol` division tests are migrated explicitly as target _lengths_ rather than silently switching to capsule volume. `UniformLengthDivision` stores one threshold per stable cell ID, samples daughter thresholds from the original uniform ranges, and uses the controller's checkpointed random stream.

The legacy `jitter_z=False` path used ambient NumPy randomness for small daughter-axis perturbations. The migrated policy preserves the xy-only perturbation geometry but deliberately moves its draws into the explicit controller stream. This is reproducible replacement behavior, not a claim of bitwise identity with an unseeded legacy NumPy global.

Every translated model requests one exact mechanics pass per biological step. The callback adapter separately maps `max_substeps` to bounded new-contact-frontier relaxation. Renderer colors are not encoded into simulation state; the viewer derives colors from typed fields.

## Coupled-rate translation

The legacy signaling callbacks expose extracellular derivatives as concentration rates and divide cell exchange by the `4 * 4 * 4 = 64` voxel volume before returning them. MicroSimulator coupled plans expose extracellular _amount_ rates; the native scatter operation performs the voxel-volume division. The migrated signal outputs therefore return the unscaled exchange amount. Intracellular equations that explicitly used `area / gridVolume` retain their division by 64.

The migrations use conventional diffusion coefficients rather than preserving the legacy implementation's accidental extra factor of one sixth. They retain the declared no-flux boundary behavior and integration choice: forward Euler for `ex3_simpleSignal.py`, and Crank-Nicolson for the other three models. The Crank-Nicolson ports set an absolute residual tolerance of `1e-12`; the engine default would accept a zero field while these models' initially small signal sources remained below its absolute threshold.

`EdgeDetectorChamber.py` initialized `targetVol` but tested the absent `target_volume` attribute against 3.0. Its migration restores the evident intended uniform division threshold of 3.5 to 4.0. This is an explicit repair, not a claim that the original typo's behavior was reproduced.
