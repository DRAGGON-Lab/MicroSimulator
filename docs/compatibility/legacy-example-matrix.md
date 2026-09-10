# Legacy example compatibility matrix

The executable matrix pins all 25 Python examples from CellModeller commit `4896f543c6250f053eea2312e628cc3a96bf7408`. Every legacy source and every migrated implementation is SHA-256 authenticated before model code executes. The runner exercises each runnable row on every device exposed by each requested backend and writes a complete JSON report, including failures.

| Legacy example | Classification | Execution path |
| --- | --- | --- |
| `ACS2012/EdgeDetectorChamber.py` | migrated | typed native model |
| `Conjugation.py` | runnable | callback adapter |
| `TimRudgeThesis/Meristem.py` | runnable | callback adapter |
| `Tutorial_1/Tutorial_1a.py` | runnable | callback adapter |
| `Tutorial_1/Tutorial_1b.py` | runnable | callback adapter |
| `Tutorial_1/Tutorial_1c.py` | runnable | callback adapter |
| `Tutorial_2/Tutorial_2a.py` | migrated | typed native model |
| `Tutorial_2/Tutorial_2b.py` | migrated | typed native model |
| `Tutorial_3/Tutorial_3.py` | migrated | typed native model |
| `colorWalk_planes_3d.py` | runnable | callback adapter |
| `ex1_simpleGrowth.py` | runnable | callback adapter |
| `ex1_simpleGrowth2D.py` | runnable | callback adapter |
| `ex1a_simpleGrowth2D.py` | runnable | callback adapter |
| `ex1a_simpleGrowth2Types.py` | runnable | callback adapter |
| `ex1b_simpleGrowth2D.py` | runnable | callback adapter |
| `ex1b_simpleGrowthRoundCell.py` | runnable | callback adapter |
| `ex2_constGene.py` | migrated | typed native model |
| `ex2a_dilution.py` | migrated | typed native model |
| `ex2b_diluteRepression.py` | migrated | typed native model |
| `ex3_simpleSignal.py` | migrated | typed native model |
| `ex4_simpleCellCellSignaling.py` | migrated | typed native model |
| `ex5_colonySector.py` | runnable | callback adapter |
| `ex5_colonySector_3d.py` | runnable | callback adapter |
| `load.py` | migration-only | `microsimulator import-legacy-pickle` |
| `sphere_constraints.py` | runnable | callback adapter |

No example is silently omitted or presently classified as deliberately retired. The 15 runnable sources are executed unchanged through the adapter, the 9 OpenCL equation models execute their typed MicroSimulator migrations, and `load.py` is represented by the separately tested one-way trusted-pickle migration workflow.

From a built development environment, run CPU and all enumerated Metal devices with:

```console
python scripts/run_legacy_example_matrix.py \
  --legacy-root /path/to/CellModeller \
  --backend cpu \
  --backend metal \
  --output build/legacy-example-matrix.json
```

The command fails if a requested backend has no devices, a pinned source has drifted, a migrated implementation has drifted, or any scenario fails to build, advance three steps, or validate its native state. `migration_only` and `deliberately_retired` are non-executing classifications and must carry a reason in the manifest.
