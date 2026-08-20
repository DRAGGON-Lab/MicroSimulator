# MicroSimulator tutorials

These tutorials introduce the MicroSimulator modeling interface through runnable biological examples. Read them in order for a guided path, or use any example independently.

## Start here

1. [Run, inspect, and resume a model](getting-started.md)
2. [Growth, division, cell types, and constraints](biophysics-and-growth.md)
3. [Intracellular species and gene circuits](intracellular-dynamics.md)
4. [Diffusible signals and cell-cell communication](signaling.md)
5. [Plasmid segregation, contacts, and conjugation](discrete-state-and-contacts.md)
6. [Checkpoints, contact graphs, and quantitative analysis](analysis.md)
7. [SimBOL circuit examples](simbol.md)
8. [Microfluidic devices: walls, flow, and washout](microfluidics.md)
9. [Solved flow: a pillar channel, Brinkman feedback, and the benchmarks](flow-solvers.md)

The examples use `uv`, the `microsimulator` command, data-only checkpoints, and the standalone viewer. Each model selects its backend explicitly and can be run headlessly for batch experiments.

## Working with the examples

Teaching models are under [`examples/tutorials`](../../examples/tutorials). Scenario parameters are JSON values passed with `--parameter`; every command in the tutorials can be run from the repository root.

The tutorials state numerical assumptions where they affect interpretation, including the meaning of cell length and volume, concentration dilution, time-step-dependent probabilities, signal units, and boundary conditions. For quantitative studies, follow the convergence and comparison guidance in each lesson rather than relying on viewer appearance alone.

Readers comparing these models with the CellModeller wiki, legacy examples, or SimBOL sources can consult [tutorial sources and model translations](../compatibility/tutorial-source-provenance.md).
