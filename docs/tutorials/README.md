# MicroSimulator tutorials

These tutorials explore cells in microfluidic devices through runnable models: geometry and flow supply the environment, while growth, mechanics, and circuits determine how populations respond. The [modeling guide](../microfluidics.md) introduces the full workflow and its assumptions. Each example can also be used independently.

For backend selection, PowerShell syntax, quoted JSON parameters, and paths with spaces, see [tutorial commands by backend and shell](commands.md#choose-a-shell). Multiline commands on this page use POSIX shell backslashes; the guide provides the PowerShell equivalents and [explicit CPU, Metal, and CUDA trap launches](commands.md#run-the-same-trap-on-cpu-metal-or-cuda).

## Start here

Follow [getting started](getting-started.md) to run a nutrient-fed trap, configure the viewer, and resume a checkpoint. Then choose a path below.

The examples use `uv`, the `microsimulator` command, data-only checkpoints, and the standalone viewer. Each model selects its backend explicitly and can be run headlessly for batch experiments.

## Devices, flow, and transport

1. [Microfluidic devices: walls, flow, and washout](microfluidics.md) connects geometry, nutrient delivery, cell growth, and outlet removal, with trap, clock, and biopixel examples.
2. [Solved flow in a pillar channel](flow-solvers.md) introduces attached founders, released daughters, stationary resistance, solver selection, and analytic benchmarks.
3. [Nutrient penetration and attached-population growth](nutrient-validation.md) measures spatial growth, conservation, and sensitivity to spatial and temporal resolution.

## Cell biology and circuits

These lessons develop the biological rules used within devices and in standalone colony models:

1. [Growth, division, cell types, and constraints](biophysics-and-growth.md)
2. [Intracellular species and gene circuits](intracellular-dynamics.md)
3. [Diffusible signals and cell-cell communication](signaling.md)
4. [Plasmid segregation, contacts, and conjugation](discrete-state-and-contacts.md)
5. [SimBOL circuit examples](simbol.md)

## Analyze an experiment

The [analysis tutorial](analysis.md) covers checkpoints, contact graphs, and quantitative output. Continue with [analysis recipes](../analysis/recipes.md) for reproducible Parquet/Zarr datasets and Polars queries.

## Working with the examples

Teaching models are under [`examples/tutorials`](../../examples/tutorials). Scenario parameters are JSON values passed with `--parameter`; every command in the tutorials can be run from the repository root.

The tutorials state numerical assumptions where they affect interpretation, including the meaning of cell length and volume, concentration dilution, time-step-dependent probabilities, signal units, and boundary conditions. For quantitative studies, follow the convergence and comparison guidance in each lesson rather than relying on viewer appearance alone.

Readers comparing these models with the CellModeller wiki, legacy examples, or SimBOL sources can consult [tutorial sources and model translations](../compatibility/tutorial-source-provenance.md).
