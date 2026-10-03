<p align="center">
  <img src="docs/assets/microsimulator-logo.png" alt="MicroSimulator" width="640">
</p>

MicroSimulator simulates microbial populations in microfluidic devices. It brings device geometry, steady fluid flow, solute transport, and individual-cell biology into one Python modeling workflow, connecting the conditions in a channel or trap to colony growth, signaling, and washout.

Models run on a C++23 engine with native CPU, Apple Metal, and NVIDIA CUDA backends. Simulations can run headlessly or with an interactive browser viewer. Device geometry and flow are optional for standalone colony models.

## Modeling cells in devices

- **Device geometry:** describe traps, channels, and pillars with mechanical walls and solid masks that constrain cells and route solutes through the fluid space.
- **Flow and transport:** solve depth-averaged Hele-Shaw/Darcy flow in shallow devices or resolved Stokes-Brinkman flow on a staggered grid, then use the velocity field for conservative advection, diffusion, and reactions.
- **Coupled culture:** compose pressure/rate-driven Stokes flow, conservative chemical transport, and nutrient-limited cell growth.
- **Growth and signaling:** combine nutrient uptake with conserved biochemical biomass, rod growth and division, contact mechanics, lineage, intracellular circuits, and diffusible signals.
- **Cells in flow:** model free-cell drift and rotation, explicitly attached populations, stationary biomass resistance, and model-defined outlet removal.
- **Reproducible experiments:** inspect live simulations, resume versioned checkpoints, run parameter sweeps, and export Parquet/Zarr datasets for quantitative analysis.

The [microfluidics modeling guide](docs/microfluidics.md) explains how these pieces fit together. The device tutorials use kinematic cell motion followed by contact relaxation and empirical stationary resistance. The [fluid and culture guide](docs/models/fluid-culture.md) covers hydrodynamic body coupling and its resolution and performance limits. The [flow benchmarks](docs/tutorials/flow-solvers.md#numerical-evidence) and [nutrient study](docs/tutorials/nutrient-validation.md) document numerical checks and refinement studies. Experimental calibration remains specific to each model.

## Quick start

MicroSimulator requires Python 3.12, CMake 3.25 or newer, Ninja, a C++23 compiler, and [uv](https://docs.astral.sh/uv/).

```console
git clone git@github.com:DRAGGON-Lab/MicroSimulator.git
cd MicroSimulator
uv sync --group dev
uv run microsimulator devices
uv run microsimulator run \
  --model examples/mother_machine.py \
  --backend cpu \
  --seed 42 \
  --steps 100 \
  --dt 0.025 \
  --output results/mother-machine.json
```

This runs a mother machine with cells confined in closed-ended growth channels and saves a restartable checkpoint. The short run checks startup and transport at the default biological timescale. Follow the [mother-machine guide](docs/tutorials/mother-machine.md) to watch growth and washout in the viewer, including an explicit accelerated-growth option for interactive demonstrations.

## Examples

| Explore                                                                | Start with                                                                                    |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Mother-cell retention by confinement, division, and descendant washout | [Mother machine](docs/tutorials/mother-machine.md)                                            |
| Nutrient delivery, colony growth, and washout in a trap                | [Microfluidic trap](examples/microfluidic_trap.py)                                            |
| Perfusion, nutrient transport, and cell growth                         | [Growing media](examples/growing_media.py)                                                    |
| Flow around pillars, attached founders, and released daughters         | [Pillar channel](examples/tutorials/pillar_channel.py)                                        |
| A quorum-sensing clock in a flowing device                             | [Danino clock](examples/tutorials/danino_clock.py)                                            |
| A single biopixel trap with documented dimensions and CAD provenance   | [Biopixel tutorial](docs/tutorials/microfluidics.md#a-source-backed-prindle-biopixel-example) |
| Nutrient penetration, biomass gain, and conservation                   | [Controlled nutrient study](docs/tutorials/nutrient-validation.md)                            |

The [tutorial suite](docs/tutorials/README.md) also covers growth, gene circuits, signaling, plasmids, contacts, and analysis. Device examples state their geometry, units, and modeling assumptions alongside the runnable code.

## Backend status

| Backend     | Status                   | Role                                       |
| ----------- | ------------------------ | ------------------------------------------ |
| CPU         | Feature complete         | Portable execution and numerical reference |
| Apple Metal | Feature complete         | Native Apple GPU execution                 |
| NVIDIA CUDA | Under active development | Native NVIDIA GPU execution                |

The flow solvers and coupled culture model have native CPU, Metal, and CUDA implementations. CPU and Metal support the complete current modeling workflow. CUDA compilation is checked, while NVIDIA runtime and application validation remain required for supported status. The [validation policy](docs/development/validation.md) defines the hardware and application acceptance criteria.

## Documentation

| Topic                      | Entry point                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| Microfluidics              | [Devices, flow, transport, and biology](docs/microfluidics.md)         |
| Tutorials                  | [Modeling tutorials](docs/tutorials/README.md)                         |
| Architecture and numerics  | [Design documents](docs/README.md#architecture-and-numerics)           |
| HPC environments           | [CPU, Metal, and CUDA setup](docs/README.md#execution-environments)    |
| Analysis and visualization | [Research output workflows](docs/README.md#analysis-and-visualization) |
| CellModeller compatibility | [Scope and evidence](docs/README.md#cellmodeller-compatibility)        |
| Development                | [Testing and validation](docs/development/validation.md)               |

The complete documentation index is available at [`docs/README.md`](docs/README.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the shared formatting command, commit hooks, readability conventions, and code quality checks.

## Origins

MicroSimulator began as a rewrite of [CellModeller](https://github.com/cellmodeller/CellModeller) and has developed into an independent microfluidics simulation system. It builds on that lineage of individual-based cell modeling with its own device, flow, transport, and experiment workflows. The [compatibility guide](docs/compatibility/README.md) documents supported CellModeller models and intentional numerical differences. Users of the former CellModeller2 package can follow the [rename guide](docs/compatibility/microsimulator-rename.md).

## License

MicroSimulator is available under the [MIT License](LICENSE).
