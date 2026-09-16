<p align="center">
  <img src="docs/assets/microsimulator-logo.png" alt="MicroSimulator" width="640">
</p>

MicroSimulator simulates microbial populations in microfluidic devices. It brings device geometry, steady fluid flow, solute transport, and individual-cell biology into one Python modeling workflow, connecting the conditions in a channel or trap to colony growth, signaling, and washout.

Models run on a C++23 engine with native CPU, Apple Metal, and NVIDIA CUDA backends. Simulations can run headlessly or with an interactive browser viewer. Device geometry and flow are optional for standalone colony models.

## Modeling cells in devices

- **Device geometry:** describe traps, channels, and pillars with mechanical walls and solid masks that constrain cells and route solutes through the fluid space.
- **Flow and transport:** solve depth-averaged Hele-Shaw/Darcy flow in shallow devices or resolved Stokes-Brinkman flow on a staggered grid, then use the velocity field for conservative advection, diffusion, and reactions.
- **Growth and signaling:** combine nutrient uptake with conserved biochemical biomass, rod growth and division, contact mechanics, lineage, intracellular circuits, and diffusible signals.
- **Cells in flow:** model free-cell drift and rotation, explicitly attached populations, stationary biomass resistance, and model-defined outlet removal.
- **Reproducible experiments:** inspect live simulations, resume versioned checkpoints, run parameter sweeps, and export Parquet/Zarr datasets for quantitative analysis.

The [microfluidics modeling guide](docs/microfluidics.md) explains how these pieces fit together. Cell motion uses a kinematic approximation followed by contact relaxation; stationary resistance is an empirical closure. The [flow benchmarks](docs/tutorials/flow-solvers.md#numerical-evidence) and [nutrient study](docs/tutorials/nutrient-validation.md) document numerical checks and refinement studies. Experimental calibration remains specific to each model.

## Quick start

MicroSimulator requires Python 3.12, CMake 3.25 or newer, Ninja, a C++23 compiler, and [uv](https://docs.astral.sh/uv/).

```console
git clone git@github.com:DRAGGON-Lab/MicroSimulator.git
cd MicroSimulator
uv sync --group dev
uv run microsimulator devices
uv run microsimulator run \
  --model examples/microfluidic_trap.py \
  --backend cpu \
  --seed 42 \
  --steps 100 \
  --dt 0.02 \
  --output results/trap.json
```

This runs a cell trap supplied with nutrient through a flowing channel and saves a restartable checkpoint. Follow [getting started](docs/tutorials/getting-started.md) to set up the viewer, or continue with [microfluidic devices](docs/tutorials/microfluidics.md) to explore walls, transport, growth, and washout.

## Examples

| Explore | Start with |
| --- | --- |
| Nutrient delivery, colony growth, and washout in a trap | [Microfluidic trap](examples/microfluidic_trap.py) |
| Flow around pillars, attached founders, and released daughters | [Pillar channel](examples/tutorials/pillar_channel.py) |
| A quorum-sensing clock in a flowing device | [Danino clock](examples/tutorials/danino_clock.py) |
| A single biopixel trap with documented dimensions and CAD provenance | [Biopixel tutorial](docs/tutorials/microfluidics.md#a-source-backed-prindle-biopixel-example) |
| Nutrient penetration, biomass gain, and conservation | [Controlled nutrient study](docs/tutorials/nutrient-validation.md) |

The [tutorial suite](docs/tutorials/README.md) also covers growth, gene circuits, signaling, plasmids, contacts, and analysis. Device examples state their geometry, units, and modeling assumptions alongside the runnable code.

## Backend status

| Backend | Status | Role |
| --- | --- | --- |
| CPU | Feature complete | Portable execution and numerical reference |
| Apple Metal | Feature complete | Native Apple GPU execution |
| NVIDIA CUDA | Under active development | Native NVIDIA GPU execution |

Both flow solvers have native CPU, Metal, and CUDA implementations. CPU and Metal support the complete current modeling workflow. CUDA compilation is checked, while NVIDIA runtime and application validation remain required for supported status. The [validation policy](docs/development/validation.md) defines the hardware and application acceptance criteria.

## Documentation

| Topic | Entry point |
| --- | --- |
| Microfluidics | [Devices, flow, transport, and biology](docs/microfluidics.md) |
| Tutorials | [Modeling tutorials](docs/tutorials/README.md) |
| Architecture and numerics | [Design documents](docs/README.md#architecture-and-numerics) |
| HPC environments | [CPU, Metal, and CUDA setup](docs/README.md#execution-environments) |
| Analysis and visualization | [Research output workflows](docs/README.md#analysis-and-visualization) |
| CellModeller compatibility | [Scope and evidence](docs/README.md#cellmodeller-compatibility) |
| Development | [Testing and validation](docs/development/validation.md) |

The complete documentation index is available at [`docs/README.md`](docs/README.md).

## Origins

MicroSimulator began as a rewrite of [CellModeller](https://github.com/cellmodeller/CellModeller) and has developed into an independent microfluidics simulation system. It builds on that lineage of individual-based cell modeling with its own device, flow, transport, and experiment workflows. The [compatibility guide](docs/compatibility/README.md) documents supported CellModeller models and intentional numerical differences. Users of the former CellModeller2 package can follow the [rename guide](docs/compatibility/microsimulator-rename.md).

## License

MicroSimulator is available under the [MIT License](LICENSE).
