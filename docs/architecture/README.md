# Architecture

MicroSimulator connects device geometry, fluid flow, solute transport, and individual-cell biology through explicit simulation interfaces. Python authors the device and model rules; the engine owns numerical state and backend execution; checkpoints, analysis datasets, and the viewer expose that state for reproducible experiments. The [modeling guide](../microfluidics.md) introduces the scientific workflow.

The CPU backend serves as the readable numerical reference. Metal and CUDA implement the same observable contracts with their native programming models; their current support status is documented in [testing and validation](../development/validation.md).

The [numerical contract](numerical-contract.md) is the best starting point for work that affects results across backends. The architecture decision records below explain why each major interface has its present shape.

## Devices, flow, and transport

- [Axis-aligned box constraints](0016-box-constraints.md)
- [Axis-aligned cylinder constraints](0018-cylinder-constraints.md)
- [Signal grid obstacles](0017-grid-obstacles.md)
- [Face-staggered signal velocity fields](0019-velocity-fields.md)
- [Depth-integrated shallow flow](0022-brinkman-flow.md)
- [Resolved MAC Stokes-Brinkman flow and benchmarks](0023-mac-stokes.md)
- [Finite-aspect flow drift on cells](0021-flow-drift.md)
- [Cell removal and washout](0020-cell-removal.md)

## Cell biology and mechanics

- [Typed dynamic contact mechanics](0002-contact-mechanics.md)
- [Explicit daughter fractions](0007-division.md)
- [Persistent fixed rod cells](0009-fixed-cells.md)
- [Biomass, growth, division, and uptake](0024-biomass-accounting.md)
- [Typed species rate plans](0003-species-rates.md)
- [Cell-occupied extracellular volume: design and CPU reference](0025-cell-occupied-volume.md)
- [Grid signaling and cell coupling](0006-grid-signaling.md)
- [Crank-Nicolson signal transport](0008-crank-nicolson-signals.md)
- [Neighbor diffusion](0010-neighbor-diffusion.md)
- [Bounded SBML import](0011-sbml-import.md)
- [Affine grid reactions](0015-affine-grid-reactions.md)

## Execution, data, and presentation

- [Independent native Metal and CUDA backends](0001-native-backends.md)
- [Deterministic batch execution](0005-batch-execution.md)
- [Restartable native model controllers](0014-native-controllers.md)
- [Versioned data-only checkpoints](0004-checkpoints.md)
- [Headless scene protocol and independent viewer](0012-viewer-boundary.md)
- [Versioned columnar analysis datasets](0013-analysis-datasets.md)
