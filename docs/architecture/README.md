# Architecture

MicroSimulator separates model authoring, backend-neutral simulation state, accelerator implementations, storage formats, and presentation. The CPU backend serves as the readable numerical reference; Metal and CUDA implement the same observable contracts with their native programming models.

The [numerical contract](numerical-contract.md) is the best starting point for work that affects results across backends. The architecture decision records below explain why each major interface has its present shape.

## Engine and execution

- [Independent native Metal and CUDA backends](0001-native-backends.md)
- [Typed dynamic contact mechanics](0002-contact-mechanics.md)
- [Deterministic batch execution](0005-batch-execution.md)
- [Explicit daughter fractions](0007-division.md)
- [Persistent fixed rod cells](0009-fixed-cells.md)
- [Restartable native model controllers](0014-native-controllers.md)
- [Axis-aligned box constraints](0016-box-constraints.md)
- [Axis-aligned cylinder constraints](0018-cylinder-constraints.md)
- [Cell removal](0020-cell-removal.md)
- [Steady Hele-Shaw-Brinkman flow solve](0022-brinkman-flow.md)
- [Staggered MAC Stokes-Brinkman solve and flow benchmarks](0023-mac-stokes.md)

## Biological dynamics and signaling

- [Typed species rate plans](0003-species-rates.md)
- [Grid signaling and cell coupling](0006-grid-signaling.md)
- [Crank-Nicolson signal transport](0008-crank-nicolson-signals.md)
- [Neighbor diffusion](0010-neighbor-diffusion.md)
- [Bounded SBML import](0011-sbml-import.md)
- [Affine grid reactions](0015-affine-grid-reactions.md)
- [Signal grid obstacles](0017-grid-obstacles.md)
- [Face-staggered signal velocity fields](0019-velocity-fields.md)

## Data and presentation

- [Versioned data-only checkpoints](0004-checkpoints.md)
- [Headless scene protocol and independent viewer](0012-viewer-boundary.md)
- [Versioned columnar analysis datasets](0013-analysis-datasets.md)

Current backend status and the tests required to support it are documented in [testing and validation](../development/validation.md).

- [Biomass accounting](0024-biomass-accounting.md)
