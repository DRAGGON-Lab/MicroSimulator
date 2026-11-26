# ADR 0029: transport and flow over fluid sites

- Status: accepted
- Date: 2026-11-26

## Context

A signal grid is a box of sites, some of them obstacles. A colony's trap is mostly fluid. A whole microfluidic chip is mostly wall: the example mixer at a 12.5 micrometer spacing has 127,000 fluid sites in a box of 4.0 million, about 3 percent, in one of its three layers.

Transport does nothing at an obstacle site. Its level is zero and stays zero. A step that visits every site of the box, or that checks or copies arrays as large as the box, costs what the box costs and not what the fluid costs, and on a chip that is thirty times too much. The same holds for the pressure solve that gives a chip its flow field, which iterates thousands of times.

A grid's specification is also large: a face velocity for every face, an obstacle flag for every site, reaction rates for every level. It changes rarely, when a model sets a new velocity field, reaction field or boundary value, and never during a step.

## Decision

**A specification is validated where it is set.** A `SignalGrid` validates its specification when it is constructed and whenever one of its setters changes it, and validates levels whenever they are set. A step relies on both and validates only the levels it produces, with `validate_signal_levels`, which reads them in parallel and copies nothing. The setters that replace a velocity or reaction field validate the field in place and put the old one back if it is refused.

**A grid has a revision.** `SignalGrid::revision()` is a number that differs between two grids and changes whenever a grid's specification does, and never when only its levels do.

**A grid has a layout.** `SignalGrid::layout()` holds what transport derives from the specification: the ascending list of fluid sites, the operator's diagonal at each fluid level, the least diagonal, from which the implicit operator's scale follows for any step, and the greatest outflow of the velocity field, from which the explicit stability bound follows. It is derived when first asked for after a revision and held until the next.

**Transport visits fluid levels.** On every backend the transport rates, the right-hand side, the Jacobi sweep, the residual and its norm run over the fluid sites of each signal. Arrays stay dense, one value per level, so levels, checkpoints, scenes and the Python interface are as they were; the entries at obstacle sites are zero and are not written. The residual norm sums over fluid levels and divides by the count of all levels, so its value is that of the dense norm.

- The CPU backend loops over the layout's fluid list in parallel pieces and keeps its working arrays with the grid between steps.
- The Metal and CUDA kernels are launched over `signal_count * fluid_count` threads and read their level from the fluid list on the device. Residual terms are stored one per fluid level, so their reduction is over the fluid alone.
- A kernel's output buffer must hold zero at obstacle sites, since the kernel does not write them. A new Metal buffer is zero. The CUDA backend clears a buffer when it is allocated, and both backends clear their output buffers if the obstacles on the device change.

**A device holds a specification until it changes.** The Metal and CUDA backends keep one set of buffers for the coefficients, reaction field, obstacles, face velocities, boundary values and fluid list, shared by plain and coupled steps, and record the revision they hold. A step whose grid has that revision uploads only its levels.

The coupled step's explicit kernel, which adds the cells' exchange to the grid once per step, runs over every level.

**The pressure solve iterates over active sites.** A `DepthAveragedFlowSystem` carries a form of itself over its active sites alone, those whose pressure it determines: the list of sites, each site's six neighbors as places in that list with the conductance to each, and the diagonal, right-hand side and boundary conductance in that order. Every backend's conjugate-gradient vectors are as long as that list, and the operator reads neighbors from it. The solved pressure is scattered to the full grid once, where the velocity reconstruction reads it.

**A device keeps the pressure solve's scalars.** On Metal and CUDA the step length and the direction weight of each iteration are computed on the device from dot products reduced on the device, and held in a small buffer with the iteration count and the reason the solve stopped. The host encodes a batch of iterations, waits once, and reads that buffer. A solve that meets its tolerance inside a batch takes steps of no length for the rest of it, so its result and its reported iteration count are those of an iteration checked every time. `batched_conjugate_gradient` in `core/conjugate_gradient.hpp` is the loop both backends share.

A dot product is reduced in three levels: fixed-width blocks of products, groups of 64 block sums, and one thread that adds the groups. One thread adding every block sum of a large system is slower than the rest of an iteration together.

## Consequences

On the mixer at a 12.5 micrometer spacing, one signal, measured on an Apple M4 Max:

| | CPU | Metal |
| --- | --- | --- |
| A step with no sweeps | 1.7 ms | 2.6 ms |
| Each sweep | 0.4 ms | 0.2 ms |

The fixed cost of a step is the copy of the levels into and out of the solve. On Metal a sweep is one command buffer, and its cost is the round trip more than the arithmetic.

The pressure solve of the same chip, two solves of about 9,000 iterations each over 0.13 million active columns, takes 1.8 s on the CPU backend and 2.2 s on Metal. On a grid that is nearly all fluid the two backends cost the same at about a quarter of a million sites, and Metal is faster above that.

- A model that changes its reaction field every step, as one with a cell-dependent sink does, re-derives the layout and re-uploads the specification every step. Its grid is a trap's, where that is cheap.
- A source rate at an obstacle site is ignored.
- Backends agree as before. A residual norm may differ in its last bits from a sum over the dense array, since the sum is grouped differently.
- The pressure solve's iteration count still doubles when the spacing is halved; its preconditioner is the diagonal.
- The resolved Stokes solver's own momentum and pressure systems run over the whole grid. It starts from a depth-averaged pressure, which is solved over active sites.

## Verification

- The signal-grid conformance test steps each backend through a change of velocity field, reaction field, boundary values, patch values and levels, and checks every step against a simulation that has held no other grid.
- The flow conformance test solves a grid that is mostly wall, with one winding channel, through ports on each backend, and checks the field, the port fluxes and the iteration count against the CPU backend.
- The existing unit, conformance and Python tests run unchanged.
