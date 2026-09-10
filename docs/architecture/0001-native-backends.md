# ADR 0001: independent native Metal and CUDA backends

- Status: accepted
- Date: 2026-08-15

## Context

Legacy CellModeller couples simulation modules through PyOpenCL contexts, queues, arrays, and raw buffer handles. A direct translation would reproduce those ownership boundaries and keep important spatial and reduction work on the host.

MicroSimulator must run the same scientific model on Apple GPUs through Metal and on NVIDIA GPUs through CUDA while allowing each platform to use its native execution and memory model.

## Decision

The shared engine is C++23. It owns model semantics, state transitions, execution ordering, events, observations, and checkpoints.

There are three independent compute backends:

1. CPU: readable reference behavior and numerical diagnosis.
2. Metal: Metal-cpp host implementation and MSL kernels.
3. CUDA: CUDA C++ host implementation and CUDA kernels.

The backend contract exposes domain operations such as advancing growth, constructing contacts, solving mechanics, and advancing fields. It does not expose a generic kernel launcher or a supposedly portable device buffer.

Static mechanics and field kernels are handwritten for each GPU. A typed biological model representation may generate native CUDA and MSL rate functions so model authors do not have to duplicate algebraic equations.

No backend may silently fall back to another backend. Unsupported backends fail during construction with a diagnostic that names the missing capability. Backends enumerate their native devices, accept an explicit zero-based device index, and report the selected index with the device name. CUDA reactivates its owned device before every operation so multiple simulations cannot accidentally inherit another instance's process-thread selection.

## Consequences

- Kernel implementations are duplicated intentionally.
- Scientific definitions, fixtures, invariants, and tolerances are shared.
- Each vertical feature slice must pass CPU, Metal, and CUDA conformance before it can be marked backend-complete.
- Metal and CUDA may fuse and schedule operations differently.
- Cross-device bitwise equality is not required; discrete event equality, specified tolerances, residual criteria, and statistical validation are.
