# ADR 0012: headless scene protocol and independent viewer

- Status: accepted
- Date: 2026-08-15

## Context

The legacy PyQt/OpenGL application owns the OpenCL device, simulator lifetime, model imports, simulation clock, pickle persistence, selection, and rendering. Models install renderers into the simulation itself. This prevents headless execution, makes renderer behavior depend on mutable implementation details, and would force CUDA and Metal concerns into the UI.

MicroSimulator needs interactive inspection on Apple and NVIDIA systems without creating a fourth simulation implementation or weakening the native backend contract.

## Decision

The simulation engine remains headless. It exposes an immutable, versioned scene frame containing only observable presentation data:

- schema name and version;
- simulation time and source backend metadata;
- rods in stable cell-ID order, including geometry and inspectable scalar values;
- optional signal-grid geometry, boundaries, and channel-major levels; and
- stable lineage parents for selection context.

The JSON interchange uses finite numbers, closed object schemas, and decimal strings for 64-bit identifiers. A frame is a projection, not a checkpoint: it contains no rate plans, solver configuration, executable source, controller state, or authority to resume a simulation.

The viewer is a separate TypeScript web application using an instanced graphics renderer. It loads scene files without server authority. Its live controller publishes the same frames and accepts only typed play, pause, step, reset, checkpoint, and frame-request commands through an authenticated loopback protocol. The controller—not the browser—constructs simulations and selects an explicit backend and device.

CUDA and Metal remain native scientific-compute APIs. Viewer graphics are not a compute backend and do not participate in numerical conformance. The CPU, Metal, and CUDA acceptance gate is equality of scene semantics for the same engine state. Renderer tests cover framing, picking, color mapping, and command state separately.

Presentation mappings are declarative viewer state. The initial mapping modes are cell type, species channel, growth rate, and fixed state. Model callbacks do not mutate RGB fields in the engine. New observable fields require a scene schema change or a typed extension, not arbitrary object traversal.

## Implementation layers

1. The scene-frame projection and strict JSON reader/writer define the data boundary.
2. The read-only viewer provides rods, grid slices, camera navigation, picking, selection details, and declarative color mapping.
3. The local controller uses the batch model and checkpoint APIs while publishing the same scene schema.
4. JSON remains the inspectable reference format. Any binary transport must preserve the same logical schema and should be justified by measurements on large colonies.

## Consequences

- Viewer work is implemented once and consumes CPU, Metal, and CUDA results through the same boundary.
- Headless batch execution has no UI dependency.
- Scene files are safe to inspect but cannot resume a run.
- Legacy sphere-cell, plant, periodic-image, collision-grid, and static-mesh renderers are explicitly retired from compatibility. A future feature in one of those domains requires a new typed engine, checkpoint, and scene proposal.
- The viewer can evolve or be replaced without changing backend kernels.
