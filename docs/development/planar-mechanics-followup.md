# Proposed follow-up: explicit planar cell mechanics

Status: specification only; not implemented. The [planarity investigation](../tutorials/planarity.md) found expected three-dimensional responses, including Z-directed degenerate contacts and inherited tilt. `jitter_z=False` must retain its current orientation-perturbation meaning. A strict planar tutorial needs a separately selected native mechanical mode.

## Public contract and state ownership

Add a native mechanical dimensionality setting with `spatial_3d` as the backward-compatible default and `planar_xy` with an explicit finite `plane_z`. This is simulation state shared by Python, CPU, Metal, and CUDA, not a viewer option or controller-only callback. Preserve the existing three-coordinate geometry API and existing capsule length, radius, and volume conventions.

For `planar_xy`, all mobile cell centers satisfy `position.z == plane_z` and all directions satisfy `direction.z == 0`, within a documented float32 representation tolerance after every public geometry mutation and completed native stage. The plane's stored value is its canonical float32 representation. Reject non-finite or unrepresentable heights. Initialization and explicit geometry edits reject appreciably off-plane centers, directions with nonzero Z beyond the declared input tolerance, and directions with no finite nonzero XY component; canonicalize accepted roundoff to exact stored plane Z and normalized XY direction. Do not silently flatten tilted checkpoint geometry or switch modes on an occupied simulation.

## Translation, rotation, growth, and division

Solve only the two in-plane translation components and rotation about Z. Assemble/project mechanical Jacobians, forces, increments, and residuals in those degrees of freedom; projecting a completed 3D solve afterwards is insufficient because its contact response was computed in a different space. Fixed cells obey the same planar geometry validation and remain stationary.

Growth changes length without introducing Z. Division uses the planar parent axis to place daughters, keeps both centers on the configured plane, and preserves the existing volume and lineage contracts. A model requesting an XYZ geometry edit or incompatible division jitter must receive a clear validation error, so a mistakenly configured tutorial does not appear to work through silent filtering. XY-only or absent division jitter works unchanged.

## Contacts and external boundaries

Compute planar closest-segment contacts and normals in XY. Crossing rods and coincident parallel rods must receive deterministic in-plane separating directions instead of the 3D cross-product/fallback directions. Specify canonical axis signs, cell-ID ordering, tie breaks, and normal sign conventions centrally; test cell order reversal and both equivalent representations of a rod axis. The same overlapping input and solver parameters must produce matching contact identities and tolerance-equivalent corrections on all available backends.

The first implementation should support Z-extruded lateral planes, axis-aligned boxes, and Z-aligned cylinders whose planar cross-sections are well-defined and whose finite-height caps fully clear a capsule on the chosen plane. Validate this compatibility when adding constraints and when adding or resizing cells. Reject tilted planes, spheres, or cap intersections until their planar mechanical semantics are explicitly implemented. A floor/ceiling touching a planar capsule must not create an unsatisfiable out-of-plane force. This mode models discs/capsules constrained to a plane within a compatible device; it does not replace finite-height 3D confinement.

## Flow and chemical fields

Project sampled drift velocity onto XY before applying mobile-cell translation. Deliberately ignore the normal component as a kinematic constraint and document that this is not a resolved reaction-force or momentum-conservation model. Preserve in-plane interpolation, obstacle handling, boundary behavior, and fixed-cell exclusions. Keep signal grids and chemistry independently three-dimensional: a 3D concentration field may be sampled at `plane_z`; a shallow grid does not automatically opt the cells into planar mode.

## Checkpoint, resume, and diagnostics

Version the native checkpoint schema to persist dimensionality and canonical plane height. Old checkpoints migrate explicitly to `spatial_3d`; missing or invalid fields in the new schema fail validation. Validate planar geometry and compatible constraints before exposing a restored simulation, without silently repairing incompatible data. Preserve mode and plane across CPU/Metal/CUDA restoration, controller restart, and clone/export paths. Reject a resume request whose requested mechanics mode conflicts with saved state.

Expose the mode and plane in scene/analysis metadata so diagnostics can distinguish an intended invariant from a visual appearance. Coordinate that schema change with the existing metadata owner; do not infer dimensionality from channel labels, camera view, signal-grid depth, or jitter configuration.

## Acceptance and validation

- Shared native fixtures cover separated cells, overlapping parallel rods, crossing rods, order-reversed pairs, arbitrary in-plane orientations, division, long growth runs, and mixed mobile/fixed cells. Check both center Z and direction Z after each relevant stage, including zero-duration controller steps.
- Test all public construction and geometry-edit paths: accepted roundoff canonicalizes consistently; inherited tilt and invalid heights fail clearly; default 3D behavior and its existing conformance fixtures remain unchanged.
- Verify planar contacts separate overlap in XY with finite residuals and deterministic identities, including degenerate ties. Check force/rotation consistency and convergence rather than only final projection onto the plane.
- Exercise each supported boundary, each rejected incompatible boundary, cell growth approaching a cap, and a prescribed flow with nonzero Z velocity. In-plane drift is preserved and out-of-plane drift is suppressed only in planar mode.
- Round-trip checkpoints with each mode, migrate old data, reject malformed/inconsistent planar state, and compare resumed trajectories against uninterrupted execution. Run the same fixture contract on CPU, Metal, and CUDA; report unavailable accelerator hardware instead of substituting CPU.
- Add an opt-in tutorial using the new mode and a paired finite-height 3D example. Documentation explains the distinct mechanical assumptions and retains the current definition of `jitter_z`.
