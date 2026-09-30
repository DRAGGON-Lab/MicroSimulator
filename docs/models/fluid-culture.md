# Fluid mechanics, solute transport, and cell growth

`Simulation.configure_culture` composes physical Stokes flow, hydrodynamic cell motion, geometric extracellular transport, and surface exchange on CPU, Metal, or CUDA. Seed cells first, then configure this mode before the first step. `examples/growing_media.py` is a complete perfusion-driven growth example; `examples/physical_flow.py` exercises the standalone hydraulic solver.

```sh
uv run python examples/growing_media.py --backend cpu --output /tmp/culture-example
```

Select `metal` or `cuda` explicitly to use that backend. An unavailable device, failed solve, unresolved geometry, or exhausted resource limit raises an error. There is no backend fallback. The current implementation is suitable for numerical evaluation and refinement studies; large-colony throughput and quantitative cell-drag accuracy have not been established.

## Units and device geometry

`FluidGridSpec` stores a three-dimensional uniform grid. `origin` is the center of voxel zero, `spacing` is a model length, and obstacle entries are Boolean wall voxels in x-major/z-minor order. `length_unit_m` and `time_unit_s` convert model lengths and times to SI. Hydrodynamic viscosity, density, pressure, applied force, torque, and total port rate use Pa s, kg/m³, Pa, N, N m, and m³/s. Solute diffusivity uses model length²/model time. Concentrations use an explicitly chosen amount unit per model volume; the same amount unit applies to a solute and its bound intracellular species.

Each `FlowPort` selects a patch on one outer grid boundary. Empty `sites` selects all fluid faces on that boundary. A pressure port specifies reservoir pressure; a rate port specifies total signed volume flow, positive outward. Remaining boundary faces and obstacles are no-slip walls. Each disconnected fluid component has its own pressure reference. A component with only rate ports must balance prescribed net outflow against cell growth. A sealed incompressible chamber cannot accommodate net cell growth.

Cells are analytic capsules with a cylindrical length and hemispherical caps. Their hydrodynamic pose uses a persistent quaternion. Closed-wall gaps are bounded by `stepping.minimum_gap_m`; singular lubrication is not resolved. A radius of two grid spacings is only a representation minimum. Quantitative results require independent fluid-grid, surface, gap, and time refinement. Cells must remain inside the device, and their immersed-boundary support must not reach an open port. Cell passage through outlets is not implemented.

## Amounts, biomass, and growth

A `Medium` is an immutable composition of named solutes, with explicit concentration units and optional provenance. Each `Solute` declares a stable identity, amount unit, and diffusivity. Nutritional requirements belong to the cell's growth model. The same transported chemical can be a nutrient for one organism and inert for another.

Each port has one hydraulic drive and one chemical boundary. `AdvectiveFeed` supplies its composition with inflowing fluid and supplies nothing at zero volume rate. `ReservoirContact` also permits diffusion to an external bath. `ConvectiveOutflow` uses interior concentration with no diffusive flux; backflow requires an explicit external composition or rejects the step. `PiecewiseConstant` schedules can change hydraulic drives and compositions independently. Stepping splits exactly at scheduled event times, and restart preserves the event cursor.

`MonodGrowth` consumes one limiting substrate. `EssentialNutrientGrowth` requires all declared nutrients, using the minimum limitation and a shared stoichiometric growth extent. `BiomassConversion` connects dry biomass in grams to biochemical volume and geometric capsule volume. The coupled solver iterates geometry, surface concentration, uptake, and growth until they agree. Only accepted nutrient uptake can produce biomass. Exported growth rates describe the realized volume change over the accepted public step.

Reserve-budget growth remains available through `ReserveBudgetGrowth` and the native `ReserveRequirement`/`CellSurfaceExchange` records. Reserve growth consumes intracellular material present at the beginning of a substep; imported material becomes available on the next substep. It cannot be combined with kinetic growth in the same configuration. Intracellular reaction plans use actual biochemical volume and its change rate. Native rate-plan arithmetic retains its existing float32 precision; zero reaction plans preserve stored amounts through dilution.

`culture_checkpoint` returns a copy containing persistent poses, biochemical volumes, intracellular amounts, extracellular fragment amounts, cumulative reservoir transfers, and the last solve reports. `fluid_fragments` describes the matching extracellular volumes. Amount arrays are fragment-major and solute-minor. `cell_surface_concentrations(id)` returns an area-weighted membrane concentration. Legacy signal-grid reaction plans and geometric mechanics are separate simulation modes and cannot be combined with the coupled culture model.

## Geometry and integration

The extracellular mesh uses convex capsule surfaces whose cross-sectional area and total volume equal the analytic capsule values. The reported surface approximation error decreases with `geometry.surface_resolution`. Actual polyhedral clipping determines fluid volumes and open face areas. Face-connected pieces within a voxel are combined; disconnected pieces remain separate. A gap smaller than the surface error is rejected and requires refinement. Thin positive-volume pieces are retained during clipping.

Old and new fragments are joined within a voxel only through positive geometric overlap. The resulting temporal control volumes retain closing and opening fragments. Endpoint face areas define the time-averaged apertures, and a constrained flux projection enforces the change in extracellular volume while retaining prescribed total port rates. Residual refinement controls local volume equations in addition to the global Krylov norm. Conservative backward-Euler advection/diffusion and implicit uptake advance amounts. Reports expose geometric-conservation and total-amount residuals. This is a first-order time scheme; a small solver residual does not establish spatial or temporal accuracy.

`Simulation.step(dt)` stages the complete interval. Body proposals bound surface displacement to at most one quarter of a spacing. Bounded retries reduce failed numerical substeps, and any unrecoverable failure restores cells, chemistry, amounts, ledgers, and time. A culture-enabled `NativeController` also restores its explicit JSON state, RNG, lifecycle changes, and step counters if regulation or integration fails. Arbitrary external effects performed by user callbacks are outside that transaction.

Division splits analytic capsule volume, biochemical volume, and intracellular amounts according to the requested fraction. The daughter cylinders have lengths `f*(L + 4*r/3) - 4*r/3` and `(1-f)*(L + 4*r/3) - 4*r/3`. Negative lengths or daughters that do not fit are rejected. An amount-preserving geometric remap updates extracellular fluid without a septation-flow transient or reservoir transfer. Daughter placement can require more axial room than the parent. Instantaneous cell removal and manual pose edits are rejected because they do not provide a conservative displacement model.

## Storage and inspection

Checkpoint version 11 stores fluid and chemical configuration, kinetic models, schedules, poses, biochemical and extracellular amounts, uptake and boundary ledgers, recipe provenance, and diagnostics. Versions 1–10 continue to load; version 10 migrates to explicit reserve-budget semantics. The polyhedral mesh is reconstructed and checked on restore. Restart tests compare continued and restored trajectories.

Scene version 5 includes physical units, biochemical volumes, dry biomass, realized growth rates, uptake totals, fragment volumes and amounts, and boundary ledgers. Versions 2–4 remain readable. The viewer displays fluid-volume-weighted voxel concentration means and labels them accordingly. Those means do not merge the authoritative fragments. Flow diagnostics describe the last fluid solve, whose geometry is the beginning of its substep.

Analysis version 5 writes `culture_frames.parquet`, `culture_cells.parquet`, `fluid_fragments.parquet`, and `chemical_transfers.parquet`. These preserve binary64 amounts, poses, volumes, units, cumulative transfers, and biological growth diagnostics. Version-4 tables remain readable by the canonical table names. `signals.zarr` contains the presentation concentration means. Use fragment amounts and volumes for conservation calculations.

## Backend and validation boundaries

Geometry, sparse coefficient assembly, and factorization setup run on the host. CPU linear algebra uses binary64. Metal and CUDA use native binary32 Krylov vectors, sparse operators, preconditioner application, and reductions, with host scalar control and checks against the original operator. Scalar defect correction retains a double-precision authoritative solution; correction solves execute on the chosen device. Coupled-body preconditioning currently applies triangular factors serially on the selected device; it is a performance limitation. Resource estimates are checked before the principal allocations.

Conformance tests cover pressure/rate ports, disconnected pressure references, a square-duct refinement limit, immersed-boundary identities, force balance, no-slip residuals, contact, growth displacement, polyhedral volume/area closure, diffusion eigenmodes, moving-volume uniform concentration, surface-transfer budgets, restart, and rollback. Actual Metal execution requires `metal_runtime_gate`; CUDA compilation is distinct from execution on an NVIDIA device. Dense contact networks, large-colony scaling, and quantitative drag/lubrication refinement remain scientific validation work.

See [code ownership and interfaces](../architecture/0027-fluid-transport-biology-code.md) for the numerical, fluid, chemical, and biological layer boundaries.
