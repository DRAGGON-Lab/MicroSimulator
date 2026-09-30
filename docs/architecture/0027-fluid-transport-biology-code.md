# Fluid mechanics, chemical transport, and growth: code ownership

The simulation composes three independent physical and biological operations. Fluid mechanics determines velocity and moving-body geometry. Chemical transport uses that geometry to advance conserved solute amounts. Growth models convert accepted nutrient uptake into biomass and geometric growth. `CultureSimulation` couples those operations transactionally; biological medium recipes remain Python authoring data.

## Native ownership

| Layer                        | Public types                                                                                      | Implementation                                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Grid                         | `GridShape`, `FlowAxis`, `MacVelocityField`                                                       | `cm/grid.hpp`                                                                                             |
| Numerical settings           | `LinearSolveParameters`                                                                           | `cm/numerics.hpp`, `core/numerics.cpp`                                                                    |
| Sparse algebra and devices   | `CsrMatrix`, `IncompleteLu`, `MultigridLevel`, `SparseMultigrid`, `NumericsDevice`                | `core/numerics_*.hpp`, `cuda/cuda_numerics.cu`, `metal/metal_numerics.mm`, `metal/kernels/numerics.metal` |
| Fluid mechanics              | `FluidGridSpec`, `FluidProperties`, `FlowPort`, `CapsuleBody`, `StokesFlowSolver`                 | `cm/stokes_flow.hpp`, `core/stokes_flow.cpp`                                                              |
| Capsule geometry and contact | `CapsuleContact`, `CapsulePolyhedron`, `ConvexPolyhedron`, `FluidGeometry`                        | `core/capsule_contacts.hpp`, `core/convex_polyhedron.hpp`, `core/fluid_geometry.cpp`                      |
| Geometric transport          | `TransportGeometry`, `GeometricFluxReport`                                                        | `cm/transport_geometry.hpp`, `core/transport_geometry.cpp`                                                |
| Chemical transport           | `Solute`, `ChemicalBoundary`, `SurfaceTransferLaw`, `SurfaceEnvironment`, `SoluteTransportSolver` | `cm/solute_transport.hpp`, `core/solute_transport.cpp`                                                    |
| Biological growth            | `CellGrowthModel`, `GrowthRequirement`, `GrowthExecutor`                                          | `cm/growth.hpp`, `core/growth.cpp`                                                                        |
| Coupled simulation           | `CultureConfiguration`, `CultureEvent`, `CultureCheckpoint`, `CultureReport`                      | `cm/culture_simulation.hpp`, `core/culture_simulation.cpp`                                                |

Sparse storage, incomplete factorization, device allocation, and linear solves do not include the fluid solver or biological headers. `fluid_multigrid.hpp` constructs the velocity hierarchy from MAC coordinates; the generic multigrid representation and cycle live in `numerics_multigrid.hpp`. The fluid API has no solute or growth-model dependency. The chemical transport API has no biological medium or growth-model dependency. `GrowthExecutor` consumes concentrations and accepted uptake through a small numerical input record; it does not own the fluid domain or transport solver.

The existing signal-grid API retains `SignalGridVelocityField` as an alias of `MacVelocityField`. Established shallow-flow and Stokes–Brinkman models retain their existing semantics.

## Python authoring

`flow.py` defines fluid properties, domain patches, pressure and volume-rate drives. `stokes.py` provides `StokesFlow`. `transport.py` defines solutes and chemical boundary conditions. `media.py` defines immutable `Medium` compositions and concentration units. `growth.py` defines organism-specific kinetics, nutrient yields, and biomass conversion. `schedules.py` provides piecewise-constant schedules. `culture.py` compiles these definitions to native records with stable solute indices and explicit unit conversion.

```python
simulation.configure_culture(
    fluid=StokesFlow(domain, properties, hydraulic_boundaries),
    transport=SoluteTransport(solutes, chemical_boundaries, initial_medium),
    cell_growth={cell_id: CellGrowth(kinetics, biomass_conversion)},
)
simulation.step(dt)
state = simulation.culture_state
```

The complete runnable configuration is [growing_media.py](../../examples/growing_media.py). It starts with nutrient-free fluid and no intracellular nutrient reserve. Feed composition, volume rate, essential nutrient requirements, and a scheduled medium switch are explicit inputs. Medium composition does not prescribe a cell growth rate.

`Simulation.configure_culture` is the Python authoring entry point. Native callers pass `CultureConfiguration` to `cm::Simulation::configure_culture`; Python exposes that record-based entry point as `_configure_culture`. `culture_state` is an immutable, named Python inspection snapshot. `culture_checkpoint` returns a copy of native restart state. `fluid_fragments` and `cell_surface_concentrations` expose geometry and membrane-local chemical state. `set_cell_force` changes the external force and torque on a coupled body.

## A coupled step

1. Split the requested interval at the next scheduled boundary event.
2. Stage cell poses, biochemical volumes, extracellular amounts, and transfer ledgers from the accepted old state.
3. Propose growth rates and a constrained Stokes body step; reject unresolved geometry or invalid contact.
4. Build `TransportGeometry` from old/new fluid fragments, temporal overlaps, apertures, and conservative face fluxes.
5. Evaluate surface uptake coefficients, solve implicit transport, and evaluate biomass production from accepted uptake.
6. Iterate until growth, uptake, and geometry agree within the coupling tolerance. Every trial starts from the same accepted old amounts.
7. Commit all state together. A failed trial reduces the substep; exhaustion restores the whole public step.

A Monod model uses one substrate. An essential-nutrient model uses a shared growth extent limited by all required nutrients and their yields. Nutrient uptake and biomass increments share the same ledger; excess accepted uptake from the converged iteration is returned to adjacent extracellular fragments. Reserve-budget growth remains a separate explicit model. Native configuration rejects mixing reserve and kinetic growth mechanisms in one culture.

Scheduled events store complete hydraulic and chemical boundary snapshots and a restartable event cursor. Schedules are compiled at configuration time. Division partitions extensive stores and uptake totals and inherits the parent's kinetic model. Biochemical volume and geometric capsule volume remain distinct quantities connected by the configured conversion.

## Persistence and device execution

Checkpoint version 11 writes `simulation.culture` and `configuration.fluid`. The version-10 reader recognizes the old `media_flow` and `medium` keys after verifying the original digest, then supplies explicit reserve-growth defaults. Scene version 5 writes `frame.culture`; version 4 remains readable through its original schema. Analysis version 5 writes `culture_frames`, `culture_cells`, `fluid_fragments`, and `chemical_transfers` tables. Version-4 datasets retain their original filenames and authenticated identities; readers resolve canonical names to those files.

CPU algebra uses binary64. Metal and CUDA execute sparse operations, preconditioner application, reductions, and growth evaluation on the selected device. Host code owns geometry, matrix assembly, factorization setup, authoritative amounts, convergence control, and transactions. Scalar defect correction checks the original double-precision operator while executing correction solves on the selected device. Device unavailability and nonconvergence raise errors; no CPU fallback is selected.

Native conformance tests cover each layer and their coupling. Python tests cover authoring, conserved uptake, schedules, restart, division, rollback, and old-format readers. CUDA compile/link checks establish build compatibility; numerical CUDA parity requires execution on NVIDIA hardware. Model limitations and refinement requirements are described in the [fluid and culture guide](../models/fluid-culture.md).
