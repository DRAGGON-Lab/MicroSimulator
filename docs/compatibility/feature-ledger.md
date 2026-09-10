# CellModeller compatibility overview

This page summarizes how familiar CellModeller capabilities map to MicroSimulator. It is intended to help researchers assess an existing model before migration; it is not a claim of bitwise parity with the original OpenCL implementation.

CPU and Metal support the current modeling surface described below. CUDA status and hardware requirements are maintained in the [testing and validation guide](../development/validation.md).

| Area | CellModeller source | MicroSimulator behavior |
| --- | --- | --- |
| Cell identity | `Simulator` | Stable IDs are distinct from compact storage slots; division records lineage. |
| Cell creation and division | `Simulator`, `CLBacterium` | Equal and fractional division use explicit, deterministic geometry and jitter policies. |
| Length growth | `CLBacterium` | The reference step is `length += rate * length * dt`. |
| Rod mechanics | `CLBacterium` | Contacts feed a diagnosed, matrix-free finite-radius mechanics solve. |
| Cell-cell contacts | `CLBacterium.cl` | Deterministic sweep-and-prune staging produces a dynamic incidence graph without a fixed contact cap. |
| Plane and sphere constraints | `CLBacterium.cl` | Typed constraint records participate directly in the mechanics system. |
| Intracellular species | `CLEulerIntegrator` | Typed rate plans use effective-volume dilution and simultaneous updates. |
| Grid signaling | `GridDiffusion` | Signal grids support diffusion, optional advection, declared boundaries, affine source/loss fields, stability checks, and checkpointed state. |
| Coupled cell-grid rates | Signal integrators | Sampling, intracellular rates, transport, scatter, and commit form one declared stage. |
| Checkpoint and resume | Pickle output | Versioned, non-executable JSON records state, provenance, integrity, and controller data. |
| Batch execution | Batch scripts | `microsimulator run` and run manifests declare backend, device, seed, parameters, stopping rules, and output policy. |
| Model orchestration | Simulator/module lifecycle | Restartable controllers own regulation, division, mechanics scheduling, model state, and runtime randomness. |
| Python callback models | Module regulator | Maintained `setup`/`init`/`update`/`divide` models run through the compatibility adapter within documented limits. |
| Pickle snapshots | `Simulator` | Trusted snapshots can be imported once into native state; exact continuation is not inferred. |
| Crank-Nicolson signaling | `CLCrankNicIntegrator` | A diagnosed native solver implements the intended semi-implicit equation. |
| Neighbor reporting | `CLBacterium` | Stable-ID neighbor views are derived from the current contact graph. |
| Fixed cells | `CLFixedPosition` | Rods can remain mechanically fixed while biological state continues to advance. |
| Neighbor diffusion | `NeighbourDiffusion` | The incomplete legacy module is not supported; contact-mediated transport requires an explicit graph-flux model. |
| SBML import | `SBMLImport` | A bounded SBML Core subset compiles to typed species-rate plans without generated source. |
| Interactive viewer | PyQt/OpenGL GUI | An independent scene consumer displays rods and signal grids without owning engine state. |
| Analysis | `Scripts` | Versioned Parquet/Zarr exports and documented dataframe recipes replace access to private solver objects. |

For source-pinned comparisons, see the [example matrix](legacy-example-matrix.md), [recorded trajectories](legacy-trajectory-evidence.md), and the subsystem references collected in [compatibility and migration](README.md).
