# Division jitter, confinement, and out-of-plane motion

MicroSimulator's native cell mechanics is three-dimensional. An initially planar colony and XY-only division jitter can remain planar in a particular run, but neither imposes a planar constraint. A shallow flow calculation or a one-layer signal grid does not change the mechanical degrees of freedom.

`UniformLengthDivision(jitter_z=False)` draws the usual three random perturbations and replaces the new Z perturbation with zero. It adds the resulting vector to each daughter's inherited direction, then the engine normalizes that direction. An inherited nonzero Z component therefore survives, and its normalized value can change when X or Y changes. Daughter centers are placed along the parent's three-dimensional axis before jitter is applied. `jitter_z=None` disables orientation jitter entirely; `jitter_z=True` includes the new Z perturbation. All three settings leave subsequent contact relaxation and flow drift three-dimensional.

## Tutorial audit

The following contracts describe the checked-in models after the founder-initialization correction in issue #22. “XY-only” refers to the added division perturbation, not a mechanical constraint. No model below implements strict 2D mechanics.

| Model or scenario | Initial plane and division jitter | Mechanical confinement and dimensional contract |
| --- | --- | --- |
| `biophysics.py`: `basics`, `two_types`, `competition` | Centers Z=0, directions in XY; XY-only jitter | No walls. All translations and rotations remain 3D. |
| `biophysics.py`: `short_cells` | Centers Z=0; XYZ jitter | No walls. Orientation leaves XY at division by design. |
| `biophysics.py`: `box` | Centers Z=0.5; XYZ jitter | Floor at Z=0 and four lateral walls; no ceiling. This is an open 3D box. |
| `gene_expression.py`: all five scenarios | Centers Z=0, directions in XY; XY-only jitter | No mechanical walls; unrestricted 3D. |
| `signaling.py`: `single_gene`, `communication` | Centers Z=0, directions in XY; XY-only jitter | Only lateral Y walls at ±16; no Z confinement. |
| `signaling.py`: `mutualism` | Centers Z=0, directions in XY; XY-only jitter | No mechanical walls; unrestricted 3D. |
| `simbol_circuits.py`: all six circuits | Centers Z=0, directions in XY; XY-only jitter | No mechanical walls; unrestricted 3D. |
| `plasmid_segregation.py`, `conjugation.py` | Centers Z=0, directions in XY; no orientation jitter | Division inherits the parent axis; contact mechanics remains unrestricted 3D. |
| `pillar_channel.py` | Centers Z=0, directions in XY; XY-only jitter | Box walls at Z=±3 and cylindrical pillars. Attached cells are fixed; released daughters move in a finite-height 3D chamber. |
| `danino_clock.py`, `examples/microfluidic_trap.py` | Centers Z=0, directions in XY; XY-only jitter | Trap walls at Z=±3. These permit nonzero Z and tilt within a finite-height 3D device. |
| `biopixel_trap.py` | Centers Z=0.825, directions in XY; XY-only jitter | Trap floor Z=0 and roof Z=1.65; the channel is taller. The thin trap encourages a monolayer but does not force identical center Z or zero tilt. |
| `examples/culture_dish.py` | Centers Z=0, directions in XY; XY-only jitter | Inside cylinder with caps at Z=±1. Finite-height confinement permits nonzero Z and tilt. |

“Monolayer” in the device tutorials describes a physical modeling intention supported by a thin cavity. Walls are soft numerical constraints, and residual overlap depends on the contact solver tolerance and number of relaxation passes. Their presence does not guarantee exact wall bounds after one pass or eliminate out-of-plane degrees of freedom.

## Reproduce and locate a departure

Run the diagnostic from the repository root after installing the development environment:

```console
uv run python scripts/diagnose_planarity.py --backend cpu --seed 17 --dt 0.02 --output results/planarity-fixtures.json
uv run python scripts/diagnose_planarity.py --backend cpu --seed 17 --dt 0.02 --model examples/tutorials/biophysics.py --scenario basics --steps 1000 --max-cells 128 --output results/planarity-basics.json
uv run python scripts/diagnose_planarity.py --backend metal --seed 17 --dt 0.02 --model examples/tutorials/biopixel_trap.py --plane-z 0.825 --steps 1000 --max-cells 128 --output results/planarity-biopixel.json
```

Use `--plane-z 0.5` for the box scenario. Other model parameters accept JSON, for example `--parameter circuit='"bba_0001"'`. The script rejects unavailable backends instead of falling back. A division batch can exceed `--max-cells`; that limit stops the next biological step rather than truncating the model's division requests.

The JSON records the backend, seed, timestep, model path and supplied parameters, model/controller state, source provenance, mechanics settings, initial and final geometry, and constraints. Every stage records maximum absolute center Z, displacement from the selected reference plane, absolute direction Z, and the change in each Z component for surviving cell IDs. The first event exceeding `1e-6` includes before/after geometry. This is a diagnostic detection tolerance, not a global engine guarantee. For an initially tilted cell, the first event is initialization; later stages still report their changes.

The probe wraps native Python entry points for the selected simulation in a dedicated process and restores them in `finally`. It observes division, geometry edits, removal, growth/chemistry, flow drift, and contact/constraint relaxation without duplicating the controller or drawing random numbers. Contact and wall relaxation use the same native entry point: the isolated contact-only and wall-only fixtures distinguish their causes. Arbitrary native work performed internally by a custom extension is outside these Python stage boundaries. Use this as a headless diagnostic, not inside a multithreaded application.

## Findings and classification

The seven fixtures in `scripts/diagnose_planarity.py` and their shared-backend assertions in `python/tests/test_planarity.py` establish the following causes independently of any long colony trajectory. The measurements below were reproduced on CPU and Apple Metal with seed 17; NVIDIA CUDA hardware was unavailable for this investigation. The same tests select CUDA when its runtime is available.

| Fixture | Observation | Classification |
| --- | --- | --- |
| Separated planar rods | Two nonoverlapping X-oriented rods retain center Z=0 and direction Z=0 through relaxation. | Planar control case; no departure. |
| Crossing rods | Two rods at the same center, directions X and Y, centerline length 4 and radius 0.5, produce a Z-directed normal. One relaxation moves their centers to approximately ±0.4. | Expected 3D contact response. Crossing rods at zero centerline separation are an overlapping initialization if intended as a nonoverlapping planar colony. |
| Coincident parallel rods | Two identical X-oriented rods at one center use the deterministic degenerate-contact fallback; the normal points in Z and relaxation separates centers to approximately ±0.4. | Expected 3D degeneracy handling, also an overlapping initialization. |
| Planar division | An intentionally oversized planar parent divides with XY-only jitter without creating a Z component. | Planar division control; no departure. The fixture intentionally bypasses tutorial founder capping. |
| Inherited tilt | A parent initialized with direction `(1, 0, 0.2)` has normalized Z≈0.196116. Division places daughters at Z≈±0.245145. Both geometry-edit calls request exactly zero added Z, although native normalization changes the daughters' direction Z. | Expected inherited geometry and normalization; an initially tilted setup cannot test preservation of a planar state. |
| Finite-height walls | A horizontal radius-0.5 rod at Z=0.8 intersects the ceiling at Z=1. One relaxation moves its center to approximately 0.6. Repeated default solves stop with approximately 0.003704 penetration; tightening the residual tolerance to `1e-8` reduces penetration below `1e-6`. The center remains near Z=0.5. | Expected soft-wall convergence within a finite-height 3D space. Default residual tolerance is `0.005`; physical confinement neither implies exact wall projection nor the plane Z=0. |
| Vertical flow | Prescribed Z velocity 0.2 over dt=0.02 moves a center from Z=1 to approximately 1.004, with no division. | Expected 3D advection. XY-only jitter does not filter flow. |

For zero closest-point separation, the CPU, Metal, and CUDA contact implementations first try the cross product of rod axes, then transverse center separation, then a deterministic perpendicular fallback. Two crossing XY axes have a Z-directed cross product; coincident X axes use a fallback that can also point in Z. Choosing these directions is consistent with the existing 3D contact contract. Suppressing them globally would change three-dimensional mechanics.

Representative tutorial runs used seed 17, dt=0.02, a maximum of 1000 steps, and a stop threshold of 128 cells. CPU and Metal reached the same step/count endpoints below; agreement here is not a claim of bitwise trajectory equivalence. Geometry was checked after each instrumented stage, including both center displacement and direction Z.

| Model/scenario | Reference Z | Completed steps / final cells | First departure |
| --- | ---: | ---: | --- |
| Biophysics: basics | 0 | 312 / 128 | None above diagnostic tolerance |
| Biophysics: two_types | 0 | 149 / 128 | None above diagnostic tolerance |
| Biophysics: competition | 0 | 351 / 130 | None above diagnostic tolerance |
| Biophysics: short_cells | 0 | 325 / 128 | Division geometry edit at time≈0.02: direction Z≈0.000531725 |
| Biophysics: box | 0.5 | 327 / 128 | Division geometry edit at time≈0.02: direction Z≈0.000531725 |
| Gene expression: constitutive | 0 | 327 / 128 | None above diagnostic tolerance |
| Signaling: single_gene | 0 | 179 / 128 | None above diagnostic tolerance |
| Pillar channel | 0 | 394 / 131 | None above diagnostic tolerance |
| Biopixel trap | 0.825 | 478 / 128 | None above diagnostic tolerance |

The two departures in this matrix are the tutorials' intentional XYZ jitter. The source audit covers additional scenarios without claiming that they all received these trajectory runs. The unchanged plane in the other runs provides reproducibility evidence for those particular initial states and durations, not proof of planar invariance.

No solver defect was demonstrated. Issue #22 separately corrected oversized tutorial founders, which could previously divide immediately; the diagnostics here use that corrected initialization. Luiza's older-version observation cannot be assigned to one particular mechanism without its original model, geometry, seed, and trajectory. The fixtures nevertheless show several reproducible paths to Z motion even with `jitter_z=False`.

If a lesson requires every center to remain at a specified Z and every axis to remain in XY, it requires an explicit planar mechanics feature. The [proposed planar-mechanics contract](../development/planar-mechanics-followup.md) defines that follow-up without changing the meaning of division jitter.
