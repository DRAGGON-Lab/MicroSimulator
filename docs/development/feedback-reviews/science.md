# Independent science review — round 1

This is an archived independent review. See the [round summary](../feedback-review.md) for final heads, corrections and combined validation. Temporary evidence paths in this report identify local artifacts; they are not repository dependencies.

Reviewed issues **13, 14, and 22**, corresponding to **PRs 31, 35, and 26**. No material correctness or acceptance finding was confirmed. No production, test, or documentation file in an issue worktree was edited, and no GitHub action was taken.

## Review identity and scope

| Issue / PR | Exact reviewed head | Diff base |
| --- | --- | --- |
| 13 / 31 | `4ccef4f3a95f02b810fa18eb76b75e0e9ff7a80e` | `b69193b9db034b4ccd484920de3068cc11c3ea2d` |
| 14 / 35 | `63adb45c02f79f045df35539164385c0e8290af0` | merge-base with issue 22: `6e544e3380556ea60282dda17f38075deff4b573` |
| 22 / 26 | `2d30fc7fde33abe06ddf0b05910c09d7d594e3f7` | `b69193b9db034b4ccd484920de3068cc11c3ea2d` |

Read the original assignment packets before implementation and evidence maps, then read the background `feedback/luiza.md`. Inspected the branch-specific source, tests, and documentation, the native transport/contact/controller contracts they refer to, and the integration occupancy implementation at `5514d99a565b1911c2a8a204e885ebd37ad6476a`. The founder implementation inherited by issue 14 is attributed to issue 22. The extra head on issue 22 is the founder-test formatting commit; it does not change the founder implementation inherited by issue 14.

Issue 13 is a design and executable-reference contribution. Issue 14 is an investigation and dimensionality specification. Missing production porosity kernels or strict planar mechanics are their explicitly stated follow-up scope, not failures of these issues.

## Findings

**None.** There are no confirmed P0–P3 findings, required source corrections, or blockers in this lane. The limitations below distinguish what the evidence proves from later native validation requirements.

## Issue 13 / PR 31 acceptance map

| Original criterion | Independently verified evidence |
| --- | --- |
| ADR selects a model, governing balance, discrete update, units, limitations | `docs/architecture/0025-cell-occupied-volume.md:7–56` chooses coarse capsule-union porosity, authoritative amount `N=W*c`, harmonic face closure, backward Euler, and an intrinsic-velocity convention. The amount flux has units A/T; aperture is multiplied into the volumetric flux once. The limitations explicitly include unresolved fluid passages and displacement flow. |
| Exact occupancy state and recomputation timing | ADR lines 15–23 and 58–73 list centers, directions, length, radius, lattice, wall mask, quadrature and cutoff; regulation/topology, post-growth, and post-drift/mechanics barriers; atomic commit/rollback; and zero-time edits. These are specified future controller changes, not claimed current implementation. |
| Executable CPU empty/partial/unequal/changing/division/removal cases | All eight `python/tests/test_occupancy_reference.py` tests pass. Cases begin at lines 21, 40, 53, and 74. Geometry uses actual capsule distance and union occupancy, not biochemical biomass deposition. Division test verifies the different geometric and biochemical volume balances. |
| Solute amount accounting with boundary/reaction terms | `occupancy_reference.py:234–299` assembles equal-and-opposite internal face terms, signed reservoir terms and implicit loss; returns a separate amount ledger. Checked-in test at line 109 passes. Independent 250-case mixed-flux probe reconstructs every local balance from final concentrations; maximum local residual 1.9762e-14 and global ledger residual 3.1975e-14. |
| Fully occupied regions without division by zero/lost solute | `occupancy_reference.py:108–169` rejects nonzero amount at zero storage, redistributes closing donors by new volume within the old/new accessible union component, excludes persistent closed separators, and rejects a closing component lacking recipients. Existing atomic-input-preservation tests pass. Independent six-voxel topology probe returned exactly `[1.5,0,4.5,0,0,9]`, preserving both separated component totals. |
| Empty-cell limit recovers existing transport | Checked-in native CPU backward-Euler comparison passes. Independently compared six anisotropic-grid cases spanning fixed, no-flux and periodic boundaries with both advection signs. Max concentration discrepancy 5.2027e-6; all pass the ADR's general native comparison gate. Independent residual checks agree with native solve reports to float32 rounding scale. |
| Numerical tolerances and spatial/timestep refinement | ADR lines 81–96 defines scaled float64 balance checks, proposed float32 ledger/drift/error gates, h/m/dt/cutoff studies and fixed physical support. Tests at lines 134 and 164 pass for first-order time refinement, second-order empty-grid spatial operator and sphere quadrature. No unsupported convergence-order claim is made for moving/occupied geometry. |
| Compatibility/checkpoints/CPU/Metal/CUDA/coupling work | ADR lines 98–104 preserves occupancy-disabled old checkpoints; specifies explicit amount-preserving conversion, versioned authoritative N/W and geometry metadata, and native state, rasterization, transport, staging, exchange, flow and independent GPU implementation work. Native checkpoint code is unchanged. |
| Modeling guide distinguishes approximation from resolved flow | `docs/microfluidics.md:34–36` identifies current full-voxel behavior and the proposed coarse model, linking the ADR and explicitly excluding resolved cell-surface flow and displaced-fluid motion. |

Additional source checks: parent/daughter geometric volume difference is correctly `2*pi*r^3/3` while biochemical B is conserved. The cell exchange rule has normalized accessible-volume weights, amount-rate scatter and exactly one division by W. The support construction and affordable uptake solve are explicitly deferred native work. An independent axis-swap geometry probe gives exactly equal porosity.

### Issue 13 numerical limits and a rejected suspected finding

The simple checked-in three-site diffusion fixture's `rtol=2e-6, atol=2e-7` tolerance is specific to that fixture. Reusing it unchanged for the independent anisotropic fixed-boundary/advection case initially produced a comparison failure (1.5705e-6 error at concentration 0.529776, relative error 2.96e-6). Investigation found no new operator or units error. Production `SignalSolveParameters` already defaults to relative residual tolerance `1e-5` in `cpp/include/cm/signals.hpp:26–32`; its solver at `cpp/core/signals.cpp:873–903` scales by the initial residual and adds a float32 representation floor. Reconstructed native residual RMS ranges from 2.4243e-6 to 4.2199e-6 versus reported values 2.3684e-6 to 4.2446e-6. The six cases pass the ADR's broader `rtol=2e-4, atol=2e-6` proposed native gate. Largest native amount-ledger residual in these order-one cases was 1.0484e-5 absolute, less than 1e-6 relative to the initial amount; float64 reference residuals were at most 1.7764e-15. This is inherited native solve tolerance, not a defect introduced by PR 31.

The reference is dense, float64 and small-case-only. No production occupancy transport, atomic geometry staging, cell-support search, checkpoint migration, weighted-flow solve or GPU porosity path was tested because those paths do not yet exist in this scoped contribution. Porous-domain scientific convergence, long native drift, cutoff variation and displacement-flow fidelity remain follow-up gates; empty-grid spatial convergence cannot stand in for them.

## Issue 14 / PR 35 acceptance map

| Original criterion | Independently verified evidence |
| --- | --- |
| Reproducible fixtures/script records model/scenario, seed, backend, dt, initial geometry, constraints and first event | `scripts/diagnose_planarity.py:51–149,273–350` captures those fields, source provenance, model identity and mechanics. It validates backend availability and restores wrapped methods in `finally`. Re-ran CLI for `short_cells` on CPU and Metal and retained both JSON outputs. |
| Planar separated, crossing/degenerate, inherited tilt, physically confined cases | `fixtures()` at lines 159–270 implements seven fixtures, including all requested geometry cases plus explicit vertical flow. Shared fixture tests pass on CPU and Metal; CUDA reports an explicit unavailable-runtime skip. |
| False Z jitter adds no new Z, preserves inherited Z | Shared test inspects both daughter `requested_direction_delta_z == 0` calls while normalization changes the stored Z orientation; planar division remains planar. The inherited-tilt fixture's first event is initialization, correctly separating it from subsequent center changes during division. |
| Classify departures with supporting evidence | `docs/tutorials/planarity.md:44–76` explains crossing and coincident degeneracy normals, inherited tilt, soft-wall convergence, vertical flow and intentional XYZ jitter. Inspected corresponding CPU/Metal/CUDA contact implementations; the documentation matches their cross-product and fallback rules. |
| Fix any demonstrated setup/implementation defect and regress | No new solver or setup defect was demonstrated by the required fixtures or inspected source. Oversized founders are separately fixed by the prerequisite issue 22, not claimed as a change authored by issue 14. Mechanism fixtures and instrumentation equivalence tests prevent regression of the diagnosed behavior. |
| Update division policy/affected tutorial actual guarantees | `division.py` docstring, `docs/tutorials/planarity.md:7–26`, controller ADR and affected tutorial guides distinguish added XY jitter, unrestricted 3D, finite-height devices and soft walls. The audit covers all scenario groups and states no current model implements strict 2D mechanics. |
| Concrete strict-2D follow-up if required | `docs/development/planar-mechanics-followup.md` specifies state and validation, constrained translations/rotations, planar contacts and deterministic ties, compatible boundaries, projected drift, independently 3D chemistry and checkpoint migration. Explicitly marked proposed, not implemented. |
| Backend-affecting fix shared fixtures and unavailable hardware | No native backend was modified. New tests parameterize all available native backends. CPU and Metal passed in this review; CUDA hardware/runtime was unavailable and skipped. Source inspection of CUDA does not constitute CUDA runtime evidence. |

Trace fidelity: `test_diagnostics_preserve_rng_and_restore_native_methods_on_failure` compares exact scene JSON and controller/RNG state against uninstrumented execution and passes; it also checks restoration after a deliberate exception. Native controller stage order matches the wrapper labels. Contact and constraint relaxation share one native call, so stage traces correctly report their combined phase; isolated contact-only and wall-only fixtures determine cause. Arbitrary custom extension work outside wrapped Python entry points is not observed and the documentation says so. Full controller/RNG payload is not serialized into each CLI report; reproducibility uses the recorded seed, supplied parameters and source provenance. This does not omit a field required by the original issue.

The independent CPU and Metal `short_cells` runs both completed 325 steps at 128 cells, with first departure `geometry_edit` at time 0.019999999552965164 and absolute orientation Z 0.0005317250033840537, matching the documented intentional XYZ jitter case. This review did not rerun the full nine-scenario trajectory matrix. Its existing outputs and source claims were read, while runtime verification here covers the shared mechanisms and this representative complete trajectory on both available backends. No assertion of all-seed/all-duration planarity is justified or made. The original older-version Luiza trajectory is unavailable, so attribution to one specific historical mechanism remains unresolved as the investigation states.

## Issue 22 / PR 26 acceptance map

| Original criterion | Independently verified evidence |
| --- | --- |
| Every ordinary native tutorial founder satisfies declared relationship | Source audit includes all 24 tutorial scenarios and `culture_dish.py` / `microfluidic_trap.py`; affected builders sample one target then cap before add. The default separate `native_controller.py` is length 3 below threshold 4. `Simulation.add_cell()` is unchanged. The 26-scenario founder matrix passes. |
| Representative seeds and every affected scenario | `test_tutorial_founders_do_not_divide_without_growth` at `python/tests/test_tutorials.py:261` covers all 26 scenarios and seeds 0, 7, 17, 71. Re-ran this matrix. |
| Zero-time regulation does not divide just from initialization | Same matrix confirms founder IDs remain after `step(0.0)`. Capping considers the actually representable native float, not just a Python double. |
| Later growth still triggers original comparison | Same matrix sets each length above its saved threshold and observes replacement on zero-time regulation. The strict comparison source is unchanged. This test targets threshold semantics rather than duplicating native growth integration. |
| Deterministic same seed and parameters | At line 293, all 26 models compare controller/RNG state and founder lengths across repeated seed-71 builds; tests pass. |
| Unbiased target sampling, no retries | At `test_division.py:108`, thresholds and final RNG state match exactly one original draw per founder. Rare Gaussian seed 3103 at `test_tutorials.py:325` is unchanged and capped without retries. Independent 1,000-founder probe matches every original uniform draw and final random state exactly. Placement draws in the culture-dish model still precede all threshold draws as before. |
| Existing checkpoint geometry and targets resume unchanged | At `test_tutorials.py:293`, all 26 models save deliberately oversized length 8 then restore exact geometry/controller state. Builder-only cap is absent from resume paths. Existing source-digest authentication still requires the original model file when resuming an old checkpoint; prose explicitly says this. |
| Explicit oversized founder tests/models remain possible | Existing `UniformLengthDivision.initialize()` and `Simulation.add_cell()` remain unchanged. Existing `test_uniform_length_division_tracks_native_identities` intentionally starts over target and still passes. Explicit `native_controller.py` initial-length parameters remain available. |
| Intentional ordinary-tutorial exceptions documented/tested | No ordinary tutorial exception exists. Low-level manual division and explicit initial-length experiments are described in `docs/tutorials/biophysics-and-growth.md:44–46`; these are distinct from the audited capped stochastic tutorials. |
| Examples and tutorial prose updated together | All affected builders change only the intended length/threshold initialization path. Guide updates cover biophysics, intracellular, signaling, microfluidics and discrete state, explain changed initial amount at unchanged concentration, and distinguish centerline length, biochemical B and geometric capsule volume. |

Additional floating-point audit: `capped_founder_length` validates finite/nonnegative inputs, rejects a minimum beyond float32 range, rounds to float32 and steps toward zero only when the rounded value exceeds the bound. Independent probes checked 831 cases around all float32 power-of-two/subnormal boundaries and confirmed the greatest representable value at or below each target. Helper tests check radius, position, direction, type and concentrations; builder diffs preserve those assignments and attached-cell state. Invalid negative Gaussian targets are explicitly rejected without retries, preserving the declared sampling contract.

Old-version checkpoint files were not independently generated in this review; compatibility evidence is source-path inspection and all-model oversized round trips. An old source-digest mismatch remains an intentional existing authentication boundary, not a newly introduced geometry clamp. Native growth implementation and every model's long trajectory were not rerun because this feature changes builder initialization only.

## Checks actually run

Each test/probe used the named worktree's `.venv/bin/python` with an explicit working directory and no inherited `VIRTUAL_ENV`.

| Worktree | Command or probe | Result |
| --- | --- | --- |
| issue-13 | `-m pytest -q python/tests/test_occupancy_reference.py` | 8 passed |
| issue-13 | `science/probe_occupancy.py` | 250 mixed local/global balances; transient topology; six native anisotropic boundary/advection comparisons; capsule/grid axis-swap symmetry; pass at stated numerical gates |
| issue-14 | `-m pytest -q python/tests/test_planarity.py` | 3 passed, 1 skipped (CUDA unavailable); CPU and Metal executed |
| issue-14 | diagnostic CLI, CPU, biophysics/short_cells, seed 17, dt 0.02, 1000-step limit, 128-cell threshold | 325 steps / 128 cells; documented intentional jitter reproduced |
| issue-14 | same diagnostic CLI with Metal | Same endpoint and first event |
| issue-22 | `-m pytest -q python/tests/test_division.py python/tests/test_tutorials.py -k 'founder or uniform_length'` | 134 passed, 31 deselected |
| issue-22 | `science/probe_founders.py` | 831 float32 boundaries and 1,000 exact random draws passed |
| all three | `git status --short` after probes | Clean worktrees |

Probe sources, raw CPU/Metal diagnostic output and compact planarity summary are colocated with this report. No source fix or new regression is requested. Native occupancy/backend implementation, unavailable CUDA execution, reconstruction of the original Luiza run, and exhaustive scientific convergence remain the explicit limits above.
