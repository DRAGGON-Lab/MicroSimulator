# ADR 0025: conservative extracellular storage with coarse geometric porosity

- Status: selected numerical design; executable CPU reference only
- Date: 2026-09-24
- Scope: issue #13. Native implementation and enabling this model in simulations are separate work.

## Decision and current behavior

Select **coarse geometric porosity** as an opt-in transport model. Estimate the union of cell capsules inside each non-wall voxel, store solute amount per voxel, and use a declared porosity closure for face conductance. This approximates cell exclusion without claiming to resolve sub-voxel fluid passages, membrane boundary layers, displacement flow, or hydrodynamic forces. It is preferable here to treating smoothed biochemical biomass density as an exact solid fraction.

Current production transport continues to use full non-wall voxel volume: `SignalGridSpec.voxel_volume()` is `hx*hy*hz`, and `cpu_coupled.cpp` divides scattered cell amount rates by that volume. `colony_volume_fraction()` deposits conserved biochemical biomass for empirical resistance; it may exceed one and is not this occupancy representation. No production defaults or checkpoint formats change in this contribution.

The reference is `python/src/microsimulator/occupancy_reference.py`. It uses float64 midpoint quadrature and a dense backward-Euler solve, intentionally unsuitable for large simulations. Its interfaces carry explicit amounts, volumes, faces, and ledgers so later backends can be compared without inheriting their implementation.

## State, geometry, and units

For voxel i, geometric volume `V_i = hx*hy*hz`, accessible fraction `epsilon_i`, accessible storage `W_i = epsilon_i V_i`, and extracellular concentration `c_i`, define the authoritative solute amount `N_i = W_i c_i`. Concentration is amount per accessible fluid volume, including in partially occupied voxels. Length has units L, time T, concentration A/L³, and amount A.

Occupancy depends only on the current cell centers, normalized directions, nonnegative centerline lengths, positive radii, lattice centers/spacing, and the existing binary transport-wall mask. It does not depend on species, growth-rate attributes, cell type, stationary attachment, or the biomass-resistance averaging radius. All cells, including mobile ones, exclude storage. Mechanical wall primitives do not define a second transport mask: only the declared voxel obstacle mask clips storage.

Each voxel uses m³ midpoint samples on its physical box centered at `origin + index*spacing`. A sample is occupied when its distance to any capsule centerline segment is at most the corresponding radius. Count the union, never summed capsule fractions, so overlaps cannot make epsilon negative. Outside-domain capsule pieces are ignored. A wall voxel has epsilon zero regardless of cells. Default reference m is 8; m is a reproducible numerical parameter requiring convergence checks. Even a degenerate lattice axis retains its physical voxel thickness for occupancy and storage; it does not turn capsules into disks.

Biochemical biomass remains `B = pi*r²*(length + 2*r)`. Geometric capsule volume is `V_geom = pi*r²*length + 4*pi*r³/3`. Native division preserves B but, for equal radii, reduces total capsule volume by `2*pi*r³/3`. This closure deliberately exposes the resulting extra fluid storage. It represents coarse division geometry, not physical septum formation or calibrated cell volume. Never transfer extracellular solute into daughters merely because their geometric occupancy changed.

## Conservative transport balance

For an internal oriented face i→j, use harmonic closure `a_f = 2 epsilon_i epsilon_j / (epsilon_i + epsilon_j)` when both voxels are accessible, otherwise zero. This is a coarse permeability/aperture approximation, not geometric face intersection. Let `K_f = D a_f A_f / d_f` and `Q_f = a_f A_f u_f`. D has units L²/T, K and Q have units L³/T, and u is intrinsic accessible-fluid velocity in L/T. Positive Q flows from i to j. The outward amount rate is

```text
F_ij = K_f (c_i - c_j) + max(Q_f, 0) c_i + min(Q_f, 0) c_j
```

The identical face value enters the neighbor with the opposite sign. With amount source S_i, accessible-fluid reaction source b_i, and first-order loss lambda_i:

```text
dN_i/dt = -sum_faces F_ij + S_i + W_i b_i - lambda_i N_i
```

After the geometric remap described below, freeze W and face coefficients over one transport step. The first native implementation should offer backward Euler:

```text
(W + dt L) c_new = N_remapped + dt (S + W b + reservoir_inflow)
N_new = W c_new
```

L has diffusion conductances, upwind outgoing fluxes, and `lambda_i W_i` on its diagonal, with negative incoming neighbor coefficients. Zero-storage rows are isolated identity rows with zero right-hand side. The reference reports before/after total amount plus separately integrated source, reaction, and boundary amounts; internal faces cancel. Do not infer conservation from changes in concentration or from an unweighted sum of concentrations.

No-flux boundaries have K=Q=0. Periodic pairs contribute one shared internal face. Fixed reservoirs retain the existing **exterior lattice-center** convention: distance d is one spacing, not a half spacing. For this closure, use exterior epsilon=1 and the same harmonic rule. Reservoir exchange is `K(c_res-c_i) - Q*c_upwind` with Q positive outward, and is included in the boundary ledger. In a singleton axis production transport has no face operator, matching the existing degenerate-axis rule. Affine b is concentration/time per accessible fluid; a physical source specified per whole voxel instead becomes an explicit amount rate, never an implicit second volume conversion.

## Velocity convention and flow coupling

The current face field is a velocity used directly by transport and also sampled by cell drift. With the existing binary wall mask and epsilon=1 on fluid sites, intrinsic velocity and whole-open-face volumetric velocity coincide. Partial porosity introduces a distinction that cannot be inferred from old arrays.

In the new opt-in mode, retain **intrinsic velocity** in `SignalGridVelocityField` and derive Q by multiplying `a_f*A_f` once. If a flow solver produces integrated flux Q, convert to intrinsic u by dividing by `a_f*A_f` once on open faces and require Q=0 on closed faces. Never multiply an already aperture-weighted flux by epsilon again. Constant-advection inputs use the same explicit convention.

Existing flow solutions generally satisfy continuity for their current binary fluid geometry, not for these new weighted faces. They must not be advertised as occupancy-consistent flow without a weighted projection/re-solve. For fixed occupancy, validate `sum Q_out = 0` in interior voxels. Moving occupancy would require `dW/dt + sum Q_out = 0` for incompressible displaced fluid; this initial model instead uses the explicit conservative remap below. That remap is a solute bookkeeping closure and does not solve fluid displacement. Prescribed velocity experiments must declare this limitation; coupling a resolved displacement-flow model is subsequent work. Transport remains amount-conservative even for a prescribed divergent field, but uniform concentration need not remain uniform.

## Occupancy changes, closed storage, and transaction order

Fractions smaller than `epsilon_cutoff = 1e-8` are treated as zero, including for face closure. The cutoff is part of the model configuration and checkpoint state, not a hidden denominator floor. Keep N fixed in each voxel whose new W remains positive, so concentration changes to N/W. Newly accessible voxels start with their existing amount, normally zero; removal does not invent extracellular solute.

For voxels changing to zero storage, redistribute their entire amount over newly accessible recipient voxels in the same face-connected component of the **union of old and new accessible voxels**, weighted by recipients' new W. Connectivity uses regular voxel neighbors including declared periodic pairs and excludes persistent wall/closed voxels. Use sorted voxel order for deterministic accumulation. Existing recipient amounts are retained. If a component closes completely while holding any positive amount, reject the entire geometry/transport transaction. If it contains exactly zero amount, closure is valid. No epsilon floor, disappearing amount, cross-wall transfer, or silent clipping is allowed. A near-zero positive W can still create high concentrations; finite/positivity and solver checks may reject the step rather than alter its mass balance.

This remap is intentionally nonlocal within its transition component and can instantaneously mix expelled solute. It does not reconstruct a membrane trajectory or predict a swept-volume velocity. Its domain, cutoff, and redistribution rule must therefore be stated in scientific use. A donor is emptied once even when several cells overlap it. Reject invalid input before changing simulation state.

The proposed opt-in controller/native stage order is:

1. Snapshot complete state and authoritative N at the last committed geometry. Apply validated regulation, divisions, removals, and their geometry callbacks. Recompute occupancy and conservatively remap if geometry changed.
2. Advance biological growth and intracellular dilution using B. Recompute occupancy for post-growth geometry and remap N again. Construct one set of accessible exchange weights, sample this remapped pre-transport concentration, evaluate biology, then solve transport/reactions and scatter cell amount rates using the same weights. This explicitly changes the sampling point from the legacy stage; it needs a separately named opt-in contract and native split-stage work.
3. Apply configured flow drift and mechanical relaxation, then recompute/remap once at final geometry. Any direct geometry edit or removal outside the controller must invoke the same barrier before the next transport/export/checkpoint operation. No geometry change may leave old W paired with new cells.
4. Commit geometry, biomass/species, N, W, time, and all ledgers together only after every stage validates. A failure restores the complete transaction, including controller/RNG state. Zero-time topology edits still require remapping. Unchanged geometry reuses occupancy without resampling.

This ordering follows the existing regulation/topology → biology → drift/mechanics outline while making its new geometry barriers explicit. It is not implemented by the reference module. Production work must add atomic staging; installing only a new denominator inside `cpu_coupled.cpp` would be incorrect.

## Cell sampling and scattering

Choose a declared physical support that reaches accessible extracellular sites, restricted to the connected fluid component containing a deterministic nearest-accessible anchor. Resolve equal-distance anchors by flattened index and explicitly validate the maximum search radius; no cross-wall search. For nonnegative geometric kernel values phi_i in that support, use `w_i = phi_i W_i / sum(phi_j W_j)`. Sampling is `sum(w_i c_i)`; a cellular extracellular amount rate J scatters `w_i J`. The concentration-rate conversion is `(w_i J)/W_i` exactly once, for accessible sites only. The same weights give a partition of unity and the sampling/scattering adjoint relation. The reference `exchange_weights` receives an already connected support; topology construction remains native implementation work.

With epsilon=1 and equal voxel volumes this reduces to the current normalized trilinear weights and J/V conversion. With resolved excluded cell centers, the old eight center-neighbor stencil may have no accessible sites; expanding or surface-based physical support is required, with a declared radius and refinement study. Reject zero accessible support. Uptake must be limited by an explicitly conservative coupled solve or reject an unaffordable step; never clamp negative extracellular concentrations. Opposite intracellular amount uses B, not geometric occupancy. Washout of a cell exports its intracellular amount through a separate biological ledger; it does not remove a voxel's extracellular solute.

## Executable reference cases and tolerances

Run `uv run python -m pytest python/tests/test_occupancy_reference.py -v`. The cases are deterministic CPU tests:

| Case | Amount evidence |
| --- | --- |
| Empty grid | Float64 reference equals existing native CPU backward Euler for `[0,1,0]`, within rtol 2e-6/atol 2e-7; total amount remains 1. Empty geometry gives epsilon=1. |
| Partially occupied closed domain and unequal storage | W=[0.5,1.5], initial c=[8,0], total N=4; diffusion tends to equal c=2 with unequal amounts [1,3]. Every-step ledger residual is below 1e-12. |
| Changing occupancy, closure, reopening | N=[2,3,1], W changes [1,1,1]→[0,0.25,0.75]; remap gives [0,3.5,2.5], total 6. Reopening keeps new-site amount zero. Entire-component closure rejects without mutation; a persistent wall prevents redistribution to an unrelated recipient. |
| Division/removal, overlap, wall-adjacent geometry | One native-shaped parent becomes two daughters with unchanged B and smaller geometric volume. Union quadrature does not double-count duplicate cells, wall voxels remain inaccessible, and remapping through division/removal preserves total N. |
| Boundaries, reactions, exchange and advection | Unequal storage, an internal advective face, inflow/outflow reservoirs, decay and cellular source all contribute to the signed ledger; residual below 1e-12. A face test verifies aperture enters Q once. |
| Refinement | Backward-Euler timestep error approximately halves for 10/20/40 steps; the empty-limit centered operator error approximately quarters for 10/20/40 voxels. Sphere quadrature at m=8/16/32 improves the finest volume error to below 2%. |

The 1e-12 reference balance tolerance applies to these order-one float64 cases. General checks scale by `max(1, |N_before|, |N_after|, sum(abs(external terms)))`; scientific units must be normalized explicitly. Initial native float32 gates: per-step relative ledger residual <=5e-6, 1000-step closed-case drift <=5e-5, and concentrations versus float64 reference within rtol 2e-4/atol 2e-6 in normalized test units. These are acceptance targets to measure, not verified GPU results.

Subsequent native validation must run h, h/2, h/4 at fixed physical cell/device dimensions and exchange support, with m, 2m, 4m independently; also dt, dt/2, dt/4. Track occupied volume, total amount, concentration L1/Linf errors, boundary/reaction/cell ledgers, cutoff crossings, and solver residuals. Require convergence of the scientific observable, not only conservation. The midpoint geometry estimate can oscillate across resolutions; never assert a universal smooth-interface order from one placement. Test rotated/translated rods, overlapping capsules, nearly blocked passages, all-zero storage, wall contacts, division/removal, and restart at a geometry barrier. Vary epsilon_cutoff by factors of ten. Native backends must implement their own kernels and pass the same cases without CPU fallback.

## Compatibility and implementation work

Old checkpoints and all ordinary runs retain occupancy-disabled full-voxel semantics. Do not reinterpret old concentration arrays as fluid-volume concentrations. Enabling occupancy on an existing state is an explicit conversion: preserve legacy amount `N=V*c`, compute W, then apply closure remapping and derive c. Such a conversion can reject a sealed component and must be recorded in provenance.

A future checkpoint version must record model kind/version, lattice and wall geometry, quadrature resolution, cutoff, exchange-support/anchor rule, remap rule, velocity convention, authoritative per-species N and committed W/geometry revision, plus any required solver history. N and W participate in integrity checks. On restore, authenticate before migration, validate N>=0 and zero amount at W=0, verify geometry/occupancy consistency, and resume at a committed barrier without repeating remapping. Record precision/backend provenance; recomputing occupancy with another quadrature algorithm may change results and requires an explicit conversion. The current reference adds no fields to checkpoints.

Required follow-up contributions are (1) native state/configuration and checkpoint conversion, (2) conservative geometry rasterization/transition connectivity, (3) CPU weighted-storage operator and atomic coupled staging, (4) independent Metal and CUDA kernels/reductions/solvers, (5) accessible cell exchange and uptake budgets, (6) weighted-flow interface/projection with explicit drift semantics, and (7) end-to-end geometry/removal/restart and refinement validation. Existing flow resistance may coexist as a calibrated closure; it must not be relabeled as geometric exclusion or counted a second time in transport porosity.
