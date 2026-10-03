# Fluid mechanics, chemical transport, and biological media

This document records the implementation plan and its scientific acceptance criteria. The [code ownership document](0027-fluid-transport-biology-code.md) describes the current interfaces and execution boundaries; the plan is not a claim that every proposed validation study has been completed.
Status: proposed implementation plan. This document describes future work; the API names below are proposals. The objective is to simulate how the flow and composition of growth medium through a device determine local nutrient availability, cellular consumption, and growth.

The companion [code and API sketch](0027-fluid-transport-biology-code.md) shows the intended Python configuration, native interfaces, coupled stepping algorithm, file changes, and representative acceptance tests.

## Intended model

Device geometry and hydraulic boundary conditions determine fluid velocity. Chemical transport uses that velocity, fluid-accessible volumes, diffusivities, and chemical boundary conditions to determine local concentrations. A biological model specifies how cells consume those chemicals and convert them into biomass. A medium is a composition supplied to the device; its nutritional effect depends on the cell model.

```mermaid
flowchart LR
    D[Device and hydraulic conditions] --> F[Fluid mechanics]
    F --> T[Chemical transport]
    M[Medium composition and feeding schedule] --> T
    T --> B[Cell uptake and growth model]
    B -->|Chemical consumption and secretion| T
    B -->|Changing cell geometry| F
```

General fluid mechanics here means reusable physical interfaces. The initial numerical model remains steady, incompressible Newtonian Stokes flow. This refactor does not introduce inertial Navier–Stokes flow, turbulence, immiscible phases, or composition-dependent viscosity. The first media comparisons use a shared carrier fluid with explicitly specified constant properties.

The first complete application will compare the same device and attached growing cells under several flow rates and nutrient compositions. It will report supplied, remaining, consumed, and discharged chemical amounts; biomass production; local concentrations; and spatial growth rates. Scheduled medium switches and a two-essential-nutrient example complete the initial user-facing scope.

## What can be reused

The current [physical flow solver](../../cpp/core/stokes_flow.cpp), [fluid geometry](../../cpp/core/fluid_geometry.cpp), and [transport solver](../../cpp/core/solute_transport.cpp) already implement most numerical infrastructure. Preserve their hydraulic, contact, geometric-conservation, transport, and backend checks while extracting biological assumptions from their interfaces.

The current [simulation integration](../../cpp/core/culture_simulation.cpp) owns persistent cell poses, biochemical volume, chemical amounts, and reservoir ledgers. It currently converts a configured cylindrical elongation rate into growth demand, limits that demand by intracellular stores, and replenishes the stores through linear membrane uptake. Retain this behavior as an explicit reserve-budget model while adding a distinct nutrient-limited growth model.

The existing [nutrient benchmark](../../scripts/run_nutrient_benchmarks.py) and [its documented equations](../tutorials/nutrient-validation.md) already connect transported nutrient loss to biomass gain. Reuse its conservation principle, metrics, and refinement procedure. Its full-voxel storage, smoothed cell coupling, biomass convention, and boundary discretization differ from the new geometric solver, so its numerical values are not an exact trajectory oracle for the new mode. Use matched limiting cases and an independent small reference model.

## Layer boundaries and ownership

| Layer                       | Owns or defines                                                                                                                                       | Inputs from another layer                                           | Excludes                                                              |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Fluid mechanics             | Grid, walls, physical properties, hydraulic patches, pressure, velocity, capsule responses, accessible fluid geometry                                 | Body poses and prescribed geometric growth from simulation coupling | Solute identities, media recipes, biomass yields, cell species        |
| Chemical transport          | Solute identities and units, diffusivities, authoritative extracellular amounts, chemical boundary conditions, surface transfers, reaction accounting | Accessible geometry and conservative volume fluxes                  | Organism-specific nutritional meaning or an independent growth update |
| Biological media and growth | Immutable medium compositions, cell-specific kinetic parameters, uptake/growth stoichiometry, optional reserves, division inheritance                 | Local chemical environment and accepted transfer amounts            | Fluid discretization and linear solvers                               |
| Simulation orchestration    | Clock, schedule position, lifecycle, mapping between cells and physical bodies, staged candidates, commit/rollback                                    | Candidates and reports from all three layers                        | A second copy of authoritative chemical or biomass state              |

The simulation owns authoritative state. Solvers consume views and return candidates; their retained resources are numerical caches. Python medium and growth definitions compile to validated data rather than becoming executable content in checkpoints. Physical body identifiers map to cells at the coupling boundary; a fluid-only run has no cell dependency.

## Concrete extraction

| Current interface or file                                 | Proposed responsibility                                                                                                       |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `FluidProperties`                                         | `FluidProperties`: viscosity and density                                                                                      |
| `StokesFlowSolver`, `stokes_flow.cpp`                     | `StokesFlowSolver`, `stokes_flow.cpp`: physical device flow, with optional capsule coupling                                   |
| `CapsuleBody`                                             | Explicit capsule body geometry; do not imply arbitrary rigid-body shapes                                                      |
| `FluidGeometry`, `fluid_geometry.cpp`                     | `FluidGeometry`, `fluid_geometry.cpp`: accessible fragments, surfaces, overlaps, and volume changes                           |
| `Solute`, `SoluteTransportSolver`, `solute_transport.cpp` | `Solute`, `SoluteTransportSolver`, `solute_transport.cpp`                                                                     |
| `ChemicalBoundary`, `SurfaceTransferLaw`                  | Chemical boundary and surface-transfer data, separate from hydraulic ports                                                    |
| `NumericsDevice`, `solve_numerics_linear`                 | Private shared numerical operations used by both fluid and transport solvers, with no biological names or types               |
| `CultureConfiguration`, `culture_simulation.cpp`          | A culture configuration composing independent fluid, chemistry, and biology definitions; simulation coupling remains explicit |

Extract common grid/face-field primitives from their current dependency on `SignalGridSpec` and `SignalGridVelocityField`. Extend the existing `flow.py`/`stokes.py` authoring surface instead of creating another parallel fluid framework. Put chemical authoring in `transport.py`, medium recipes in `media.py`, and growth definitions in `growth.py`.

Retain the shallow and Stokes–Brinkman solvers as distinct models. Their normalized inlet-speed and porosity assumptions must not silently acquire the pressure/rate semantics or cell-scale capabilities of the physical solver. Introduce adapters only for compatible units, geometry, and flux contracts; reject unsupported combinations explicitly. Existing device wall descriptions should materialize the physical grid and port patches without requiring a legacy signal grid or mechanical constraints.

## Medium and feeding definitions

A `Solute` has a stable identity, display label, diffusion coefficient, and declared amount basis. A `Medium` is a named immutable mapping from solute identities to concentrations, with units and optional composition provenance. The simulation compiles the union of solutes used by all media before stepping. An omitted concentration for a registered solute is zero; an unknown identity or incompatible unit is an error. Reordering labels or changing a recipe must not reorder numerical channels.

`Perfusion` binds a medium to an existing port and a hydraulic drive. Permit either prescribed volume flow or pressure, with outlet pressure as the usual reference. Support piecewise-constant changes to medium composition and hydraulic drive independently. Define schedules as right-continuous at event times, split steps exactly at discontinuities, and checkpoint the schedule and its position. Validate hydraulic compatibility across connected components at every event. Interactive changes enter the same validated event mechanism.

Convert common authoring units, such as microliters/minute and millimolar, at the boundary. Native transport retains amount per model volume with an explicit per-solute amount basis. Molecular solutes can use mol; a documented effective nutrient pool can use a mass basis. Cross-basis conversion requires the relevant molecular weight. Use a small supported conversion set; do not introduce a general units language.

Keep hydraulic and chemical boundary semantics separate. A normal perfusion inlet specifies the supplied chemical flux associated with its incoming volume flow. A convective outlet removes chemicals at the interior concentration and has zero diffusive flux. Explicit reservoir-contact boundaries may supply chemicals by diffusion even when flow is zero. Backflow requires a defined incoming composition or rejects the configuration. The present transport solver applies diffusive reservoir contact at every port; preserve that behavior for existing configurations, but do not silently use it for the new perfusion default.

Recipes do not contain universal growth rates or a single nutritional-quality score. Those are properties of an organism/model under stated conditions. Represent complex media through measured components or explicitly calibrated effective pools; a name such as LB does not establish its transport or nutritional parameters. Different recipes initially share one declared carrier-fluid property set. Mixing media with incompatible physical properties is unsupported until a property-mixture law is implemented.

## Growth models and units

Provide two explicit growth models initially: the existing intracellular-reserve budget behavior, and a single-substrate `MonodGrowth` model with maximum specific biomass growth rate, half-saturation concentration, biomass yield, and a substrate identity. Maintenance and decay default to zero. Add `EssentialNutrientGrowth` next, using the declared Liebig minimum rule `mu = mu_max * min_j(c_j / (K_j + c_j))` over explicitly required nutrients, with consumption requirements for each. Alternative carbon sources, diauxic switching, adaptation, and inhibition require their own stated models; they must not arise from an implicit sum or minimum over all recipe ingredients.

For the single-substrate model, let M be biomass mass and c the membrane-area-weighted substrate concentration:

```text
mu(c) = mu_max * c / (K_s + c)
dM/dt = mu(c) * M
substrate consumption rate = (dM/dt) / Y
```

The numerical implementation must derive actual biomass gain from accepted substrate consumption, `Delta M = Y * U`. It must not independently grow a cell by the requested rate and then attempt to remove unavailable nutrient. A positive implicit uptake formulation with bounded nonlinear iteration can reuse the present surface sink: its linearized coefficient is proportional to `mu_max * M / (Y * surface_area * (K_s + c))`. Recompute the concentration-dependent coefficient until the kinetic residual passes, rather than accepting an arbitrarily lagged denominator during a sharp feed change. Freeze old biomass consistently for the first-order scheme and verify temporal convergence.

For multiple essential nutrients, use one accepted growth extent with stoichiometric consumption of every required nutrient. Do not solve independent unlimited uptake demands, take the minimum growth afterward, and lose excess nutrient. Explicit reserve models may retain excess uptake in their corresponding intracellular pools. Every chemical transfer and reaction has one owner; an intracellular rate plan must not consume a substrate a second time for the same growth event.

Preserve the existing biochemical volume as authoritative state and declare a biomass density converting it to dry biomass mass: `M = rho_b * V_bio`. Use yields in mass or molar units compatible with the substrate's declared basis. For the current fixed-radius capsule mapping, `Delta V_geom = Delta V_bio / kappa` and `Delta length = Delta V_geom / (pi*r^2)`. Do not apply a biomass-specific growth rate directly to cylindrical length. Keep legacy `growth_rate`/rate-instruction meanings intact and expose the new realized specific biomass growth rate separately. Model bindings, biomass density, reserves, and parameters inherit explicitly at division.

Reuse the existing typed rate arithmetic for custom intracellular biology and native execution. Introduce only the additional local-environment inputs and growth/transfer outputs needed by these concrete models. The first release does not require a universal metabolic-network compiler. Constant and saturating surface exchange plus declared consumption/product stoichiometry provide the initial chemical reaction scope; additional bulk reactions can reuse rate-plan expressions behind an amount-conserving source contract.

Monod kinetics provides a baseline model, not a guarantee of physiological response during rapid medium switching. Experimental microfluidic work has measured history-dependent growth under nutrient fluctuations, so schedule fidelity and an optional future adaptation state are distinct from the first equilibrium growth law. See [Nguyen et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8209047/). Spatial nutrient gradients and local growth are also directly studied in [Hornung et al., 2018](https://arxiv.org/abs/1802.05858).

## Coupled stepping and backend execution

For each candidate interval, evaluate the feeding schedule, propose uptake/growth, predict changed cell geometry, solve the corresponding fluid displacement, and advance chemical amounts. Use the accepted chemical transfers to correct the biomass and geometry proposal. Repeat from the same beginning-of-step state until nutrient, growth, and geometric-volume residuals agree. A candidate iteration must never spend material or accumulate reservoir ledgers twice. If the bounded coupling iteration fails, reduce the interval and retry; retain complete rollback on failure.

This driver keeps the existing geometric-conservation projection, minimum-gap checks, quaternion state, amount-preserving division, controller/RNG transaction, and checkpoint guarantees. Initially demonstrate attached cells with no translational motion, then add division and the existing freely moving capsule mode under the same biological definitions. Hydrodynamic forces must not be an obligatory part of defining a medium or testing a growth law.

Implement every added numerical operation on CPU, Metal, and CUDA in the phase that introduces it. Python performs authoring and event scheduling; native code performs per-cell kinetics and numerical updates. Reuse shared validated operation definitions and immutable parameter arrays. Host geometry, assembly, and scalar convergence control remain explicit. CPU execution is the reference, and the declared GPU precision/tolerance contract remains visible. No Python per-cell loop or hidden CPU numerical fallback should stand in for a missing native growth operation.

Retain the existing geometric and chemical precision contracts during extraction. For the new small acceptance fixtures, target CPU/GPU agreement within 0.1% for integrated biomass and uptake, and normalized chemical budget residual below 1e-5 with a documented absolute scale near zero. Use measured solver residuals to distinguish linear convergence from discretization error. Freeze justified fixture tolerances before accepting results; do not widen them solely to obtain a pass.

## Ordered implementation phases

| Phase                                 | Changes                                                                                                                                                                                      | Completion evidence                                                                                                                                                                |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Establish contracts and references | Record current behavior; define units, port semantics, biomass conversion, and target model equations. Add independent well-mixed growth and transport reference cases.                      | Existing suites remain green; references distinguish biomass growth from elongation and nutrient supply from intracellular reserves.                                               |
| 2. Extract general physics            | Move/rename flow, geometry, capsule coupling, and shared numerical operations; remove chemical-grid dependencies; add a device-grid authoring adapter.                                       | A fluid-only pressure/rate example runs with zero solutes and zero cells on each backend; existing hydraulic/body results are preserved.                                           |
| 3. Extract chemical transport         | Separate hydraulic patches from chemical boundaries; expose chemical state and paired surface transfers; add the perfusion boundary semantics.                                               | An inert-tracer pulse, diffusion-only case, mixed inflow/outflow case, and moving-volume case conserve amounts without any growth model.                                           |
| 4. Add media and feeding              | Implement immutable compositions, stable solute indexing, bounded unit conversion, independent flow/composition schedules, and restartable events.                                           | Two feeds mix predictably; composition switches propagate through transport; restart across a scheduled switch reproduces the same continuation.                                   |
| 5. Couple nutritional growth          | Add single-substrate Monod growth, explicit biomass conversion, and bounded uptake/geometry iteration; retain reserve mode; then add two-essential-nutrient growth and division inheritance. | No-substrate/no-reserve control produces no biomass; accepted substrate consumption determines biomass gain; changing the limiting nutrient changes the correct growth constraint. |
| 6. Deliver the experiment workflow    | Compose fluid/chemistry/biology configuration in `Simulation` and the native model runner; update checkpoints, analysis, viewer labels, examples, and documentation.                         | Flow/composition sweep and starvation/refeeding examples run end to end; output includes realized growth, nutrient budgets, recipe identities, schedule, and parameter provenance. |
| 7. Validate and characterize          | Run spatial, surface, timestep, and coupling-tolerance refinement; exercise CPU/Metal/CUDA hardware; record runtime/memory and remaining physical limitations.                               | Numerical acceptance gates pass on actual supported devices; throughput claims use measured evidence; compile-only CUDA results remain labeled as such.                            |

Persistence work accompanies each state addition; phase 6 completes the application presentation rather than postponing restart design. Backend checks accompany every numerical phase. The first reviewable implementation should contain phases 1–2 only, with no change to existing biological behavior. The first biological milestone is phase 5's single-substrate flow/composition comparison; multi-nutrient behavior follows that gate before the complete workflow is declared done.

## Acceptance experiments

| Experiment                               | Required observation                                                                                                                                                                                                                            |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fluid with no chemistry                  | Hydraulic results are independent of whether a medium definition exists.                                                                                                                                                                        |
| Nonreactive chemical pulse               | Correct transport, arrival/clearance behavior, and input/output inventory without cells.                                                                                                                                                        |
| Well-mixed single-substrate reference    | Saturating specific growth and the expected yield relationship; depletion agrees with an independent reference integrator.                                                                                                                      |
| Same medium, different flow rates        | In a deliberately supply-limited geometry, increased delivery changes local substrate and growth; a separately saturated control approaches its configured growth ceiling. Do not require universal monotonicity in every geometry.             |
| Same flow, different medium compositions | Nutrient concentration and identity determine the response through the configured cell model; changing an inert constituent does not change growth.                                                                                             |
| Essential nutrient omitted               | Growth is limited by the missing requirement even with abundant other substrate; explicit reserves, if present, are included in the budget.                                                                                                     |
| Starvation and refeeding                 | A verified closed/no-feed control stops growth once any declared reserves are exhausted; refeeding restarts the selected kinetic model after transport delivers substrate. Diffusive reservoir supply is either counted or disabled explicitly. |
| Two feeds and a timed switch             | The field reflects transport and mixing; cells do not instantly inherit a reservoir composition merely because its label changed.                                                                                                               |
| Growth and division                      | Extracellular amounts, biomass, intracellular stores, lineage, and recipe/model bindings survive geometry changes without unaccounted creation or loss.                                                                                         |
| Failure and restart                      | Failures roll back schedule position, biochemical state, fluid/chemical state, RNG, ledgers, and time; restart resumes each phase of a feed schedule.                                                                                           |

Use a few attached cells and defined nutrient compositions first, with initial nutrient/reserve amounts small enough that the response is supply-limited. Record local concentrations, growth, biomass, input and output nutrient amounts, uptake, and inferred limitation. Compare at least three spatial resolutions and successive timestep halvings; the initial target is less than 2% change in integrated biomass and uptake at the finest refinements, with decreasing errors and independent checks of surface geometry. Backend agreement alone does not establish physical accuracy.

For a closed single-substrate test with no maintenance or reserves, the explicit check is `extracellular substrate + biomass gained / yield = initial substrate`. Open cases add initial/final reserves and the signed boundary ledger. Multi-nutrient and product models use their declared stoichiometric balances; do not sum unrelated chemical species amounts and call that mass conservation.

## Compatibility and deliverables

Rename the new, unreleased `Media*` APIs directly; add a temporary alias only if an actual consumer needs it. Preserve established flow/signal APIs and the existing checkpoint version-10 reader. Pure renaming should not change serialized meaning. Add a new version only for new state/contracts, with explicit migration of old runs to the reserve-budget behavior; never reinterpret old `growth_rate`, amount units, or port boundaries as the new model. Preserve supported legacy scene/analysis readers and document any renamed tables or fields. Continue to distinguish fragment quantities from voxel means in visualization.

Checkpoint all active medium definitions, solute order and units, feeding schedules and event position, growth-model bindings, parameter provenance, and declared stores. Analysis should expose actual specific biomass growth, uptake by nutrient, biomass produced, reservoir supply/discharge, and model identity. Recipe names are metadata, not additional conserved chemical fields.

The finished user workflow must allow a model author to choose a device, fluid model, medium composition, perfusion schedule, and cellular growth model independently, then compare experiments using the same analysis path. Ship one clearly labeled illustrative defined-medium example and one two-essential-nutrient example. Organism-specific recipes or kinetic parameters require provenance and calibration evidence. Cell exit/washout, singular lubrication, large-colony acceleration, physiological adaptation, gas exchange, and full metabolic models remain separate work with explicit boundaries.
