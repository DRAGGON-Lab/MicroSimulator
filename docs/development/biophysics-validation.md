# Biophysics audit and validation envelope

This document audits MicroSimulator as a numerical model of rod-shaped bacterial colonies in
microfluidic devices. It is a traceability record, not an experimental calibration. Unless a model
author supplies organism-, medium-, device-, and dataset-specific evidence, outputs are mechanistic
or phenomenological simulations rather than biological predictions.

CPU is the scientific reference implementation. Metal and CUDA runs establish backend consistency
with that reference; agreement between implementations cannot establish that an equation is a good
description of an experiment.

## Evidence labels

- **Verified** means a governing equation or invariant has an executable analytic, conservation, or
  refinement check.
- **Phenomenological** means the implementation is internally tested but intentionally represents a
  biological process with a simplified closure.
- **Uncalibrated** means parameters have model units and no dataset-specific fit is claimed.
- **Scientifically questionable** identifies a convention that can be numerically consistent while
  being easy to misinterpret physically. It is retained for compatibility and should only be changed
  in a separately reviewed contribution.

## Equation and unit matrix

The base length and time units are selected by each model; microfluidic tutorials conventionally use
micrometres and seconds. Concentrations and amounts are likewise model-defined. Dimensional
consistency therefore requires the model author to use one coherent unit system.

| Quantity | Governing equation and required units | Implementation | Evidence | Interpretation and status |
| --- | --- | --- | --- | --- |
| Rod geometry | Centerline segment length `L`, radius `r`; end-to-end extent `L + 2r`. Geometric capsule volume is `pi r^2 L + 4 pi r^3/3`. | `cpp/core/world_state.cpp`; `python/src/microsimulator/biomass.py` | contact geometry tests; `division_conservation` | Constant-radius spherocylinder; **verified geometry**, **phenomenological morphology**. |
| Biochemical biomass | `V_b = pi r^2 (L + 2r)`. A concentration `c` represents amount `c V_b`. | `cpp/core/species.cpp:128`; `biomass.py` | `biomass_and_dilution`, `division_conservation` | This is larger than capsule volume by `2 pi r^3/3`. It makes division amount-conserving, but is not literal geometric volume: **scientifically questionable compatibility convention**. |
| Axial growth | Forward Euler: `L_(n+1) = L_n (1 + mu dt)`, approaching `L(t)=L0 exp(mu t)` as `dt -> 0`; `mu` has `time^-1`. Radius is constant. | `cpp/core/world_state.cpp:311` | `growth_recurrence` | Exponential axial extension, no width regulation: **verified numerical law**, **uncalibrated**. |
| Dilution | After growth, `c_new = c_old V_old/V_new`, before reaction rates are applied. | `cpp/cpu/cpu_species.cpp:103`; `cpp/cpu/cpu_coupled.cpp:145` | `biomass_and_dilution` | Conserves intracellular amount in a growing cell: **verified**. |
| Division | `L1=f(L-2r)`, `L2=(1-f)(L-2r)`; daughters are separated by `2r`, inherit concentration, type, growth rate, fixed state, and record the parent ID. | `cpp/core/world_state.cpp:230` | `division_conservation`; `python/tests/test_biomass.py` | Conserves `V_b` and intracellular amount, not capsule volume: **verified convention**, biologically **phenomenological**. |
| Cell contact | Closest centerline points define signed separation `s=d-(r_i+r_j)` and the contact normal. A pair contributes two endpoint-weighted directed rows when appropriate. | `cpp/cpu/cpu_contacts.cpp`; `cpp/core/contact_graph.cpp` | contact geometry suite; `contacts_and_mechanics` | Frictionless soft overlap correction, not adhesion or cell-wall deformation: **verified geometry**, **phenomenological mechanics**. |
| Quasi-static mechanics | Solve the symmetric correction operator for translations, rotations, and length correction. `mu_a` weights length and `1/gamma` regularizes axial change. Corrections are applied without inertia. | `cpp/cpu/cpu_mechanics.cpp:186`; `cpp/core/mechanics_integration.cpp` | mechanics operator tests; overlap, fixed-cell and centre-of-mass checks; extended sensitivity | Solver parameters are not elastic moduli or drag coefficients: **verified numerics**, **uncalibrated phenomenology**. |
| Flow translation | `dx/dt = u(x)` from fluid-only interpolation; fixed cells have zero drift. | `cpp/core/simulation.cpp:108` | `flow_translation_and_rotation` | Cells are perfect tracers with no slip, settling, wall drag, or hydrodynamic cell-cell coupling: **phenomenological**. |
| Flow rotation | `dp/dt = Omega p + lambda[E p - p(p.E.p)]`, `lambda=(a^2-1)/(a^2+1)`, with equivalent-spheroid aspect `a=(L+2r)/(2r)`. | `cpp/core/simulation.cpp:139` | finite-aspect shear and rigid-rotation convergence tests | Jeffery closure for a spheroid, not a resolved capsule near walls: **verified against the closure**, **uncalibrated approximation**. |
| Intracellular reactions | User rate plan supplies `dc/dt=f(c,s,x,L,r,mu,...)`; Forward Euler follows growth dilution. Rate units are concentration/time. | `cpp/cpu/cpu_species.cpp`; `cpp/cpu/cpu_coupled.cpp` | rate-plan and coupled-rate suites | Equation semantics are **verified**; biological networks are entirely user-specified. |
| Cell-grid exchange | Cell signal output is an amount/time and is scattered conservatively as concentration/time after division by voxel volume. Opposite cell/grid fluxes conserve total amount. | `cpp/cpu/cpu_coupled.cpp` | `membrane_exchange`; coupling foundation tests | Kernel interpolation is a numerical coupling, not an explicit membrane: **verified conservation**, **phenomenological exchange**. |
| Diffusion | `dc/dt = D nabla^2 c`; `D` has `length^2/time`. Central second differences; Forward Euler, Crank-Nicolson, or backward Euler time integration. | `cpp/core/signals.cpp:778` | decay, positivity, conservation, and refinement checks | **Verified** for the tested grids and boundary types. |
| Advection | `dc/dt = -div(u c)` with conservative first-order upwind face flux; `u` has `length/time`. | `cpp/core/signals.cpp:778` | periodic transport conservation and refinement | Conservative and non-negative under its step restrictions, but numerically diffusive: **verified numerics**. |
| Reactions and boundaries | Affine grid reaction `dc/dt=q-kc`; no-flux, periodic, and fixed ghost-centre boundary values. | `cpp/core/signals.cpp:224`; `cpp/core/signals.cpp:778` | signal-grid tests; `transport` | Boundary values are ghost-centre concentrations one spacing from a site: **verified convention**. |
| Depth-averaged flow | `u=-M grad(p)`, `div(u)=0`; default clear-gap mobility scales with squared gap height. | `cpp/core/flow_system.hpp`; `python/src/microsimulator/flow.py` | uniform duct, mobility split, and thin-gap cross-check | Hele-Shaw/Darcy closure: **verified against analytic cases**, valid only when depth averaging is justified. |
| Resolved flow | Steady incompressible Stokes-Brinkman: `-grad(p)+mu nabla^2 u-alpha u=0`, `div(u)=0`, with nondimensional solver coefficients. | `cpp/core/flow_system.hpp`; `cpp/cpu/cpu_flow.cpp`; `python/src/microsimulator/stokes.py` | plane/square duct and two-layer Brinkman benchmarks | **Verified numerical solver**; physical viscosity/pressure scaling and porous drag require model calibration. |
| Nutrient-limited growth | Tutorial closure `mu(S)=mu_max S/(K_S+S)` and `Delta biomass = Y Delta nutrient`; `K_S` has concentration units, `Y` biomass/nutrient. | `scripts/run_nutrient_benchmarks.py`; tutorial models | `monod_and_yield`; extended nutrient balance and refinement | Monod limits and accounting are **verified**; all biological parameters are **uncalibrated**. |
| Biomass flow resistance | Smoothed biomass volume fraction is mapped to mobility/drag using model parameters and a physical averaging radius. | `python/src/microsimulator/flow.py` | nutrient sensitivity study | Empirical porous-medium closure: **phenomenological and uncalibrated**. |
| Attachment and detachment | `fixed` cells do not drift or mechanically move. Tutorial controllers choose which daughter remains fixed and remove cells at outlets. | `cpp/core/simulation.cpp`; `python/src/microsimulator/controller.py`; device examples | fixed-cell and application workflow tests | A model rule, not a force-based adhesion or detachment prediction: **phenomenological**. |

## Executable evidence

Run the normal CPU gate with:

```console
uv run python scripts/run_biophysics_benchmarks.py \
  --suite quick --backend cpu --output build/biophysics-validation.json
```

For the standard Docker workflow, mount the checkout at `/workspace`, build/install the locked
project inside the container, and run the same command there. Evidence is written through the mount
to the host's `build/` directory; no host compiler or Python environment is used.

The output records the commit, backend device, elapsed time, metrics, exceptions, and an explicit
`calibration_claim: false`. A failed check still produces the JSON artifact and exits nonzero.

The release/research gate adds transport and mechanics sensitivity, the existing analytic flow and
controlled nutrient studies, and focused growing/dividing colony and integrated-device workflows:

```console
uv run python scripts/run_biophysics_benchmarks.py \
  --suite extended --backend cpu --output build/biophysics-validation-extended.json
```

Generated evidence belongs under `build/` and is not committed. Metal or CUDA may be selected only
when that native backend is available. Such a result is backend-conformance evidence; run and pass
CPU on the same commit before interpreting it.

### Implementation-time result

On 2026-09-16 the quick suite passed 8/8 checks and the extended suite passed 15/15 checks in the
project's Linux Docker environment. Selected extended metrics were:

| Measure | Result |
| --- | ---: |
| Growth recurrence order against the exponential limit | 0.958 |
| Intracellular amount error under dilution | `1.49e-8` relative |
| Division biomass and species-amount errors | 0, 0 |
| Mechanics overlap, initial to one relaxation | 0.200 to 0.040 |
| Jeffery shear direction error | `2.36e-4` |
| Periodic transport mass error | `3.73e-9` |
| Transport refinement order, fine pair | 0.973 |
| Exchange refinement order, fine pair | 1.009 |
| Maximum exchange amount-balance error | `2.00e-8` relative |
| Nutrient-study balance error across cases | at most `7.13e-5` relative |
| Analytic flow checks | 10/10 passed |

These figures characterize the reviewed working tree and are not release evidence because the
benchmark implementation was not yet committed when they were generated. Regenerate both JSON
artifacts from the final commit; only those artifacts can support a release claim.

## Validity envelope

The present model is suitable for studying qualitative and numerical consequences of these declared
assumptions:

- constant-radius rods undergoing exponential axial growth and discrete division;
- frictionless, quasi-static overlap relaxation with no inertia, explicit adhesion, or viscoelastic
  cell wall;
- passive-tracer translation and equivalent-spheroid Jeffery rotation in prescribed flow;
- conservative finite-volume-like solute transport on a Cartesian grid, with first-order upwind
  advection;
- depth-averaged flow in shallow gaps or resolved steady Stokes-Brinkman flow;
- empirical biomass resistance and controller-defined attachment, detachment, and washout; and
- user-specified intracellular kinetics, exchange, Monod constants, yields, and boundary reservoirs.

It does not by itself support claims about absolute growth, nutrient penetration, mechanical stress,
detachment thresholds, flow occlusion, lineage fitness, or device performance in a real organism or
device. Those claims require parameter inference and validation against the relevant measurements,
including uncertainty and grid/timestep sensitivity at the fitted operating point.

## Primary references

- Monod, “The Growth of Bacterial Cultures” (1949),
  [DOI 10.1146/annurev.mi.03.100149.002103](https://doi.org/10.1146/annurev.mi.03.100149.002103).
- Jeffery, “The Motion of Ellipsoidal Particles Immersed in a Viscous Fluid” (1922),
  [DOI 10.1098/rspa.1922.0078](https://doi.org/10.1098/rspa.1922.0078).
- Brinkman, “A Calculation of the Viscous Force Exerted by a Flowing Fluid on a Dense Swarm of
  Particles” (1949), [DOI 10.1007/BF02120313](https://doi.org/10.1007/BF02120313).
- Rudge et al., “Computational Modeling of Synthetic Microbial Biofilms” (2012),
  [CellModeller paper](https://cellmodeller.github.io/CellModeller/pdfs/acs2012.pdf), for the
  individual-based modeling lineage. MicroSimulator's equations and validations are the repository
  sources named above; compatibility is not evidence of calibration.
