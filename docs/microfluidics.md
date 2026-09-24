# Modeling cells in microfluidic devices

MicroSimulator connects the physical environment of a microfluidic device to the behavior of individual cells. A model can follow nutrient delivery into a trap, growth and division within its walls, signaling between cells, and the removal of cells carried into an outlet. Python defines the device and biological rules; the engine advances cell mechanics and chemistry using a shared transport grid and velocity field.

## From device to experiment

1. **Describe the device.** Define the channels, cavities, and obstacles that shape the experiment. Device helpers produce mechanical constraints, solid transport masks, and inlet and outlet boundaries from one description. The [device tutorial](tutorials/microfluidics.md) covers traps and channels; the [pillar example](tutorials/flow-solvers.md) shows curved obstacles.
2. **Solve the flow.** Choose a shallow or resolved steady-flow model, set the mean inlet speed, and obtain the face velocities used by solute transport. Refresh the field when the model's stationary resistance changes.
3. **Add biology.** Define founders, growth and division rules, intracellular rates, and exchange with extracellular nutrients or signals. Growth, dilution, division, and uptake share an explicit [biochemical biomass convention](architecture/0024-biomass-accounting.md).
4. **Define attachment and exit rules.** Fixed cells represent an attached population. Free cells drift and rotate in the velocity field before contact relaxation. Model rules determine attachment, daughter release, and washout, while stable identifiers retain lineage history.
5. **Run and measure.** Use the [live viewer](../viewer/README.md) to inspect the model, [batch runs](formats/run-manifest-v1.md) to compare parameters and seeds, and [analysis datasets](analysis/recipes.md) to measure the resulting populations and fields. Checkpoints preserve the state needed to resume an experiment.

The [microfluidic trap](../examples/microfluidic_trap.py) is a compact starting model. The [Danino clock](../examples/tutorials/danino_clock.py) adds a quorum-sensing circuit; the [biopixel tutorial](tutorials/microfluidics.md#a-source-backed-prindle-biopixel-example) combines a single-trap model with published cavity dimensions and a supplied CAD layout whose provenance is documented.

## Choosing a flow model

Both solvers calculate steady, inertia-free device flow and normalize the result to a prescribed mean inlet speed. They expose the same velocity-field interface for transport, with native implementations on CPU, Metal, and CUDA. See [backend status](../README.md#backend-status) for current support.

| Model | Appropriate geometry and question | Numerical scope |
| --- | --- | --- |
| [Depth-averaged Hele-Shaw/Darcy flow](architecture/0022-brinkman-flow.md) | Shallow channels and traps; nutrient delivery and flux routing around obstacles | One pressure per contiguous depth column above a common floor. Gap height sets conductance, and an empirical mobility can represent stationary biomass. The model averages across the gap and omits in-plane viscous stresses. |
| [Resolved Stokes-Brinkman flow](architecture/0023-mac-stokes.md) | Velocity profiles and three-dimensional obstructions on meshes that resolve the passages | Face velocities and voxel pressures solve the coupled momentum and continuity equations with no-slip voxel walls and optional stationary porous drag. Accuracy depends on gap resolution and geometry refinement. |

The shallow model lifts integrated fluxes conservatively onto the transport grid. Its reconstructed field does not resolve wall shear. The resolved solver reports momentum and block residuals, divergence, and minimum gap resolution; quantitative use also requires mesh convergence. The [flow tutorial](tutorials/flow-solvers.md) links these choices to analytic benchmarks.

## Coupling flow, transport, and cells

Solutes move through the device by conservative advection and diffusion with reactions and cellular exchange. Solid masks close transport faces at walls, and cell sampling and deposition respect connected fluid space. Nutrient availability can regulate growth, while cells consume nutrient or release signals according to the model's rate equations.

Attached biomass can change flow resistance through a conservatively smoothed density field. This provides an empirical feedback between a stationary population and its nutrient supply. The resistance coefficient, smoothing radius, and flow-refresh interval are explicit model choices. Freely moving cells are excluded from the stationary resistance used by the tutorials.

Free-cell motion uses the local velocity and a finite-aspect Jeffery orientation approximation, followed by contact relaxation. This kinematic coupling approximates rods as equivalent spheroids for rotation. Cell-scale hydrodynamic forces, lubrication, and predictive adhesion or detachment are outside its scope. The [flow-drift design](architecture/0021-flow-drift.md) specifies the approximation and integration limits.

## Cell-occupied extracellular volume

Current native transport stores concentration per full non-wall voxel; cells do not yet exclude extracellular storage. The smoothed biochemical biomass density used for flow resistance is neither bounded geometric occupancy nor a resolved fluid fraction. [ADR 0025](architecture/0025-cell-occupied-volume.md) selects an opt-in coarse geometric-porosity model and provides executable CPU reference cases, including conservative amount remapping as geometry changes. It is a numerical design, not an enabled production feature. Its harmonic face closure and component-level redistribution do not resolve fluid passages around individual cells, membrane transport layers, or displacement flow; geometric, spatial, and timestep refinement remain required.

## Interpreting results

The [analytic flow benchmarks](tutorials/flow-solvers.md#numerical-evidence) test profile convergence, flux routing, and agreement between the solvers in a shared thin-gap regime. The [controlled nutrient study](tutorials/nutrient-validation.md) measures spatial growth, nutrient balance, and sensitivity to grid spacing, timestep, and flow-refresh interval. It isolates attached-population growth and transport; the interactive tutorials exercise division, mechanics, and washout separately.

These studies establish numerical behavior under their stated assumptions. For comparison with an experiment, supply measured geometry, boundary conditions, transport and biological parameters, then check convergence and parameter sensitivity for the quantities being compared. The device tutorials distinguish published dimensions, supplied drawings, and illustrative model choices. The [validation guide](development/validation.md) explains the separate requirements for numerical, backend, and application evidence.
