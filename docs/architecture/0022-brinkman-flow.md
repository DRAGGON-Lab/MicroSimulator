# ADR 0022: depth-integrated shallow flow

- Status: accepted
- Date: 2026-08-16
- Amended: 2026-09-10

## Decision

`microsimulator.flow` solves one pressure unknown per fluid x/y column. For gap height H and depth-averaged mobility m, the equations are `div_xy(q) = 0`, `q = -H m grad_xy(p)`. The default mobility is proportional to H², giving the required cubic gap conductance H³. Supplied mobility must be constant through each depth column. Harmonic face conductances represent series resistance. This is a lubrication/Darcy closure: it omits in-plane viscous stresses and requires a shallow gap and slowly varying geometry for physical accuracy. It is not the full Brinkman equation.

Supported masks have contiguous fluid columns above a common planar floor, with flow along x or y. Overhangs, vertically disconnected channels, and depth-varying mobility are rejected; they require resolved flow. H is the authored staircase height, not an exact continuous-device height. Geometry convergence remains necessary, particularly near sharp depth changes.

Pressure is fixed at the flow-axis inlet and outlet, with no flux elsewhere. Native CPU, Metal, and CUDA operators solve the reduced system by Jacobi-preconditioned conjugate gradient. Accelerator vectors stay on device. Fields use binary32, with default pressure residual tolerance `1e-6`. Disconnected inlet/outlet geometry and periodic boundaries are rejected.

The depth-integrated horizontal face flux is distributed over the overlapping open depth. A reconstructed vertical flux balances horizontal redistribution within each column, while distributing the small pressure residual over depth and keeping floor and roof closed. This is a conservative transport reconstruction, not a resolved vertical velocity profile. Conservation holds to pressure-solve and floating-point error; pressure residual alone is not a relative flux error bound. The complete field is scaled to the requested mean inlet speed.

`colony_volume_fraction` deposits the conserved biochemical biomass B defined in ADR 0024, using exact voxel integrals of a separable tent kernel with a fixed physical averaging radius (default 4 length units). Masked support is restricted to a face-connected fluid component and renormalized to conserve amount. The raw density is never clipped. `colony_mobility` depth-averages this density and caps it only inside the empirical Kozeny-Carman resistance closure. Its coefficient and averaging radius require calibration. `colony_species_density` deposits concentration times B with the same kernel.

The caller decides which biomass is stationary and belongs in a stationary resistance model. Freely advected cells are not automatically a porous matrix. `Simulation.set_velocity_field` validates and atomically swaps a field; the caller chooses and convergence-tests its refresh interval.

## Validation

Tests cover cubic flux partition between unequal depths, flux continuity and uniform-tracer preservation across a depth step, invalid column geometry, mass-conserving deposition and exact aggregation under refinement, native backend conformance, and the uniform duct/pillar benchmarks. `scripts/run_flow_benchmarks.py` compares the closure with resolved flow in its shared regime. Shallow flow does not supply no-slip wall shear or near-wall rod velocities.
