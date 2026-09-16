# ADR 0023: resolved Stokes-Brinkman flow

- Status: accepted
- Date: 2026-08-18
- Amended: 2026-09-10

## Decision

`microsimulator.stokes` solves `mu lap(v) - mu d v - grad(p) = 0`, `div(v) = 0` on a staggered Cartesian grid, with velocity on faces and pressure at voxel centers. The inverse permeability d has units of inverse length squared. Stationary biomass can contribute empirical Brinkman drag using the conserved, smoothed density defined in ADR 0022. The closure needs calibration and does not describe freely moving cells or resolved cell-scale hydrodynamic forces.

No-slip walls follow obstacle voxel boundaries. Normal wall velocities are zero and tangential components use reflected ghosts. Collapsed axes are invariant directions, not resolved one-cell gaps. Flow-axis boundaries prescribe ghost pressures with zero-gradient normal velocity; developing inlet flow requires upstream padding. Linear solutions are scaled to the requested mean inlet speed, removing the pressure amplitude and viscosity from the velocity comparison.

Write momentum as `A v + G p = f`, continuity as `D v = 0`, with `G = -D^T` for the uniform-grid discrete inner products and nonnegative symmetric A on active velocity faces. A known linear inlet-to-outlet pressure profile is removed analytically, so the unknown pressure is a correction and the forcing is distributed through the channel. Flexible GMRES solves this fixed block system. Momentum CG and a diagonal approximation to `-D diag(A)^-1 G` act only as a variable block preconditioner. An inexact inner solve therefore cannot change the outer operator. Continuity is scaled by inverse minimum spacing to balance the block residual. Restart length is 40, with twice-modified Gram-Schmidt. Convergence requires a freshly computed residual of the original block equations, not just a recursive Krylov estimate.

CPU, Metal, and CUDA keep their own native operator and vector implementations; accelerator Krylov vectors remain on device. The report includes the true scaled block relative residual, momentum relative residual, physical divergence RMS after speed normalization, iteration counts, and minimum transverse gap resolution. Field storage is binary32; the default outer and inner tolerances are `1e-6`. Fine binary32 second differences can prevent convergence at `1e-6`; the fine Brinkman benchmark explicitly requests `1e-5`, well below its discretization error. An unattainable requested tolerance causes a reported failure and is never silently relaxed. A loose inner tolerance affects performance but should preserve the converged field within the outer tolerance and binary32 error. Disconnected inlet/outlet geometry is rejected; unforced sealed components remain at zero flow with an arbitrary pressure gauge.

## Validation and limits

The analytic benchmark script and Python tests measure plane-Poiseuille and two-layer Brinkman profile convergence, the square-duct peak/mean ratio, shallow/resolved consistency, and gap-resolution error. Native tests check operator adjointness, viscous symmetry and nonnegative energy; conformance compares heterogeneous native backend fields. A separate loose-inner-tolerance regression checks true residuals and solution invariance.

The model assumptions, staircase geometry, boundary treatment, and discretization all limit accuracy. Planar aligned-wall profile convergence does not establish second-order accuracy around arbitrary curved obstacles. Device helpers classify voxel centers against the same continuous geometry used by mechanics; planar wall displacement is at most half a spacing, while narrow or curved features require geometry convergence. A one-voxel gap does not resolve a parabolic profile. `min_gap_voxels` is a diagnostic, not an accuracy certificate or an automatic solver-selection rule. Experimental PIV comparisons must include sampling uncertainty and useful baselines; qualitative agreement alone does not validate colony coupling.
