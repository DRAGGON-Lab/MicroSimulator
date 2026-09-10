# Grid signaling compatibility

This reference compares CellModeller's grid-signaling behavior with the numerical model used by MicroSimulator. The relevant sources are `Signalling/GridDiffusion.py`, `Integration/CLEulerSigIntegrator.py`, `Integration/CLEulerSigIntegrator.cl`, and the signaling examples.

## State and geometry

`GridDiffusion` stores one flattened `float32` field per signal with shape `(signal, x, y, z)`. Grid locations are `origin + index * spacing` and cell values are obtained by trilinear interpolation over the enclosing eight locations. The implementation requires equal spacing in all three axes even though its public arguments appear anisotropic.

Weights outside the grid are set to zero rather than renormalized. A cell near or beyond an edge can therefore sample less than a full partition of unity and can lose production during scatter. The OpenCL path maps invalid corner indices to zero and relies on their zero weight to suppress that alias.

Initial levels are homogeneous per signal. The selected integrator owns the field, while `GridDiffusion` owns its geometry and transport coefficients. Pickles save the flattened signal field but do not define an independent, versioned grid-state schema.

## Crank-Nicolson path

`CLCrankNicIntegrator` constructs the intended operator `I - dt/2 T`, uses SciPy GMRES to approximate a central Green's function, and forms the right-hand side `(I + dt/2 T)c_n + dt f(c_n)`. The final call to `scipy.ndimage.convolve`, however, neither assigns its return value nor supplies an output array. The inverse application is therefore discarded in shipped code. Observable execution leaves only the explicit right-hand side in the signal field.

MicroSimulator treats the comments and operator construction as the recoverable scientific intent and implements the full semi-implicit equation. It does not reproduce the discarded-convolution behavior, truncated Green's function, or legacy coefficient scaling.

## Transport

The explicit integrator asks SciPy to compute a nearest-neighbor Laplacian and then multiplies it by `D / (6 h^2)`. Since SciPy's Laplacian already sums the six neighbor differences, this is one sixth of the conventional discretization for a physical diffusion coefficient `D`. Examples pass values such as `10.0` without documenting whether they compensate for this factor.

Optional advection is a scalar per signal and acts only along the first grid axis through a centered three-point convolution. Advection always uses nearest-edge extension, independently of the diffusion boundary selection. Diffusion delegates boundary behavior to SciPy mode strings; a constant boundary takes the signal's homogeneous initial level as its value. No CFL or positivity condition is checked before forward Euler updates.

These scaling, direction, and boundary mismatches are implementation history, not authoritative physical semantics.

## Cell coupling

For each step the legacy Euler path:

1. dilutes intracellular species using the old/new cell-volume ratio;
2. computes grid transport from the old signal field;
3. samples old grid signals at every cell;
4. evaluates injected species and signal-rate OpenCL source;
5. scatters each cell signal rate with the same trilinear weights;
6. updates grid signals and intracellular species with forward Euler; and
7. resamples updated grid signals for Python-visible cell state.

Rate evaluation is parallel, but grid transport runs through SciPy on the host and scatter is completed by host sorting and segmented summation. The coupled stage is therefore not device resident.

The injected functions receive grid volume, cell surface area, effective cell volume, cell type, species, and sampled signals. Units are not enforced. The examples divide both intracellular and extracellular exchange expressions by grid volume inside user source. There is no typed guarantee that the species and grid exchange terms conserve amount.

## MicroSimulator behavior

MicroSimulator preserves the useful structure: signal-major fields, trilinear sample/scatter, pre-step signal sampling, post-growth species dilution, and a simultaneous Euler update. It does not preserve the extra one-sixth diffusion factor, x-only centered advection, implicit SciPy boundary strings, truncated edge weights, arbitrary OpenCL source injection, or host-side scatter.

Models using these APIs need an explicit translation of their intended coefficients and units. This cannot be automatic because the CellModeller API does not reveal whether a model compensated for the historical scaling choices.
