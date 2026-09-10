# Diffusible signals and cell-cell communication

This tutorial introduces extracellular grids, diffusion, cell-grid exchange, sender-receiver communication, and two-strain mutualism. Run the scenarios in `examples/tutorials/signaling.py` with a small time step such as `0.01`.

## Grid geometry and units

A `SignalGridSpec` declares channel count, lattice shape, physical origin, spacing, diffusion coefficients, advection velocities, integration method, and six boundary conditions. Grid levels are concentrations. A coupled rate plan returns:

- one concentration-per-time derivative for every intracellular species; and
- one amount-per-time extracellular source for every signal.

The engine samples and scatters with the same trilinear weights. It divides an amount source by voxel volume while scattering, so a membrane exchange term can be written once and conserved with opposite signs inside and outside.

The tutorial grids use Crank–Nicolson for transport and no-flux boundaries. Cell reactions and cell-grid exchange remain one simultaneous coupled stage. Grid spacing is four and the single-gene production rate is one.

### Optional affine field reactions

A grid may also declare a `SignalGridAffineReaction`. Its `source_rates` and `loss_rates` arrays have exactly `grid.level_count` entries in the same signal-major layout as the levels. At every lattice site the native signal operator adds

```text
source_rate - loss_rate * concentration.
```

Both coefficient arrays must be finite and non-negative. A relaxation toward a non-negative target `C` at rate `k`, `k(C - c)`, is represented by source `k*C` and loss `k`. Coordinate masks are authoring-time operations that fill these arrays; they are not runtime callbacks. The complete arrays are validated and checkpointed, so backend changes and resume cannot alter which sites react.

## 1. A signaling gene in a chamber

```console
uv run microsimulator view \
  --model examples/tutorials/signaling.py \
  --parameter scenario='"single_gene"' \
  --seed 42 \
  --dt 0.01 \
  --open
```

There is one intracellular pool `x` and one extracellular signal `s`. With cell surface area `A` and voxel volume `Vg = 4^3 = 64`, define the amount flux

```text
J = 0.1 (s - x) A.
```

The coupled equations are

```text
dx/dt = 1 + J / Vg
grid source = -J.
```

When `x > s`, `J` is negative: the cell loses concentration and the grid gains the corresponding amount. The extracellular output is an amount rate, and the engine performs the grid-volume conversion exactly once.

Two inward-facing planes at `y = -16` and `y = 16` confine cells. The signal grid and mechanical chamber are independent declarations; matching their extents is a model choice, not an implicit renderer behavior.

## 2. Sender-receiver communication

```console
uv run microsimulator view \
  --model examples/tutorials/signaling.py \
  --parameter scenario='"communication"' \
  --seed 42 \
  --dt 0.01 \
  --open
```

The founders are type 0 at `x = -10` and type 1 at `x = 10`. The channels are:

|   Channel | Meaning                        |
| --------: | ------------------------------ |
| species 0 | intracellular signal pool `x0` |
| species 1 | sender reporter (RFP)          |
| species 2 | receiver reporter (GFP)        |
|  signal 0 | extracellular signal           |

Type 0 constitutively produces `x0` and species 1. Type 1 imports `x0` and produces species 2 through

```text
x0^2 / (5e-5 + x0^2).
```

Use cell-type coloring to identify sender and receiver lineages, species channel 1 or 2 for reporter concentrations, and a signal-grid slice for the shared field. Viewer color scales are frame-relative; export numerical data before comparing intensity across times or runs.

## 3. Two-strain mutualism

```console
uv run microsimulator view \
  --model examples/tutorials/signaling.py \
  --parameter scenario='"mutualism"' \
  --seed 42 \
  --dt 0.01 \
  --open
```

Type 0 produces alpha and requires beta; type 1 produces beta and requires alpha. Each chemical has an intracellular pool and an extracellular grid channel. Membrane coefficients and production rates are one in the tutorial.

The regulation callback uses the partner’s intracellular concentration:

```text
growth = 0.1 + 0.9 partner / (0.1 + partner).
```

This is a saturating phenomenological mutualism rule, not a biomass or nutrient balance. Compare it against a control with constant growth rates before attributing increased intermixing to cooperation.

## Numerical checks

- Confirm all cells remain inside the closed grid bounds. An out-of-domain cell is a model error and fails before mutation.
- For Forward Euler grids, obey the documented diffusion/advection/loss stability bound. These lessons select Crank–Nicolson transport, but coupled cell sources are still explicit and can require a smaller `dt`.
- Export `signals.zarr` and inspect named `(frame, channel, x, y, z)` axes. Never infer axis order from flattened storage.
- Check conservation for a model with production disabled: integrated grid amount plus intracellular amount should change only through declared boundaries and numerical tolerance.

## Experiments

- Set the mutualist growth response to a constant one and compare mixing at the same total cell count.
- Give the signal grid fixed reservoir boundaries and state the reservoir concentration explicitly.
- Add advection to the chamber model. Interpret velocity in physical distance per simulation-time unit and check the upwind stability contribution.
- Change grid spacing without changing a membrane amount flux. Explain why the concentration change per voxel changes with voxel volume.
