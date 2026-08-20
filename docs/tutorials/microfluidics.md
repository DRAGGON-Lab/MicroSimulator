# Microfluidic devices: walls, flow, and washout

This tutorial builds models that live inside devices: geometry that confines cells, blocks
chemistry, and carries media. Four examples cover the range:

| Model | Device | Demonstrates |
| --- | --- | --- |
| [`examples/culture_dish.py`](../../examples/culture_dish.py) | round dish | one inside-cylinder constraint as a dish |
| [`examples/microfluidic_trap.py`](../../examples/microfluidic_trap.py) | trap + channel | flow, obstacles, drift, washout |
| [`examples/tutorials/danino_clock.py`](../../examples/tutorials/danino_clock.py) | trap + channel | the full quorum clock in a device |
| [`examples/tutorials/biopixel_trap.py`](../../examples/tutorials/biopixel_trap.py) | biopixel array trap | reported cavity, CAD layout, monolayer model |

Run any of them live:

```console
uv run cm view --model examples/microfluidic_trap.py --seed 42 --dt 0.02 --backend metal --open
```

## Walls that cells and chemistry both respect

Mechanical walls are typed external constraints: infinite planes, spheres, axis-aligned
boxes, and z-aligned cylinders, each with an inside or outside permitted region. A round
culture dish is a single inside cylinder whose barrel is the wall and whose caps confine the
monolayer:

```python
dish = CylinderConstraintInit()
dish.radius = 30.0
dish.half_height = 1.0
dish.allowed_region = ConstraintRegion.INSIDE
simulation.add_cylinder_constraint(dish)
```

Constraints alone are invisible to signals. The signal grid's obstacle mask closes every
lattice face between fluid and solid voxels, so diffusion and advection stop at walls, and
sampling near a wall renormalizes over fluid sites. Keeping the mask consistent with the
constraints is an authoring concern, which the device helpers handle.

## Devices from one description

`cellmodeller2.microfluidics.TrapChannelDevice` describes an open-sided trap fed by a
straight channel and projects that one description into every engine input:

```python
from cellmodeller2.microfluidics import TrapChannelDevice

DEVICE = TrapChannelDevice(mean_flow_speed=20.0)
DEVICE.add_constraints(simulation)                     # box walls for mechanics
DEVICE.apply_to_grid(grid, inlet_values=[10.0], outlet_values=[0.0])
```

`apply_to_grid` materializes the solid mask, fixed inlet and outlet boundaries on the y axis, and the numerically solved steady device flow on the grid's face-staggered velocity field (see the next section). Flow runs through the channel, circulates weakly at the open trap face, and the dead-end trap exchanges with the channel chiefly by diffusion in this model.

## Flow on signals and on cells

The velocity field advects every signal with conservative upwind face fluxes under both
integrators. Cells feel the same field through explicit drift: with
`MechanicsConfig(flow_drift=True)`, the controller advects each non-fixed cell by the fluid
velocity sampled at its endpoints before contact relaxation, so escaped cells travel down the
channel and rods rotate in shear. Contact relaxation then resolves any overlap the drift
produced against walls or neighbors.

## Washout

Cells that reach the end of the channel leave the system through plan removals:

```python
def _regulate(step):
    divisions = DIVISION.requests(step)
    washed = tuple(cell.id for cell in step.cells if abs(cell.position.y) > WASHOUT_Y)
    if washed:
        DIVISION.forget(step, washed)
        divisions = tuple(r for r in divisions if r.parent_id not in washed)
    return StepPlan(updates=..., divisions=divisions, removals=washed)
```

Removal keeps stable identifiers and lineage history, so analysis can count washout events
and trace removed cells' ancestry from checkpoints.

## Numerical flow: arbitrary geometry and colony feedback

Device flow fields are solved, not authored: `cellmodeller2.flow` computes the steady
Hele-Shaw–Brinkman problem over the grid's fluid voxels and returns the same face-staggered
field the engine consumes. `apply_to_grid` runs this solve for every device, and it works
for any mask geometry — junctions, bends, pillars, a CAD-derived layout — not just straight
channels. The solver is also available directly for grids built without a device helper:

```python
from cellmodeller2.flow import colony_mobility, solve_flow_field

field, report = solve_flow_field(grid, mean_inlet_speed=20.0)   # Stokes limit
grid.velocity_field = field
```

The solve is a variable-coefficient pressure problem (`div(m grad p) = 0`), so the returned
fluxes conserve mass per voxel and vanish on wall faces by construction; the flow-axis
boundaries must be `FIXED` to act as inlet and outlet, and the linear solution is rescaled to
the requested mean inlet speed. With uniform mobility this is the Stokes limit of the
depth-averaged closure — correct routing through any mask, plug profile across the channel
width (side-wall boundary layers, of order the gap height, are outside the closure).

The mobility field is where Brinkman feedback enters: `colony_mobility` rasterizes the
colony's volume fraction and adds Kozeny–Carman style drag, so media diverts around a packed
trap and seeps through its edges. Because the field is data, regulation code re-solves as
the colony grows and swaps it into the running simulation — the trap models do this every
`RESOLVE_INTERVAL` steps:

```python
def _regulate(step: ControllerStep) -> StepPlan:
    if step.completed_steps and step.completed_steps % RESOLVE_INTERVAL == 0:
        mobility = colony_mobility(GRID, step.cells, drag_coefficient=DRAG_COEFFICIENT)
        field, _ = solve_flow_field(GRID, mean_inlet_speed=FLOW_SPEED, mobility=mobility)
        step.simulation.set_velocity_field(field)
    ...
```

`Simulation.set_velocity_field` validates the replacement against the full grid
specification before swapping it; transport, drift, and checkpoints all use whichever field
is current. The re-solve cadence is a model choice, and the trade is cost against staleness:
the trap models re-solve every hundred steps, which at `dt = 0.02` is a couple of doublings
of colony growth, so the field the drift and transport see lags the colony by that much.
Shorten the interval where the blockage matters quantitatively. The drag coefficient is a
modeling parameter (how strongly a packed colony resists through-flow relative to the open
channel), not a measured constant.

### Resolved flow: the MAC Stokes–Brinkman solver

When a study needs the flow the closure cannot express — viscous boundary layers on side
walls, the true cross-channel profile, resolved wall shear — `cellmodeller2.stokes` solves
the full staggered-grid Stokes–Brinkman problem with the same call shape and returns the
same engine-ready field:

```python
from cellmodeller2.stokes import colony_drag, solve_stokes_field

field, report = solve_stokes_field(grid, mean_inlet_speed=20.0)
field, report = solve_stokes_field(
    grid, mean_inlet_speed=20.0, drag=colony_drag(grid, cells, drag_coefficient=0.4)
)
```

It costs far more than the Hele-Shaw solve, so devices keep the closure for authoring and
in-loop feedback; the MAC solver anchors it where the grid resolves the gap. Each solve
reports `min_gap_voxels`, the fluid voxels across its narrowest channel: below about four
the MAC solve over-predicts that channel's flux and the depth-averaged closure is the more
accurate model, which is why the shallow device grids here stay on the closure. Both
solvers run against literature and exact references in `scripts/run_flow_benchmarks.py` —
plane Poiseuille and the two-layer Brinkman channel against their exact solutions with
measured second-order convergence, the Shah–London square-duct peak-to-mean ratio, and a
thin-gap cross-check in which the depth-averaged MAC solution reproduces the Hele-Shaw flux
split around a pillar:

```console
uv run python scripts/run_flow_benchmarks.py          # CI-gating benchmark table
uv run python scripts/run_flow_benchmarks.py --fine   # doubled resolutions
```

The next tutorial, [Solved flow](flow-solvers.md), exercises all of this machinery on a
pillar-array channel built without any device helper.

## A source-backed Prindle biopixel example

The [`prindle.dwg` and `prindle.dxf` files](devices) supplied with this tutorial are associated with the sensing-array project reported by Prindle et al. in [Nature 481, 39–44 (2012)](https://www.nature.com/articles/nature10722). Their provenance is recorded beside the files. The repository does not assert that this drawing is the exact fabrication revision used for the published experiments.

The example deliberately separates three kinds of information:

| Basis | Values used or observed | Role in the example |
| --- | --- | --- |
| Published methods | trapping region 100 x 85 x 1.65 micrometers; 25-micrometer trap spacing; nominal arrays of 500 and 12,000 biopixels | source of the modeled cavity dimensions and context for the array scale |
| Supplied CAD | 496 matching model-space `Layer-2` outlines in a 16 x 31 layout; raw outline size 0.110 x 0.100 drawing units; raw row pitch 0.125 | validates the supplied layout and its source-specific scale, but does not define cavity walls or layer thicknesses |
| Model choices | one 100 x 85 x 1.65 cavity beside a 100 x 10 x 300 micrometer channel; 10-micrometer numerical walls; mean inlet speed 20 micrometers per model time unit; chosen nutrient, drag, and re-solve parameters | defines a qualitative single-trap simulation, not a calibrated reconstruction of the experimental device |

The trapping-region dimensions and spacing come from the [published supplementary methods](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnature10722/MediaObjects/41586_2012_BFnature10722_MOESM313_ESM.pdf), not from subtracting a guessed wall inset from the CAD. `BiopixelTrapDevice` therefore defaults to a 100 x 85 x 1.65 micrometer cavity. Its channel dimensions, wall thickness, and flow speed remain ordinary constructor parameters:

```python
from cellmodeller2.microfluidics import BiopixelTrapDevice

DEVICE = BiopixelTrapDevice(mean_flow_speed=20.0)
```

### Reading the supplied CAD layout

`cellmodeller2.masks` is a bounded, data-only reader for model-space `LWPOLYLINE` geometry. It returns drawing coordinates unchanged unless the caller provides an explicit, source-specific `unit_scale`:

```python
from cellmodeller2.masks import extract_rectangles, load_mask_polylines, match_rectangles

polylines = load_mask_polylines("docs/tutorials/devices/prindle.dxf")
raw_rectangles = extract_rectangles(polylines, layer="Layer-2")
raw_traps = match_rectangles(raw_rectangles, 0.110, 0.100, tolerance=0.001)

rectangles_um = extract_rectangles(polylines, layer="Layer-2", unit_scale=1000.0)
traps_um = match_rectangles(rectangles_um, 110.0, 100.0, tolerance=1.0)
```

For this file, treating one drawing unit as one millimeter is an inference corroborated by the publication: the raw 0.100 outline dimension maps to the reported 100-micrometer trap dimension, and the raw 0.125 row pitch maps to that dimension plus the reported 25-micrometer spacing. The DXF also stores `$INSUNITS=1`; [Autodesk documents `INSUNITS` as automatic insertion-scaling metadata and code 1 as inches](https://help.autodesk.com/cloudhelp/2026/ENU/AutoCAD-Core/files/GUID-A58A87BB-482B-4042-A00A-EEF55A2B4FD8.htm), which does not reconcile with these feature sizes. The reader therefore does not infer physical units from this header or impose the conversion on other drawings.

Both raw and scaled queries yield 496 outlines in a 16 x 31 layout. Their centers span 2.4 x 3.75 millimeters after the inferred conversion; the rows have a 125-micrometer pitch, while column pitches are 135, 160, or 172.5 micrometers. The published device is described nominally as having 500 biopixels, so the documented result preserves the distinction between the paper's nominal count and this file's exact count.

With `include_blocks=True`, the reader also exposes geometry in unplaced block definitions and records each block name. It does not apply `INSERT` transforms. The supplied file contains substantial `Layer-5` block geometry, but without a process map the tutorial does not assign that layer a physical role or infer cross-layer registration from it.

The executable example loads and checks this layout, then simulates one cavity using the independently published dimensions. That single-trap reduction assumes one selected local inlet condition; it does not assert uniform flow across the array, reproduce the array manifold, or include inter-trap coupling. Run it live:

```console
uv run cm view --model examples/tutorials/biopixel_trap.py --seed 5 --dt 0.02 --backend metal --open
```

## Units and timescales

Model lengths are expressed in micrometers. Only the 100 x 85 x 1.65 trapping region is taken from the published methods; the table above identifies the remaining geometry and transport inputs as model choices.

Time is a model growth scale. `growth_rate` is the exponential rate of cell length, so `BASE_GROWTH_RATE = 1.0` gives a doubling time of `ln 2 ≈ 0.69` model time units. Mapping that doubling to a biological duration, such as 30 minutes, is illustrative and would make one model time unit about 43 minutes; it is not a calibration performed by this example. Nutrient and AHL levels are dimensionless concentration scales set by their inlet values and coupling parameters.

For the biopixel example's configured channel values, `U = 20`, `L = 100`, and `D = 40` give a nominal channel-scale Péclet number `U L / D = 50`. That number characterizes this model only. Velocity is nonuniform, flow inside the dead-end cavity is much weaker, and no experimental flow or diffusivity measurements are fitted here, so the example makes no claim of experimental Péclet-number fidelity.

The model also does not reproduce an experimentally established separation between transport and growth timescales. Its initial signal field is primed with inlet media, and its transport coefficients are chosen for a tractable tutorial run. Quantitative comparison with an experiment would require measured boundary conditions and material properties, grid and timestep convergence, and sensitivity analysis over the channel, transport, drag, and feedback parameters.

## Numerical guidance

- Choose `dt` so the largest per-step drift, `max_speed * dt`, stays below a cell radius;
  `solve_flow_field` reports `max_speed`, and the trap examples use `dt = 0.02` with a mean
  channel speed of 20.
- Forward Euler enforces its stability bound from the per-site advective outflow; the trap
  models select Crank-Nicolson.
- The implicit solve's relative tolerance is the accuracy the step delivers: it asks for
  that reduction of the residual the step starts with, so a model gets what it asked for
  regardless of its concentration scale. These models keep the engine defaults.
- Let the lattice of site centers cover every position a cell can reach, with about a voxel
  of margin past each wall: contact relaxation lets a crowded cell press slightly into a
  wall, and sampling outside the lattice is an error.
- Keep the mechanics walls enclosing the solid mask. The device helpers voxelize
  conservatively — a site is solid only when its whole voxel lies inside a wall — so the
  voxel holding any reachable position stays fluid and a cell against a wall always has a
  fluid site to sample. A hand-built mask needs the same rule; the
  [pillar channel](flow-solvers.md) shows it for curved walls.
- A sampling position whose whole stencil is solid raises an error rather than returning
  zero.
