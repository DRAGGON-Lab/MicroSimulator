# Microfluidic devices: walls, flow, and washout

This tutorial connects device geometry, flowing media, and cell biology in runnable MicroSimulator models. The [modeling guide](../microfluidics.md) introduces the workflow and the choice of flow solver. Four examples cover the range:

These models use XY-only division jitter and finite-height 3D confinement. A thin cavity can encourage a monolayer, but it does not force a common center Z or eliminate tilt; walls are soft constraints whose residual depends on relaxation tolerance and passes. The [dimensionality audit](planarity.md) lists each device and reproduces these distinctions.

The microfluidic-trap, Danino, biopixel, and pillar tutorial founders request centerline length 3.5, capped at their single sampled target in [3.2, 3.8]. Attachment, position, radius, and concentrations are preserved. This affects new construction only; saved geometry is restored unchanged. See [founder initialization and volume conventions](biophysics-and-growth.md#length-and-volume).

| Model | Device | Demonstrates |
| --- | --- | --- |
| [`examples/culture_dish.py`](../../examples/culture_dish.py) | round dish | one inside-cylinder constraint as a dish |
| [`examples/microfluidic_trap.py`](../../examples/microfluidic_trap.py) | trap + channel | flow, obstacles, drift, washout |
| [`examples/tutorials/danino_clock.py`](../../examples/tutorials/danino_clock.py) | trap + channel | the full quorum clock in a device |
| [`examples/tutorials/biopixel_trap.py`](../../examples/tutorials/biopixel_trap.py) | biopixel array trap | reported cavity, CAD layout, monolayer model |

Run any of them live:

```console
uv run microsimulator view --model examples/microfluidic_trap.py --seed 42 --dt 0.02 --backend metal --open
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

`microsimulator.microfluidics.TrapChannelDevice` describes an open-sided trap fed by a
straight channel and projects that one description into every engine input:

```python
from microsimulator.microfluidics import TrapChannelDevice

DEVICE = TrapChannelDevice(mean_flow_speed=20.0)
DEVICE.add_constraints(simulation)                     # box walls for mechanics
DEVICE.apply_to_grid(
    grid,
    inlet_values=[10.0],
    outlet_values=[0.0],
    simulation=simulation,
)
```

`apply_to_grid` materializes the solid mask, fixed inlet and outlet boundaries on the y axis, and the numerically solved steady device flow on the grid's face-staggered velocity field (see the next section). Passing the model's `Simulation` makes the solve execute through the backend selected by the runner. Flow runs through the channel, circulates weakly at the open trap face, and the dead-end trap exchanges with the channel chiefly by diffusion in this model.

## Flow on signals and on cells

The velocity field advects every signal with conservative upwind face fluxes under all three signal integrators. With `MechanicsConfig(flow_drift=True)`, the controller translates each non-fixed cell using the velocity sampled at its center and rotates it with a finite-aspect Jeffery approximation, so escaped cells travel down the channel and rods rotate in shear. Contact relaxation then resolves overlaps with walls or neighbors. The [flow-drift design](../architecture/0021-flow-drift.md) describes this kinematic approximation and its integration limits.

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

## Numerical flow and stationary resistance

`microsimulator.flow` solves `div_xy(H*m*grad_xy(p)) = 0` with one pressure per depth column. The default mobility is proportional to H², so integrated flux has the required H³ gap dependence. Harmonic face conductance and conservative lifting provide the face field used by transport. The shallow solver requires contiguous columns above a common floor; full three-dimensional obstructions require `microsimulator.stokes`. Both normalize velocity to a prescribed inlet speed and execute on the selected native backend.

```python
from microsimulator.flow import colony_mobility, solve_flow_field

mobility = colony_mobility(
    grid, (cell for cell in cells if cell.fixed),
    drag_coefficient=100.0, averaging_radius=4.0,
)
field, report = solve_flow_field(
    grid, mean_inlet_speed=20.0, mobility=mobility, simulation=simulation,
)
simulation.set_velocity_field(field)
```

Stationary resistance uses explicitly attached cells. A trapped but freely moving population is not automatically a stationary porous matrix. The four examples apply that distinction; the pillar model supplies attached founders. Biomass is deposited conservatively over a fixed physical radius, and only the resistance formula caps density. The drag coefficient and smoothing radius require calibration; grid size and refresh interval require numerical sensitivity checks.

For resolved wall profiles use `solve_stokes_field`, with the same field interface and adequately resolved gaps. It solves a fixed Stokes-Brinkman block system by flexible GMRES and reports true momentum/block residuals and divergence. `min_gap_voxels` is diagnostic rather than a guarantee. The [flow tutorial](flow-solvers.md) gives the solver assumptions and analytic checks, and [nutrient validation](nutrient-validation.md) measures spatial growth, conservation, and refinement effects.

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
from microsimulator.microfluidics import BiopixelTrapDevice

DEVICE = BiopixelTrapDevice(mean_flow_speed=20.0)
```

### Reading the supplied CAD layout

`microsimulator.masks` is a bounded, data-only reader for model-space `LWPOLYLINE` geometry. It returns drawing coordinates unchanged unless the caller provides an explicit, source-specific `unit_scale`:

```python
from microsimulator.masks import extract_rectangles, load_mask_polylines, match_rectangles

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
uv run microsimulator view --model examples/tutorials/biopixel_trap.py --seed 5 --dt 0.02 --backend metal --open
```

## Units and timescales

Model lengths are expressed in micrometers. Only the 100 x 85 x 1.65 trapping region is taken from the published methods; the table above identifies the remaining geometry and transport inputs as model choices.

Time is a model growth scale. `growth_rate` is the exponential rate of cell length, so `BASE_GROWTH_RATE = 1.0` doubles cylindrical length in `ln 2 ≈ 0.69` model time units. Biochemical biomass includes an end contribution and therefore does not obey that exact exponential law. Mapping that doubling to a biological duration, such as 30 minutes, is illustrative and would make one model time unit about 43 minutes; it is not a calibration performed by this example. Nutrient and AHL levels are dimensionless concentration scales set by their inlet values and coupling parameters.

For the biopixel example's configured channel values, `U = 20`, `L = 100`, and `D = 40` give a nominal channel-scale Péclet number `U L / D = 50`. That number characterizes this model only. Velocity is nonuniform, flow inside the dead-end cavity is much weaker, and no experimental flow or diffusivity measurements are fitted here, so the example makes no claim of experimental Péclet-number fidelity.

The model also does not reproduce an experimentally established separation between transport and growth timescales. Its initial signal field is primed with inlet media, and its transport coefficients are chosen for a tractable tutorial run. Quantitative comparison with an experiment would require measured boundary conditions and material properties, grid and timestep convergence, and sensitivity analysis over the channel, transport, drag, and feedback parameters.

## Numerical guidance

- Choose `dt` so the largest per-step drift, `max_speed * dt`, stays below a cell radius;
  `solve_flow_field` reports `max_speed`, and the trap examples use `dt = 0.02` with a mean
  channel speed of 20.
- Forward Euler enforces its stability bound from the per-site advective outflow; the trap models select backward Euler for stiff transport and affine losses. Explicit cellular uptake still constrains the timestep.
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

## Uptake and time integration

The tutorials consume the actual biochemical biomass increment `Delta B / yield`, where `B = pi*r²*(length + 2*r)`. Division conserves this amount, which is distinct from geometric capsule volume. `growth_rate * B` is not the realized biomass rate under the length-growth law.

Backward Euler is the baseline for their stiff transport and affine losses. Cellular sinks remain explicit; native biological failure restores growth, species, signals, time, and the prior solver report. Controller regulation, division callbacks, and mechanics are separate operations. No rejected step may be counted as successful growth. The Danino circuit is a qualitative example: its AHL secretion uses intracellular concentration times B, and its AiiA loss uses a conservatively smoothed enzyme amount. Its parameters and oscillations are not experimentally calibrated.

The biopixel model explicitly uses signal absolute residual tolerance `1e-5` because binary32 noise at concentration 10 and its fine depth spacing prevents reliable convergence at `1e-6`. This is a model-scale numerical choice, not a biological accuracy claim.
