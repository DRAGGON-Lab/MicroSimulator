# Microfluidic devices: walls, flow, and washout

This tutorial connects device geometry, flowing media, and cell biology in runnable MicroSimulator models. The [modeling guide](../microfluidics.md) introduces the workflow and the choice of flow solver. Four examples cover the range:

These models use XY-only division jitter and finite-height 3D confinement. A thin cavity can encourage a monolayer, but it does not force a common center Z or eliminate tilt; walls and neighbors are held apart by contact forces, and a contact overlaps by its force divided by its stiffness, with walls a hundred times stiffer than cells by default. The [dimensionality audit](planarity.md) lists each device and reproduces these distinctions.

For backend selection, PowerShell syntax, quoted JSON parameters, and paths with spaces, see [tutorial commands by backend and shell](commands.md#choose-a-shell). Multiline commands on this page use POSIX shell backslashes; the guide provides the PowerShell equivalents and [explicit CPU, Metal, and CUDA trap launches](commands.md#run-the-same-trap-on-cpu-metal-or-cuda).

The microfluidic-trap, Danino, biopixel, and pillar tutorial founders request centerline length 3.5, capped at their single sampled target in [3.2, 3.8]. Attachment, position, radius, and concentrations are preserved. This affects new construction only; saved geometry is restored unchanged. See [founder initialization and volume conventions](biophysics-and-growth.md#length-and-volume).

| Model | Device | Demonstrates |
| --- | --- | --- |
| [`examples/culture_dish.py`](../../examples/culture_dish.py) | round dish | one inside-cylinder constraint as a dish |
| [`examples/microfluidic_trap.py`](../../examples/microfluidic_trap.py) | monolayer trap + channel | flow, obstacles, drift, crowding-limited growth, washout |
| [`examples/tutorials/danino_clock.py`](../../examples/tutorials/danino_clock.py) | trap + channel | the full quorum clock in a device |
| [`examples/tutorials/biopixel_trap.py`](../../examples/tutorials/biopixel_trap.py) | biopixel array trap | reported cavity, CAD layout, monolayer model |

A device can also be read from a chip design; see [a device from a chip design](#a-device-from-a-chip-design).

Run any of them live:

```console
uv run microsimulator view --model examples/microfluidic_trap.py --seed 42 --dt 0.02 --backend cpu --open
```

The trap example is sized to one population: a 40 by 20 micrometer trap, 1.5 micrometers deep, beside a 20 micrometer channel. A few hundred cells fill it as a monolayer. It solves growth and contact together with [contact dynamics](../architecture/0026-contact-dynamics.md): cells push each other toward the mouth with contact forces, and a cell's elongation falls linearly with the axial load on it, stopping at a stall load. The cells at the back carry the load of everything between them and the mouth, so they slow and stall, while cells at the mouth keep growing, spill into the channel, and wash out. The trap holds about 260 cells with continuous turnover. The stall load is a model choice, not a measured mechanical response.

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

The constructor's keyword arguments set the trap, channel, and wall dimensions; the trap example passes its own, and the Danino clock uses the defaults. `apply_to_grid` materializes the solid mask, fixed inlet and outlet boundaries on the y axis, and the numerically solved steady device flow on the grid's face-staggered velocity field (see the next section). Passing the model's `Simulation` makes the solve execute through the backend selected by the runner. Flow runs through the channel, circulates weakly at the open trap face, and the dead-end trap exchanges with the channel chiefly by diffusion in this model.

## A device from a chip design

A chip drawn in the [`microfluidics` design tool](https://github.com/marpaia/microfluidics) need not be stated again here. That tool writes each design as a scene, one JSON document holding the chip's rectangles by layer, each layer's height, and the flow it solved. `microsimulator.design` reads the scene as plain data, and `LayoutDevice` projects a window of it into the same engine inputs as the devices above:

```python
from microsimulator.design import load_design_scene
from microsimulator.microfluidics import LayoutDevice

scene = load_design_scene("trap_chamber.scene.json")
trap = scene.instance("trap")                    # one placed region, by its path in the design
window = scene.window(trap.low, trap.high)       # its fluid, in coordinates local to the window
DEVICE = LayoutDevice(window.boxes, window.size, mean_flow_speed=20.0)

origin, shape = DEVICE.lattice((5.0, 5.0, 0.825))
DEVICE.add_constraints(simulation)
DEVICE.apply_to_grid(grid, inlet_values=[10.0], outlet_values=[0.0], simulation=simulation)
```

`python -c "import json; from microfluidics.project import Project; print(json.dumps(Project('trap_chamber.py').scene()))"` writes a design's scene with the design tool installed; MicroSimulator needs only the file.

A window is any rectangle of the chip, in the design's micrometers. Its low corner becomes the device's origin and the glass its floor at `z = 0`, so positions stay small beside the chip's own millimeter coordinates; `window.local(x, y)` maps a point of the design, and `window.origin` is the low corner. Every rectangle is clipped to the window and rises from the floor to its layer's height. Of a row of repeated features, such as the trenches of a mother machine, only those that reach the window are placed.

Without ports, `LayoutDevice` takes media in through the window face named by `inlet` (the lower y face unless stated; any of the four sides) and out through the face opposite. Both must cut fluid, and the channel continues past them as it meets them. The other two sides are walls: fluid the window cuts there ends there, so a window is drawn wide enough to hold what matters across the flow. [Ports](#several-inlets-ports-and-mixers) state several inlets and outlets on any sides. `lattice(spacing)` gives a grid origin and shape whose site centers cover the window with a voxel of wall beyond each side, and `gap_height(x, y)` gives the fluid's height above a point of the floor.

Walls follow the drawn edges exactly, with the half-voxel staircase of any mask. For the window above, which draws the biopixel cavity and channel, the solid mask and solved flow match `BiopixelTrapDevice` site for site on the same lattice.

The scene also carries what the design says of how the chip is run. `scene.flows_of("trap")` gives the flow through a region with its mean speed in millimeters per second, `scene.viscosity_mpa_s` the viscosity, and `scene.fraction(species, vertex)` the share of a fed species arriving at a vertex of the flow graph, from 0 to 1, which scales a reservoir concentration into an inlet value. These are physical quantities, and `mean_flow_speed` is in model units: converting needs the model's choice of seconds per model time unit. A chip flushed at microliters per minute moves its medium at millimeters to hundreds of millimeters per second, many orders above the speeds these tutorials use, and the limit on drift per step under [numerical guidance](#numerical-guidance) then binds for cells loose in the channel. `scene.passes` reports whether the design passed the design tool's own checks; the reader does not refuse one that did not.

The reader takes scenes of format `0.6`, at patch `0.6.0` or later, and refuses any other version, since a field of another version may mean something else. It reads the fields it names, passes over every other, and executes nothing.

## Several inlets: ports and mixers

Where channels meet, as at the junction of a mixer, each inlet carries its own liquid at its own flow rate. A device states that with ports: openings in the window's sides, each with a speed or a pressure and with the media outside it. A design window lists the channels it cuts as `crossings`, which give each port's side and span:

```python
from microsimulator.microfluidics import LayoutDevice, LayoutPort

meet = scene.instance("meet")                    # the junction of a three-liquid mixer
window = scene.window(meet.low, meet.high)
inlets = sorted((c for c in window.crossings if c.side == GridFace.Y_LOWER), key=lambda c: c.low)
(outlet,) = (c for c in window.crossings if c.side == GridFace.X_UPPER)

ports = [
    LayoutPort(c.side, c.low, c.high, speed=speed, values=media)
    for c, speed, media in zip(inlets, speeds, compositions)
]
ports.append(LayoutPort(outlet.side, outlet.low, outlet.high))      # holds pressure zero
DEVICE = LayoutDevice(window.boxes, window.size, ports=tuple(ports))

origin, shape = DEVICE.lattice((50.0, 50.0, 50.0))
DEVICE.apply_to_grid(grid, inlet_values, outlet_values, simulation=simulation)
```

A port with a `speed` carries that mean speed over its open area, positive into the device and negative out of it. A port without one holds a `pressure`, zero unless stated, and carries whatever flow that draws; a device needs at least one, usually its outlet. `values` is the media just outside the port, which enters where the port's flow runs inward. The geometry continues past the window through each port, and the rest of every side is wall. `lattice` puts one layer of sites outside each side, so that each port opens through it.

`apply_to_grid` makes every side that holds a port a fixed boundary with a patch at each port carrying that port's media, and solves the flow through the ports. `DEVICE.solve_flow(grid, simulation=simulation)` solves it again and returns each port's pressure, flux and mean speed; `resolved=True` there, or in `apply_to_grid`, uses the resolved Stokes solver in place of the shallow one. `DEVICE.set_port_values(simulation, port, values)` changes one port's media during a run.

Which crossing is an inlet, at what speed and with what media is the model's to state. The scene supplies what a design knows: `scene.flows_of(channel)` gives each feeding channel's mean speed, and `scene.fraction(species, vertex)` what arrives at a junction. For a mixer whose liquids are carried through in seconds, the natural model time unit is the second, and a speed in millimeters per second is a thousand model units of micrometers per second. The test of the design tool's own mixer builds its junction this way: three inlets at the design's speeds, each with one pure liquid, and at a steady state the stream leaving holds the liquids in the shares the design's flow solution states, to within a percent.

The design tool takes liquids as fully mixed at every vertex. A simulated junction does not: beside each other in one channel, liquids mix only as fast as they diffuse, and the whole chip can be simulated to see how far. [`examples/mixer_device.py`](../../examples/mixer_device.py) builds the mixer's whole chip, with its three inlets held at one pressure so that the split among the liquids is the flow solver's own: 19.7, 29.8 and 50.5 percent against the design's 20, 30 and 50.

How far the liquids have mixed is a harder question than how they split, and the answer depends on the grid and the advection scheme. [`scripts/run_mixer_study.py`](../../scripts/run_mixer_study.py) measures the segregation of the half-share liquid across the mixing channel, 1 for streams side by side and 0 for a blend, at three sections, and compares it with what diffusion alone gives a plug flow in a channel of that width:

| Spacing | Sites across | Scheme | Just joined | After one bend | Serpentine end |
| --- | --- | --- | --- | --- | --- |
| 50 | 4 | upwind | 0.323 | 0.013 | 0.000 |
| 50 | 4 | van Leer | 0.493 | 0.116 | 0.000 |
| 25 | 8 | upwind | 0.449 | 0.059 | 0.000 |
| 25 | 8 | van Leer | 0.586 | 0.252 | 0.009 |
| 12.5 | 16 | upwind | 0.529 | 0.131 | 0.000 |
| 12.5 | 16 | van Leer | 0.621 | 0.296 | 0.018 |
| | | expected | 0.646 | 0.303 | 0.019 |

Three things follow. First-order upwinding blends the streams at each bend, and refinement alone recovers that slowly: at 16 sites across it is still worse than the limited scheme at 8. The limited scheme at 16 sites across reproduces the expected values to within a few percent, and at 8 it is close where the streams are still distinct. And for this mixer the answer itself is that the liquids leave mostly but not wholly mixed: the difference between the streams has fallen to about 15 percent of what it began as, because the serpentine holds the stream for nearly two of diffusion's own time constants across the channel. A mixing question therefore needs the limited scheme, a spacing that puts about sixteen sites across the channel, and this kind of check; four sites across answers the split and nothing about the mixing. Transport and the flow solve [visit the fluid sites alone](../architecture/0029-transport-over-fluid-sites.md), so the fine grid costs what its channels cost and not what the chip's whole box does: the study's finest row takes 20 to 25 seconds, flow solve included. The expected values assume the shallow solver's plug flow; a real channel's velocity profile adds dispersion that neither they nor this solver include.

Underneath, `microsimulator.flow.solve_port_flow_field` and `microsimulator.stokes.solve_port_stokes_field` take a grid and a list of `FlowPort`, each a face of the grid, a `GridFacePatch` of its boundary sites, and a speed or a pressure. A fixed boundary's `patches` give parts of one face their own media, and `Simulation.set_signal_boundary_patch_values(face, patch, values)` changes one. A fixed face is a reservoir wherever fluid touches it, port or not, so a model that opens ports by hand keeps the rest of each port's face solid, as `LayoutDevice` does.

## Changing the inlet media

The media entering at the inlet carries one concentration per signal, and the solved flow
carries that composition down the channel. `apply_to_grid` sets the starting composition;
`set_inlet_values` replaces it between steps, so a model can switch reservoirs, pulse an
inducer, or ramp a nutrient as an experiment's valves and pumps would:

```python
from microsimulator.microfluidics import InletSchedule

# Signal 0 is nutrient, signal 1 an inducer supplied from t = 5 until t = 8.
INLET = InletSchedule.pulse([10.0, 0.0], [10.0, 2.0], start=5.0, end=8.0)

def _regulate(step):
    DEVICE.set_inlet_values(step.simulation, INLET.values_at(step.time))
    return StepPlan(...)
```

`InletSchedule(initial, times, values)` holds `initial` until `times[0]` and `values[i]` from
`times[i]`; `ramp=True` varies the composition linearly between the listed times instead of
switching. The composition set in a regulation callback applies to the step that follows it.
Because the schedule depends only on model time and the current inlet values are part of the
checkpointed grid, a resumed run continues on the same composition.

A change at the inlet reaches a point downstream after the transit time of the flow, and the
first-order upwind transport spreads a sharp switch over a few voxels on the way. Outside a
device, `Simulation.set_signal_boundary_values(face, values)` replaces the values of any
fixed boundary of the signal grid.

## Flow on signals and on cells

The velocity field advects every signal with conservative upwind face fluxes under all three signal integrators. With `MechanicsConfig(flow_drift=True)`, the controller translates each non-fixed cell using the velocity sampled at its center and rotates it with a finite-aspect Jeffery approximation, so escaped cells travel down the channel and rods rotate in shear. That velocity is part of the contact solve, so cells the flow presses together or into a wall do not overlap. The [flow-drift design](../architecture/0021-flow-drift.md) describes this kinematic approximation and its integration limits.

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

## Numerical flow and drag from cells

`microsimulator.flow` solves `div_xy(H*m*grad_xy(p)) = 0` with one pressure per depth column. The default mobility is proportional to H², so integrated flux has the required H³ gap dependence. Harmonic face conductance and conservative lifting provide the face field used by transport. The shallow solver requires contiguous columns above a common floor; full three-dimensional obstructions require `microsimulator.stokes`. Both support a prescribed inlet speed and execute on the selected native backend.

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

The pillar example uses the stationary closure above for attached founders. The trap uses `colony_flow_coupling` with every cell and its center velocity from contact mechanics. It passes both the returned mobility and body force to `solve_flow_field`. A stalled cell resists passing medium, and a moving cell can also drive flow. At the default timestep of 0.02, the trap refreshes the flow every 0.1 model time units. Biomass is deposited conservatively over a fixed physical radius, and only the resistance formula caps density. The drag coefficient and smoothing radius require calibration; grid size and refresh interval require numerical sensitivity checks.

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
uv run microsimulator view --model examples/tutorials/biopixel_trap.py --seed 5 --dt 0.02 --backend cpu --open
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
- Transport upwinds to first order unless the grid names another scheme. That blends streams wherever flow crosses the grid at an angle, which matters when the question is how far liquids have mixed and seldom otherwise. `grid.advection_scheme = SignalAdvectionKind.VAN_LEER` selects the [limited second-order scheme](../architecture/0028-limited-advection.md), which keeps a front sharp and never overshoots. Its implicit solve settles only while one step carries the fastest fluid across about six sites or fewer; a longer step is refused with a message that says so. Its explicit stability bound is half as long in its advective part.
- The implicit solve's relative tolerance is the accuracy the step delivers: it asks for
  that reduction of the residual the step starts with, so a model gets what it asked for
  regardless of its concentration scale. These models keep the engine defaults.
- Let the lattice of site centers cover every position a cell can reach, with about a voxel
  of margin past each wall: sampling outside the lattice is an error.
- Keep the mechanics walls enclosing the solid mask. The device helpers mark a site solid
  when its center lies inside a wall, and the contact solve keeps every cell center about a
  cell radius clear of each wall, so a cell against a wall always has a fluid site to
  sample. A hand-built mask needs the same rule; the
  [pillar channel](flow-solvers.md) shows it for curved walls.
- A sampling position whose whole stencil is solid raises an error rather than returning
  zero.

## Uptake and time integration

The tutorials consume the actual biochemical biomass increment `Delta B / yield`, where `B = pi*r²*(length + 2*r)`. Division conserves this amount, which is distinct from geometric capsule volume. `growth_rate * B` is not the realized biomass rate under the length-growth law.

Backward Euler is the baseline for their stiff transport and affine losses. Cellular sinks remain explicit; native biological failure restores growth, species, signals, time, and the prior solver report. Controller regulation, division callbacks, and mechanics are separate operations. No rejected step may be counted as successful growth. The Danino circuit is a qualitative example: its AHL secretion uses intracellular concentration times B, and its AiiA loss uses a conservatively smoothed enzyme amount. Its parameters and oscillations are not experimentally calibrated.

The biopixel model explicitly uses signal absolute residual tolerance `1e-5` because binary32 noise at concentration 10 and its fine depth spacing prevents reliable convergence at `1e-6`. This is a model-scale numerical choice, not a biological accuracy claim.
