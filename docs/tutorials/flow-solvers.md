# Solved flow: a pillar channel, Brinkman feedback, and the benchmarks

The previous tutorial builds devices from the packaged helpers, whose grids and flow come
preassembled. This one goes a level down and demonstrates the flow machinery itself on
geometry no helper covers and no formula describes: a monolayer channel crossed by a
staggered array of cylindrical pillars, with colonies adhered in the pillar wakes shedding
daughters into the stream. Everything lives in one model:

```console
uv run cm view --model examples/tutorials/pillar_channel.py --seed 7 --dt 0.01 --backend metal --open
```

[`examples/tutorials/pillar_channel.py`](../../examples/tutorials/pillar_channel.py)
demonstrates, in order: authoring walls and a solid mask directly, solving the flow through
an arbitrary geometry, adhesion with released daughters, Brinkman colony feedback on the
flow, and washout. The end state is a steady flow of cells: anchored lineages grow in the
wakes while the stream continuously carries their offspring between the pillars and out of
the channel.

## Geometry the solver has to earn

The channel is one inside-region box; each pillar is one outside-region z-cylinder:

```python
pillar = CylinderConstraintInit()
pillar.center = Vec3(x, y, 0.0)
pillar.radius = PILLAR_RADIUS
pillar.allowed_region = ConstraintRegion.OUTSIDE
simulation.add_cylinder_constraint(pillar)
```

The signal grid needs the same geometry as a voxel mask, and curved walls raise a rule that
axis-aligned devices never surface: **the mechanics geometry must enclose the solid mask**.
Mechanics keeps cell centers outside the smooth cylinder; the mask is stair-stepped. If a
voxel whose center is barely inside the circle is marked solid, its corners poke out past
the cylinder wall, a cell hugging the wall can stand inside a solid voxel, and sampling
signals at its center is an error. The model therefore voxelizes conservatively — a voxel
is solid only when it lies *entirely* inside the pillar:

```python
core = PILLAR_RADIUS - 0.5 * math.hypot(spacing.x, spacing.y)
solid = (px - x) ** 2 + (py - y) ** 2 < core * core
```

With that rule every reachable cell position is in fluid, and the stair-stepped flow
blockage errs on the small side by the same half-diagonal margin.

## Solving flow where no profile exists

An analytic profile for a pillar array does not exist; the field comes from the numerical
solve, exactly as in the device helpers:

```python
field, report = solve_flow_field(grid, mean_inlet_speed=FLOW_SPEED, mobility=gap_mobility(grid))
grid.velocity_field = field
```

The solved field is conservative per voxel and routes around every pillar. At a mean inlet
speed of 20 the plug away from the array runs at 20 as requested — the solve normalizes over
the open inlet faces, so blocked columns cannot inflate it — and the gaps beside the center
pillar carry ≈31, because the pillars take cross-section and the same flux has to fit
through what is left. Flow speeds up exactly where the physical device would.

`report.max_speed` gives the number the `dt` bound needs. Drift is an explicit step, so a
cell must not cross more than about its own radius per step: keep `max_speed * dt` below
`CELL_RADIUS`. Here `max_speed` is 37.7, so `--dt 0.01` leaves a comfortable margin and
`--dt 0.02` would exceed it.

## Adhesion: anchored mothers, shed daughters

Founders are placed in the pillar wakes with `fixed = True` — mechanics and drift never
move them. Daughters inherit adhesion, so each division decides who stays:

```python
def _divided(step, event):
    DIVISION.on_division(step, event)
    if event.parent.fixed:
        released = ...  # the daughter farther from the adhesion site
        step.simulation.set_cell_fixed(released.id, False)
```

The daughter nearer the adhesion site keeps the anchor; the other is released and the
stream takes it. Anchoring by *site* rather than by daughter order matters: division
displaces both daughters by half a cell length, and a lineage that anchors whichever
daughter comes first random-walks away from its wake — and since fixed cells are never
pushed by mechanics, a walking anchor eventually stands inside a pillar. Site anchoring
keeps each attached cell within about a cell length of where its founder adhered,
indefinitely.

Released cells drift with the local fluid velocity (`MechanicsConfig(flow_drift=True)`),
slowly in the wake, then fast in the gaps, and leave through plan removals at the channel
end — the same washout pattern as the trap models. Run at seed 7 for 1600 steps at
`dt = 0.01` and the population reaches a steady state: three anchored cells, on the order
of 140 in transit, and about 1200 washed out, with lineage recording every one.

## The colony pushes back on the flow

Like the trap models, the regulation step re-solves the flow at a fixed cadence with the
colony rasterized into Brinkman drag and swaps the field into the running simulation:

```python
if step.completed_steps and step.completed_steps % RESOLVE_INTERVAL == 0:
    mobility = colony_mobility(GRID, step.cells, base=GAP_MOBILITY, drag_coefficient=DRAG_COEFFICIENT)
    field, _ = solve_flow_field(GRID, mean_inlet_speed=FLOW_SPEED, mobility=mobility)
    step.simulation.set_velocity_field(field)
```

Here the feedback has a visible consequence: as a wake colony thickens it plugs its own
gap, the solve routes more of the flux through the neighboring gaps, and shed daughters
increasingly take the fast lanes around the crowd.

## Checking the closure with the resolved solver

The Hele-Shaw field the model runs on is a depth-averaged closure. The MAC Stokes–Brinkman
solver resolves the same problem with viscous boundary layers on every wall, and takes the
identical grid:

```python
from cellmodeller2.stokes import solve_stokes_field

resolved, report = solve_stokes_field(GRID, mean_inlet_speed=FLOW_SPEED)
```

Depth-averaging the resolved field reproduces the closure's flux split around obstacles to
under a percent in the thin-gap regime — that agreement is enforced continuously by the
benchmark suite, which validates both solvers against literature and exact references
(plane Poiseuille and the two-layer Brinkman channel against their exact solutions with
measured second-order convergence, the Shah–London square-duct peak-to-mean ratio 2.0962,
and the cross-solver thin-gap check):

```console
uv run python scripts/run_flow_benchmarks.py          # CI-gating benchmark table
uv run python scripts/run_flow_benchmarks.py --fine   # doubled resolutions
```

### Which solver a grid deserves

The two solvers are accurate in opposite regimes, and the grid decides which.

The MAC solve resolves a no-slip profile only where it has voxels to resolve it in. Every
solve reports `min_gap_voxels`, the fluid voxels across its narrowest channel. One voxel
carries roughly two and a half times the flux the true parabolic profile would; four
voxels bring that within about ten percent and eight within a few percent. The pillar grid
here is one voxel deep in z, so its `min_gap_voxels` is 1 and the *depth-averaged* MAC
answer is the meaningful one — a comparison of in-plane flux splits, not of absolute
speeds.

The Hele-Shaw closure has the mirror-image property. It carries the gap-height physics
analytically in its mobility, so it is accurate for shallow channels at any z resolution,
but it solves for the depth-averaged velocity: every z layer of a column gets the column's
mean. In a channel resolved across its depth, the cells near the floor drift at the mean
rather than at the slower speed the true profile gives them, and a rod sees no shear
across the gap.

So: use the closure for device authoring and the in-model re-solve cadence, and reach for
`solve_stokes_field` when a study needs resolved wall shear or true cross-channel profiles
*and* the grid resolves the gap with at least four voxels. Refining z to reach that costs
grid sites in every other subsystem too.

## What to watch in the viewer

- Streamwise plug flow entering the array, cells accelerating visibly through the gaps.
- Wake colonies growing in place while a comet tail of released cells stretches
  downstream from each pillar.
- Cells vanishing at the channel end as washout removes them; lineage keeps their
  ancestry for analysis.
