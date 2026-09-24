# Solved flow in a pillar channel

The [pillar-channel model](../../examples/tutorials/pillar_channel.py) combines cylindrical walls, a depth-integrated flow calculation, attached founder lineages, and released daughters:

Cells retain three-dimensional mechanics inside the channel walls at Z=±3. XY-only division jitter and depth-integrated flow do not impose a planar cell constraint; fixed founders remain attached while released daughters can move and tilt within the finite-height chamber. See the [dimensionality audit](planarity.md).

```console
uv run microsimulator view --model examples/tutorials/pillar_channel.py --seed 7 --dt 0.01 --backend metal --open
```

## Geometry and flow

Mechanics uses continuous cylinders. The transport mask classifies voxel centers against those same cylinders, without shrinking their radii. Staircase walls approximate the physical geometry; curved-wall error and narrow gaps require refinement. Fluid-only, face-connected interpolation keeps solid concentrations out of cell exchange. A cell with no fluid sampling support is rejected rather than given artificial access through a wall.

The channel's six-micrometer depth is represented by two three-micrometer fluid layers. The shallow solver uses one pressure per x/y column and depth-integrated conductance `H*m`, with base `m` proportional to `H²`. It routes flux around the pillars and lifts it conservatively to the signal grid:

```python
field, report = solve_flow_field(
    grid, mean_inlet_speed=FLOW_SPEED,
    mobility=gap_mobility(grid), simulation=simulation,
)
grid.velocity_field = field
```

The shallow approximation requires contiguous depth columns with a common floor. It does not resolve wall shear, flow within the gap, or the in-plane viscous stresses of a full Brinkman model. CPU, Metal, and CUDA implement the numerical solve independently, retaining accelerator vectors on device.

## Attached cells and prescribed shedding

Fixed founders remain attached. At division, the daughter nearer its lineage's adhesion site keeps the anchor and the farther daughter is released. This is a prescribed attachment rule, not a detachment prediction. Free daughters move at the sampled fluid velocity with a finite-aspect Jeffery orientation approximation, then undergo contact relaxation and model-defined outlet removal.

Drift uses midpoint substeps bounded by grid displacement and angular change. The angular limit no longer clips the total rotation. The full growth/drift/contact split is still first order, so reduce the outer timestep and check mechanics convergence for quantitative motion.

Only attached cells contribute stationary porous resistance:

```python
mobility = colony_mobility(
    GRID, (cell for cell in step.cells if cell.fixed),
    base=GAP_MOBILITY, drag_coefficient=DRAG_COEFFICIENT,
    averaging_radius=4.0,
)
field, report = solve_flow_field(
    GRID, mean_inlet_speed=FLOW_SPEED,
    mobility=mobility, simulation=step.simulation,
)
step.simulation.set_velocity_field(field)
```

The amount-conserving smoothing radius has physical units and stays fixed under grid refinement. The density is capped only inside the empirical resistance formula; biomass itself is conserved. Three small anchors should not be presented as a validated bulk biofilm blockage experiment. Freely advected daughters do not form a stationary matrix.

Nutrient uptake in the tutorial equals the actual increment of biochemical biomass `B = pi*r²*(length + 2*r)` divided by the chosen yield. Backward Euler treats transport and affine loss implicitly; explicit cell uptake still requires an affordable step, and a rejected native biological step rolls back growth and chemistry.

## Numerical evidence

```console
uv run python scripts/run_flow_benchmarks.py --backend cpu
uv run python scripts/run_flow_benchmarks.py --backend metal
uv run python scripts/run_flow_benchmarks.py --backend cpu --fine
```

The analytic suite checks plane and square ducts, a two-layer Brinkman channel, shallow routing, and agreement in a common thin-gap regime. Resolved flow uses flexible GMRES on a fixed velocity-pressure operator; inexact momentum solves are preconditioners. Its report includes freshly computed momentum and block residuals and physical divergence. Fine binary32 grids may need an explicitly looser tolerance, as the Brinkman benchmark documents.

Use `solve_stokes_field` for resolved profiles when the mesh resolves the gap. `min_gap_voxels` identifies poorly resolved passages but does not certify accuracy. The [nutrient validation study](nutrient-validation.md) provides a separate, quantitative attached-population example with nutrient balance and grid, timestep, and flow-refresh sensitivity. It establishes numerical behavior under stated parameters, not biological calibration.
