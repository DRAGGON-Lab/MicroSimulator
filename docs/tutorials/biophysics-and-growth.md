# Growth, division, cell types, and constraints

This tutorial introduces cell geometry, growth, division, lineage, cell types, mechanical constraints, and competition. Its five runnable scenarios are defined in `examples/tutorials/biophysics.py`.

## 1. A founder that grows and divides

Run the basic model:

```console
uv run microsimulator view \
  --model examples/tutorials/biophysics.py \
  --parameter scenario='"basics"' \
  --seed 42 \
  --dt 0.05 \
  --open
```

`build(context)` obtains the backend chosen by the runner, creates one founder, and returns a restartable controller. The important founder fields are:

```python
founder.position = Vec3(0.0, 0.0, 0.0)
founder.direction = Vec3(1.0, 0.0, 0.0)
founder.length = 3.5
founder.radius = 0.5
founder.growth_rate = 1.0
founder.cell_type = 0
```

`position` is the capsule center. `direction` is normalized by the engine. `length` is the cylindrical centerline length; the full end-to-end capsule length is `length + 2 * radius`. `cell_type` is an inheritable signed integer used for categorical biology and viewer coloring.

The controller stores one stochastic division target per stable cell ID. On each step, its regulation callback returns `CellUpdate` records and a `DivisionRequest` for every cell above its target. Division replaces the parent with two new stable IDs, records lineage, copies concentration state, and invokes the division callback to sample two new targets.

### Length and volume

The tutorial uses centerline length as its division threshold. MicroSimulator uses the effective capsule volume

```text
V = pi r^2 (length + 2r)
```

for concentration dilution and cell-grid exchange. If an experiment requires a volume-based division rule, compute that threshold explicitly in the regulation callback.

## 2. Two founder types

```console
uv run microsimulator view \
  --model examples/tutorials/biophysics.py \
  --parameter scenario='"two_types"' \
  --seed 42 \
  --dt 0.02 \
  --open
```

The model places type 0 at `x = -10` and type 1 at `x = 10`. Both use the same growth and division policy. In the viewer, select `Cell type` coloring. This exercise isolates initial condition and lineage effects from rate differences.

## 3. Short, nearly round rods

```console
uv run microsimulator view \
  --model examples/tutorials/biophysics.py \
  --parameter scenario='"short_cells"' \
  --seed 42 \
  --dt 0.01 \
  --open
```

This scenario lowers the post-founder division length to produce short spherocylinders. It does not simulate a distinct spherical cell morphology. Sphere *constraints* are available for bounding rod cells, but they do not change cell shape.

## 4. Type-dependent competition in a growth zone

```console
uv run microsimulator view \
  --model examples/tutorials/biophysics.py \
  --parameter scenario='"competition"' \
  --seed 7 \
  --dt 0.01 \
  --open
```

The three founder types use the following parameters:

| Cell type | Division target range | Active growth rate |
| --------: | --------------------: | -----------------: |
|         0 |               1.0–1.5 |                2.0 |
|         1 |               2.0–2.5 |                1.1 |
|         2 |               3.5–4.0 |                0.8 |

At every regulation step, the model finds the largest cell-center `y` coordinate. A cell grows only when it lies less than five length units behind that leading edge. The computation is based on the immutable snapshots in `ControllerStep`, so all cells see the same pre-update colony state.

Use `Growth rate` coloring to see the active zone and `Cell type` coloring to see competition. A type's abundance is not determined by growth rate alone: shape, orientation, contact topology, stochastic target lengths, and spatial position affect which lineages remain near the frontier.

## 5. A three-dimensional open box

```console
uv run microsimulator view \
  --model examples/tutorials/biophysics.py \
  --parameter scenario='"box"' \
  --seed 42 \
  --dt 0.01 \
  --open
```

Five `PlaneConstraintInit` values define a floor and four walls. Each plane has a point, an inward unit normal, and a positive row coefficient. The permitted half-space lies in the inward direction. The founder begins at `z = 0.5`, so its radius-0.5 capsule initially touches rather than penetrates the floor.

The tutorial sets `MechanicsConfig(gamma=20)`. Here, `gamma` is the positive length regularizer in the seven-degree-of-freedom mechanics operator. It affects how strongly changes in cell length are penalized during relaxation.

After each biological step the controller runs a mechanics pass. Pair and constraint contacts are re-derived dynamically; there is no `max_planes`, `max_contacts`, or fixed contact-table capacity to tune.

## Experiments

- Change the three growth rates while keeping target ranges fixed. Compare lineage abundance at a fixed cell count, not at an arbitrary wall-clock time.
- Change the growth-zone width in `_callbacks` and compare radial or vertical growth-rate profiles with the analysis recipes.
- Add a ceiling plane to make a closed box, or replace the four side planes with an inside-sphere constraint.
- Run the same seed on CPU and Metal and compare checkpoint state. Backend identity will differ; scientific state should satisfy the project’s conformance tolerances.
