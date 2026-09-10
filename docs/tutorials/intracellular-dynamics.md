# Intracellular species and gene circuits

This tutorial introduces intracellular concentrations, growth dilution, typed rate equations, gene-circuit feedback, and quantitative time-course analysis. The runnable scenarios are collected in `examples/tutorials/gene_expression.py`.

## The native species contract

A simulation declares one immutable species count. Each cell contains exactly that many finite single-precision concentrations. A typed rate plan returns one concentration-per-time derivative for each channel.

For a biological step, MicroSimulator:

1. advances capsule length;
2. dilutes concentrations by `V_old / V_new`;
3. evaluates every rate from the complete diluted state; and
4. commits the simultaneous explicit Euler update `c_next = c_diluted + dt * rate`.

Equal division copies concentrations to both daughters. Since their effective volumes sum to the parent volume, this preserves total amount. Negative finite concentrations are not silently clamped; choose a stable `dt` and equations whose numerical behavior matches the intended model.

`RatePlanBuilder` is Python authoring syntax for a validated equation graph. The graph—not Python or injected C—is checkpointed and interpreted by each native backend.

## 1. Constitutive production

```console
uv run microsimulator view \
  --model examples/tutorials/gene_expression.py \
  --parameter scenario='"constitutive"' \
  --seed 42 \
  --dt 0.01 \
  --open
```

The model has one channel, `x`, with

```text
dx/dt = 2.
```

The rate plan is:

```python
rates = RatePlanBuilder()
simulation.set_species_rate_plan(
    rates.species_plan(1, (rates.constant(2.0),))
)
```

Choose `Species` coloring and channel 0 in the viewer. Production competes with growth dilution. Cells whose effective volume grows more slowly dilute less, which can create colony-scale concentration structure without a position term in the equation.

The `legacy_constitutive` scenario provides an alternative parameterization with production rate one, growth rate two, and a shorter 2.5–3.0 division-target range.

## 2. Dilution without production

```console
uv run microsimulator view \
  --model examples/tutorials/gene_expression.py \
  --parameter scenario='"dilution"' \
  --seed 42 \
  --dt 0.01 \
  --open
```

The founder starts at `x = 10` and the explicit chemical rate is zero. Any decline is therefore caused by cell growth dilution. Division itself copies concentration; it does not halve it. This distinction is essential when interpreting a concentration rather than a discrete molecule count.

## 3. Derepression through dilution

```console
uv run microsimulator view \
  --model examples/tutorials/gene_expression.py \
  --parameter scenario='"derepression"' \
  --seed 42 \
  --dt 0.01 \
  --open
```

The channels are `x0 = repressor` and `x1 = reporter`, initially `(10, 0)`:

```text
dx0/dt = 0
dx1/dt = 4 / (4 + x0^2).
```

As growth dilutes `x0`, reporter production approaches one. Inspect both channels rather than treating a color change as quantitative evidence.

## 4. Activator-inhibitor oscillator

```console
uv run microsimulator view \
  --model examples/tutorials/gene_expression.py \
  --parameter scenario='"oscillator"' \
  --seed 42 \
  --dt 0.005 \
  --open
```

With `x = species[0]` and `y = species[1]`, the model uses:

```text
dx/dt = 2(1 + x^2) / (1 + x^2 + y^2) - x
dy/dt = 2(1 + x^2) / (1 + x^2) - y.
```

The circuit is adapted from Guantes and Poyatos, PLoS Computational Biology 2(3), 2006, DOI `10.1371/journal.pcbi.0020030`, as cited by the original tutorial. The teaching model is not a parameter fit to one organism.

Whether the model produces a sustained limit cycle depends on parameters, dilution, initial conditions, integration step, and simulation duration. Plot a time course before interpreting oscillatory behavior from a single rendered frame.

## Quantitative check

Create periodic checkpoints, export them, and plot or inspect one stable cell lineage:

```console
uv run microsimulator run \
  --model examples/tutorials/gene_expression.py \
  --parameter scenario='"oscillator"' \
  --seed 42 \
  --steps 400 \
  --dt 0.005 \
  --checkpoint-every 20 \
  --output results/oscillator.json

uv run microsimulator export-analysis \
  results/oscillator.step-*.json \
  --output results/oscillator.dataset
```

The `species.parquet` table contains `frame_index`, stable `cell_id`, channel, and level. A colony-wide mean can hide phase dispersion; inspect individual lineages and state how cells born or lost between frames are handled.

## Experiments

- Multiply constitutive production by `rates.growth_rate()` and test whether spatial concentration differences decrease.
- Add first-order degradation to the constitutive model and derive the expected steady concentration under a fixed growth rate.
- Sweep oscillator `dt` before interpreting phase drift. Numerical instability and biological desynchronization are different claims.
- Couple one species to growth in the regulation callback. Record that this makes mechanics, dilution, and the gene circuit a feedback system.
