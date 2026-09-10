# Run, inspect, and resume a tutorial

## Prepare the workspace

MicroSimulator requires Python 3.12, CMake, Ninja, a C++23 compiler, and `uv`. From the repository root:

```console
uv sync --group dev --extra analysis --extra viewer
uv run microsimulator devices
uv run pytest
```

The CPU backend is the portable reference. On a supported Mac, `metal` is the feature-complete native GPU choice. CUDA tutorials should be treated as development validation until the project status table says otherwise. Every command selects its backend explicitly; no tutorial silently falls back.

Build the browser viewer once:

```console
pnpm --dir viewer install
pnpm --dir viewer build
```

## Run a deterministic model

Tutorial options are JSON-valued model parameters. A string therefore needs JSON quotes inside the shell argument:

```console
uv run microsimulator run \
  --model examples/tutorials/biophysics.py \
  --parameter scenario='"basics"' \
  --backend cpu \
  --seed 42 \
  --steps 100 \
  --dt 0.05 \
  --checkpoint-every 20 \
  --output results/tutorial-basics.json
```

The seed belongs to the model's dedicated random stream. It controls founder variation, daughter-axis jitter, stochastic plasmid partitioning, and conjugation events where applicable. The checkpoint records the model digest, seed, parameters, controller state, random state, native state, and run provenance.

Use `--stop-cell-count` for a bounded colony experiment:

```console
uv run microsimulator run \
  --model examples/tutorials/biophysics.py \
  --parameter scenario='"competition"' \
  --backend cpu \
  --seed 7 \
  --steps 1000 \
  --dt 0.02 \
  --stop-cell-count 256 \
  --output results/competition.json
```

The maximum step count is still required so a non-growing model always terminates.

## Inspect a live simulation

```console
uv run microsimulator view \
  --model examples/tutorials/gene_expression.py \
  --parameter scenario='"oscillator"' \
  --backend cpu \
  --seed 42 \
  --dt 0.01 \
  --checkpoint-output results/oscillator-live.json \
  --open
```

The viewer can play, pause, step, reset, and request a checkpoint. Choose `Species` coloring to inspect intracellular channels, `Cell type` for strain or discrete-state categories, `Growth rate` for regulated growth, and enable a signal slice for signaling models. Selecting a cell shows its stable ID, lineage parent, geometry, type, growth rate, and ordered species values.

The browser owns only presentation state. Python owns the clock, model, backend, checkpoint path, and random state.

## Resume exactly

Controller-backed checkpoints must be resumed with the same model source, seed, and parameters. MicroSimulator verifies the source digest before running the file:

```console
uv run microsimulator run \
  --model examples/tutorials/gene_expression.py \
  --resume results/oscillator-live.json \
  --parameter scenario='"oscillator"' \
  --backend cpu \
  --seed 42 \
  --steps 100 \
  --dt 0.01 \
  --output results/oscillator-resumed.json
```

Do not add `--overwrite` casually. Runs fail before mutation when a final or planned periodic output already exists.

## Read a model file

A native teaching model generally has four layers:

1. validated parameters and constants;
2. a `build(context)` function that constructs native state;
3. regulation and division callbacks that return typed changes; and
4. `resume(context, checkpoint)` that restores the same behavior around the checkpoint's exact native state.

Species and coupled rate plans are immutable equation data stored in the checkpoint. Python callbacks never inject C, MSL, or CUDA source. Each backend interprets the same validated equation graph through its own implementation.
