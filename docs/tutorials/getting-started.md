# Run, inspect, and resume a tutorial

Start with a cell trap supplied by a flowing nutrient channel. This model combines device walls, a steady flow solve, solute transport, nutrient-dependent growth, and cell motion. The [microfluidics tutorial](microfluidics.md) explains the model, and the [modeling guide](../microfluidics.md) introduces the broader workflow.

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

Run the trap on the CPU backend and save a checkpoint every 20 steps:

```console
uv run microsimulator run \
  --model examples/microfluidic_trap.py \
  --backend cpu \
  --seed 42 \
  --steps 100 \
  --dt 0.02 \
  --checkpoint-every 20 \
  --output results/tutorial-trap.json
```

The seed belongs to the model's dedicated random stream. It controls founder variation, daughter-axis jitter, stochastic plasmid partitioning, and conjugation events where applicable. The checkpoint records the model digest, seed, parameters, controller state, random state, native state, and run provenance.

Other tutorials expose JSON-valued model parameters. A string needs JSON quotes inside the shell argument. For example, the biophysics tutorial uses a `scenario` parameter; add `--stop-cell-count` to bound the colony size:

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
  --model examples/microfluidic_trap.py \
  --backend cpu \
  --seed 42 \
  --dt 0.02 \
  --checkpoint-output results/trap-live.json \
  --open
```

The viewer can play, pause, step, reset, and request a checkpoint. For the trap, enable a nutrient signal slice and choose `Growth rate` coloring to inspect the population alongside its environment. Other models can use `Species` coloring for intracellular channels or `Cell type` for strain or discrete-state categories. Selecting a cell shows its stable ID, lineage parent, geometry, type, growth rate, and ordered species values.

The browser owns only presentation state. Python owns the clock, model, backend, checkpoint path, and random state.

## Resume exactly

Controller-backed checkpoints must be resumed with the same model source, seed, and parameters. MicroSimulator verifies the source digest before running the file:

```console
uv run microsimulator run \
  --model examples/microfluidic_trap.py \
  --resume results/tutorial-trap.json \
  --backend cpu \
  --seed 42 \
  --steps 100 \
  --dt 0.02 \
  --output results/trap-resumed.json
```

Do not add `--overwrite` casually. Runs fail before mutation when a final or planned periodic output already exists.

## Read a model file

A native teaching model generally has four layers:

1. validated parameters and constants;
2. a `build(context)` function that constructs native state;
3. regulation and division callbacks that return typed changes; and
4. `resume(context, checkpoint)` that restores the same behavior around the checkpoint's exact native state.

Species and coupled rate plans are immutable equation data stored in the checkpoint. Python callbacks never inject C, MSL, or CUDA source. Each backend interprets the same validated equation graph through its own implementation.
