# MicroSimulator scene viewer

The MicroSimulator viewer displays cells, device walls, and signal fields so you can inspect a population in its microfluidic environment. Use it to explore saved scenes or follow a live simulation with growth-rate coloring, nutrient slices, and individual-cell inspection.

The viewer is a TypeScript and Three.js client for `microsimulator-scene` documents. Standalone mode reads scene files; live mode sends typed controls to a Python-owned engine session and verifies every returned scene document. Python owns the model, simulation clock, backend, and checkpoint writer.

## Run locally

From the repository root:

```console
uv run python examples/viewer_scene.py --output viewer-demo.scene.json
pnpm --dir viewer install
pnpm --dir viewer dev
```

Open the local URL printed by Vite and load `viewer-demo.scene.json`. The viewer also accepts a scene by drag and drop.

## Run a live session

From the repository root:

```console
uv sync --group dev --extra viewer
pnpm --dir viewer install
pnpm --dir viewer build
uv run microsimulator view \
  --model examples/microfluidic_trap.py \
  --backend cpu \
  --seed 42 \
  --dt 0.02 \
  --checkpoint-output results/trap-live.json \
  --open
```

Without `--open`, open the tokenized loopback URL printed by `microsimulator`. The live transport can play, pause, advance one step, rebuild the original model, write to the configured checkpoint destination, and stop the session. Camera position, display mapping, grid slice, and selected-cell identity survive frame updates.

### Stop one model and start another

Click **Stop session** or press **Ctrl+C** once in the terminal running the server. The current individual step or checkpoint write finishes, the browser displays **Stopped**, and the command returns to the prompt. A large playback batch does not have to finish. Start another `microsimulator view` command using the same port and open the new printed URL. Reset rebuilds the current model; Pause keeps its process available; closing the browser pauses it and allows reconnection. Stop does not automatically save a checkpoint: use Checkpoint first if you need restartable state.

For example, after stopping the trap model above, launch a different model on the same default port:

```console
uv run microsimulator view --model examples/tutorials/biophysics.py --backend cpu --seed 42 --dt 0.02 --port 8765 --open
```

The command is a single line and also works in PowerShell where the Python/native build is available. To distinguish Windows console behavior from browser behavior, use this manual verification procedure in an attached PowerShell or Command Prompt console:

1. Record the Windows version, terminal application/version, Python version, and exact launch command. Start the command above and click Stop while paused. Confirm the prompt returns, then start the second model on port 8765.
2. Repeat with Play active and `--frame-steps 10000`. Confirm Stopping transitions to Stopped without finishing the entire batch.
3. Repeat using Ctrl+C once, both paused and playing. Confirm the prompt returns without `taskkill`, then immediately start another model on the same port.
4. Close only the browser tab during Play, then reopen the printed URL. Confirm the session remains available and paused.

The automated `python/tests/test_viewer_shutdown.py` suite covers same-socket Stop, checkpoint completion, worker cleanup, and repeated real subprocess restarts. Its Stop/port-reuse test is portable to Windows; the SIGINT subprocess case runs on POSIX. Windows console Ctrl+C must be checked with the attached-console procedure above; passing the POSIX case does not establish Windows behavior.

The `Windows live-session shutdown` GitHub Actions job builds the CPU extension on `windows-2025` and runs the portable server and shutdown tests with dependencies from `uv.lock`. Its uploaded report records Windows, PowerShell, Python, backend availability, and individual test results. The detached CI process cannot substitute for the attached-console Ctrl+C check.

## Capabilities

- SHA-256 verification over the Python writer's RFC 8785 canonical frame;
- strict scene v2 structural and numerical validation;
- instanced cylinder and sphere rendering for exact spherocylinder geometry;
- device walls rendered from plane, sphere, box, and cylinder constraints;
- orbit, pan, zoom, colony framing, raycast picking, and selection highlighting;
- a draggable camera-synchronized flat-corner view cube with readable labels, shortest-path single-click snapping, and double-click label leveling;
- exact cell geometry, lineage, type, fixed state, growth, and species inspection;
- categorical cell-type and fixed-state color maps;
- perceptual growth-rate and species color maps;
- selectable signal channel, axis, and grid slice;
- authenticated same-origin live frames and typed transport controls; and
- stable camera, display, slice, and selection state during playback.

Cell IDs remain decimal strings throughout the browser because their unsigned 64-bit range exceeds JavaScript's exact integer range.

## Validate

```console
pnpm --dir viewer format:check
pnpm --dir viewer check
pnpm --dir viewer test
pnpm --dir viewer build
```

The unit suite includes a Python-authored scene fixture whose digest contains floating-point values that ordinary Python and JavaScript JSON serializers spell differently. Passing that test is the cross-language integrity gate.
