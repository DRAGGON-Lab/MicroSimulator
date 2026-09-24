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

The automated `python/tests/test_viewer_shutdown.py` suite covers same-socket Stop, checkpoint completion, worker cleanup, and repeated real subprocess restarts. It sends SIGINT on POSIX. On Windows it starts each viewer in an isolated console and uses a separate attached sender to deliver a real [Windows CTRL_C_EVENT](https://learn.microsoft.com/en-us/windows/console/generateconsolectrlevent), leaving the test runner unaffected. Both paths verify orderly browser notifications, clean process exit, and three different models reusing the same port. This exercises the operating-system interruption path; use the manual procedure above to check a particular interactive terminal application and keyboard configuration.

The `Windows live-session shutdown` GitHub Actions job builds the CPU extension on `windows-2025` and runs the server and shutdown tests, including isolated-console Ctrl+C, with dependencies from `uv.lock`. Its uploaded report records Windows, PowerShell, Python, backend availability, and individual test results.

## Capabilities

- SHA-256 verification over the Python writer's RFC 8785 canonical frame;
- strict scene v2 structural and numerical validation;
- instanced open cylinders and matching hemispheres for continuous capsule surfaces;
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

## Dataset presentation lifecycle

Opening a scene file, live session, or recording begins a new dataset. Call `DatasetPresentationState.beginDataset()` and `ColonyViewer.beginDataset()` once, then present its first frame with `setFrame(frame, true)` to fit the camera. Ordinary updates, a reset of the same live model, and recording seeks use `setFrame(frame)` without beginning a dataset. Neither simulation time returning to zero nor a changed signal-grid shape identifies a new dataset.

`DatasetPresentationState.datasetId` scopes numerical channel identities; display labels are not identities. Retained preferences are separate from the effective values returned by `forFrame()`. Temporarily absent channels or smaller grids use valid display indices without erasing the user's selections, signal visibility, or chosen slice. The first available signal grid initializes a default slice once. Feature-specific display state should reset only in the explicit `newDataset` block in `presentScene()`.

The ground reference grid is separate from the scientific signal lattice. Its square extent and origin come from the first frame's finite device geometry (boxes, spheres, and cylinders), or from the initial cell capsule bounds when no finite device exists. Infinite plane constraints are excluded. The extent is at least 10 scene distance units with 20 equal divisions; the grid plane is 0.01 units below the lesser of the initial lower Z bound and zero. An initially empty dataset uses a 10-unit grid centered on the world origin. These values remain fixed even when cells or device geometry appear later, the colony expands beyond the grid, or all cells disappear. Opening another dataset initializes a new reference grid; camera Fit never changes its geometry.

`browser/reference-grid.mjs` verifies the reference grid and presentation lifecycle in Chromium against a running Vite server. It uses Playwright (`@playwright/test`) and its installed Chromium; a shared installation can be supplied through `MICROSIMULATOR_PLAYWRIGHT_MODULE` as an absolute module filename. Set `VIEWER_URL` if the server is not on `http://127.0.0.1:4320`, and `EVIDENCE_DIR` to choose the screenshot directory. The test observes renderer transforms through test-only request instrumentation and introduces no production debug interface.

## Concentration color ranges

Species coloring and signal slices each offer Automatic and Fixed color ranges. Automatic uses the current frame's species extrema or the selected signal slice's extrema. Constant automatic data uses the midpoint color and a uniform legend; empty automatic data shows “no values” without numerical bounds. Fixed uses the entered minimum and maximum across frames and slices. Values outside that interval use endpoint colors; the underlying concentrations and inspector values remain unchanged.

Switching to Fixed starts from the current extrema (with finite padding for constant data), or restores that channel's previously entered fixed bounds. Edit both bounds and choose Apply range or press Enter. Bounds accept finite decimal numbers, including negative numbers and scientific notation, with minimum strictly less than maximum. Invalid or incomplete edits show an explanation and leave the last valid range active. Legends always describe the active range rather than unsubmitted text.

Settings belong to the numerical species or signal channel within the current dataset. They survive temporarily missing channels/grids, live updates, same-model reset, and frame seeking; opening another dataset restores automatic defaults. The shared `resolveScalarRange()` and `normalizeScalar()` APIs reject non-finite data explicitly, retain zero-valued and negative data, and avoid overflowing the difference between extreme finite bounds. They produce display intensity only and do not modify model data.

`browser/scalar-ranges.mjs` checks actual cell instance colors, signal texture pixels, legends, validation messages, keyboard interaction, and dataset transitions in Chromium. Run it with a Vite server on port 4315, or set `VIEWER_URL`, using the same optional Playwright module and evidence-directory environment variables as the reference-grid test.

## Channel labels

Model-defined species and signal names appear in channel selectors, the species legend, and cell inspection. Duplicate names include their channel indices; unnamed channels retain `Channel N`. Names are presentation text; indices continue to identify selected channels. Current readers accept scene v2 and v3, while writers emit v3. See the [authoring guide](../docs/models/channel-labels.md) and [scene v3 schema](../docs/formats/scene-v3.md).

### Device geometry visibility

Use **Show device geometry** in the Scene panel to hide all mechanical constraint meshes and their outlines. The control is disabled when the current frame contains no geometry, while its preference is retained for later frames. Visibility persists through live updates, reset, and replay seeks, and defaults to enabled on opening another dataset. Cells, selection, the reference grid, signal slices, camera pose, and the existing Fit bounds policy are independent of this display setting. No simulation constraint or transport obstacle is changed.

`browser/device-visibility.mjs` verifies all four constraint types, outlines, keyboard toggling, cell picking, sibling visibility, frame/reset retention, missing geometry, camera and Fit invariance, and new-dataset defaults against Vite on port 4323. It uses the same Playwright module and evidence-directory options as the reference-grid browser test.

## Composite species colors

Choose Species composite to display several intracellular channels together. The first two available channels initially use red and green and are enabled; additional channels start disabled. Enable or disable each channel with its checkbox. Display settings exposes its tint (a six-digit sRGB hexadecimal color), the same Automatic/Fixed range editor used by single-species coloring, and controls for reordering the list. Tint or range changes apply when submitted, and invalid edits preserve the active value.

For each cell, every enabled channel is independently normalized with its selected range. Tints are decoded from sRGB into linear RGB; normalized intensity multiplies each linear tint, the contributions are added, and each summed component is clipped to one. The result is encoded back to sRGB for the existing renderer interface, which converts its instance colors to linear RGB. Channel-list order has no effect on the result. Full red and full green therefore produce yellow. The legend lists enabled channel names, tints, and active bounds. Three.js uses a small approximation in its sRGB encoding function; conversion tests bound the resulting error below 0.00001, well below an 8-bit color step.

When all channels are disabled or unavailable, cells use neutral gray and the legend says no channels are active. With active channels and fixed zero-based bounds, zero intensity is black. Automatic constant data uses the same midpoint convention as single-species coloring, including a constant zero field; choose fixed zero-based bounds when zero should mean no displayed contribution. These colors are a presentation mapping, not calibrated fluorescence measurements. Lighting, tone mapping, and selection highlighting can further affect the final pixel appearance.

Tints, visibility, list order, and ranges remain associated with numerical channel identity when labels change or data temporarily disappears. Single-species and composite views share each species channel's range. Same-model reset and frame seeking retain settings; opening another dataset restores defaults. The cell inspector, picking, selection highlights, and lineage values continue using the original cell state.

`browser/composite-species.mjs` exercises red-only, green-only, co-expressing, and zero-expression cells in a moving colony, verifies the rendered instance colors, and checks controls, ordering, picking, highlighting, lineage, and dataset transitions. It uses a Vite server on port 4319 (or `VIEWER_URL`) and the same Playwright/evidence environment variables as the other browser checks.

## Replay a recording

Record periodic checkpoints using the short native growth/division/removal example, then list the checkpoint paths in the order they should play:

```sh
uv run microsimulator run --model examples/replay_demo.py --backend cpu --seed 17 --steps 5 --dt 0.2 --checkpoint-every 1 --output run/replay.json
uv run microsimulator export-replay run/replay.step-00000001.json run/replay.step-00000002.json run/replay.step-00000003.json run/replay.step-00000004.json run/replay.step-00000005.json --output run/replay-bundle
pnpm --dir viewer dev
```

Open the displayed viewer URL, choose **Open recording**, and select the `run/replay-bundle` folder. Select the folder itself, containing `manifest.json` and `frames`, rather than one frame file. The standalone viewer reads the selected local files; no simulation server or source GPU is required.

Use Play/Pause, Previous/Next, the frame slider and Frames/s. Slider arrow keys seek one recorded frame; the buttons also work with keyboard focus. Manual seeking pauses playback. The transport displays a one-based frame position and recorded simulation time, while the manifest uses zero-based ordinals. Equal-time frames remain individually selectable. Playback stops at the end; Play then restarts from frame one. Opening a static scene ends the recording session.

Source paths are used exactly in command-line order; avoid relying on shell globs to establish chronological ordering. The final `run/replay.json` duplicates the last periodic state in this example and is intentionally omitted. Decreasing times cause an error. The exporter refuses existing destinations; choose a new bundle directory for another export. Model parameters and source are unnecessary for export, and no callbacks execute during playback.

The reader loads frames on demand through a bounded three-frame/64 MiB accounting-budget LRU cache; it does not decode the whole recording. Oversized frames are uncached, and renderer/current-load allocations exist outside that cache. See the [replay format](../docs/formats/replay-v1.md) for integrity, provenance, resource bounds and failure behavior. This first implementation imports checkpoint sequences; live recording, video export and timeline-based simulation restart are separate features.

For browser regression checks, generate native fixtures with `.venv/bin/python viewer/browser/replay-fixtures.py /tmp/replay-fixtures`, run the viewer on port 4326, then run `viewer/browser/replay.mjs` with `REPLAY_FIXTURES=/tmp/replay-fixtures` and `MICROSIMULATOR_PLAYWRIGHT_MODULE` pointing to an installed Playwright module. This uses the existing shared browser harness and adds no production debug API.

Capsule geometry tests verify the scene's cylindrical centerline length, constant radius, spherical ends, zero-length sphere case, arbitrary orientation, outward topology, exact equator positions/normals, and ray picking. The selected-cell overlay uses the same geometry with radius increased by 8%; the cap centers retain the original centerline length. Three instanced draw calls represent the colony, independent of cell count. Replacing frames disposes both geometry and instance buffers.

For visual and GPU-resource checks, see [the capsule browser regression](browser/README.md). Mesh tessellation and pixel aliasing can still affect silhouettes at distant zoom levels; the shared tangent joins specifically remove overlapping end disks and mismatched sphere/cylinder boundaries. The viewer does not smooth or modify simulated cell motion.
