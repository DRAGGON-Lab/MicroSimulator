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

Without `--open`, open the tokenized loopback URL printed by `microsimulator`. The live transport can play, pause, advance one step, rebuild the original model, and write to the configured checkpoint destination. Camera position, display mapping, grid slice, and selected-cell identity survive frame updates.

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

## Dataset presentation lifecycle

Opening a scene file, live session, or recording begins a new dataset. Call `DatasetPresentationState.beginDataset()` and `ColonyViewer.beginDataset()` once, then present its first frame with `setFrame(frame, true)` to fit the camera. Ordinary updates, a reset of the same live model, and recording seeks use `setFrame(frame)` without beginning a dataset. Neither simulation time returning to zero nor a changed signal-grid shape identifies a new dataset.

`DatasetPresentationState.datasetId` scopes numerical channel identities; display labels are not identities. Retained preferences are separate from the effective values returned by `forFrame()`. Temporarily absent channels or smaller grids use valid display indices without erasing the user's selections, signal visibility, or chosen slice. The first available signal grid initializes a default slice once. Feature-specific display state should reset only in the explicit `newDataset` block in `presentScene()`.

The ground reference grid is separate from the scientific signal lattice. Its square extent and origin come from the first frame's finite device geometry (boxes, spheres, and cylinders), or from the initial cell capsule bounds when no finite device exists. Infinite plane constraints are excluded. The extent is at least 10 scene distance units with 20 equal divisions; the grid plane is 0.01 units below the lesser of the initial lower Z bound and zero. An initially empty dataset uses a 10-unit grid centered on the world origin. These values remain fixed even when cells or device geometry appear later, the colony expands beyond the grid, or all cells disappear. Opening another dataset initializes a new reference grid; camera Fit never changes its geometry.

`browser/reference-grid.mjs` verifies the reference grid and presentation lifecycle in Chromium against a running Vite server. It uses Playwright (`@playwright/test`) and its installed Chromium; a shared installation can be supplied through `MICROSIMULATOR_PLAYWRIGHT_MODULE` as an absolute module filename. Set `VIEWER_URL` if the server is not on `http://127.0.0.1:4320`, and `EVIDENCE_DIR` to choose the screenshot directory. The test observes renderer transforms through test-only request instrumentation and introduces no production debug interface.
## Channel labels

Model-defined species and signal names appear in channel selectors, the species legend, and cell inspection. Duplicate names include their channel indices; unnamed channels retain `Channel N`. Names are presentation text; indices continue to identify selected channels. Current readers accept scene v2 and v3, while writers emit v3. See the [authoring guide](../docs/models/channel-labels.md) and [scene v3 schema](../docs/formats/scene-v3.md).

### Device geometry visibility

Use **Show device geometry** in the Scene panel to hide all mechanical constraint meshes and their outlines. The control is disabled when the current frame contains no geometry, while its preference is retained for later frames. Visibility persists through live updates, reset, and replay seeks, and defaults to enabled on opening another dataset. Cells, selection, the reference grid, signal slices, camera pose, and the existing Fit bounds policy are independent of this display setting. No simulation constraint or transport obstacle is changed.

`browser/device-visibility.mjs` verifies all four constraint types, outlines, keyboard toggling, cell picking, sibling visibility, frame/reset retention, missing geometry, camera and Fit invariance, and new-dataset defaults against Vite on port 4323. It uses the same Playwright module and evidence-directory options as the reference-grid browser test.
