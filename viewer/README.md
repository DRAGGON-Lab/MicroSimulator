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
