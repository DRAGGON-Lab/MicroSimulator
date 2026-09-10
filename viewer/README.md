# MicroSimulator scene viewer

This package is a TypeScript and Three.js consumer of `microsimulator-scene` documents. It has no Python bridge, model loader, simulation clock, checkpoint writer, CUDA context, or Metal device access. Standalone mode reads scene files; live mode sends a closed control vocabulary to the Python-owned engine session and verifies every returned scene document.

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
pnpm --dir viewer build
uv run microsimulator view \
  --model examples/batch_model.py \
  --dt 0.05 \
  --checkpoint-output results/live.json \
  --open
```

Without `--open`, open the tokenized loopback URL printed by `microsimulator`. The live transport can play, pause, advance one step, rebuild the original model, and write to the configured checkpoint destination. Camera position, display mapping, grid slice, and selected-cell identity survive frame updates.

## Capabilities

- SHA-256 verification over the Python writer's RFC 8785 canonical frame;
- strict scene v1 structural and numerical validation;
- instanced cylinder and sphere rendering for exact spherocylinder geometry;
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
