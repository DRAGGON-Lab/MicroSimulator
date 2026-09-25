# Capsule rendering verification

`capsules.mjs` exercises the real viewer renderer through a development-server-only test injection. It records the isolated rod and dense colony at near/far camera distances, 48 frames of prescribed colony motion, stable-ID selection through frame reordering/removal, colors, pointer picking, selection geometry, and a selected zero-length cell. The generated fixtures use arbitrary 3D rod directions; no model or scientific output is modified.

Start this worktree's Vite server and run the script from the repository root with an existing Playwright installation:

```console
pnpm --dir viewer install --frozen-lockfile
pnpm --dir viewer dev --host 127.0.0.1 --port 4321
```

In another terminal:

```console
MICROSIMULATOR_PLAYWRIGHT_MODULE=/absolute/path/to/@playwright/test/index.mjs CAPSULE_MODE=corrected EVIDENCE_DIR=/tmp/capsule-corrected node viewer/browser/capsules.mjs
```

`VIEWER_URL` selects another server. For a before/after comparison, run the same script against a separate checkout of `b69193b9` served on another port, using `CAPSULE_MODE=baseline` and a different evidence directory. The script changes only the browser's fetched development module to observe the renderer; it does not patch the checkout. Baseline mode records rather than asserts the corrected GPU-resource and highlight invariants. Both runs use the same camera, lighting, colors, timestep sequence, and fixture parameters.

The performance fixture renders 512 cells after ten warmup frames, measuring 60 complete frame replacements (transform upload, coloring, rendering, and `gl.finish()` GPU synchronization). It reports median/p95 wall times, browser/WebGL renderer, three draw calls, per-mesh vertices/triangles, and geometry/buffer counts. Timings include CPU work and synchronization and are local regression evidence, not a cross-device benchmark. It tracks actual `createBuffer`/`deleteBuffer` calls to detect instance-buffer leaks that `renderer.info.memory.geometries` alone misses. Corrected mode requires stable buffer counts across replacements and zero tracked buffers after an empty frame and viewer disposal.

Inspect the PNGs and recorded WebM listed in `results.json`. Compare the tangent joins under identical lighting, distinguishing the reproduced ring seams from silhouette tessellation, pixel aliasing, and the fixture's deliberately changing orientation/position. Do not use test/build success alone to declare the visual artifact resolved.

## Combined feedback campaign

The validation branch includes `feedback-fixtures.py` and `feedback-integration.mjs` to exercise the completed issue stack together. The native exporter produces a growth/division/removal recording and constant-concentration presentation snapshots with missing/reappearing signal grids and device geometry. The browser check verifies fixed species/signal ranges, composite tints/order/enabled state, device visibility, stable-ID selection, camera/reference-grid retention, reverse/rapid seeking, damaged-frame recovery, keyboard navigation and the supported 880px layout. These synthetic presentation transitions are not claims about evolving native device geometry.

```console
uv run --no-sync python viewer/browser/feedback-fixtures.py /tmp/microsimulator-feedback/fixtures-v3
pnpm --dir viewer dev --host 127.0.0.1 --port 4330
```

In another POSIX terminal, run:

```console
MICROSIMULATOR_PLAYWRIGHT_MODULE=/absolute/path/to/@playwright/test/index.mjs EVIDENCE_DIR=/tmp/microsimulator-feedback node viewer/browser/feedback-integration.mjs
```

Use a new fixture/output directory for another export. Existing issue browser hooks accept Vite's cache-busting module query strings so hot reload during combined validation does not disable test-only observation. These scripts never install production debug globals.
