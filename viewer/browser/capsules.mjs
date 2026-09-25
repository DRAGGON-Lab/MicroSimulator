// Vite browser regression: same fixture/camera/light for baseline and corrected meshes.
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";

const { chromium } = await import(
  process.env.MICROSIMULATOR_PLAYWRIGHT_MODULE ?? "@playwright/test"
);
const url = process.env.VIEWER_URL ?? "http://127.0.0.1:4321";
const mode = process.env.CAPSULE_MODE ?? "corrected";
const evidence =
  process.env.EVIDENCE_DIR ?? `/tmp/microsimulator-capsules-${mode}`;
await mkdir(evidence, { recursive: true });
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1440, height: 960 },
  recordVideo: { dir: evidence, size: { width: 1440, height: 960 } },
});
const page = await context.newPage();
const errors = [];
page.on("pageerror", (error) => {
  errors.push(error.message);
  console.error(error.message);
});
page.on("console", (message) => {
  if (message.type() === "error") console.error(message.text());
});
page.on("requestfailed", (request) =>
  console.error(request.url(), request.failure()),
);
await page.route("**/src/colony-viewer.ts*", async (route) => {
  const response = await route.fetch();
  const source = await response.text();
  const marker = "this.onSelection = onSelection;";
  assert.equal(source.split(marker).length, 2);
  await route.fulfill({
    response,
    body: source.replace(marker, `${marker}\nglobalThis.__testViewer = this;`),
  });
});

try {
  await page.goto(url);
  await page.waitForFunction(() => globalThis.__testViewer !== undefined);
  await page.evaluate(() => {
    const v = globalThis.__testViewer;
    v.renderer.setAnimationLoop(null);
    v.controls.enableDamping = false;
    v.grid.visible = false;
    const gl = v.renderer.getContext();
    const create = gl.createBuffer.bind(gl);
    const remove = gl.deleteBuffer.bind(gl);
    const buffers = new Set();
    gl.createBuffer = () => {
      const b = create();
      buffers.add(b);
      return b;
    };
    gl.deleteBuffer = (b) => {
      buffers.delete(b);
      return remove(b);
    };
    globalThis.__liveBuffers = buffers;
    globalThis.__fixture = (count = 1, time = 0) => {
      const columns = count === 1 ? 1 : Math.ceil(Math.sqrt(count * 2));
      const rows = Math.ceil(count / columns);
      return {
        time,
        backend: {
          kind: "cpu",
          name: "synthetic geometry fixture",
          device: "host",
          deviceIndex: 0,
          native: true,
        },
        speciesCount: 0,
        signalGrid: null,
        constraints: { boxes: [], cylinders: [], planes: [], spheres: [] },
        cells: Array.from({ length: count }, (_, i) => {
          const angle = 0.4 + 0.09 * Math.sin(i * 0.7 + time);
          const tilt = 0.15 * Math.cos(i * 0.9 + time);
          return {
            id: String(i + 1),
            parentId: null,
            slot: i,
            position: [
              ((i % columns) - (columns - 1) / 2) * 1.9 +
                0.04 * Math.sin(time + i),
              (Math.floor(i / columns) - (rows - 1) / 2) * 1.05,
              0,
            ],
            direction: [
              Math.cos(angle) * Math.cos(tilt),
              Math.sin(angle) * Math.cos(tilt),
              Math.sin(tilt),
            ],
            length: count === 1 ? 3 : 1.05,
            radius: count === 1 ? 0.6 : 0.4,
            growthRate: 0,
            cellType: i % 3,
            fixed: false,
            species: [],
          };
        }),
      };
    };
    globalThis.__present = (count, time, distance) => {
      const frame = globalThis.__fixture(count, time);
      v.setFrame(frame, false);
      v.grid.visible = false;
      v.setCellColors(
        frame.cells.map((cell) =>
          cell.cellType === 0
            ? [0.65, 0.85, 0.72]
            : cell.cellType === 1
              ? [0.85, 0.55, 0.3]
              : [0.45, 0.65, 0.9],
        ),
      );
      if (distance !== undefined) {
        v.camera.position.set(0, -distance, distance * 0.65);
        v.camera.up.set(0, 0, 1);
        v.controls.target.set(0, 0, 0);
        v.camera.lookAt(v.controls.target);
        v.controls.update();
      }
      v.renderer.render(v.scene, v.camera);
    };
    document.querySelector("#empty-state").hidden = true;
  });
  for (const [name, count, distance] of [
    ["isolated-near", 1, 6],
    ["isolated-far", 1, 18],
    ["colony-near", 64, 12],
    ["colony-far", 64, 32],
  ]) {
    await page.evaluate(
      ([n, d]) => globalThis.__present(n, 0, d),
      [count, distance],
    );
    await page
      .locator("#canvas-host")
      .screenshot({ path: `${evidence}/${name}.png` });
  }
  for (let frame = 0; frame < 48; frame++) {
    await page.evaluate((time) => {
      globalThis.__present(64, time, 12);
      return new Promise(requestAnimationFrame);
    }, frame / 24);
    if (frame % 12 === 0) {
      await page
        .locator("#canvas-host")
        .screenshot({ path: `${evidence}/moving-${frame}.png` });
    }
  }
  const metrics = await page.evaluate(async () => {
    const v = globalThis.__testViewer;
    const samples = [];
    const resources = [];
    for (let frame = 0; frame < 70; frame++) {
      await new Promise(requestAnimationFrame);
      const start = performance.now();
      globalThis.__present(512, frame / 24, 60);
      // Complete GPU work so timings compare actual rendering, not only enqueue time.
      v.renderer.getContext().finish();
      if (frame >= 10) samples.push(performance.now() - start);
      if (frame === 10 || frame === 69) {
        resources.push({
          geometries: v.renderer.info.memory.geometries,
          buffers: globalThis.__liveBuffers.size,
        });
      }
    }
    const info = { ...v.renderer.info.render };
    const gl = v.renderer.getContext();
    const debug = gl.getExtension("WEBGL_debug_renderer_info");
    const device = debug
      ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL)
      : gl.getParameter(gl.RENDERER);
    const vertices = v.cellMeshes.map(
      (mesh) => mesh.geometry.getAttribute("position").count,
    );
    const triangles = v.cellMeshes.map((mesh) => mesh.geometry.index.count / 3);
    samples.sort((a, b) => a - b);
    v.setFrame(globalThis.__fixture(0), false);
    v.renderer.render(v.scene, v.camera);
    return {
      samples: samples.length,
      device,
      medianMilliseconds: samples[30],
      p95Milliseconds: samples[57],
      render: info,
      vertices,
      triangles,
      resources,
      emptyBuffers: globalThis.__liveBuffers.size,
    };
  });

  const behavior = await page.evaluate(() => {
    const v = globalThis.__testViewer;
    globalThis.__present(2, 0, 6);
    v.selectCell(1);
    const selected = v.selectedCellId;
    const frame = globalThis.__fixture(2, 1);
    frame.cells.reverse();
    frame.cells = frame.cells.map((cell, slot) => ({ ...cell, slot }));
    v.setFrame(frame, false);
    v.setCellColors([
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const retained = v.selectedCellId;
    const highlights = v.highlight.children.map((mesh) => {
      mesh.updateMatrixWorld(true);
      return { matrix: mesh.matrixWorld.elements.slice() };
    });
    const colors = v.cellMeshes.map((mesh) =>
      Array.from(mesh.instanceColor.array),
    );
    v.selectCell(null);
    const center = v.camera.position
      .clone()
      .fromArray(frame.cells[0].position)
      .project(v.camera);
    const canvas = v.renderer.domElement.getBoundingClientRect();
    v.renderer.render(v.scene, v.camera);
    return {
      selected,
      retained,
      highlights,
      colors,
      point: {
        x: canvas.x + ((center.x + 1) * canvas.width) / 2,
        y: canvas.y + ((1 - center.y) * canvas.height) / 2,
      },
    };
  });
  assert.equal(behavior.selected, "2");
  assert.equal(behavior.retained, "2");
  for (const colors of behavior.colors)
    assert.deepEqual(colors, [1, 0, 0, 0, 1, 0]);
  await page.mouse.click(behavior.point.x, behavior.point.y);
  assert.equal(
    await page.evaluate(() => globalThis.__testViewer.selectedCellId),
    "2",
  );
  const highlightError = await page.evaluate(() => {
    const v = globalThis.__testViewer;
    const selected = v.cells.find((cell) => cell.id === v.selectedCellId);
    const center = v.camera.position.clone().fromArray(selected.position);
    const axis = center.clone().fromArray(selected.direction).normalize();
    let error = 0;
    for (const mesh of v.highlight.children) {
      mesh.updateMatrixWorld(true);
      const positions = mesh.geometry.getAttribute("position");
      for (let i = 0; i < positions.count; i++) {
        const point = center
          .clone()
          .fromBufferAttribute(positions, i)
          .applyMatrix4(mesh.matrixWorld);
        const along = point.clone().sub(center).dot(axis);
        const nearest = center
          .clone()
          .addScaledVector(
            axis,
            Math.max(
              -selected.length / 2,
              Math.min(selected.length / 2, along),
            ),
          );
        error = Math.max(
          error,
          Math.abs(point.distanceTo(nearest) - selected.radius * 1.08),
        );
      }
    }
    v.renderer.render(v.scene, v.camera);
    return error;
  });
  if (mode === "corrected") assert.ok(highlightError < 1e-6);
  await page
    .locator("#canvas-host")
    .screenshot({ path: `${evidence}/selected.png` });
  const removed = await page.evaluate(() => {
    const v = globalThis.__testViewer;
    v.setFrame(globalThis.__fixture(1), false);
    return { selected: v.selectedCellId, visible: v.highlight.visible };
  });
  assert.deepEqual(removed, { selected: null, visible: false });
  if (mode === "corrected") {
    const zero = await page.evaluate(() => {
      const v = globalThis.__testViewer;
      const frame = globalThis.__fixture(1);
      frame.cells[0].length = 0;
      v.setFrame(frame, false);
      v.setCellColors([[0.6, 0.8, 0.7]]);
      v.selectCell(0);
      v.renderer.render(v.scene, v.camera);
      return v.highlight.children.every((mesh) =>
        mesh.matrix.elements.every(Number.isFinite),
      );
    });
    assert.equal(zero, true, "zero-length selection has finite transforms");
    await page
      .locator("#canvas-host")
      .screenshot({ path: `${evidence}/zero-length-selected.png` });
  }
  if (mode === "corrected") {
    assert.equal(
      metrics.render.calls,
      3,
      "cell rendering stays at three instanced draws",
    );
    assert.equal(
      metrics.resources[0].buffers,
      metrics.resources[1].buffers,
      "frame replacement must free old instance buffers",
    );
    assert.equal(
      metrics.emptyBuffers,
      0,
      "empty frame must free cell GPU buffers",
    );
  }
  assert.deepEqual(errors, []);
  const disposedBuffers = await page.evaluate(() => {
    globalThis.__testViewer.dispose();
    return globalThis.__liveBuffers.size;
  });
  if (mode === "corrected") assert.equal(disposedBuffers, 0);
  const report = {
    mode,
    browser: browser.version(),
    fixture:
      "512 cells; 10 warmup + 60 measured replacement/color/render/gl.finish frames; fixed camera and lighting; 64-cell prescribed motion video",
    metrics,
    behavior,
    highlightError,
    disposedBuffers,
    video: await page.video().path(),
  };
  await writeFile(`${evidence}/results.json`, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally {
  await context.close();
  await browser.close();
}
