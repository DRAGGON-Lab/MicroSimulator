// Run against the Vite development server. No production debug hooks are added.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir } from "node:fs/promises";
import canonicalize from "canonicalize";

const { chromium, expect } = await import(
  process.env.MICROSIMULATOR_PLAYWRIGHT_MODULE ?? "@playwright/test"
);
const url = process.env.VIEWER_URL ?? "http://127.0.0.1:4323";
const evidence =
  process.env.EVIDENCE_DIR ?? "/tmp/microsimulator-device-visibility";
await mkdir(evidence, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));

// Observe the real application instance rather than replacing its renderer.
await page.route("**/src/colony-viewer.ts", async (route) => {
  const response = await route.fetch();
  const source = await response.text();
  const marker = "this.onSelection = onSelection;";
  assert.equal(source.split(marker).length, 2);
  await route.fulfill({
    response,
    body: source.replace(marker, `${marker}\nglobalThis.__testViewer = this;`),
  });
});
let socket;
await page.routeWebSocket("**/api/v1/session?*", (connection) => {
  socket = connection;
});
const boundary = { kind: "no_flux", values: [] };
function signalGrid(shape = [5, 7, 9], signalCount = 3) {
  return {
    signal_count: signalCount,
    shape,
    origin: [-2, -3, 0],
    spacing: [1, 1, 1],
    boundaries: {
      x_lower: boundary,
      x_upper: boundary,
      y_lower: boundary,
      y_upper: boundary,
      z_lower: boundary,
      z_upper: boundary,
    },
    levels: Array(shape.reduce((a, b) => a * b, signalCount)).fill(1),
  };
}
const cell = {
  id: "1",
  parent_id: null,
  slot: 0,
  position: [0, 0, 0.6],
  direction: [1, 0, 0],
  length: 2,
  radius: 0.5,
  growth_rate: 0.1,
  cell_type: 0,
  fixed: false,
  species: [1, 2, 3],
};
const base = {
  backend: {
    kind: "cpu",
    name: "CPU fixture",
    device: "host",
    device_index: 0,
    native: true,
  },
  time: 0,
  species_count: 3,
  cells: [cell],
  constraints: { boxes: [], cylinders: [], planes: [], spheres: [] },
  signal_grid: signalGrid(),
};
function scene(frame) {
  return {
    format: "microsimulator-scene",
    version: 2,
    producer: { name: "microsimulator", version: "test" },
    integrity: {
      algorithm: "sha256",
      frame: createHash("sha256").update(canonicalize(frame)).digest("hex"),
    },
    frame,
  };
}
let revision = 0;
async function send(frame) {
  socket.send(
    JSON.stringify({
      type: "frame",
      revision: revision++,
      completed_steps: revision,
      playing: false,
      checkpoint_enabled: false,
      scene: scene(frame),
    }),
  );
  await expect(page.locator("#time-chip")).toHaveText(`t = ${frame.time}`);
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      ),
  );
}
const constraints = {
  boxes: [
    {
      id: "1",
      center: [4, 0, 1],
      half_extents: [1, 2, 1],
      allowed_region: "outside",
      coefficient: 1,
    },
  ],
  spheres: [
    {
      id: "2",
      center: [-4, 0, 1],
      radius: 1.5,
      allowed_region: "outside",
      coefficient: 1,
    },
  ],
  cylinders: [
    {
      id: "3",
      center: [0, 4, 1],
      radius: 1,
      half_height: 2,
      allowed_region: "outside",
      coefficient: 1,
    },
  ],
  planes: [
    { id: "4", point: [0, 0, -0.5], inward_normal: [0, 0, 1], coefficient: 1 },
  ],
};
const all = { ...base, constraints };
async function snapshot() {
  return page.evaluate(() => {
    const v = globalThis.__testViewer;
    v.grid.updateMatrixWorld(true);
    return {
      camera: [
        ...v.camera.position.toArray(),
        ...v.camera.quaternion.toArray(),
      ],
      bounds: [v.sceneBounds.min.toArray(), v.sceneBounds.max.toArray()],
      grid: v.grid.matrixWorld.elements.slice(),
      visible: v.device.visible,
      deviceChildren: v.device.children.length,
      cells: v.colony.visible && v.cellMeshes.length > 0,
      highlight: v.highlight.visible,
      signal: v.signal.visible && v.signal.children.length > 0,
    };
  });
}
function assertSnapshot(actual, expected, message) {
  actual.camera.forEach((value, index) =>
    assert.ok(Math.abs(value - expected.camera[index]) < 1e-9, message),
  );
  assert.deepEqual(
    { ...actual, camera: [] },
    { ...expected, camera: [] },
    message,
  );
}
async function pickCell() {
  const point = await page.evaluate(() => {
    const v = globalThis.__testViewer;
    const rect = v.renderer.domElement.getBoundingClientRect();
    const p = v.camera.position.clone().set(0, 0, 0.6).project(v.camera);
    return [
      rect.left + ((p.x + 1) * rect.width) / 2,
      rect.top + ((1 - p.y) * rect.height) / 2,
    ];
  });
  await page.mouse.click(...point);
  await expect(page.locator("#selection-title")).toHaveText("Cell 1");
}
try {
  await page.goto(`${url}/?token=device-test`);
  await expect.poll(() => socket !== undefined).toBe(true);
  await send(all);
  const toggle = page.getByLabel("Show device geometry", { exact: true });
  await expect(toggle).toBeEnabled();
  await expect(toggle).toBeChecked();
  await pickCell();
  const initial = await snapshot();
  assert.equal(
    initial.deviceChildren,
    6,
    "all four meshes plus box/cylinder outlines",
  );
  assert.ok(initial.cells && initial.signal && initial.highlight);
  await page.screenshot({ path: `${evidence}/visible.png` });
  await toggle.focus();
  await page.keyboard.press("Space");
  await expect(toggle).not.toBeChecked();
  const hidden = await snapshot();
  assertSnapshot(hidden, { ...initial, visible: false });
  await page.screenshot({ path: `${evidence}/hidden.png` });
  await page.locator("#clear-selection").click();
  await pickCell();
  for (const time of [1, 20, 3, 0]) {
    await send({ ...all, time });
    await expect(toggle).not.toBeChecked();
    const current = await snapshot();
    assertSnapshot(
      current,
      hidden,
      "updates/reset/reverse-time retain display state",
    );
  }
  await send({ ...base, time: 21 });
  await expect(toggle).toBeDisabled();
  await expect(toggle).not.toBeChecked();
  await send({ ...all, time: 22 });
  await expect(toggle).toBeEnabled();
  await expect(toggle).not.toBeChecked();
  await page.locator("#fit-button").click();
  await expect
    .poll(async () =>
      page.evaluate(() => globalThis.__testViewer.cameraTransition === null),
    )
    .toBe(true);
  const fittedHidden = await snapshot();
  await toggle.focus();
  await page.keyboard.press("Space");
  const shown = await snapshot();
  assertSnapshot(shown, { ...fittedHidden, visible: true });
  await page.locator("#fit-button").click();
  await expect
    .poll(async () =>
      page.evaluate(() => globalThis.__testViewer.cameraTransition === null),
    )
    .toBe(true);
  assertSnapshot(
    await snapshot(),
    shown,
    "Fit bounds and pose are visibility independent",
  );
  // A separate standalone file opening starts a new dataset and restores enabled.
  await page.goto(url);
  const file = {
    name: "device.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(scene(all))),
  };
  await page.locator("#scene-file").setInputFiles(file);
  await expect(toggle).toBeChecked();
  await toggle.uncheck();
  await page.locator("#scene-file").setInputFiles(file);
  await expect(toggle).toBeChecked();
  await pickCell();
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      result: "passed",
      browser: browser.version(),
      assertions:
        "device meshes/outlines, sibling visibility, keyboard, picking, camera, Fit, live updates/reset, missing geometry, new dataset",
    }),
  );
} finally {
  await browser.close();
}
