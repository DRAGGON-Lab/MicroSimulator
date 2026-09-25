// Run against the Vite development server. No production debug hooks are added.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir } from "node:fs/promises";
import canonicalize from "canonicalize";

const { chromium, expect } = await import(
  process.env.MICROSIMULATOR_PLAYWRIGHT_MODULE ?? "@playwright/test"
);
const url = process.env.VIEWER_URL ?? "http://127.0.0.1:4320";
const evidence =
  process.env.EVIDENCE_DIR ?? "/tmp/microsimulator-reference-grid";
await mkdir(evidence, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));

// Observe the real application instance rather than replacing its renderer.
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
async function snapshot() {
  return page.evaluate(() => {
    const v = globalThis.__testViewer;
    v.grid.updateMatrixWorld(true);
    v.camera.updateMatrixWorld(true);
    const point = v.camera.position
      .clone()
      .set(0, 0, 0)
      .applyMatrix4(v.grid.matrixWorld)
      .project(v.camera);
    return {
      grid: v.grid.matrixWorld.elements.slice(),
      camera: v.camera.position.toArray(),
      target: v.controls.target.toArray(),
      pixelOrigin: point.toArray(),
    };
  });
}
function assertStationary(actual, expected, message) {
  assert.deepEqual(actual.grid, expected.grid, message);
  for (const key of ["camera", "target", "pixelOrigin"]) {
    actual[key].forEach((value, index) => {
      assert.ok(
        Math.abs(value - expected[key][index]) < 1e-9,
        `${message}: ${key}[${index}]`,
      );
    });
  }
}
try {
  await page.goto(`${url}/?token=reference-grid-test`);
  await expect.poll(() => socket !== undefined).toBe(true);
  await send(base);
  await page.locator("#signal-visible").focus();
  await page.keyboard.press("Space");
  await page.evaluate(() => {
    const v = globalThis.__testViewer;
    v.controls.enableDamping = false;
    v.camera.position.multiplyScalar(4);
    v.controls.update();
  });
  const initial = await snapshot();
  await page.screenshot({ path: `${evidence}/initial.png` });
  await send({ ...base, time: 1, cells: [{ ...cell, length: 14 }] });
  assertStationary(
    await snapshot(),
    initial,
    "growth must preserve world grid and stationary-camera projection",
  );
  await page.screenshot({ path: `${evidence}/grown.png` });
  await send({
    ...base,
    time: 2,
    cells: [
      { ...cell, position: [5, -8, -20] },
      { ...cell, id: "2", slot: 1, position: [-5, 2, 3] },
    ],
  });
  assertStationary(
    await snapshot(),
    initial,
    "translation and division must preserve grid and camera",
  );
  await send({ ...base, time: 3, cells: [] });
  assertStationary(
    await snapshot(),
    initial,
    "empty frames must preserve grid and camera",
  );
  await send(base);
  assertStationary(
    await snapshot(),
    initial,
    "live reset must preserve grid and camera",
  );

  // Exercise UI preferences through a missing grid and reduced channel counts.
  await page.locator("#color-mode").selectOption("species");
  await page.locator("#species-channel").selectOption("2");
  await page.locator("#signal-channel").selectOption("2");
  await page.locator("#signal-slice").focus();
  await page.keyboard.press("End");
  await page.locator("#signal-axis").selectOption("x");
  await expect(page.locator("#signal-slice")).toHaveValue("4");
  await page.locator("#signal-axis").selectOption("z");
  await expect(page.locator("#signal-slice")).toHaveValue("8");
  await send({
    ...base,
    time: 4,
    species_count: 0,
    cells: [],
    signal_grid: null,
  });
  await send({
    ...base,
    time: 5,
    species_count: 1,
    cells: [],
    signal_grid: signalGrid([2, 2, 2], 1),
  });
  await expect(page.locator("#signal-slice")).toHaveValue("1");
  await send({ ...base, time: 6 });
  await expect(page.locator("#species-channel")).toHaveValue("2");
  await expect(page.locator("#signal-channel")).toHaveValue("2");
  await expect(page.locator("#signal-slice")).toHaveValue("8");
  await expect(page.locator("#signal-visible")).not.toBeChecked();

  const canvas = page.locator("#canvas-host canvas");
  const bounds = await canvas.boundingBox();
  const x = bounds.x + bounds.width / 2;
  const y = bounds.y + bounds.height / 2;
  for (const button of ["left", "right"]) {
    await page.mouse.move(x, y);
    await page.mouse.down({ button });
    await page.mouse.move(x + 80, y + 40, { steps: 8 });
    await page.mouse.up({ button });
    assert.deepEqual(
      (await snapshot()).grid,
      initial.grid,
      `${button} drag changes camera only`,
    );
  }
  await page.mouse.wheel(0, 150);
  assert.deepEqual(
    (await snapshot()).grid,
    initial.grid,
    "zoom changes camera only",
  );
  await page.locator("#fit-button").click();
  await expect
    .poll(async () =>
      page.evaluate(() => globalThis.__testViewer.cameraTransition === null),
    )
    .toBe(true);
  assert.deepEqual(
    (await snapshot()).grid,
    initial.grid,
    "Fit changes camera only",
  );

  // Explicit new dataset, including the initially empty case, in the real renderer.
  const emptyThenDevice = await page.evaluate(() => {
    const v = globalThis.__testViewer;
    const empty = {
      time: 0,
      backend: {
        kind: "cpu",
        name: "CPU",
        device: "host",
        deviceIndex: 0,
        native: true,
      },
      speciesCount: 0,
      cells: [],
      constraints: {
        boxes: [],
        cylinders: [],
        planes: [
          {
            id: "1",
            point: [1e30, 0, 0],
            inwardNormal: [1, 0, 0],
            coefficient: 1,
          },
        ],
        spheres: [],
      },
      signalGrid: null,
    };
    v.beginDataset();
    v.setFrame(empty, true);
    const before = {
      position: v.grid.position.toArray(),
      scale: v.grid.scale.toArray(),
    };
    const device = {
      ...empty,
      constraints: {
        ...empty.constraints,
        boxes: [
          {
            id: "2",
            center: [30, 40, -5],
            halfExtents: [50, 25, 3],
            coefficient: 1,
            allowedRegion: "inside",
          },
        ],
      },
    };
    v.setFrame(device);
    const retained = {
      position: v.grid.position.toArray(),
      scale: v.grid.scale.toArray(),
    };
    v.beginDataset();
    v.setFrame(device, true);
    return {
      before,
      retained,
      reopened: {
        position: v.grid.position.toArray(),
        scale: v.grid.scale.toArray(),
      },
    };
  });
  assert.deepEqual(emptyThenDevice.before, {
    position: [0, 0, -0.01],
    scale: [0.5, 0.5, 0.5],
  });
  assert.deepEqual(emptyThenDevice.retained, emptyThenDevice.before);
  assert.deepEqual(emptyThenDevice.reopened, {
    position: [30, 40, -8.01],
    scale: [5, 5, 5],
  });
  // Opening files goes through the application's explicit dataset reset path.
  await page.goto(url);
  const openFile = async (frame) => {
    await page.locator("#scene-file").setInputFiles({
      name: "same-name.scene.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(scene(frame))),
    });
    await expect(page.locator("#time-chip")).toHaveText(`t = ${frame.time}`);
  };
  await openFile(base);
  await page.locator("#color-mode").selectOption("species");
  await page.locator("#species-channel").selectOption("2");
  await page.locator("#signal-visible").focus();
  await page.keyboard.press("Space");
  await openFile({
    ...base,
    time: 11,
    constraints: {
      ...base.constraints,
      boxes: [
        {
          id: "1",
          center: [30, 40, -5],
          half_extents: [50, 25, 3],
          coefficient: 1,
          allowed_region: "inside",
        },
      ],
    },
  });
  await expect(page.locator("#color-mode")).toHaveValue("cell-type");
  await expect(page.locator("#species-channel")).toHaveValue("0");
  await expect(page.locator("#signal-visible")).toBeChecked();
  assert.deepEqual(
    await page.evaluate(() => ({
      position: globalThis.__testViewer.grid.position.toArray(),
      scale: globalThis.__testViewer.grid.scale.toArray(),
    })),
    { position: [30, 40, -8.01], scale: [5, 5, 5] },
  );
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify(
      {
        result: "passed",
        browser: browser.version(),
        assertions:
          "grid world/projected coordinates; camera orbit/pan/zoom/Fit; growth/XYZ/division/removal/reset; missing channels/grid; axis round trip; initially empty/new dataset",
        evidence,
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
}
