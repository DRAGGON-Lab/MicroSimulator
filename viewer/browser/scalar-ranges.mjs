import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir } from "node:fs/promises";
import canonicalize from "canonicalize";
const { chromium, expect } = await import(
  process.env.MICROSIMULATOR_PLAYWRIGHT_MODULE ?? "@playwright/test"
);
const url = process.env.VIEWER_URL ?? "http://127.0.0.1:4315";
const evidence =
  process.env.EVIDENCE_DIR ?? "/tmp/microsimulator-scalar-ranges";
await mkdir(evidence, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
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
const grid = {
  signal_count: 2,
  shape: [2, 1, 2],
  origin: [-2, -2, -1],
  spacing: [4, 1, 1],
  boundaries: {
    x_lower: boundary,
    x_upper: boundary,
    y_lower: boundary,
    y_upper: boundary,
    z_lower: boundary,
    z_upper: boundary,
  },
  levels: [2, 2, 4, 40, 8, 8, 6, 60],
};
const first = {
  id: "1",
  parent_id: null,
  slot: 0,
  position: [-1, 0, 0.6],
  direction: [0, 1, 0],
  length: 2,
  radius: 0.5,
  growth_rate: 0.1,
  cell_type: 0,
  fixed: false,
  species: [2, 8],
};
const second = {
  ...first,
  id: "2",
  slot: 1,
  position: [1, 0, 0.6],
  species: [4, 6],
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
  species_count: 2,
  cells: [first, second],
  constraints: { boxes: [], cylinders: [], planes: [], spheres: [] },
  signal_grid: grid,
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
}
const control = (kind, field) =>
  page.getByRole(field === "range mode" ? "combobox" : "textbox", {
    name: `${kind} ${field}`,
    exact: true,
  });
async function fixed(kind, minimum, maximum) {
  await control(kind, "range mode").focus();
  await page.keyboard.press("f");
  await page.keyboard.press("Tab");
  await expect(control(kind, "range mode")).toHaveValue("fixed");
  await control(kind, "minimum").fill(String(minimum));
  await control(kind, "maximum").fill(String(maximum));
  await control(kind, "maximum").press("Enter");
}
async function colors() {
  return page.evaluate(() => ({
    cells: Array.from(
      globalThis.__testViewer.cellMeshes[0]?.instanceColor?.array ?? [],
    ),
    signal: Array.from(globalThis.__testViewer.signalTexture?.image.data ?? []),
  }));
}
async function expectedColor(intensity) {
  return page.evaluate(async (intensity) => {
    const { rgbBytes, viridis } = await import("/src/color.ts");
    const { Color, SRGBColorSpace } =
      await import("/node_modules/.vite/deps/three.js");
    const rgb = viridis(intensity);
    const linear = new Color().setRGB(...rgb, SRGBColorSpace);
    return {
      signal: [...rgbBytes(rgb), 205],
      cell: [linear.r, linear.g, linear.b].map(Math.fround),
    };
  }, intensity);
}
try {
  await page.goto(`${url}/?token=scalar-range-test`);
  await expect.poll(() => socket !== undefined).toBe(true);
  await send(base);
  await page.locator("#color-mode").selectOption("species");
  await fixed("Species", 0, 10);
  await fixed("Signal", 0, 10);
  await expect(page.locator("#legend-mode")).toHaveText("Fixed");
  await expect(page.locator("#signal-legend-mode")).toHaveText("Fixed");
  const initial = await colors();
  const at02 = await expectedColor(0.2);
  assert.deepEqual(initial.cells.slice(0, 3), at02.cell);
  assert.deepEqual(initial.signal.slice(0, 4), at02.signal);
  await page.screenshot({ path: `${evidence}/fixed-ranges.png` });

  await send({
    ...base,
    time: 1,
    cells: [first, { ...second, species: [40, 6] }],
    signal_grid: { ...grid, levels: [2, 2, 40, 40, 8, 8, 6, 60] },
  });
  const changed = await colors();
  assert.deepEqual(changed.cells.slice(0, 3), initial.cells.slice(0, 3));
  assert.deepEqual(changed.signal.slice(0, 4), initial.signal.slice(0, 4));
  await page.locator("#signal-slice").focus();
  await page.keyboard.press("End");
  assert.deepEqual(
    (await colors()).signal.slice(0, 4),
    initial.signal.slice(0, 4),
  );
  await expect(page.locator("#legend-max")).toHaveText("10");
  await expect(page.locator("#signal-legend-max")).toHaveText("10");

  const clipped = {
    ...base,
    time: 2,
    cells: [
      { ...first, species: [-5, 8] },
      { ...second, species: [50, 6] },
    ],
    signal_grid: { ...grid, levels: [-5, -5, 50, 50, 8, 8, 6, 60] },
  };
  await send(clipped);
  const endpoints = await colors();
  const low = await expectedColor(0);
  const high = await expectedColor(1);
  assert.deepEqual(endpoints.cells, [...low.cell, ...high.cell]);
  assert.deepEqual(endpoints.signal, [...low.signal, ...high.signal]);
  await page.evaluate(() => globalThis.__testViewer.selectCell(0));
  await expect(page.locator("#species-values code").first()).toHaveText("-5");
  assert.deepEqual(
    await page.evaluate(() =>
      globalThis.__testViewer.cells.map((cell) => cell.species[0]),
    ),
    [-5, 50],
  );

  for (const kind of ["Species", "Signal"]) {
    await control(kind, "minimum").fill("20");
    await control(kind, "maximum").press("Enter");
    await expect(
      page
        .getByRole("form", { name: `${kind} concentration range` })
        .getByRole("status"),
    ).toContainText("Minimum must be less than maximum");
    assert.deepEqual(await colors(), endpoints);
    await control(kind, "minimum").fill("Infinity");
    await control(kind, "maximum").press("Enter");
    await expect(
      page
        .getByRole("form", { name: `${kind} concentration range` })
        .getByRole("status"),
    ).toContainText("finite number");
    assert.deepEqual(await colors(), endpoints);
    await control(kind, "minimum").fill("");
    await control(kind, "maximum").press("Enter");
    await expect(
      page
        .getByRole("form", { name: `${kind} concentration range` })
        .getByRole("status"),
    ).toContainText("finite number");
    await fixed(kind, 0, 10);
  }

  await page.locator("#species-channel").selectOption("1");
  await expect(control("Species", "range mode")).toHaveValue("automatic");
  await fixed("Species", -100, 100);
  await page.locator("#species-channel").selectOption("0");
  await expect(control("Species", "minimum")).toHaveValue("0");
  await expect(control("Species", "maximum")).toHaveValue("10");
  await page.locator("#signal-channel").selectOption("1");
  await expect(control("Signal", "range mode")).toHaveValue("automatic");
  await fixed("Signal", -2, 2);
  await page.locator("#signal-channel").selectOption("0");
  await expect(control("Signal", "maximum")).toHaveValue("10");
  await send({
    ...base,
    time: 3,
    species_count: 0,
    cells: [],
    signal_grid: null,
  });
  await send(base); // Same live model reset / seek back to time zero.
  await expect(control("Species", "range mode")).toHaveValue("fixed");
  await expect(control("Species", "maximum")).toHaveValue("10");
  await expect(control("Signal", "maximum")).toHaveValue("10");
  assert.deepEqual((await colors()).cells, initial.cells);

  for (const kind of ["Species", "Signal"]) {
    await control(kind, "range mode").focus();
    await page.keyboard.press("a");
    await page.keyboard.press("Tab");
  }
  await send({
    ...base,
    time: 4,
    cells: [
      { ...first, species: [7, 7] },
      { ...second, species: [7, 7] },
    ],
    signal_grid: { ...grid, levels: Array(8).fill(7) },
  });
  await expect(page.locator("#legend-mode")).toHaveText("Automatic · constant");
  await expect(page.locator("#signal-legend-mode")).toHaveText(
    "Automatic · constant",
  );
  const midpoint = await expectedColor(0.5);
  assert.deepEqual((await colors()).cells, [
    ...midpoint.cell,
    ...midpoint.cell,
  ]);
  assert.deepEqual((await colors()).signal, [
    ...midpoint.signal,
    ...midpoint.signal,
  ]);
  await send({ ...base, time: 5, cells: [], signal_grid: null });
  await expect(page.locator("#legend-mode")).toHaveText(
    "Automatic · no values",
  );
  await expect(page.locator("#legend-min")).toHaveText("—");
  await expect(page.locator("#legend-ramp")).toBeHidden();
  await send(base);
  await control("Species", "range mode").selectOption("fixed");
  await expect(control("Species", "maximum")).toHaveValue("10");
  await control("Signal", "range mode").selectOption("fixed");
  await expect(control("Signal", "maximum")).toHaveValue("10");

  await page.goto(url);
  const open = async (value) => {
    await page.locator("#scene-file").setInputFiles({
      name: "scene.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(scene(value))),
    });
    await expect(page.locator("#time-chip")).toHaveText(`t = ${value.time}`);
  };
  await open(base);
  await page.locator("#color-mode").selectOption("species");
  await fixed("Species", 0, 10);
  await fixed("Signal", 0, 10);
  await open({ ...base, time: 99 });
  await page.locator("#color-mode").selectOption("species");
  await expect(control("Species", "range mode")).toHaveValue("automatic");
  await expect(control("Signal", "range mode")).toHaveValue("automatic");
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify(
      {
        result: "passed",
        browser: browser.version(),
        evidence,
        coverage:
          "cell instance RGB, signal texture RGBA, fixed scales across frames/slices, endpoints and inspector, invalid bounds, keyboard controls, independent channels, missing data, reset/seek, constant/empty, new dataset",
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
}
