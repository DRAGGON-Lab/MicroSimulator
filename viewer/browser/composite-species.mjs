import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir } from "node:fs/promises";
import canonicalize from "canonicalize";
const { chromium, expect } = await import(
  process.env.MICROSIMULATOR_PLAYWRIGHT_MODULE ?? "@playwright/test"
);
const url = process.env.VIEWER_URL ?? "http://127.0.0.1:4319";
const evidence =
  process.env.EVIDENCE_DIR ?? "/tmp/microsimulator-composite-species";
await mkdir(evidence, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
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
  channel_metadata: {
    species: ["Red reporter", "Green reporter"],
    signals: [],
  },
  cells: [
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],
  ].map((species, slot) => ({
    id: String(slot + 1),
    parent_id: slot === 2 ? "1" : null,
    slot,
    position: [slot % 2 === 0 ? -2 : 2, slot < 2 ? -2 : 2, 0.6],
    direction: [1, 0, 0],
    length: 1.5,
    radius: 0.45,
    growth_rate: 0.1,
    cell_type: slot,
    fixed: false,
    species,
  })),
  constraints: { boxes: [], cylinders: [], planes: [], spheres: [] },
  signal_grid: null,
};
function scene(frame) {
  return {
    format: "microsimulator-scene",
    version: 3,
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
const row = (index) =>
  page.locator(`.composite-channel[data-channel="${index}"]`);
async function fixed(index, minimum, maximum) {
  const group = row(index);
  if (!(await group.locator("details").evaluate((details) => details.open)))
    await group.locator("summary").click();
  await group
    .getByRole("combobox", { name: "Species range mode", exact: true })
    .selectOption("fixed");
  await group
    .getByRole("textbox", { name: "Species minimum", exact: true })
    .fill(String(minimum));
  await group
    .getByRole("textbox", { name: "Species maximum", exact: true })
    .fill(String(maximum));
  await group
    .getByRole("textbox", { name: "Species maximum", exact: true })
    .press("Enter");
}
async function colors() {
  return page.evaluate(() =>
    Array.from(
      globalThis.__testViewer.cellMeshes[0]?.instanceColor?.array ?? [],
    ),
  );
}
function close(actual, expected, tolerance = 1e-5) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) =>
    assert.ok(
      Math.abs(value - expected[index]) < tolerance,
      `component ${index}: ${value} != ${expected[index]}`,
    ),
  );
}
async function clickCell(index) {
  const position = await page.evaluate((index) => {
    const v = globalThis.__testViewer;
    v.camera.updateMatrixWorld(true);
    const point = v.camera.position
      .clone()
      .fromArray(v.cells[index].position)
      .project(v.camera);
    const bounds = v.renderer.domElement.getBoundingClientRect();
    return {
      x: bounds.x + ((point.x + 1) * bounds.width) / 2,
      y: bounds.y + ((1 - point.y) * bounds.height) / 2,
    };
  }, index);
  await page.mouse.click(position.x, position.y);
}
try {
  await page.goto(`${url}/?token=composite-test`);
  await expect.poll(() => socket !== undefined).toBe(true);
  await send(base);
  await page.locator("#color-mode").selectOption("composite");
  await fixed(0, 0, 1);
  await fixed(1, 0, 1);
  const redGreenYellowBlack = [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0];
  close(await colors(), redGreenYellowBlack);
  await expect(page.locator("#composite-legend")).toContainText(
    "Red reporter · #ff0000 · Fixed 0 to 1",
  );
  await expect(page.locator("#composite-legend")).toContainText(
    "Green reporter · #00ff00 · Fixed 0 to 1",
  );
  await row(0).locator("summary").click();
  await row(1).locator("summary").click();
  await page.evaluate(() => {
    const v = globalThis.__testViewer;
    v.controls.enableDamping = false;
    v.camera.position
      .sub(v.controls.target)
      .multiplyScalar(1.25)
      .add(v.controls.target);
    v.controls.update();
  });
  await page.screenshot({ path: `${evidence}/expression-patterns.png` });
  await clickCell(2);
  await expect(page.locator("#selection-title")).toHaveText("Cell 3");
  await expect(
    page
      .locator("#cell-details > div")
      .filter({ has: page.locator("dt", { hasText: "Parent" }) })
      .locator("dd"),
  ).toHaveText("1");
  await expect(page.locator("#species-values code")).toHaveText(["1", "1"]);
  assert.equal(
    await page.evaluate(() => globalThis.__testViewer.highlight.visible),
    true,
  );
  await page.screenshot({ path: `${evidence}/selected-patterns.png` });

  for (let time = 1; time <= 3; time += 1) {
    await send({
      ...base,
      time,
      cells: base.cells.map((cell) => ({
        ...cell,
        position: [
          cell.position[0] + time / 3,
          cell.position[1] - time / 4,
          cell.position[2],
        ],
      })),
    });
    close(await colors(), redGreenYellowBlack);
    await expect(page.locator("#selection-title")).toHaveText("Cell 3");
    assert.equal(
      await page.evaluate(() => globalThis.__testViewer.highlight.visible),
      true,
    );
  }
  await page.screenshot({ path: `${evidence}/moving-patterns.png` });
  await row(1)
    .getByRole("checkbox", { name: "Enable Green reporter" })
    .uncheck();
  close(await colors(), [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]);
  await row(0).getByRole("checkbox", { name: "Enable Red reporter" }).uncheck();
  await expect(page.locator("#composite-legend")).toContainText(
    "No channels active",
  );
  const neutral = await page.evaluate(async () => {
    const { COMPOSITE_NEUTRAL } = await import("/src/composite-color.ts");
    const { Color, SRGBColorSpace } =
      await import("/node_modules/.vite/deps/three.js");
    const value = new Color().setRGB(...COMPOSITE_NEUTRAL, SRGBColorSpace);
    return [value.r, value.g, value.b];
  });
  close(await colors(), [...neutral, ...neutral, ...neutral, ...neutral]);
  await row(0).getByRole("checkbox").check();
  await row(1).getByRole("checkbox").check();
  close(await colors(), redGreenYellowBlack);
  await row(1).locator("summary").click();
  const beforeOrder = await colors();
  await row(1).getByRole("button", { name: "Move Green reporter up" }).click();
  assert.deepEqual(
    await page
      .locator(".composite-channel")
      .evaluateAll((rows) => rows.map((row) => row.dataset.channel)),
    ["1", "0"],
  );
  assert.deepEqual(await colors(), beforeOrder);
  await expect(
    row(1).getByRole("textbox", { name: "Species maximum", exact: true }),
  ).toHaveValue("1");

  await fixed(0, 0, 2);
  await page.locator("#color-mode").selectOption("species");
  await page.locator("#species-channel").selectOption("0");
  await expect(
    page
      .locator("#species-range")
      .getByRole("textbox", { name: "Species maximum", exact: true }),
  ).toHaveValue("2");
  await page
    .locator("#species-range")
    .getByRole("textbox", { name: "Species maximum", exact: true })
    .fill("3");
  await page
    .locator("#species-range")
    .getByRole("textbox", { name: "Species maximum", exact: true })
    .press("Enter");
  await page.locator("#color-mode").selectOption("composite");
  await expect(
    row(0).getByRole("textbox", { name: "Species maximum", exact: true }),
  ).toHaveValue("3");
  await fixed(0, 0, 1);
  await row(0).getByRole("textbox", { name: "Channel tint" }).fill("#808080");
  await row(0).getByRole("textbox", { name: "Channel tint" }).press("Enter");
  await send({
    ...base,
    time: 4,
    cells: [{ ...base.cells[0], species: [0.5, 0] }, ...base.cells.slice(1)],
  });
  const decode = (value) => ((value + 0.055) / 1.055) ** 2.4;
  close((await colors()).slice(0, 3), Array(3).fill(0.5 * decode(128 / 255)));
  const validTintColors = await colors();
  await row(0).getByRole("textbox", { name: "Channel tint" }).fill("invalid");
  await row(0).getByRole("textbox", { name: "Channel tint" }).press("Enter");
  await expect(
    row(0).locator(".composite-tint-form [role=status]"),
  ).toContainText("six-digit");
  assert.deepEqual(await colors(), validTintColors);
  await row(0).getByRole("textbox", { name: "Channel tint" }).fill("#ff0000");
  await row(0).getByRole("textbox", { name: "Channel tint" }).press("Enter");
  await send({
    ...base,
    time: 5,
    cells: [{ ...base.cells[0], species: [-1, 2] }, ...base.cells.slice(1)],
  });
  close((await colors()).slice(0, 3), [0, 1, 0]);
  await clickCell(0);
  await expect(page.locator("#species-values code")).toHaveText(["-1", "2"]);

  await row(1).getByRole("checkbox").uncheck();
  await send({
    ...base,
    time: 6,
    species_count: 0,
    channel_metadata: { species: [], signals: [] },
    cells: [],
  });
  await expect(page.locator("#selection-title")).toHaveText("No cell selected");
  await send(base); // Backward seek / live reset contract.
  await expect(page.locator("#color-mode")).toHaveValue("composite");
  await expect(row(1).getByRole("checkbox")).not.toBeChecked();
  await expect(
    row(0).getByRole("textbox", { name: "Species maximum", exact: true }),
  ).toHaveValue("1");
  assert.deepEqual(
    await page
      .locator(".composite-channel")
      .evaluateAll((rows) => rows.map((row) => row.dataset.channel)),
    ["1", "0"],
  );

  await page.goto(url);
  const open = async (frame) => {
    await page.locator("#scene-file").setInputFiles({
      name: "named.scene.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(scene(frame))),
    });
    await expect(page.locator("#time-chip")).toHaveText(`t = ${frame.time}`);
  };
  await open(base);
  await page.locator("#color-mode").selectOption("composite");
  await row(0).getByRole("checkbox").uncheck();
  await fixed(1, 0, 20);
  await open({ ...base, time: 99 });
  await page.locator("#color-mode").selectOption("composite");
  await expect(row(0).getByRole("checkbox")).toBeChecked();
  await row(1).locator("summary").click();
  await expect(
    row(1).getByRole("combobox", { name: "Species range mode", exact: true }),
  ).toHaveValue("automatic");
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify(
      {
        result: "passed",
        browser: browser.version(),
        evidence,
        coverage:
          "linear RGB/colorspace, coexpression/motion, disable/neutral, reorder, shared ranges, tint validation, clipping, picking/highlight/lineage, absent channels/reset/new dataset",
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
}
