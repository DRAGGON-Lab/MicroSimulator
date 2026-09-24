import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import canonicalize from "canonicalize";

const { chromium, expect } = await import(
  process.env.MICROSIMULATOR_PLAYWRIGHT_MODULE ?? "@playwright/test"
);
const url = process.env.VIEWER_URL ?? "http://127.0.0.1:4330";
const evidence = process.env.EVIDENCE_DIR ?? "/tmp/microsimulator-feedback";
const fixtures = process.env.REPLAY_FIXTURES ?? `${evidence}/fixtures-v3`;
await mkdir(evidence, { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 960 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/src/colony-viewer.ts*", async (route) => {
    const response = await route.fetch();
    const source = await response.text();
    const marker = "this.onSelection = onSelection;";
    assert.equal(source.split(marker).length, 2);
    await route.fulfill({
      response,
      body: source.replace(
        marker,
        `${marker}\nglobalThis.__testViewer = this;`,
      ),
    });
  });
  await page.route("**/src/replay-bundle.ts*", async (route) => {
    const response = await route.fetch();
    const source = await response.text();
    const marker = "const bytes = await this.read(file, signal);";
    assert.equal(source.split(marker).length, 2);
    await route.fulfill({
      response,
      body: source.replace(
        marker,
        `await globalThis.__delayReplayLoad?.(ordinal);\n${marker}`,
      ),
    });
  });
  await page.goto(url);
  async function open(name, count) {
    await page
      .locator("#recording-folder")
      .setInputFiles(path.resolve(fixtures, name));
    await expect(page.locator("#replay-position")).toContainText(
      `1 / ${count}`,
    );
    await expect(page.locator("#replay-message")).toHaveText("");
  }
  async function seek(ordinal, count) {
    await page.locator("#replay-timeline").fill(String(ordinal));
    await page.locator("#replay-timeline").dispatchEvent("input");
    await expect(page.locator("#replay-position")).toContainText(
      `${ordinal + 1} / ${count}`,
    );
    await expect(page.locator("#replay-message")).toHaveText("");
  }
  const snapshot = () =>
    page.evaluate(() => {
      const viewer = globalThis.__testViewer;
      viewer.grid.updateMatrixWorld(true);
      return {
        camera: viewer.camera.position.toArray(),
        target: viewer.controls.target.toArray(),
        grid: viewer.grid.matrixWorld.elements.slice(),
      };
    });
  function assertStationary(actual, expected) {
    assert.deepEqual(actual.grid, expected.grid);
    for (const key of ["camera", "target"])
      actual[key].forEach((value, index) => {
        assert.ok(
          Math.abs(value - expected[key][index]) < 1e-9,
          `${key}[${index}] remains stationary`,
        );
      });
  }
  await open("lifecycle", 5);
  await expect(page.locator("#time-chip")).toHaveText("t = 0.2");
  await page.evaluate(() => {
    const v = globalThis.__testViewer;
    v.selectCell(0);
    v.controls.enableDamping = false;
    v.camera.position.multiplyScalar(2);
    v.controls.update();
  });
  const initial = await snapshot();
  await expect(page.locator("#selection-title")).toHaveText("Cell 1");
  await seek(1, 5);
  await expect(page.locator("#selection-title")).toHaveText("No cell selected");
  await expect(page.locator("#cell-count")).toHaveText("2");
  await page.evaluate(() => globalThis.__testViewer.selectCell(1));
  await expect(page.locator("#selection-title")).toHaveText("Cell 3");
  await seek(2, 5);
  await expect(page.locator("#selection-title")).toHaveText("Cell 3");
  await expect(page.locator("#cell-count")).toHaveText("1");
  await expect(page.locator("#cell-details")).toContainText("Slot0");
  await seek(1, 5);
  await expect(page.locator("#cell-details")).toContainText("Slot1");
  assertStationary(await snapshot(), initial);
  await page.selectOption("#color-mode", "species");
  await expect(page.locator("#legend-title")).toHaveText("Reporter");
  // Delay an uncached frame and supersede it while its decode worker is occupied.
  await page.evaluate(() => {
    globalThis.__delayReplayLoad = (ordinal) =>
      ordinal === 4
        ? new Promise((resolve) => {
            globalThis.__releaseReplay = resolve;
          })
        : undefined;
  });
  await page.locator("#replay-timeline").fill("4");
  await page.locator("#replay-timeline").dispatchEvent("input");
  await expect
    .poll(() => page.evaluate(() => typeof globalThis.__releaseReplay))
    .toBe("function");
  await page.locator("#replay-timeline").fill("3");
  await page.locator("#replay-timeline").dispatchEvent("input");
  await page.evaluate(() => {
    globalThis.__releaseReplay();
    globalThis.__delayReplayLoad = undefined;
  });
  await expect(page.locator("#replay-position")).toContainText("4 / 5");
  await expect(page.locator("#time-chip")).toHaveText("t = 0.8");
  await expect(page.locator("#legend-title")).toHaveText("Reporter");
  await seek(0, 5);
  await page.locator("#replay-fps").fill("20");
  await page.locator("#replay-fps").dispatchEvent("change");
  await page.locator("#replay-play").click();
  await expect(page.locator("#replay-position")).toContainText("5 / 5");
  await expect(page.locator("#replay-play")).toHaveText("Play");
  await page.locator("#replay-previous").focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#replay-position")).toContainText("4 / 5");
  assertStationary(await snapshot(), initial);
  await page.screenshot({ path: `${evidence}/lifecycle-replay.png` });
  await open("grids", 4);
  async function fixed(root, kind, minimum, maximum) {
    await root.getByLabel(`${kind} range mode`).selectOption("fixed");
    await root
      .getByLabel(`${kind} minimum`, { exact: true })
      .fill(String(minimum));
    await root
      .getByLabel(`${kind} maximum`, { exact: true })
      .fill(String(maximum));
    await root
      .getByRole("button", { name: "Apply range", exact: true })
      .click();
  }
  await page.selectOption("#color-mode", "species");
  await fixed(page.locator("#species-range"), "Species", 0, 1);
  await page.selectOption("#species-channel", "1");
  await fixed(page.locator("#species-range"), "Species", 0, 2);
  await page.selectOption("#signal-channel", "1");
  await fixed(page.locator("#signal-color-range"), "Signal", 0, 2);
  await page.locator("#signal-slice").fill("2");
  await page.locator("#signal-slice").dispatchEvent("input");
  await page.locator("#signal-visible").focus();
  await page.keyboard.press("Space");
  await page.getByLabel("Show device geometry", { exact: true }).focus();
  await page.keyboard.press("Space");
  await page.selectOption("#color-mode", "composite");
  const blue = page.locator('.composite-channel[data-channel="1"]');
  await blue.locator("summary").click();
  await blue.getByLabel("Channel tint").fill("#0000ff");
  await blue.getByRole("button", { name: "Apply tint", exact: true }).click();
  await blue.getByRole("button", { name: "Move Red up", exact: true }).click();
  const colors = () =>
    page.evaluate(() =>
      Array.from(globalThis.__testViewer.cellMeshes[0].instanceColor.array),
    );
  const composite = await colors();
  assert.ok(Math.abs(composite[0] - 0.25) < 1e-5);
  assert.equal(composite[1], 0);
  assert.ok(Math.abs(composite[2] - 0.375) < 1e-5);
  await page.getByLabel("Enable Green", { exact: true }).uncheck();
  const retainedColors = await colors();
  assert.equal(retainedColors[0], 0);
  await page.evaluate(() => {
    const v = globalThis.__testViewer;
    v.selectCell(0);
    v.controls.enableDamping = false;
    v.camera.position.multiplyScalar(1.3);
    v.controls.update();
  });
  const gridInitial = await snapshot();
  await seek(1, 4);
  await expect(page.locator("#signal-section")).toBeHidden();
  await expect(page.locator("#device-visible")).toBeDisabled();
  await seek(2, 4);
  await expect(page.locator("#signal-slice")).toHaveValue("0");
  await seek(3, 4);
  await expect(page.locator("#signal-slice")).toHaveValue("2");
  await expect(page.locator("#signal-channel")).toHaveValue("1");
  await expect(page.locator("#species-channel")).toHaveValue("1");
  await expect(page.locator("#signal-visible")).not.toBeChecked();
  await expect(page.locator("#device-visible")).not.toBeChecked();
  await expect(page.locator("#device-visible")).toBeEnabled();
  await expect(page.locator("#selection-title")).toHaveText("Cell 1");
  await expect(page.locator("#composite-legend")).toContainText(
    "Red · #0000ff · Fixed 0 to 2",
  );
  await expect(
    page.getByLabel("Enable Green", { exact: true }),
  ).not.toBeChecked();
  assert.deepEqual(await colors(), retainedColors);
  assert.deepEqual(
    await page
      .locator(".composite-channel")
      .evaluateAll((rows) => rows.map((row) => row.dataset.channel)),
    ["1", "0"],
  );
  assert.equal(
    await page.evaluate(() => globalThis.__testViewer.device.visible),
    false,
  );
  await expect(
    page
      .locator("#signal-color-range")
      .getByLabel("Signal maximum", { exact: true }),
  ).toHaveValue("2");
  await seek(0, 4);
  assert.deepEqual(await colors(), retainedColors);
  assertStationary(await snapshot(), gridInitial);
  await page.selectOption("#color-mode", "species");
  await expect(
    page.locator("#species-range").getByLabel("Species range mode"),
  ).toHaveValue("fixed");
  await expect(
    page
      .locator("#species-range")
      .getByLabel("Species maximum", { exact: true }),
  ).toHaveValue("2");
  await page.selectOption("#color-mode", "composite");
  assert.deepEqual(await colors(), retainedColors);
  await page.screenshot({ path: `${evidence}/grid-replay.png` });
  await page.setViewportSize({ width: 880, height: 720 });
  for (const id of [
    "#replay-play",
    "#replay-timeline",
    "#replay-fps",
    "#recording-open",
  ]) {
    const bounds = await page.locator(id).boundingBox();
    assert.ok(
      bounds && bounds.x >= 0 && bounds.x + bounds.width <= 880,
      `${id} fits the supported narrow layout`,
    );
  }
  await expect(page.locator("#replay-position")).toBeVisible();
  await page.screenshot({ path: `${evidence}/narrow-replay.png` });
  await page.setViewportSize({ width: 1440, height: 960 });
  // A damaged frame is reported persistently while retaining the last good frame.
  const malformed = `${evidence}/malformed`;
  await cp(`${fixtures}/lifecycle`, malformed, {
    recursive: true,
    force: true,
  });
  const manifest = JSON.parse(
    await readFile(`${malformed}/manifest.json`, "utf8"),
  );
  const broken = Buffer.from("not a scene");
  const entry = manifest.recording.frames[2];
  entry.bytes = broken.byteLength;
  entry.sha256 = createHash("sha256").update(broken).digest("hex");
  await writeFile(`${malformed}/${entry.file}`, broken);
  manifest.integrity.recording = createHash("sha256")
    .update(canonicalize(manifest.recording))
    .digest("hex");
  await writeFile(`${malformed}/manifest.json`, JSON.stringify(manifest));
  await page.locator("#recording-folder").setInputFiles(malformed);
  await expect(page.locator("#replay-position")).toContainText("1 / 5");
  await page.locator("#replay-timeline").fill("2");
  await page.locator("#replay-timeline").dispatchEvent("input");
  await expect(page.locator("#replay-message")).toContainText(
    "frame 2 (frames/00000002.scene.json)",
  );
  await expect(page.locator("#replay-position")).toContainText("1 / 5");
  await seek(1, 5);
  // Opening a static scene closes replay; delayed recording work cannot replace it.
  await page
    .locator("#scene-file")
    .setInputFiles(`${fixtures}/grids/frames/00000000.scene.json`);
  await expect(page.locator("#replay-transport")).toBeHidden();
  await expect(page.locator("#color-mode")).toHaveValue("cell-type");
  await expect(page.locator("#species-channel")).toHaveValue("0");
  await expect(page.locator("#device-visible")).toBeChecked();
  await page.selectOption("#color-mode", "species");
  await expect(
    page.locator("#species-range").getByLabel("Species range mode"),
  ).toHaveValue("automatic");
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      status: "passed",
      coverage:
        "combined fixed species/signal ranges, composite tint/order/enabled state, device visibility, native exported topology, reverse steps, stable ID selection, time, rapid seeks, playback fps/end, metadata, missing/changing grids, camera/grid/preferences, malformed frame attribution/recovery, keyboard, narrow layout, new dataset reset",
    }),
  );
} finally {
  await browser.close();
}
