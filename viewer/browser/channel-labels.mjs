// Uses the shared external Playwright harness; no application debug API is shipped.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile } from "node:fs/promises";
import canonicalize from "canonicalize";

const { chromium, expect } = await import(
  process.env.MICROSIMULATOR_PLAYWRIGHT_MODULE ?? "@playwright/test"
);
const url = process.env.VIEWER_URL ?? "http://127.0.0.1:4318";
const evidence =
  process.env.EVIDENCE_DIR ?? "/tmp/microsimulator-channel-labels";
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
  let socket;
  await page.routeWebSocket("**/api/v1/session?*", (connection) => {
    socket = connection;
  });
  await page.goto(`${url}/?token=fixture`);
  await expect.poll(() => socket !== undefined).toBe(true);
  const document = JSON.parse(
    await readFile(
      new URL("../tests/fixtures/channels-v3.scene.json", import.meta.url),
      "utf8",
    ),
  );
  let revision = 0;
  async function send() {
    document.integrity.frame = createHash("sha256")
      .update(canonicalize(document.frame))
      .digest("hex");
    socket.send(
      JSON.stringify({
        type: "frame",
        revision: revision++,
        completed_steps: revision,
        playing: false,
        checkpoint_enabled: false,
        scene: document,
      }),
    );
  }
  await send();
  await expect(page.locator("#species-channel option")).toHaveText([
    "<b>α 🧪</b> [0]",
    "<b>α 🧪</b> [1]",
  ]);
  await expect(page.locator("#signal-channel option")).toHaveText([
    "Channel 0",
    "Channel 1",
  ]);
  await page.selectOption("#color-mode", "species");
  await page.selectOption("#species-channel", "1");
  await page.selectOption("#signal-channel", "1");
  await expect(page.locator("#legend-title")).toHaveText("<b>α 🧪</b> [1]");
  await page.evaluate(() => globalThis.__testViewer.selectCell(0));
  await expect(page.locator("#species-values li span")).toHaveText([
    "<b>α 🧪</b> [0]",
    "<b>α 🧪</b> [1]",
  ]);
  await expect(
    page.locator("#species-values b, #species-channel b, #legend-title b"),
  ).toHaveCount(0);
  document.frame.channel_metadata = {
    species: ["Green reporter", "Red reporter"],
    signals: ["Nutrient", "Extracellular cue"],
  };
  document.frame.time = 1;
  await send();
  await expect(page.locator("#species-channel")).toHaveValue("1");
  await expect(page.locator("#signal-channel")).toHaveValue("1");
  await expect(page.locator("#species-channel option")).toHaveText([
    "Green reporter",
    "Red reporter",
  ]);
  await expect(page.locator("#signal-channel option")).toHaveText([
    "Nutrient",
    "Extracellular cue",
  ]);
  await expect(page.locator("#legend-title")).toHaveText("Red reporter");
  await expect(page.locator("#species-values li span")).toHaveText([
    "Green reporter",
    "Red reporter",
  ]);
  document.frame.time = 0;
  await send();
  await expect(page.locator("#time-chip")).toHaveText("t = 0");
  await expect(page.locator("#species-channel")).toHaveValue("1");
  // Literal labels may imitate automatically generated duplicate suffixes.
  document.frame.species_count = 3;
  for (const cell of document.frame.cells) cell.species.push(0.25);
  document.frame.channel_metadata.species = ["GFP", "GFP", "GFP [0]"];
  await send();
  const indexed = ["GFP [0]", "GFP [1]", "GFP [0] [2]"];
  await expect(page.locator("#species-channel option")).toHaveText(indexed);
  await expect(page.locator("#species-values li span")).toHaveText(indexed);
  await expect(page.locator("#species-channel")).toHaveValue("1");
  await page.selectOption("#species-channel", "2");
  await expect(page.locator("#legend-title")).toHaveText("GFP [0] [2]");
  await page.screenshot({ path: `${evidence}/named-channels.png` });
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      status: "passed",
      assertions:
        "selector names, duplicate and generated-suffix disambiguation, HTML text safety, Unicode, inspector, legend, index-stable renamed frames, reset time",
      screenshot: `${evidence}/named-channels.png`,
    }),
  );
} finally {
  await browser.close();
}
