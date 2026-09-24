// Real Python server + browser, including process exit and same-port restart.
// Run from the repository root after building this worktree's Python/viewer.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { createInterface } from "node:readline";

const { chromium, expect } = await import(
  process.env.MICROSIMULATOR_PLAYWRIGHT_MODULE ?? "@playwright/test"
);
const evidence =
  process.env.EVIDENCE_DIR ?? "/tmp/microsimulator-live-shutdown";
const python =
  process.env.MICROSIMULATOR_PYTHON ?? path.resolve(".venv/bin/python");
const port = process.env.VIEWER_PORT ?? "4327";
const temporary = await mkdtemp(path.join(tmpdir(), "microsimulator-stop-"));
await mkdir(evidence, { recursive: true });
await writeFile(
  path.join(temporary, "model.py"),
  `
import time
from microsimulator import CellInit
class Model:
    def __init__(self, context):
        self.simulation = context.simulation()
        self.simulation.add_cell(CellInit())
    def step(self, dt):
        time.sleep(0.02)
        self.simulation.step(dt)
    def controller_state(self):
        return {"kind": "shutdown-browser-fixture"}
def build(context):
    return Model(context)
`,
);

const browser = await chromium.launch({ headless: true });
let processUnderTest;
const errors = [];
const results = [];
try {
  for (const mode of ["paused", "playing", "reconnect", "interrupt"]) {
    if (mode === "interrupt" && process.platform === "win32") continue;
    const child = spawn(
      python,
      [
        "-m",
        "microsimulator",
        "view",
        "--model",
        path.join(temporary, "model.py"),
        "--backend",
        "cpu",
        "--dt",
        "0.01",
        "--frame-steps",
        "10000",
        "--viewer-dist",
        path.resolve("viewer/dist"),
        "--port",
        port,
        "--checkpoint-output",
        path.join(temporary, "checkpoint.json"),
      ],
      { stdio: ["ignore", "pipe", "pipe"] },
    );
    processUnderTest = child;
    let stderr = "";
    child.stderr.on("data", (data) => {
      stderr += data.toString();
    });
    const exit = new Promise((resolve) =>
      child.once("exit", (code, signal) => resolve({ code, signal })),
    );
    const lines = createInterface({ input: child.stdout });
    const url = await new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error("server startup timed out")),
        10000,
      );
      lines.on("line", (line) => {
        if (line.startsWith("MicroSimulator live viewer: ")) {
          clearTimeout(timer);
          resolve(line.slice("MicroSimulator live viewer: ".length));
        }
      });
      child.once("exit", () => {
        clearTimeout(timer);
        reject(new Error(stderr));
      });
      child.once("error", reject);
    });
    let page = await browser.newPage({
      viewport: { width: 1440, height: 960 },
    });
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(url);
    await expect(page.locator("#live-label")).toHaveText("Paused");
    await expect(page.locator("#live-stop")).toBeEnabled();
    if (mode === "reconnect") {
      await page.close();
      // The existing viewer deliberately requires a minimum width of 880px.
      page = await browser.newPage({ viewport: { width: 880, height: 844 } });
      page.on("pageerror", (error) => errors.push(error.message));
      await page.goto(url);
      await expect(page.locator("#live-label")).toHaveText("Paused");
      assert.equal(child.exitCode, null);
      await page.locator("#live-checkpoint").click();
      await expect(page.locator("#status")).toContainText("Checkpoint saved");
      assert.ok(
        (await readFile(path.join(temporary, "checkpoint.json"))).length > 100,
      );
    }
    if (mode === "playing") {
      await page.locator("#live-play").click();
      await expect(page.locator("#live-label")).toHaveText("Running");
    }
    const start = performance.now();
    if (mode === "interrupt") {
      child.kill("SIGINT");
    } else {
      await page.locator("#live-stop").focus();
      await page.keyboard.press("Enter");
    }
    await expect(page.locator("#live-label")).toHaveText("Stopped", {
      timeout: 5000,
    });
    const toolbar = await page.locator("#live-transport").boundingBox();
    const viewport = await page.locator("#canvas-host").boundingBox();
    assert.ok(toolbar.x >= viewport.x);
    assert.ok(toolbar.x + toolbar.width <= viewport.x + viewport.width);
    await expect(page.locator("#status")).toContainText("Session stopped");
    for (const id of ["play", "step", "reset", "checkpoint", "stop"]) {
      await expect(page.locator(`#live-${id}`)).toBeDisabled();
    }
    const stopped = await Promise.race([
      exit,
      new Promise((_, reject) => {
        const timer = setTimeout(
          () => reject(new Error("server did not exit")),
          5000,
        );
        timer.unref();
      }),
    ]);
    assert.deepEqual(stopped, { code: 0, signal: null });
    assert.equal(stderr, "");
    await page.screenshot({ path: `${evidence}/${mode}-stopped.png` });
    results.push({
      mode,
      stopMilliseconds: performance.now() - start,
      exitCode: stopped.code,
    });
    await page.close();
    lines.close();
    processUnderTest = undefined;
  }
  assert.deepEqual(errors, []);
  const result = {
    result: "passed",
    browser: browser.version(),
    platform: process.platform,
    port,
    results,
  };
  await writeFile(`${evidence}/results.json`, JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
} finally {
  processUnderTest?.kill("SIGKILL");
  await browser.close();
  await rm(temporary, { recursive: true, force: true });
}
