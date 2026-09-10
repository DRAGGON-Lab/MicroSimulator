import canonicalize from "canonicalize";
import { describe, expect, it } from "vitest";

import { parseLiveMessage } from "../src/live";

const FRAME = {
  backend: {
    device: "host",
    device_index: 0,
    kind: "cpu",
    name: "CPU reference",
    native: false,
  },
  cells: [],
  signal_grid: null,
  species_count: 0,
  time: 0,
};

async function scene(): Promise<Record<string, unknown>> {
  const canonical = canonicalize(FRAME);
  if (canonical === undefined) {
    throw new Error("fixture is not canonicalizable");
  }
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(canonical),
  );
  return {
    format: "microsimulator-scene",
    version: 1,
    producer: { name: "microsimulator", version: "0.1.0" },
    integrity: {
      algorithm: "sha256",
      frame: [...new Uint8Array(digest)]
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join(""),
    },
    frame: FRAME,
  };
}

describe("live viewer protocol", () => {
  it("verifies embedded scene documents", async () => {
    const message = await parseLiveMessage(
      JSON.stringify({
        type: "frame",
        revision: 2,
        completed_steps: 8,
        playing: true,
        checkpoint_enabled: false,
        scene: await scene(),
      }),
    );
    expect(message.type).toBe("frame");
    if (message.type === "frame") {
      expect(message.revision).toBe(2);
      expect(message.completedSteps).toBe(8);
      expect(message.frame.backend.kind).toBe("cpu");
    }
  });

  it("rejects unknown fields and malformed scene data", async () => {
    await expect(
      parseLiveMessage(
        JSON.stringify({ type: "error", message: "reason", source: "hidden" }),
      ),
    ).rejects.toThrow(/unknown keys.*source/);

    const document = await scene();
    (document.frame as { time: number }).time = 1;
    await expect(
      parseLiveMessage(
        JSON.stringify({
          type: "frame",
          revision: 0,
          completed_steps: 0,
          playing: false,
          checkpoint_enabled: false,
          scene: document,
        }),
      ),
    ).rejects.toThrow("frame digest does not match");
  });

  it("reads checkpoint and error results", async () => {
    await expect(
      parseLiveMessage('{"type":"checkpoint","path":"/tmp/run.cm2.json"}'),
    ).resolves.toEqual({ type: "checkpoint", path: "/tmp/run.cm2.json" });
    await expect(
      parseLiveMessage('{"type":"error","message":"not configured"}'),
    ).resolves.toEqual({ type: "error", message: "not configured" });
  });
});
