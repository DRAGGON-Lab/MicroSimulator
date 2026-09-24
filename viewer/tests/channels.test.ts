import canonicalize from "canonicalize";
import { describe, expect, it } from "vitest";

import { mapCellColors } from "../src/color";
import { channelLabel, parseScene } from "../src/scene";
import pythonScene from "./fixtures/channels-v3.scene.json?raw";

async function sign(frame: unknown, version = 3): Promise<string> {
  const bytes = new TextEncoder().encode(canonicalize(frame));
  const digest = [
    ...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
  ]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
  return JSON.stringify({
    format: "microsimulator-scene",
    version,
    producer: { name: "test", version: "1" },
    integrity: { algorithm: "sha256", frame: digest },
    frame,
  });
}

interface Document {
  version: number;
  frame: {
    time: number;
    channel_metadata?: { species: unknown[]; signals: unknown[] };
  };
}

const document = (): Document => JSON.parse(pythonScene) as Document;

describe("channel metadata", () => {
  it("reads the Python-produced v3 fixture with Unicode, markup and missing labels", async () => {
    const frame = await parseScene(pythonScene);
    expect(frame.channelMetadata).toEqual({
      species: ["<b>α 🧪</b>", "<b>α 🧪</b>"],
      signals: [null, " "],
    });
    expect(channelLabel(frame, "species", 0)).toBe("<b>α 🧪</b> [0]");
    expect(channelLabel(frame, "species", 1)).toBe("<b>α 🧪</b> [1]");
    expect(channelLabel(frame, "signals", 0)).toBe("Channel 0");
    expect(channelLabel(frame, "signals", 1)).toBe("Channel 1");
    expect(() => channelLabel(frame, "species", 2)).toThrow(RangeError);
  });

  it("keeps numerical values/color selection independent of label text", async () => {
    const frame = await parseScene(pythonScene);
    const config = { mode: "species" as const, speciesIndex: 1 };
    const renamed = {
      ...frame,
      channelMetadata: {
        species: ["Changed", "Chosen"],
        signals: [null, null],
      },
    };
    const before = mapCellColors(frame, config);
    const after = mapCellColors(renamed, config);
    expect(before.colors).toEqual(after.colors);
    expect(after.title).toBe("Chosen");
    expect(renamed.cells).toEqual(frame.cells);
  });

  it("disambiguates explicit names that collide with missing-label fallbacks", async () => {
    const frame = await parseScene(pythonScene);
    const labels = {
      ...frame,
      channelMetadata: { species: [null, "Channel 0"], signals: [] },
    };
    expect(channelLabel(labels, "species", 0)).toBe("Channel 0 [0]");
    expect(channelLabel(labels, "species", 1)).toBe("Channel 0 [1]");
  });

  it("authenticates v2 without inserting metadata before digest verification", async () => {
    const old = document();
    delete old.frame.channel_metadata;
    const encoded = await sign(old.frame, 2);
    const frame = await parseScene(encoded);
    expect(frame.channelMetadata).toEqual({
      species: [null, null],
      signals: [null, null],
    });
    expect(channelLabel(frame, "species", 1)).toBe("Channel 1");
    const tampered = JSON.parse(encoded) as Document;
    tampered.frame.time += 1;
    await expect(parseScene(JSON.stringify(tampered))).rejects.toThrow(
      "frame digest does not match",
    );
  });

  it("rejects label tampering even when channel counts are unchanged", async () => {
    const tampered = document();
    tampered.frame.channel_metadata!.species[0] = "forged";
    await expect(parseScene(JSON.stringify(tampered))).rejects.toThrow(
      "frame digest does not match",
    );
  });

  it("rejects invalid Unicode before integrity verification", async () => {
    const invalid = document();
    invalid.frame.channel_metadata!.species[0] = "\ud800";
    await expect(parseScene(JSON.stringify(invalid))).rejects.toThrow(
      "Lone surrogate",
    );
  });

  it.each([
    { species: [null], signals: [null, null] },
    { species: [null, 3], signals: [null, null] },
    { species: [null, null], signals: [null, null], extra: true },
  ])(
    "rejects malformed metadata after valid integrity verification: %j",
    async (metadata) => {
      const invalid = document();
      invalid.frame.channel_metadata = metadata;
      await expect(parseScene(await sign(invalid.frame))).rejects.toThrow();
    },
  );
});
