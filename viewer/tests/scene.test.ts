import canonicalize from "canonicalize";
import { describe, expect, it } from "vitest";

import { parseScene, SceneFormatError } from "../src/scene";

const PYTHON_SCENE = `{
  "format": "microsimulator-scene",
  "frame": {
    "backend": {
      "device": "host",
      "device_index": 0,
      "kind": "cpu",
      "name": "CPU reference",
      "native": false
    },
    "cells": [
      {
        "cell_type": -2,
        "direction": [1.0, 0.0, 0.0],
        "fixed": false,
        "growth_rate": 0.125,
        "id": "9223372036854775815",
        "length": 4.0,
        "parent_id": "1",
        "position": [1.0, 2.5, -3.0],
        "radius": 0.5,
        "slot": 0,
        "species": [7.25]
      }
    ],
    "signal_grid": null,
    "species_count": 1,
    "time": 1.0
  },
  "integrity": {
    "algorithm": "sha256",
    "frame": "a10da18274e64b7c33d19f9cf8e84560f200254e8b7931a753890c431bc5af04"
  },
  "producer": {"name": "microsimulator", "version": "0.1.0"},
  "version": 1
}`;

async function digest(value: unknown): Promise<string> {
  const encoded = canonicalize(value);
  if (encoded === undefined) {
    throw new Error("fixture is not canonicalizable");
  }
  const result = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(encoded),
  );
  return [...new Uint8Array(result)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

describe("scene reader", () => {
  it("verifies and reads a Python-authored RFC 8785 scene", async () => {
    const frame = await parseScene(PYTHON_SCENE);
    expect(frame.time).toBe(1);
    expect(frame.backend.kind).toBe("cpu");
    expect(frame.cells[0]?.id).toBe("9223372036854775815");
    expect(frame.cells[0]?.position).toEqual([1, 2.5, -3]);
    expect(frame.cells[0]?.species).toEqual([7.25]);
  });

  it("reads the previous scene format and still rejects tampering", async () => {
    const document = JSON.parse(PYTHON_SCENE) as {
      format: string;
      frame: { time: number };
    };
    document.format = "cellmodeller2-scene";
    expect(await parseScene(JSON.stringify(document))).toEqual(
      await parseScene(PYTHON_SCENE),
    );
    document.frame.time = 2;
    await expect(parseScene(JSON.stringify(document))).rejects.toThrow(
      "frame digest does not match",
    );
  });

  it("rejects a modified frame", async () => {
    const document = JSON.parse(PYTHON_SCENE) as { frame: { time: number } };
    document.frame.time = 2;
    await expect(parseScene(JSON.stringify(document))).rejects.toThrow(
      "frame digest does not match",
    );
  });

  it("rejects unknown fields even when the frame is correctly signed", async () => {
    const document = JSON.parse(PYTHON_SCENE) as {
      frame: { cells: Array<Record<string, unknown>> };
      integrity: { frame: string };
    };
    const cell = document.frame.cells[0];
    if (cell === undefined) {
      throw new Error("fixture cell is missing");
    }
    cell.color = [1, 0, 0];
    document.integrity.frame = await digest(document.frame);
    await expect(parseScene(JSON.stringify(document))).rejects.toThrow(
      /unknown keys.*color/,
    );
  });

  it("rejects invalid documents explicitly", async () => {
    await expect(parseScene("")).rejects.toBeInstanceOf(SceneFormatError);
    await expect(parseScene("[]")).rejects.toThrow("expected an object");
    await expect(parseScene("{")).rejects.toThrow("not valid JSON");
  });
});
