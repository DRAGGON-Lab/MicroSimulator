import canonicalize from "canonicalize";
import { describe, expect, it } from "vitest";

import { MAX_SCENE_CHANNELS, parseScene, SceneFormatError } from "../src/scene";

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
        "direction": [
          1.0,
          0.0,
          0.0
        ],
        "fixed": false,
        "growth_rate": 0.125,
        "id": "9223372036854775815",
        "length": 4.0,
        "parent_id": "1",
        "position": [
          1.0,
          2.5,
          -3.0
        ],
        "radius": 0.5,
        "slot": 0,
        "species": [
          7.25
        ]
      }
    ],
    "constraints": {
      "boxes": [
        {
          "allowed_region": "outside",
          "center": [
            4.0,
            -1.0,
            0.5
          ],
          "coefficient": 0.75,
          "half_extents": [
            1.5,
            0.5,
            2.0
          ],
          "id": "3"
        }
      ],
      "cylinders": [
        {
          "allowed_region": "inside",
          "center": [
            0.0,
            0.0,
            1.0
          ],
          "coefficient": 1.0,
          "half_height": 2.0,
          "id": "4",
          "radius": 6.0
        }
      ],
      "planes": [
        {
          "coefficient": 1.25,
          "id": "2",
          "inward_normal": [
            0.0,
            0.0,
            1.0
          ],
          "point": [
            0.0,
            0.0,
            -1.0
          ]
        }
      ],
      "spheres": []
    },
    "signal_grid": null,
    "species_count": 1,
    "time": 1.0
  },
  "integrity": {
    "algorithm": "sha256",
    "frame": "1f64bdd2bd5e230f094922c4c4701a3919972a8f26ae462aa9b253709e056584"
  },
  "producer": {
    "name": "microsimulator",
    "version": "0.1.0"
  },
  "version": 2
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

async function channelBudgetScene(
  version: number,
  speciesCount: number,
  signalCount: number,
  complete: boolean,
): Promise<string> {
  const boundary = { kind: "no_flux", values: [] };
  const frame = {
    time: 0,
    backend: {
      kind: "cpu",
      name: "CPU reference",
      device: "host",
      device_index: 0,
      native: false,
    },
    species_count: speciesCount,
    cells: [],
    constraints: { planes: [], spheres: [], boxes: [], cylinders: [] },
    signal_grid: {
      signal_count: signalCount,
      shape: [1, 1, 1],
      origin: [0, 0, 0],
      spacing: [1, 1, 1],
      boundaries: {
        x_lower: boundary,
        x_upper: boundary,
        y_lower: boundary,
        y_upper: boundary,
        z_lower: boundary,
        z_upper: boundary,
      },
      levels: Array<number>(complete ? signalCount : 1).fill(0),
    },
    ...(version === 3
      ? {
          channel_metadata: {
            species: Array<null>(complete ? speciesCount : 0).fill(null),
            signals: Array<null>(complete ? signalCount : 0).fill(null),
          },
        }
      : {}),
  };
  return JSON.stringify({
    format: "microsimulator-scene",
    version,
    producer: { name: "microsimulator", version: "0.1.0" },
    frame,
    integrity: { algorithm: "sha256", frame: await digest(frame) },
  });
}

describe("scene reader", () => {
  it.each([2, 3])(
    "accepts both channel groups at the inclusive scene budget in v%i",
    async (version) => {
      const frame = await parseScene(
        await channelBudgetScene(
          version,
          MAX_SCENE_CHANNELS,
          MAX_SCENE_CHANNELS,
          true,
        ),
      );
      expect(frame.cells).toEqual([]);
      expect(frame.channelMetadata.species).toEqual(
        Array<null>(MAX_SCENE_CHANNELS).fill(null),
      );
      expect(frame.channelMetadata.signals).toEqual(
        Array<null>(MAX_SCENE_CHANNELS).fill(null),
      );
      expect(frame.signalGrid?.levels).toHaveLength(MAX_SCENE_CHANNELS);
    },
  );

  it.each([2, 3])(
    "rejects tiny signed oversized channel claims before allocating v%i labels",
    async (version) => {
      for (const count of [MAX_SCENE_CHANNELS + 1, 2 ** 32 - 1]) {
        for (const group of ["species", "signals"] as const) {
          const encoded = await channelBudgetScene(
            version,
            group === "species" ? count : 0,
            group === "signals" ? count : 1,
            false,
          );
          expect(encoded.length).toBeLessThan(2000);
          await expect(parseScene(encoded)).rejects.toThrow(
            `${group === "species" ? "$.frame.species_count" : "$.frame.signal_grid.signal_count"}: exceeds scene presentation channel budget of 4096 per group`,
          );
        }
      }
    },
  );

  it("verifies and reads a Python-authored RFC 8785 scene", async () => {
    const frame = await parseScene(PYTHON_SCENE);
    expect(frame.time).toBe(1);
    expect(frame.backend.kind).toBe("cpu");
    expect(frame.cells[0]?.id).toBe("9223372036854775815");
    expect(frame.cells[0]?.position).toEqual([1, 2.5, -3]);
    expect(frame.cells[0]?.species).toEqual([7.25]);
    expect(frame.constraints.planes[0]?.id).toBe("2");
    expect(frame.constraints.planes[0]?.inwardNormal).toEqual([0, 0, 1]);
    expect(frame.constraints.boxes[0]?.id).toBe("3");
    expect(frame.constraints.boxes[0]?.halfExtents).toEqual([1.5, 0.5, 2]);
    expect(frame.constraints.boxes[0]?.allowedRegion).toBe("outside");
    expect(frame.constraints.cylinders[0]?.id).toBe("4");
    expect(frame.constraints.cylinders[0]?.radius).toBe(6);
    expect(frame.constraints.cylinders[0]?.halfHeight).toBe(2);
    expect(frame.constraints.cylinders[0]?.allowedRegion).toBe("inside");
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
