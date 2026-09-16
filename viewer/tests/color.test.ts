import { describe, expect, it } from "vitest";

import { mapCellColors, viridis } from "../src/color";
import type { SceneCell, SceneFrame } from "../src/scene";

function cell(
  slot: number,
  cellType: number,
  growthRate: number,
  species: readonly number[],
): SceneCell {
  return {
    id: String(slot + 1),
    parentId: null,
    slot,
    position: [slot, 0, 0],
    direction: [1, 0, 0],
    length: 2,
    radius: 0.5,
    growthRate,
    cellType,
    fixed: slot === 1,
    species,
  };
}

const frame: SceneFrame = {
  time: 0,
  backend: {
    kind: "cpu",
    name: "CPU",
    device: "host",
    deviceIndex: 0,
    native: false,
  },
  speciesCount: 2,
  constraints: { planes: [], spheres: [], boxes: [], cylinders: [] },
  cells: [
    cell(0, -1, 0.5, [2, 8]),
    cell(1, 2, 1.5, [4, 6]),
    cell(2, -1, 1, [3, 7]),
  ],
  signalGrid: null,
};

describe("cell color mappings", () => {
  it("assigns stable categorical colors for signed cell types", () => {
    const mapping = mapCellColors(frame, {
      mode: "cell-type",
      speciesIndex: 0,
    });
    expect(mapping.colors[0]).toEqual(mapping.colors[2]);
    expect(mapping.colors[0]).not.toEqual(mapping.colors[1]);
    expect(mapping.minimum).toBeNull();
  });

  it("normalizes scalar channels to the full perceptual ramp", () => {
    const mapping = mapCellColors(frame, { mode: "species", speciesIndex: 0 });
    expect(mapping.minimum).toBe(2);
    expect(mapping.maximum).toBe(4);
    expect(mapping.colors[0]).toEqual(viridis(0));
    expect(mapping.colors[1]).toEqual(viridis(1));
    expect(mapping.colors[2]).toEqual(viridis(0.5));
  });

  it("rejects an unavailable species channel", () => {
    expect(() =>
      mapCellColors(frame, { mode: "species", speciesIndex: 2 }),
    ).toThrow("out of range");
  });
});
