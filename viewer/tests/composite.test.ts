import { describe, expect, it } from "vitest";
import { mapCellColors } from "../src/color";
import {
  COMPOSITE_NEUTRAL,
  mapCompositeSpecies,
  type CompositeChannelConfig,
} from "../src/composite-color";
import { CompositeSpeciesState } from "../src/composite-state";
import { DatasetScalarRanges } from "../src/scalar-range";
import type { SceneFrame } from "../src/scene";

const frame: SceneFrame = {
  time: 0,
  backend: {
    kind: "cpu",
    name: "CPU",
    device: "host",
    deviceIndex: 0,
    native: true,
  },
  speciesCount: 2,
  channelMetadata: { species: ["Red reporter", "Green reporter"], signals: [] },
  cells: [
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],
  ].map((species, slot) => ({
    id: String(slot + 1),
    parentId: null,
    slot,
    position: [slot, 0, 0],
    direction: [1, 0, 0],
    length: 2,
    radius: 0.5,
    growthRate: 0,
    fixed: false,
    cellType: 0,
    species,
  })),
  constraints: { planes: [], spheres: [], boxes: [], cylinders: [] },
  signalGrid: null,
};
const channels: readonly CompositeChannelConfig[] = [
  {
    index: 0,
    enabled: true,
    tint: "#ff0000",
    range: { mode: "fixed", minimum: 0, maximum: 1 },
  },
  {
    index: 1,
    enabled: true,
    tint: "#00ff00",
    range: { mode: "fixed", minimum: 0, maximum: 1 },
  },
];

function expectColor(
  actual: readonly number[] | undefined,
  expected: readonly number[],
  tolerance = 1e-10,
): void {
  expect(actual).toHaveLength(3);
  expected.forEach((value, index) =>
    expect(Math.abs(actual![index]! - value)).toBeLessThanOrEqual(tolerance),
  );
}

describe("composite species blending", () => {
  it("distinguishes red, green, coexpression yellow and zero expression", () => {
    const mapping = mapCompositeSpecies(frame, channels);
    [
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 0],
    ].forEach((color, index) => expectColor(mapping.colors[index], color));
    expect(mapping.channels.map((channel) => channel.label)).toEqual([
      "Red reporter",
      "Green reporter",
    ]);
  });
  it("is exactly independent of channel order and disabling removes only that contribution", () => {
    const expected = mapCompositeSpecies(frame, channels);
    expect(mapCompositeSpecies(frame, [...channels].reverse()).colors).toEqual(
      expected.colors,
    );
    const disabled = mapCompositeSpecies(
      frame,
      channels.map((channel) => ({ ...channel, enabled: channel.index === 0 })),
    );
    expectColor(disabled.colors[2], [1, 0, 0]);
    expectColor(disabled.colors[1], [0, 0, 0]);
    expect(disabled.channels).toHaveLength(1);
  });
  it("weights and adds in linear RGB, then encodes once for the existing renderer boundary", () => {
    const half = {
      ...frame,
      cells: [{ ...frame.cells[0]!, species: [0.5, 0] }],
    };
    const color = mapCompositeSpecies(half, channels).colors[0];
    // Three.js encodes with its 0.41666 exponent approximation.
    expectColor(color, [0.7353569830524495, 0, 0], 1e-5);
    const decode = (value: number) =>
      value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    expect(Math.abs(decode(color![0]) - 0.5)).toBeLessThan(1e-5);
    const gray = mapCompositeSpecies(half, [
      { ...channels[0]!, tint: "#808080" },
    ]).colors[0]!;
    for (const value of gray)
      expect(Math.abs(decode(value) - 0.5 * decode(128 / 255))).toBeLessThan(
        1e-5,
      );
  });
  it("clamps summed components after addition and clips concentrations without mutating values", () => {
    const input = {
      ...frame,
      cells: [
        { ...frame.cells[0]!, species: [-10, 40] },
        { ...frame.cells[1]!, species: [40, 40] },
      ],
    };
    const mapping = mapCompositeSpecies(
      input,
      channels.map((channel) => ({ ...channel, tint: "#ff0000" })),
    );
    expectColor(mapping.colors[0], [1, 0, 0]);
    expectColor(mapping.colors[1], [1, 0, 0]);
    expect(input.cells.map((cell) => cell.species)).toEqual([
      [-10, 40],
      [40, 40],
    ]);
  });
  it("uses neutral gray when no channels are enabled and handles empty data", () => {
    const mapping = mapCompositeSpecies(
      frame,
      channels.map((channel) => ({ ...channel, enabled: false })),
    );
    expect(mapping.colors).toEqual(frame.cells.map(() => COMPOSITE_NEUTRAL));
    expect(mapping.channels).toEqual([]);
    expect(
      mapCompositeSpecies({ ...frame, cells: [] }, channels).colors,
    ).toEqual([]);
  });
  it("uses the shared automatic constant midpoint and validates channel identity/tints", () => {
    const constant = {
      ...frame,
      cells: [{ ...frame.cells[0]!, species: [7, 0] }],
    };
    expectColor(
      mapCompositeSpecies(constant, [
        { ...channels[0]!, range: { mode: "automatic" } },
      ]).colors[0],
      [0.7353569830524495, 0, 0],
      1e-5,
    );
    expect(() =>
      mapCompositeSpecies(frame, [channels[0]!, channels[0]!]),
    ).toThrow("duplicate");
    expect(() =>
      mapCompositeSpecies(frame, [{ ...channels[0]!, index: 20 }]),
    ).toThrow("out of range");
    expect(() =>
      mapCompositeSpecies(frame, [{ ...channels[0]!, tint: "red" }]),
    ).toThrow("six-digit");
  });
  it("integrates with existing color modes without changing cell identity", () => {
    const mapped = mapCellColors(frame, {
      mode: "composite",
      speciesIndex: 0,
      compositeChannels: channels,
    });
    expect(mapped.colors).toEqual(mapCompositeSpecies(frame, channels).colors);
    expect(mapped.composite).toHaveLength(2);
    expect(frame.cells.map((cell) => cell.id)).toEqual(["1", "2", "3", "4"]);
    expect(
      mapCellColors(frame, { mode: "species", speciesIndex: 0 }).title,
    ).toBe("Red reporter");
  });
});

describe("composite channel settings", () => {
  it("preserves tint/range/enabled identity through reordering and temporarily missing channels", () => {
    const state = new CompositeSpeciesState();
    const ranges = new DatasetScalarRanges();
    ranges.set("species", 0, { mode: "fixed", minimum: -2, maximum: 10 });
    ranges.set("species", 1, { mode: "fixed", minimum: 1, maximum: 8 });
    state.setTint(1, "#ABCDEF");
    state.setEnabled(1, false);
    state.move(1, -1, 2);
    expect(state.indices(2)).toEqual([1, 0]);
    expect(state.indices(0)).toEqual([]);
    expect(state.configuration(2, ranges)).toEqual([
      {
        index: 1,
        tint: "#abcdef",
        enabled: false,
        range: { mode: "fixed", minimum: 1, maximum: 8 },
      },
      {
        index: 0,
        tint: "#ff0000",
        enabled: true,
        range: { mode: "fixed", minimum: -2, maximum: 10 },
      },
    ]);
    state.setEnabled(1, true);
    expect(state.configuration(2, ranges)[0]?.range).toMatchObject({
      minimum: 1,
    });
    expect(() => state.setTint(1, "invalid")).toThrow();
    expect(state.get(1).tint).toBe("#abcdef");
    state.beginDataset();
    ranges.beginDataset();
    expect(state.indices(2)).toEqual([0, 1]);
    expect(state.get(1)).toEqual({ enabled: true, tint: "#00ff00" });
    expect(state.configuration(2, ranges)[0]?.range).toEqual({
      mode: "automatic",
    });
  });
});
