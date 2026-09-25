import { describe, expect, it } from "vitest";
import { mapScalarColors, viridis } from "../src/color";
import {
  DatasetScalarRanges,
  normalizeScalar,
  parseFixedScalarRange,
  resolveScalarRange,
  suggestedFixedRange,
} from "../src/scalar-range";

const fixed = { mode: "fixed" as const, minimum: 0, maximum: 10 };

describe("scalar normalization", () => {
  it("keeps shared values identical across different frame and slice extrema", () => {
    const first = mapScalarColors([2, 4], fixed);
    const second = mapScalarColors([2, 40], fixed);
    expect(first.colors[0]).toEqual(second.colors[0]);
    expect(first.colors[0]).toEqual(viridis(0.2));
    expect(second.colors[1]).toEqual(viridis(1));
  });
  it("clips display intensity without modifying negative/out-of-range data", () => {
    const values = Object.freeze([-30, -10, 0, 10, 300]);
    expect(
      mapScalarColors(values, { mode: "fixed", minimum: -10, maximum: 10 })
        .colors,
    ).toEqual([viridis(0), viridis(0), viridis(0.5), viridis(1), viridis(1)]);
    expect(values).toEqual([-30, -10, 0, 10, 300]);
  });
  it("represents automatic empty data without invented extrema", () => {
    expect(mapScalarColors([])).toEqual({
      colors: [],
      range: { mode: "automatic", minimum: null, maximum: null, count: 0 },
    });
    expect(mapScalarColors([], fixed)).toEqual({
      colors: [],
      range: { ...fixed, count: 0 },
    });
  });
  it("uses the midpoint for constant automatic data and fixed bounds for constant fixed data", () => {
    expect(mapScalarColors([7, 7]).colors).toEqual([
      viridis(0.5),
      viridis(0.5),
    ]);
    expect(mapScalarColors([7, 7], fixed).colors).toEqual([
      viridis(0.7),
      viridis(0.7),
    ]);
    expect(resolveScalarRange([7, 7])).toMatchObject({
      minimum: 7,
      maximum: 7,
    });
  });
  it("rejects non-finite source values rather than generating NaN colors", () => {
    for (const invalid of [NaN, Infinity, -Infinity]) {
      expect(() => mapScalarColors([invalid])).toThrow("must be finite");
      expect(() =>
        normalizeScalar(invalid, resolveScalarRange([0, 1])),
      ).toThrow("must be finite");
    }
  });
  it("handles extreme finite bounds without overflowing normalization", () => {
    const range = resolveScalarRange([-Number.MAX_VALUE, 0, Number.MAX_VALUE], {
      mode: "fixed",
      minimum: -Number.MAX_VALUE,
      maximum: Number.MAX_VALUE,
    });
    expect(normalizeScalar(0, range)).toBe(0.5);
    expect(normalizeScalar(-Number.MAX_VALUE / 2, range)).toBe(0.25);
    expect(normalizeScalar(Number.MAX_VALUE / 2, range)).toBe(0.75);
    const tiny = resolveScalarRange([0], {
      mode: "fixed",
      minimum: 0,
      maximum: Number.MIN_VALUE,
    });
    expect(normalizeScalar(Number.MIN_VALUE, tiny)).toBe(1);
  });
  it("suggests valid bounds for empty, constant, and extreme automatic ranges", () => {
    for (const values of [
      [],
      [0],
      [7],
      [-7],
      [Number.MAX_VALUE],
      [-Number.MAX_VALUE],
      [2, 40],
    ]) {
      const suggestion = suggestedFixedRange(resolveScalarRange(values));
      expect(Number.isFinite(suggestion.minimum)).toBe(true);
      expect(Number.isFinite(suggestion.maximum)).toBe(true);
      expect(suggestion.minimum).toBeLessThan(suggestion.maximum);
    }
  });
});

describe("fixed range validation and channel state", () => {
  it("accepts negative finite numbers and scientific notation", () => {
    expect(parseFixedScalarRange(" -1e2 ", ".5")).toEqual({
      mode: "fixed",
      minimum: -100,
      maximum: 0.5,
    });
  });
  it("rejects missing, nonnumeric, infinite, equal and reversed bounds", () => {
    for (const [minimum, maximum] of [
      ["", "1"],
      ["0", "NaN"],
      ["0", "Infinity"],
      ["0", "1e309"],
      ["0x10", "20"],
      ["2", "2"],
      ["3", "2"],
    ]) {
      expect(() => parseFixedScalarRange(minimum!, maximum!)).toThrow();
    }
  });
  it("retains the last valid configuration after invalid changes", () => {
    const state = new DatasetScalarRanges();
    state.set("species", 0, fixed);
    for (const bounds of [
      { minimum: 2, maximum: 2 },
      { minimum: 3, maximum: 2 },
      { minimum: NaN, maximum: 2 },
      { minimum: 0, maximum: Infinity },
    ]) {
      expect(() =>
        state.set("species", 0, { mode: "fixed", ...bounds }),
      ).toThrow();
      expect(state.get("species", 0)).toEqual(fixed);
    }
  });
  it("separates channel kind/index, remembers fixed bounds, and resets only on a new dataset", () => {
    const state = new DatasetScalarRanges();
    state.set("species", 0, fixed);
    state.set("species", 1, { mode: "fixed", minimum: -1, maximum: 1 });
    state.set("signals", 0, { mode: "fixed", minimum: 10, maximum: 20 });
    expect(state.get("species", 0)).toEqual(fixed);
    expect(state.get("species", 2)).toEqual({ mode: "automatic" });
    state.set("species", 0, { mode: "automatic" });
    expect(state.lastFixed("species", 0)).toEqual(fixed);
    expect(state.get("species", 1)).toMatchObject({ minimum: -1 });
    expect(state.get("signals", 0)).toMatchObject({ minimum: 10 });
    state.beginDataset();
    expect(state.get("species", 0)).toEqual({ mode: "automatic" });
    expect(state.lastFixed("species", 0)).toBeNull();
    expect(state.get("signals", 0)).toEqual({ mode: "automatic" });
  });
});
