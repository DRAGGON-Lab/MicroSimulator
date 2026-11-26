import { describe, expect, it } from "vitest";
import { flowSlice, streamlines, vectorArrows } from "../src/flow-field";

import { flatSignalIndex, signalSlice, sliceDimension } from "../src/grid";
import type { SceneGridBoundary, SceneSignalGrid } from "../src/scene";

const boundary: SceneGridBoundary = { kind: "no_flux", values: [] };
const grid: SceneSignalGrid = {
  signalCount: 2,
  shape: [2, 3, 2],
  origin: [10, 20, 30],
  spacing: [0.5, 1, 2],
  boundaries: {
    xLower: boundary,
    xUpper: boundary,
    yLower: boundary,
    yUpper: boundary,
    zLower: boundary,
    zUpper: boundary,
  },
  levels: Array.from({ length: 24 }, (_, index) => index),
};

describe("signal grid slicing", () => {
  it("uses the engine's channel-major x-y-z indexing", () => {
    expect(flatSignalIndex(grid, 0, 1, 2, 1)).toBe(11);
    expect(flatSignalIndex(grid, 1, 1, 2, 1)).toBe(23);
  });

  it("extracts an XY slice with x varying across texture rows", () => {
    const slice = signalSlice(grid, 1, "z", 1);
    expect(slice.width).toBe(2);
    expect(slice.height).toBe(3);
    expect(slice.values).toEqual([13, 19, 15, 21, 17, 23]);
    expect(slice.center).toEqual([10.25, 21, 32]);
    expect(slice.horizontalSpan).toBe(0.5);
    expect(slice.verticalSpan).toBe(2);
  });

  it("reports dimensions and rejects invalid coordinates", () => {
    expect(sliceDimension(grid, "x")).toBe(2);
    expect(sliceDimension(grid, "y")).toBe(3);
    expect(sliceDimension(grid, "z")).toBe(2);
    expect(() => signalSlice(grid, 0, "x", 2)).toThrow("out of range");
    expect(() => flatSignalIndex(grid, 2, 0, 0, 0)).toThrow("out of range");
  });
});

describe("velocity overlays", () => {
  it("centers staggered components and projects onto the selected physical plane", () => {
    const shape = [2, 3, 2] as const;
    const [nx, ny, nz] = shape;
    // Linear face coordinates make wrong MAC strides or a one-face offset observable.
    const faces = [
      Float32Array.from({ length: (nx + 1) * ny * nz }, (_, i) =>
        Math.floor(i / (ny * nz)),
      ),
      Float32Array.from(
        { length: nx * (ny + 1) * nz },
        (_, i) => -Math.floor(i / nz) % (ny + 1),
      ),
      Float32Array.from(
        { length: nx * ny * (nz + 1) },
        (_, i) => 2 * (i % (nz + 1)),
      ),
    ] as const;
    const field = { ...grid, velocityFaces: faces };
    for (const [axis, peak] of [
      ["x", Math.hypot(2.5, 3)],
      ["y", Math.hypot(1.5, 3)],
      ["z", Math.hypot(1.5, 2.5)],
    ] as const) {
      const slice = flowSlice(field, axis, 1);
      expect(slice.peakSpeed).toBeCloseTo(peak);
      const normal = axis === "x" ? 0 : axis === "y" ? 1 : 2;
      expect(slice.center[normal]).toBe(
        grid.origin[normal] + grid.spacing[normal],
      );
    }
    const xy = flowSlice(field, "z", 0);
    expect(xy.velocity.slice(0, 2)).toEqual(new Float32Array([0.5, -0.5]));
    expect(xy.velocity.slice(-2)).toEqual(new Float32Array([1.5, -2.5]));
    expect(xy.change).toBeNull();
    const unchanged = flowSlice(field, "z", 0, faces);
    expect(unchanged.peakChange).toBe(0);
    expect(unchanged.changeScale).toBeCloseTo(xy.peakSpeed);

    const reversed = faces[0].map((value) => -value);
    const difference = flowSlice(field, "z", 0, [reversed, faces[1], faces[2]]);
    expect(difference.velocity).toEqual(xy.velocity);
    expect(difference.peakChange).toBe(3);
    expect(difference.change).toEqual(new Float32Array([1, 3, 1, 3, 1, 3]));
  });

  it("maps vector direction and speed to arrow displacement on anisotropic slices", () => {
    const field = flowSlice(
      {
        ...grid,
        velocityFaces: [
          new Float32Array(18),
          new Float32Array(16),
          new Float32Array(18),
        ],
      },
      "z",
      0,
    );
    expect(
      vectorArrows(field, { reference: field, scaleSpeed: 0, gain: 1 }),
    ).toEqual([]);
    field.velocity.set([3, 4, -1.5, -2]);
    field.peakSpeed = 5;
    const display = { reference: field, scaleSpeed: field.peakSpeed, gain: 1 };
    const arrows = vectorArrows(field, display);
    expect(arrows).toHaveLength(2);
    const fast = arrows[0]!,
      slow = arrows[1]!;
    expect(fast.x).toBe(-0.25);
    expect(slow.x).toBe(0.25);
    expect(fast.y).toBe(-1);
    expect(fast.v / fast.u).toBeCloseTo(4 / 3);
    expect(fast.u).toBeGreaterThan(0);
    expect(slow.u).toBeCloseTo(-fast.u / 2);
    expect(slow.v).toBeCloseTo(-fast.v / 2);
    expect(fast.tipOpacity).toBe(0);

    const slower = {
      ...field,
      peakSpeed: 2.5,
      velocity: field.velocity.map((v) => v / 2),
    };
    const changed = vectorArrows(slower, display)[0]!;
    expect(changed.u).toBeCloseTo(fast.u / 2);
    expect(changed.v).toBeCloseTo(fast.v / 2);
    expect(changed.u0).toBe(fast.u);
    expect(changed.v0).toBe(fast.v);
    expect(changed.tipOpacity).toBeGreaterThan(0);
    const enlarged = vectorArrows(slower, { ...display, gain: 4 })[0]!;
    expect(enlarged.u).toBeCloseTo(4 * changed.u);
    expect(enlarged.v0).toBeCloseTo(4 * changed.v0);
    expect(enlarged.x).toBe(changed.x);

    const stopped = {
      ...field,
      peakSpeed: 0,
      velocity: new Float32Array(field.velocity.length),
    };
    const tip = vectorArrows(stopped, display)[0]!;
    expect(tip.u).toBe(0);
    expect(tip.v).toBe(0);
    expect(tip.tipOpacity).toBeGreaterThan(0);
  });

  it("follows downstream in physical coordinates and terminates at a stopped strip", () => {
    const uniform = flowSlice(
      {
        ...grid,
        velocityFaces: [
          new Float32Array(18).fill(1),
          new Float32Array(16).fill(1),
          new Float32Array(18),
        ],
      },
      "z",
      0,
    );
    const field = {
      ...uniform,
      dx: 2,
      dy: 1,
      width: 8,
      height: 12,
      velocity: new Float32Array(2 * 8 * 12).fill(1),
    };
    for (let row = 0; row < 12; row++)
      field.velocity.fill(0, 2 * (row * 8 + 4), 2 * (row * 8 + 5));
    const strokes = streamlines(field);
    expect(strokes.length).toBeGreaterThan(0);
    expect(streamlines(field)).toEqual(strokes);
    for (const { points } of strokes) {
      for (let i = 3; i < points.length; i += 3) {
        const x = points[i]!,
          y = points[i + 1]!;
        expect(x).toBeGreaterThan(points[i - 3]!);
        expect(y - points[i - 2]!).toBeCloseTo(2 * (x - points[i - 3]!));
        expect(points[i + 2]).toBeGreaterThan(points[i - 1]!);
        expect(Math.round(x)).not.toBe(4);
        expect(x < 3.5).toBe(points[0]! < 3.5);
      }
    }
    expect(streamlines({ ...field, peakSpeed: 0 })).toEqual([]);
  });
});
