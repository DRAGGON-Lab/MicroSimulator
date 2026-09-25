import { describe, expect, it } from "vitest";

import {
  DatasetReferenceGrid,
  initialReferenceGrid,
} from "../src/reference-grid";
import type { SceneCell, SceneConstraints, SceneFrame } from "../src/scene";

const constraints: SceneConstraints = {
  boxes: [],
  spheres: [],
  cylinders: [],
  planes: [],
};
const cell: SceneCell = {
  id: "1",
  parentId: null,
  slot: 0,
  position: [3, 4, 2],
  direction: [1, 0, 0],
  length: 4,
  radius: 0.5,
  growthRate: 0,
  cellType: 0,
  fixed: false,
  species: [],
};
const frame: SceneFrame = {
  time: 0,
  backend: {
    kind: "cpu",
    name: "CPU",
    device: "host",
    deviceIndex: 0,
    native: true,
  },
  speciesCount: 0,
  cells: [cell],
  constraints,
  signalGrid: null,
};

describe("reference grid initialization", () => {
  it("uses initial colony bounds with a minimum ten-unit extent", () => {
    expect(initialReferenceGrid(frame)).toEqual({
      extent: 10,
      spacing: 0.5,
      position: [3, 4, -0.01],
    });
    expect(
      initialReferenceGrid({ ...frame, cells: [{ ...cell, length: 40 }] }),
    ).toEqual({ extent: 41, spacing: 2.05, position: [3, 4, -0.01] });
  });

  it("uses finite device bounds even when cells are far outside them", () => {
    expect(
      initialReferenceGrid({
        ...frame,
        cells: [{ ...cell, position: [1000, 2000, -99] }],
        constraints: {
          ...constraints,
          boxes: [
            {
              id: "1",
              center: [10, -20, -3],
              halfExtents: [8, 2, 1],
              coefficient: 1,
              allowedRegion: "inside",
            },
          ],
        },
      }),
    ).toEqual({ extent: 16, spacing: 0.8, position: [10, -20, -4.01] });
  });

  it("accounts for spherical and cylindrical device extents", () => {
    expect(
      initialReferenceGrid({
        ...frame,
        constraints: {
          ...constraints,
          spheres: [
            {
              id: "1",
              center: [50, 60, -8],
              radius: 6,
              coefficient: 1,
              allowedRegion: "outside",
            },
          ],
        },
      }),
    ).toEqual({ extent: 12, spacing: 0.6, position: [50, 60, -14.01] });
    expect(
      initialReferenceGrid({
        ...frame,
        constraints: {
          ...constraints,
          cylinders: [
            {
              id: "1",
              center: [-100, 40, 10],
              radius: 8,
              halfHeight: 2,
              coefficient: 1,
              allowedRegion: "inside",
            },
          ],
        },
      }),
    ).toEqual({ extent: 16, spacing: 0.8, position: [-100, 40, -0.01] });
  });

  it("ignores unbounded planes, including their arbitrary origin", () => {
    const planes = [
      {
        id: "1",
        point: [1e30, -1e30, -1e30] as const,
        inwardNormal: [0, 0, 1] as const,
        coefficient: 1,
      },
    ];
    expect(
      initialReferenceGrid({
        ...frame,
        constraints: { ...constraints, planes },
      }),
    ).toEqual(initialReferenceGrid(frame));
    expect(
      initialReferenceGrid({
        ...frame,
        cells: [],
        constraints: { ...constraints, planes },
      }),
    ).toEqual({ extent: 10, spacing: 0.5, position: [0, 0, -0.01] });
  });
});

describe("reference grid dataset lifecycle", () => {
  it("preserves grid intersections through growth, XYZ motion, division, empty frames, reset and seek", () => {
    const grid = new DatasetReferenceGrid();
    grid.beginDataset();
    const initial = grid.forFrame(frame);
    const changedFrames = [
      { ...frame, time: 5, cells: [{ ...cell, length: 400 }] },
      {
        ...frame,
        time: 6,
        cells: [{ ...cell, position: [100, -200, -300] as const }],
      },
      {
        ...frame,
        time: 7,
        cells: [
          cell,
          { ...cell, id: "2", slot: 1, position: [-50, 20, 30] as const },
        ],
      },
      { ...frame, time: 8, cells: [] },
      { ...frame, time: 0 },
      { ...frame, time: 4 },
    ];
    const intersections = (value: typeof initial) =>
      Array.from({ length: 21 }, (_, i) => [
        value.position[0] - value.extent / 2 + i * value.spacing,
        value.position[1] - value.extent / 2 + i * value.spacing,
        value.position[2],
      ]);
    for (const next of changedFrames) {
      expect(grid.forFrame(next)).toBe(initial);
      expect(intersections(grid.forFrame(next))).toEqual(
        intersections(initial),
      );
    }
  });

  it("keeps the fallback from an initially empty dataset when cells or devices arrive", () => {
    const grid = new DatasetReferenceGrid();
    const initial = grid.forFrame({ ...frame, cells: [] });
    expect(grid.forFrame(frame)).toBe(initial);
    const next = {
      ...frame,
      constraints: {
        ...constraints,
        boxes: [
          {
            id: "1",
            center: [20, 30, 0] as const,
            halfExtents: [50, 50, 50] as const,
            coefficient: 1,
            allowedRegion: "inside" as const,
          },
        ],
      },
    };
    expect(grid.forFrame(next)).toBe(initial);
    grid.beginDataset();
    expect(grid.forFrame(next)).toEqual({
      extent: 100,
      spacing: 5,
      position: [20, 30, -50.01],
    });
  });
});
