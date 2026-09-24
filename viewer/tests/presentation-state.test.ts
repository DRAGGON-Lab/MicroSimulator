import { describe, expect, it } from "vitest";
import { DatasetPresentationState } from "../src/presentation-state";
import type { SceneFrame, SceneSignalGrid } from "../src/scene";

const boundary = { kind: "no_flux" as const, values: [] };
const grid: SceneSignalGrid = {
  signalCount: 3,
  shape: [5, 7, 9],
  origin: [0, 0, 0],
  spacing: [1, 1, 1],
  boundaries: {
    xLower: boundary,
    xUpper: boundary,
    yLower: boundary,
    yUpper: boundary,
    zLower: boundary,
    zUpper: boundary,
  },
  levels: [],
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
  speciesCount: 3,
  cells: [],
  constraints: { planes: [], spheres: [], boxes: [], cylinders: [] },
  signalGrid: grid,
};

describe("dataset presentation lifecycle", () => {
  it("resets defaults only on an explicit new dataset", () => {
    const state = new DatasetPresentationState();
    state.beginDataset();
    expect(state.datasetId).toBe(1);
    expect(state.forFrame(frame).signalSlice).toBe(4);
    Object.assign(state.preferences, {
      colorMode: "growth-rate",
      signalVisible: false,
      signalAxis: "x",
      signalSlice: 3,
    });
    for (const time of [1, 100, 0, 20, 2]) {
      expect(state.forFrame({ ...frame, time })).toMatchObject({
        colorMode: "growth-rate",
        signalVisible: false,
        signalAxis: "x",
        signalSlice: 3,
      });
      expect(state.datasetId).toBe(1);
    }
    state.beginDataset();
    expect(state.datasetId).toBe(2);
    expect(state.forFrame(frame)).toEqual({
      colorMode: "cell-type",
      speciesChannel: 0,
      signalVisible: true,
      signalChannel: 0,
      signalAxis: "z",
      signalSlice: 4,
    });
  });

  it("retains channel and slice choices through absent or smaller grids", () => {
    const state = new DatasetPresentationState();
    state.beginDataset();
    Object.assign(state.preferences, {
      colorMode: "species",
      speciesChannel: 2,
      signalChannel: 2,
      signalVisible: false,
      signalAxis: "y",
      signalSlice: 6,
    });
    expect(
      state.forFrame({ ...frame, speciesCount: 0, signalGrid: null }),
    ).toMatchObject({
      colorMode: "cell-type",
      speciesChannel: 0,
      signalChannel: 0,
      signalSlice: 0,
    });
    expect(
      state.forFrame({
        ...frame,
        speciesCount: 1,
        signalGrid: { ...grid, signalCount: 1, shape: [2, 2, 2] },
      }),
    ).toMatchObject({
      colorMode: "species",
      speciesChannel: 0,
      signalChannel: 0,
      signalSlice: 1,
    });
    expect(state.forFrame(frame)).toMatchObject({
      colorMode: "species",
      speciesChannel: 2,
      signalChannel: 2,
      signalVisible: false,
      signalAxis: "y",
      signalSlice: 6,
    });
  });

  it("initializes the slice once when an initially missing grid arrives", () => {
    const state = new DatasetPresentationState();
    state.beginDataset();
    state.forFrame({ ...frame, signalGrid: null });
    expect(state.preferences.signalSlice).toBeNull();
    expect(state.forFrame(frame).signalSlice).toBe(4);
    expect(
      state.forFrame({ ...frame, signalGrid: { ...grid, shape: [5, 7, 99] } })
        .signalSlice,
    ).toBe(4);
  });
});
