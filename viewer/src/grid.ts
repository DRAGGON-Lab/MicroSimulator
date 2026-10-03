import type { SceneSignalGrid, Vector3 } from "./scene";

export type SliceAxis = "x" | "y" | "z";

export interface SignalSlice {
  readonly axis: SliceAxis;
  readonly index: number;
  readonly channel: number;
  readonly width: number;
  readonly height: number;
  readonly values: readonly number[];
  readonly center: Vector3;
  readonly horizontal: Vector3;
  readonly vertical: Vector3;
  readonly horizontalSpan: number;
  readonly verticalSpan: number;
}

export function flatSignalIndex(
  grid: SceneSignalGrid,
  channel: number,
  x: number,
  y: number,
  z: number,
): number {
  const [sizeX, sizeY, sizeZ] = grid.shape;

  if (channel < 0 || channel >= grid.signalCount) {
    throw new RangeError(`signal channel ${channel} is out of range`);
  }

  if (x < 0 || x >= sizeX || y < 0 || y >= sizeY || z < 0 || z >= sizeZ) {
    throw new RangeError(
      `signal coordinate (${x}, ${y}, ${z}) is out of range`,
    );
  }

  return channel * sizeX * sizeY * sizeZ + x * sizeY * sizeZ + y * sizeZ + z;
}

function span(dimension: number, spacing: number): number {
  return Math.max(spacing, (dimension - 1) * spacing);
}

export function sliceDimension(grid: SceneSignalGrid, axis: SliceAxis): number {
  return grid.shape[axis === "x" ? 0 : axis === "y" ? 1 : 2];
}

function sliceValues(
  grid: SceneSignalGrid,
  channel: number,
  axis: SliceAxis,
  index: number,
): number[] {
  const [sizeX, sizeY, sizeZ] = grid.shape;
  const values: number[] = [];

  if (axis === "x") {
    for (let z = 0; z < sizeZ; z += 1) {
      for (let y = 0; y < sizeY; y += 1) {
        values.push(
          grid.levels[flatSignalIndex(grid, channel, index, y, z)] ?? 0,
        );
      }
    }

    return values;
  }

  if (axis === "y") {
    for (let z = 0; z < sizeZ; z += 1) {
      for (let x = 0; x < sizeX; x += 1) {
        values.push(
          grid.levels[flatSignalIndex(grid, channel, x, index, z)] ?? 0,
        );
      }
    }

    return values;
  }

  for (let y = 0; y < sizeY; y += 1) {
    for (let x = 0; x < sizeX; x += 1) {
      values.push(
        grid.levels[flatSignalIndex(grid, channel, x, y, index)] ?? 0,
      );
    }
  }

  return values;
}

export function signalSlice(
  grid: SceneSignalGrid,
  channel: number,
  axis: SliceAxis,
  index: number,
): SignalSlice {
  const dimension = sliceDimension(grid, axis);

  if (!Number.isInteger(index) || index < 0 || index >= dimension) {
    throw new RangeError(`${axis} slice ${index} is out of range`);
  }

  const [sizeX, sizeY, sizeZ] = grid.shape;
  const [originX, originY, originZ] = grid.origin;
  const [spacingX, spacingY, spacingZ] = grid.spacing;
  const centerX = originX + ((sizeX - 1) * spacingX) / 2;
  const centerY = originY + ((sizeY - 1) * spacingY) / 2;
  const centerZ = originZ + ((sizeZ - 1) * spacingZ) / 2;

  const values = sliceValues(grid, channel, axis, index);

  if (axis === "x") {
    return {
      axis,
      index,
      channel,
      width: sizeY,
      height: sizeZ,
      values,
      center: [originX + index * spacingX, centerY, centerZ],
      horizontal: [0, 1, 0],
      vertical: [0, 0, 1],
      horizontalSpan: span(sizeY, spacingY),
      verticalSpan: span(sizeZ, spacingZ),
    };
  }

  if (axis === "y") {
    return {
      axis,
      index,
      channel,
      width: sizeX,
      height: sizeZ,
      values,
      center: [centerX, originY + index * spacingY, centerZ],
      horizontal: [1, 0, 0],
      vertical: [0, 0, 1],
      horizontalSpan: span(sizeX, spacingX),
      verticalSpan: span(sizeZ, spacingZ),
    };
  }

  return {
    axis,
    index,
    channel,
    width: sizeX,
    height: sizeY,
    values,
    center: [centerX, centerY, originZ + index * spacingZ],
    horizontal: [1, 0, 0],
    vertical: [0, 1, 0],
    horizontalSpan: span(sizeX, spacingX),
    verticalSpan: span(sizeY, spacingY),
  };
}
