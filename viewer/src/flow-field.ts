import type { SliceAxis } from "./grid";
import type { SceneSignalGrid } from "./scene";

type PlaneAxis = 0 | 1 | 2;

export interface FlowSlice {
  width: number;
  height: number;
  horizontal: PlaneAxis;
  vertical: PlaneAxis;
  center: [number, number, number];
  dx: number;
  dy: number;
  velocity: Float32Array;
  change: Float32Array | null;
  peakSpeed: number;
  peakChange: number;
  changeScale: number;
}

export function flowSlice(
  grid: SceneSignalGrid,
  axis: SliceAxis,
  slice: number,
  reference?: SceneSignalGrid["velocityFaces"],
): FlowSlice {
  const faces = grid.velocityFaces;
  if (!faces) throw new Error("the grid has no velocity field");
  const normal = axis === "x" ? 0 : axis === "y" ? 1 : 2;
  if (!Number.isInteger(slice) || slice < 0 || slice >= grid.shape[normal])
    throw new RangeError("velocity slice is outside the grid");
  const horizontal = axis === "x" ? 1 : 0;
  const vertical = axis === "z" ? 1 : 2;
  const width = grid.shape[horizontal],
    height = grid.shape[vertical];
  const center: [number, number, number] = [...grid.origin];
  center[normal] += slice * grid.spacing[normal];
  const dx = grid.spacing[horizontal],
    dy = grid.spacing[vertical];
  center[horizontal] += ((width - 1) * dx) / 2;
  center[vertical] += ((height - 1) * dy) / 2;
  const velocity = new Float32Array(2 * width * height);
  const change = reference ? new Float32Array(width * height) : null;
  let peakSpeed = 0,
    peakChange = 0,
    referencePeak = 0;

  const [, ny, nz] = grid.shape;
  const strides = [ny * nz, nz, 1] as const;
  const site = [0, 0, 0] as [number, number, number];
  site[normal] = slice;
  for (let row = 0; row < height; row++) {
    site[vertical] = row;
    for (let col = 0; col < width; col++) {
      site[horizontal] = col;
      const [x, y, z] = site;
      const offsets = [
        (x * ny + y) * nz + z,
        (x * (ny + 1) + y) * nz + z,
        (x * ny + y) * (nz + 1) + z,
      ];
      const centered = (field: typeof faces, a: PlaneAxis) => {
        const i = offsets[a]!;
        return (field[a][i]! + field[a][i + strides[a]]!) / 2;
      };
      const i = row * width + col;
      const u = centered(faces, horizontal),
        v = centered(faces, vertical);
      velocity[2 * i] = u;
      velocity[2 * i + 1] = v;
      peakSpeed = Math.max(peakSpeed, Math.hypot(u, v));
      if (reference && change) {
        const u0 = centered(reference, horizontal),
          v0 = centered(reference, vertical);
        referencePeak = Math.max(referencePeak, Math.hypot(u0, v0));
        const magnitude = Math.hypot(u - u0, v - v0);
        change[i] = magnitude;
        peakChange = Math.max(peakChange, magnitude);
      }
    }
  }
  return {
    width,
    height,
    horizontal,
    vertical,
    center,
    dx,
    dy,
    velocity,
    change,
    peakSpeed,
    peakChange,
    changeScale: Math.max(peakChange, referencePeak),
  };
}

// Sample in grid coordinates, with texel centers at integer coordinates.
function sample(
  field: FlowSlice,
  x: number,
  y: number,
  out: [number, number],
): boolean {
  const { width, height, velocity } = field;
  if (x < -0.5 || x >= width - 0.5 || y < -0.5 || y >= height - 0.5)
    return false;
  const nearest = 2 * (Math.round(y) * width + Math.round(x));
  if (velocity[nearest] === 0 && velocity[nearest + 1] === 0) return false;
  x = Math.max(0, x);
  y = Math.max(0, y);
  const col = Math.floor(x),
    row = Math.floor(y);
  const right = Math.min(col + 1, width - 1),
    top = Math.min(row + 1, height - 1);
  const fx = x - col,
    fy = y - row;
  for (let a = 0; a < 2; a++) {
    const bottom =
      (1 - fx) * velocity[2 * (row * width + col) + a]! +
      fx * velocity[2 * (row * width + right) + a]!;
    const upper =
      (1 - fx) * velocity[2 * (top * width + col) + a]! +
      fx * velocity[2 * (top * width + right) + a]!;
    out[a] = bottom * (1 - fy) + upper * fy;
  }
  return true;
}

export interface FlowStroke {
  // Interleaved grid x, grid y and simulated travel time since the seed.
  points: number[];
  phase: number;
}

/** Target separation in scene length units, independent of camera zoom. */
export function streamlineSpacing(field: FlowSlice): number {
  return Math.max(field.width * field.dx, field.height * field.dy) / 96;
}

export function streamlines(field: FlowSlice): FlowStroke[] {
  if (field.peakSpeed === 0) return [];
  const { width, height, dx, dy } = field;
  const spacing = streamlineSpacing(field);
  // Reserve a band around completed curves so neighboring seeds fill gaps
  // instead of redrawing the same channel. Store the mask in physical space.
  const bin = spacing / 3;
  const cols = Math.ceil((width * dx) / bin),
    rows = Math.ceil((height * dy) / bin);
  const seedMask = new Uint8Array(cols * rows);
  const centerlines = new Uint8Array(cols * rows);
  const column = (x: number) =>
    Math.min(cols - 1, Math.floor(((x + 0.5) * dx) / bin));
  const row = (y: number) =>
    Math.min(rows - 1, Math.floor(((y + 0.5) * dy) / bin));
  const binIndex = (x: number, y: number) => row(y) * cols + column(x);
  const vector: [number, number] = [0, 0];

  function trace(x: number, y: number, sign: number): number[] {
    const points: number[] = [];
    let time = 0,
      distance = 0;
    // Midpoint integration uses physical velocity and signed simulation time.
    // Limit steps in grid coordinates to resolve walls on anisotropic grids.
    while (
      points.length < 3 * 768 &&
      distance < 115 * spacing &&
      sample(field, x, y, vector)
    ) {
      const [u, v] = vector;
      const speed = Math.hypot(u, v);
      if (speed < field.peakSpeed * 0.005) break;
      const dt =
        sign *
        Math.min(
          spacing / (3 * speed),
          0.4 / Math.max(Math.abs(u / dx), Math.abs(v / dy)),
        );
      if (
        !sample(field, x + (0.5 * dt * u) / dx, y + (0.5 * dt * v) / dy, vector)
      )
        break;
      const nextX = x + (dt * vector[0]) / dx,
        nextY = y + (dt * vector[1]) / dy;
      if (
        !sample(field, nextX, nextY, vector) ||
        centerlines[binIndex(nextX, nextY)]
      )
        break;
      const moved = Math.hypot((nextX - x) * dx, (nextY - y) * dy);
      if (moved < spacing * 1e-6) break;
      distance += moved;
      time += dt;
      x = nextX;
      y = nextY;
      points.push(x, y, time);
    }
    return points;
  }

  const strokes: FlowStroke[] = [];
  for (let iy = 1; iy < rows + 1; iy += 3) {
    for (let ix = 1; ix < cols + 1; ix += 3) {
      const x = Math.min(width - 0.5 - 1e-6, ((ix + 0.5) * bin) / dx - 0.5);
      const y = Math.min(height - 0.5 - 1e-6, ((iy + 0.5) * bin) / dy - 0.5);
      if (seedMask[binIndex(x, y)] || !sample(field, x, y, vector)) continue;
      const back = trace(x, y, -1),
        forward = trace(x, y, 1);
      if (!back.length && !forward.length) continue;
      const points: number[] = [];
      for (let i = back.length - 3; i >= 0; i -= 3)
        points.push(back[i]!, back[i + 1]!, back[i + 2]!);
      points.push(x, y, 0, ...forward);
      strokes.push({ points, phase: (ix * 0.61803399 + iy * 0.41421356) % 1 });
      for (let i = 0; i < points.length; i += 3) {
        const cx = column(points[i]!),
          cy = row(points[i + 1]!);
        // A converging path can enter the seed exclusion band. Only stop it
        // when it meets the narrower band occupied by the actual centerline.
        centerlines[cy * cols + cx] = 1;
        for (
          let sy = Math.max(0, cy - 2);
          sy <= Math.min(rows - 1, cy + 2);
          sy++
        ) {
          seedMask.fill(
            1,
            sy * cols + Math.max(0, cx - 2),
            sy * cols + Math.min(cols, cx + 3),
          );
        }
      }
    }
  }
  return strokes;
}

export interface VectorDisplay {
  reference: FlowSlice;
  scaleSpeed: number;
  gain: number;
}

/** Sample native sites. Both velocities use one physical length scale and the
 * same tail, so tip displacement measures the local change in velocity. */
export function vectorArrows(field: FlowSlice, display: VectorDisplay) {
  if (display.scaleSpeed === 0) return [];
  const { width, height, dx, dy } = field;
  const pitch = Math.max(width * dx, height * dy) / 60;
  const strideX = Math.max(1, Math.round(pitch / dx));
  const strideY = Math.max(1, Math.round(pitch / dy));
  const stepX = Math.min(width, strideX) * dx,
    stepY = Math.min(height, strideY) * dy;
  const spacing =
    width === 1 ? stepY : height === 1 ? stepX : Math.min(stepX, stepY);
  const scale = (0.96 * spacing * display.gain) / display.scaleSpeed;
  const arrows = [];
  for (
    let row = Math.floor(((height - 1) % strideY) / 2);
    row < height;
    row += strideY
  ) {
    for (
      let col = Math.floor(((width - 1) % strideX) / 2);
      col < width;
      col += strideX
    ) {
      const i = 2 * (row * width + col);
      const u = field.velocity[i]! * scale,
        v = field.velocity[i + 1]! * scale;
      const u0 = display.reference.velocity[i]! * scale,
        v0 = display.reference.velocity[i + 1]! * scale;
      if (u === 0 && v === 0 && u0 === 0 && v0 === 0) continue;
      arrows.push({
        x: (col - (width - 1) / 2) * dx,
        y: (row - (height - 1) / 2) * dy,
        u,
        v,
        u0,
        v0,
        tipOpacity: Math.min(1, Math.hypot(u - u0, v - v0) / (0.2 * spacing)),
      });
    }
  }
  return arrows;
}
