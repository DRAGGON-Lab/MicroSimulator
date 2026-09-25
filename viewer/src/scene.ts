import canonicalize from "canonicalize";

export const SCENE_FORMAT = "microsimulator-scene";
export const SCENE_VERSION = 3;
export const MAX_SCENE_BYTES = 1 << 30;
// Presentation resource budget; this does not limit native engine counts.
export const MAX_SCENE_CHANNELS = 4096;

const UINT32_MAX = 2 ** 32 - 1;
const UINT64_MAX = (1n << 64n) - 1n;
const INT32_MIN = -(2 ** 31);
const INT32_MAX = 2 ** 31 - 1;
const FLOAT32_MAX = 3.4028234663852886e38;

export type BackendKind = "cpu" | "metal" | "cuda";
export type BoundaryKind = "no_flux" | "periodic" | "fixed";
export type RegionKind = "outside" | "inside";
export type Vector3 = readonly [number, number, number];

export interface SceneBackend {
  readonly kind: BackendKind;
  readonly name: string;
  readonly device: string;
  readonly deviceIndex: number;
  readonly native: boolean;
}

export interface SceneCell {
  readonly id: string;
  readonly parentId: string | null;
  readonly slot: number;
  readonly position: Vector3;
  readonly direction: Vector3;
  readonly length: number;
  readonly radius: number;
  readonly growthRate: number;
  readonly cellType: number;
  readonly fixed: boolean;
  readonly species: readonly number[];
}

export interface ScenePlaneConstraint {
  readonly id: string;
  readonly point: Vector3;
  readonly inwardNormal: Vector3;
  readonly coefficient: number;
}

export interface SceneSphereConstraint {
  readonly id: string;
  readonly center: Vector3;
  readonly radius: number;
  readonly coefficient: number;
  readonly allowedRegion: RegionKind;
}

export interface SceneBoxConstraint {
  readonly id: string;
  readonly center: Vector3;
  readonly halfExtents: Vector3;
  readonly coefficient: number;
  readonly allowedRegion: RegionKind;
}

export interface SceneCylinderConstraint {
  readonly id: string;
  readonly center: Vector3;
  readonly radius: number;
  readonly halfHeight: number;
  readonly coefficient: number;
  readonly allowedRegion: RegionKind;
}

export interface SceneConstraints {
  readonly planes: readonly ScenePlaneConstraint[];
  readonly spheres: readonly SceneSphereConstraint[];
  readonly boxes: readonly SceneBoxConstraint[];
  readonly cylinders: readonly SceneCylinderConstraint[];
}

export interface SceneGridBoundary {
  readonly kind: BoundaryKind;
  readonly values: readonly number[];
}

export interface SceneSignalGrid {
  readonly signalCount: number;
  readonly shape: readonly [number, number, number];
  readonly origin: Vector3;
  readonly spacing: Vector3;
  readonly boundaries: Readonly<{
    xLower: SceneGridBoundary;
    xUpper: SceneGridBoundary;
    yLower: SceneGridBoundary;
    yUpper: SceneGridBoundary;
    zLower: SceneGridBoundary;
    zUpper: SceneGridBoundary;
  }>;
  readonly levels: readonly number[];
}

export type ChannelKind = "species" | "signals";
export interface ChannelMetadata {
  readonly species: readonly (string | null)[];
  readonly signals: readonly (string | null)[];
}

// Metadata arrays are readonly. Weak keys retain no discarded frame/history;
// resolving once also avoids rebuilding a whole group for every selector row.
const displayChannelLabels = new WeakMap<
  readonly (string | null)[],
  readonly string[]
>();

/** Presentation only. Numerical indices, never labels, identify channels. */
export function channelLabel(
  frame: SceneFrame,
  kind: ChannelKind,
  index: number,
): string {
  const labels = frame.channelMetadata[kind];
  if (!Number.isInteger(index) || index < 0 || index >= labels.length) {
    throw new RangeError(`${kind} channel ${index} is out of range`);
  }
  let resolved = displayChannelLabels.get(labels);
  if (resolved === undefined) {
    const names = labels.map((label, slot) =>
      label === null || label.trim() === ""
        ? `Channel ${slot}`
        : label.replace(/[\t\n\f\r ]+/g, " ").replace(/^ | $/g, ""),
    );
    const counts = new Map<string, number>();
    for (const name of names) counts.set(name, (counts.get(name) ?? 0) + 1);
    resolved = names.map((name, slot) =>
      counts.get(name)! > 1 ? `${name} [${slot}]` : name,
    );
    // A supplied name can imitate an automatically indexed duplicate, e.g.
    // ["GFP", "GFP", "GFP [0]"]. In that case index the entire group: the
    // distinct final indices guarantee uniqueness even for nested suffixes.
    if (new Set(resolved).size !== resolved.length) {
      resolved = names.map((name, slot) => `${name} [${slot}]`);
    }
    displayChannelLabels.set(labels, resolved);
  }
  return resolved[index]!;
}

export interface SceneFrame {
  readonly time: number;
  readonly backend: SceneBackend;
  readonly speciesCount: number;
  readonly channelMetadata: ChannelMetadata;
  readonly cells: readonly SceneCell[];
  readonly constraints: SceneConstraints;
  readonly signalGrid: SceneSignalGrid | null;
}

export class SceneFormatError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "SceneFormatError";
  }
}

function fail(path: string, message: string): never {
  throw new SceneFormatError(`${path}: ${message}`);
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return fail(path, "expected an object");
  }
  return value as Record<string, unknown>;
}

function array(value: unknown, path: string): readonly unknown[] {
  if (!Array.isArray(value)) {
    return fail(path, "expected an array");
  }
  return value;
}

function exactKeys(
  value: Record<string, unknown>,
  path: string,
  expected: readonly string[],
): void {
  const wanted = new Set(expected);
  const missing = expected.filter((key) => !(key in value));
  const unknown = Object.keys(value).filter((key) => !wanted.has(key));
  if (missing.length > 0) {
    fail(path, `missing keys ${JSON.stringify(missing.toSorted())}`);
  }
  if (unknown.length > 0) {
    fail(path, `unknown keys ${JSON.stringify(unknown.toSorted())}`);
  }
}

function string(value: unknown, path: string): string {
  if (typeof value !== "string") {
    return fail(path, "expected a string");
  }
  return value;
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") {
    return fail(path, "expected a boolean");
  }
  return value;
}

function number(value: unknown, path: string, float32 = false): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fail(path, "expected a finite number");
  }
  if (float32 && Math.abs(value) > FLOAT32_MAX) {
    return fail(path, "number is outside the finite float32 range");
  }
  return value;
}

function integer(
  value: unknown,
  path: string,
  minimum: number,
  maximum: number,
): number {
  const result = number(value, path);
  if (!Number.isSafeInteger(result)) {
    return fail(path, "expected a safe integer");
  }
  if (result < minimum || result > maximum) {
    return fail(path, `integer is outside [${minimum}, ${maximum}]`);
  }
  return result;
}

function identifier(value: unknown, path: string): string {
  const result = string(value, path);
  if (!/^[1-9][0-9]*$/.test(result)) {
    return fail(path, "expected a canonical positive decimal uint64 string");
  }
  if (BigInt(result) > UINT64_MAX) {
    return fail(path, "identifier is outside the positive uint64 range");
  }
  return result;
}

function tuple3(value: unknown, path: string, positive = false): Vector3 {
  const values = array(value, path);
  if (values.length !== 3) {
    return fail(path, "expected exactly three values");
  }
  const result: Vector3 = [
    number(values[0], `${path}[0]`, true),
    number(values[1], `${path}[1]`, true),
    number(values[2], `${path}[2]`, true),
  ];
  if (positive && result.some((component) => component <= 0)) {
    return fail(path, "values must be positive");
  }
  return result;
}

function floatArray(value: unknown, path: string): readonly number[] {
  return array(value, path).map((item, index) =>
    number(item, `${path}[${index}]`, true),
  );
}

function parseBackend(value: unknown, path: string): SceneBackend {
  const data = record(value, path);
  exactKeys(data, path, ["kind", "name", "device", "device_index", "native"]);
  const kind = string(data.kind, `${path}.kind`);
  if (kind !== "cpu" && kind !== "metal" && kind !== "cuda") {
    return fail(`${path}.kind`, `unknown backend kind ${JSON.stringify(kind)}`);
  }
  const name = string(data.name, `${path}.name`);
  const device = string(data.device, `${path}.device`);
  if (name.length === 0 || device.length === 0) {
    return fail(path, "name and device must not be empty");
  }
  return {
    kind,
    name,
    device,
    deviceIndex: integer(
      data.device_index,
      `${path}.device_index`,
      0,
      UINT32_MAX,
    ),
    native: boolean(data.native, `${path}.native`),
  };
}

function parseCell(
  value: unknown,
  path: string,
  speciesCount: number,
): SceneCell {
  const data = record(value, path);
  exactKeys(data, path, [
    "id",
    "parent_id",
    "slot",
    "position",
    "direction",
    "length",
    "radius",
    "growth_rate",
    "cell_type",
    "fixed",
    "species",
  ]);
  const id = identifier(data.id, `${path}.id`);
  const parentId =
    data.parent_id === null
      ? null
      : identifier(data.parent_id, `${path}.parent_id`);
  if (parentId !== null && BigInt(parentId) >= BigInt(id)) {
    return fail(`${path}.parent_id`, "must precede the child identifier");
  }
  const direction = tuple3(data.direction, `${path}.direction`);
  const norm = Math.hypot(...direction);
  if (Math.abs(norm - 1) > 1e-5) {
    return fail(`${path}.direction`, "must be normalized");
  }
  const length = number(data.length, `${path}.length`, true);
  const radius = number(data.radius, `${path}.radius`, true);
  if (length < 0) {
    return fail(`${path}.length`, "must be non-negative");
  }
  if (radius <= 0) {
    return fail(`${path}.radius`, "must be positive");
  }
  const species = floatArray(data.species, `${path}.species`);
  if (species.length !== speciesCount) {
    return fail(`${path}.species`, `expected ${speciesCount} values`);
  }
  return {
    id,
    parentId,
    slot: integer(data.slot, `${path}.slot`, 0, UINT32_MAX - 1),
    position: tuple3(data.position, `${path}.position`),
    direction,
    length,
    radius,
    growthRate: number(data.growth_rate, `${path}.growth_rate`, true),
    cellType: integer(
      data.cell_type,
      `${path}.cell_type`,
      INT32_MIN,
      INT32_MAX,
    ),
    fixed: boolean(data.fixed, `${path}.fixed`),
    species,
  };
}

function parseBoundary(
  value: unknown,
  path: string,
  signalCount: number,
): SceneGridBoundary {
  const data = record(value, path);
  exactKeys(data, path, ["kind", "values"]);
  const kind = string(data.kind, `${path}.kind`);
  if (kind !== "no_flux" && kind !== "periodic" && kind !== "fixed") {
    return fail(
      `${path}.kind`,
      `unknown boundary kind ${JSON.stringify(kind)}`,
    );
  }
  const values = floatArray(data.values, `${path}.values`);
  const expected = kind === "fixed" ? signalCount : 0;
  if (values.length !== expected) {
    return fail(
      `${path}.values`,
      `expected ${expected} values for ${kind} boundary`,
    );
  }
  return { kind, values };
}

function positiveNumber(value: unknown, path: string): number {
  const result = number(value, path, true);
  if (result <= 0) {
    return fail(path, "must be positive");
  }
  return result;
}

function parseRegion(value: unknown, path: string): RegionKind {
  const region = string(value, path);
  if (region !== "outside" && region !== "inside") {
    return fail(path, `unknown region kind ${JSON.stringify(region)}`);
  }
  return region;
}

function parsePlaneConstraint(
  value: unknown,
  path: string,
): ScenePlaneConstraint {
  const data = record(value, path);
  exactKeys(data, path, ["id", "point", "inward_normal", "coefficient"]);
  const inwardNormal = tuple3(data.inward_normal, `${path}.inward_normal`);
  if (Math.abs(Math.hypot(...inwardNormal) - 1) > 1e-5) {
    return fail(`${path}.inward_normal`, "must be normalized");
  }
  return {
    id: identifier(data.id, `${path}.id`),
    point: tuple3(data.point, `${path}.point`),
    inwardNormal,
    coefficient: positiveNumber(data.coefficient, `${path}.coefficient`),
  };
}

function parseSphereConstraint(
  value: unknown,
  path: string,
): SceneSphereConstraint {
  const data = record(value, path);
  exactKeys(data, path, [
    "id",
    "center",
    "radius",
    "coefficient",
    "allowed_region",
  ]);
  return {
    id: identifier(data.id, `${path}.id`),
    center: tuple3(data.center, `${path}.center`),
    radius: positiveNumber(data.radius, `${path}.radius`),
    coefficient: positiveNumber(data.coefficient, `${path}.coefficient`),
    allowedRegion: parseRegion(data.allowed_region, `${path}.allowed_region`),
  };
}

function parseBoxConstraint(value: unknown, path: string): SceneBoxConstraint {
  const data = record(value, path);
  exactKeys(data, path, [
    "id",
    "center",
    "half_extents",
    "coefficient",
    "allowed_region",
  ]);
  return {
    id: identifier(data.id, `${path}.id`),
    center: tuple3(data.center, `${path}.center`),
    halfExtents: tuple3(data.half_extents, `${path}.half_extents`, true),
    coefficient: positiveNumber(data.coefficient, `${path}.coefficient`),
    allowedRegion: parseRegion(data.allowed_region, `${path}.allowed_region`),
  };
}

function parseCylinderConstraint(
  value: unknown,
  path: string,
): SceneCylinderConstraint {
  const data = record(value, path);
  exactKeys(data, path, [
    "id",
    "center",
    "radius",
    "half_height",
    "coefficient",
    "allowed_region",
  ]);
  return {
    id: identifier(data.id, `${path}.id`),
    center: tuple3(data.center, `${path}.center`),
    radius: positiveNumber(data.radius, `${path}.radius`),
    halfHeight: positiveNumber(data.half_height, `${path}.half_height`),
    coefficient: positiveNumber(data.coefficient, `${path}.coefficient`),
    allowedRegion: parseRegion(data.allowed_region, `${path}.allowed_region`),
  };
}

function parseConstraints(value: unknown, path: string): SceneConstraints {
  const data = record(value, path);
  exactKeys(data, path, ["planes", "spheres", "boxes", "cylinders"]);
  const constraints: SceneConstraints = {
    planes: array(data.planes, `${path}.planes`).map((item, index) =>
      parsePlaneConstraint(item, `${path}.planes[${index}]`),
    ),
    spheres: array(data.spheres, `${path}.spheres`).map((item, index) =>
      parseSphereConstraint(item, `${path}.spheres[${index}]`),
    ),
    boxes: array(data.boxes, `${path}.boxes`).map((item, index) =>
      parseBoxConstraint(item, `${path}.boxes[${index}]`),
    ),
    cylinders: array(data.cylinders, `${path}.cylinders`).map((item, index) =>
      parseCylinderConstraint(item, `${path}.cylinders[${index}]`),
    ),
  };
  const identifiers = new Set<string>();
  for (const kind of [
    constraints.planes,
    constraints.spheres,
    constraints.boxes,
    constraints.cylinders,
  ] as const) {
    for (const constraint of kind) {
      if (identifiers.has(constraint.id)) {
        return fail(path, `duplicate constraint identifier ${constraint.id}`);
      }
      identifiers.add(constraint.id);
    }
  }
  return constraints;
}

function parseSignalGrid(value: unknown, path: string): SceneSignalGrid | null {
  if (value === null) {
    return null;
  }
  const data = record(value, path);
  exactKeys(data, path, [
    "signal_count",
    "shape",
    "origin",
    "spacing",
    "boundaries",
    "levels",
  ]);
  const signalCount = sceneChannelCount(
    data.signal_count,
    `${path}.signal_count`,
    1,
  );
  const shapeValues = array(data.shape, `${path}.shape`);
  if (shapeValues.length !== 3) {
    return fail(`${path}.shape`, "expected exactly three dimensions");
  }
  const shape: readonly [number, number, number] = [
    integer(shapeValues[0], `${path}.shape[0]`, 1, UINT32_MAX),
    integer(shapeValues[1], `${path}.shape[1]`, 1, UINT32_MAX),
    integer(shapeValues[2], `${path}.shape[2]`, 1, UINT32_MAX),
  ];
  const boundaries = record(data.boundaries, `${path}.boundaries`);
  exactKeys(boundaries, `${path}.boundaries`, [
    "x_lower",
    "x_upper",
    "y_lower",
    "y_upper",
    "z_lower",
    "z_upper",
  ]);
  const levels = floatArray(data.levels, `${path}.levels`);
  const expectedLevels = signalCount * shape[0] * shape[1] * shape[2];
  if (
    !Number.isSafeInteger(expectedLevels) ||
    levels.length !== expectedLevels
  ) {
    return fail(`${path}.levels`, `expected ${expectedLevels} values`);
  }
  return {
    signalCount,
    shape,
    origin: tuple3(data.origin, `${path}.origin`),
    spacing: tuple3(data.spacing, `${path}.spacing`, true),
    boundaries: {
      xLower: parseBoundary(
        boundaries.x_lower,
        `${path}.boundaries.x_lower`,
        signalCount,
      ),
      xUpper: parseBoundary(
        boundaries.x_upper,
        `${path}.boundaries.x_upper`,
        signalCount,
      ),
      yLower: parseBoundary(
        boundaries.y_lower,
        `${path}.boundaries.y_lower`,
        signalCount,
      ),
      yUpper: parseBoundary(
        boundaries.y_upper,
        `${path}.boundaries.y_upper`,
        signalCount,
      ),
      zLower: parseBoundary(
        boundaries.z_lower,
        `${path}.boundaries.z_lower`,
        signalCount,
      ),
      zUpper: parseBoundary(
        boundaries.z_upper,
        `${path}.boundaries.z_upper`,
        signalCount,
      ),
    },
    levels,
  };
}

function parseChannelMetadata(
  value: unknown,
  path: string,
  speciesCount: number,
  signalCount: number,
): ChannelMetadata {
  const data = record(value, path);
  exactKeys(data, path, ["species", "signals"]);
  function labels(
    kind: ChannelKind,
    count: number,
  ): readonly (string | null)[] {
    const values = array(data[kind], `${path}.${kind}`);
    if (values.length !== count) {
      return fail(
        `${path}.${kind}`,
        `expected ${count} labels, got ${values.length}`,
      );
    }
    return values.map((value, index) => {
      if (value === null) return null;
      const label = string(value, `${path}.${kind}[${index}]`);
      if (/[\uD800-\uDFFF]/u.test(label))
        return fail(
          `${path}.${kind}[${index}]`,
          "invalid Unicode scalar value",
        );
      return label;
    });
  }
  return {
    species: labels("species", speciesCount),
    signals: labels("signals", signalCount),
  };
}

function sceneChannelCount(value: unknown, path: string, minimum = 0): number {
  const count = integer(value, path, minimum, UINT32_MAX);
  if (count > MAX_SCENE_CHANNELS) {
    return fail(
      path,
      `exceeds scene presentation channel budget of ${MAX_SCENE_CHANNELS} per group`,
    );
  }
  return count;
}

function parseFrame(value: unknown, path: string, version: number): SceneFrame {
  const data = record(value, path);
  exactKeys(data, path, [
    "time",
    "backend",
    "species_count",
    "cells",
    "constraints",
    "signal_grid",
    ...(version >= 3 ? ["channel_metadata"] : []),
  ]);
  const time = number(data.time, `${path}.time`);
  if (time < 0) {
    return fail(`${path}.time`, "must be non-negative");
  }
  const speciesCount = sceneChannelCount(
    data.species_count,
    `${path}.species_count`,
  );
  const cells = array(data.cells, `${path}.cells`).map((item, index) =>
    parseCell(item, `${path}.cells[${index}]`, speciesCount),
  );
  const identifiers = new Set<string>();
  for (const [index, cell] of cells.entries()) {
    if (cell.slot !== index) {
      return fail(
        `${path}.cells[${index}].slot`,
        "cells must be compact and ordered by slot",
      );
    }
    if (identifiers.has(cell.id)) {
      return fail(`${path}.cells[${index}].id`, "duplicate cell identifier");
    }
    identifiers.add(cell.id);
  }
  const signalGrid = parseSignalGrid(data.signal_grid, `${path}.signal_grid`);
  const signalCount = signalGrid?.signalCount ?? 0;
  // Digest verification has already completed against the unmodified source frame.
  const channelMetadata =
    version >= 3
      ? parseChannelMetadata(
          data.channel_metadata,
          `${path}.channel_metadata`,
          speciesCount,
          signalCount,
        )
      : {
          species: Array<string | null>(speciesCount).fill(null),
          signals: Array<string | null>(signalCount).fill(null),
        };
  return {
    time,
    channelMetadata,
    backend: parseBackend(data.backend, `${path}.backend`),
    speciesCount,
    cells,
    constraints: parseConstraints(data.constraints, `${path}.constraints`),
    signalGrid,
  };
}

async function sha256(value: string): Promise<string> {
  if (globalThis.crypto?.subtle === undefined) {
    throw new SceneFormatError(
      "SHA-256 verification requires the Web Crypto API",
    );
  }
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export async function parseScene(source: string): Promise<SceneFrame> {
  if (source.length === 0) {
    throw new SceneFormatError("scene is empty");
  }
  if (new TextEncoder().encode(source).byteLength > MAX_SCENE_BYTES) {
    throw new SceneFormatError(
      `scene exceeds the ${MAX_SCENE_BYTES}-byte limit`,
    );
  }

  let decoded: unknown;
  try {
    decoded = JSON.parse(source) as unknown;
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new SceneFormatError(`scene is not valid JSON: ${detail}`);
  }

  const root = record(decoded, "$");
  exactKeys(root, "$", ["format", "version", "producer", "integrity", "frame"]);
  if (
    ![SCENE_FORMAT, "cellmodeller2-scene"].includes(
      string(root.format, "$.format"),
    )
  ) {
    return fail("$.format", "not a MicroSimulator scene");
  }
  const version = integer(root.version, "$.version", 0, UINT32_MAX);
  if (version !== 2 && version !== SCENE_VERSION) {
    return fail(
      "$.version",
      `unsupported scene version ${String(root.version)}`,
    );
  }
  const producer = record(root.producer, "$.producer");
  exactKeys(producer, "$.producer", ["name", "version"]);
  string(producer.name, "$.producer.name");
  string(producer.version, "$.producer.version");
  const integrity = record(root.integrity, "$.integrity");
  exactKeys(integrity, "$.integrity", ["algorithm", "frame"]);
  if (string(integrity.algorithm, "$.integrity.algorithm") !== "sha256") {
    return fail("$.integrity.algorithm", "unsupported integrity algorithm");
  }
  const expectedDigest = string(integrity.frame, "$.integrity.frame");
  if (!/^[0-9a-f]{64}$/.test(expectedDigest)) {
    return fail("$.integrity.frame", "expected a lowercase SHA-256 digest");
  }
  let canonical: string | undefined;
  try {
    canonical = canonicalize(root.frame);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new SceneFormatError(`scene cannot be canonicalized: ${detail}`);
  }
  if (canonical === undefined) {
    return fail("$.frame", "cannot be represented by RFC 8785 canonical JSON");
  }
  const actualDigest = await sha256(canonical);
  if (actualDigest !== expectedDigest) {
    return fail("$.integrity.frame", "frame digest does not match");
  }
  return parseFrame(root.frame, "$.frame", version);
}
