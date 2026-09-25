import canonicalize from "canonicalize";
import {
  MAX_SCENE_BYTES,
  parseScene,
  parseSceneBackend,
  type SceneBackend,
  type SceneFrame,
} from "./scene";

export const MAX_MANIFEST_BYTES = 16 * 1024 * 1024;
export const MAX_REPLAY_FRAMES = 100_000;
export interface ReplayEntry {
  readonly ordinal: number;
  readonly time: number;
  readonly file: string;
  readonly bytes: number;
  readonly sha256: string;
  readonly checkpointSha256: string;
  readonly sourceBackend: SceneBackend;
}
export interface ReplayManifest {
  readonly exportBackend: SceneBackend;
  readonly frames: readonly ReplayEntry[];
}
export class ReplayFormatError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "ReplayFormatError";
  }
}
function fail(path: string, message: string): never {
  throw new ReplayFormatError(`${path}: ${message}`);
}
function object(
  value: unknown,
  path: string,
  keys: readonly string[],
): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    return fail(path, "expected an object");
  const record = value as Record<string, unknown>;
  if (
    keys.some((key) => !(key in record)) ||
    Object.keys(record).some((key) => !keys.includes(key))
  )
    return fail(path, `expected exactly ${keys.join(", ")}`);
  return record;
}
function number(value: unknown, path: string, integer = false): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < 0 ||
    (integer && !Number.isSafeInteger(value))
  )
    return fail(
      path,
      `expected a nonnegative ${integer ? "safe integer" : "finite number"}`,
    );
  return value;
}
function digest(value: unknown, path: string): string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value))
    return fail(path, "expected lowercase SHA-256");
  return value;
}
export async function sha256(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const result = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(result)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}
function reference(value: unknown, path: string): string {
  // No URI schemes, absolute paths, backslashes, percent escapes or dot segments.
  if (
    typeof value !== "string" ||
    !/^[A-Za-z0-9_-][A-Za-z0-9._-]*(\/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$/.test(
      value,
    ) ||
    !value.endsWith(".scene.json")
  )
    return fail(path, "expected a safe relative .scene.json file path");
  return value;
}
export async function parseReplayManifest(
  source: string,
): Promise<ReplayManifest> {
  if (new TextEncoder().encode(source).byteLength > MAX_MANIFEST_BYTES)
    return fail("manifest", "exceeds 16 MiB limit");
  let decoded: unknown;
  try {
    decoded = JSON.parse(source);
  } catch {
    return fail("manifest", "invalid JSON");
  }
  const root = object(decoded, "manifest", [
    "format",
    "version",
    "integrity",
    "recording",
  ]);
  if (root.format !== "microsimulator-replay" || root.version !== 1)
    return fail("manifest", "unsupported replay format/version");
  const integrity = object(root.integrity, "manifest.integrity", [
    "algorithm",
    "recording",
  ]);
  if (integrity.algorithm !== "sha256")
    return fail("manifest.integrity", "unsupported algorithm");
  const expected = digest(integrity.recording, "manifest.integrity.recording");
  let canonical: string | undefined;
  try {
    canonical = canonicalize(root.recording);
  } catch {
    return fail("manifest.recording", "cannot be canonicalized");
  }
  if (
    canonical === undefined ||
    (await sha256(new TextEncoder().encode(canonical))) !== expected
  )
    return fail("manifest.integrity", "recording digest does not match");
  const recording = object(root.recording, "manifest.recording", [
    "export_backend",
    "frames",
  ]);
  if (
    !Array.isArray(recording.frames) ||
    recording.frames.length < 1 ||
    recording.frames.length > MAX_REPLAY_FRAMES
  )
    return fail("manifest.frames", `expected 1 to ${MAX_REPLAY_FRAMES} frames`);
  let previousTime = -1;
  const references = new Set<string>();
  const frames = recording.frames.map(
    (value: unknown, ordinal): ReplayEntry => {
      const path = `frame ${ordinal}`;
      const entry = object(value, path, [
        "ordinal",
        "time",
        "file",
        "bytes",
        "sha256",
        "checkpoint_sha256",
        "source_backend",
      ]);
      if (number(entry.ordinal, `${path}.ordinal`, true) !== ordinal)
        return fail(
          path,
          "ordinals must be contiguous and match manifest order",
        );
      const time = number(entry.time, `${path}.time`);
      if (time < previousTime)
        return fail(path, "time precedes previous frame");
      previousTime = time;
      const file = reference(entry.file, `${path}.file`);
      if (references.has(file))
        return fail(path, `duplicate file reference ${file}`);
      references.add(file);
      const bytes = number(entry.bytes, `${path}.bytes`, true);
      if (bytes === 0 || bytes > MAX_SCENE_BYTES)
        return fail(path, "scene size outside permitted range");
      return {
        ordinal,
        time,
        file,
        bytes,
        sha256: digest(entry.sha256, `${path}.sha256`),
        checkpointSha256: digest(
          entry.checkpoint_sha256,
          `${path}.checkpoint_sha256`,
        ),
        sourceBackend: parseSceneBackend(
          entry.source_backend,
          `${path}.source_backend`,
        ),
      };
    },
  );
  return {
    exportBackend: parseSceneBackend(
      recording.export_backend,
      "manifest.export_backend",
    ),
    frames,
  };
}

/** FileReader cancellation prevents old dataset opens retaining large input buffers. */
function readBytes(
  file: File,
  signal: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  return new Promise((resolve, reject) => {
    signal.throwIfAborted();
    const reader = new FileReader();
    const cleanup = () => signal.removeEventListener("abort", abort);
    const abort = () => reader.abort();
    reader.onload = () => {
      cleanup();
      resolve(new Uint8Array(reader.result as ArrayBuffer));
    };
    reader.onerror = () => {
      cleanup();
      reject(reader.error ?? new Error("could not read file"));
    };
    reader.onabort = () => {
      cleanup();
      reject(new DOMException("Load canceled", "AbortError"));
    };
    signal.addEventListener("abort", abort, { once: true });
    reader.readAsArrayBuffer(file);
  });
}
export class ReplayBundle {
  public constructor(
    public readonly manifest: ReplayManifest,
    private readonly files: ReadonlyMap<string, File>,
    private readonly read: (
      file: File,
      signal: AbortSignal,
    ) => Promise<Uint8Array<ArrayBuffer>> = readBytes,
  ) {}

  public static async open(
    files: readonly File[],
    signal: AbortSignal,
  ): Promise<ReplayBundle> {
    const manifests = files.filter((file) => file.name === "manifest.json");
    if (manifests.length !== 1)
      return fail(
        "recording",
        "select one bundle folder containing exactly one manifest.json",
      );
    const manifestFile = manifests[0]!;
    if (manifestFile.size > MAX_MANIFEST_BYTES)
      return fail("manifest", "exceeds 16 MiB limit");
    const manifestPath = manifestFile.webkitRelativePath || manifestFile.name;
    const prefix = manifestPath.slice(0, -"manifest.json".length);
    const bytes = await readBytes(manifestFile, signal);
    const manifest = await parseReplayManifest(
      new TextDecoder("utf-8", { fatal: true }).decode(bytes),
    );
    const wanted = new Set(manifest.frames.map((entry) => entry.file));
    const index = new Map<string, File>();
    for (const file of files) {
      const path = file.webkitRelativePath || file.name;
      if (!path.startsWith(prefix)) continue;
      const relative = path.slice(prefix.length);
      if (!wanted.has(relative)) continue;
      if (index.has(relative))
        return fail("recording", `duplicate file ${relative}`);
      index.set(relative, file);
    }
    signal.throwIfAborted();
    return new ReplayBundle(manifest, index);
  }

  public async load(ordinal: number, signal: AbortSignal): Promise<SceneFrame> {
    const entry = this.manifest.frames[ordinal];
    if (entry === undefined)
      throw new RangeError(`frame ${ordinal} is out of range`);
    try {
      signal.throwIfAborted();
      const file = this.files.get(entry.file);
      if (file === undefined) throw new Error("missing file");
      if (file.size !== entry.bytes)
        throw new Error(
          `byte length mismatch: expected ${entry.bytes}, found ${file.size}`,
        );
      const bytes = await this.read(file, signal);
      if ((await sha256(bytes)) !== entry.sha256)
        throw new Error("scene file digest does not match");
      signal.throwIfAborted();
      const frame = await parseScene(
        new TextDecoder("utf-8", { fatal: true }).decode(bytes),
      );
      signal.throwIfAborted();
      if (frame.time !== entry.time)
        throw new Error("scene time does not match manifest");
      if (JSON.stringify(frame.backend) !== JSON.stringify(entry.sourceBackend))
        throw new Error("scene source backend does not match manifest");
      return frame;
    } catch (error) {
      if (signal.aborted) throw signal.reason;
      const detail = error instanceof Error ? error.message : String(error);
      return fail(`frame ${ordinal} (${entry.file})`, detail);
    }
  }
}
