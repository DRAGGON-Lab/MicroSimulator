import {
  parseChannelLabels,
  parseScene,
  parseSignalGridLayout,
  signalGridLevelCount,
  type SceneFrame,
} from "./scene";

export type LiveConnectionState =
  "connecting" | "connected" | "stopping" | "stopped" | "closed";

export interface LiveFrameMessage {
  readonly type: "frame";
  readonly revision: number;
  readonly completedSteps: number;
  readonly playing: boolean;
  readonly checkpointEnabled: boolean;
  readonly frame: SceneFrame;
}

export interface LiveCheckpointMessage {
  readonly type: "checkpoint";
  readonly path: string;
}

export interface LiveErrorMessage {
  readonly type: "error";
  readonly message: string;
}

export interface LiveSessionMessage {
  readonly type: "session";
  readonly state: "stopping" | "stopped";
}

export type LiveMessage =
  | LiveFrameMessage
  | LiveCheckpointMessage
  | LiveErrorMessage
  | LiveSessionMessage;

export type LiveCommand =
  | Readonly<{
      type: "frame" | "play" | "pause" | "reset" | "checkpoint" | "stop";
    }>
  | Readonly<{ type: "step"; steps?: number }>;

export interface LiveCallbacks {
  readonly message: (message: LiveMessage) => void;
  readonly state: (state: LiveConnectionState) => void;
  readonly protocolError: (message: string) => void;
}

export class LiveProtocolError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "LiveProtocolError";
  }
}

function fail(path: string, message: string): never {
  throw new LiveProtocolError(`${path}: ${message}`);
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return fail(path, "expected an object");
  }
  return value as Record<string, unknown>;
}

function exactKeys(
  value: Record<string, unknown>,
  path: string,
  expected: readonly string[],
): void {
  const allowed = new Set(expected);
  const missing = expected.filter((key) => !(key in value));
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (missing.length > 0) {
    fail(path, `missing keys ${JSON.stringify(missing.toSorted())}`);
  }
  if (unknown.length > 0) {
    fail(path, `unknown keys ${JSON.stringify(unknown.toSorted())}`);
  }
}

function nonnegativeInteger(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    return fail(path, "expected a non-negative safe integer");
  }
  return value;
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") {
    return fail(path, "expected a Boolean");
  }
  return value;
}

function string(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    return fail(path, "expected a non-empty string");
  }
  return value;
}

/**
 * A text message as decoded, before it is validated. A frame whose signal grid
 * is not null awaits the binary message that carries the grid's levels.
 */
export interface LiveHeader {
  readonly message: Record<string, unknown>;
  readonly awaitsLevels: boolean;
}

export function readLiveHeader(encoded: string): LiveHeader {
  let value: unknown;
  try {
    value = JSON.parse(encoded) as unknown;
  } catch {
    return fail("$", "message is not valid JSON");
  }
  const message = record(value, "$");
  return {
    message,
    awaitsLevels:
      message.type === "frame" &&
      message.signal_grid !== null &&
      message.signal_grid !== undefined,
  };
}

function hex(bytes: ArrayBuffer): string {
  return [...new Uint8Array(bytes)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

/** Little-endian float32 bytes as numbers, on a host of either byte order. */
function float32LittleEndian(bytes: ArrayBuffer): Float32Array {
  const hostIsLittleEndian =
    new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
  if (hostIsLittleEndian) {
    return new Float32Array(bytes);
  }
  const view = new DataView(bytes);
  const values = new Float32Array(bytes.byteLength / 4);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = view.getFloat32(index * 4, true);
  }
  return values;
}

/**
 * The frame a header and its levels describe: the header's scene, with the
 * signal grid the header describes holding the levels that came as bytes.
 */
async function liveFrame(
  message: Record<string, unknown>,
  levels: ArrayBuffer | undefined,
): Promise<SceneFrame> {
  const scene = await parseScene(JSON.stringify(message.scene));
  if (message.signal_grid === null) {
    if (levels !== undefined) {
      return fail("$.signal_grid", "levels arrived for a frame without a grid");
    }
    return scene;
  }
  const path = "$.signal_grid";
  const described = record(message.signal_grid, path);
  exactKeys(described, path, [
    "signal_count",
    "shape",
    "origin",
    "spacing",
    "boundaries",
    "labels",
    "levels",
  ]);
  if (scene.signalGrid !== null) {
    return fail(path, "the frame's scene already holds a signal grid");
  }
  const layout = parseSignalGridLayout(described, path);
  const labels = parseChannelLabels(
    described.labels,
    `${path}.labels`,
    layout.signalCount,
  );
  const stated = record(described.levels, `${path}.levels`);
  exactKeys(stated, `${path}.levels`, ["encoding", "byte_length", "sha256"]);
  if (stated.encoding !== "float32-le") {
    return fail(`${path}.levels.encoding`, "unsupported level encoding");
  }
  const byteLength = nonnegativeInteger(
    stated.byte_length,
    `${path}.levels.byte_length`,
  );
  const digest = string(stated.sha256, `${path}.levels.sha256`);
  if (byteLength !== signalGridLevelCount(layout) * 4) {
    return fail(
      `${path}.levels.byte_length`,
      `expected ${signalGridLevelCount(layout) * 4} bytes for the grid`,
    );
  }
  if (levels === undefined) {
    return fail(`${path}.levels`, "the frame's levels did not arrive");
  }
  if (levels.byteLength !== byteLength) {
    return fail(
      `${path}.levels`,
      `expected ${byteLength} bytes, got ${levels.byteLength}`,
    );
  }
  if (hex(await crypto.subtle.digest("SHA-256", levels)) !== digest) {
    return fail(`${path}.levels`, "level digest does not match");
  }
  const values = float32LittleEndian(levels);
  if (!values.every((value) => Number.isFinite(value))) {
    return fail(`${path}.levels`, "expected finite levels");
  }
  return {
    ...scene,
    channelMetadata: {
      species: scene.channelMetadata.species,
      signals: labels,
    },
    signalGrid: { ...layout, levels: values },
  };
}

/**
 * Validate one server message. `levels` is the binary message that follows a
 * frame whose signal grid is not null.
 */
export async function parseLiveMessage(
  encoded: string,
  levels?: ArrayBuffer,
): Promise<LiveMessage> {
  return completeLiveMessage(readLiveHeader(encoded), levels);
}

export async function completeLiveMessage(
  header: LiveHeader,
  levels?: ArrayBuffer,
): Promise<LiveMessage> {
  const message = header.message;
  const type = message.type;
  if (type === "frame") {
    exactKeys(message, "$", [
      "type",
      "revision",
      "completed_steps",
      "playing",
      "checkpoint_enabled",
      "scene",
      "signal_grid",
    ]);
    return {
      type,
      revision: nonnegativeInteger(message.revision, "$.revision"),
      completedSteps: nonnegativeInteger(
        message.completed_steps,
        "$.completed_steps",
      ),
      playing: boolean(message.playing, "$.playing"),
      checkpointEnabled: boolean(
        message.checkpoint_enabled,
        "$.checkpoint_enabled",
      ),
      frame: await liveFrame(message, levels),
    };
  }
  if (levels !== undefined) {
    return fail("$", "levels arrived for a message that is not a frame");
  }
  if (type === "checkpoint") {
    exactKeys(message, "$", ["type", "path"]);
    return { type, path: string(message.path, "$.path") };
  }
  if (type === "error") {
    exactKeys(message, "$", ["type", "message"]);
    return { type, message: string(message.message, "$.message") };
  }
  if (type === "session") {
    exactKeys(message, "$", ["type", "state"]);
    if (message.state !== "stopping" && message.state !== "stopped") {
      return fail("$.state", "unknown session state");
    }
    return { type, state: message.state };
  }
  return fail("$.type", "unknown message type");
}

export class LiveConnection {
  private socket: WebSocket | null = null;
  private messageQueue = Promise.resolve();
  private stopping = false;
  private stopped = false;
  // A frame header whose levels have not yet arrived.
  private awaitingLevels: LiveHeader | null = null;

  public constructor(
    private readonly token: string,
    private readonly callbacks: LiveCallbacks,
  ) {
    if (token.length === 0) {
      throw new LiveProtocolError("live session token is empty");
    }
  }

  public connect(): void {
    if (this.socket !== null) {
      throw new LiveProtocolError("live connection already started");
    }
    const url = new URL("/api/v2/session", window.location.href);
    url.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    url.searchParams.set("token", this.token);
    this.callbacks.state("connecting");
    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    socket.addEventListener("open", () => this.callbacks.state("connected"));
    socket.addEventListener("message", (event) => {
      // A frame with a signal grid is a text header followed by one binary
      // message of levels. Other text messages may arrive between the two; a
      // later header replaces one whose levels never came.
      const data: unknown = event.data;
      let header: LiveHeader;
      let levels: ArrayBuffer | undefined;
      if (typeof data === "string") {
        try {
          header = readLiveHeader(data);
        } catch (error) {
          this.callbacks.protocolError(
            error instanceof Error ? error.message : String(error),
          );
          return;
        }
        if (header.awaitsLevels) {
          this.awaitingLevels = header;
          return;
        }
      } else if (data instanceof ArrayBuffer && this.awaitingLevels !== null) {
        header = this.awaitingLevels;
        levels = data;
        this.awaitingLevels = null;
      } else {
        this.callbacks.protocolError(
          "live server sent levels that no frame announced",
        );
        return;
      }
      this.messageQueue = this.messageQueue
        .then(async () => {
          if (this.stopped) return;
          const message = await completeLiveMessage(header, levels);
          if (message.type === "session") {
            this.stopping = true;
            this.stopped = message.state === "stopped";
            this.callbacks.state(message.state);
          }
          this.callbacks.message(message);
        })
        .catch((error: unknown) => {
          this.callbacks.protocolError(
            error instanceof Error ? error.message : String(error),
          );
        });
    });
    socket.addEventListener("error", () => {
      void this.messageQueue.then(() => {
        if (!this.stopped)
          this.callbacks.protocolError("live connection failed");
      });
    });
    socket.addEventListener("close", () => {
      // Scene digest verification is asynchronous. Drain messages before close,
      // otherwise a valid stopped notification can race with disconnected UI.
      void this.messageQueue.then(() => {
        if (!this.stopped) this.callbacks.state("closed");
      });
    });
  }

  public send(command: LiveCommand): void {
    if (this.stopping) {
      if (command.type === "stop") return;
      throw new LiveProtocolError("live session is stopping or stopped");
    }
    if (this.socket?.readyState !== WebSocket.OPEN) {
      throw new LiveProtocolError("live connection is not ready");
    }
    this.socket.send(JSON.stringify(command));
    if (command.type === "stop") {
      this.stopping = true;
      this.callbacks.state("stopping");
    }
  }

  public close(): void {
    this.socket?.close(1000, "viewer closed");
  }
}
