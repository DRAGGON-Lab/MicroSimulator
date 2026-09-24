import { parseScene, type SceneFrame } from "./scene";

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

export async function parseLiveMessage(encoded: string): Promise<LiveMessage> {
  let value: unknown;
  try {
    value = JSON.parse(encoded) as unknown;
  } catch {
    return fail("$", "message is not valid JSON");
  }
  const message = record(value, "$");
  const type = message.type;
  if (type === "frame") {
    exactKeys(message, "$", [
      "type",
      "revision",
      "completed_steps",
      "playing",
      "checkpoint_enabled",
      "scene",
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
      frame: await parseScene(JSON.stringify(message.scene)),
    };
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
    const url = new URL("/api/v1/session", window.location.href);
    url.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    url.searchParams.set("token", this.token);
    this.callbacks.state("connecting");
    const socket = new WebSocket(url);
    this.socket = socket;
    socket.addEventListener("open", () => this.callbacks.state("connected"));
    socket.addEventListener("message", (event) => {
      const encoded = event.data;
      if (typeof encoded !== "string") {
        this.callbacks.protocolError("live server sent a non-text message");
        return;
      }
      this.messageQueue = this.messageQueue
        .then(async () => {
          if (this.stopped) return;
          const message = await parseLiveMessage(encoded);
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
