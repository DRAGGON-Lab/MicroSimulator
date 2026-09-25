import canonicalize from "canonicalize";
import { afterEach, describe, expect, it, vi } from "vitest";

import { LiveConnection, parseLiveMessage } from "../src/live";

const FRAME = {
  backend: {
    device: "host",
    device_index: 0,
    kind: "cpu",
    name: "CPU reference",
    native: false,
  },
  cells: [],
  constraints: { boxes: [], cylinders: [], planes: [], spheres: [] },
  signal_grid: null,
  species_count: 0,
  time: 0,
};

async function scene(): Promise<Record<string, unknown>> {
  const canonical = canonicalize(FRAME);
  if (canonical === undefined) {
    throw new Error("fixture is not canonicalizable");
  }
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(canonical),
  );
  return {
    format: "microsimulator-scene",
    version: 2,
    producer: { name: "microsimulator", version: "0.1.0" },
    integrity: {
      algorithm: "sha256",
      frame: [...new Uint8Array(digest)]
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join(""),
    },
    frame: FRAME,
  };
}

describe("live viewer protocol", () => {
  it("verifies embedded scene documents", async () => {
    const message = await parseLiveMessage(
      JSON.stringify({
        type: "frame",
        revision: 2,
        completed_steps: 8,
        playing: true,
        checkpoint_enabled: false,
        scene: await scene(),
      }),
    );
    expect(message.type).toBe("frame");
    if (message.type === "frame") {
      expect(message.revision).toBe(2);
      expect(message.completedSteps).toBe(8);
      expect(message.frame.backend.kind).toBe("cpu");
    }
  });

  it("rejects unknown fields and malformed scene data", async () => {
    await expect(
      parseLiveMessage(
        JSON.stringify({ type: "error", message: "reason", source: "hidden" }),
      ),
    ).rejects.toThrow(/unknown keys.*source/);

    const document = await scene();
    (document.frame as { time: number }).time = 1;
    await expect(
      parseLiveMessage(
        JSON.stringify({
          type: "frame",
          revision: 0,
          completed_steps: 0,
          playing: false,
          checkpoint_enabled: false,
          scene: document,
        }),
      ),
    ).rejects.toThrow("frame digest does not match");
  });

  it("reads checkpoint and error results", async () => {
    await expect(
      parseLiveMessage('{"type":"checkpoint","path":"/tmp/run.cm2.json"}'),
    ).resolves.toEqual({ type: "checkpoint", path: "/tmp/run.cm2.json" });
    await expect(
      parseLiveMessage('{"type":"error","message":"not configured"}'),
    ).resolves.toEqual({ type: "error", message: "not configured" });
  });

  it("validates closed session lifecycle messages", async () => {
    await expect(
      parseLiveMessage('{"type":"session","state":"stopped"}'),
    ).resolves.toEqual({ type: "session", state: "stopped" });
    await expect(
      parseLiveMessage('{"type":"session","state":"ready"}'),
    ).rejects.toThrow("unknown session state");
    await expect(
      parseLiveMessage('{"type":"session","state":"stopped","force":true}'),
    ).rejects.toThrow("unknown keys");
  });
});

class Socket extends EventTarget {
  static readonly OPEN = 1;
  static current: Socket;
  readonly readyState = Socket.OPEN;
  readonly sent: string[] = [];

  constructor() {
    super();
    Socket.current = this;
  }

  send(value: string): void {
    this.sent.push(value);
  }
  close(): void {
    this.dispatchEvent(new Event("close"));
  }
  receive(value: unknown): void {
    this.dispatchEvent(
      new MessageEvent("message", { data: JSON.stringify(value) }),
    );
  }
}

describe("live connection lifecycle", () => {
  afterEach(() => vi.unstubAllGlobals());

  function connect() {
    vi.stubGlobal("WebSocket", Socket);
    vi.stubGlobal("window", {
      location: { href: "http://127.0.0.1:8765/", protocol: "http:" },
    });
    const callbacks = {
      state: vi.fn(),
      message: vi.fn(),
      protocolError: vi.fn(),
    };
    const connection = new LiveConnection("token", callbacks);
    connection.connect();
    Socket.current.dispatchEvent(new Event("open"));
    return { connection, callbacks, socket: Socket.current };
  }

  it("disables further work immediately after Stop and tolerates repeated Stop", () => {
    const { connection, callbacks, socket } = connect();
    connection.send({ type: "stop" });
    connection.send({ type: "stop" });
    expect(socket.sent).toEqual(['{"type":"stop"}']);
    expect(callbacks.state).toHaveBeenLastCalledWith("stopping");
    for (const type of ["play", "step", "reset", "checkpoint"] as const) {
      expect(() => connection.send({ type })).toThrow("stopping or stopped");
    }
  });

  it("drains asynchronous frame verification before treating close as intentional", async () => {
    const { callbacks, socket } = connect();
    socket.receive({
      type: "frame",
      revision: 0,
      completed_steps: 0,
      playing: false,
      checkpoint_enabled: false,
      scene: await scene(),
    });
    socket.receive({ type: "session", state: "stopping" });
    socket.receive({ type: "session", state: "stopped" });
    socket.receive({ type: "error", message: "late rejected command" });
    socket.close();
    socket.dispatchEvent(new Event("error"));
    await vi.waitFor(() => expect(callbacks.message).toHaveBeenCalledTimes(3));
    expect(callbacks.state.mock.calls.map(([state]) => state)).toEqual([
      "connecting",
      "connected",
      "stopping",
      "stopped",
    ]);
    expect(callbacks.protocolError).not.toHaveBeenCalled();
  });

  it("keeps unexpected disconnects distinguishable from completed shutdown", async () => {
    const { connection, callbacks, socket } = connect();
    connection.send({ type: "stop" });
    socket.close();
    await vi.waitFor(() =>
      expect(callbacks.state).toHaveBeenLastCalledWith("closed"),
    );
  });
});
