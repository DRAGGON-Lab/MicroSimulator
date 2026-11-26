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

const LEVELS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5];

/** A frame whose signal grid, two signals on 2 by 1 by 3 sites, travels as bytes. */
async function gridFrame(): Promise<{
  header: Record<string, unknown>;
  bytes: ArrayBuffer;
}> {
  const bytes = new Float32Array(LEVELS).buffer;
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const open = { kind: "no_flux", values: [] };
  return {
    bytes,
    header: {
      type: "frame",
      revision: 3,
      completed_steps: 5,
      playing: false,
      checkpoint_enabled: false,
      scene: await scene(),
      signal_grid: {
        signal_count: 2,
        shape: [2, 1, 3],
        origin: [0, 0, 0],
        spacing: [1, 2, 0.5],
        boundaries: {
          x_lower: { kind: "fixed", values: [1, 0] },
          x_upper: open,
          y_lower: open,
          y_upper: open,
          z_lower: open,
          z_upper: open,
        },
        labels: ["Nutrient", null],
        levels: {
          encoding: "float32-le",
          byte_length: bytes.byteLength,
          sha256: [...new Uint8Array(digest)]
            .map((byte) => byte.toString(16).padStart(2, "0"))
            .join(""),
        },
      },
    },
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
        signal_grid: null,
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
          signal_grid: null,
        }),
      ),
    ).rejects.toThrow("frame digest does not match");
  });

  it("reads checkpoint and error results", async () => {
    await expect(
      parseLiveMessage('{"type":"checkpoint","path":"/tmp/run.json"}'),
    ).resolves.toEqual({ type: "checkpoint", path: "/tmp/run.json" });
    await expect(
      parseLiveMessage('{"type":"error","message":"not configured"}'),
    ).resolves.toEqual({ type: "error", message: "not configured" });
  });

  it("joins a frame's binary levels to the grid its header describes", async () => {
    const { header, bytes } = await gridFrame();
    const message = await parseLiveMessage(JSON.stringify(header), bytes);
    expect(message.type).toBe("frame");
    if (message.type !== "frame") return;
    const grid = message.frame.signalGrid;
    expect(grid).not.toBeNull();
    expect(grid?.signalCount).toBe(2);
    expect(grid?.shape).toEqual([2, 1, 3]);
    expect(grid?.spacing).toEqual([1, 2, 0.5]);
    expect(grid?.boundaries.xLower).toEqual({ kind: "fixed", values: [1, 0] });
    expect(Array.from(grid?.levels ?? [])).toEqual(LEVELS);
    expect(message.frame.channelMetadata.signals).toEqual(["Nutrient", null]);
  });

  it("verifies and splits staggered face data after the signal levels", async () => {
    const { header, bytes } = await gridFrame();
    const faces = Float32Array.from({ length: 9 + 12 + 8 }, (_, i) => i - 5);
    const hash = await crypto.subtle.digest("SHA-256", faces.buffer);
    Object.assign(header.signal_grid as object, {
      velocity_faces: {
        encoding: "float32-le",
        byte_length: faces.byteLength,
        sha256: Array.from(new Uint8Array(hash), (b) =>
          b.toString(16).padStart(2, "0"),
        ).join(""),
      },
    });
    const packet = new Uint8Array(bytes.byteLength + faces.byteLength);
    packet.set(new Uint8Array(bytes));
    packet.set(new Uint8Array(faces.buffer), bytes.byteLength);
    const encoded = JSON.stringify(header);
    const message = await parseLiveMessage(encoded, packet.buffer);
    if (message.type !== "frame") throw Error("expected frame");
    const grid = message.frame.signalGrid;
    expect(Array.from(grid!.levels)).toEqual(LEVELS);
    expect(grid!.velocityFaces!.map((a) => [a.length, a[0]])).toEqual([
      [9, -5],
      [12, 4],
      [8, 16],
    ]);
    packet[packet.byteLength - 1] = 255;
    await expect(parseLiveMessage(encoded, packet.buffer)).rejects.toThrow(
      "velocity digest",
    );
  });

  it("refuses levels that are missing, short, altered or unannounced", async () => {
    const { header, bytes } = await gridFrame();
    const encoded = JSON.stringify(header);
    await expect(parseLiveMessage(encoded)).rejects.toThrow(
      "levels did not arrive",
    );
    await expect(parseLiveMessage(encoded, bytes.slice(4))).rejects.toThrow(
      "expected 48 bytes, got 44",
    );
    const altered = bytes.slice(0);
    new Float32Array(altered)[0] = 9;
    await expect(parseLiveMessage(encoded, altered)).rejects.toThrow(
      "level digest does not match",
    );
    const grid = header.signal_grid as { levels: { byte_length: number } };
    grid.levels.byte_length = 44;
    await expect(
      parseLiveMessage(JSON.stringify(header), bytes),
    ).rejects.toThrow("expected 48 bytes for the grid");

    const plain = JSON.stringify({
      type: "frame",
      revision: 0,
      completed_steps: 0,
      playing: false,
      checkpoint_enabled: false,
      scene: await scene(),
      signal_grid: null,
    });
    await expect(parseLiveMessage(plain, bytes)).rejects.toThrow(
      "levels arrived for a frame without a grid",
    );
    await expect(
      parseLiveMessage('{"type":"error","message":"reason"}', bytes),
    ).rejects.toThrow("not a frame");
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
  receiveBytes(value: ArrayBuffer): void {
    this.dispatchEvent(new MessageEvent("message", { data: value }));
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
      signal_grid: null,
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

  it("delivers a frame once its levels arrive, and other messages meanwhile", async () => {
    const { callbacks, socket } = connect();
    const { header, bytes } = await gridFrame();
    socket.receive(header);
    socket.receive({ type: "error", message: "between header and levels" });
    await vi.waitFor(() => expect(callbacks.message).toHaveBeenCalledTimes(1));
    expect(callbacks.message.mock.calls[0]?.[0]).toEqual({
      type: "error",
      message: "between header and levels",
    });
    socket.receiveBytes(bytes);
    await vi.waitFor(() => expect(callbacks.message).toHaveBeenCalledTimes(2));
    const frame = callbacks.message.mock.calls[1]?.[0] as {
      type: string;
      revision: number;
      frame: { signalGrid: { levels: ArrayLike<number> } };
    };
    expect(frame.type).toBe("frame");
    expect(frame.revision).toBe(3);
    expect(Array.from(frame.frame.signalGrid.levels)).toEqual(LEVELS);
    expect(callbacks.protocolError).not.toHaveBeenCalled();
  });

  it("reports levels no frame announced, and drops a header whose levels never came", async () => {
    const { callbacks, socket } = connect();
    const { header, bytes } = await gridFrame();
    socket.receiveBytes(bytes);
    expect(callbacks.protocolError).toHaveBeenCalledWith(
      "live server sent levels that no frame announced",
    );
    callbacks.protocolError.mockClear();

    // A second header replaces the first, whose levels never came.
    socket.receive({ ...header, revision: 1 });
    socket.receive({ ...header, revision: 2 });
    socket.receiveBytes(bytes);
    await vi.waitFor(() => expect(callbacks.message).toHaveBeenCalledTimes(1));
    expect(
      (callbacks.message.mock.calls[0]?.[0] as { revision: number }).revision,
    ).toBe(2);
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
