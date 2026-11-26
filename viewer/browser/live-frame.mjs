// A live frame as the server sends it: a text header whose scene holds no
// signal grid, then the grid's levels as one binary message.
import { createHash } from "node:crypto";

/**
 * Split a scene frame into what a live frame carries: the frame without its
 * signal grid, the grid as the header describes it, and its levels as
 * little-endian float32 bytes. A frame without a grid has neither of the last
 * two.
 */
export function liveParts(frame) {
  const grid = frame.signal_grid ?? null;
  if (grid === null) {
    return { frame, signalGrid: null, levels: null };
  }
  const { levels, ...layout } = grid;
  const bytes = Buffer.from(new Float32Array(levels).buffer);
  const metadata = frame.channel_metadata;
  return {
    frame: {
      ...frame,
      signal_grid: null,
      ...(metadata === undefined
        ? {}
        : { channel_metadata: { ...metadata, signals: [] } }),
    },
    signalGrid: {
      ...layout,
      labels: metadata?.signals ?? Array(grid.signal_count).fill(null),
      levels: {
        encoding: "float32-le",
        byte_length: bytes.byteLength,
        sha256: createHash("sha256").update(bytes).digest("hex"),
      },
    },
    levels: bytes,
  };
}

/**
 * Send one live frame on a routed WebSocket. `envelope` holds the header's
 * fields apart from the scene and grid, and `document(frame)` wraps a scene
 * frame in its scene document.
 */
export function sendLiveFrame(socket, envelope, frame, document) {
  const parts = liveParts(frame);
  socket.send(
    JSON.stringify({
      type: "frame",
      ...envelope,
      scene: document(parts.frame),
      signal_grid: parts.signalGrid,
    }),
  );
  if (parts.levels !== null) {
    socket.send(parts.levels);
  }
}
