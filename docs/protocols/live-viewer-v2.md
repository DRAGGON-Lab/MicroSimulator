# Live viewer protocol v2

The live viewer protocol connects the independent browser renderer to one Python-owned simulation session. It does not expose simulation objects, Python evaluation, model loading, or client-selected filesystem paths.

## Authority boundary

The server binds to loopback by default and creates a random bearer token for each process. A WebSocket upgrade at `/api/v2/session` succeeds only when its `token` query value matches that bearer token and its `Origin` exactly matches the HTTP server origin. Commands are limited to 4096 UTF-8 bytes. The server does not enable CORS.

Static viewer assets have no engine authority. Possession of the per-process token grants control of that process, so callers must not publish or persist the live URL.

## Server messages

Every new connection receives a frame. The server also broadcasts a frame after a state-changing command and after every configured playback batch. A frame is a text message, its header, followed by one binary message when the model has a signal grid:

```json
{
  "type": "frame",
  "revision": 4,
  "completed_steps": 12,
  "playing": false,
  "checkpoint_enabled": true,
  "scene": {},
  "signal_grid": {
    "signal_count": 2,
    "shape": [64, 48, 3],
    "origin": [0.0, 0.0, 0.0],
    "spacing": [1.0, 1.0, 1.0],
    "boundaries": {},
    "labels": ["Nutrient", null],
    "levels": { "encoding": "float32-le", "byte_length": 73728, "sha256": "…" }
  }
}
```

`scene` is a complete scene document at the current scene format version, including its RFC 8785 SHA-256 integrity value, whose own `signal_grid` is null and whose signal labels are empty. `revision` changes once per step batch or reset. The browser must verify and validate the document as it would a scene file.

The header's `signal_grid` describes the model's signal grid, or is null when the model has none. Its `signal_count`, `shape`, `origin`, `spacing` and `boundaries` are as a scene states them, and `labels` holds one string or null per signal. The grid's levels are not in the header. They are the binary message that follows it: `byte_length` bytes of little-endian IEEE 754 binary32, one value per signal and site in the scene's order, channel-major with `z` varying fastest, then `y`, then `x`. These are the engine's own values, so a level reads the same as in a scene file. `sha256` is the lowercase hexadecimal SHA-256 of those bytes; the browser must check the length against the grid and the digest against the bytes, and reject non-finite values, before showing the frame. A frame whose `signal_grid` is null has no binary message.

A client may append `velocity=1` to its WebSocket query. When the simulation has a staggered velocity field, the header then includes `signal_grid.velocity_faces` with the same encoding, byte length and digest fields as `levels`. Its bytes follow the levels **in the same binary message**. The block holds x, y and z face arrays, each with z varying fastest, of shapes (X+1,Y,Z), (X,Y+1,Z) and (X,Y,Z+1). Each block has its own digest. Cells, levels and velocities come from one capture. A client that omits the query receives the original levels-only frame.

Levels travel as bytes because they are most of a frame: a grid of a quarter of a million sites and four signals is 3 MB of binary32 and about 13 MB as decimal text, which the server would encode and the browser parse, canonicalize and hash for every frame.

No other frame's messages come between a header and its levels on one connection, but a `checkpoint`, `error` or `session` message may. A client pairs a binary message with the latest header that announced levels, and treats a binary message that no header announced as a protocol error. A header whose levels never arrive, because delivery to a slow receiver timed out, is superseded by the next header.

A successful checkpoint command returns the server-configured absolute path:

```json
{ "type": "checkpoint", "path": "/configured/path/run.json" }
```

Rejected commands and model failures return a data-only error:

```json
{ "type": "error", "message": "reason" }
```

Intentional shutdown sends lifecycle notifications before closing the WebSocket with code 1000:

```json
{"type":"session","state":"stopping"}
{"type":"session","state":"stopped"}
```

`stopping` means admission of new work has ended. `stopped` means the active operation has finished and the simulation worker has terminated. Clients should process previously received messages (including asynchronous scene verification) before interpreting the subsequent socket close. A close without `stopped` is still an unexpected disconnect. Use a viewer built from the same release as the server.

## Client commands

The vocabulary is closed. Unknown fields are rejected.

```json
{"type":"frame"}
{"type":"step","steps":1}
{"type":"play"}
{"type":"pause"}
{"type":"reset"}
{"type":"checkpoint"}
{"type":"stop"}
```

`steps` defaults to one and is bounded to 1 through 10,000. Playback advances the configured number of steps per published frame. A step or reset first pauses playback. Disconnecting the final client pauses the simulation.

Reset calls the original server-side model factory again with its original backend, device, seed, parameters, and resume source. Checkpoint writes only to the destination configured when the server starts and atomically replaces that file. It preserves controller state for any runnable model implementing the `SimulationController` protocol.

## Stop and restart

Stop ends this server process and releases its listening port. It is authenticated through the same token and exact-origin WebSocket upgrade as every other command. The reader handles Stop immediately, including while an earlier command on that same socket is executing. Ordinary commands retain per-client ordering in a bounded queue of 32; additional queued commands receive an error rather than blocking Stop.

Shutdown is cooperative: an in-progress individual simulation step finishes, then the rest of its batch is skipped. An already-running reset, scene capture, or atomic checkpoint write also finishes. There is no timeout that kills the worker in the middle of model state mutation or file replacement. A model operation that never returns will therefore keep the session in `stopping`. Queued operations and newly received Frame, Play, Step, Pause, Reset, and Checkpoint commands are rejected once stopping begins; repeated Stop requests are idempotent. Closing the final browser connection still pauses the session and allows reconnection.

Browser Stop and terminal Ctrl+C share the same worker/socket cleanup path. After `stopped`, the server closes client sockets and its application runner, exits, and releases the port. Launch the next `microsimulator view` command from the terminal and open its newly printed URL; each process has a new token. The old browser keeps its last rendered frame and displays Stopped.

Network delivery has separate deadlines from cooperative model work. Initial frame writes, broadcasts, and lifecycle notifications allow one second per receiver; independent lifecycle deliveries run concurrently. The complete WebSocket close operation, including writing and draining its close frame, also has a one-second deadline. An unresponsive connection is then aborted, discarding queued network bytes so neither its initial-send handler nor the application runner waits indefinitely for a reader. Responsive clients still receive `stopping`, `stopped`, and a normal code-1000 close. A stalled client may miss these notifications and observe an unexpected disconnect; this does not cancel an active simulation operation or interrupt an atomic checkpoint write.
