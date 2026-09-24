# Live viewer protocol v1

The live viewer protocol connects the independent browser renderer to one Python-owned simulation session. It does not expose simulation objects, Python evaluation, model loading, or client-selected filesystem paths.

## Authority boundary

The server binds to loopback by default and creates a random bearer token for each process. A WebSocket upgrade at `/api/v1/session` succeeds only when its `token` query value matches that bearer token and its `Origin` exactly matches the HTTP server origin. Commands are limited to 4096 UTF-8 bytes. The server does not enable CORS.

Static viewer assets have no engine authority. Possession of the per-process token grants control of that process, so callers must not publish or persist the live URL.

## Server messages

Every new connection receives a `frame` message. The server also broadcasts a frame after a state-changing command and after every configured playback batch:

```json
{
  "type": "frame",
  "revision": 4,
  "completed_steps": 12,
  "playing": false,
  "checkpoint_enabled": true,
  "scene": {}
}
```

`scene` is a complete scene document at the current scene format version, including its RFC 8785 SHA-256 integrity value. `revision` changes once per step batch or reset. The browser must verify and validate each document as it would a scene file.

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

`stopping` means admission of new work has ended. `stopped` means the active operation has finished and the simulation worker has terminated. Clients should process previously received messages (including asynchronous scene verification) before interpreting the subsequent socket close. A close without `stopped` is still an unexpected disconnect. These messages extend the v1 vocabulary; use a viewer built from the same release as the server.

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

Reset calls the original server-side model factory again with its original backend, device, seed, parameters, and resume source. Checkpoint writes only to the destination configured when the server starts and atomically replaces that file. It preserves controller state for any runnable model implementing the `SimulationController` protocol, including the legacy compatibility adapter.

## Stop and restart

Stop ends this server process and releases its listening port. It is authenticated through the same token and exact-origin WebSocket upgrade as every other command. The reader handles Stop immediately, including while an earlier command on that same socket is executing. Ordinary commands retain per-client ordering in a bounded queue of 32; additional queued commands receive an error rather than blocking Stop.

Shutdown is cooperative: an in-progress individual simulation step finishes, then the rest of its batch is skipped. An already-running reset, scene capture, or atomic checkpoint write also finishes. There is no timeout that kills the worker in the middle of model state mutation or file replacement. A model operation that never returns will therefore keep the session in `stopping`. Queued operations and newly received Frame, Play, Step, Pause, Reset, and Checkpoint commands are rejected once stopping begins; repeated Stop requests are idempotent. Closing the final browser connection still pauses the session and allows reconnection.

Browser Stop and terminal Ctrl+C share the same worker/socket cleanup path. After `stopped`, the server closes client sockets and its application runner, exits, and releases the port. Launch the next `microsimulator view` command from the terminal and open its newly printed URL; each process has a new token. The old browser keeps its last rendered frame and displays Stopped.
