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

## Client commands

The vocabulary is closed. Unknown fields are rejected.

```json
{"type":"frame"}
{"type":"step","steps":1}
{"type":"play"}
{"type":"pause"}
{"type":"reset"}
{"type":"checkpoint"}
```

`steps` defaults to one and is bounded to 1 through 10,000. Playback advances the configured number of steps per published frame. A step or reset first pauses playback. Disconnecting the final client pauses the simulation.

Reset calls the original server-side model factory again with its original backend, device, seed, parameters, and resume source. Checkpoint writes only to the destination configured when the server starts and atomically replaces that file. It preserves controller state for any runnable model implementing the `SimulationController` protocol, including the legacy compatibility adapter.
