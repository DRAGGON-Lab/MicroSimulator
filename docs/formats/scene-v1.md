# MicroSimulator scene format v1

The scene format is an immutable presentation snapshot. It is safe data for a viewer, not a simulation checkpoint: it has no equations, solver settings, controller state, Python source, or resume authority.

## Envelope

A document is UTF-8 JSON with these root fields:

```json
{
  "format": "microsimulator-scene",
  "version": 1,
  "producer": { "name": "microsimulator", "version": "0.1.0" },
  "integrity": { "algorithm": "sha256", "frame": "..." },
  "frame": {}
}
```

The digest is SHA-256 over the frame encoded with the RFC 8785 JSON Canonicalization Scheme (JCS). This gives Python and browser readers the same object-key, string, and ECMAScript-number representation. Readers reject missing, duplicate, and unknown fields. The maximum encoded size is 1 GiB.

## Frame

`frame` contains:

- `time`: non-negative simulation time;
- `backend`: source kind, backend name, device name, zero-based device index, and whether the backend is native;
- `species_count`: the fixed number of per-cell species values;
- `cells`: rods in compact slot order; and
- `signal_grid`: a scalar grid or `null`.

Each cell records its stable `id`, optional lineage `parent_id`, compact `slot`, position, normalized direction, cylindrical length, radius, growth rate, cell type, fixed state, and ordered species values. IDs are canonical positive decimal strings because the engine's unsigned 64-bit identity range exceeds JavaScript's exact integer range. Slots and other bounded integers remain JSON numbers.

## Signal grid

A grid records `signal_count`, three-dimensional `shape`, `origin`, `spacing`, six typed boundaries, and flattened `levels`. Fixed boundaries carry one value per signal; no-flux and periodic boundaries carry an empty value array.

Levels are channel-major. Within a channel, `z` varies fastest, then `y`, then `x`. For shape `(X, Y, Z)`, the offset is:

```text
signal * X * Y * Z + x * Y * Z + y * Z + z
```

The scene preserves all channels. A viewer chooses a channel and slice as presentation state; it does not mutate the frame.

## Compatibility

Writers always emit the current version. Readers accept version 1 exactly and fail closed on later versions until an explicit migration is defined. Backend conformance compares frame semantics while ignoring the expected backend identity fields. Pixel output is tested separately by the viewer.
