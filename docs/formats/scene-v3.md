# MicroSimulator scene format v3

Version 3 retains the [version 2 envelope, geometry, constraints and grid representation](scene-v2.md) and adds one required field inside the integrity-protected frame:

```json
"channel_metadata": {
  "species": ["Green reporter", "Red reporter"],
  "signals": ["Nutrient", null]
}
```

Both groups are required arrays. `species` has exactly `species_count` entries; `signals` has exactly `signal_grid.signal_count` entries, or zero when the grid is null. Each entry is a Unicode scalar string or null. Null is the canonical serialized representation of an unspecified slot; unspecified groups are expanded to null-filled arrays. Empty and whitespace-only strings are retained verbatim and use the same display fallback as null. Duplicate names are valid. Unknown metadata fields and invalid lengths or types are errors.

Scene presentation has a channel-count budget of **4096 species and 4096 signals independently**, inclusive (`MAX_SCENE_CHANNELS` in Python and TypeScript). This budget applies to both v2 and v3, including empty colonies. Readers check each claimed count before expanding missing labels, copying channel data, or constructing viewer controls; an oversized count raises a scene-format error identifying the count and budget. Python applies the same limit to `SceneFrame` construction, `capture_scene`, parsing/loading, and encoding/saving. The existing encoded-size and grid-shape checks remain separate. This presentation budget does not limit native simulation counts or checkpoint restoration, and exporters reject oversized scenes rather than silently truncating channels.

The entire frame, including channel metadata, is hashed with RFC 8785 canonical JSON and SHA-256. Digests detect corruption; they do not authenticate a publisher. Labels must be rendered as text, never interpreted as HTML or executable code.

Readers accept versions 2 and 3. They verify a version-2 frame's original digest and exact version-2 keys first, then return the current in-memory representation with null-filled channel arrays. A version-2 file containing a `channel_metadata` field is invalid, even with a matching digest. Readers reject all other versions. Writers emit only version 3.

Python `SceneFrame.channel_metadata` and TypeScript `SceneFrame.channelMetadata` expose the same ordered values. TypeScript `channelLabel(frame, "species" | "signals", index)` provides missing-label fallback and duplicate-name disambiguation. Indices identify channels; display names never identify settings or alter stored numerical values. See the [authoring guide](../models/channel-labels.md) for native, low-level and SBML examples.
