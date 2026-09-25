import canonicalize from "canonicalize";
import { describe, expect, it } from "vitest";
import {
  MAX_MANIFEST_BYTES,
  ReplayBundle,
  parseReplayManifest,
  sha256,
} from "../src/replay-bundle";
import source from "./fixtures/channels-v3.scene.json?raw";

const backend = {
  kind: "cpu",
  name: "cpu-reference",
  device: "host",
  device_index: 0,
  native: true,
};
const bytes = new TextEncoder().encode(source);
async function document() {
  return {
    format: "microsimulator-replay",
    version: 1,
    integrity: { algorithm: "sha256", recording: "" },
    recording: {
      export_backend: backend,
      frames: [
        {
          ordinal: 0,
          time: 0,
          file: "frames/z-first.scene.json",
          bytes: bytes.length,
          sha256: await sha256(bytes),
          checkpoint_sha256: "a".repeat(64),
          source_backend: backend,
        },
        {
          ordinal: 1,
          time: 0,
          file: "frames/a-second.scene.json",
          bytes: bytes.length,
          sha256: await sha256(bytes),
          checkpoint_sha256: "b".repeat(64),
          source_backend: backend,
        },
      ],
    },
  };
}
async function sign(value: Awaited<ReturnType<typeof document>>) {
  value.integrity.recording = await sha256(
    new TextEncoder().encode(canonicalize(value.recording)),
  );
  return JSON.stringify(value);
}
const read = async (file: File) => new Uint8Array(await file.arrayBuffer());

describe("replay manifest validation", () => {
  it("uses manifest order and permits distinct ordinals at equal times", async () => {
    const manifest = await parseReplayManifest(await sign(await document()));
    expect(manifest.frames.map((entry) => entry.file)).toEqual([
      "frames/z-first.scene.json",
      "frames/a-second.scene.json",
    ]);
    expect(manifest.frames.map((entry) => entry.time)).toEqual([0, 0]);
  });
  it.each([
    "/root.scene.json",
    "../evil.scene.json",
    "frames/../evil.scene.json",
    "frames/%2e%2e/evil.scene.json",
    "https://example.com/a.scene.json",
    "C:\\a.scene.json",
    "frames\\a.scene.json",
    "./a.scene.json",
  ])("rejects unsafe path %s", async (path) => {
    const value = await document();
    value.recording.frames[0]!.file = path;
    await expect(parseReplayManifest(await sign(value))).rejects.toThrow(
      "safe relative",
    );
  });
  it("rejects duplicate references, ordinals and decreasing timestamps", async () => {
    const duplicate = await document();
    duplicate.recording.frames[1]!.file = duplicate.recording.frames[0]!.file;
    await expect(parseReplayManifest(await sign(duplicate))).rejects.toThrow(
      "duplicate file",
    );
    const ordinal = await document();
    ordinal.recording.frames[1]!.ordinal = 0;
    await expect(parseReplayManifest(await sign(ordinal))).rejects.toThrow(
      "ordinals",
    );
    const time = await document();
    time.recording.frames[0]!.time = 1;
    await expect(parseReplayManifest(await sign(time))).rejects.toThrow(
      "precedes",
    );
  });
  it("rejects unsupported, damaged, over-limit and empty manifests", async () => {
    const invalid = await document();
    const encoded = await sign(invalid);
    invalid.recording.frames[0]!.time = 1;
    await expect(parseReplayManifest(JSON.stringify(invalid))).rejects.toThrow(
      "digest",
    );
    await expect(
      parseReplayManifest(encoded.replace('"version":1', '"version":2')),
    ).rejects.toThrow("version");
    await expect(
      parseReplayManifest(" ".repeat(MAX_MANIFEST_BYTES + 1)),
    ).rejects.toThrow("limit");
    const empty = await document();
    empty.recording.frames = [];
    await expect(parseReplayManifest(await sign(empty))).rejects.toThrow(
      "expected 1",
    );
  });
});

describe("on-demand scene loading", () => {
  it("verifies exact file bytes and existing scene integrity, including source identity", async () => {
    const manifest = await parseReplayManifest(await sign(await document()));
    let reads = 0;
    const files = new Map([
      [manifest.frames[0]!.file, new File([source], "first.scene.json")],
    ]);
    const bundle = new ReplayBundle(manifest, files, async (file) => {
      reads++;
      return read(file);
    });
    expect(reads).toBe(0);
    expect(
      (await bundle.load(0, new AbortController().signal)).channelMetadata
        .species[0],
    ).toContain("α");
    expect(reads).toBe(1);
    await expect(bundle.load(1, new AbortController().signal)).rejects.toThrow(
      "frame 1 (frames/a-second.scene.json): missing file",
    );
  });
  it("attributes size and digest corruption to a specific frame", async () => {
    const manifest = await parseReplayManifest(await sign(await document()));
    const path = manifest.frames[0]!.file;
    const short = new ReplayBundle(
      manifest,
      new Map([[path, new File(["bad"], "bad.scene.json")]]),
      read,
    );
    await expect(short.load(0, new AbortController().signal)).rejects.toThrow(
      "frame 0",
    );
    const corrupt = source.replace(
      "cpu-reference",
      "cpu-reference".toUpperCase(),
    );
    const damaged = new ReplayBundle(
      manifest,
      new Map([[path, new File([corrupt], "bad.scene.json")]]),
      read,
    );
    await expect(damaged.load(0, new AbortController().signal)).rejects.toThrow(
      "scene file digest",
    );
  });
  it("rejects scene time/backend mismatch even when scene and file digests are valid", async () => {
    const value = await document();
    value.recording.frames = value.recording.frames.slice(0, 1);
    value.recording.frames[0]!.time = 1;
    const manifest = await parseReplayManifest(await sign(value));
    const bundle = new ReplayBundle(
      manifest,
      new Map([
        [manifest.frames[0]!.file, new File([source], "frame.scene.json")],
      ]),
      read,
    );
    await expect(bundle.load(0, new AbortController().signal)).rejects.toThrow(
      "scene time does not match",
    );
    value.recording.frames[0]!.time = 0;
    value.recording.frames[0]!.source_backend = {
      ...backend,
      device: "different",
    };
    const mismatch = await parseReplayManifest(await sign(value));
    await expect(
      new ReplayBundle(
        mismatch,
        new Map([
          [mismatch.frames[0]!.file, new File([source], "frame.scene.json")],
        ]),
        read,
      ).load(0, new AbortController().signal),
    ).rejects.toThrow("source backend");
  });
});
