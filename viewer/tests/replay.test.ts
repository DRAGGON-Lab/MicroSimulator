import { afterEach, describe, expect, it, vi } from "vitest";
import { parseScene, type SceneFrame } from "../src/scene";
import {
  ReplayCache,
  ReplayController,
  frameWeight,
  type FrameLoader,
  type ReplayState,
} from "../src/replay";
import source from "./fixtures/channels-v3.scene.json?raw";

const flush = async () => {
  for (let index = 0; index < 20; index++) await Promise.resolve();
};
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
function player(loader: FrameLoader, count = 5) {
  const frames: number[] = [];
  const states: ReplayState[] = [];
  const controller = new ReplayController(count, loader, {
    frame: (_, ordinal) => frames.push(ordinal),
    state: (state) => states.push(state),
  });
  return { controller, frames, states };
}
afterEach(() => vi.useRealTimers());

describe("bounded replay loading", () => {
  it("evicts least-recently-used frames and revisits without unbounded history", async () => {
    const frame = await parseScene(source);
    const loads: number[] = [];
    const cache = new ReplayCache(
      async (ordinal) => {
        loads.push(ordinal);
        return frame;
      },
      3,
      3 * frameWeight(frame),
    );
    const signal = new AbortController().signal;
    for (const ordinal of [0, 1, 2, 0, 3, 0, 1])
      await cache.get(ordinal, signal);
    expect(loads).toEqual([0, 1, 2, 3, 1]);
    expect(cache.size).toBe(3);
    expect(cache.bytes).toBeLessThanOrEqual(cache.maxBytes);
    cache.clear();
    expect(cache.bytes).toBe(0);
    expect(cache.size).toBe(0);
  });
  it("does not cache oversized frames and enforces the byte budget before count limit", async () => {
    const frame = await parseScene(source);
    const cache = new ReplayCache(async () => frame, 3, frameWeight(frame));
    const signal = new AbortController().signal;
    await cache.get(0, signal);
    await cache.get(1, signal);
    expect(cache.size).toBe(1);
    const oversized = new ReplayCache(async () => frame, 3, 1);
    await oversized.get(0, signal);
    expect(oversized.size).toBe(0);
    expect(oversized.bytes).toBe(0);
  });
  it("coalesces pending seeks and never presents a late completed frame", async () => {
    const frame = await parseScene(source);
    const delayed = deferred<SceneFrame>();
    const loads: number[] = [];
    const { controller, frames } = player(async (ordinal) => {
      loads.push(ordinal);
      return ordinal === 0 ? delayed.promise : frame;
    });
    controller.seek(0);
    controller.seek(1);
    controller.seek(4);
    expect(loads).toEqual([0]);
    delayed.resolve(frame);
    await flush();
    expect(loads).toEqual([0, 4]);
    expect(frames).toEqual([4]);
    expect(controller.state.index).toBe(4);
    expect(controller.cache.size).toBe(1);
    controller.dispose();
  });
  it("ignores stale failures and reports current failures with frame ordinal", async () => {
    const frame = await parseScene(source);
    const delayed = deferred<SceneFrame>();
    const { controller, frames } = player(async (ordinal) => {
      if (ordinal === 0) return delayed.promise;
      if (ordinal === 2) throw new Error("damaged.scene.json digest mismatch");
      return frame;
    });
    controller.seek(0);
    controller.seek(1);
    delayed.reject(new Error("stale error"));
    await flush();
    expect(frames).toEqual([1]);
    expect(controller.state.error).toBeNull();
    controller.seek(2);
    await flush();
    expect(controller.state.error).toContain("Frame 3 (ordinal 2)");
    expect(controller.state.error).toContain("damaged.scene.json");
    expect(controller.state.index).toBe(1);
    expect(controller.state.playing).toBe(false);
    controller.seek(3);
    await flush();
    expect(controller.state.error).toBeNull();
    controller.dispose();
  });
  it("aborts and suppresses old dataset completion after disposal", async () => {
    const frame = await parseScene(source);
    const delayed = deferred<SceneFrame>();
    let signal: AbortSignal | undefined;
    const { controller, frames, states } = player(async (_, active) => {
      signal = active;
      return delayed.promise;
    });
    controller.seek(0);
    const emitted = states.length;
    controller.dispose();
    expect(signal?.aborted).toBe(true);
    delayed.resolve(frame);
    await flush();
    expect(frames).toEqual([]);
    expect(states.length).toBe(emitted);
    expect(controller.cache.size).toBe(0);
  });
});

describe("recorded-frame playback", () => {
  it.each([4, null])(
    "starts playback from a pending seek when the displayed index is %s",
    async (displayedIndex) => {
      const frame = await parseScene(source);
      vi.useFakeTimers();
      const delayed = deferred<SceneFrame>();
      const loads: number[] = [];
      let pendingSignal: AbortSignal | undefined;
      const { controller, frames } = player(async (ordinal, signal) => {
        loads.push(ordinal);
        if (ordinal === 2) {
          pendingSignal = signal;
          return delayed.promise;
        }
        return frame;
      });
      const initial = displayedIndex === null ? [] : [displayedIndex];
      if (displayedIndex !== null) {
        controller.seek(displayedIndex);
        await flush();
      }
      controller.setFps(20);
      controller.seek(2);
      controller.play();
      controller.play();
      expect(controller.state).toMatchObject({
        index: displayedIndex,
        requestedIndex: 2,
        playing: true,
        loading: true,
        error: null,
      });
      expect(pendingSignal?.aborted).toBe(false);
      await vi.advanceTimersByTimeAsync(1000);
      expect(loads).toEqual([...initial, 2]);
      expect(frames).toEqual(initial);
      delayed.resolve(frame);
      await flush();
      expect(frames).toEqual([...initial, 2]);
      expect(controller.state.loading).toBe(false);
      await vi.advanceTimersByTimeAsync(49);
      expect(controller.state.index).toBe(2);
      await vi.advanceTimersByTimeAsync(1);
      expect(frames).toEqual([...initial, 2, 3]);
      controller.dispose();
    },
  );
  it.each([
    { displayed: 0, requested: 3 },
    { displayed: 3, requested: 1 },
    { displayed: 4, requested: 2 },
    { displayed: null, requested: 2 },
  ])(
    "Play retries failed frame $requested from displayed frame $displayed",
    async ({ displayed, requested }) => {
      const frame = await parseScene(source);
      vi.useFakeTimers();
      const loads: number[] = [];
      let attempts = 0;
      const { controller, frames } = player(async (ordinal) => {
        loads.push(ordinal);
        if (ordinal === requested && ++attempts <= 2)
          throw new Error("temporary read failure");
        return frame;
      });
      const initial = displayed === null ? [] : [displayed];
      if (displayed !== null) {
        controller.seek(displayed);
        await flush();
      }
      controller.setFps(20);
      controller.seek(requested);
      await flush();
      for (let retry = 0; retry < 2; retry++) {
        expect(controller.state).toMatchObject({
          index: displayed,
          requestedIndex: requested,
          playing: false,
          loading: false,
        });
        expect(controller.state.error).toContain(`ordinal ${requested}`);
        await vi.advanceTimersByTimeAsync(1000);
        expect(frames).toEqual(initial);
        controller.play();
        expect(controller.state).toMatchObject({
          requestedIndex: requested,
          playing: true,
          loading: true,
          error: null,
        });
        await flush();
      }
      expect(loads).toEqual([...initial, requested, requested, requested]);
      expect(frames).toEqual([...initial, requested]);
      expect(controller.state).toMatchObject({
        index: requested,
        requestedIndex: requested,
        playing: true,
        loading: false,
        error: null,
      });
      await vi.advanceTimersByTimeAsync(49);
      expect(controller.state.index).toBe(requested);
      await vi.advanceTimersByTimeAsync(1);
      expect(controller.state.index).toBe(requested + 1);
      controller.dispose();
    },
  );
  it("steps both ways, plays all recorded frames at selected fps, and stops at the end", async () => {
    const frame = await parseScene(source);
    vi.useFakeTimers();
    const { controller, frames } = player(async () => frame, 3);
    controller.seek(0);
    await flush();
    controller.step(1);
    await flush();
    controller.step(-1);
    await flush();
    expect(frames).toEqual([0, 1, 0]);
    controller.setFps(20);
    controller.play();
    await vi.advanceTimersByTimeAsync(49);
    expect(controller.state.index).toBe(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(controller.state.index).toBe(1);
    await vi.advanceTimersByTimeAsync(50);
    expect(controller.state.index).toBe(2);
    expect(controller.state.playing).toBe(false);
    controller.play();
    await flush();
    expect(controller.state.index).toBe(0);
    controller.pause();
    await vi.advanceTimersByTimeAsync(1000);
    expect(controller.state.index).toBe(0);
    controller.dispose();
  });
  it("manual seeking pauses playback and invalid fps keeps the previous rate", async () => {
    const frame = await parseScene(source);
    vi.useFakeTimers();
    const { controller } = player(async () => frame);
    controller.seek(0);
    await flush();
    controller.play();
    controller.seek(3);
    await flush();
    await vi.advanceTimersByTimeAsync(1000);
    expect(controller.state.index).toBe(3);
    expect(controller.state.playing).toBe(false);
    for (const value of [NaN, Infinity, 0, 121])
      expect(() => controller.setFps(value)).toThrow(RangeError);
    expect(controller.state.fps).toBe(10);
    expect(() => controller.seek(5)).toThrow(RangeError);
    controller.dispose();
  });
});
