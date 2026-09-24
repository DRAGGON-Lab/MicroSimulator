import type { SceneFrame } from "./scene";

export const REPLAY_CACHE_FRAMES = 3;
export const REPLAY_CACHE_BYTES = 64 * 1024 * 1024;
export type FrameLoader = (
  ordinal: number,
  signal: AbortSignal,
) => Promise<SceneFrame>;

/** Conservative accounting units, not a claim about a particular JS engine's heap. */
export function frameWeight(frame: SceneFrame): number {
  let bytes = 1024;
  for (const cell of frame.cells)
    bytes +=
      512 +
      16 * cell.species.length +
      2 * (cell.id.length + (cell.parentId?.length ?? 0));
  for (const labels of [
    frame.channelMetadata.species,
    frame.channelMetadata.signals,
  ]) {
    for (const label of labels) bytes += 32 + 2 * (label?.length ?? 0);
  }
  const grid = frame.signalGrid;
  if (grid !== null) {
    bytes += 1024 + 16 * grid.levels.length;
    for (const boundary of Object.values(grid.boundaries))
      bytes += 128 + 16 * boundary.values.length;
  }
  for (const constraints of Object.values(frame.constraints))
    bytes += 512 * constraints.length;
  return bytes;
}
export class ReplayCache {
  private readonly values = new Map<
    number,
    { frame: SceneFrame; weight: number }
  >();
  public bytes = 0;
  public constructor(
    private readonly loader: FrameLoader,
    public readonly maxFrames = REPLAY_CACHE_FRAMES,
    public readonly maxBytes = REPLAY_CACHE_BYTES,
  ) {
    if (
      !Number.isSafeInteger(maxFrames) ||
      maxFrames < 1 ||
      !Number.isSafeInteger(maxBytes) ||
      maxBytes < 1
    )
      throw new RangeError("cache limits must be positive safe integers");
  }
  public get size(): number {
    return this.values.size;
  }
  public clear(): void {
    this.values.clear();
    this.bytes = 0;
  }
  public async get(ordinal: number, signal: AbortSignal): Promise<SceneFrame> {
    signal.throwIfAborted();
    const cached = this.values.get(ordinal);
    if (cached !== undefined) {
      this.values.delete(ordinal);
      this.values.set(ordinal, cached);
      return cached.frame;
    }
    const frame = await this.loader(ordinal, signal);
    signal.throwIfAborted();
    const weight = frameWeight(frame);
    if (weight <= this.maxBytes) {
      while (
        this.values.size >= this.maxFrames ||
        this.bytes + weight > this.maxBytes
      ) {
        const key = this.values.keys().next().value;
        if (key === undefined) break;
        this.bytes -= this.values.get(key)!.weight;
        this.values.delete(key);
      }
      this.values.set(ordinal, { frame, weight });
      this.bytes += weight;
    }
    return frame;
  }
}
export interface ReplayState {
  readonly index: number | null;
  readonly requestedIndex: number;
  readonly playing: boolean;
  readonly loading: boolean;
  readonly fps: number;
  readonly error: string | null;
}
export interface ReplayCallbacks {
  readonly frame: (frame: SceneFrame, ordinal: number) => void;
  readonly state: (state: ReplayState) => void;
}

/** Latest seek wins. A single decode worker prevents unbounded seek fan-out. */
export class ReplayController {
  public readonly cache: ReplayCache;
  private index: number | null = null;
  private requestedIndex = 0;
  private playing = false;
  private loading = false;
  private fps = 10;
  private error: string | null = null;
  private disposed = false;
  private version = 0;
  private pending: { ordinal: number; version: number } | null = null;
  private running = false;
  private active: AbortController | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;

  public constructor(
    public readonly frameCount: number,
    loader: FrameLoader,
    private readonly callbacks: ReplayCallbacks,
    cache?: ReplayCache,
  ) {
    if (!Number.isSafeInteger(frameCount) || frameCount < 1)
      throw new RangeError("recording must contain frames");
    this.cache = cache ?? new ReplayCache(loader);
  }
  public get state(): ReplayState {
    return {
      index: this.index,
      requestedIndex: this.requestedIndex,
      playing: this.playing,
      loading: this.loading,
      fps: this.fps,
      error: this.error,
    };
  }
  private emit(): void {
    if (!this.disposed) this.callbacks.state(this.state);
  }
  private clearTimer(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
  }
  public pause(): void {
    this.playing = false;
    this.clearTimer();
    this.emit();
  }
  public play(): void {
    if (this.disposed) return;
    this.playing = true;
    this.error = null;
    if (this.index === null || this.index === this.frameCount - 1)
      this.request(0);
    else if (!this.loading) this.schedule();
    this.emit();
  }
  public setFps(fps: number): void {
    if (!Number.isFinite(fps) || fps < 1 || fps > 120)
      throw new RangeError("Playback frame rate must be between 1 and 120 fps");
    this.fps = fps;
    if (this.playing && !this.loading) this.schedule();
    this.emit();
  }
  public seek(ordinal: number): void {
    if (
      !Number.isSafeInteger(ordinal) ||
      ordinal < 0 ||
      ordinal >= this.frameCount
    )
      throw new RangeError(`frame ${ordinal} is out of range`);
    this.pause();
    this.request(ordinal);
  }
  public step(delta: -1 | 1): void {
    this.seek(
      Math.max(0, Math.min(this.frameCount - 1, this.requestedIndex + delta)),
    );
  }
  private request(ordinal: number): void {
    if (this.disposed) return;
    this.clearTimer();
    this.requestedIndex = ordinal;
    this.pending = { ordinal, version: ++this.version };
    this.loading = true;
    this.error = null;
    this.active?.abort();
    this.emit();
    void this.drain();
  }
  private async drain(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      while (this.pending !== null && !this.disposed) {
        const request = this.pending;
        this.pending = null;
        const active = new AbortController();
        this.active = active;
        try {
          const frame = await this.cache.get(request.ordinal, active.signal);
          if (this.disposed || request.version !== this.version) continue;
          this.index = request.ordinal;
          this.loading = false;
          this.callbacks.frame(frame, request.ordinal);
          this.emit();
          if (this.playing) this.schedule();
        } catch (error) {
          if (this.disposed || request.version !== this.version) continue;
          this.loading = false;
          this.playing = false;
          this.error = `Frame ${request.ordinal + 1} (ordinal ${request.ordinal}): ${error instanceof Error ? error.message : String(error)}`;
          this.emit();
        } finally {
          this.active = null;
        }
      }
    } finally {
      this.running = false;
    }
  }
  private schedule(): void {
    this.clearTimer();
    if (this.index === null || this.loading || !this.playing || this.disposed)
      return;
    if (this.index >= this.frameCount - 1) {
      this.playing = false;
      this.emit();
      return;
    }
    this.timer = setTimeout(
      () => this.request((this.index ?? 0) + 1),
      1000 / this.fps,
    );
  }
  public dispose(): void {
    this.disposed = true;
    this.playing = false;
    this.pending = null;
    ++this.version;
    this.clearTimer();
    this.active?.abort();
    this.cache.clear();
  }
}
