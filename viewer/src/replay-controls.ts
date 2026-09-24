import { ReplayController, type ReplayState } from "./replay";
import { ReplayBundle } from "./replay-bundle";
import type { SceneFrame } from "./scene";

export class ReplayControls {
  private player: ReplayController | null = null;
  private opening: AbortController | null = null;
  private readonly timeline: HTMLInputElement;
  private readonly position: HTMLOutputElement;
  private readonly play: HTMLButtonElement;
  private readonly previous: HTMLButtonElement;
  private readonly next: HTMLButtonElement;
  private readonly fps: HTMLInputElement;
  private readonly message: HTMLElement;

  public constructor(
    private readonly host: HTMLElement,
    private readonly present: (frame: SceneFrame, newDataset: boolean) => void,
    private readonly reportError: (message: string) => void,
  ) {
    const element = <T extends HTMLElement>(id: string): T => {
      const found = host.querySelector<T>(`#${id}`);
      if (found === null) throw new Error(`missing replay control ${id}`);
      return found;
    };
    this.timeline = element("replay-timeline");
    this.position = element("replay-position");
    this.play = element("replay-play");
    this.previous = element("replay-previous");
    this.next = element("replay-next");
    this.fps = element("replay-fps");
    this.message = element("replay-message");
    this.play.addEventListener("click", () => {
      if (this.player?.state.playing) this.player.pause();
      else this.player?.play();
    });
    this.previous.addEventListener("click", () => this.player?.step(-1));
    this.next.addEventListener("click", () => this.player?.step(1));
    this.timeline.addEventListener("input", () =>
      this.player?.seek(Number(this.timeline.value)),
    );
    this.fps.addEventListener("change", () => {
      try {
        this.player?.setFps(this.fps.valueAsNumber);
        this.fps.setCustomValidity("");
      } catch (error) {
        this.fps.setCustomValidity(
          error instanceof Error ? error.message : String(error),
        );
        this.fps.reportValidity();
      }
    });
  }
  public close(): void {
    this.opening?.abort();
    this.opening = null;
    this.player?.dispose();
    this.player = null;
    this.host.hidden = true;
  }
  public async open(files: readonly File[]): Promise<void> {
    this.close();
    const opening = new AbortController();
    this.opening = opening;
    try {
      const bundle = await ReplayBundle.open(files, opening.signal);
      if (opening.signal.aborted) return;
      let first = true;
      const player = new ReplayController(
        bundle.manifest.frames.length,
        (ordinal, signal) => bundle.load(ordinal, signal),
        {
          frame: (frame) => {
            this.present(frame, first);
            first = false;
          },
          state: (state) =>
            this.update(
              state,
              bundle.manifest.frames.length,
              state.index === null
                ? null
                : bundle.manifest.frames[state.index]!.time,
            ),
        },
      );
      this.player = player;
      this.timeline.max = String(bundle.manifest.frames.length - 1);
      this.fps.value = "10";
      this.fps.setCustomValidity("");
      this.host.hidden = false;
      player.seek(0);
    } catch (error) {
      if (!opening.signal.aborted)
        this.reportError(
          error instanceof Error ? error.message : String(error),
        );
    }
  }
  private update(state: ReplayState, count: number, time: number | null): void {
    this.timeline.value = String(state.requestedIndex);
    this.timeline.setAttribute(
      "aria-valuetext",
      `Frame ${state.requestedIndex + 1} of ${count}`,
    );
    this.position.value =
      state.index === null
        ? `— / ${count}`
        : `${state.index + 1} / ${count} · t = ${time?.toLocaleString(undefined, { maximumSignificantDigits: 7 }) ?? "—"}`;
    this.play.textContent = state.playing ? "Pause" : "Play";
    this.previous.disabled = state.requestedIndex === 0;
    this.next.disabled = state.requestedIndex === count - 1;
    this.message.textContent =
      state.error ??
      (state.loading ? `Loading frame ${state.requestedIndex + 1}…` : "");
    this.message.dataset.kind = state.error === null ? "info" : "error";
  }
}
