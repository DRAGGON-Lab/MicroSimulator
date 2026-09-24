import type { ColorMode } from "./color";
import { sliceDimension, type SliceAxis } from "./grid";
import type { SceneFrame } from "./scene";

export interface PresentationPreferences {
  colorMode: ColorMode;
  speciesChannel: number;
  signalVisible: boolean;
  deviceVisible: boolean;
  signalChannel: number;
  signalAxis: SliceAxis;
  signalSlice: number | null;
}

function defaults(): PresentationPreferences {
  return {
    colorMode: "cell-type",
    speciesChannel: 0,
    signalVisible: true,
    deviceVisible: true,
    signalChannel: 0,
    signalAxis: "z",
    signalSlice: null,
  };
}

/** One identity per explicit file/session/recording open, never per frame. */
export class DatasetPresentationState {
  public datasetId = 0;
  public preferences = defaults();

  public beginDataset(): void {
    this.datasetId += 1;
    this.preferences = defaults();
  }

  /** Clamp only the displayed values, retaining choices for later frames. */
  public forFrame(frame: SceneFrame): PresentationPreferences & {
    signalSlice: number;
  } {
    const grid = frame.signalGrid;
    if (grid !== null && this.preferences.signalSlice === null) {
      this.preferences.signalSlice = Math.floor(
        (sliceDimension(grid, this.preferences.signalAxis) - 1) / 2,
      );
    }
    return {
      ...this.preferences,
      colorMode:
        frame.speciesCount === 0 && this.preferences.colorMode === "species"
          ? "cell-type"
          : this.preferences.colorMode,
      speciesChannel: Math.min(
        this.preferences.speciesChannel,
        Math.max(frame.speciesCount - 1, 0),
      ),
      signalChannel: Math.min(
        this.preferences.signalChannel,
        Math.max((grid?.signalCount ?? 0) - 1, 0),
      ),
      signalSlice: Math.min(
        this.preferences.signalSlice ?? 0,
        grid === null
          ? 0
          : sliceDimension(grid, this.preferences.signalAxis) - 1,
      ),
    };
  }
}
