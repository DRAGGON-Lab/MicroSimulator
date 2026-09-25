import { channelLabel, type SceneFrame } from "./scene";
import {
  mapCompositeSpecies,
  type CompositeChannelConfig,
  type CompositeChannelLegend,
} from "./composite-color";
import {
  AUTOMATIC_SCALAR_RANGE,
  normalizeScalar,
  resolveScalarRange,
  type ResolvedScalarRange,
  type ScalarRangeConfig,
} from "./scalar-range";

export type RGB = readonly [number, number, number];
export type ColorMode =
  "cell-type" | "species" | "composite" | "growth-rate" | "fixed";

export interface ColorConfig {
  readonly mode: ColorMode;
  readonly speciesIndex: number;
  readonly range?: ScalarRangeConfig;
  readonly compositeChannels?: readonly CompositeChannelConfig[];
}

export interface ColorMapping {
  readonly colors: readonly RGB[];
  readonly title: string;
  readonly minimum: number | null;
  readonly maximum: number | null;
  readonly range: ResolvedScalarRange | null;
  readonly composite?: readonly CompositeChannelLegend[];
}

const TYPE_PALETTE: readonly RGB[] = [
  [0.545, 0.878, 0.741],
  [0.455, 0.718, 0.929],
  [0.957, 0.706, 0.424],
  [0.808, 0.596, 0.91],
  [0.949, 0.525, 0.58],
  [0.58, 0.82, 0.867],
  [0.757, 0.827, 0.482],
  [0.89, 0.643, 0.824],
];

const VIRIDIS: readonly (readonly [number, RGB])[] = [
  [0, [0.267, 0.005, 0.329]],
  [0.25, [0.23, 0.322, 0.545]],
  [0.5, [0.128, 0.567, 0.551]],
  [0.75, [0.369, 0.789, 0.383]],
  [1, [0.993, 0.906, 0.144]],
];

function interpolate(left: RGB, right: RGB, fraction: number): RGB {
  return [
    left[0] + (right[0] - left[0]) * fraction,
    left[1] + (right[1] - left[1]) * fraction,
    left[2] + (right[2] - left[2]) * fraction,
  ];
}

export function viridis(value: number): RGB {
  const clamped = Math.min(1, Math.max(0, value));
  for (let index = 1; index < VIRIDIS.length; index += 1) {
    const right = VIRIDIS[index];
    const left = VIRIDIS[index - 1];
    if (left !== undefined && right !== undefined && clamped <= right[0]) {
      const span = right[0] - left[0];
      return interpolate(
        left[1],
        right[1],
        span === 0 ? 0 : (clamped - left[0]) / span,
      );
    }
  }
  return VIRIDIS.at(-1)?.[1] ?? [1, 1, 1];
}

export function mapScalarColors(
  values: readonly number[],
  config: ScalarRangeConfig = AUTOMATIC_SCALAR_RANGE,
): { colors: readonly RGB[]; range: ResolvedScalarRange } {
  const range = resolveScalarRange(values, config);
  return {
    colors: values.map((value) => viridis(normalizeScalar(value, range))),
    range,
  };
}

function scalarMapping(
  values: readonly number[],
  title: string,
  config: ScalarRangeConfig = AUTOMATIC_SCALAR_RANGE,
): ColorMapping {
  const mapping = mapScalarColors(values, config);
  return {
    ...mapping,
    title,
    minimum: mapping.range.minimum,
    maximum: mapping.range.maximum,
  };
}

export function mapCellColors(
  frame: SceneFrame,
  config: ColorConfig,
): ColorMapping {
  switch (config.mode) {
    case "composite": {
      const mapping = mapCompositeSpecies(
        frame,
        config.compositeChannels ?? [],
      );
      return {
        colors: mapping.colors,
        title: "Species composite",
        minimum: null,
        maximum: null,
        range: null,
        composite: mapping.channels,
      };
    }
    case "cell-type":
      return {
        colors: frame.cells.map((cell) => {
          const index =
            ((cell.cellType % TYPE_PALETTE.length) + TYPE_PALETTE.length) %
            TYPE_PALETTE.length;
          return TYPE_PALETTE[index] ?? [1, 1, 1];
        }),
        title: "Cell type",
        minimum: null,
        maximum: null,
        range: null,
      };
    case "fixed":
      return {
        colors: frame.cells.map((cell) =>
          cell.fixed ? [0.957, 0.553, 0.337] : [0.455, 0.718, 0.929],
        ),
        title: "Fixed state",
        minimum: null,
        maximum: null,
        range: null,
      };
    case "growth-rate":
      return scalarMapping(
        frame.cells.map((cell) => cell.growthRate),
        "Growth rate",
      );
    case "species": {
      if (
        config.speciesIndex < 0 ||
        config.speciesIndex >= frame.speciesCount
      ) {
        throw new RangeError(
          `species channel ${config.speciesIndex} is out of range`,
        );
      }
      return scalarMapping(
        frame.cells.map((cell) => cell.species[config.speciesIndex] ?? 0),
        channelLabel(frame, "species", config.speciesIndex),
        config.range,
      );
    }
  }
}

export function rgbBytes(color: RGB): readonly [number, number, number] {
  return [
    Math.round(Math.min(1, Math.max(0, color[0])) * 255),
    Math.round(Math.min(1, Math.max(0, color[1])) * 255),
    Math.round(Math.min(1, Math.max(0, color[2])) * 255),
  ];
}
