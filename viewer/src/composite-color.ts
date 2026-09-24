import { Color, SRGBColorSpace } from "three";
import type { RGB } from "./color";
import { channelLabel, type SceneFrame } from "./scene";
import {
  normalizeScalar,
  resolveScalarRange,
  type ResolvedScalarRange,
  type ScalarRangeConfig,
} from "./scalar-range";

export const COMPOSITE_NEUTRAL: RGB = [0.55, 0.6, 0.58];
export interface CompositeChannelConfig {
  readonly index: number;
  readonly enabled: boolean;
  /** A six-digit CSS hexadecimal color in sRGB display space. */
  readonly tint: string;
  readonly range: ScalarRangeConfig;
}
export interface CompositeChannelLegend {
  readonly index: number;
  readonly label: string;
  readonly tint: string;
  readonly range: ResolvedScalarRange;
}

export function validateTint(tint: string): void {
  if (!/^#[0-9a-fA-F]{6}$/.test(tint))
    throw new RangeError(
      "Use a six-digit hexadecimal tint, for example #ff0000.",
    );
}

/** Return sRGB display colors, as required by ColonyViewer.setCellColors(). */
export function mapCompositeSpecies(
  frame: SceneFrame,
  configuration: readonly CompositeChannelConfig[],
): { colors: readonly RGB[]; channels: readonly CompositeChannelLegend[] } {
  const seen = new Set<number>();
  const active = configuration.filter((channel) => channel.enabled);
  for (const channel of active) {
    if (
      !Number.isSafeInteger(channel.index) ||
      channel.index < 0 ||
      channel.index >= frame.speciesCount
    )
      throw new RangeError(`species channel ${channel.index} is out of range`);
    if (seen.has(channel.index))
      throw new RangeError(`duplicate species channel ${channel.index}`);
    seen.add(channel.index);
    validateTint(channel.tint);
  }
  // Canonical summation order makes results bitwise independent of UI order.
  const resolved = active
    .toSorted((a, b) => a.index - b.index)
    .map((channel) => ({
      channel,
      tint: new Color(channel.tint), // Three.js decodes CSS sRGB into linear RGB.
      range: resolveScalarRange(
        frame.cells.map((cell) => cell.species[channel.index] ?? 0),
        channel.range,
      ),
    }));
  const colors: RGB[] = frame.cells.map((cell) => {
    if (resolved.length === 0) return COMPOSITE_NEUTRAL;
    const linear = new Color(0, 0, 0);
    for (const { channel, tint, range } of resolved) {
      const intensity = normalizeScalar(
        cell.species[channel.index] ?? 0,
        range,
      );
      linear.r += intensity * tint.r;
      linear.g += intensity * tint.g;
      linear.b += intensity * tint.b;
    }
    linear.r = Math.min(1, linear.r);
    linear.g = Math.min(1, linear.g);
    linear.b = Math.min(1, linear.b);
    const display = linear.getRGB({ r: 0, g: 0, b: 0 }, SRGBColorSpace);
    return [display.r, display.g, display.b];
  });
  const byIndex = new Map(
    resolved.map((value) => [value.channel.index, value]),
  );
  return {
    colors,
    channels: active.map(({ index, tint }) => ({
      index,
      tint,
      label: channelLabel(frame, "species", index),
      range: byIndex.get(index)!.range,
    })),
  };
}
