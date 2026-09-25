export type ScalarRangeConfig =
  | Readonly<{ mode: "automatic" }>
  | Readonly<{ mode: "fixed"; minimum: number; maximum: number }>;
export type FixedScalarRange = Extract<ScalarRangeConfig, { mode: "fixed" }>;
export type ScalarChannelKind = "species" | "signals";
export interface ResolvedScalarRange {
  readonly mode: ScalarRangeConfig["mode"];
  readonly minimum: number | null;
  readonly maximum: number | null;
  readonly count: number;
}
export const AUTOMATIC_SCALAR_RANGE: ScalarRangeConfig = Object.freeze({
  mode: "automatic",
});

export function validateScalarRange(config: ScalarRangeConfig): void {
  if (config.mode === "automatic") return;
  if (!Number.isFinite(config.minimum))
    throw new RangeError("Minimum must be a finite number.");
  if (!Number.isFinite(config.maximum))
    throw new RangeError("Maximum must be a finite number.");
  if (config.minimum >= config.maximum)
    throw new RangeError("Minimum must be less than maximum.");
}

export function parseFixedScalarRange(
  minimum: string,
  maximum: string,
): FixedScalarRange {
  const number = (text: string, label: string): number => {
    if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(text.trim())) {
      throw new RangeError(`${label} must be a finite number.`);
    }
    return Number(text);
  };
  const result = {
    mode: "fixed" as const,
    minimum: number(minimum, "Minimum"),
    maximum: number(maximum, "Maximum"),
  };
  validateScalarRange(result);
  return result;
}

/** Empty automatic ranges have no bounds; fixed bounds still describe a configured scale. */
export function resolveScalarRange(
  values: readonly number[],
  config: ScalarRangeConfig = AUTOMATIC_SCALAR_RANGE,
): ResolvedScalarRange {
  validateScalarRange(config);
  let minimum: number | null = null;
  let maximum: number | null = null;
  for (const value of values) {
    if (!Number.isFinite(value))
      throw new RangeError("Scalar values must be finite.");
    minimum = minimum === null ? value : Math.min(minimum, value);
    maximum = maximum === null ? value : Math.max(maximum, value);
  }
  return {
    mode: config.mode,
    minimum: config.mode === "fixed" ? config.minimum : minimum,
    maximum: config.mode === "fixed" ? config.maximum : maximum,
    count: values.length,
  };
}

/** Return display intensity only; the source value is never modified. */
export function normalizeScalar(
  value: number,
  range: ResolvedScalarRange,
): number {
  if (!Number.isFinite(value))
    throw new RangeError("Scalar values must be finite.");
  const { minimum, maximum } = range;
  if (minimum === null || maximum === null)
    throw new RangeError("Cannot normalize without scalar bounds.");
  if (maximum === minimum) return 0.5;
  if (value <= minimum) return 0;
  if (value >= maximum) return 1;
  const span = maximum - minimum;
  // Extreme finite user bounds can overflow their difference. Scaling both
  // numerator and denominator preserves the ratio without overflowing.
  const normalized = Number.isFinite(span)
    ? (value - minimum) / span
    : (value / 2 - minimum / 2) / (maximum / 2 - minimum / 2);
  return Math.min(1, Math.max(0, normalized));
}

/** Valid initial bounds when switching from automatic to fixed display. */
export function suggestedFixedRange(
  range: ResolvedScalarRange,
): FixedScalarRange {
  const { minimum, maximum } = range;
  if (minimum === null || maximum === null)
    return { mode: "fixed", minimum: 0, maximum: 1 };
  if (minimum < maximum) return { mode: "fixed", minimum, maximum };
  const padding = Math.max(Math.abs(minimum) / 2, 0.5);
  return {
    mode: "fixed",
    minimum: Math.max(-Number.MAX_VALUE, minimum - padding),
    maximum: Math.min(Number.MAX_VALUE, maximum + padding),
  };
}

/** Dataset-local settings keyed by kind and numerical channel index, never labels. */
export class DatasetScalarRanges {
  private readonly channels = new Map<
    string,
    { active: ScalarRangeConfig; fixed: FixedScalarRange | null }
  >();
  private key(kind: ScalarChannelKind, index: number): string {
    if (!Number.isSafeInteger(index) || index < 0)
      throw new RangeError(
        "Channel index must be a non-negative safe integer.",
      );
    return `${kind}:${index}`;
  }
  public beginDataset(): void {
    this.channels.clear();
  }
  public get(kind: ScalarChannelKind, index: number): ScalarRangeConfig {
    return (
      this.channels.get(this.key(kind, index))?.active ?? AUTOMATIC_SCALAR_RANGE
    );
  }
  public lastFixed(
    kind: ScalarChannelKind,
    index: number,
  ): FixedScalarRange | null {
    return this.channels.get(this.key(kind, index))?.fixed ?? null;
  }
  public set(
    kind: ScalarChannelKind,
    index: number,
    config: ScalarRangeConfig,
  ): void {
    validateScalarRange(config);
    const key = this.key(kind, index);
    const active = Object.freeze({ ...config });
    this.channels.set(key, {
      active,
      fixed:
        active.mode === "fixed"
          ? active
          : (this.channels.get(key)?.fixed ?? null),
    });
  }
}
