import { validateTint, type CompositeChannelConfig } from "./composite-color";
import type { DatasetScalarRanges } from "./scalar-range";

const TINTS = [
  "#ff0000",
  "#00ff00",
  "#0080ff",
  "#ff00ff",
  "#00ffff",
  "#ff8000",
] as const;
export interface CompositeChannelPreference {
  readonly enabled: boolean;
  readonly tint: string;
}

/**
 * Which channels of one group, species or signals, a composite shows, in what
 * tints and in what order. The first `initiallyEnabled` channels start on.
 */
export class CompositeChannelState {
  private readonly channels = new Map<number, CompositeChannelPreference>();
  private order: number[] = [];
  public constructor(
    private readonly group: "species" | "signals" = "species",
    private readonly initiallyEnabled = 2,
  ) {}
  public beginDataset(): void {
    this.channels.clear();
    this.order = [];
  }
  public get(index: number): CompositeChannelPreference {
    if (!Number.isSafeInteger(index) || index < 0)
      throw new RangeError(
        "Channel index must be a non-negative safe integer.",
      );
    return (
      this.channels.get(index) ?? {
        enabled: index < this.initiallyEnabled,
        tint: TINTS[index % TINTS.length]!,
      }
    );
  }
  public setEnabled(index: number, enabled: boolean): void {
    this.channels.set(index, { ...this.get(index), enabled });
  }
  public setTint(index: number, tint: string): void {
    validateTint(tint);
    this.channels.set(index, { ...this.get(index), tint: tint.toLowerCase() });
  }
  public indices(count: number): readonly number[] {
    for (let index = 0; index < count; index += 1)
      if (!this.order.includes(index)) this.order.push(index);
    return this.order.filter((index) => index < count);
  }
  public move(index: number, direction: -1 | 1, count: number): void {
    const visible = this.indices(count);
    const neighbor = visible[visible.indexOf(index) + direction];
    if (neighbor === undefined || !visible.includes(index)) return;
    const from = this.order.indexOf(index);
    const to = this.order.indexOf(neighbor);
    [this.order[from], this.order[to]] = [neighbor, index];
  }
  public configuration(
    count: number,
    ranges: DatasetScalarRanges,
  ): readonly CompositeChannelConfig[] {
    return this.indices(count).map((index) => ({
      index,
      ...this.get(index),
      range: ranges.get(this.group, index),
    }));
  }
}
