import type { CompositeChannelLegend } from "./composite-color";
import { CompositeChannelState } from "./composite-state";
import { ScalarRangeControls } from "./scalar-range-controls";
import { DatasetScalarRanges, resolveScalarRange } from "./scalar-range";
import { channelLabel, type SceneFrame } from "./scene";

interface ChannelRow {
  root: HTMLElement;
  label: HTMLElement;
  enabled: HTMLInputElement;
  swatch: HTMLElement;
  tint: HTMLInputElement;
  tintError: HTMLElement;
  range: ScalarRangeControls;
  up: HTMLButtonElement;
  down: HTMLButtonElement;
}

/** The channels of one group that a composite draws from, and how to read them. */
export interface CompositeChannelSource {
  readonly group: "species" | "signals";
  /** The id of the legend element; range controls take ids that begin with `rangeIdPrefix`. */
  readonly legendId: string;
  readonly rangeIdPrefix: string;
  /** What the legend says when no channel is enabled. */
  readonly emptyText: string;
  count(frame: SceneFrame): number;
  /** One channel's values over what is displayed: every cell, or the pixels of a slice. */
  values(frame: SceneFrame, index: number): readonly number[];
}

export const SPECIES_COMPOSITE_SOURCE: CompositeChannelSource = {
  group: "species",
  legendId: "composite-legend",
  rangeIdPrefix: "composite-range",
  emptyText: "No channels active. Cells use neutral gray.",
  count: (frame) => frame.speciesCount,
  values: (frame, index) => frame.cells.map((cell) => cell.species[index] ?? 0),
};

export class CompositeChannelControls {
  private readonly rows = new Map<number, ChannelRow>();
  private readonly list = document.createElement("div");
  private readonly legend = document.createElement("div");
  private visibleOrder = "";
  private count = 0;

  public constructor(
    root: HTMLElement,
    private readonly state: CompositeChannelState,
    private readonly ranges: DatasetScalarRanges,
    private readonly onChange: () => void,
    private readonly formatNumber: (value: number) => string,
    private readonly source: CompositeChannelSource = SPECIES_COMPOSITE_SOURCE,
  ) {
    this.legend.id = source.legendId;
    this.legend.className = "composite-legend";
    this.legend.setAttribute("aria-label", "Composite channel legend");
    root.append(this.list, this.legend);
  }

  public beginDataset(): void {
    this.rows.clear();
    this.list.replaceChildren();
    this.legend.replaceChildren();
    this.visibleOrder = "";
  }

  public update(
    frame: SceneFrame,
    channels: readonly CompositeChannelLegend[],
  ): void {
    this.count = this.source.count(frame);
    const indices = this.state.indices(this.count);
    for (const [position, index] of indices.entries()) {
      let row = this.rows.get(index);
      if (row === undefined) {
        row = this.createRow(index);
        this.rows.set(index, row);
      }
      const name = channelLabel(frame, this.source.group, index);
      row.label.textContent = name;
      row.root.setAttribute("aria-label", `${name} composite channel`);
      row.enabled.setAttribute("aria-label", `Enable ${name}`);
      row.up.setAttribute("aria-label", `Move ${name} up`);
      row.down.setAttribute("aria-label", `Move ${name} down`);
      row.up.disabled = position === 0;
      row.down.disabled = position === indices.length - 1;
      const preference = this.state.get(index);
      row.enabled.checked = preference.enabled;
      row.swatch.style.backgroundColor = preference.tint;
      row.range.bind(
        index,
        resolveScalarRange(
          this.source.values(frame, index),
          this.ranges.get(this.source.group, index),
        ),
      );
    }
    const order = indices.join(",");
    if (order !== this.visibleOrder) {
      this.list.replaceChildren(
        ...indices.map((index) => this.rows.get(index)!.root),
      );
      this.visibleOrder = order;
    }
    this.legend.replaceChildren();
    if (channels.length === 0) {
      this.legend.textContent = this.source.emptyText;
      return;
    }
    for (const channel of channels) {
      const item = document.createElement("div");
      item.className = "composite-legend-channel";
      const swatch = document.createElement("span");
      swatch.className = "composite-swatch";
      swatch.style.backgroundColor = channel.tint;
      swatch.setAttribute("aria-hidden", "true");
      const description = document.createElement("span");
      const bounds =
        channel.range.minimum === null || channel.range.maximum === null
          ? "no values"
          : `${this.formatNumber(channel.range.minimum)} to ${this.formatNumber(channel.range.maximum)}`;
      description.textContent = `${channel.label} · ${channel.tint} · ${channel.range.mode === "fixed" ? "Fixed" : "Automatic"} ${bounds}${channel.range.count === 0 && channel.range.minimum !== null ? " (no values)" : ""}`;
      item.append(swatch, description);
      this.legend.append(item);
    }
  }

  private createRow(index: number): ChannelRow {
    const root = document.createElement("div");
    root.className = "composite-channel";
    root.dataset.channel = String(index);
    root.setAttribute("role", "group");
    const heading = document.createElement("label");
    heading.className = "composite-channel-heading";
    const enabled = document.createElement("input");
    enabled.type = "checkbox";
    const swatch = document.createElement("span");
    swatch.className = "composite-swatch";
    swatch.setAttribute("aria-hidden", "true");
    const label = document.createElement("span");
    heading.append(enabled, swatch, label);
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Display settings";
    const tintForm = document.createElement("form");
    tintForm.className = "composite-tint-form";
    const tintLabel = document.createElement("label");
    tintLabel.className = "field";
    tintLabel.textContent = "Tint (sRGB hex)";
    const tint = document.createElement("input");
    tint.type = "text";
    tint.value = this.state.get(index).tint;
    tint.spellcheck = false;
    tint.setAttribute("aria-label", "Channel tint");
    tintLabel.append(tint);
    const tintApply = document.createElement("button");
    tintApply.className = "button button-secondary";
    tintApply.type = "submit";
    tintApply.textContent = "Apply tint";
    const tintError = document.createElement("p");
    tintError.className = "scalar-range-message";
    tintError.setAttribute("role", "status");
    tintForm.append(tintLabel, tintApply, tintError);
    const rangeRoot = document.createElement("div");
    rangeRoot.id = `${this.source.rangeIdPrefix}-${index}`;
    const range = new ScalarRangeControls(
      rangeRoot,
      this.ranges,
      this.source.group,
      this.onChange,
    );
    const order = document.createElement("div");
    order.className = "composite-channel-order";
    const moveButton = (direction: -1 | 1): HTMLButtonElement => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "button button-secondary";
      button.textContent = direction === -1 ? "Move up" : "Move down";
      button.addEventListener("click", () => {
        this.state.move(index, direction, this.count);
        this.onChange();
        this.rows.get(index)?.[direction === -1 ? "down" : "up"].focus();
      });
      order.append(button);
      return button;
    };
    const up = moveButton(-1);
    const down = moveButton(1);
    details.append(summary, tintForm, rangeRoot, order);
    root.append(heading, details);
    enabled.addEventListener("change", () => {
      this.state.setEnabled(index, enabled.checked);
      this.onChange();
    });
    tintForm.addEventListener("submit", (event) => {
      event.preventDefault();
      try {
        this.state.setTint(index, tint.value.trim());
        tint.value = this.state.get(index).tint;
        tint.removeAttribute("aria-invalid");
        tintError.textContent = "";
        this.onChange();
      } catch (error) {
        tint.setAttribute("aria-invalid", "true");
        tintError.textContent =
          error instanceof Error ? error.message : String(error);
      }
    });
    return { root, label, enabled, swatch, tint, tintError, range, up, down };
  }
}
