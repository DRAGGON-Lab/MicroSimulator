import {
  DatasetScalarRanges,
  parseFixedScalarRange,
  suggestedFixedRange,
  type ResolvedScalarRange,
  type ScalarChannelKind,
  type ScalarRangeConfig,
} from "./scalar-range";

/** Reusable keyboard-accessible editor; draft text never changes the active range. */
export class ScalarRangeControls {
  private readonly mode: HTMLSelectElement;
  private readonly fixedFields: HTMLElement;
  private readonly minimum: HTMLInputElement;
  private readonly maximum: HTMLInputElement;
  private readonly message: HTMLElement;
  private channel: number | null = null;
  private active: ScalarRangeConfig | null = null;
  private resolved: ResolvedScalarRange | null = null;

  public constructor(
    root: HTMLElement,
    private readonly ranges: DatasetScalarRanges,
    private readonly kind: ScalarChannelKind,
    private readonly onChange: () => void,
  ) {
    const label = kind === "species" ? "Species" : "Signal";
    const form = document.createElement("form");
    form.noValidate = true;
    form.className = "scalar-range-control";
    form.setAttribute("aria-label", `${label} concentration range`);
    const modeLabel = document.createElement("label");
    modeLabel.className = "field";
    const modeText = document.createElement("span");
    modeText.textContent = "Color range";
    this.mode = document.createElement("select");
    this.mode.setAttribute("aria-label", `${label} range mode`);
    for (const [value, text] of [
      ["automatic", "Automatic"],
      ["fixed", "Fixed"],
    ] as const) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      this.mode.append(option);
    }
    modeLabel.append(modeText, this.mode);
    this.fixedFields = document.createElement("div");
    this.fixedFields.className = "scalar-range-fields";
    const input = (title: string): HTMLInputElement => {
      const labelElement = document.createElement("label");
      labelElement.className = "field";
      const text = document.createElement("span");
      text.textContent = title;
      const element = document.createElement("input");
      element.type = "text";
      element.inputMode = "decimal";
      element.setAttribute("aria-label", `${label} ${title.toLowerCase()}`);
      element.setAttribute("aria-describedby", `${root.id}-message`);
      labelElement.append(text, element);
      this.fixedFields.append(labelElement);
      return element;
    };
    this.minimum = input("Minimum");
    this.maximum = input("Maximum");
    const apply = document.createElement("button");
    apply.type = "submit";
    apply.className = "button button-secondary";
    apply.textContent = "Apply range";
    this.message = document.createElement("p");
    this.message.id = `${root.id}-message`;
    this.message.className = "scalar-range-message";
    this.message.setAttribute("role", "status");
    this.fixedFields.append(apply, this.message);
    form.append(modeLabel, this.fixedFields);
    root.append(form);
    this.mode.addEventListener("change", () => {
      if (this.channel === null || this.resolved === null) return;
      const config =
        this.mode.value === "automatic"
          ? { mode: "automatic" as const }
          : (this.ranges.lastFixed(this.kind, this.channel) ??
            suggestedFixedRange(this.resolved));
      this.ranges.set(this.kind, this.channel, config);
      this.syncFields();
      this.onChange();
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (this.channel === null || this.mode.value !== "fixed") return;
      try {
        const config = parseFixedScalarRange(
          this.minimum.value,
          this.maximum.value,
        );
        this.ranges.set(this.kind, this.channel, config);
        this.syncFields();
        this.onChange();
      } catch (error) {
        this.message.textContent = `${error instanceof Error ? error.message : String(error)} The previous range remains active.`;
        this.message.dataset.error = "true";
        this.minimum.setAttribute("aria-invalid", "true");
        this.maximum.setAttribute("aria-invalid", "true");
      }
    });
  }

  public beginDataset(): void {
    this.channel = null;
    this.active = null;
    this.resolved = null;
  }

  public bind(channel: number, resolved: ResolvedScalarRange): void {
    this.resolved = resolved;
    if (
      this.channel !== channel ||
      this.active !== this.ranges.get(this.kind, channel)
    ) {
      this.channel = channel;
      this.syncFields();
    }
  }

  private syncFields(): void {
    if (this.channel === null || this.resolved === null) return;
    const active = this.ranges.get(this.kind, this.channel);
    this.active = active;
    const fixed =
      this.ranges.lastFixed(this.kind, this.channel) ??
      suggestedFixedRange(this.resolved);
    this.mode.value = active.mode;
    this.fixedFields.hidden = active.mode !== "fixed";
    this.minimum.value = String(fixed.minimum);
    this.maximum.value = String(fixed.maximum);
    this.minimum.removeAttribute("aria-invalid");
    this.maximum.removeAttribute("aria-invalid");
    this.message.textContent = "Apply or press Enter to use edited bounds.";
    delete this.message.dataset.error;
  }
}
