import "./style.css";

import { mapCellColors, type ColorMode } from "./color";
import { ColonyViewer } from "./colony-viewer";
import { signalSlice, sliceDimension, type SliceAxis } from "./grid";
import { DatasetPresentationState } from "./presentation-state";
import {
  LiveConnection,
  type LiveConnectionState,
  type LiveFrameMessage,
  type LiveMessage,
} from "./live";
import {
  MAX_SCENE_BYTES,
  channelLabel,
  parseScene,
  type SceneCell,
  type SceneFrame,
} from "./scene";

function required<T extends HTMLElement>(id: string): T {
  const value = document.querySelector<T>(`#${id}`);
  if (value === null) {
    throw new Error(`missing required element #${id}`);
  }
  return value;
}

const viewport = required<HTMLElement>("viewport");
const canvasHost = required<HTMLElement>("canvas-host");
const viewCubeElement = required<HTMLElement>("view-cube");
const fileInput = required<HTMLInputElement>("scene-file");
const fitButton = required<HTMLButtonElement>("fit-button");
const emptyState = required<HTMLElement>("empty-state");
const status = required<HTMLElement>("status");
const backendChip = required<HTMLElement>("backend-chip");
const timeChip = required<HTMLElement>("time-chip");
const countChip = required<HTMLElement>("count-chip");
const cellCount = required<HTMLElement>("cell-count");
const colorMode = required<HTMLSelectElement>("color-mode");
const speciesField = required<HTMLElement>("species-field");
const speciesChannel = required<HTMLSelectElement>("species-channel");
const colorLegend = required<HTMLElement>("color-legend");
const legendMinimum = required<HTMLElement>("legend-min");
const legendMaximum = required<HTMLElement>("legend-max");
const legendTitle = required<HTMLElement>("legend-title");
const signalSection = required<HTMLElement>("signal-section");
const signalVisible = required<HTMLInputElement>("signal-visible");
const signalChannel = required<HTMLSelectElement>("signal-channel");
const signalAxis = required<HTMLSelectElement>("signal-axis");
const signalRange = required<HTMLInputElement>("signal-slice");
const sliceValue = required<HTMLOutputElement>("slice-value");
const backendDetail = required<HTMLElement>("backend-detail");
const deviceDetail = required<HTMLElement>("device-detail");
const speciesCount = required<HTMLElement>("species-count");
const gridShape = required<HTMLElement>("grid-shape");
const selectionTitle = required<HTMLElement>("selection-title");
const selectionHint = required<HTMLElement>("selection-hint");
const clearSelection = required<HTMLButtonElement>("clear-selection");
const cellDetails = required<HTMLElement>("cell-details");
const speciesDetails = required<HTMLElement>("species-details");
const speciesValues = required<HTMLOListElement>("species-values");
const liveTransport = required<HTMLElement>("live-transport");
const liveLabel = required<HTMLElement>("live-label");
const livePlay = required<HTMLButtonElement>("live-play");
const liveStep = required<HTMLButtonElement>("live-step");
const liveReset = required<HTMLButtonElement>("live-reset");
const liveCheckpoint = required<HTMLButtonElement>("live-checkpoint");

let frame: SceneFrame | null = null;
let statusToken = 0;
let dragDepth = 0;
let liveConnected = false;
let livePlaying = false;
let liveCheckpointEnabled = false;
let liveConnection: LiveConnection | null = null;
const presentation = new DatasetPresentationState();

const viewer = new ColonyViewer(canvasHost, viewCubeElement, updateSelection);

function formatNumber(value: number): string {
  if (value === 0) {
    return "0";
  }
  const magnitude = Math.abs(value);
  return magnitude >= 10_000 || magnitude < 0.001
    ? value.toExponential(4)
    : value.toLocaleString(undefined, { maximumSignificantDigits: 7 });
}

function setStatus(message: string, kind: "info" | "error" = "info"): void {
  const token = ++statusToken;
  status.textContent = message;
  status.dataset.kind = kind;
  status.hidden = false;
  window.setTimeout(
    () => {
      if (token === statusToken) {
        status.hidden = true;
      }
    },
    kind === "error" ? 7000 : 3500,
  );
}

function options(
  select: HTMLSelectElement,
  count: number,
  label: (index: number) => string,
  selected: number,
): void {
  select.replaceChildren();
  for (let index = 0; index < count; index += 1) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = label(index);
    select.append(option);
  }
  select.value = String(selected);
}

function selectedInteger(
  control: HTMLSelectElement | HTMLInputElement,
): number {
  const value = Number.parseInt(control.value, 10);
  return Number.isFinite(value) ? value : 0;
}

function updateColors(): void {
  if (frame === null) {
    return;
  }
  const mode = colorMode.value as ColorMode;
  speciesField.hidden = mode !== "species";
  const mapping = mapCellColors(frame, {
    mode,
    speciesIndex: selectedInteger(speciesChannel),
  });
  viewer.setCellColors(mapping.colors);
  const scalar = mapping.minimum !== null && mapping.maximum !== null;
  colorLegend.hidden = !scalar;
  legendTitle.textContent = mapping.title;
  legendMinimum.textContent = scalar ? formatNumber(mapping.minimum ?? 0) : "—";
  legendMaximum.textContent = scalar ? formatNumber(mapping.maximum ?? 0) : "—";
}

function updateSignalRange(): void {
  if (frame?.signalGrid === null || frame === null) {
    return;
  }
  const axis = signalAxis.value as SliceAxis;
  const maximum = sliceDimension(frame.signalGrid, axis) - 1;
  signalRange.max = String(maximum);
  signalRange.value = String(presentation.forFrame(frame).signalSlice);
  sliceValue.value = signalRange.value;
}

function updateSignal(): void {
  if (frame?.signalGrid === null || frame === null || !signalVisible.checked) {
    viewer.setSignalSlice(null);
    return;
  }
  const axis = signalAxis.value as SliceAxis;
  const value = signalSlice(
    frame.signalGrid,
    selectedInteger(signalChannel),
    axis,
    selectedInteger(signalRange),
  );
  viewer.setSignalSlice(value);
}

function detail(label: string, value: string): HTMLDivElement {
  const row = document.createElement("div");
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = label;
  description.textContent = value;
  row.append(term, description);
  return row;
}

function updateSelection(cell: SceneCell | null): void {
  cellDetails.replaceChildren();
  speciesValues.replaceChildren();
  if (cell === null) {
    selectionTitle.textContent = "No cell selected";
    selectionHint.hidden = false;
    cellDetails.hidden = true;
    speciesDetails.hidden = true;
    clearSelection.disabled = true;
    return;
  }
  selectionTitle.textContent = `Cell ${cell.id}`;
  selectionHint.hidden = true;
  clearSelection.disabled = false;
  cellDetails.hidden = false;
  cellDetails.append(
    detail("Slot", String(cell.slot)),
    detail("Parent", cell.parentId ?? "Founder"),
    detail("Type", String(cell.cellType)),
    detail("Position", cell.position.map(formatNumber).join(", ")),
    detail("Direction", cell.direction.map(formatNumber).join(", ")),
    detail("Length", formatNumber(cell.length)),
    detail("Radius", formatNumber(cell.radius)),
    detail("Growth", formatNumber(cell.growthRate)),
    detail("Fixed", cell.fixed ? "Yes" : "No"),
  );
  speciesDetails.hidden = cell.species.length === 0;
  for (const [index, value] of cell.species.entries()) {
    const item = document.createElement("li");
    const label = document.createElement("span");
    const encoded = document.createElement("code");
    label.textContent =
      frame === null
        ? `Channel ${index}`
        : channelLabel(frame, "species", index);
    encoded.textContent = formatNumber(value);
    item.append(label, encoded);
    speciesValues.append(item);
  }
}

function presentScene(
  next: SceneFrame,
  label: string,
  {
    newDataset = false,
    announce = true,
  }: { newDataset?: boolean; announce?: boolean } = {},
): void {
  if (newDataset) {
    presentation.beginDataset();
    viewer.beginDataset();
  }
  const display = presentation.forFrame(next);
  frame = next;
  viewer.setFrame(next, newDataset);
  fitButton.disabled = false;
  colorMode.disabled = false;
  emptyState.hidden = true;
  backendChip.textContent = `${next.backend.kind.toUpperCase()} · ${next.backend.native ? "native" : "reference"}`;
  timeChip.textContent = `t = ${formatNumber(next.time)}`;
  countChip.textContent = `${next.cells.length.toLocaleString()} ${next.cells.length === 1 ? "cell" : "cells"}`;
  cellCount.textContent = next.cells.length.toLocaleString();
  backendDetail.textContent = next.backend.name;
  deviceDetail.textContent = `${next.backend.device} · ${next.backend.deviceIndex}`;
  speciesCount.textContent = String(next.speciesCount);
  gridShape.textContent =
    next.signalGrid === null ? "None" : next.signalGrid.shape.join(" × ");

  colorMode.value = display.colorMode;
  options(
    speciesChannel,
    next.speciesCount,
    (index) => channelLabel(next, "species", index),
    display.speciesChannel,
  );
  const speciesOption = colorMode.querySelector<HTMLOptionElement>(
    'option[value="species"]',
  );
  if (speciesOption !== null) {
    speciesOption.disabled = next.speciesCount === 0;
  }

  signalSection.hidden = next.signalGrid === null;
  signalVisible.checked = display.signalVisible;
  signalAxis.value = display.signalAxis;
  if (next.signalGrid !== null) {
    options(
      signalChannel,
      next.signalGrid.signalCount,
      (index) => channelLabel(next, "signals", index),
      display.signalChannel,
    );
    signalRange.max = String(
      sliceDimension(next.signalGrid, display.signalAxis) - 1,
    );
    signalRange.value = String(display.signalSlice);
    updateSignalRange();
  }
  updateColors();
  updateSignal();
  if (announce) {
    setStatus(`Opened ${label}`);
  }
}

async function loadFile(file: File): Promise<void> {
  if (file.size > MAX_SCENE_BYTES) {
    setStatus(
      `Scene exceeds the ${MAX_SCENE_BYTES.toLocaleString()}-byte limit`,
      "error",
    );
    return;
  }
  try {
    const next = await parseScene(await file.text());
    presentScene(next, file.name, { newDataset: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message, "error");
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file !== undefined) {
    void loadFile(file);
  }
  fileInput.value = "";
});

fitButton.addEventListener("click", () => viewer.fitColony());
clearSelection.addEventListener("click", () => viewer.selectCell(null));
colorMode.addEventListener("change", () => {
  presentation.preferences.colorMode = colorMode.value as ColorMode;
  updateColors();
});
speciesChannel.addEventListener("change", () => {
  presentation.preferences.speciesChannel = selectedInteger(speciesChannel);
  updateColors();
});
signalVisible.addEventListener("change", () => {
  presentation.preferences.signalVisible = signalVisible.checked;
  updateSignal();
});
signalChannel.addEventListener("change", () => {
  presentation.preferences.signalChannel = selectedInteger(signalChannel);
  updateSignal();
});
signalAxis.addEventListener("change", () => {
  presentation.preferences.signalAxis = signalAxis.value as SliceAxis;
  updateSignalRange();
  updateSignal();
});
signalRange.addEventListener("input", () => {
  presentation.preferences.signalSlice = selectedInteger(signalRange);
  sliceValue.value = signalRange.value;
  updateSignal();
});

for (const eventName of ["dragenter", "dragover"] as const) {
  viewport.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (eventName === "dragenter") {
      dragDepth += 1;
    }
    viewport.dataset.dragging = "true";
  });
}
viewport.addEventListener("dragleave", (event) => {
  event.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) {
    delete viewport.dataset.dragging;
  }
});
viewport.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  delete viewport.dataset.dragging;
  if (liveConnection !== null) {
    return;
  }
  const file = event.dataTransfer?.files[0];
  if (file !== undefined) {
    void loadFile(file);
  }
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    viewer.selectCell(null);
  }
});

function updateLiveControls(): void {
  livePlay.disabled = !liveConnected;
  liveStep.disabled = !liveConnected;
  liveReset.disabled = !liveConnected;
  liveCheckpoint.disabled = !liveConnected || !liveCheckpointEnabled;
  livePlay.textContent = livePlaying ? "Pause" : "Play";
}

function liveState(state: LiveConnectionState): void {
  liveConnected = state === "connected";
  if (!liveConnected) {
    livePlaying = false;
  }
  liveLabel.textContent =
    state === "connecting"
      ? "Connecting"
      : state === "connected"
        ? "Live"
        : "Disconnected";
  liveTransport.dataset.state = state;
  updateLiveControls();
  if (state === "closed") {
    setStatus("Live simulation disconnected", "error");
  }
}

function liveFrame(message: LiveFrameMessage): void {
  const first = frame === null;
  livePlaying = message.playing;
  liveCheckpointEnabled = message.checkpointEnabled;
  if (liveConnected) {
    liveLabel.textContent = message.playing ? "Running" : "Paused";
  }
  presentScene(message.frame, "live simulation", {
    newDataset: first,
    announce: first,
  });
  updateLiveControls();
}

function liveMessage(message: LiveMessage): void {
  if (message.type === "frame") {
    liveFrame(message);
  } else if (message.type === "checkpoint") {
    setStatus(`Checkpoint saved to ${message.path}`);
  } else {
    setStatus(message.message, "error");
  }
}

function sendLive(
  command:
    | { type: "play" | "pause" | "reset" | "checkpoint" }
    | { type: "step"; steps: number },
): void {
  try {
    liveConnection?.send(command);
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), "error");
  }
}

const liveToken = new URL(window.location.href).searchParams.get("token");
if (liveToken !== null) {
  liveTransport.hidden = false;
  for (const action of document.querySelectorAll<HTMLElement>(
    ".scene-open-action",
  )) {
    action.hidden = true;
  }
  const emptyTitle = emptyState.querySelector<HTMLElement>("h1");
  const emptyDescription = emptyState.querySelector<HTMLElement>("p");
  if (emptyTitle !== null && emptyDescription !== null) {
    emptyTitle.textContent = "Connecting to simulation";
    emptyDescription.textContent =
      "Waiting for the first verified scene frame from the local engine.";
  }
  liveConnection = new LiveConnection(liveToken, {
    message: liveMessage,
    state: liveState,
    protocolError: (message) => setStatus(message, "error"),
  });
  liveConnection.connect();
}

livePlay.addEventListener("click", () =>
  sendLive({ type: livePlaying ? "pause" : "play" }),
);
liveStep.addEventListener("click", () => sendLive({ type: "step", steps: 1 }));
liveReset.addEventListener("click", () => sendLive({ type: "reset" }));
liveCheckpoint.addEventListener("click", () =>
  sendLive({ type: "checkpoint" }),
);

window.addEventListener(
  "beforeunload",
  () => {
    liveConnection?.close();
    viewer.dispose();
  },
  { once: true },
);
