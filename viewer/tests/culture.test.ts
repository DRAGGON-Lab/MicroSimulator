import canonicalize from "canonicalize";
import { describe, expect, it } from "vitest";
import { parseScene } from "../src/scene";
import currentScene from "./fixtures/culture-v5.scene.json?raw";
import pythonScene from "./fixtures/media-v4.scene.json?raw";

describe("physical culture scenes", () => {
  it("reads native nutrient uptake and growth in a version-5 culture", async () => {
    const frame = await parseScene(currentScene);
    expect(frame.culture?.soluteAmountUnits).toEqual(["mol", "mol"]);
    const cell = frame.culture!.cells[0]!;
    expect(cell.dryBiomassG).toBeGreaterThan(0);
    expect(cell.realizedSpecificRatePerHour).toBeGreaterThan(0);
    expect(cell.uptakeTotals[0]).toBeGreaterThan(0);
    expect(cell.uptakeTotals[1]).toBe(0);
    expect(cell.biomassProducedG / cell.uptakeTotals[0]!).toBeCloseTo(90, 8);
  });
  it("retains Python fragment amounts, poses and physical units", async () => {
    const frame = await parseScene(pythonScene);
    expect(frame.culture?.lengthUnitM).toBe(1e-6);
    expect(frame.culture?.cells[0]?.biochemicalVolume).toBe(80);
    expect(frame.culture?.cells[0]?.speciesAmounts).toEqual([240]);

    for (const [i, component] of frame.culture!.cells[0]!.orientation.entries())
      expect(component).toBeCloseTo(i === 0 ? 1 : 0, 12);

    expect(frame.culture?.fragments).toHaveLength(208);

    for (const fragment of frame.culture!.fragments)
      expect(fragment.amounts[0]! / fragment.volume).toBeCloseTo(2, 12);

    expect(frame.signalGrid?.signalCount).toBe(1);
  });
  it("rejects a signed or inconsistent fragment amount after digest verification", async () => {
    const document = JSON.parse(pythonScene) as {
      frame: { media: { fragments: { amounts: number[] }[] } };
      integrity: { frame: string };
    };
    document.frame.media.fragments[0]!.amounts[0] = -1;
    const digest = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(canonicalize(document.frame)),
    );
    document.integrity.frame = [...new Uint8Array(digest)]
      .map((x) => x.toString(16).padStart(2, "0"))
      .join("");
    await expect(parseScene(JSON.stringify(document))).rejects.toThrow(
      "invalid culture amount array",
    );
  });
});
