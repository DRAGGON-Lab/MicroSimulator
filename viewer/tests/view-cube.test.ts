import { describe, expect, it } from "vitest";
import { Quaternion, Vector3 } from "three";

import {
  canonicalViewQuaternion,
  DEFAULT_VIEW_TARGET,
  dragOrbitOffset,
  faceLabelBasis,
  interpolateViewDirection,
  interpolateViewOrientation,
  shortestViewQuaternion,
  targetDirection,
  VIEW_CUBE_TARGETS,
} from "../src/view-cube";

describe("view cube targets", () => {
  it("defines six faces and eight flat corners", () => {
    expect(
      VIEW_CUBE_TARGETS.filter((target) => target.kind === "face"),
    ).toHaveLength(6);
    expect(
      VIEW_CUBE_TARGETS.filter((target) => target.kind === "corner"),
    ).toHaveLength(8);
    expect(new Set(VIEW_CUBE_TARGETS.map((target) => target.id)).size).toBe(14);
  });

  it("uses the viewer's Z-up and negative-Y-front convention", () => {
    const targets = new Map(
      VIEW_CUBE_TARGETS.map((target) => [target.id, target]),
    );
    expect(targets.get("top")?.direction).toEqual([0, 0, 1]);
    expect(targets.get("bottom")?.direction).toEqual([0, 0, -1]);
    expect(targets.get("front")?.direction).toEqual([0, -1, 0]);
    expect(targets.get("back")?.direction).toEqual([0, 1, 0]);
    expect(targets.get("right")?.direction).toEqual([1, 0, 0]);
    expect(targets.get("left")?.direction).toEqual([-1, 0, 0]);
  });

  it("uses the exact top-front-right corner as the fitted view", () => {
    expect(DEFAULT_VIEW_TARGET.id).toBe("top-front-right");
    expect(DEFAULT_VIEW_TARGET.direction).toEqual([1, -1, 1]);
    expect(targetDirection(DEFAULT_VIEW_TARGET).length()).toBeCloseTo(1, 12);
  });

  it("orients every static face label upright in its selected view", () => {
    for (const target of VIEW_CUBE_TARGETS.filter(
      (candidate) => candidate.kind === "face",
    )) {
      const { right, up, normal } = faceLabelBasis(target.direction);
      expect(right.dot(up)).toBeCloseTo(0, 12);
      expect(right.dot(normal)).toBeCloseTo(0, 12);
      expect(up.dot(normal)).toBeCloseTo(0, 12);
      expect(
        new Vector3().crossVectors(right, up).distanceTo(normal),
      ).toBeLessThan(1e-12);
      expect(up.toArray()).toEqual(
        Math.abs(normal.z) > 0.99 ? [0, 1, 0] : [0, 0, 1],
      );
    }
  });
});

describe("view direction interpolation", () => {
  it("reaches exact normalized endpoints", () => {
    const start = new Vector3(1, -2, 3);
    const end = new Vector3(-4, 2, -1);
    expect(
      interpolateViewDirection(start, end, 0).distanceTo(
        start.clone().normalize(),
      ),
    ).toBeLessThan(1e-12);
    expect(
      interpolateViewDirection(start, end, 1).distanceTo(
        end.clone().normalize(),
      ),
    ).toBeLessThan(1e-12);
  });

  it("stays finite and normalized between opposite views", () => {
    const direction = interpolateViewDirection(
      new Vector3(0, 0, 1),
      new Vector3(0, 0, -1),
      0.5,
    );
    expect(direction.toArray().every(Number.isFinite)).toBe(true);
    expect(direction.length()).toBeCloseTo(1, 12);
  });

  it("clamps interpolation outside the animation interval", () => {
    const start = new Vector3(1, 0, 0);
    const end = new Vector3(0, 1, 0);
    expect(interpolateViewDirection(start, end, -1).toArray()).toEqual([
      1, 0, 0,
    ]);
    expect(
      interpolateViewDirection(start, end, 2).distanceTo(end),
    ).toBeLessThan(1e-12);
  });
});

describe("view orientation transitions", () => {
  it("uses the minimum rotation for a single-click view change", () => {
    const start = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), 0.63);
    const endDirection = new Vector3(1, -1, 1).normalize();
    const end = shortestViewQuaternion(start, endDirection);
    const startDirection = new Vector3(0, 0, 1).applyQuaternion(start);

    expect(
      new Vector3(0, 0, 1).applyQuaternion(end).distanceTo(endDirection),
    ).toBeLessThan(1e-12);
    expect(start.angleTo(end)).toBeCloseTo(
      startDirection.angleTo(endDirection),
      12,
    );
  });

  it("levels every face label after a double-click", () => {
    for (const target of VIEW_CUBE_TARGETS.filter(
      (candidate) => candidate.kind === "face",
    )) {
      const orientation = canonicalViewQuaternion(target);
      const screenUp = new Vector3(0, 1, 0).applyQuaternion(orientation);
      const viewDirection = new Vector3(0, 0, 1).applyQuaternion(orientation);
      const label = faceLabelBasis(target.direction);

      expect(screenUp.distanceTo(label.up)).toBeLessThan(1e-12);
      expect(viewDirection.distanceTo(label.normal)).toBeLessThan(1e-12);
    }
  });

  it("interpolates orientation smoothly and monotonically", () => {
    const start = new Quaternion();
    const end = canonicalViewQuaternion(DEFAULT_VIEW_TARGET);
    const samples = [0, 0.25, 0.5, 0.75, 1].map((fraction) =>
      interpolateViewOrientation(start, end, fraction),
    );

    expect(samples[0]?.angleTo(start)).toBeLessThan(1e-12);
    expect(samples.at(-1)?.angleTo(end)).toBeLessThan(1e-12);
    for (let index = 1; index < samples.length; index += 1) {
      expect(samples[index]?.angleTo(start)).toBeGreaterThan(
        samples[index - 1]?.angleTo(start) ?? -1,
      );
    }
  });
});

describe("view cube dragging", () => {
  it("orbits around a Z-up target without changing camera distance", () => {
    const offset = new Vector3(8, -10, 7);
    const result = dragOrbitOffset(offset, new Vector3(0, 0, 1), 24, -13);
    expect(result.length()).toBeCloseTo(offset.length(), 12);
    expect(result.distanceTo(offset)).toBeGreaterThan(0.1);
    expect(result.toArray().every(Number.isFinite)).toBe(true);
  });

  it("keeps vertical dragging away from the orbit poles", () => {
    const result = dragOrbitOffset(
      new Vector3(1, 0, 0),
      new Vector3(0, 0, 1),
      0,
      1_000_000,
    );
    expect(result.length()).toBeCloseTo(1, 12);
    expect(Math.abs(result.z)).toBeLessThan(1);
  });
});
