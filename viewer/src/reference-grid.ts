import { Box3, Vector3 } from "three";

import type { SceneFrame, Vector3 as SceneVector3 } from "./scene";

export const REFERENCE_GRID_DIVISIONS = 20;
export const MINIMUM_REFERENCE_GRID_EXTENT = 10;

export interface ReferenceGridLayout {
  readonly extent: number;
  readonly spacing: number;
  readonly position: SceneVector3;
}

/** Finite device geometry is independent of the colony's current envelope. */
function initialBounds(frame: SceneFrame): Box3 {
  const device = new Box3();
  const include = (center: SceneVector3, extent: SceneVector3): void => {
    const origin = new Vector3().fromArray(center);
    const radius = new Vector3().fromArray(extent);
    device.union(
      new Box3(origin.clone().sub(radius), origin.clone().add(radius)),
    );
  };
  for (const box of frame.constraints.boxes) {
    include(box.center, box.halfExtents);
  }
  for (const sphere of frame.constraints.spheres) {
    include(sphere.center, [sphere.radius, sphere.radius, sphere.radius]);
  }
  for (const cylinder of frame.constraints.cylinders) {
    include(cylinder.center, [
      cylinder.radius,
      cylinder.radius,
      cylinder.halfHeight,
    ]);
  }
  // Planes are infinite. Their visualization extent must never size the grid.
  if (!device.isEmpty()) {
    return device;
  }
  const colony = new Box3();
  for (const cell of frame.cells) {
    const center = new Vector3().fromArray(cell.position);
    const half = new Vector3()
      .fromArray(cell.direction)
      .normalize()
      .multiplyScalar(cell.length / 2);
    colony.union(
      new Box3()
        .setFromPoints([center.clone().sub(half), center.clone().add(half)])
        .expandByScalar(cell.radius),
    );
  }
  return colony.isEmpty() ? new Box3(new Vector3(), new Vector3()) : colony;
}

export function initialReferenceGrid(frame: SceneFrame): ReferenceGridLayout {
  const bounds = initialBounds(frame);
  const size = bounds.getSize(new Vector3());
  const center = bounds.getCenter(new Vector3());
  const extent = Math.max(size.x, size.y, MINIMUM_REFERENCE_GRID_EXTENT);
  return Object.freeze({
    extent,
    spacing: extent / REFERENCE_GRID_DIVISIONS,
    position: Object.freeze([
      center.x,
      center.y,
      Math.min(bounds.min.z, 0) - 0.01,
    ]) as SceneVector3,
  });
}

/** Geometry is chosen from the first frame, including an empty first frame. */
export class DatasetReferenceGrid {
  private layout: ReferenceGridLayout | null = null;

  public beginDataset(): void {
    this.layout = null;
  }

  public forFrame(frame: SceneFrame): ReferenceGridLayout {
    this.layout ??= initialReferenceGrid(frame);
    return this.layout;
  }
}
