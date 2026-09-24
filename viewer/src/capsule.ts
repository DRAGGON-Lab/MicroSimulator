import {
  Box3,
  CylinderGeometry,
  Matrix4,
  Quaternion,
  SphereGeometry,
  Vector3,
} from "three";

import type { SceneCell } from "./scene";

/** Three shared surfaces, with matching radial samples at both tangent joins. */
export function capsuleGeometries(
  radialSegments = 24,
  capSegments = 6,
): readonly [CylinderGeometry, SphereGeometry, SphereGeometry] {
  if (
    !Number.isInteger(radialSegments) ||
    radialSegments < 3 ||
    !Number.isInteger(capSegments) ||
    capSegments < 2
  ) {
    throw new RangeError(
      "capsule segments require radial >= 3 and cap >= 2 integers",
    );
  }
  const cylinder = new CylinderGeometry(1, 1, 1, radialSegments, 1, true);
  const lower = new SphereGeometry(
    1,
    radialSegments,
    capSegments,
    0,
    Math.PI * 2,
    Math.PI / 2,
    Math.PI / 2,
  );
  const upper = new SphereGeometry(
    1,
    radialSegments,
    capSegments,
    0,
    Math.PI * 2,
    0,
    Math.PI / 2,
  );
  const ring = cylinder.getAttribute("position");
  for (const [cap, equator] of [
    [lower, 0],
    [upper, capSegments],
  ] as const) {
    // SphereGeometry and CylinderGeometry use different azimuth origins.
    // Matching phases alone is insufficient: snap the equator to exactly the
    // same Float32 ring and radial normals, including its duplicated UV seam.
    cap.rotateY(Math.PI / 2);
    const position = cap.getAttribute("position");
    const normal = cap.getAttribute("normal");
    for (let index = 0; index <= radialSegments; index++) {
      const vertex = equator * (radialSegments + 1) + index;
      position.setXYZ(vertex, ring.getX(index), 0, ring.getZ(index));
      normal.setXYZ(vertex, ring.getX(index), 0, ring.getZ(index));
    }
  }
  return [cylinder, lower, upper];
}

/** Reusable scratch transforms; cap radius is never stretched by rod length. */
export class CapsuleTransform {
  public readonly matrices = [
    new Matrix4(),
    new Matrix4(),
    new Matrix4(),
  ] as const;
  public readonly bounds = new Box3();
  private readonly center = new Vector3();
  private readonly direction = new Vector3();
  private readonly endpoint = new Vector3();
  private readonly first = new Vector3();
  private readonly second = new Vector3();
  private readonly scale = new Vector3();
  private readonly orientation = new Quaternion();
  private readonly up = new Vector3(0, 1, 0);

  public update(cell: SceneCell, radiusScale = 1): void {
    const radius = cell.radius * radiusScale;
    this.center.fromArray(cell.position);
    this.direction.fromArray(cell.direction).normalize();
    this.orientation.setFromUnitVectors(this.up, this.direction);
    this.endpoint.copy(this.direction).multiplyScalar(cell.length / 2);
    this.first.copy(this.center).sub(this.endpoint);
    this.second.copy(this.center).add(this.endpoint);
    this.matrices[0].compose(
      this.center,
      this.orientation,
      this.scale.set(radius, cell.length, radius),
    );
    this.scale.setScalar(radius);
    this.matrices[1].compose(this.first, this.orientation, this.scale);
    this.matrices[2].compose(this.second, this.orientation, this.scale);
    this.bounds
      .makeEmpty()
      .expandByPoint(this.first)
      .expandByPoint(this.second)
      .expandByScalar(radius);
  }
}
