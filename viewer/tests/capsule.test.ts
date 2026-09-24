import { describe, expect, it } from "vitest";
import {
  BufferGeometry,
  InstancedMesh,
  Matrix3,
  Matrix4,
  MeshBasicMaterial,
  Raycaster,
  Vector3,
} from "three";

import { capsuleGeometries, CapsuleTransform } from "../src/capsule";
import type { SceneCell } from "../src/scene";

function cell(
  length: number,
  radius: number,
  direction: readonly [number, number, number],
): SceneCell {
  return {
    id: "1",
    parentId: null,
    slot: 0,
    position: [-2, 3, 4],
    direction,
    length,
    radius,
    cellType: 0,
    growthRate: 0,
    fixed: false,
    species: [],
  };
}

function vertex(
  geometry: BufferGeometry,
  attribute: "position" | "normal",
  index: number,
): Vector3 {
  return new Vector3().fromBufferAttribute(
    geometry.getAttribute(attribute),
    index,
  );
}

describe("matched capsule surfaces", () => {
  it("uses only an open cylinder and outward hemispheres, with identical seam normals and samples", () => {
    const [cylinder, lower, upper] = capsuleGeometries();
    const width = cylinder.parameters.radialSegments + 1;
    const capRows = upper.parameters.heightSegments;
    expect(cylinder.parameters.openEnded).toBe(true);
    expect(cylinder.index!.count / 3).toBe(2 * (width - 1));
    for (let j = 0; j < width; j++) {
      for (const [cap, row, cylinderRow] of [
        [lower, 0, 1],
        [upper, capRows, 0],
      ] as const) {
        const side = vertex(cylinder, "position", cylinderRow * width + j);
        side.y = 0;
        expect(vertex(cap, "position", row * width + j).toArray()).toEqual(
          side.toArray(),
        );
        expect(vertex(cap, "normal", row * width + j).toArray()).toEqual(
          vertex(cylinder, "normal", cylinderRow * width + j).toArray(),
        );
      }
    }
    for (const [cap, sign] of [
      [lower, -1],
      [upper, 1],
    ] as const) {
      for (let j = 0; j < cap.getAttribute("position").count; j++) {
        expect(vertex(cap, "position", j).y * sign).toBeGreaterThanOrEqual(0);
      }
      const indices = cap.index!;
      for (let j = 0; j < indices.count; j += 3) {
        const a = vertex(cap, "position", indices.getX(j));
        const b = vertex(cap, "position", indices.getX(j + 1));
        const c = vertex(cap, "position", indices.getX(j + 2));
        const outward = b.clone().sub(a).cross(c.clone().sub(a));
        expect(outward.dot(a.clone().add(b).add(c))).toBeGreaterThan(0);
      }
    }
    for (const geometry of [cylinder, lower, upper]) geometry.dispose();
  });

  it.each([
    [0, 0.5],
    [0.0001, 0.08],
    [0.2, 2],
    [3, 0.5],
    [100, 0.01],
  ])(
    "preserves the capsule surface, radius, and axial extent for length %s and radius %s",
    (length, radius) => {
      const geometries = capsuleGeometries();
      for (const direction of [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
        [2, -3, 4],
      ] as const) {
        const c = cell(length, radius, direction);
        const transform = new CapsuleTransform();
        transform.update(c);
        const center = new Vector3().fromArray(c.position);
        const axis = new Vector3().fromArray(direction).normalize();
        const axial: number[] = [];
        for (const [part, geometry] of geometries.entries()) {
          // Use the same Float32 instance-matrix representation sent to WebGL.
          const matrix = new Matrix4().fromArray(
            new Float32Array(transform.matrices[part]!.elements),
          );
          for (let j = 0; j < geometry.getAttribute("position").count; j++) {
            const point = vertex(geometry, "position", j).applyMatrix4(matrix);
            const local = point.clone().sub(center);
            const along = local.dot(axis);
            axial.push(along);
            const nearest = center
              .clone()
              .addScaledVector(
                axis,
                Math.max(-length / 2, Math.min(length / 2, along)),
              );
            expect(Math.abs(point.distanceTo(nearest) - radius)).toBeLessThan(
              1e-5,
            );
            expect(
              transform.bounds
                .clone()
                .expandByScalar(1e-5)
                .containsPoint(point),
            ).toBe(true);
          }
        }
        expect(Math.min(...axial)).toBeCloseTo(-length / 2 - radius, 5);
        expect(Math.max(...axial)).toBeCloseTo(length / 2 + radius, 5);
        const width = geometries[0].parameters.radialSegments + 1;
        for (const [part, row, cylinderRow] of [
          [1, 0, 1],
          [2, geometries[2].parameters.heightSegments, 0],
        ] as const) {
          for (let j = 0; j < width; j++) {
            const a = vertex(
              geometries[0],
              "position",
              cylinderRow * width + j,
            ).applyMatrix4(transform.matrices[0]);
            const b = vertex(
              geometries[part],
              "position",
              row * width + j,
            ).applyMatrix4(transform.matrices[part]);
            expect(a.distanceTo(b)).toBeLessThan(1e-10);
            if (length > 0) {
              const normalA = vertex(
                geometries[0],
                "normal",
                cylinderRow * width + j,
              )
                .applyMatrix3(
                  new Matrix3().getNormalMatrix(transform.matrices[0]),
                )
                .normalize();
              const normalB = vertex(
                geometries[part],
                "normal",
                row * width + j,
              )
                .applyMatrix3(
                  new Matrix3().getNormalMatrix(transform.matrices[part]),
                )
                .normalize();
              expect(normalA.distanceTo(normalB)).toBeLessThan(1e-10);
            }
          }
        }
      }
      for (const geometry of geometries) geometry.dispose();
    },
  );

  it.each([0, 3])(
    "ray-picks the outer hemisphere tips and preserves instance IDs for length %s",
    (length) => {
      const c = cell(length, 0.5, [2, -3, 4]);
      const transform = new CapsuleTransform();
      transform.update(c);
      const material = new MeshBasicMaterial();
      const meshes = capsuleGeometries().map((geometry, i) => {
        const mesh = new InstancedMesh(geometry, material, 1);
        mesh.setMatrixAt(0, transform.matrices[i]!);
        mesh.computeBoundingSphere();
        return mesh;
      });
      const center = new Vector3().fromArray(c.position);
      const axis = new Vector3().fromArray(c.direction).normalize();
      for (const sign of [-1, 1]) {
        const ray = new Raycaster(
          center.clone().addScaledVector(axis, sign * 10),
          axis.clone().multiplyScalar(-sign),
        );
        const hits = ray.intersectObjects(meshes);
        expect(hits.length).toBeGreaterThan(0);
        expect(hits[0]!.instanceId).toBe(0);
        expect(hits[0]!.distance).toBeCloseTo(10 - length / 2 - c.radius, 5);
      }
      for (const mesh of meshes) {
        mesh.dispose();
        mesh.geometry.dispose();
      }
      material.dispose();
    },
  );

  it("inflates selection radius without moving cap centers or stretching ends", () => {
    const transform = new CapsuleTransform();
    const c = cell(3, 0.5, [2, -3, 4]);
    transform.update(c, 1.08);
    for (const matrix of transform.matrices.slice(1)) {
      const position = new Vector3().setFromMatrixPosition(matrix);
      expect(
        position.distanceTo(new Vector3().fromArray(c.position)),
      ).toBeCloseTo(c.length / 2, 12);
      for (const scale of new Vector3().setFromMatrixScale(matrix).toArray()) {
        expect(scale).toBeCloseTo(0.54, 12);
      }
    }
  });
});
