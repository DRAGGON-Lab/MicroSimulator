import {
  BufferGeometry,
  CanvasTexture,
  Color,
  DoubleSide,
  EdgesGeometry,
  Float32BufferAttribute,
  Group,
  LineBasicMaterial,
  LineSegments,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  OrthographicCamera,
  Quaternion,
  Raycaster,
  Scene,
  Shape,
  ShapeGeometry,
  Spherical,
  SRGBColorSpace,
  Vector2,
  Vector3,
  type Camera,
  type WebGLRenderer,
} from "three";

export type ViewCubeTargetKind = "face" | "corner";

export interface ViewCubeTarget {
  readonly id: string;
  readonly kind: ViewCubeTargetKind;
  readonly label: string;
  readonly axisLabel: string;
  readonly direction: readonly [number, number, number];
}

const FACE_TARGETS: readonly ViewCubeTarget[] = [
  {
    id: "right",
    kind: "face",
    label: "Right",
    axisLabel: "+X",
    direction: [1, 0, 0],
  },
  {
    id: "left",
    kind: "face",
    label: "Left",
    axisLabel: "−X",
    direction: [-1, 0, 0],
  },
  {
    id: "front",
    kind: "face",
    label: "Front",
    axisLabel: "−Y",
    direction: [0, -1, 0],
  },
  {
    id: "back",
    kind: "face",
    label: "Back",
    axisLabel: "+Y",
    direction: [0, 1, 0],
  },
  {
    id: "top",
    kind: "face",
    label: "Top",
    axisLabel: "+Z",
    direction: [0, 0, 1],
  },
  {
    id: "bottom",
    kind: "face",
    label: "Bottom",
    axisLabel: "−Z",
    direction: [0, 0, -1],
  },
];

const X_NAMES = new Map<number, [string, string]>([
  [1, ["Right", "+X"]],
  [-1, ["Left", "−X"]],
]);
const Y_NAMES = new Map<number, [string, string]>([
  [1, ["Back", "+Y"]],
  [-1, ["Front", "−Y"]],
]);
const Z_NAMES = new Map<number, [string, string]>([
  [1, ["Top", "+Z"]],
  [-1, ["Bottom", "−Z"]],
]);

function cornerTarget(x: number, y: number, z: number): ViewCubeTarget {
  const xName = X_NAMES.get(x);
  const yName = Y_NAMES.get(y);
  const zName = Z_NAMES.get(z);
  if (xName === undefined || yName === undefined || zName === undefined) {
    throw new Error("corner direction components must be +1 or -1");
  }
  return {
    id: `${z > 0 ? "top" : "bottom"}-${y < 0 ? "front" : "back"}-${x > 0 ? "right" : "left"}`,
    kind: "corner",
    label: `${zName[0]} · ${yName[0]} · ${xName[0]}`,
    axisLabel: `${xName[1]}, ${yName[1]}, ${zName[1]}`,
    direction: [x, y, z],
  };
}

const CORNER_TARGETS: readonly ViewCubeTarget[] = [-1, 1].flatMap((z) =>
  [-1, 1].flatMap((y) => [-1, 1].map((x) => cornerTarget(x, y, z))),
);

export const VIEW_CUBE_TARGETS: readonly ViewCubeTarget[] = [
  ...FACE_TARGETS,
  ...CORNER_TARGETS,
];

export const DEFAULT_VIEW_TARGET: ViewCubeTarget = (() => {
  const target = VIEW_CUBE_TARGETS.find(
    (candidate) => candidate.id === "top-front-right",
  );
  if (target === undefined) {
    throw new Error("default view cube target is missing");
  }
  return target;
})();

export function targetDirection(target: ViewCubeTarget): Vector3 {
  return new Vector3(...target.direction).normalize();
}

export function canonicalViewQuaternion(
  target: ViewCubeTarget,
  result = new Quaternion(),
): Quaternion {
  const normal = targetDirection(target);
  const preferredUp =
    Math.abs(normal.z) > 0.99 ? new Vector3(0, 1, 0) : new Vector3(0, 0, 1);
  const up = preferredUp
    .addScaledVector(normal, -preferredUp.dot(normal))
    .normalize();
  const right = new Vector3().crossVectors(up, normal).normalize();
  return result.setFromRotationMatrix(
    new Matrix4().makeBasis(right, up, normal),
  );
}

export function shortestViewQuaternion(
  start: Quaternion,
  endDirection: Vector3,
  result = new Quaternion(),
): Quaternion {
  const startDirection = new Vector3(0, 0, 1).applyQuaternion(start);
  const turn = new Quaternion().setFromUnitVectors(
    startDirection,
    endDirection.clone().normalize(),
  );
  return result.copy(turn).multiply(start).normalize();
}

export function interpolateViewOrientation(
  start: Quaternion,
  end: Quaternion,
  fraction: number,
  result = new Quaternion(),
): Quaternion {
  return result
    .slerpQuaternions(start, end, Math.min(Math.max(fraction, 0), 1))
    .normalize();
}

export function interpolateViewDirection(
  start: Vector3,
  end: Vector3,
  fraction: number,
  result = new Vector3(),
): Vector3 {
  const from = start.clone().normalize();
  const to = end.clone().normalize();
  const rotation = new Quaternion().setFromUnitVectors(from, to);
  const partial = new Quaternion().slerpQuaternions(
    new Quaternion(),
    rotation,
    Math.min(Math.max(fraction, 0), 1),
  );
  return result.copy(from).applyQuaternion(partial).normalize();
}

export function dragOrbitOffset(
  offset: Vector3,
  up: Vector3,
  deltaX: number,
  deltaY: number,
  result = new Vector3(),
): Vector3 {
  const toOrbitSpace = new Quaternion().setFromUnitVectors(
    up,
    new Vector3(0, 1, 0),
  );
  result.copy(offset).applyQuaternion(toOrbitSpace);
  const spherical = new Spherical().setFromVector3(result);
  const radiansPerPixel = 0.012;
  spherical.theta -= deltaX * radiansPerPixel;
  spherical.phi = Math.min(
    Math.max(spherical.phi - deltaY * radiansPerPixel, 1.0e-4),
    Math.PI - 1.0e-4,
  );
  return result
    .setFromSpherical(spherical)
    .applyQuaternion(toOrbitSpace.invert());
}

function labelTexture(target: ViewCubeTarget): CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const context = canvas.getContext("2d");
  if (context === null) {
    throw new Error("2D canvas is unavailable for view cube labels");
  }
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#f2f5f3";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.font = "600 42px Inter, system-ui, sans-serif";
  context.fillText(target.label, 128, 108);
  context.fillStyle = "#8be0bd";
  context.font = "600 31px ui-monospace, monospace";
  context.fillText(target.axisLabel, 128, 157);
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

function faceGeometry(): ShapeGeometry {
  const inner = 0.44;
  const shape = new Shape();
  shape.moveTo(-inner, -1);
  shape.lineTo(inner, -1);
  shape.lineTo(1, -inner);
  shape.lineTo(1, inner);
  shape.lineTo(inner, 1);
  shape.lineTo(-inner, 1);
  shape.lineTo(-1, inner);
  shape.lineTo(-1, -inner);
  shape.closePath();
  const geometry = new ShapeGeometry(shape);
  const positions = geometry.getAttribute("position");
  const uv: number[] = [];
  for (let index = 0; index < positions.count; index += 1) {
    uv.push((positions.getX(index) + 1) / 2, (positions.getY(index) + 1) / 2);
  }
  geometry.setAttribute("uv", new Float32BufferAttribute(uv, 2));
  return geometry;
}

function cornerGeometry(
  direction: readonly [number, number, number],
): BufferGeometry {
  const [x, y, z] = direction;
  const inner = 0.44;
  const geometry = new BufferGeometry();
  geometry.setAttribute(
    "position",
    new Float32BufferAttribute(
      [x, y, z * inner, x, y * inner, z, x * inner, y, z],
      3,
    ),
  );
  geometry.setIndex([0, 1, 2]);
  geometry.computeVertexNormals();
  return geometry;
}

export function faceLabelBasis(direction: readonly [number, number, number]): {
  right: Vector3;
  up: Vector3;
  normal: Vector3;
} {
  const normal = new Vector3(...direction).normalize();
  const up =
    Math.abs(normal.z) > 0.99 ? new Vector3(0, 1, 0) : new Vector3(0, 0, 1);
  const right = new Vector3()
    .copy(normal)
    .multiplyScalar(-1)
    .cross(up)
    .normalize();
  return { right, up, normal };
}

function orientFace(
  mesh: Mesh,
  direction: readonly [number, number, number],
): void {
  const { right, up, normal } = faceLabelBasis(direction);
  mesh.quaternion.setFromRotationMatrix(
    new Matrix4().makeBasis(right, up, normal),
  );
  mesh.position.copy(normal);
}

function disposeObject(group: Group): void {
  group.traverse((object) => {
    if (object instanceof Mesh || object instanceof LineSegments) {
      object.geometry.dispose();
      const materials = Array.isArray(object.material)
        ? object.material
        : [object.material];
      for (const material of materials) {
        if (material instanceof MeshBasicMaterial) {
          material.map?.dispose();
        }
        material.dispose();
      }
    }
  });
}

export class ViewCube {
  private readonly scene = new Scene();
  private readonly camera = new OrthographicCamera(
    -1.72,
    1.72,
    1.72,
    -1.72,
    0.1,
    20,
  );
  private readonly cube = new Group();
  private readonly raycaster = new Raycaster();
  private readonly pointer = new Vector2();
  private readonly targets: Mesh[] = [];
  private readonly targetByMesh = new Map<Mesh, ViewCubeTarget>();
  private hovered: Mesh | null = null;
  private pointerId: number | null = null;
  private pointerOrigin = new Vector2();
  private previousPointer = new Vector2();
  private dragged = false;
  private pendingClick: {
    readonly target: ViewCubeTarget;
    readonly timer: number;
  } | null = null;

  public constructor(
    private readonly element: HTMLElement,
    private readonly onSelect: (
      target: ViewCubeTarget,
      alignLabel: boolean,
    ) => void,
    private readonly onDrag: (deltaX: number, deltaY: number) => void,
  ) {
    this.scene.background = new Color(0x141918);
    this.camera.position.set(0, 0, 5);
    this.camera.lookAt(0, 0, 0);
    this.scene.add(this.cube);

    for (const target of FACE_TARGETS) {
      const material = new MeshBasicMaterial({
        color: 0x35423d,
        map: labelTexture(target),
        side: DoubleSide,
        toneMapped: false,
      });
      const mesh = new Mesh(faceGeometry(), material);
      orientFace(mesh, target.direction);
      this.addTarget(mesh, target);
    }
    for (const target of CORNER_TARGETS) {
      const mesh = new Mesh(
        cornerGeometry(target.direction),
        new MeshBasicMaterial({
          color: 0x60736b,
          side: DoubleSide,
          toneMapped: false,
        }),
      );
      this.addTarget(mesh, target);
    }
    this.element.addEventListener("pointermove", this.handlePointerMove);
    this.element.addEventListener("pointerleave", this.handlePointerLeave);
    this.element.addEventListener("pointerdown", this.handlePointerDown);
    this.element.addEventListener("pointerup", this.handlePointerUp);
    this.element.addEventListener("pointercancel", this.handlePointerCancel);
  }

  public setVisible(visible: boolean): void {
    this.element.hidden = !visible;
    if (!visible) {
      this.setHovered(null);
    }
  }

  public sync(camera: Camera): void {
    this.cube.quaternion.copy(camera.quaternion).invert();
  }

  public render(renderer: WebGLRenderer): void {
    if (this.element.hidden) {
      return;
    }
    const canvasBounds = renderer.domElement.getBoundingClientRect();
    const bounds = this.element.getBoundingClientRect();
    const left = bounds.left - canvasBounds.left;
    const bottom = canvasBounds.bottom - bounds.bottom;
    renderer.setScissorTest(true);
    renderer.setScissor(left, bottom, bounds.width, bounds.height);
    renderer.setViewport(left, bottom, bounds.width, bounds.height);
    renderer.clear(true, true, true);
    renderer.render(this.scene, this.camera);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, canvasBounds.width, canvasBounds.height);
  }

  public dispose(): void {
    this.cancelPendingClick();
    this.element.removeEventListener("pointermove", this.handlePointerMove);
    this.element.removeEventListener("pointerleave", this.handlePointerLeave);
    this.element.removeEventListener("pointerdown", this.handlePointerDown);
    this.element.removeEventListener("pointerup", this.handlePointerUp);
    this.element.removeEventListener("pointercancel", this.handlePointerCancel);
    disposeObject(this.cube);
    this.cube.clear();
  }

  private addTarget(mesh: Mesh, target: ViewCubeTarget): void {
    const edges = new LineSegments(
      new EdgesGeometry(mesh.geometry),
      new LineBasicMaterial({
        color: 0x9adbc8,
        transparent: true,
        opacity: 0.58,
      }),
    );
    mesh.add(edges);
    this.targets.push(mesh);
    this.targetByMesh.set(mesh, target);
    this.cube.add(mesh);
  }

  private pick(event: PointerEvent): Mesh | null {
    const bounds = this.element.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) {
      return null;
    }
    this.pointer.set(
      ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
      -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const intersection = this.raycaster.intersectObjects(
      this.targets,
      false,
    )[0];
    return intersection?.object instanceof Mesh ? intersection.object : null;
  }

  private setHovered(mesh: Mesh | null): void {
    if (mesh === this.hovered) {
      return;
    }
    if (
      this.hovered !== null &&
      this.hovered.material instanceof MeshBasicMaterial
    ) {
      const previous = this.targetByMesh.get(this.hovered);
      this.hovered.material.color.setHex(
        previous?.kind === "corner" ? 0x60736b : 0x35423d,
      );
    }
    this.hovered = mesh;
    const target = mesh === null ? undefined : this.targetByMesh.get(mesh);
    if (mesh !== null && mesh.material instanceof MeshBasicMaterial) {
      mesh.material.color.setHex(0x5ea98c);
    }
    this.element.dataset.hover = target === undefined ? "false" : "true";
    this.element.title =
      target === undefined ? "" : `${target.label} (${target.axisLabel})`;
  }

  private readonly handlePointerMove = (event: PointerEvent): void => {
    event.preventDefault();
    event.stopPropagation();
    if (event.pointerId === this.pointerId) {
      const current = new Vector2(event.clientX, event.clientY);
      if (current.distanceTo(this.pointerOrigin) > 4) {
        this.dragged = true;
      }
      if (this.dragged) {
        this.cancelPendingClick();
        this.setHovered(null);
        this.element.dataset.dragging = "true";
        this.onDrag(
          current.x - this.previousPointer.x,
          current.y - this.previousPointer.y,
        );
      }
      this.previousPointer.copy(current);
      return;
    }
    this.setHovered(this.pick(event));
  };

  private readonly handlePointerLeave = (): void => {
    if (this.pointerId === null) {
      this.setHovered(null);
    }
  };

  private readonly handlePointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || this.pointerId !== null) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    this.pointerId = event.pointerId;
    this.dragged = false;
    this.pointerOrigin.set(event.clientX, event.clientY);
    this.previousPointer.copy(this.pointerOrigin);
    this.element.setPointerCapture(event.pointerId);
  };

  private readonly handlePointerUp = (event: PointerEvent): void => {
    if (event.pointerId !== this.pointerId) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (!this.dragged) {
      const target = this.pick(event);
      if (target !== null) {
        const definition = this.targetByMesh.get(target);
        if (definition !== undefined) {
          this.selectTarget(definition);
        }
      }
    }
    this.finishPointer(event);
    this.setHovered(this.pick(event));
  };

  private readonly handlePointerCancel = (event: PointerEvent): void => {
    if (event.pointerId === this.pointerId) {
      this.finishPointer(event);
      this.setHovered(null);
    }
  };

  private finishPointer(event: PointerEvent): void {
    if (this.element.hasPointerCapture(event.pointerId)) {
      this.element.releasePointerCapture(event.pointerId);
    }
    this.pointerId = null;
    this.dragged = false;
    delete this.element.dataset.dragging;
  }

  private selectTarget(target: ViewCubeTarget): void {
    if (this.pendingClick?.target.id === target.id) {
      window.clearTimeout(this.pendingClick.timer);
      this.pendingClick = null;
      this.onSelect(target, true);
      return;
    }
    if (this.pendingClick !== null) {
      window.clearTimeout(this.pendingClick.timer);
      this.onSelect(this.pendingClick.target, false);
    }
    const timer = window.setTimeout(() => {
      if (this.pendingClick?.timer !== timer) {
        return;
      }
      this.pendingClick = null;
      this.onSelect(target, false);
    }, 240);
    this.pendingClick = { target, timer };
  }

  private cancelPendingClick(): void {
    if (this.pendingClick === null) {
      return;
    }
    window.clearTimeout(this.pendingClick.timer);
    this.pendingClick = null;
  }
}
