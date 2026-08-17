import {
  ACESFilmicToneMapping,
  AmbientLight,
  Box3,
  BoxGeometry,
  Color,
  CylinderGeometry,
  DataTexture,
  DirectionalLight,
  DoubleSide,
  EdgesGeometry,
  GridHelper,
  Group,
  InstancedMesh,
  LinearFilter,
  LineBasicMaterial,
  LineSegments,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Quaternion,
  Raycaster,
  RGBAFormat,
  Scene,
  SphereGeometry,
  SRGBColorSpace,
  UnsignedByteType,
  Vector2,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { rgbBytes, viridis, type RGB } from "./color";
import type { SignalSlice } from "./grid";
import type { SceneCell, SceneConstraints, SceneFrame } from "./scene";

type SelectionCallback = (cell: SceneCell | null) => void;

function disposeGroup(group: Group): void {
  for (const child of [...group.children]) {
    group.remove(child);
    if (
      child instanceof Mesh ||
      child instanceof InstancedMesh ||
      child instanceof LineSegments
    ) {
      child.geometry.dispose();
      const materials = Array.isArray(child.material)
        ? child.material
        : [child.material];
      for (const material of materials) {
        material.dispose();
      }
    }
  }
}

function compose(
  position: Vector3,
  orientation: Quaternion,
  scale: Vector3,
  target: Matrix4,
): Matrix4 {
  return target.compose(position, orientation, scale);
}

export class ColonyViewer {
  private readonly renderer: WebGLRenderer;
  private readonly scene = new Scene();
  private readonly camera = new PerspectiveCamera(42, 1, 0.01, 10_000);
  private readonly controls: OrbitControls;
  private readonly colony = new Group();
  private readonly device = new Group();
  private readonly signal = new Group();
  private readonly highlight = new Group();
  private readonly grid = new GridHelper(20, 20, 0x34413c, 0x222b27);
  private readonly raycaster = new Raycaster();
  private readonly pointer = new Vector2();
  private readonly resizeObserver: ResizeObserver;
  private readonly onSelection: SelectionCallback;
  private cellMeshes: InstancedMesh[] = [];
  private cells: readonly SceneCell[] = [];
  private pointerOrigin: Vector2 | null = null;
  private signalTexture: DataTexture | null = null;
  private selectedCellId: string | null = null;
  private sceneBounds = new Box3(new Vector3(-1, -1, -1), new Vector3(1, 1, 1));

  public constructor(host: HTMLElement, onSelection: SelectionCallback) {
    this.onSelection = onSelection;
    this.scene.background = new Color(0x0b0f0e);
    this.renderer = new WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    host.append(this.renderer.domElement);

    this.camera.up.set(0, 0, 1);
    this.camera.position.set(11, -14, 11);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.target.set(0, 0, 0);

    const ambient = new AmbientLight(0xffffff, 1.4);
    const key = new DirectionalLight(0xffffff, 3.4);
    const fill = new DirectionalLight(0x8be0bd, 1.1);
    key.position.set(7, -9, 12);
    fill.position.set(-10, 5, 4);
    this.scene.add(
      ambient,
      key,
      fill,
      this.grid,
      this.signal,
      this.colony,
      this.device,
      this.highlight,
    );
    this.grid.rotateX(Math.PI / 2);
    this.grid.position.z = -0.002;
    this.highlight.visible = false;

    this.renderer.domElement.addEventListener(
      "pointerdown",
      this.handlePointerDown,
    );
    this.renderer.domElement.addEventListener(
      "pointerup",
      this.handlePointerUp,
    );
    this.resizeObserver = new ResizeObserver(() => this.resize(host));
    this.resizeObserver.observe(host);
    this.resize(host);
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  public setFrame(frame: SceneFrame, fit = true): void {
    disposeGroup(this.colony);
    this.cellMeshes = [];
    this.cells = frame.cells;
    this.highlight.visible = false;

    if (frame.cells.length === 0) {
      this.selectedCellId = null;
      this.onSelection(null);
      const deviceBounds = new Box3();
      this.buildDevice(frame.constraints, deviceBounds);
      this.sceneBounds = deviceBounds.isEmpty()
        ? new Box3(new Vector3(-1, -1, -1), new Vector3(1, 1, 1))
        : deviceBounds;
      this.configureReferenceGrid(this.sceneBounds);
      if (fit) {
        this.fitColony();
      }
      return;
    }

    const cylinderGeometry = new CylinderGeometry(1, 1, 1, 14, 1, false);
    const capGeometry = new SphereGeometry(1, 14, 9);
    const material = new MeshStandardMaterial({
      roughness: 0.62,
      metalness: 0.04,
    });
    const cylinders = new InstancedMesh(
      cylinderGeometry,
      material,
      frame.cells.length,
    );
    const firstCaps = new InstancedMesh(
      capGeometry,
      material,
      frame.cells.length,
    );
    const secondCaps = new InstancedMesh(
      capGeometry.clone(),
      material,
      frame.cells.length,
    );
    cylinders.name = "cell-cylinders";
    firstCaps.name = "cell-first-caps";
    secondCaps.name = "cell-second-caps";

    const matrix = new Matrix4();
    const orientation = new Quaternion();
    const center = new Vector3();
    const direction = new Vector3();
    const endpoint = new Vector3();
    const up = new Vector3(0, 1, 0);
    const bounds = new Box3();

    for (const [index, cell] of frame.cells.entries()) {
      center.fromArray(cell.position);
      direction.fromArray(cell.direction).normalize();
      orientation.setFromUnitVectors(up, direction);
      compose(
        center,
        orientation,
        new Vector3(cell.radius, Math.max(cell.length, 1e-7), cell.radius),
        matrix,
      );
      cylinders.setMatrixAt(index, matrix);

      endpoint.copy(direction).multiplyScalar(cell.length / 2);
      compose(
        new Vector3().copy(center).sub(endpoint),
        orientation.identity(),
        new Vector3(cell.radius, cell.radius, cell.radius),
        matrix,
      );
      firstCaps.setMatrixAt(index, matrix);
      compose(
        new Vector3().copy(center).add(endpoint),
        orientation,
        new Vector3(cell.radius, cell.radius, cell.radius),
        matrix,
      );
      secondCaps.setMatrixAt(index, matrix);

      const firstEndpoint = new Vector3().copy(center).sub(endpoint);
      const secondEndpoint = new Vector3().copy(center).add(endpoint);
      const cellBounds = new Box3()
        .setFromPoints([firstEndpoint, secondEndpoint])
        .expandByScalar(cell.radius);
      bounds.union(cellBounds);
    }

    cylinders.instanceMatrix.needsUpdate = true;
    firstCaps.instanceMatrix.needsUpdate = true;
    secondCaps.instanceMatrix.needsUpdate = true;
    cylinders.computeBoundingSphere();
    firstCaps.computeBoundingSphere();
    secondCaps.computeBoundingSphere();
    this.cellMeshes = [cylinders, firstCaps, secondCaps];
    this.colony.add(...this.cellMeshes);
    this.buildDevice(frame.constraints, bounds);
    this.sceneBounds = bounds;
    this.configureReferenceGrid(bounds);
    const selectedIndex = frame.cells.findIndex(
      (cell) => cell.id === this.selectedCellId,
    );
    if (selectedIndex >= 0) {
      const selected = frame.cells[selectedIndex];
      if (selected !== undefined) {
        this.buildHighlight(selected);
        this.onSelection(selected);
      }
    } else {
      this.selectedCellId = null;
      this.onSelection(null);
    }
    if (fit) {
      this.fitColony();
    }
  }

  public setCellColors(colors: readonly RGB[]): void {
    if (colors.length !== this.cells.length) {
      throw new RangeError(
        `expected ${this.cells.length} cell colors, received ${colors.length}`,
      );
    }
    const color = new Color();
    for (const mesh of this.cellMeshes) {
      for (const [index, value] of colors.entries()) {
        color.setRGB(value[0], value[1], value[2], SRGBColorSpace);
        mesh.setColorAt(index, color);
      }
      if (mesh.instanceColor !== null) {
        mesh.instanceColor.needsUpdate = true;
      }
    }
  }

  public setSignalSlice(slice: SignalSlice | null): void {
    this.signalTexture?.dispose();
    this.signalTexture = null;
    disposeGroup(this.signal);
    if (slice === null) {
      return;
    }
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    for (const value of slice.values) {
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    const span = maximum - minimum;
    const pixels = new Uint8Array(slice.width * slice.height * 4);
    for (const [index, value] of slice.values.entries()) {
      const color = rgbBytes(
        viridis(span === 0 ? 0.5 : (value - minimum) / span),
      );
      const offset = index * 4;
      pixels[offset] = color[0];
      pixels[offset + 1] = color[1];
      pixels[offset + 2] = color[2];
      pixels[offset + 3] = 205;
    }
    const texture = new DataTexture(
      pixels,
      slice.width,
      slice.height,
      RGBAFormat,
      UnsignedByteType,
    );
    texture.colorSpace = SRGBColorSpace;
    texture.magFilter = LinearFilter;
    texture.minFilter = LinearFilter;
    texture.needsUpdate = true;
    this.signalTexture = texture;

    const geometry = new PlaneGeometry(
      slice.horizontalSpan,
      slice.verticalSpan,
    );
    const material = new MeshBasicMaterial({
      map: texture,
      transparent: true,
      opacity: 0.74,
      side: DoubleSide,
      depthWrite: false,
    });
    const mesh = new Mesh(geometry, material);
    const horizontal = new Vector3().fromArray(slice.horizontal);
    const vertical = new Vector3().fromArray(slice.vertical);
    const normal = new Vector3().crossVectors(horizontal, vertical).normalize();
    const basis = new Matrix4().makeBasis(horizontal, vertical, normal);
    mesh.quaternion.setFromRotationMatrix(basis);
    mesh.position.fromArray(slice.center);
    mesh.renderOrder = 1;
    this.signal.add(mesh);
  }

  public selectCell(index: number | null): void {
    if (index === null) {
      this.selectedCellId = null;
      this.highlight.visible = false;
      this.onSelection(null);
      return;
    }
    const cell = this.cells[index];
    if (cell === undefined) {
      throw new RangeError(`cell slot ${index} is out of range`);
    }
    this.selectedCellId = cell.id;
    this.buildHighlight(cell);
    this.onSelection(cell);
  }

  public fitColony(): void {
    const center = this.sceneBounds.getCenter(new Vector3());
    const size = this.sceneBounds.getSize(new Vector3());
    const radius = Math.max(size.length() / 2, 1);
    const direction = new Vector3(1, -1.25, 0.9).normalize();
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(direction, radius * 2.8);
    this.camera.near = Math.max(radius / 1000, 0.001);
    this.camera.far = Math.max(radius * 100, 100);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  public dispose(): void {
    this.renderer.setAnimationLoop(null);
    disposeGroup(this.device);
    this.resizeObserver.disconnect();
    this.renderer.domElement.removeEventListener(
      "pointerdown",
      this.handlePointerDown,
    );
    this.renderer.domElement.removeEventListener(
      "pointerup",
      this.handlePointerUp,
    );
    disposeGroup(this.colony);
    this.signalTexture?.dispose();
    this.signalTexture = null;
    disposeGroup(this.signal);
    disposeGroup(this.highlight);
    this.grid.geometry.dispose();
    const materials = Array.isArray(this.grid.material)
      ? this.grid.material
      : [this.grid.material];
    for (const material of materials) {
      material.dispose();
    }
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private readonly handlePointerDown = (event: PointerEvent): void => {
    if (event.button === 0) {
      this.pointerOrigin = new Vector2(event.clientX, event.clientY);
    }
  };

  private readonly handlePointerUp = (event: PointerEvent): void => {
    if (event.button !== 0 || this.pointerOrigin === null) {
      return;
    }
    const distance = this.pointerOrigin.distanceTo(
      new Vector2(event.clientX, event.clientY),
    );
    this.pointerOrigin = null;
    if (distance > 4) {
      return;
    }
    const bounds = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(
      ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
      -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const intersection = this.raycaster.intersectObjects(
      this.cellMeshes,
      false,
    )[0];
    this.selectCell(intersection?.instanceId ?? null);
  };

  private buildDevice(constraints: SceneConstraints, bounds: Box3): void {
    disposeGroup(this.device);
    const wallColor = 0xd7e8e3;
    const chamberColor = 0xcfdcea;
    const edgeMaterial = new LineBasicMaterial({
      color: 0x9adbc8,
      transparent: true,
      opacity: 0.5,
    });

    const translucent = (color: number, opacity: number) =>
      new MeshStandardMaterial({
        color,
        transparent: true,
        opacity,
        roughness: 0.3,
        metalness: 0,
        depthWrite: false,
        side: DoubleSide,
      });

    for (const box of constraints.boxes) {
      const outside = box.allowedRegion === "outside";
      const geometry = new BoxGeometry(
        box.halfExtents[0] * 2,
        box.halfExtents[1] * 2,
        box.halfExtents[2] * 2,
      );
      const mesh = new Mesh(
        geometry,
        translucent(outside ? wallColor : chamberColor, outside ? 0.22 : 0.1),
      );
      mesh.position.fromArray(box.center);
      mesh.renderOrder = 2;
      const edges = new LineSegments(new EdgesGeometry(geometry), edgeMaterial);
      edges.position.fromArray(box.center);
      edges.renderOrder = 3;
      this.device.add(mesh, edges);
      const center = new Vector3().fromArray(box.center);
      const extent = new Vector3().fromArray(box.halfExtents);
      bounds.union(
        new Box3(
          new Vector3().copy(center).sub(extent),
          new Vector3().copy(center).add(extent),
        ),
      );
    }

    for (const sphere of constraints.spheres) {
      const outside = sphere.allowedRegion === "outside";
      const geometry = new SphereGeometry(sphere.radius, 24, 16);
      const mesh = new Mesh(
        geometry,
        translucent(outside ? wallColor : chamberColor, outside ? 0.22 : 0.1),
      );
      mesh.position.fromArray(sphere.center);
      mesh.renderOrder = 2;
      this.device.add(mesh);
      const center = new Vector3().fromArray(sphere.center);
      bounds.union(
        new Box3(
          new Vector3().copy(center).subScalar(sphere.radius),
          new Vector3().copy(center).addScalar(sphere.radius),
        ),
      );
    }

    const zToAxis = new Quaternion().setFromUnitVectors(
      new Vector3(0, 1, 0),
      new Vector3(0, 0, 1),
    );
    for (const cylinder of constraints.cylinders) {
      const outside = cylinder.allowedRegion === "outside";
      const geometry = new CylinderGeometry(
        cylinder.radius,
        cylinder.radius,
        cylinder.halfHeight * 2,
        48,
        1,
        !outside,
      );
      const mesh = new Mesh(
        geometry,
        translucent(outside ? wallColor : chamberColor, outside ? 0.22 : 0.12),
      );
      mesh.position.fromArray(cylinder.center);
      mesh.quaternion.copy(zToAxis);
      mesh.renderOrder = 2;
      const edges = new LineSegments(
        new EdgesGeometry(geometry, 30),
        edgeMaterial,
      );
      edges.position.fromArray(cylinder.center);
      edges.quaternion.copy(zToAxis);
      edges.renderOrder = 3;
      this.device.add(mesh, edges);
      const center = new Vector3().fromArray(cylinder.center);
      const extent = new Vector3(
        cylinder.radius,
        cylinder.radius,
        cylinder.halfHeight,
      );
      bounds.union(
        new Box3(
          new Vector3().copy(center).sub(extent),
          new Vector3().copy(center).add(extent),
        ),
      );
    }

    if (constraints.planes.length > 0) {
      const focus = bounds.isEmpty()
        ? new Vector3()
        : bounds.getCenter(new Vector3());
      const extent = bounds.isEmpty()
        ? 20
        : Math.max(bounds.getSize(new Vector3()).length() * 1.2, 10);
      for (const plane of constraints.planes) {
        const point = new Vector3().fromArray(plane.point);
        const normal = new Vector3().fromArray(plane.inwardNormal).normalize();
        const mesh = new Mesh(
          new PlaneGeometry(extent, extent),
          translucent(wallColor, 0.07),
        );
        mesh.quaternion.setFromUnitVectors(new Vector3(0, 0, 1), normal);
        const offset = new Vector3().copy(focus).sub(point).dot(normal);
        mesh.position.copy(focus).addScaledVector(normal, -offset);
        mesh.renderOrder = 2;
        this.device.add(mesh);
      }
    }
  }

  private buildHighlight(cell: SceneCell): void {
    disposeGroup(this.highlight);
    const material = new MeshBasicMaterial({
      color: 0xffffff,
      wireframe: true,
      transparent: true,
      opacity: 0.92,
      depthTest: false,
    });
    const cylinder = new Mesh(
      new CylinderGeometry(1, 1, 1, 18, 1, false),
      material,
    );
    const firstCap = new Mesh(new SphereGeometry(1, 18, 10), material);
    const secondCap = new Mesh(new SphereGeometry(1, 18, 10), material);
    const center = new Vector3().fromArray(cell.position);
    const direction = new Vector3().fromArray(cell.direction).normalize();
    const orientation = new Quaternion().setFromUnitVectors(
      new Vector3(0, 1, 0),
      direction,
    );
    const endpoint = new Vector3()
      .copy(direction)
      .multiplyScalar(cell.length / 2);
    const radius = cell.radius * 1.08;
    cylinder.position.copy(center);
    cylinder.quaternion.copy(orientation);
    cylinder.scale.set(radius, Math.max(cell.length, 1e-7), radius);
    firstCap.position.copy(center).sub(endpoint);
    firstCap.scale.setScalar(radius);
    secondCap.position.copy(center).add(endpoint);
    secondCap.scale.setScalar(radius);
    this.highlight.add(cylinder, firstCap, secondCap);
    this.highlight.visible = true;
  }

  private configureReferenceGrid(bounds: Box3): void {
    const size = bounds.getSize(new Vector3());
    const center = bounds.getCenter(new Vector3());
    const extent = Math.max(size.x, size.y, 10);
    this.grid.scale.set(extent / 20, extent / 20, extent / 20);
    this.grid.position.set(
      center.x,
      center.y,
      Math.min(bounds.min.z, 0) - 0.01,
    );
  }

  private resize(host: HTMLElement): void {
    const width = Math.max(host.clientWidth, 1);
    const height = Math.max(host.clientHeight, 1);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }
}
