import {
  BufferAttribute,
  BufferGeometry,
  DataTexture,
  DoubleSide,
  Group,
  LinearFilter,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  PlaneGeometry,
  Points,
  PointsMaterial,
  ShaderMaterial,
  SRGBColorSpace,
  TextureLoader,
  Vector2,
  Vector3,
} from "three";
import {
  streamlines,
  streamlineSpacing,
  vectorArrows,
  type FlowSlice,
  type VectorDisplay,
} from "./flow-field";
import referenceTipUrl from "./flow-reference-tip.svg?url";
import arrowUrl from "./flow-arrow.svg?url";

// The line is stationary in the instantaneous field. Its highlight travels
// forward using the simulated flight time along this particular streamline.
const strokeMaterial = () =>
  new ShaderMaterial({
    uniforms: {
      viewport: { value: new Vector2(1, 1) },
      simTime: { value: 0 },
      comparison: { value: 0 },
    },
    vertexShader: `
    attribute vec4 stroke;
    attribute vec3 ribbon;
    uniform vec2 viewport;
    uniform float comparison;
    varying vec4 trace;
    varying float edge;
    void main() {
      trace = stroke;
      edge = ribbon.z;
      vec4 clip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      vec4 tangent = projectionMatrix * modelViewMatrix * vec4(ribbon.xy, 0.0, 0.0);
      vec2 direction = (tangent.xy * clip.w - clip.xy * tangent.w) * viewport;
      vec2 normal = vec2(-direction.y, direction.x) / max(length(direction), 1e-8);
      float halfWidth = mix(1.65, 1.32, comparison);
      clip.xy += edge * halfWidth * 2.0 * normal / viewport * clip.w;
      gl_Position = clip;
    }`,
    fragmentShader: `
    uniform float simTime;
    uniform float comparison;
    varying vec4 trace;
    varying float edge;
    void main() {
      float phase = fract(trace.y - simTime + trace.z);
      float glint = smoothstep(0.4, 0.86, phase) * (1.0 - smoothstep(0.86, 1.0, phase));
      float resolved = 1.0 - smoothstep(0.12, 0.5, fwidth(trace.y));
      float light = mix(0.3, glint, resolved * smoothstep(0.1, 0.45, trace.w));
      float across = abs(edge);
      float core = 1.0 - smoothstep(0.32, 0.7, across);
      vec3 dye = mix(vec3(0.12, 0.79, 0.86), vec3(0.86, 1.0, 0.96), light);
      vec3 ink = mix(vec3(0.035, 0.17, 0.24), dye, core);
      float opacity = (1.0 - smoothstep(0.76, 1.0, across)) * smoothstep(0.0, 0.8, trace.x);
      gl_FragColor = vec4(ink, opacity * (0.73 + 0.26 * trace.w) * mix(1.0, 0.53, comparison));
    }`,
    side: DoubleSide,
    forceSinglePass: true,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });

function strokeGeometry(field: FlowSlice, period: number): BufferGeometry {
  const positions: number[] = [],
    attributes: number[] = [],
    ribbons: number[] = [],
    indices: number[] = [];
  for (const { points, phase } of streamlines(field)) {
    const last = points.length - 3;
    let distance = 0;
    const lengths = [0];
    for (let j = 3; j <= last; j += 3) {
      distance += Math.hypot(
        (points[j]! - points[j - 3]!) * field.dx,
        (points[j + 1]! - points[j - 2]!) * field.dy,
      );
      lengths.push(distance);
    }
    for (let j = 0; j <= last; j += 3) {
      const index = positions.length / 3;
      const x = (points[j]! + 0.5) / field.width - 0.5;
      const y = (points[j + 1]! + 0.5) / field.height - 0.5;
      const before = Math.max(0, j - 3),
        after = Math.min(last, j + 3);
      const tx = (points[after]! - points[before]!) / field.width;
      const ty = (points[after + 1]! - points[before + 1]!) / field.height;
      positions.push(x, y, 0, x, y, 0);
      ribbons.push(tx, ty, -1, tx, ty, 1);
      const col = Math.round(points[j]!),
        row = Math.round(points[j + 1]!);
      const i = 2 * (row * field.width + col);
      const strength = Math.sqrt(
        Math.hypot(field.velocity[i]!, field.velocity[i + 1]!) /
          field.peakSpeed,
      );
      const along = lengths[j / 3]!;
      const taper =
        Math.min(along, distance - along) / streamlineSpacing(field);
      const travel = points[j + 2]! / period;
      attributes.push(
        taper,
        travel,
        phase,
        strength,
        taper,
        travel,
        phase,
        strength,
      );
      if (j > 0)
        indices.push(index - 2, index - 1, index, index, index - 1, index + 1);
    }
  }
  const geo = new BufferGeometry();
  geo.setAttribute(
    "position",
    new BufferAttribute(new Float32Array(positions), 3),
  );
  geo.setAttribute(
    "stroke",
    new BufferAttribute(new Float32Array(attributes), 4),
  );
  geo.setAttribute("ribbon", new BufferAttribute(new Float32Array(ribbons), 3));
  geo.setIndex(indices);
  return geo;
}

// Rotate the glyph in physical coordinates before mapping to the unit slice.
function vectorGeometry(field: FlowSlice, display: VectorDisplay) {
  const tips: number[] = [],
    tipColors: number[] = [];
  const positions: number[] = [],
    uvs: number[] = [],
    indices: number[] = [];
  const width = field.width * field.dx,
    height = field.height * field.dy;
  for (const { x, y, u, v, u0, v0, tipOpacity } of vectorArrows(
    field,
    display,
  )) {
    if (tipOpacity > 0) {
      tips.push((x + u0) / width, (y + v0) / height, 0);
      tipColors.push(1, 1, 1, tipOpacity);
    }
    if (u === 0 && v === 0) continue;
    const base = positions.length / 3;
    for (const [along, across] of [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ] as const) {
      const b = (across - 0.5) * (24 / 64);
      positions.push(
        (x + along * u - b * v) / width,
        (y + along * v + b * u) / height,
        0,
      );
      uvs.push(along, across);
    }
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  const geometry = new BufferGeometry();
  geometry.setAttribute(
    "position",
    new BufferAttribute(new Float32Array(positions), 3),
  );
  geometry.setAttribute("uv", new BufferAttribute(new Float32Array(uvs), 2));
  geometry.setIndex(indices);
  const markers = new BufferGeometry();
  markers.setAttribute(
    "position",
    new BufferAttribute(new Float32Array(tips), 3),
  );
  markers.setAttribute(
    "color",
    new BufferAttribute(new Float32Array(tipColors), 4),
  );
  return { arrows: geometry, markers };
}

// Stops in display sRGB, including opacity so zero change is fully clear.
const changeRamp = [
  [112, 40, 115, 0],
  [206, 63, 111, 152],
  [255, 116, 74, 202],
  [255, 171, 92, 209],
  [255, 219, 152, 220],
] as const;

export class FlowLayer extends Group {
  private readonly wash = new Mesh(
    new PlaneGeometry(1, 1),
    new MeshBasicMaterial({
      transparent: true,
      depthTest: false,
      depthWrite: false,
      side: DoubleSide,
      toneMapped: false,
      forceSinglePass: true,
    }),
  );
  private readonly arrowTexture = new TextureLoader().load(arrowUrl);
  private readonly vectors = new Mesh(
    new BufferGeometry(),
    new MeshBasicMaterial({
      map: this.arrowTexture,
      transparent: true,
      side: DoubleSide,
      forceSinglePass: true,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    }),
  );
  private readonly tipTexture = new TextureLoader().load(referenceTipUrl);
  private readonly referenceTips = new Points(
    new BufferGeometry(),
    new PointsMaterial({
      map: this.tipTexture,
      vertexColors: true,
      size: 5,
      sizeAttenuation: false,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    }),
  );
  private previousVectorDisplay: VectorDisplay | null = null;
  private readonly strokes = new Mesh(new BufferGeometry(), strokeMaterial());
  private texture: DataTexture | null = null;
  private pulsePeriod = 0;
  private previous: FlowSlice | null = null;
  private readonly reducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  );

  constructor() {
    super();
    this.visible = false;
    this.renderOrder = 3;
    this.wash.renderOrder = 3;
    this.strokes.renderOrder = 4;
    this.referenceTips.renderOrder = 4;
    this.vectors.renderOrder = 5;
    this.tipTexture.colorSpace = SRGBColorSpace;
    this.arrowTexture.colorSpace = SRGBColorSpace;
    this.add(this.wash, this.strokes, this.referenceTips, this.vectors);
  }

  public setViewport(width: number, height: number): void {
    this.strokes.material.uniforms.viewport!.value.set(width, height);
  }

  public update(
    field: FlowSlice | null,
    time: number,
    vectors: VectorDisplay | null,
  ): void {
    this.visible = field !== null;
    if (!field) {
      this.previous = null;
      this.previousVectorDisplay = null;
      return;
    }
    this.strokes.material.uniforms.comparison!.value = field.change ? 1 : 0;
    const old = this.previous;
    this.previous = field;
    const samePlane =
      old &&
      old.width === field.width &&
      old.height === field.height &&
      old.dx === field.dx &&
      old.dy === field.dy &&
      old.horizontal === field.horizontal &&
      old.vertical === field.vertical &&
      old.center.every((v, i) => v === field.center[i]);
    if (!samePlane || !this.pulsePeriod) {
      this.pulsePeriod = field.peakSpeed
        ? (9 * streamlineSpacing(field)) / field.peakSpeed
        : 0;
    }
    this.strokes.material.uniforms.simTime!.value =
      this.reducedMotion.matches || !this.pulsePeriod
        ? 0
        : time / this.pulsePeriod;
    if (!samePlane) {
      const h = new Vector3().setComponent(field.horizontal, 1),
        v = new Vector3().setComponent(field.vertical, 1);
      this.quaternion.setFromRotationMatrix(
        new Matrix4().makeBasis(h, v, new Vector3().crossVectors(h, v)),
      );
      this.position.fromArray(field.center);
      this.scale.set(field.width * field.dx, field.height * field.dy, 1);
    }
    const lastVectors = this.previousVectorDisplay;
    this.previousVectorDisplay = vectors;
    this.vectors.visible = this.referenceTips.visible = vectors !== null;
    this.strokes.visible = vectors === null;
    const fieldChanged =
      !samePlane || !field.velocity.every((v, i) => v === old.velocity[i]);
    if (
      vectors &&
      (fieldChanged ||
        vectors.reference !== lastVectors?.reference ||
        vectors.gain !== lastVectors.gain ||
        vectors.scaleSpeed !== lastVectors.scaleSpeed)
    ) {
      const geometry = vectorGeometry(field, vectors);
      this.vectors.geometry.dispose();
      this.referenceTips.geometry.dispose();
      this.vectors.geometry = geometry.arrows;
      this.referenceTips.geometry = geometry.markers;
    } else if (!vectors && (fieldChanged || lastVectors)) {
      this.strokes.geometry.dispose();
      this.strokes.geometry = strokeGeometry(field, this.pulsePeriod);
    }
    const change = field.change;
    this.wash.visible = !!change && field.peakChange > 0;
    if (!change || !this.wash.visible) return;
    if (
      samePlane &&
      old.change &&
      old.changeScale === field.changeScale &&
      change.every((v, i) => v === old.change![i])
    )
      return;
    if (
      !this.texture ||
      this.texture.image.width !== field.width ||
      this.texture.image.height !== field.height
    ) {
      this.texture?.dispose();
      this.texture = new DataTexture(
        new Uint8Array(change.length * 4),
        field.width,
        field.height,
      );
      this.texture.magFilter = this.texture.minFilter = LinearFilter;
      this.texture.colorSpace = SRGBColorSpace;
      this.wash.material.map = this.texture;
      this.wash.material.needsUpdate = true;
    }
    const pixels = this.texture.image.data!;
    for (let i = 0; i < change.length; i++) {
      const t =
        Math.min(1, change[i]! / field.changeScale) * (changeRamp.length - 1);
      const low = Math.min(Math.floor(t), changeRamp.length - 2);
      for (let a = 0; a < 4; a++) {
        const start = changeRamp[low]![a]!,
          end = changeRamp[low + 1]![a]!;
        pixels[4 * i + a] = Math.round(start + (end - start) * (t - low));
      }
    }
    this.texture.needsUpdate = true;
  }

  public dispose(): void {
    this.texture?.dispose();
    this.strokes.geometry.dispose();
    this.strokes.material.dispose();
    this.vectors.geometry.dispose();
    this.vectors.material.dispose();
    this.arrowTexture.dispose();
    this.referenceTips.geometry.dispose();
    this.referenceTips.material.dispose();
    this.tipTexture.dispose();
    this.wash.geometry.dispose();
    this.wash.material.dispose();
  }
}
