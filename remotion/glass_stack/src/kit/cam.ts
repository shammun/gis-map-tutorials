// Camera maths. Pure functions of their inputs. One camera description drives BOTH the WebGL render (View3D) and the SVG
// overlay (projector), so labels and leader lines drawn in SVG sit exactly on the 3-D objects.
import * as THREE from "three";

export type V3 = [number, number, number];
export type Pt = [number, number];

/** A resolved camera: position, look-at target, up vector, vertical fov (perspective) or visible world height
 * (orthographic), and the principal point pp (screen px inside the canvas) where the target is drawn. */
export type CamState = { pos: V3; target: V3; up: V3; fov: number; ortho?: number; pp: Pt };

export const D2R = Math.PI / 180;
export const add3 = (a: V3, b: V3, s = 1): V3 => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];
export const sub3 = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const mul3 = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot3 = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm3 = (a: V3): V3 => { const L = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / L, a[1] / L, a[2] / L]; };
export const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
export const easeIO = (t: number) => { const x = clamp01(t); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };

/** Apply a CamState to a three.js camera for a W × H canvas, shifting the principal point to st.pp. */
export const applyCam = (c: THREE.PerspectiveCamera | THREE.OrthographicCamera, st: CamState, W: number, H: number) => {
  if ((c as THREE.PerspectiveCamera).isPerspectiveCamera) {
    const p = c as THREE.PerspectiveCamera;
    p.fov = st.fov; p.aspect = W / H; p.near = 0.01; p.far = 200;
  } else {
    const o = c as THREE.OrthographicCamera;
    const hh = (st.ortho ?? 10) / 2, hw = (hh * W) / H;
    o.left = -hw; o.right = hw; o.top = hh; o.bottom = -hh; o.near = -100; o.far = 200;
  }
  c.position.set(...st.pos);
  c.up.set(...st.up);
  c.lookAt(...st.target);
  c.updateMatrixWorld(true);
  c.updateProjectionMatrix();
  const dx = (st.pp[0] - W / 2) / (W / 2), dy = -(st.pp[1] - H / 2) / (H / 2);
  const e = c.projectionMatrix.elements;
  if ((c as THREE.PerspectiveCamera).isPerspectiveCamera) { e[8] = -dx; e[9] = -dy; } else { e[12] += dx; e[13] += dy; }
  c.projectionMatrixInverse.copy(c.projectionMatrix).invert();
};

const _pc = new THREE.PerspectiveCamera(), _oc = new THREE.OrthographicCamera();
const _v = new THREE.Vector3();
/** Screen projector for overlays: world point → [x, y] in page px (canvas offset ox, oy), plus visibility. */
export type Projector = { p: (v: V3) => Pt; vis: (v: V3) => boolean; depth: (v: V3) => number; wpp: (v: V3) => number; st: CamState };
export const projector = (st: CamState, W: number, H: number, ox = 0, oy = 0): Projector => {
  const c = st.ortho ? _oc : _pc;
  applyCam(c, st, W, H);
  const m = c.projectionMatrix.clone().multiply(c.matrixWorldInverse);
  const fwd = norm3(sub3(st.target, st.pos));
  const p = (v: V3): Pt => {
    _v.set(v[0], v[1], v[2]).applyMatrix4(m);
    return [ox + ((_v.x + 1) / 2) * W, oy + ((1 - _v.y) / 2) * H];
  };
  const depth = (v: V3) => dot3(sub3(v, st.pos), fwd);
  const wpp = (v: V3) => (st.ortho ? st.ortho / H : (2 * Math.max(0.05, depth(v)) * Math.tan((st.fov * D2R) / 2)) / H);
  return { p, vis: (v) => st.ortho !== undefined || depth(v) > 0.02, depth, wpp, st };
};
