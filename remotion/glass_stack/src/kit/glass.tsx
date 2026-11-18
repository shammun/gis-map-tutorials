// GlassStack3D: an exploded stack of thin glass panes in front of a relief globe. Each pane carries one field as the
// ORTHOGRAPHIC hemisphere seen along the pane normal d = ll(lat0, lon0), so the panes line up with the globe from any
// camera angle. Fields cross-fade between forecast lead times in a two-texture shader (one clock for all panes).
// A pure function of its props; textures load with delayRender.
import React, { useEffect, useMemo, useState } from "react";
import { cancelRender, continueRender, delayRender } from "remotion";
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { CamState, D2R, V3, add3, mul3 } from "./cam";
import { enAt, ll } from "./globe";

// ------------------------------------------------------------------------------------------------ loaders
const cache = new Map<string, THREE.Texture>();
const wait = new Map<string, Promise<THREE.Texture>>();
const load = (url: string, srgb: boolean) => {
  let p = wait.get(url);
  if (!p) {
    p = new Promise<THREE.Texture>((res, rej) => new THREE.TextureLoader().load(url, (t) => {
      // pane textures pass through the shader untouched (raw PNG bytes = what the builder wrote); the globe map is sRGB
      t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
      t.anisotropy = 8; t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter;
      cache.set(url, t); res(t);
    }, undefined, rej));
    wait.set(url, p);
  }
  return p;
};
/** all textures of a list, or null until every one is loaded (one delayRender for the set) */
export const useTexSet = (urls: string[], srgb = false): THREE.Texture[] | null => {
  const key = urls.join("|");
  const ready = () => (urls.every((u) => cache.has(u)) ? urls.map((u) => cache.get(u)!) : null);
  const [set, setSet] = useState<THREE.Texture[] | null>(ready);
  const [h] = useState(() => (ready() ? null : delayRender(`glass textures ${urls.length}`, { timeoutInMilliseconds: 180000 })));
  useEffect(() => {
    if (set) return;
    Promise.all(urls.map((u) => load(u, srgb))).then((ts) => setSet(ts)).catch((e) => cancelRender(e));
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (set && h !== null) continueRender(h); }, [set, h]);
  return set;
};

// ------------------------------------------------------------------------------------------------ geometry helpers
/** the stack frame: d = pane normal (towards the viewer side), e = east, n = north at (lat0, lon0) */
export const stackBasis = (lat0: number, lon0: number) => { const { e, n } = enAt(lat0, lon0); return { d: ll(lat0, lon0), e, n }; };
/** a world point from stack coordinates (along e, along n, along d) */
export const sp = (B: ReturnType<typeof stackBasis>, x: number, y: number, z: number): V3 => add3(add3(mul3(B.e, x), B.n, y), B.d, z);

/** Orbit camera in the stack frame: azimuth from the normal d towards east e, elevation towards north n. */
export const glassCam = (lat0: number, lon0: number, o: { az: number; el: number; dist: number; look?: [number, number, number]; fov?: number; pp?: [number, number] }): CamState => {
  const B = stackBasis(lat0, lon0), a = o.az * D2R, el = o.el * D2R;
  const L = o.look ?? [0, 0, 0];
  const target = sp(B, L[0], L[1], L[2]);
  const dir = add3(add3(mul3(B.d, Math.cos(el) * Math.cos(a)), B.e, Math.cos(el) * Math.sin(a)), B.n, Math.sin(el));
  return { pos: add3(target, dir, o.dist), target, up: B.n, fov: o.fov ?? 30, pp: o.pp ?? [960, 450] };
};

/** unit sphere with explicit equirectangular UVs: u = (lon + 180)/360, v = (lat + 90)/180, positions from ll() */
const sphereGeo = (nLat = 96, nLon = 192) => {
  const pos: number[] = [], uv: number[] = [], nor: number[] = [], idx: number[] = [];
  for (let i = 0; i <= nLat; i++) {
    const lat = -90 + (180 * i) / nLat;
    for (let j = 0; j <= nLon; j++) {
      const lon = -180 + (360 * j) / nLon, p = ll(lat, lon);
      pos.push(...p); nor.push(...p); uv.push(j / nLon, i / nLat);
    }
  }
  for (let i = 0; i < nLat; i++) for (let j = 0; j < nLon; j++) {
    const a = i * (nLon + 1) + j;
    idx.push(a, a + 1, a + nLon + 1, a + 1, a + nLon + 2, a + nLon + 1);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  return g;
};

/** flat rounded-rectangle ring (the pane's edge highlight) of outer half-size h, corner r, width w */
const ringGeo = (h: number, r: number, w: number) => {
  const rr = (s: THREE.Shape | THREE.Path, H: number, R: number) => {
    s.moveTo(-H + R, -H); s.lineTo(H - R, -H); s.quadraticCurveTo(H, -H, H, -H + R); s.lineTo(H, H - R);
    s.quadraticCurveTo(H, H, H - R, H); s.lineTo(-H + R, H); s.quadraticCurveTo(-H, H, -H, H - R); s.lineTo(-H, -H + R);
    s.quadraticCurveTo(-H, -H, -H + R, -H);
  };
  const s = new THREE.Shape(); rr(s, h, r);
  const hole = new THREE.Path(); rr(hole, h - w, Math.max(0.001, r - w)); s.holes.push(hole);
  return new THREE.ShapeGeometry(s, 8);
};

const VERT = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const FRAG = `uniform sampler2D a; uniform sampler2D b; uniform float w; uniform float op; varying vec2 vUv;
void main(){ vec4 ca = texture2D(a, vUv); vec4 cb = texture2D(b, vUv); vec4 c = mix(ca, cb, w); gl_FragColor = vec4(c.rgb, c.a * op); }`;

export type GlassPane = {
  /** one texture per lead (orthographic hemisphere, transparent outside the disc) */
  frames: THREE.Texture[];
  /** pane opacity 0..1 (fade-in / dim) */
  o?: number;
  /** lift along north (stack units; the globe radius is 1) and towards the viewer along d */
  lift?: number;
  /** glass tint */
  tint?: string;
};

/**
 * GlassStack3D (inside a View3D):
 *   lat0, lon0  the pane normal d (the hemisphere every pane shows)
 *   panes       nearest-to-globe first; pane k sits at d-offset z0 + k·gap
 *   t           fractional lead index shared by all panes (0 = frames[0]; 2.4 = 60 % of the way from frames[2] to frames[3])
 *   relief      the globe's equirectangular texture (row 0 = 90N, column 0 = 180W)
 *   half        half-size of a pane (globe radius 1; the data disc has radius 1)
 */
export const GlassStack3D: React.FC<{ lat0: number; lon0: number; panes: GlassPane[]; t: number; relief: THREE.Texture | null; z0?: number; gap: number;
  half?: number; globeO?: number }> = ({ lat0, lon0, panes, t, relief, z0 = 1.25, gap, half = 1.16, globeO = 1 }) => {
  const B = useMemo(() => stackBasis(lat0, lon0), [lat0, lon0]);
  const basis = useMemo(() => new THREE.Matrix4().makeBasis(new THREE.Vector3(...B.e), new THREE.Vector3(...B.n), new THREE.Vector3(...B.d)), [B]);
  const sph = useMemo(() => sphereGeo(), []);
  const globeMat = useMemo(() => new THREE.MeshStandardMaterial({ roughness: 0.92, metalness: 0 }), []);
  if (globeMat.map !== relief) { globeMat.map = relief; globeMat.needsUpdate = true; }
  globeMat.transparent = globeO < 1; globeMat.opacity = globeO;
  const slab = useMemo(() => new RoundedBoxGeometry(2 * half, 2 * half, 0.03, 4, 0.07), [half]);
  const ring = useMemo(() => ringGeo(half, 0.07, 0.012), [half]);
  const disc = useMemo(() => new THREE.PlaneGeometry(2, 2), []);
  const glassMats = useMemo(() => panes.map(() => new THREE.MeshPhysicalMaterial({ color: "#c9d8ea", roughness: 0.12, metalness: 0, clearcoat: 1,
    transparent: true, depthWrite: false, side: THREE.DoubleSide })), [panes.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const ringMats = useMemo(() => panes.map(() => new THREE.MeshBasicMaterial({ color: "#eef4ff", transparent: true, depthWrite: false, side: THREE.DoubleSide })), [panes.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const dataMats = useMemo(() => panes.map(() => new THREE.ShaderMaterial({ uniforms: { a: { value: null }, b: { value: null }, w: { value: 0 }, op: { value: 1 } },
    vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false, side: THREE.DoubleSide })), [panes.length]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <group>
      {relief ? <mesh geometry={sph} material={globeMat} renderOrder={0} /> : null}
      <group matrixAutoUpdate={false} matrix={basis}>
        {panes.map((P, k) => {
          const o = P.o ?? 1;
          if (o <= 0.003) return null;
          const n = P.frames.length, tc = Math.max(0, Math.min(n - 1, t)), i0 = Math.min(n - 1, Math.floor(tc)), i1 = Math.min(n - 1, i0 + 1);
          const dm = dataMats[k];
          dm.uniforms.a.value = P.frames[i0]; dm.uniforms.b.value = P.frames[i1]; dm.uniforms.w.value = tc - i0; dm.uniforms.op.value = o;
          const gm = glassMats[k]; gm.opacity = 0.13 * o; if (P.tint) gm.color.set(P.tint);
          const rm = ringMats[k]; rm.opacity = 0.55 * o;
          const lift = P.lift ?? 0;
          const z = z0 + k * gap + 0.25 * lift;
          return (
            <group key={k} position={[0, lift, z]}>
              <mesh geometry={slab} material={gm} renderOrder={10 + 3 * k} />
              <mesh geometry={disc} material={dm} position={[0, 0, 0.017]} renderOrder={11 + 3 * k} />
              <mesh geometry={ring} material={rm} position={[0, 0, 0.018]} renderOrder={12 + 3 * k} />
            </group>
          );
        })}
      </group>
    </group>
  );
};

/** screen anchors of pane k (stack coordinates → world), for labels and leader lines drawn in SVG */
export const paneCorner = (lat0: number, lon0: number, k: number, gap: number, lift: number, sx: number, sy: number, z0 = 1.25, half = 1.16): V3 =>
  sp(stackBasis(lat0, lon0), sx * half, sy * half + lift, z0 + k * gap + 0.25 * lift + 0.018);
