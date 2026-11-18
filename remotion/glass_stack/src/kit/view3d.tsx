// View3D: a transparent three.js canvas whose camera is set from a CamState on every frame. A pure function of its props
// (no useFrame, no requestAnimationFrame, no random numbers), so every frame renders the same way every time.
import React, { createContext, useContext, useLayoutEffect, useMemo } from "react";
import { ThreeCanvas } from "@remotion/three";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { CamState, Projector, V3, applyCam, projector } from "./cam";

type Ctx = { st: CamState; W: number; H: number; pr: Projector };
const View3DCtx = createContext<Ctx | null>(null);
export const useView3D = () => {
  const c = useContext(View3DCtx);
  if (!c) throw new Error("3-D component used outside View3D");
  return c;
};

const CamRig: React.FC<{ st: CamState; W: number; H: number }> = ({ st, W, H }) => {
  const set = useThree((s) => s.set);
  const persp = useMemo(() => { const c = new THREE.PerspectiveCamera(); (c as unknown as { manual: boolean }).manual = true; return c; }, []);
  const orth = useMemo(() => { const c = new THREE.OrthographicCamera(); (c as unknown as { manual: boolean }).manual = true; return c; }, []);
  useLayoutEffect(() => {
    const c = st.ortho ? orth : persp;
    applyCam(c, st, W, H);
    set({ camera: c });
  });
  return null;
};

/** Advance the Three.js canvas once after every commit — needed because @remotion/three only advances when the frame
 * number changes, so any mesh that loads data asynchronously would stay blank in stills without this kick. Data
 * components that load asynchronously should also add their own <Kick deps={[data]} /> with the loaded data as deps. */
export const Kick: React.FC<{ deps?: unknown[] }> = ({ deps }) => {
  const advance = useThree((s) => s.advance);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useLayoutEffect(() => { advance(performance.now()); }, deps);
  return null;
};

/** A transparent ThreeCanvas at (x, y) of size w × h, camera from `st`, two lights (ambient + one key light). */
export const View3D: React.FC<{ st: CamState; x?: number; y?: number; w?: number; h?: number; o?: number; children: React.ReactNode;
  key3?: V3; amb?: number; dir?: number; style?: React.CSSProperties }> = ({ st, x = 0, y = 0, w = 1920, h = 1080, o = 1, children,
  key3 = [-4, -6, 9], amb = 1.15, dir = 2.0, style }) => {
  const pr = useMemo(() => projector(st, w, h, x, y), [st, w, h, x, y]);
  if (o <= 0) return null;
  return (
    <ThreeCanvas width={w} height={h} style={{ position: "absolute", left: x, top: y, opacity: o, ...style }}
      gl={{ antialias: true, alpha: true, preserveDrawingBuffer: true }} camera={{ position: [0, 0, 10], fov: st.fov }}>
      <CamRig st={st} W={w} H={h} />
      <Kick />
      <ambientLight intensity={amb} />
      <directionalLight position={key3} intensity={dir} />
      <directionalLight position={[6, 4, -3]} intensity={0.45} />
      <View3DCtx.Provider value={{ st, W: w, H: h, pr }}>{children}</View3DCtx.Provider>
    </ThreeCanvas>
  );
};
