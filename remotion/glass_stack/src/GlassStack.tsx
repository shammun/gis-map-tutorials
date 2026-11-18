// GlassStack: 26 seconds of the glass-pane stack on NOAA GFS data read from dynamical.org. Five panes (temperature,
// isobars, precipitable water, cloud, precipitation) fan out in front of a relief globe, the isobar pane lifts out and
// the forecast steps from +6 h to +48 h. No narration. The pictures and the manifest in the public folder are written
// by `python -m gismaps.glass.remotion <public folder>` (see README.md).
import React, { useEffect, useState } from "react";
import { AbsoluteFill, cancelRender, continueRender, delayRender, staticFile, useCurrentFrame } from "remotion";
import { View3D } from "./kit/view3d";
import { easeIO, projector } from "./kit/cam";
import { ll } from "./kit/globe";
import { C, FONT, Lbl, Svg, ramp, textW } from "./kit/draw";
import { GlassStack3D, paneCorner, sp, stackBasis, glassCam, useTexSet } from "./kit/glass";

export const FPS = 30;
export const FRAMES = 780;

type Manifest = { run: string; leads_h: number[]; centre: [number, number]; ranges: Record<string, [number, number]>; ramps: Record<string, string[]>;
  atlantic_low: { lead_h: number; hPa: number; lat: number; lon: number }[] };

const useManifest = () => {
  const [m, setM] = useState<Manifest | null>(null);
  const [h] = useState(() => delayRender("manifest"));
  useEffect(() => { fetch(staticFile("manifest.json")).then((r) => r.json()).then((j) => { setM(j); continueRender(h); }).catch((e) => cancelRender(e)); }, [h]);
  return m;
};

// panes from the globe outwards
const FIELDS = ["t2m", "mslp", "pwat", "tcc", "prate"] as const;
const NAME: Record<string, string> = { t2m: "Temperature at 2 m", mslp: "Sea-level pressure", pwat: "Precipitable water", tcc: "Cloud cover", prate: "Precipitation rate" };
const UNIT: Record<string, string> = { t2m: "°C", mslp: "", pwat: "kg m⁻²", tcc: "%", prate: "mm h⁻¹" };
const TICKS: Record<string, [string, string]> = { t2m: ["−30", "30"], mslp: ["", ""], pwat: ["0", "60"], tcc: ["0", "100"], prate: ["0.1", "10"] };
const NL = 8;
const LIFT_K = 1; // the isobar pane lifts out

const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const fmt = (d: Date) => `${String(d.getUTCHours()).padStart(2, "0")} UTC ${d.getUTCDate()} ${MON[d.getUTCMonth()]} ${d.getUTCFullYear()}`;

/** the lead clock: hold each lead 50 frames, cross-fade over 25 (7 steps from f 90) */
const leadAt = (f: number) => {
  const s = (f - 90) / 75;
  if (s <= 0) return 0;
  if (s >= NL - 1) return NL - 1;
  const i = Math.floor(s), r = (s - i) * 75;
  return i + easeIO(Math.max(0, (r - 50) / 25));
};

const ColourBar: React.FC<{ x: number; y: number; w: number; h: number; stops: string[]; id: string; o: number }> = ({ x, y, w, h, stops, id, o }) => (
  <g opacity={o}>
    <defs><linearGradient id={id} x1="0" y1="0" x2="1" y2="0">{stops.map((c, i) => <stop key={i} offset={i / (stops.length - 1)} stopColor={c} />)}</linearGradient></defs>
    <rect x={x} y={y} width={w} height={h} rx={4} fill={`url(#${id})`} stroke={C.muted} strokeWidth={1.5} />
  </g>
);

export const GlassStack: React.FC<{ debugAxis?: boolean }> = ({ debugAxis = false }) => {
  const f = useCurrentFrame();
  const man = useManifest();
  const relief = useTexSet([staticFile("relief.png")], true);
  const urls = FIELDS.flatMap((k) => Array.from({ length: NL }, (_, i) => staticFile(`${k}_${i}.png`)));
  const tex = useTexSet(urls);
  const lat0 = man?.centre[0] ?? 45, lon0 = man?.centre[1] ?? -30;

  // ---- choreography
  const fadeIn = ramp(f, 0, 40);
  const gap = 0.07 + 0.5 * ramp(f, 45, 210);
  const az = -(30 + 10 * ramp(f, 45, 210) + 9 * Math.max(0, Math.min(1, (f - 210) / 570)));
  const el = 12 + 8 * ramp(f, 45, 300);
  const lift = 0.85 * (ramp(f, 330, 410) - ramp(f, 570, 640));
  const dim = 1 - 0.65 * (ramp(f, 330, 410) - ramp(f, 570, 640));
  const t = leadAt(f);
  // debugAxis: look straight down the pane normal with a near-orthographic lens (fov 2.4 deg from 150 radii): every pane's
  // disc must then sit exactly on the globe's silhouette and its coast lines on the globe's coast lines (the alignment check)
  const st = debugAxis ? glassCam(lat0, lon0, { az: 0, el: 0, dist: 150, look: [0, 0, 0], fov: 2.4, pp: [660, 500] })
    : glassCam(lat0, lon0, { az, el, dist: 13.2, look: [0, 0.2, 1.9], fov: 30, pp: [660, 480] });
  const pr = projector(st, 1920, 1080);

  const panes = tex ? FIELDS.map((k, i) => ({ frames: tex.slice(i * NL, (i + 1) * NL), o: debugAxis ? (i === 4 ? 0.9 : 0) : fadeIn * (i === LIFT_K ? 1 : dim), lift: i === LIFT_K && !debugAxis ? lift : 0 })) : [];

  // ---- overlay numbers
  const runD = man ? new Date(man.run.slice(0, 19) + "Z") : null;
  const li = Math.round(t), leadH = man ? man.leads_h[li] : 0;
  const valid = runD ? new Date(runD.getTime() + leadH * 3600e3) : null;

  // legend rows (outermost pane on top) and leader lines to each pane's upper-right corner
  const LX = 1300, rowY = (r: number) => 250 + r * 122;
  const order = [...FIELDS]; // innermost on top: with the camera above the stack its corner is the highest, so leaders never cross
  // the Atlantic low on the lifted isobar pane (interpolated between leads), from the manifest
  let low: { p: [number, number]; hPa: number } | null = null;
  if (man && lift > 0.6) {
    const i0 = Math.floor(t), i1 = Math.min(NL - 1, i0 + 1), w = t - i0, A = man.atlantic_low[i0], Bq = man.atlantic_low[i1];
    const la = A.lat + (Bq.lat - A.lat) * w, lo = A.lon + (Bq.lon - A.lon) * w;
    const B = stackBasis(lat0, lon0), q = ll(la, lo);
    const u = q[0] * B.e[0] + q[1] * B.e[1] + q[2] * B.e[2], v = q[0] * B.n[0] + q[1] * B.n[1] + q[2] * B.n[2];
    const p = pr.p(sp(B, u, v + lift, 1.25 + LIFT_K * gap + 0.25 * lift + 0.02));
    low = { p, hPa: Math.round(A.hPa + (Bq.hPa - A.hPa) * w) };
  }
  const lowO = ramp(f, 420, 450) * (1 - ramp(f, 545, 570));

  return (
    <AbsoluteFill style={{ backgroundColor: C.bg, fontFamily: FONT }}>
      <View3D st={st} amb={1.0} dir={1.7} key3={[st.pos[0] * 1.2 + 3 * stackBasis(lat0, lon0).n[0], st.pos[1] * 1.2 + 3 * stackBasis(lat0, lon0).n[1], st.pos[2] * 1.2 + 3 * stackBasis(lat0, lon0).n[2]]}>
        <GlassStack3D lat0={lat0} lon0={lon0} panes={panes} t={t} relief={relief ? relief[0] : null} gap={gap} globeO={fadeIn} />
      </View3D>
      <Svg>
        {/* what it shows + run / valid time */}
        <Lbl x={80} y={92} t="Five layers of one weather forecast" s={54} c={C.key} a="start" />
        {runD ? <Lbl x={80} y={152} t={`NOAA GFS · run ${fmt(runD)}`} s={44} c={C.muted} a="start" /> : null}
        {valid ? <>
          <Lbl x={1300} y={92} t={`+${leadH} h`} s={54} c={C.gold} a="start" />
          <Lbl x={1300 + textW(`+${leadH} h`, 54) + 22} y={92} t="forecast" s={44} c={C.muted} a="start" />
          <Lbl x={1300} y={152} t={`valid ${fmt(valid)}`} s={44} c={C.ink} a="start" />
        </> : null}
        {/* legend: label, colour bar with end values and unit; a gold leader to the pane */}
        {man ? order.map((k, r) => {
          const kk = FIELDS.indexOf(k), y = rowY(r);
          const active = kk === LIFT_K ? 1 : (dim - 0.35) / 0.65;
          const corner = pr.p(paneCorner(lat0, lon0, kk, gap, kk === LIFT_K ? lift : 0, 0.92, 0.92));
          const bx = LX + textW(TICKS[k][0], 44) + (TICKS[k][0] ? 16 : 0), bw = 230;
          return (
            <g key={k} opacity={fadeIn * (0.35 + 0.65 * active)}>
              <g opacity={active * active}>
                <line x1={LX - 22} y1={y - 14} x2={corner[0]} y2={corner[1]} stroke={C.gold} strokeWidth={3} strokeOpacity={0.55} />
                <circle cx={corner[0]} cy={corner[1]} r={6} fill={C.gold} fillOpacity={0.8} />
              </g>
              <Lbl x={LX} y={y} t={NAME[k]} s={44} c={C.key} a="start" />
              {k === "mslp" ? <>
                <line x1={LX} y1={y + 46} x2={LX + 120} y2={y + 46} stroke="#f4ecd2" strokeWidth={5} />
                <Lbl x={LX + 140} y={y + 62} t="isobars every 4 hPa" s={44} c={C.ink} a="start" />
              </> : <>
                <Lbl x={LX} y={y + 62} t={TICKS[k][0]} s={44} c={C.muted} a="start" />
                <ColourBar x={bx} y={y + 30} w={bw} h={26} stops={man.ramps[k]} id={`cb_${k}`} o={1} />
                <Lbl x={bx + bw + 16} y={y + 62} t={`${TICKS[k][1]} ${UNIT[k]}`} s={44} c={C.muted} a="start" />
              </>}
            </g>
          );
        }) : null}
        {/* the North Atlantic low on the lifted isobar pane */}
        {low && lowO > 0 ? <g opacity={lowO}>
          <circle cx={low.p[0]} cy={low.p[1]} r={16} fill="none" stroke={C.gold} strokeWidth={5} />
          <line x1={low.p[0] - 14} y1={low.p[1] - 14} x2={low.p[0] - 120} y2={low.p[1] - 110} stroke={C.gold} strokeWidth={5} />
          <Lbl x={low.p[0] - 130} y={low.p[1] - 122} t={`Low · ${low.hPa} hPa`} s={44} c={C.key} a="end" />
        </g> : null}
        {/* source line (below the picture, never on it) */}
        <text x={80} y={852} fontSize={30} fill={C.muted} fontFamily={FONT}>
          Data: dynamical.org / NOAA GFS (CC BY 4.0) · relief: ETOPO 2022 (NOAA NCEI) · each pane: the hemisphere centred on 45°N, 30°W
        </text>
        <text x={80} y={898} fontSize={30} fontWeight={700} fill="#f1ede4" fontFamily={FONT}>Map created by Shammunul Islam</text>
      </Svg>
    </AbsoluteFill>
  );
};
