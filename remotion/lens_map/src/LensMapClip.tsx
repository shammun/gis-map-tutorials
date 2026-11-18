// "One storm, many realities": three round lenses over one dark base map of the north-east Atlantic, each showing a
// different ERA5 field of the SAME hour of Storm Ciarán (1–2 November 2023): temperature at 850 hPa, specific humidity at
// 850 hPa, sea-level isobars with moving 10-m wind streaks. The lenses open one by one, sweep across the map, then follow
// the storm while one clock steps all three fields.
// The layers and manifest.json are made in Python (see ../README.md).
// Style after Fan Mo, #30DayMapChallenge 2026 Day 14 ("Borgesian map"). Map created by Shammunul Islam.
import React, { useEffect, useState } from "react";
import { AbsoluteFill, cancelRender, continueRender, delayRender, staticFile, useCurrentFrame } from "remotion";
import { C, FONT, PulseRing, SERIES_PAL, ramp } from "./kit/util";
import { LensMap, LensPt, LensSpec, lensR } from "./kit/lens";

export const FPS = 30;
export const FRAMES = 1230;

type Manifest = { w: number; h: number; km_per_px: number; times: string[]; low: { x: number; y: number; hPa: number }[];
  wind: { step: number; nx: number; ny: number; scale: number; uv: number[][] }; ramps: Record<string, string[]>; ranges: Record<string, [number, number]> };

const useManifest = () => {
  const [m, setM] = useState<Manifest | null>(null);
  const [h] = useState(() => delayRender("manifest"));
  useEffect(() => { fetch(staticFile("manifest.json")).then((r) => r.json()).then((j) => { setM(j); continueRender(h); }).catch((e) => cancelRender(e)); }, [h]);
  return m;
};

// ---- layout (page px)
const MX = 60, MY = 196, MW = 1140, MH = 625; // the map
const COLX = 1262, BARX = 1238; // text column
const BLOCK_Y = [252, 444, 636];
const R = 180;
const SERIF = "Georgia, 'Times New Roman', serif";
const RIM = { t: "#e8833a", q: "#2aa198", p: "#f4ecd2" };
const WIND = "#8fd0ff";

// ---- timeline (frames)
const F = { ctx: [0, 45], open: [[50, 95], [125, 170], [200, 245]], sweep: [300, 545], run: [565, 985], end: [1000, 1050] } as const;
const LAST = 15; // 03 UTC 2 Nov: the low is deepest (954 hPa) and the three lenses still fit on the map
const STEP = (F.run[1] - F.run[0]) / LAST; // frames per hour

const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const stamp = (iso: string) => `${iso.slice(11, 13)} UTC · ${Number(iso.slice(8, 10))} ${MON[Number(iso.slice(5, 7)) - 1]} ${iso.slice(0, 4)}`;
const pad = (i: number) => String(i).padStart(2, "0");
const lerp2 = (a: LensPt, b: LensPt, t: number): LensPt => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const hash = (n: number) => { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); };

/** bilinear wind (map px components, m/s) at a map pixel, cross-faded between two hours */
const windAt = (W: Manifest["wind"], i0: number, i1: number, m: number, x: number, y: number): [number, number] => {
  const gx = clamp(x / W.step - 0.5, 0, W.nx - 1.001), gy = clamp(y / W.step - 0.5, 0, W.ny - 1.001);
  const i = Math.floor(gx), j = Math.floor(gy), fx = gx - i, fy = gy - j;
  const at = (uv: number[], c: number) => {
    const k = (jj: number, ii: number) => uv[(jj * W.nx + ii) * 2 + c];
    return ((k(j, i) * (1 - fx) + k(j, i + 1) * fx) * (1 - fy) + (k(j + 1, i) * (1 - fx) + k(j + 1, i + 1) * fx) * fy) / W.scale;
  };
  const a0 = W.uv[i0], a1 = W.uv[i1];
  return [at(a0, 0) * (1 - m) + at(a1, 0) * m, at(a0, 1) * (1 - m) + at(a1, 1) * m];
};

/** Moving wind streaks inside a disc: each streak is a short streamline of the 10-m wind (length ∝ speed, 2.6 px per m/s)
 * with a bright dash running tail → head at a speed ∝ the wind speed, and a small head. Seeds are fixed to the map. */
const WindStreaks: React.FC<{ W: Manifest["wind"]; i0: number; i1: number; m: number; c: LensPt; r: number; f: number; w: number; h: number }> = ({ W, i0, i1, m, c, r, f, w, h }) => {
  const S = 31, out: React.ReactNode[] = [];
  for (let gy = 0; gy * S < h; gy++) for (let gx = 0; gx * S < w; gx++) {
    const id = gy * 97 + gx;
    let x = (gx + 0.15 + 0.7 * hash(id)) * S, y = (gy + 0.15 + 0.7 * hash(id + 0.5)) * S;
    if (Math.hypot(x - c[0], y - c[1]) > r - 5) continue;
    const v0 = windAt(W, i0, i1, m, x, y), sp0 = Math.hypot(v0[0], v0[1]);
    if (sp0 < 2.5) continue;
    const N = 7, len = 2.6 * sp0, ds = len / N;
    let d = `M${x.toFixed(1)} ${y.toFixed(1)}`, ux = v0[0] / sp0, uy = v0[1] / sp0;
    for (let k = 0; k < N; k++) {
      const v = windAt(W, i0, i1, m, x, y), sp = Math.hypot(v[0], v[1]) || 1;
      ux = v[0] / sp; uy = v[1] / sp; x += ux * ds; y += uy * ds;
      d += `L${x.toFixed(1)} ${y.toFixed(1)}`;
    }
    const dash = 0.42 * len, per = dash + len, off = -(((f * 0.12 * sp0 + hash(id + 0.25) * per) % per)) + dash;
    const hs = 7.5, bx = x - ux * hs * 1.5, by = y - uy * hs * 1.5;
    out.push(
      <g key={id}>
        <path d={d} fill="none" stroke={WIND} strokeOpacity={0.32} strokeWidth={2.6} strokeLinecap="round" />
        <path d={d} fill="none" stroke="#e9f6ff" strokeWidth={2.8} strokeLinecap="round" strokeDasharray={`${dash.toFixed(1)} ${len.toFixed(1)}`} strokeDashoffset={off.toFixed(2)} />
        <polygon points={`${x.toFixed(1)},${y.toFixed(1)} ${(bx - uy * hs * 0.62).toFixed(1)},${(by + ux * hs * 0.62).toFixed(1)} ${(bx + uy * hs * 0.62).toFixed(1)},${(by - ux * hs * 0.62).toFixed(1)}`} fill={WIND} />
      </g>,
    );
  }
  return <>{out}</>;
};

const Bar: React.FC<{ x: number; y: number; w: number; stops: string[]; id: string; tick?: boolean }> = ({ x, y, w, stops, id, tick }) => (
  <g>
    <defs><linearGradient id={id} x1="0" y1="0" x2="1" y2="0">{stops.map((c, i) => <stop key={i} offset={i / (stops.length - 1)} stopColor={c} />)}</linearGradient></defs>
    <rect x={x} y={y - 13} width={w} height={26} rx={4} fill={`url(#${id})`} stroke={C.muted} strokeWidth={1.5} />
    {tick ? <line x1={x + w / 2} y1={y - 19} x2={x + w / 2} y2={y + 19} stroke={C.bg} strokeWidth={3} /> : null}
  </g>
);
const Tx: React.FC<{ x: number; y: number; t: string; s?: number; c?: string; w?: number; serif?: boolean; italic?: boolean; a?: "start" | "middle" | "end"; ls?: number }> = ({
  x, y, t, s = 44, c = C.ink, w = 600, serif, italic, a = "start", ls }) => (
  <text x={x} y={y} fill={c} fontSize={s} fontFamily={serif ? SERIF : FONT} fontWeight={w} fontStyle={italic ? "italic" : undefined} textAnchor={a}
    dominantBaseline="central" letterSpacing={ls}>{t}</text>
);

export const LensMapClip: React.FC = () => {
  const f = useCurrentFrame();
  const man = useManifest();
  if (!man) return <AbsoluteFill style={{ backgroundColor: C.bg }} />;

  // ---- one clock: hour index (float). Each hour holds, then cross-fades to the next over its last 10 frames.
  const run = clamp((f - F.run[0]) / STEP, 0, LAST);
  const i0 = Math.min(LAST, Math.floor(run)), i1 = Math.min(LAST, i0 + 1);
  const mixH = clamp(((run - i0) * STEP - (STEP - 10)) / 10, 0, 1);
  const lowAt = (t: number): LensPt => { const a = Math.min(LAST, Math.floor(t)), b = Math.min(LAST, a + 1), w = t - a; return [man.low[a].x + (man.low[b].x - man.low[a].x) * w, man.low[a].y + (man.low[b].y - man.low[a].y) * w]; };
  const hPa = man.low[i0].hPa + (man.low[i1].hPa - man.low[i0].hPa) * (run - i0);
  const shown = mixH > 0.5 ? i1 : i0;

  // ---- lens centres (map px). P rides on the low; T and Q sit to its west (cold air behind, moist air to the south-west)
  const P: LensPt = lowAt(run);
  const homeT: LensPt = [P[0] - 185, clamp(P[1] - 80, R + 6, MH - R - 6)];
  // Q swings round P at a fixed distance: south-west of the low while the low is near the map's bottom edge, due south
  // (in the moist air feeding the storm) once there is room
  const qdy = Math.min(172, MH - R - 6 - P[1]), qs = clamp((qdy - 67) / (172 - 67), 0, 1);
  const homeQ: LensPt = [P[0] - Math.sqrt(Math.max(30 * 30, 176 * 176 - qdy * qdy)), P[1] + qdy];
  const out = (a: number, b: number, c: number) => ramp(f, a, b) - ramp(f, b + 10, c); // 0 → 1 → 0
  const Tc = lerp2(homeT, [905, 235], out(F.sweep[0], F.sweep[0] + 110, F.sweep[0] + 215));
  const Qc = lerp2(homeQ, [900, 430], out(F.sweep[0] + 30, F.sweep[0] + 140, F.sweep[1]));
  const open = F.open.map(([a, b]) => clamp((f - a) / (b - a), 0, 1));

  const url = (k: string, i: number) => staticFile(`${k}_${pad(i)}.png`);
  const lay = (k: string) => ({ src: url(k, i0), src2: i1 !== i0 ? url(k, i1) : undefined, mix: mixH });
  const lenses: LensSpec[] = [
    { c: Tc, r: R, rim: RIM.t, open: open[0], fill: lay("t"), overlap: 0 },
    { c: Qc, r: R, rim: RIM.q, open: open[1], fill: lay("q"), over: lay("ql"), overlap: 0 },
    { c: P, r: R, rim: RIM.p, open: open[2], fill: { src: staticFile("p.png") }, over: lay("pl"), overlap: 0,
      inside: <WindStreaks W={man.wind} i0={i0} i1={i1} m={mixH} c={P} r={lensR(R, open[2])} f={f} w={MW} h={MH} /> },
  ];

  // ---- labels: leaders only while the lenses are at home (opening and the final composition)
  const leadO = (k: number) => Math.max(ramp(f, F.open[k][0] + 30, F.open[k][1] + 15) * (1 - ramp(f, F.sweep[0] - 20, F.sweep[0] - 5)), ramp(f, F.end[0], F.end[0] + 30));
  const lblO = (k: number) => ramp(f, F.open[k][0] + 12, F.open[k][1]);
  const rimPt = (c: LensPt, deg: number): LensPt => [MX + c[0] + R * Math.cos((deg * Math.PI) / 180), MY + c[1] + R * Math.sin((deg * Math.PI) / 180)];
  const ends: LensPt[] = [rimPt(Tc, -60), rimPt(Qc, 60 - 35 * qs), rimPt(P, 0)];
  const blockOf = [0, 2, 1]; // lens index → block row (T top, P middle, Q bottom: the order of their rim points, so leaders never cross)
  const titleO = ramp(f, 4, 34), ctxO = ramp(f, F.ctx[0], F.ctx[1]);
  const endO = ramp(f, F.end[0] + 10, F.end[1]);
  const lowO = ramp(f, F.open[2][1] - 5, F.open[2][1] + 20);
  const NAMES = ["TEMPERATURE", "MOISTURE", "PRESSURE & WIND"], THEME = ["Warmth · 1.5 km up", "Fuel · vapour, 1.5 km up", "Motion · 4-hPa isobars"];
  const RIMS = [RIM.t, RIM.q, RIM.p];

  return (
    <AbsoluteFill style={{ backgroundColor: C.bg, fontFamily: FONT }}>
      <LensMap x={MX} y={MY} w={MW} h={MH} base={staticFile("ctx.png")} baseO={ctxO} lenses={lenses}
        under={/* where the low has been: its tracked centre at each past hour */
          <g opacity={lowO}>{man.low.slice(0, i0 + 1).map((p, i) => <circle key={i} cx={p.x} cy={p.y} r={4.5} fill={C.gold} fillOpacity={0.75} />)}</g>}>
        {/* the low the pressure lens rides on: L at the tracked sea-level-pressure minimum */}
        <g opacity={lowO}>
          <PulseRing x={P[0]} y={P[1]} f={f} r0={26} r1={58} color={C.gold} w={4} pal={SERIES_PAL} o={0.9} />
          <text x={P[0]} y={P[1] + 2} fill={C.key} fontSize={54} fontWeight={700} fontFamily={FONT} textAnchor="middle" dominantBaseline="central"
            stroke="#0e1524" strokeWidth={7} strokeLinejoin="round" paintOrder="stroke">L</text>
        </g>
      </LensMap>
      <svg width={1920} height={1080} style={{ position: "absolute", left: 0, top: 0 }}>
        {/* title (always there) and the closing line */}
        <g opacity={titleO} transform={`translate(0 ${(1 - titleO) * 14})`}>
          <text x={MX} y={82} fontSize={76} fontFamily={SERIF} fontWeight={700} fill={C.key} dominantBaseline="central">
            One storm, many <tspan fill={RIM.t} fontStyle="italic">realities.</tspan>
          </text>
        </g>
        <g opacity={endO}><Tx x={MX + 2} y={154} t="Each field shows one part. Together they show the storm." serif italic c={C.ink} w={400} /></g>
        {/* what the map shows: event, valid time, the tracked central pressure */}
        <g opacity={ctxO}>
          <Tx x={COLX} y={66} t="Storm Ciarán" s={54} c={C.key} w={700} />
          <Tx x={COLX} y={124} t={stamp(man.times[shown])} c={C.gold} />
          <g opacity={lowO}><Tx x={COLX} y={180} t={`low centre ${Math.round(hPa)} hPa`} c={C.ink} w={500} /></g>
        </g>
        {/* lens labels: coloured bar + caps name + italic theme + colour bar / key; a leader to the lens */}
        {[0, 1, 2].map((k) => {
          const y = BLOCK_Y[blockOf[k]], e = ends[k], a: LensPt = [BARX - 16, y + 28];
          return (
            <g key={k} opacity={lblO(k)}>
              <g opacity={leadO(k) * 0.9}>
                <line x1={a[0]} y1={a[1]} x2={e[0]} y2={e[1]} stroke={RIMS[k]} strokeWidth={3} strokeOpacity={0.75} />
                <circle cx={e[0]} cy={e[1]} r={6} fill={RIMS[k]} />
              </g>
              <rect x={BARX} y={y} width={9} height={160} fill={RIMS[k]} />
              <Tx x={COLX} y={y + 26} t={NAMES[k]} w={700} c={C.key} ls={1} />
              <Tx x={COLX} y={y + 80} t={THEME[k]} serif italic c={RIMS[k]} w={400} />
              {k === 0 ? <>
                <Tx x={COLX} y={y + 136} t="−12" c={C.muted} w={500} />
                <Bar x={COLX + 92} y={y + 136} w={250} stops={man.ramps.t} id="llm_t" tick />
                <Tx x={COLX + 358} y={y + 136} t="+12 °C" c={C.muted} w={500} />
              </> : k === 1 ? <>
                <Tx x={COLX} y={y + 136} t="0" c={C.muted} w={500} />
                <Bar x={COLX + 92} y={y + 136} w={250} stops={man.ramps.q} id="llm_q" />
                <Tx x={COLX + 358} y={y + 136} t="10 g/kg" c={C.muted} w={500} />
              </> : <>
                <line x1={COLX} y1={y + 136} x2={COLX + 64} y2={y + 136} stroke={RIM.p} strokeWidth={5} />
                <Tx x={COLX + 80} y={y + 136} t="isobar" c={C.muted} w={500} />
                <line x1={COLX + 250} y1={y + 136} x2={COLX + 300} y2={y + 136} stroke={WIND} strokeWidth={5} strokeLinecap="round" />
                <polygon points={`${COLX + 318},${y + 136} ${COLX + 298},${y + 125} ${COLX + 298},${y + 147}`} fill={WIND} />
                <Tx x={COLX + 334} y={y + 136} t="10-m wind" c={C.muted} w={500} />
              </>}
            </g>
          );
        })}
        {/* source line: below the map, never on it */}
        <text x={MX} y={852} fontSize={30} fill={C.muted} fontFamily={FONT} opacity={ctxO} dominantBaseline="central">
          Data: ERA5 (Copernicus C3S / ECMWF) · ETOPO 2022 · Natural Earth · lens style after Fan Mo, #30DayMapChallenge 2026 Day 14
        </text>
        <text x={MX} y={902} fontSize={29} fontWeight={700} fill="#f1ede4" fontFamily={FONT} opacity={ctxO} dominantBaseline="central">
          Map created by Shammunul Islam
        </text>
      </svg>
    </AbsoluteFill>
  );
};
