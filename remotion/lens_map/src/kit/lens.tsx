// LensMap — round "lenses" over ONE base map, each lens showing a different layer of the SAME map (another field,
// another dataset, another time). Style after Fan Mo, #30DayMapChallenge 2026 Day 14 ("Borgesian map": one city, many
// maps). The numpy twin of this rule is gismaps.lens.common.composite.
//
// Every layer is an image drawn on the same pixel grid as the base map (same projection, same extent), so a lens can sit
// anywhere and slide: what it shows always lines up with the map around it.
//   fill   opaque layer (a colour-mapped field, a thematic map). Drawn at alpha 1 where the lens lies on the base map and
//          at `overlap` (0..1) where it lies on an EARLIER lens.
//   over   transparent line work (contours, isobars). Drawn at full strength everywhere inside the lens.
//   overlap = 0.5  the two fills blend (the poster look; colours in the overlap no longer match a colour bar)
//   overlap = 0    the lower lens keeps its exact colours and the upper lens adds only its lines: the overlap is a true
//                  overlay and every colour still matches its colour bar. Use this for data fields.
// Pure function of its props (deterministic). No text inside: labels, colour bars and the source line belong to the caller.
import React from "react";
import { Img } from "remotion";

export type LensPt = [number, number];
/** one image, or a cross-fade from `src` to `src2` by `mix` 0..1 (time steps on one clock) */
export type LensLayer = { src: string; src2?: string; mix?: number };
export type LensSpec = {
  /** centre in map pixels (the layer images' own pixel grid) and radius in map pixels */
  c: LensPt; r: number;
  /** rim colour; `open` 0..1 is the reveal clock (the lens grows from its centre, its layer fades in) */
  rim: string; open?: number;
  fill: LensLayer; over?: LensLayer;
  /** alpha of `fill` where this lens lies on an earlier one (default 0.5) */
  overlap?: number;
  /** extra SVG drawn inside the lens in map pixels (animated wind streaks, a marker), clipped to the disc */
  inside?: React.ReactNode;
  rimW?: number;
};

const easeOut = (t: number) => 1 - Math.pow(1 - Math.max(0, Math.min(1, t)), 3);
/** radius on screen for a reveal clock 0..1 */
export const lensR = (r: number, open = 1) => r * easeOut(open);
/** point on the rim of a lens in the direction of `to` (for leader lines), in map pixels */
export const lensRimPoint = (c: LensPt, r: number, to: LensPt, inset = 0): LensPt => {
  const dx = to[0] - c[0], dy = to[1] - c[1], L = Math.hypot(dx, dy) || 1;
  return [c[0] + (dx / L) * (r - inset), c[1] + (dy / L) * (r - inset)];
};

const Layer: React.FC<{ l: LensLayer; w: number; h: number; transparent?: boolean }> = ({ l, w, h, transparent }) => {
  const m = l.src2 ? Math.max(0, Math.min(1, l.mix ?? 0)) : 0;
  const st: React.CSSProperties = { position: "absolute", left: 0, top: 0, width: w, height: h };
  return (
    <>
      {m < 0.999 ? <Img src={l.src} style={{ ...st, opacity: transparent ? 1 - m : 1 }} /> : null}
      {l.src2 && m > 0.001 ? <Img src={l.src2} style={{ ...st, opacity: m }} /> : null}
    </>
  );
};

/**
 * LensMap: base map at (x, y), size w × h on the page; `iw` × `ih` = pixel size of the layer images (default w × h).
 * Lenses are drawn in order; rims on top. `baseO` fades the base map. Everything is clipped to the map rectangle.
 * `under` = SVG (map px) on the base map but under the lenses (a storm track); `children` = SVG on top of everything.
 */
export const LensMap: React.FC<{ x: number; y: number; w: number; h: number; iw?: number; ih?: number; base: string; baseO?: number;
  lenses: LensSpec[]; glow?: boolean; under?: React.ReactNode; children?: React.ReactNode }> = ({ x, y, w, h, iw, ih, base, baseO = 1, lenses, glow = true, under, children }) => {
  const W = iw ?? w, H = ih ?? h;
  const R = lenses.map((L) => lensR(L.r, L.open ?? 1));
  const clip = (j: number): string => `circle(${R[j].toFixed(2)}px at ${lenses[j].c[0].toFixed(2)}px ${lenses[j].c[1].toFixed(2)}px)`;
  const full: React.CSSProperties = { position: "absolute", left: 0, top: 0, width: W, height: H };
  return (
    <div style={{ position: "absolute", left: x, top: y, width: w, height: h, overflow: "hidden" }}>
      <div style={{ ...full, transform: `scale(${w / W}, ${h / H})`, transformOrigin: "0 0" }}>
        <Img src={base} style={{ ...full, opacity: baseO }} />
        {under ? <svg width={W} height={H} style={{ position: "absolute", left: 0, top: 0 }}>{under}</svg> : null}
        {lenses.map((L, j) => {
          if (R[j] < 0.5) return null;
          const ov = L.overlap ?? 0.5;
          // alpha mask: `ov` inside every earlier lens, 1 elsewhere (the masks multiply)
          const gr = ov < 0.999 ? lenses.slice(0, j).map((E, i) => (R[i] < 0.5 ? null
            : `radial-gradient(circle at ${E.c[0].toFixed(2)}px ${E.c[1].toFixed(2)}px, rgba(0,0,0,${ov}) ${(R[i] - 0.5).toFixed(2)}px, #000 ${(R[i] + 0.5).toFixed(2)}px)`))
            .filter(Boolean) as string[] : [];
          const mask: React.CSSProperties = gr.length ? { maskImage: gr.join(", "), WebkitMaskImage: gr.join(", "), maskComposite: "intersect",
            WebkitMaskComposite: "source-in", maskRepeat: "no-repeat", WebkitMaskRepeat: "no-repeat", maskSize: `${W}px ${H}px`, WebkitMaskSize: `${W}px ${H}px` } : {};
          return (
            <div key={`f${j}`} style={{ ...full, clipPath: clip(j), opacity: Math.min(1, (L.open ?? 1) * 2.5) }}>
              <div style={{ ...full, ...mask }}><Layer l={L.fill} w={W} h={H} /></div>
            </div>
          );
        })}
        {lenses.map((L, j) => (R[j] < 0.5 || (!L.over && !L.inside) ? null : (
          <div key={`o${j}`} style={{ ...full, clipPath: clip(j), opacity: Math.min(1, (L.open ?? 1) * 2.5) }}>
            {L.over ? <Layer l={L.over} w={W} h={H} transparent /> : null}
            {L.inside ? <svg width={W} height={H} style={{ position: "absolute", left: 0, top: 0 }}>{L.inside}</svg> : null}
          </div>
        )))}
        <svg width={W} height={H} style={{ position: "absolute", left: 0, top: 0 }}>
          {lenses.map((L, j) => (R[j] < 0.5 ? null : (
            <g key={j}>
              {glow ? <circle cx={L.c[0]} cy={L.c[1]} r={R[j]} fill="none" stroke={L.rim} strokeWidth={(L.rimW ?? 4) * 3.5} strokeOpacity={0.14} /> : null}
              <circle cx={L.c[0]} cy={L.c[1]} r={R[j]} fill="none" stroke={L.rim} strokeWidth={L.rimW ?? 4} />
            </g>
          )))}
          {children}
        </svg>
      </div>
    </div>
  );
};
