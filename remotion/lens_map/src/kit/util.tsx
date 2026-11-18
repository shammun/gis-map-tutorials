// Small shared pieces of the clip: colours, the font, a smooth ramp and the pulsing ring that marks the low.
import React from "react";

/** Colour roles of the dark page. */
export const C = { bg: "#171f33", ink: "#e8ecf4", key: "#f4ecd2", gold: "#d8b23a", muted: "#9aa6bd" } as const;

export const FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif";

export type Pal = { bg: string; panel: string; ink: string; accent: string; accent2: string; muted: string; rule: string; warn: string; font: string };
export const SERIES_PAL: Pal = { bg: "#171f33", panel: "#1d2740", ink: "#e8ecf4", accent: "#d8b23a", accent2: "#f4ecd2", muted: "#9aa6bd",
  rule: "#3a4766", warn: "#e0707a", font: FONT };

/** Smooth 0→1 ramp between frames a and b (clamped). A pure function of the frame, so every render is the same. */
export const ramp = (frame: number, a: number, b: number): number => {
  if (b <= a) return frame >= b ? 1 : 0;
  const t = Math.min(1, Math.max(0, (frame - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

const c01 = (x: number) => Math.max(0, Math.min(1, x));
const eOut = (t: number) => 1 - Math.pow(1 - c01(t), 3);
const mix = (a: number, b: number, t: number) => a + (b - a) * t;

/** Rings that grow and fade round a point (SVG). */
export const PulseRing: React.FC<{ x: number; y: number; f: number; period?: number; r0?: number; r1?: number; color?: string; n?: number; w?: number; o?: number; dot?: number; pal?: Pal }> = ({
  x, y, f, period = 45, r0 = 10, r1 = 62, color, n = 2, w = 5, o = 1, dot = 0, pal = SERIES_PAL }) => {
  if (o <= 0.003) return null;
  const col = color ?? pal.accent;
  return (
    <g opacity={o}>
      {Array.from({ length: n }, (_, i) => {
        const u = (((f / period + i / n) % 1) + 1) % 1;
        return <circle key={i} cx={x} cy={y} r={mix(r0, r1, eOut(u))} fill="none" stroke={col} strokeWidth={w * (1 - 0.5 * u)} strokeOpacity={(1 - u) * (1 - u)} />;
      })}
      {dot > 0 ? <circle cx={x} cy={y} r={dot} fill={col} /> : null}
    </g>
  );
};
