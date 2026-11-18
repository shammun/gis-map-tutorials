// Colours, type and the SVG overlay (sizes in px at 1920 x 1080). Pure functions of their props.
import React from "react";

export const FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif";
export const C = {
  bg: "#171f33", // the page
  ink: "#e8ecf4",
  key: "#f4ecd2", // warm white: names
  gold: "#d8b23a", // pointers and the lead time
  muted: "#9aa6bd", // units, ticks, sources
} as const;

/** smooth 0 -> 1 ramp between frames a and b (clamped) */
export const ramp = (frame: number, a: number, b: number): number => {
  if (b <= a) return frame >= b ? 1 : 0;
  const t = Math.min(1, Math.max(0, (frame - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

export const Svg: React.FC<{ children: React.ReactNode; o?: number }> = ({ children, o = 1 }) => (
  <svg width={1920} height={1080} style={{ position: "absolute", left: 0, top: 0, opacity: o }}>{children}</svg>
);

/** SVG text label with a halo in the page colour */
export const Lbl: React.FC<{ x: number; y: number; t: string; c?: string; s?: number; a?: "start" | "middle" | "end"; o?: number; b?: boolean }> = ({
  x, y, t, c = C.ink, s = 54, a = "start", o = 1, b = true }) => {
  if (o <= 0) return null;
  return (
    <text x={x} y={y} fill={c} fontSize={s} fontFamily={FONT} fontWeight={b ? 650 : 500} textAnchor={a} dominantBaseline="central"
      stroke={C.bg} strokeWidth={0.18 * s} strokeLinejoin="round" paintOrder="stroke" opacity={o}>
      {t}
    </text>
  );
};
/** rough text width (Segoe UI semibold) for layout decisions */
export const textW = (t: string, s: number) => [...t].reduce((w, ch) => w + s * (/[ .,:;()|'·]/.test(ch) ? 0.3 : /[mwMW]/.test(ch) ? 0.85 : /[A-Z0-9°]/.test(ch) ? 0.6 : 0.53), 0);
