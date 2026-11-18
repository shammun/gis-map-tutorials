// Globe coordinates: unit sphere, Y up (north pole), longitude 0 on +Z, longitude 90 E on +X.
import { D2R, V3 } from "./cam";

/** (lat, lon, r) -> globe coordinates */
export const ll = (lat: number, lon: number, r = 1): V3 => {
  const la = lat * D2R, lo = lon * D2R;
  return [r * Math.cos(la) * Math.sin(lo), r * Math.sin(la), r * Math.cos(la) * Math.cos(lo)];
};
/** local east / north / up unit vectors at (lat, lon) */
export const enAt = (lat: number, lon: number): { e: V3; n: V3; u: V3 } => {
  const la = lat * D2R, lo = lon * D2R;
  return { e: [Math.cos(lo), 0, -Math.sin(lo)], n: [-Math.sin(la) * Math.sin(lo), Math.cos(la), -Math.sin(la) * Math.cos(lo)], u: ll(lat, lon) };
};
