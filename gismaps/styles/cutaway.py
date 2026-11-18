"""gismaps.styles.cutaway: a globe with a wedge cut out, the interior painted on the cut faces.

The picture: a globe shows a field on its surface. A wedge (a range of longitudes, north of the Equator) is removed,
and the three flat faces of the cut (two meridians and the equatorial plane) show the same quantity below the
surface. One object then holds a map and three vertical sections that meet at the right places.

How it is drawn: one ray per pixel (orthographic camera). The ray meets the sphere; if that point is outside the wedge
it is a surface pixel. If it is inside the wedge the ray flies on through the empty wedge until it leaves through one
of the three planes: that point (its radius and latitude or longitude) is a face pixel. No mesh, no z-buffer, exact
at every camera angle, about half a second per frame in numpy.

This map: annual-mean ocean temperature (World Ocean Atlas 2023) from the surface to 2000 m in the Pacific, cut
along 180 degrees, 110 W and the Equator. Depth is hugely exaggerated and labelled so.

API
---
``raycast(n, lon_c, lat_c, lon_a, lon_b)``  -> per-pixel kind (surface / face A / face B / equator face), lon, lat, radius
``OceanCutaway(data)``                      -> ``.globe(...)`` RGBA, ``.frame(...)``, ``.poster(path)``, ``.animate(path)``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import common as C

BG, SURF, FACE_A, FACE_B, FACE_EQ, VOID = 0, 1, 2, 3, 4, 5


def _basis(lon_c: float, lat_c: float):
    """Camera basis: f points from Earth's centre to the camera, r to screen right (east), u to screen up."""
    lo, la = np.deg2rad(lon_c), np.deg2rad(lat_c)
    f = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    r = np.array([-np.sin(lo), np.cos(lo), 0.0])
    u = np.cross(f, r)
    return f, r, u


def project(lon, lat, radius, lon_c: float, lat_c: float):
    """3-D point (lon, lat in degrees, radius in Earth radii) -> screen (x right, y up) in Earth radii, and depth."""
    f, r, u = _basis(lon_c, lat_c)
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    p = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1) * np.asarray(radius)[..., None]
    return p @ r, p @ u, p @ f


def raycast(n: int, lon_c: float, lat_c: float, lon_a: float, lon_b: float) -> dict:
    """Orthographic ray cast of a unit sphere with the wedge lon_a..lon_b (east of lon_a), lat >= 0 removed.

    Returns arrays (n, n): ``kind`` (BG, SURF, FACE_A, FACE_B, FACE_EQ, VOID), ``lon``, ``lat`` (degrees), ``rad``
    (0..1, radius of the hit point), ``shade`` (0..1 diffuse light) and ``xyz`` of the hit point.
    The wedge must be narrower than 180 degrees (it is then the intersection of three half-spaces).
    """
    f, r, u = _basis(lon_c, lat_c)
    s = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(s * 1.0, -s * 1.0)
    rr = X**2 + Y**2
    inside = rr < 1
    Z = np.sqrt(np.clip(1 - rr, 0, 1))
    P = X[..., None] * r + Y[..., None] * u + Z[..., None] * f              # front point of the sphere
    lon = np.rad2deg(np.arctan2(P[..., 1], P[..., 0]))
    lat = np.rad2deg(np.arcsin(np.clip(P[..., 2], -1, 1)))
    width = (lon_b - lon_a) % 360
    in_wedge = inside & (lat > 0) & (((lon - lon_a) % 360) < width) & (width > 1e-6)
    kind = np.where(inside, SURF, BG).astype(np.uint8)
    rad = np.where(inside, 1.0, np.nan)
    light = -0.45 * r + 0.55 * u + 0.70 * f
    light /= np.linalg.norm(light)
    shade = np.clip(P @ light, 0, 1)

    if in_wedge.any():
        a, b = np.deg2rad(lon_a), np.deg2rad(lon_b)
        normals = [np.array([np.sin(a), -np.cos(a), 0.0]),     # outward normal of the plane lon = lon_a
                   np.array([-np.sin(b), np.cos(b), 0.0]),     # outward normal of the plane lon = lon_b
                   np.array([0.0, 0.0, -1.0])]                 # outward normal of the equatorial plane
        d = -f                                                  # every ray travels away from the camera
        S = P[in_wedge]
        t_best = np.full(len(S), np.inf)
        k_best = np.zeros(len(S), np.uint8)
        for k, nrm in enumerate(normals):
            dn = d @ nrm
            if dn > 1e-9:                                       # the ray moves towards this plane
                t = -(S @ nrm) / dn
                better = (t >= 0) & (t < t_best)
                t_best[better] = t[better]
                k_best[better] = k
        H = S + np.where(np.isfinite(t_best), t_best, 0)[:, None] * d
        hr = np.linalg.norm(H, axis=1)
        ok = np.isfinite(t_best) & (hr <= 1.0)
        kk = np.where(ok, np.array([FACE_A, FACE_B, FACE_EQ], np.uint8)[k_best], VOID)
        kind[in_wedge] = kk
        rad[in_wedge] = np.where(ok, hr, np.nan)
        lon[in_wedge] = np.where(ok, np.rad2deg(np.arctan2(H[:, 1], H[:, 0])), np.nan)
        lat[in_wedge] = np.where(ok, np.rad2deg(np.arcsin(np.clip(H[:, 2] / np.maximum(hr, 1e-9), -1, 1))), np.nan)
        face_shade = np.array([max(0.0, -nrm @ light) for nrm in normals])     # the faces look INTO the wedge
        shade[in_wedge] = np.where(ok, 0.35 + 0.65 * face_shade[k_best], 0)
        P = P.copy()
        P[in_wedge] = H
    return {"kind": kind, "lon": lon, "lat": lat, "rad": rad, "shade": shade, "xyz": P}


class OceanCutaway:
    """Pacific temperature, surface to ``zmax`` m (WOA23 annual mean), on a cut-away globe."""

    W, H = 1920, 1080
    GX, GY, GD = 70, 70, 940          # globe box (left, top, diameter) in frame pixels
    R_IN = 0.50                       # radius at which the deepest level (zmax) is drawn

    def __init__(self, data: dict, lon_a: float = 180.0, lon_b: float = 250.0, zmax: float = 1000.0,
                 vmin: float = -2.0, vmax: float = 30.0):
        import cmocean
        self.cmap = cmocean.cm.thermal
        self.lat, self.lon, self.depth = data["lat"], data["lon"], data["depth_m"]
        t = data["t_c100"].astype(np.float32) / 100.0
        self.t = np.where(data["t_c100"] == -9999, np.nan, t)              # (depth, lat, lon)
        self.lon_a, self.lon_b, self.zmax, self.vmin, self.vmax = lon_a, lon_b, zmax, vmin, vmax
        self.land = C.land_raster(-180, 180, -90, 90, 0.1, "110m")
        sst = self.t[0]
        idx = ndi.distance_transform_edt(~np.isfinite(sst), return_distances=False, return_indices=True)
        self.sst = sst[tuple(idx)]                                         # coast gaps filled with the nearest ocean value

    # -- samplers (all bilinear on the 1-degree grid)
    def _rc(self, lat, lon):
        return (lat - self.lat[0]), ((lon - self.lon[0]) % 360)

    def surface_t(self, lat, lon):
        r, c = self._rc(lat, lon)
        f = np.concatenate([self.sst, self.sst[:, :1]], 1)
        return ndi.map_coordinates(f, [r, c], order=1, mode="nearest")

    def interior_t(self, lat, lon, rad):
        """Temperature at (lat, lon, radius): radius 1 = surface, R_IN = zmax; NaN below the sea floor / under land."""
        z = (1 - rad) / (1 - self.R_IN) * self.zmax
        k = np.interp(z, self.depth, np.arange(len(self.depth)))
        r, c = self._rc(lat, lon)
        f = np.concatenate([self.t, self.t[:, :, :1]], 2)
        nan = ~np.isfinite(f)
        v = ndi.map_coordinates(np.where(nan, 0, f), [k, r, c], order=1, mode="nearest")
        w = ndi.map_coordinates((~nan).astype(np.float32), [k, r, c], order=1, mode="nearest")
        return np.where((w > 0.6) & (rad >= self.R_IN), v / np.maximum(w, 1e-6), np.nan)

    def land_at(self, lat, lon):
        r = (90 - lat) / 0.1 - 0.5
        c = ((lon + 180) % 360) / 0.1 - 0.5
        return ndi.map_coordinates(self.land.astype(np.float32), [r, c], order=1, mode="nearest")

    # -- the globe image
    def globe(self, n: int, lon_c: float, lat_c: float, lon_b: float | None = None, ss: int = 2) -> np.ndarray:
        """RGBA (n, n, 4) of the cut-away globe seen from above (lon_c, lat_c). ``ss`` = supersampling factor."""
        lon_b = self.lon_b if lon_b is None else lon_b
        m = n * ss
        rc = raycast(m, lon_c, lat_c, self.lon_a, lon_b)
        kind, lon, lat, rad = rc["kind"], np.nan_to_num(rc["lon"]), np.nan_to_num(rc["lat"]), np.nan_to_num(rc["rad"], nan=1.0)
        surf = kind == SURF
        face = (kind == FACE_A) | (kind == FACE_B) | (kind == FACE_EQ)
        val = np.full((m, m), np.nan, np.float32)
        val[surf] = self.surface_t(lat[surf], lon[surf])
        val[face] = self.interior_t(lat[face], lon[face], rad[face])
        rgb = C.colourise(val, self.cmap, self.vmin, self.vmax)
        # land on the surface, rock on the faces (below the sea floor, under land, and the core that is not shown)
        landa = np.zeros((m, m), np.float32)
        landa[surf] = self.land_at(lat[surf], lon[surf])
        rgb = C.over(rgb, C.rgb("#565d70"), landa)
        rgb = C.over(rgb, C.rgb("#aeb7cc"), C.coast_alpha(ndi.gaussian_filter(landa, 0.8), 1.0 * ss) * surf * 0.55)
        rock = face & ~np.isfinite(val)
        core = face & (rad < self.R_IN)
        rgb[rock] = C.rgb("#565d70")
        rgb[core] = C.rgb("#1b2233") * (0.7 + 0.6 * rad[core, None])
        # isotherms every 5 degrees on the faces, depth arcs every 500 m
        band = np.floor(np.nan_to_num(val, nan=-99) / 5.0)
        iso = (ndi.maximum_filter(band, 3) != ndi.minimum_filter(band, 3)) & face & ~rock & ~ndi.binary_dilation(rock, iterations=2)
        rgb = C.over(rgb, np.zeros(3, np.float32), ndi.gaussian_filter(iso.astype(np.float32), 0.6 * ss) * 0.55)
        for z in np.arange(self.zmax / 4, self.zmax + 1, self.zmax / 4):
            rk = 1 - z / self.zmax * (1 - self.R_IN)
            arc = np.clip(1 - np.abs(rad - rk) * m / 2 / (0.9 * ss), 0, 1) * face
            rgb = C.over(rgb, np.ones(3, np.float32), arc * 0.38)
        # light, then the outline of every part
        shade = np.where(surf, 0.62 + 0.42 * rc["shade"], 0.72 + 0.34 * rc["shade"])
        rgb = np.clip(rgb * shade[..., None], 0, 1)
        edges = (ndi.maximum_filter(kind, 3) != ndi.minimum_filter(kind, 3)) & (kind != BG)
        rgb = C.over(rgb, C.rgb("#f1ede4"), ndi.gaussian_filter(edges.astype(np.float32), 0.5 * ss) * 0.9)
        rgb[kind == VOID] = C.rgb("#05070d")
        alpha = (kind != BG).astype(np.float32)
        out = np.dstack([rgb, alpha])
        if ss > 1:
            out = out.reshape(n, ss, n, ss, 4).mean(axis=(1, 3))
        return out

    # -- the frame
    def frame(self, path, lon_c: float = 212.0, lat_c: float = 30.0, opening: float = 1.0, labels: float = 1.0,
              scale: float = 1.0) -> Path:
        """``opening`` 0..1 opens the wedge from lon_a eastwards; ``labels`` 0..1 fades the face labels in."""
        if path is None:                     # not this process's frame (see common.Movie)
            return None
        lon_b = self.lon_a + (self.lon_b - self.lon_a) * opening
        fig, ax = C.canvas(self.W, self.H, scale=scale)
        n = self.GD if not C._PREVIEW else max(64, int(self.GD * C._PREVIEW["scale"]))   # a preview casts fewer rays
        g = self.globe(n, lon_c, lat_c, lon_b)
        # a soft halo so the globe sits in the dark frame
        pad = 60
        halo = np.zeros((self.GD + 2 * pad, self.GD + 2 * pad), np.float32)
        a = g[..., 3] if n == self.GD else ndi.zoom(g[..., 3], self.GD / n, order=1)
        halo[pad:pad + min(self.GD, a.shape[0]), pad:pad + min(self.GD, a.shape[1])] = a[:self.GD, :self.GD]
        glow = ndi.gaussian_filter(halo, 22) * 0.35
        C.show(ax, np.dstack([np.broadcast_to(C.rgb("#5f7fb8"), halo.shape + (3,)), glow]), self.GX - pad, self.GY - pad, z=1)
        C.show(ax, g, self.GX, self.GY, self.GD, self.GD, z=2)
        cx, cy, R = self.GX + self.GD / 2, self.GY + self.GD / 2, self.GD / 2

        def scr(lon, lat, rad=1.0):
            x, y, _ = project(lon, lat, rad, lon_c, lat_c)
            return cx + x * R, cy - y * R

        if labels > 0.02 and opening > 0.98:
            la = float(labels)
            # depth ticks along the edge where the 180-degree face meets the equatorial face
            for z in np.arange(0, self.zmax + 1, self.zmax / 4):
                rk = 1 - z / self.zmax * (1 - self.R_IN)
                x, y = scr(self.lon_a, 0, rk)
                ax.plot([x - 9, x], [y + 5, y], color=C.INK, lw=1.2, alpha=la, zorder=6)
                C.halo(C.text(ax, x - 13, y + 13, f"{z:.0f} m" if z else "surface", 17, ha="right", alpha=la, zorder=6), 2.4)
            for txt, lon, lat, rad, ha, dx, dy in ((f"{_lonname(self.lon_a)} meridian", self.lon_a, 52, 0.80, "center", 0, 0),
                                                 (f"{_lonname(self.lon_b)}", self.lon_b, 40, 0.80, "center", 0, 0),
                                                 ("Equator", (self.lon_a + self.lon_b) / 2, 0, 0.78, "center", 0, 0)):
                x, y = scr(lon, lat, rad)
                C.halo(C.text(ax, x + dx, y + dy, txt, 20, ha=ha, weight="bold", alpha=la, zorder=6), 3.0, "#141a28")
            x, y = scr((self.lon_a + self.lon_b) / 2, 0, 0.27)
            C.halo(C.text(ax, x, y, f"below {self.zmax:.0f} m: not shown", 16, colour=C.MUTED, ha="center", alpha=la, zorder=6), 2.0)
        # text column
        tx = 1090
        yy = C.title_block(ax, tx, 70, "Inside the Pacific", f"Ocean temperature from the surface to {self.zmax:.0f} m", size=54)
        notes = ["The globe shows the temperature of the sea surface.",
                 "A wedge is cut out north of the Equator, and its three",
                 "faces show the temperature below: warm water is a thin",
                 "skin, a few hundred metres deep, on a cold ocean.",
                 "",
                 "Along the Equator the warm layer is deep in the west",
                 "and shallow in the east."]
        for i, line in enumerate(notes):
            C.text(ax, tx, yy + 46 + i * 28, line, 20, colour="#c9cfdd")
        C.colourbar(ax, tx, 520, 620, 24, self.cmap, self.vmin, self.vmax, [0, 5, 10, 15, 20, 25, 30],
                    "TEMPERATURE (°C)   ·   dark lines on the faces: every 5 °C", size=17)
        exag = (1 - self.R_IN) * 6371e3 / self.zmax
        for i, line in enumerate([f"Depth is NOT to scale: {self.zmax:.0f} m is drawn as {100 * (1 - self.R_IN):.0f} % of Earth's radius,",
                                  f"about {round(exag, -2):,.0f} times too deep. At true scale the layer shown",
                                  "would be thinner than the white outline of the globe.",
                                  f"White arcs: every {self.zmax / 4:.0f} m. Grey: land and sea floor. Dark centre: not shown."]):
            C.text(ax, tx, 640 + i * 27, line, 18, colour=C.AMBER if i == 0 else C.MUTED)
        C.credit_block(ax, tx, self.H - 110, ["Temperature: World Ocean Atlas 2023 annual mean (all decades), NOAA NCEI (public domain)",
                                              "Land: Natural Earth (public domain)"], size=16)
        return C.finish(fig, path)

    def poster(self, path, scale: float = 2.0) -> Path:
        return self.frame(path, scale=scale)

    def animate(self, out, frames_dir, fps: int = 30) -> Path:
        mv = C.Movie(frames_dir, fps)
        n1, n2, n3 = 75, 75, 150
        for i in range(n1):                                   # the whole globe turns into place
            self.frame(mv.path(), lon_c=150 + 62 * C.ease(i / (n1 - 1)), lat_c=12 + 18 * C.ease(i / (n1 - 1)), opening=0, labels=0)
        for i in range(n2):                                   # the wedge opens eastwards from 180 degrees
            self.frame(mv.path(), opening=C.ease((i + 1) / n2), labels=0)
        for i in range(n3):                                   # labels in, a slow swing to give the faces depth
            sw = np.sin(2 * np.pi * i / n3)
            self.frame(mv.path(), lon_c=212 + 9 * sw, lat_c=30 + 4 * np.sin(4 * np.pi * i / n3), labels=min(1, i / 20))
        mv.hold(fps * 2)
        return mv.encode(out)


def _lonname(lon: float) -> str:
    lon = (lon + 180) % 360 - 180
    return "180°" if abs(abs(lon) - 180) < 1e-6 else f"{abs(lon):g}°{'W' if lon < 0 else 'E'}"
