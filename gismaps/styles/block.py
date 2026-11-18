"""gismaps.styles.block: a 3-D block diagram (terrain slab seen obliquely) with a data field draped on it and a linked chart.

The picture: a rectangular piece of the Earth cut out like a cake, relief exaggerated, seen from above and to one
side, with walls and a soft shadow. A field is draped on the surface and changes with time; a small chart of the same
field at one or two places carries the playhead, so map and chart are read together.

How it is drawn (no 3-D library): the camera is parallel and looks obliquely at the slab. The slab is first turned so
the view direction is "up the rows"; then, column by column, a pixel at screen height sy sees the nearest row whose
(exaggerated) surface reaches that height. That search is done once for the whole image with one ``minimum.at`` and one
``minimum.accumulate``. The result is a lookup table "screen pixel -> (lat, lon)"; every later frame is only a
sampling of the day's field through that table, so an animation costs a fraction of a second per frame.

This map: Storm Ciaran's 10-m wind (ERA5, hourly) on the relief of north-west Europe (ETOPO 2022).

API
---
``Block(z, lat, lon, azimuth, tilt, km_per_px, zexag, ...)``  geometry; ``.sample(grid_field, lats, lons)``,
                                                               ``.compose(rgb_surface)``, ``.screen(lat, lon)``
``CiaranBlock(wind_data, terrain_data)``                       ``.frame(hour)`` / ``.poster(path)`` / ``.animate(path)``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import common as C

BIG = np.iinfo(np.int32).max


class Block:
    """Oblique parallel view of a terrain slab.

    Parameters
    ----------
    z : (ny, nx) heights in metres on a regular lat/lon grid (row 0 = south). Sea is drawn flat at 0.
    lat, lon : 1-D coordinates of ``z``.
    azimuth : degrees; the direction the camera looks towards, clockwise from north (0 = looking north).
    tilt : degrees above the horizon of the camera (90 = straight down, a map; 30-40 = a block diagram).
    km_per_px : ground scale across the screen.
    zexag : vertical exaggeration (state it on the figure).
    thick_km : thickness of the slab's wall below sea level, in exaggerated screen terms (km at the vertical scale).
    """

    def __init__(self, z, lat, lon, azimuth: float = 25.0, tilt: float = 34.0, km_per_px: float = 1.75, zexag: float = 40.0,
                 thick_m: float = 1500.0, W: int = 1500, H: int = 980, ss: int = 2, y_base: float | None = None):
        self.lat, self.lon = np.asarray(lat, float), np.asarray(lon, float)
        self.W, self.H, self.ss = W, H, ss
        self.zexag, self.azimuth, self.tilt = zexag, azimuth, tilt
        z = np.maximum(np.asarray(z, np.float32), 0)                        # the sea is a flat surface at 0 m
        ny, nx = z.shape
        lat0 = float(self.lat.mean())
        dy = 111.195 * abs(self.lat[1] - self.lat[0])                       # km per row
        dx = 111.195 * abs(self.lon[1] - self.lon[0]) * np.cos(np.deg2rad(lat0))   # km per column at the middle latitude
        Wp, kpp = W * ss, km_per_px / ss
        a, t = np.deg2rad(azimuth), np.deg2rad(tilt)
        Lx, Ly = nx * dx, ny * dy
        depth = abs(Lx * np.sin(a)) + abs(Ly * np.cos(a))
        V = int(np.ceil(depth / kpp)) + 4
        u = (np.arange(Wp) - Wp / 2) * kpp                                  # km to the right of the slab's centre
        v = (np.arange(V) - V / 2) * kpp                                    # km away from the camera
        U, Vv = np.meshgrid(u, v)
        east = U * np.cos(a) + Vv * np.sin(a)                               # km east / north of the centre
        north = -U * np.sin(a) + Vv * np.cos(a)
        sc = east / dx + (nx - 1) / 2                                       # fractional column / row in z
        sr = north / dy + (ny - 1) / 2
        valid = (sc >= 0) & (sc <= nx - 1) & (sr >= 0) & (sr <= ny - 1)
        zr = ndi.map_coordinates(z, [sr, sc], order=1, mode="nearest")
        self.px_per_m = zexag / 1000.0 / kpp * np.cos(t)                    # screen px per metre of height
        thick = thick_m * self.px_per_m
        Hp = H * ss
        y0 = (Hp - 30 * ss - thick) if y_base is None else y_base * ss      # screen y of sea level at the nearest row
        rowstep = kpp * np.sin(t) / kpp                                     # screen px per row
        sy = y0 - np.arange(V)[:, None] * rowstep - zr * self.px_per_m      # screen y of each surface cell
        self._geom = (y0, rowstep, V, Wp, Hp, kpp, dx, dy, nx, ny, a)
        # nearest row whose surface reaches each screen pixel
        m = np.clip(np.floor(sy), 0, Hp - 1).astype(np.int32)
        rows = np.broadcast_to(np.arange(V, dtype=np.int32)[:, None], (V, Wp))
        cols = np.broadcast_to(np.arange(Wp, dtype=np.int32)[None, :], (V, Wp))
        A = np.full((Hp, Wp), BIG, np.int32)
        np.minimum.at(A, (m[valid], cols[valid]), rows[valid])
        idx = np.minimum.accumulate(A, axis=0)                              # every pixel BELOW a surface top sees it too
        hit = idx < BIG
        front = np.where(valid.any(0), valid.argmax(0), -1)                 # nearest valid row of each column
        r = np.where(hit, idx, 0)
        cc = np.broadcast_to(np.arange(Wp)[None, :], (Hp, Wp))
        SY = np.broadcast_to(np.arange(Hp)[:, None], (Hp, Wp))
        top_front = m[np.clip(front, 0, V - 1), np.arange(Wp)]
        base_front = y0 - np.clip(front, 0, V - 1) * rowstep + thick
        is_front = hit & (r == front[None, :])
        wall = is_front & (SY > top_front[None, :])
        surf = hit & ~wall
        wall &= SY <= base_front[None, :]
        self.surf, self.wall = surf, wall
        self.pr = np.where(surf | wall, sr[r, cc], 0).astype(np.float32)    # fractional row / col in z of every pixel
        self.pc = np.where(surf | wall, sc[r, cc], 0).astype(np.float32)
        self.plat = self.lat[0] + self.pr * (self.lat[1] - self.lat[0])
        self.plon = self.lon[0] + self.pc * (self.lon[1] - self.lon[0])
        # which wall: the nearest cell lies on the south/north edge or on the west/east edge of the slab
        fr, fc = sr[np.clip(front, 0, V - 1), np.arange(Wp)], sc[np.clip(front, 0, V - 1), np.arange(Wp)]
        ns_edge = np.minimum(fr, ny - 1 - fr) * dy < np.minimum(fc, nx - 1 - fc) * dx
        self.wall_light = np.where(ns_edge, 1.0, 0.62)[None, :] * np.ones((Hp, 1))
        self.depth_in_wall = (SY - top_front[None, :]) / max(thick, 1)
        # relief shading from the north-west, on the terrain grid, then through the lookup
        zs = ndi.gaussian_filter(z, 0.6) * zexag / 1000.0
        gy, gx = np.gradient(zs, dy, dx)
        lx, ly, lz = -0.55, 0.55, 0.63                                      # light from the north-west, 39 degrees up
        shade = (-gx * lx - gy * ly + lz) / np.sqrt(gx**2 + gy**2 + 1)
        self.shade = self.sample_index(np.clip(shade, 0, 1.3))
        self.height = self.sample_index(z)
        self.alpha = (surf | wall).astype(np.float32)
        self._shadow = None
        self._z_for_labels = z

    def sample_index(self, arr: np.ndarray, order: int = 1) -> np.ndarray:
        """Array on the terrain grid -> supersampled screen pixels."""
        return ndi.map_coordinates(np.asarray(arr, np.float32), [self.pr, self.pc], order=order, mode="nearest")

    def sample(self, field: np.ndarray, lats: np.ndarray, lons: np.ndarray, order: int = 1) -> np.ndarray:
        """Field on another regular lat/lon grid (e.g. a reanalysis) -> supersampled screen pixels."""
        r = (self.plat - lats[0]) / (lats[1] - lats[0])
        c = (self.plon - lons[0]) / (lons[1] - lons[0])
        return ndi.map_coordinates(np.asarray(field, np.float32), [r, c], order=order, mode="nearest")

    def compose(self, rgb: np.ndarray, wall_rgb=("#39415a", "#232a3d"), shadow: bool = True) -> np.ndarray:
        """Supersampled surface colours (Hp, Wp, 3) -> final RGBA (H, W, 4) with walls, light and a ground shadow."""
        out = np.zeros(rgb.shape[:2] + (3,), np.float32)
        lit = np.clip(rgb * (0.42 + 0.75 * self.shade[..., None]), 0, 1)
        out[self.surf] = lit[self.surf]
        top, bot = C.rgb(wall_rgb[0]), C.rgb(wall_rgb[1])
        d = np.clip(self.depth_in_wall, 0, 1)[..., None]
        wall = (top * (1 - d) + bot * d) * self.wall_light[..., None]
        strata = (np.sin(self.depth_in_wall * 38) > 0.86)[..., None] * 0.05
        out[self.wall] = np.clip(wall + strata, 0, 1)[self.wall]
        # a light line along the top edge of the walls
        edge = self.wall & ndi.binary_dilation(self.surf, iterations=self.ss)
        out[edge] = np.clip(out[edge] + 0.22, 0, 1)
        a = self.alpha
        if shadow:
            if self._shadow is None:                                         # the geometry never changes: compute once
                self._shadow = C.drop_shadow(a, 26 * self.ss, 20 * self.ss, 22 * self.ss, 0.55)
            sh = self._shadow
            a_all = np.maximum(a, sh * (1 - a))
            out = out * a[..., None]                                         # shadow pixels are black with alpha sh
            rgba = np.dstack([np.where(a_all[..., None] > 0, out / np.maximum(a_all[..., None], 1e-6), 0), a_all])
        else:
            rgba = np.dstack([out, a])
        s = self.ss
        H, W = rgba.shape[0] // s, rgba.shape[1] // s
        # average premultiplied so edges do not fringe
        pm = np.dstack([rgba[..., :3] * rgba[..., 3:], rgba[..., 3]]).reshape(H, s, W, s, 4).mean(axis=(1, 3))
        return np.dstack([pm[..., :3] / np.maximum(pm[..., 3:], 1e-6), pm[..., 3]])

    def screen(self, lat, lon, z_m: float | np.ndarray | None = None):
        """(lat, lon[, height]) -> screen pixel (x, y) in the final image; height defaults to the terrain there."""
        y0, rowstep, V, Wp, Hp, kpp, dx, dy, nx, ny, a = self._geom
        sc = (np.asarray(lon, float) - self.lon[0]) / (self.lon[1] - self.lon[0])
        sr = (np.asarray(lat, float) - self.lat[0]) / (self.lat[1] - self.lat[0])
        east, north = (sc - (nx - 1) / 2) * dx, (sr - (ny - 1) / 2) * dy
        u = east * np.cos(a) - north * np.sin(a)
        v = east * np.sin(a) + north * np.cos(a)
        if z_m is None:
            z_m = ndi.map_coordinates(self._z_for_labels, [np.atleast_1d(sr), np.atleast_1d(sc)], order=1, mode="nearest")
            z_m = z_m if np.ndim(lat) else float(z_m[0])
        x = u / kpp + Wp / 2
        y = y0 - (v / kpp + V / 2) * rowstep - np.asarray(z_m) * self.px_per_m
        return x / self.ss, y / self.ss


class CiaranBlock:
    """Storm Ciaran's 10-m wind draped on the relief of north-west Europe, with a linked chart at two places."""

    W, H = 1920, 1080
    BX, BY = 250, 150                  # where the block image sits in the frame
    PLACES = [("Brest", 48.39, -4.49), ("London", 51.51, -0.13), ("Paris", 48.86, 2.35), ("Dublin", 53.35, -6.26),
              ("Madrid", 40.42, -3.70), ("Milan", 45.46, 9.19)]
    SERIES = [("Brest", 48.39, -4.49, C.AMBER), ("London", 51.51, -0.13, "#7fc4ff")]
    VMIN, VMAX, CUT = 8.0, 30.0, 8.0   # colours start at CUT m/s; calmer air leaves the terrain bare

    def __init__(self, wind: dict, terrain: dict, azimuth: float = 24.0, tilt: float = 35.0):
        from matplotlib.colors import LinearSegmentedColormap
        # a "gale" ramp: teal -> sand -> orange -> red -> pale pink, so the core of the gale is the lightest
        # colour and reads first on the dark ground (not a rainbow: one pass through warm hues after the teal)
        self.cmap = LinearSegmentedColormap.from_list("gale", ["#2b6f8f", "#4fb0a5", "#e9d87a", "#f39a3e", "#e2503a", "#f6c9d8"])
        self.wlat, self.wlon = wind["lat"], wind["lon"]
        self.time = [str(t) for t in wind["time"]]
        self.speed = np.hypot(wind["u10_cms"].astype(np.float32), wind["v10_cms"].astype(np.float32)) / 100.0
        self.msl = wind["msl_hpa10"].astype(np.float32) / 10.0
        z, lat, lon = terrain["z_m"], terrain["lat"], terrain["lon"]
        b = self.block = Block(z, lat, lon, azimuth=azimuth, tilt=tilt, km_per_px=1.72, zexag=40.0, W=1500, H=900)
        # the bare ground: a dark sea, land tinted by height (all muted so the wind colours carry the picture)
        h = b.height
        land = b.sample_index((z > 0).astype(np.float32)) > 0.5
        tint = np.clip(h / 2500.0, 0, 1)[..., None]
        ground = np.where(land[..., None], C.rgb("#5b647a") * (1 - tint) + C.rgb("#c9cdd6") * tint, C.rgb("#1d2b4a"))
        self.ground = ground.astype(np.float32)
        self.series = []
        for name, la, lo, col in self.SERIES:
            i, j = int(np.argmin(np.abs(self.wlat - la))), int(np.argmin(np.abs(self.wlon - lo)))
            self.series.append((name, col, self.speed[:, i, j]))

    def frame(self, hour: float, path, scale: float = 1.0) -> Path:
        """``hour`` = hours after 00 UTC on 1 November 2023 (0..47), fractional hours are blended linearly."""
        if path is None:                     # not this process's frame (see common.Movie)
            return None
        b = self.block
        k = int(np.clip(np.floor(hour), 0, len(self.time) - 2))
        t = float(np.clip(hour - k, 0, 1))
        sp = (1 - t) * self.speed[k] + t * self.speed[k + 1]
        s = b.sample(sp, self.wlat, self.wlon)
        col = C.colourise(s, self.cmap, self.VMIN, self.VMAX)
        a = np.clip((s - self.CUT) / 4.0, 0, 1) * 0.92                   # fades in between 8 and 12 m/s
        img = b.compose(C.over(self.ground, col, a))
        fig, ax = C.canvas(self.W, self.H, scale=scale)
        C.show(ax, img, self.BX, self.BY, z=2)
        for name, la, lo in self.PLACES:
            x, y = b.screen(la, lo)
            ax.plot([self.BX + x, self.BX + x], [self.BY + y, self.BY + y - 26], color=C.INK, lw=1.0, zorder=5)
            ax.plot(self.BX + x, self.BY + y, "o", ms=3.5, color=C.INK, zorder=5)
            C.halo(C.text(ax, self.BX + x, self.BY + y - 32, name, 19, ha="center", zorder=6), 2.8)
        # north arrow from the geometry itself
        x0, y0 = b.screen(41.0, 9.4, 0.0)
        x1, y1 = b.screen(42.6, 9.4, 0.0)
        ax_, ay_ = 1730, 930
        dx, dy = (x1 - x0), (y1 - y0)
        nrm = np.hypot(dx, dy)
        ax.annotate("", xy=(ax_ + dx / nrm * 70, ay_ + dy / nrm * 70), xytext=(ax_, ay_),
                    arrowprops=dict(arrowstyle="-|>", color=C.INK, lw=2.0), zorder=6)
        C.text(ax, ax_ + dx / nrm * 88, ay_ + dy / nrm * 88 + 8, "N", 22, weight="bold", ha="center")
        # title
        C.title_block(ax, 56, 30, "Storm Ciarán crosses north-west Europe", "Wind 10 m above the ground, on the real relief", size=44)
        C.text(ax, 56, 146, f"Relief: vertical ×{b.zexag:.0f} (not to scale). Sea drawn flat.", 17, colour=C.MUTED)
        d = int(self.time[k][8:10]) if t < 0.5 else int(self.time[k + 1][8:10])
        hh = int(self.time[k][11:13]) if t < 0.5 else int(self.time[k + 1][11:13])
        C.text(ax, 56, 250, f"{hh:02d} UTC", 64, colour=C.AMBER, family=C.SERIF, weight="bold")
        C.text(ax, 56, 292, f"{d} November 2023", 30, family=C.SERIF)
        # linked chart (top right)
        cx, cy, cw, ch = 1330, 70, 520, 170
        ax.add_patch(__import__("matplotlib.pyplot").pyplot.Rectangle((cx - 64, cy - 44), cw + 96, ch + 100, fc="#0e1528", ec="#2a3550",
                                                                    lw=1, zorder=3))
        top = 30.0
        hrs = np.arange(len(self.time))
        for v in (0, 10, 20, 30):
            yy = cy + ch - ch * v / top
            ax.plot([cx, cx + cw], [yy, yy], color="#2a3550", lw=0.8, zorder=4)
            C.text(ax, cx - 10, yy + 6, f"{v}", 16, colour=C.MUTED, ha="right", zorder=4)
        C.text(ax, cx - 48, cy - 16, "m/s", 16, colour=C.MUTED, zorder=4)
        for kk, lab in ((0, "00 UTC\n1 Nov"), (12, "12"), (24, "00 UTC\n2 Nov"), (36, "12"), (47, "23")):
            xx = cx + cw * kk / 47
            C.text(ax, xx, cy + ch + 22, lab.split("\n")[0], 15, colour=C.MUTED, ha="center", zorder=4)
            if "\n" in lab:
                C.text(ax, xx, cy + ch + 40, lab.split("\n")[1], 15, colour=C.MUTED, ha="center", zorder=4)
        for i, (name, colr, ser) in enumerate(self.series):
            xs = cx + cw * hrs / 47
            ys = cy + ch - ch * ser / top
            ax.plot(xs, ys, color=colr, lw=1.2, alpha=0.35, zorder=5)
            n = k + 1
            hx = np.r_[xs[:n], cx + cw * hour / 47]
            now = (1 - t) * ser[k] + t * ser[k + 1]
            hy = np.r_[ys[:n], cy + ch - ch * now / top]
            ax.plot(hx, hy, color=colr, lw=2.6, zorder=6)
            ax.plot(hx[-1], hy[-1], "o", ms=7, color=colr, zorder=7)
            C.text(ax, cx + 12 + i * 250, cy - 16, f"{name}: {now:.0f} m/s", 19, colour=colr, weight="bold", zorder=6)
        px = cx + cw * hour / 47
        ax.plot([px, px], [cy, cy + ch], color=C.INK, lw=1.0, alpha=0.6, zorder=5)
        C.colourbar(ax, 56, 880, 420, 22, self.cmap, self.VMIN, self.VMAX, [8, 12, 16, 20, 24, 28],
                    "WIND SPEED AT 10 m (m/s)  ·  bare ground: under 8 m/s", size=16)
        C.credit_block(ax, 56, self.H - 96, ["Wind: ERA5 hourly, Copernicus Climate Change Service (C3S) / ECMWF (grid cell nearest each city)",
                                             "Relief: ETOPO 2022, NOAA NCEI (public domain)"], size=16)
        return C.finish(fig, path)

    def poster(self, path, hour: float = 27.0, scale: float = 2.0) -> Path:
        return self.frame(hour, path, scale)

    def animate(self, out, frames_dir, fps: int = 30, per_hour: int = 6) -> Path:
        mv = C.Movie(frames_dir, fps)
        self.frame(0.0, mv.path())
        mv.hold(fps // 2)
        for i in range(1, 47 * per_hour + 1):
            self.frame(i / per_hour, mv.path())
        mv.hold(fps * 2)
        return mv.encode(out)
