"""gismaps.styles.epochs: epoch-stacked footprints ("where it was, time after time, on one map").

The picture: the outline of one thing (a river, an ice edge, a storm's low-pressure core, a heat dome) at N times.
Each time gets one colour of an ordered ramp. The animation shows each epoch alone, then the stack: oldest at the
bottom, newest on top, so the parts that only the older epochs covered stay visible as crescents.

This map: the area with mean-sea-level pressure below 960 hPa inside Storm Ciaran, every 3 hours (ERA5).

API
---
``footprint(field, threshold, seed_rc, below=True)`` -> boolean mask of the connected region that holds the seed
``track_minimum(fields, seed_rc, radius)``           -> follow a local minimum step by step
``epoch_colours(n, cmap)``                            -> n ordered colours
``stack(base, masks, colours, upto, ...)``            -> RGB of the stacked footprints
``CiaranFootprint(data)``                             -> ``.frame(...)`` / ``.poster(path)`` / ``.animate(path)``
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from . import common as C

PLACES = [("Brest", 48.39, -4.49), ("London", 51.51, -0.13), ("Dublin", 53.35, -6.26), ("Paris", 48.86, 2.35),
          ("Amsterdam", 52.37, 4.90), ("Edinburgh", 55.95, -3.19)]


def track_minimum(fields: np.ndarray, seed_rc: tuple[int, int], seed_i: int, radius: int = 10) -> np.ndarray:
    """Follow a local minimum through a (time, y, x) stack from ``seed_i`` both ways: at each step take the minimum
    within ``radius`` grid cells of the previous position (never the minimum of the whole box: another, deeper low
    may sit in the same map). Returns integer (time, 2) row/col."""
    n = fields.shape[0]
    rc = np.zeros((n, 2), int)

    def local(i, r0, c0):
        a, b = max(r0 - radius, 0), max(c0 - radius, 0)
        win = fields[i, a:r0 + radius + 1, b:c0 + radius + 1]
        j = np.unravel_index(np.argmin(win), win.shape)
        return a + j[0], b + j[1]

    rc[seed_i] = local(seed_i, *seed_rc)
    for i in range(seed_i + 1, n):
        rc[i] = local(i, *rc[i - 1])
    for i in range(seed_i - 1, -1, -1):
        rc[i] = local(i, *rc[i + 1])
    return rc


def footprint(field: np.ndarray, threshold: float, seed_rc: tuple[int, int], below: bool = True) -> np.ndarray:
    """Connected region of ``field < threshold`` (or ``>``) that contains ``seed_rc``; empty if the seed is outside."""
    m = field < threshold if below else field > threshold
    lab, _ = ndi.label(m)
    k = lab[seed_rc]
    return (lab == k) if k else np.zeros_like(m)


def epoch_colours(n: int, cmap="plasma", lo: float = 0.12, hi: float = 0.95) -> np.ndarray:
    """n colours in order along a perceptually ordered colour map (first = oldest)."""
    return plt.get_cmap(cmap)(np.linspace(lo, hi, n))[:, :3].astype(np.float32)


def outline(alpha: np.ndarray, width: float = 1.5) -> np.ndarray:
    """Outline (alpha) of a soft mask."""
    return C.coast_alpha(alpha, width)


def stack(base: np.ndarray, alphas: list[np.ndarray], colours: np.ndarray, upto: int, opacity: float = 0.93,
          solo: int | None = None, edge=(0.03, 0.04, 0.08)) -> np.ndarray:
    """Stack footprints 0..upto-1 (oldest first). ``solo`` = index drawn filled while the others are outlines only."""
    out = base.copy()
    for k in range(upto):
        if solo is None or k == solo:
            out = C.over(out, colours[k], alphas[k] * opacity)
            out = C.over(out, np.array(edge, np.float32), outline(alphas[k], 1.2) * 0.9)
        else:
            out = C.over(out, colours[k], outline(alphas[k], 1.6) * 0.95)
    return out


class CiaranFootprint:
    """Storm Ciaran, 1-2 November 2023: the area below ``threshold`` hPa every ``step`` hours (ERA5)."""

    W, H = 1920, 1080
    MX, MY, MW, MH = 640, 40, 1240, 1000

    def __init__(self, data: dict, threshold: float = 960.0, step: int = 3, first: str = "2023-11-01T21",
                 last: str = "2023-11-02T21"):
        self.lat, self.lon = data["lat"], data["lon"]
        self.time = [str(t) for t in data["time"]]
        self.p = data["msl_hpa10"].astype(np.float32) / 10.0
        self.threshold = threshold
        g = self.grid = C.MapGrid.fit("+proj=lcc +lat_1=45 +lat_2=58 +lat_0=52 +lon_0=-3.5", -60e3, 30e3, 2050e3, self.MW, self.MH)
        land = g.land("50m")
        base = g.ground(ocean="#0e1524", land="#1b2540")
        self.base = C.over(base, C.rgb("#55648c"), C.coast_alpha(land, 1.1) * 0.9)
        self.borders = g.lines(C.border_lines("50m"))
        # the low: seeded where Ciaran's centre was at 12 UTC on 1 November (48 N, 17 W), then followed hour by hour
        i0 = self.time.index("2023-11-01T12")
        seed = (int(np.argmin(np.abs(self.lat - 48.0))), int(np.argmin(np.abs(self.lon + 17.0))))
        self.rc = track_minimum(self.p, seed, i0, radius=8)
        self.idx = list(range(self.time.index(first), self.time.index(last) + 1, step))
        self.colours = epoch_colours(len(self.idx))
        self.alphas, self.centres, self.pmin, self.fields = [], [], [], []
        for i in self.idx:
            m = footprint(self.p[i], threshold, tuple(self.rc[i]))
            # sample the pressure itself onto the pixels and threshold there: a smooth edge, not 0.25-degree stairs
            pp = g.sample(np.where(ndi.binary_dilation(m, iterations=3), self.p[i], threshold + 5), self.lat, self.lon, order=3)
            self.alphas.append(np.clip((threshold - np.nan_to_num(pp, nan=threshold + 5)) / 0.15 + 0.5, 0, 1))
            self.fields.append(g.sample(self.p[i], self.lat, self.lon, order=3))
            r, c = self.rc[i]
            self.centres.append(g.xy(self.lon[c], self.lat[r]))
            self.pmin.append(float(self.p[i, r, c]))
        tx, ty = g.xy(self.lon[self.rc[:, 1]], self.lat[self.rc[:, 0]])
        self.track = (ndi.gaussian_filter1d(tx, 1.5), ndi.gaussian_filter1d(ty, 1.5))

    @staticmethod
    def when(t: str) -> str:
        return f"{int(t[11:13]):02d} UTC, {int(t[8:10])} Nov"

    def frame(self, path, upto: int | None = None, solo: int | None = None, scale: float = 1.0) -> Path:
        """``upto`` epochs on the map (default all); ``solo`` = that epoch filled with its isobars, others outlined."""
        if path is None:                     # not this process's frame (see common.Movie)
            return None
        n = len(self.idx)
        upto = n if upto is None else upto
        g = self.grid
        fig, ax = C.canvas(self.W, self.H, scale=scale)
        img = stack(self.base, self.alphas, self.colours, upto, solo=solo)
        C.show(ax, img, self.MX, self.MY)
        ox, oy = self.MX, self.MY
        for ln in self.borders:
            C.clip_to(ax.plot(ox + ln[:, 0], oy + ln[:, 1], color="#55648c", lw=0.6, alpha=0.7, zorder=2), ax, ox, oy, g.w, g.h)
        if solo is not None:  # the isobars of that hour: the footprint is one of them, filled
            f = self.fields[solo][::4, ::4]
            xs = ox + (np.arange(f.shape[1]) * 4 + 2)
            ys = oy + (np.arange(f.shape[0]) * 4 + 2)
            ax.contour(xs, ys, f, levels=np.arange(940, 1040, 4), colors="#aab4cc", linewidths=0.7, alpha=0.55, zorder=3)
        k1 = self.idx[upto - 1] if upto else self.idx[0]
        sel = slice(self.idx[0], k1 + 1)
        ax.plot(ox + self.track[0][sel], oy + self.track[1][sel], color=C.INK, lw=1.6, alpha=0.85, zorder=4)
        for k in range(upto):
            cx, cy = self.centres[k]
            ax.plot(ox + cx, oy + cy, "o", ms=6.5, mfc=self.colours[k], mec=C.INK, mew=1.1, zorder=5)
        for name, la, lo in PLACES:
            x, y = g.xy(lo, la)
            ax.plot(ox + x, oy + y, "o", ms=3.2, color=C.INK, zorder=6)
            t = C.text(ax, ox + x + 9, oy + y + 6, name, 19, zorder=6)
            C.halo(t)
        ax.plot([ox, ox + g.w, ox + g.w, ox, ox], [oy, oy, oy + g.h, oy + g.h, oy], color="#2a3550", lw=1.0, zorder=7)
        # scale bar (true near 45 and 58 N on this conformal conic map)
        bx, by, km = ox + 40, oy + g.h - 40, 300
        ax.plot([bx, bx + km * 1e3 / g.mpp], [by, by], color=C.INK, lw=2.2, zorder=8)
        C.text(ax, bx + km * 1e3 / g.mpp + 10, by + 6, f"{km} km", 17, zorder=8)
        # text column
        yy = C.title_block(ax, 50, 40, "Storm Ciarán's footprint",
                           f"Where sea-level pressure was below {self.threshold:g} hPa,\nevery 3 hours".split("\n")[0], size=42)
        C.text(ax, 50, yy + 30, "every 3 hours, 1–2 November 2023", 24, family=C.SERIF, style="italic", colour="#d9d4c7")
        C.text(ax, 50, yy + 66, "Newest on top. The crescents left behind show", 18, colour=C.MUTED)
        C.text(ax, 50, yy + 90, "where the deep core had been. The line is the", 18, colour=C.MUTED)
        C.text(ax, 50, yy + 114, "path of the lowest pressure.", 18, colour=C.MUTED)
        if solo is not None:
            C.text(ax, 50, yy + 138, "Thin lines: isobars every 4 hPa at this hour.", 18, colour=C.MUTED)
        ly = yy + 196
        C.text(ax, 50, ly, "TIME", 16, colour=C.MUTED, weight="bold")
        C.text(ax, 330, ly, "LOWEST PRESSURE", 16, colour=C.MUTED, weight="bold")
        for k in range(n):
            y = ly + 20 + k * 42
            on = k < upto
            a = 1.0 if on else 0.22
            ax.add_patch(plt.Rectangle((50, y), 46, 26, fc=self.colours[k], ec="none", alpha=a, zorder=5))
            hi = solo == k
            C.text(ax, 112, y + 21, self.when(self.time[self.idx[k]]), 22, colour=C.AMBER if hi else C.INK, alpha=a,
                   weight="bold" if hi else "normal")
            C.text(ax, 330, y + 21, f"{self.pmin[k]:.0f} hPa", 22, colour=C.AMBER if hi else C.INK, alpha=a,
                   weight="bold" if hi else "normal")
        C.credit_block(ax, 50, self.H - 92, ["Pressure: ERA5 hourly, Copernicus Climate Change Service (C3S) / ECMWF",
                                             "Land and borders: Natural Earth (public domain)"], size=16)
        return C.finish(fig, path)

    def poster(self, path, scale: float = 2.0) -> Path:
        return self.frame(path, scale=scale)

    def animate(self, out, frames_dir, fps: int = 30, solo_s: float = 1.1, build_s: float = 0.35, hold_s: float = 3.0) -> Path:
        mv = C.Movie(frames_dir, fps)
        n = len(self.idx)
        for k in range(n):                       # each epoch alone, with its isobars
            self.frame(mv.path(), upto=k + 1, solo=k)
            mv.hold(int(solo_s * fps) - 1)
        for k in range(1, n + 1):                # then the stack, oldest first
            self.frame(mv.path(), upto=k)
            mv.hold(int(build_s * fps) - 1)
        mv.hold(int(hold_s * fps))
        return mv.encode(out)
