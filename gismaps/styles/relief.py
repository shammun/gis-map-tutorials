"""gismaps.styles.relief: a data field drawn as lit, terraced relief ("the more there is, the higher the land").

The picture: the field is treated as a height. It is cut into filled bands (each band = one colour of the colour bar),
the bands are lit from the north-west like a relief model, the land floats above a dark sea on a soft shadow, and a
large counter (year, month) steps through time. High values are literally peaks, so the eye finds them at once.

This map: the water vapour held in the whole air column (total column water vapour) over South Asia, month by
month, 1991-2020 means (ERA5). The summer monsoon is the range that rises over the plains; the Himalaya and Tibet stay
a dry plateau.

API
---
``terrace(field, levels, mix)``                       -> a height array with steps at the band edges
``relief_rgb(field, levels, cmap, mask, ...)``         -> lit, banded RGB + alpha
``float_on(ground, rgb, alpha, ...)``                  -> lay it on the ground with a drop shadow
``VapourRelief(data)``                                 -> ``.frame(month)`` / ``.poster(path)`` / ``.animate(path)``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import common as C

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
CITIES = [("Dhaka", 23.81, 90.41, "r"), ("Delhi", 28.61, 77.21, "r"), ("Mumbai", 19.08, 72.88, "l"),
          ("Chennai", 13.08, 80.27, "r"), ("Karachi", 24.86, 67.01, "l"), ("Kathmandu", 27.72, 85.32, "t"),
          ("Lhasa", 29.65, 91.14, "r"), ("Colombo", 6.93, 79.86, "l"), ("Kolkata", 22.57, 88.36, "l")]


def terrace(field: np.ndarray, levels, mix: float = 0.45, soften: float = 1.5) -> np.ndarray:
    """Height for shading: the smooth field mixed with its band index, so each band edge becomes a small step."""
    lv = np.asarray(levels, float)
    step = np.clip(np.digitize(field, lv) - 1, 0, len(lv) - 2).astype(np.float32)
    frac = np.interp(field, lv, np.arange(len(lv)))                     # the field in units of bands
    return ndi.gaussian_filter((1 - mix) * frac + mix * step, soften)


def relief_rgb(field: np.ndarray, levels, cmap, mask: np.ndarray, zfactor: float = 3.4, ambient: float = 0.58,
               light: float = 0.62, line: float = 0.28):
    """Banded colours lit as relief. Returns (rgb, alpha). ``mask`` 0..1 says where the field is drawn (land)."""
    f = np.nan_to_num(field, nan=float(levels[0]))
    rgb = C.colourise(f, cmap, levels[0], levels[-1], levels=levels)
    z = terrace(f, levels)
    shade = C.hillshade(z, 315, 42, zfactor)
    rgb = np.clip(rgb * (ambient + light * shade[..., None]), 0, 1)
    # a fine dark line on every band edge (the contour lines of the model)
    band = np.digitize(f, np.asarray(levels, float))
    edge = (ndi.maximum_filter(band, 3) != ndi.minimum_filter(band, 3)).astype(np.float32)
    rgb = C.over(rgb, np.zeros(3, np.float32), ndi.gaussian_filter(edge, 0.7) * line)
    return rgb, mask


def float_on(ground: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, lift: float = 1.0) -> np.ndarray:
    """Ground + soft shadow + the lit layer (``lift`` scales the shadow offset)."""
    out = C.over(ground, np.zeros(3, np.float32), C.drop_shadow(alpha, 10 * lift, 14 * lift, 12, 0.6))
    out = C.over(out, rgb, alpha)
    # a light rim on the north-west shore, a dark one on the south-east: the land reads as a slab
    gy, gx = np.gradient(alpha)
    rim = np.clip(-(gx + gy) * 3, 0, 1) * (alpha > 0.05)
    return C.over(out, np.ones(3, np.float32), rim * 0.35)


class VapourRelief:
    """Total column water vapour over South Asia, by month (ERA5 1991-2020 means; or the NCEP fallback)."""

    W, H = 1920, 1080
    MX, MY, MW, MH = 760, 30, 1120, 1020
    LEVELS = np.arange(0, 80, 5)

    def __init__(self, data: dict):
        import cmocean
        self.cmap = cmocean.cm.haline
        self.lat, self.lon, self.tcwv = data["lat"], data["lon"], data["tcwv"]
        self.source = str(data["source"])
        g = self.grid = C.MapGrid.fit("+proj=laea +lat_0=21.5 +lon_0=81", 0, 0, 4300e3, self.MW, self.MH)
        self.land = g.land("50m", soft=1.0)
        self.ground = np.broadcast_to(C.rgb("#0c2733"), (g.h, g.w, 3)).copy()
        self.borders = g.lines(C.border_lines("50m"))
        self.px = [g.sample(self.tcwv[m], self.lat, self.lon, order=3) for m in range(12)]
        # mean over the land in view, weighted by nothing: the map is equal-area (Lambert azimuthal)
        w = self.land * np.isfinite(self.px[0])
        self.mean = np.array([float(np.nansum(p * w) / w.sum()) for p in self.px])

    def field(self, month: float) -> np.ndarray:
        """Field at a fractional month (0 = mid-January ... 11 = mid-December), linear between monthly means."""
        a = int(np.floor(month)) % 12
        t = month - np.floor(month)
        return (1 - t) * self.px[a] + t * self.px[(a + 1) % 12]

    def frame(self, month: float, path, scale: float = 1.0) -> Path:
        if path is None:                     # not this process's frame (see common.Movie)
            return None
        g = self.grid
        f = self.field(month)
        rgb, alpha = relief_rgb(f, self.LEVELS, self.cmap, self.land * np.isfinite(f))
        img = float_on(self.ground, rgb, alpha)
        fig, ax = C.canvas(self.W, self.H, bg="#0a1f29", scale=scale)
        C.show(ax, img, self.MX, self.MY)
        ox, oy = self.MX, self.MY
        for ln in self.borders:
            C.clip_to(ax.plot(ox + ln[:, 0], oy + ln[:, 1], color="#0a1f29", lw=0.9, alpha=0.55, zorder=2), ax, ox, oy, g.w, g.h)
        for name, la, lo, side in CITIES:
            x, y = g.xy(lo, la)
            ax.plot(ox + x, oy + y, "o", ms=4.2, mfc=C.INK, mec="#0a1f29", mew=0.8, zorder=6)
            dx, ha = (10, "left") if side in "rt" else (-10, "right")
            dy = -12 if side == "t" else 7
            C.halo(C.text(ax, ox + x + dx, oy + y + dy, name, 20, ha=ha, zorder=6), 2.8, "#0a1f29")
        for name, la, lo in (("Arabian Sea", 14.5, 64.5), ("Bay of Bengal", 13.5, 89.0), ("Tibetan Plateau", 33.3, 88.0)):
            x, y = g.xy(lo, la)
            C.halo(C.text(ax, ox + x, oy + y, name, 19, colour="#b9c7cf", style="italic", family=C.SERIF, ha="center", zorder=6),
                   2.4, "#0a1f29")
        # text column
        yy = C.title_block(ax, 56, 44, "Where the water vapour sits", "South Asia through the year", size=44)
        for i, line in enumerate(["Height and colour show the water vapour in the whole", "column of air above each place, as the depth of water",
                                  "it would make if it all fell as rain (mm)."]):
            C.text(ax, 56, yy + 40 + i * 25, line, 18.5, colour=C.MUTED)
        m = int(np.round(month)) % 12
        C.text(ax, 56, 430, MONTHS[m], 88, colour=C.AMBER, family=C.SERIF, weight="bold")
        mean = float(np.interp(month % 12, np.arange(13), np.r_[self.mean, self.mean[0]]))
        C.text(ax, 56, 486, f"{mean:.0f} mm", 46, family=C.SERIF)
        C.text(ax, 56, 516, "MEAN OVER THE LAND IN VIEW", 15, colour=C.MUTED, weight="bold")
        # the year as twelve bars, this month lit
        bx, by, bw, bh = 56, 560, 520, 150
        top = self.mean.max() * 1.1
        for k in range(12):
            hgt = bh * self.mean[k] / top
            lit = k == m
            ax.add_patch(__import__("matplotlib.pyplot").pyplot.Rectangle((bx + k * bw / 12 + 4, by + bh - hgt), bw / 12 - 8, hgt,
                                                                        fc=C.AMBER if lit else "#3d5560", ec="none", zorder=5))
            C.text(ax, bx + (k + 0.5) * bw / 12, by + bh + 24, MONTHS[k][0], 17, colour=C.INK if lit else C.MUTED, ha="center")
        C.text(ax, bx + bw + 14, by + bh - bh * self.mean.max() / top + 6, f"{self.mean.max():.0f} mm", 16, colour=C.MUTED)
        C.text(ax, bx + bw + 14, by + bh - bh * self.mean.min() / top + 6, f"{self.mean.min():.0f} mm", 16, colour=C.MUTED)
        C.colourbar(ax, 56, 830, 520, 22, _banded(self.cmap, len(self.LEVELS) - 1), 0, 75, [0, 10, 20, 30, 40, 50, 60, 70],
                    "TOTAL COLUMN WATER VAPOUR (mm = kg per square metre), bands of 5 mm", size=16)
        src = "ERA5 monthly means 1991–2020, Copernicus C3S / ECMWF" if "ERA5" in self.source \
            else self.source
        C.credit_block(ax, 56, self.H - 92, [f"Water vapour: {src}", "Land and borders: Natural Earth (public domain)"], size=16)
        return C.finish(fig, path)

    def poster(self, path, month: int = 6, scale: float = 2.0) -> Path:
        return self.frame(float(month), path, scale)

    def animate(self, out, frames_dir, fps: int = 30, per_month: int = 24, hold: int = 12, loops: float = 1.0) -> Path:
        mv = C.Movie(frames_dir, fps)
        self.frame(0.0, mv.path())
        mv.hold(fps)
        for k in range(int(12 * loops)):
            for j in range(1, per_month + 1):
                self.frame(k + C.ease(j / per_month), mv.path())
            mv.hold(hold)
        mv.hold(fps)
        return mv.encode(out)


def _banded(cmap, n: int):
    """A colour map cut into n flat bands (so the colour bar shows exactly the colours on the map)."""
    from matplotlib.colors import ListedColormap
    return ListedColormap(cmap((np.arange(n) + 0.5) / n))
