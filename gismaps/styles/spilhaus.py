"""gismaps.styles.spilhaus: the world ocean as one uncut body of water (Spilhaus projection) with a linked chart.

The picture: most world maps cut the ocean at the edges and keep the land whole. The Spilhaus projection (Athelstan
Spilhaus, 1942; in PROJ since 9.6 as ``+proj=spilhaus``) does the opposite: its two "poles" are on land (in China and
Argentina), the cuts run through continents, and every ocean is one connected sheet. An ocean field then reads as one
circulation. A small chart of a few numbers taken from the same data carries the playhead.

This map: sea-surface temperature and sea ice through 2025 (NOAA OISST v2.1), with the sea-ice extent of each
hemisphere (NSIDC Sea Ice Index).

API
---
``spilhaus_grid(w)``                    -> a ``MapGrid`` holding the whole Spilhaus square
``OceanYear(data)``                     -> ``.frame(day)`` / ``.poster(path)`` / ``.animate(path)``
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi

from . import common as C

#: half-side of the Spilhaus square in projected metres (the forward projection of the far corner, 113.06 W 49.56 N)
HALF = 11_740_000.0
ICE = LinearSegmentedColormap.from_list("ice", ["#6f9fc4", "#cfe6f5", "#ffffff"])
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
MONTH_START = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])


def spilhaus_grid(w: int = 1000) -> C.MapGrid:
    """Pixel grid over the whole Spilhaus square (needs PROJ >= 9.6: ``pip install -U pyproj``)."""
    return C.MapGrid("+proj=spilhaus", -HALF, HALF, -HALF, HALF, w)


class OceanYear:
    """Sea-surface temperature and sea ice through one year on the Spilhaus map."""

    W, H = 1920, 1080
    MX, MY, MW = 40, 40, 1000

    def __init__(self, data: dict):
        import cmocean
        self.cmap = cmocean.cm.thermal
        self.lat, self.lon = data["lat"], data["lon"]
        self.dates = [dt.date.fromisoformat(str(d)) for d in data["date"]]
        self.doy = np.array([d.timetuple().tm_yday - 1 for d in self.dates], float)
        self.year = self.dates[0].year
        sst = data["sst_c100"].astype(np.float32) / 100.0
        sst[data["sst_c100"] == -9999] = np.nan
        ice = data["ice_pct"].astype(np.float32) / 100.0
        # OISST's ice field has stray "ice" in water far too warm for it (around 52 S in summer): keep ice only
        # where the sea surface is colder than 1.5 degrees C
        ice = np.where(np.nan_to_num(sst, nan=9) <= 1.5, ice, 0)
        g = self.grid = spilhaus_grid(self.MW)
        self.sst = np.stack([g.sample(s, self.lat, self.lon, wrap=True) for s in sst])
        self.ice = np.stack([np.nan_to_num(g.sample(i, self.lat, self.lon, wrap=True)) for i in ice])
        self.land = g.land("110m", res=0.1, soft=0.8)
        self.ext_n, self.ext_s = data["extent_n"], data["extent_s"]
        self.ext_src = str(data["extent_source"])

    def fields(self, day: float):
        """SST and ice at calendar ``day`` (0..364): linear blend of the two nearest snapshots."""
        day = float(np.clip(day, self.doy[0], self.doy[-1]))
        k = int(np.clip(np.searchsorted(self.doy, day, side="right") - 1, 0, len(self.doy) - 2))
        t = (day - self.doy[k]) / (self.doy[k + 1] - self.doy[k])
        return (1 - t) * self.sst[k] + t * self.sst[k + 1], (1 - t) * self.ice[k] + t * self.ice[k + 1]

    def map_rgb(self, day: float) -> np.ndarray:
        sst, ice = self.fields(day)
        rgb = C.colourise(sst, self.cmap, -2, 30)
        rgb = C.over(rgb, ICE(np.clip((ice - 0.15) / 0.85, 0, 1))[..., :3].astype(np.float32), np.clip((ice - 0.15) / 0.25, 0, 1))
        land = np.maximum(self.land, ~np.isfinite(sst) * 1.0)
        rgb = C.over(rgb, C.rgb("#171d2c"), ndi.gaussian_filter(land, 0.6))
        return C.over(rgb, C.rgb("#59647f"), C.coast_alpha(self.land, 1.0) * 0.7)

    def frame(self, day: float, path, scale: float = 1.0) -> Path:
        if path is None:                     # not this process's frame (see common.Movie)
            return None
        g = self.grid
        fig, ax = C.canvas(self.W, self.H, scale=scale)
        C.show(ax, self.map_rgb(day), self.MX, self.MY)
        ox, oy = self.MX, self.MY
        ax.plot([ox, ox + g.w, ox + g.w, ox, ox], [oy, oy, oy + g.h, oy + g.h, oy], color="#2a3550", lw=1.0, zorder=4)
        for name, la, lo in (("PACIFIC", -5, -140), ("ATLANTIC", 28, -45), ("INDIAN", -22, 78), ("SOUTHERN OCEAN", -52, -125),
                             ("ARCTIC", 82, -150), ("Antarctica", -80, 60), ("Africa", 8, 22), ("Australia", -25, 134),
                             ("North America", 44, -100), ("South America", -12, -58), ("Asia", 58, 95)):
            x, y = g.xy(lo, la)
            if not (90 < x < g.w - 90 and 20 < y < g.h - 20):      # a label that would leave the map is dropped
                continue
            ocean = name.isupper()
            C.halo(C.text(ax, ox + x, oy + y, name, 17 if ocean else 15, colour=C.INK if ocean else "#8a93a8", ha="center",
                          weight="bold" if ocean else "normal", style="normal" if ocean else "italic", zorder=5), 2.6, "#10141f")
        tx = 1090
        yy = C.title_block(ax, tx, 44, "One ocean", f"Sea-surface temperature and sea ice through {self.year}", size=56)
        for i, line in enumerate(["This map is cut through the continents, not the sea,",
                                  "so all the oceans show as one connected body of water",
                                  "(Spilhaus projection). Shapes and areas are distorted,",
                                  "most of all near the corners."]):
            C.text(ax, tx, yy + 44 + i * 27, line, 19, colour="#c9cfdd")
        d = int(np.clip(round(day), 0, 364))
        date = dt.date(self.year, 1, 1) + dt.timedelta(days=d)
        C.text(ax, tx, 392, f"{date.day} {MONTHS[date.month - 1]}", 62, colour=C.AMBER, family=C.SERIF, weight="bold")
        C.colourbar(ax, tx, 462, 440, 20, self.cmap, -2, 30, [0, 10, 20, 30], "SEA-SURFACE TEMPERATURE (°C)", size=16)
        C.colourbar(ax, tx + 500, 462, 220, 20, ICE, 15, 100, [15, 50, 100], "SEA ICE (% of the sea covered)", size=16)
        # linked chart: sea-ice extent of each hemisphere
        cx, cy, cw, ch = tx + 50, 610, 670, 230
        top = 20.0
        C.text(ax, tx, cy - 34, "SEA-ICE EXTENT (million km²)", 16, colour=C.MUTED, weight="bold")
        for v in (0, 5, 10, 15, 20):
            y = cy + ch - ch * v / top
            ax.plot([cx, cx + cw], [y, y], color="#2a3550", lw=0.8, zorder=4)
            C.text(ax, cx - 10, y + 6, f"{v}", 16, colour=C.MUTED, ha="right")
        for m in range(12):
            C.text(ax, cx + cw * (MONTH_START[m] + 15) / 365, cy + ch + 26, MONTHS[m][0], 16, colour=C.MUTED, ha="center")
        days = np.arange(365)
        for ser, col, name, dy in ((self.ext_n, "#f2b33d", "Arctic", -14), (self.ext_s, "#7fc4ff", "Antarctic", -14)):
            xs, ys = cx + cw * (days + 0.5) / 365, cy + ch - ch * ser / top
            ax.plot(xs, ys, color=col, lw=1.2, alpha=0.35, zorder=5)
            ax.plot(xs[:d + 1], ys[:d + 1], color=col, lw=2.8, zorder=6)
            ax.plot(xs[d], ys[d], "o", ms=7.5, color=col, zorder=7)
            lx = float(np.clip(xs[d] + 12, cx, cx + cw - 150))
            C.halo(C.text(ax, lx, ys[d] + dy, f"{name} {ser[d]:.1f}", 19, colour=col, weight="bold", zorder=8), 2.6)
        px = cx + cw * (d + 0.5) / 365
        ax.plot([px, px], [cy, cy + ch], color=C.INK, lw=1.0, alpha=0.5, zorder=5)
        C.credit_block(ax, tx, self.H - 132, ["Temperature and ice on the map: NOAA OISST v2.1, NOAA NCEI (public domain);",
                                              "  snapshots on the 1st and 16th of each month, blended in between",
                                              "Extent chart: Sea Ice Index v4, NSIDC / NOAA    ·    Land: Natural Earth"], size=16)
        return C.finish(fig, path)

    def poster(self, path, day: float = 258.0, scale: float = 2.0) -> Path:
        return self.frame(day, path, scale)

    def animate(self, out, frames_dir, fps: int = 30, days_per_frame: float = 1.5) -> Path:
        mv = C.Movie(frames_dir, fps)
        self.frame(0.0, mv.path())
        mv.hold(fps // 2)
        for d in np.arange(days_per_frame, self.doy[-1] + 1e-6, days_per_frame):
            self.frame(float(d), mv.path())
        mv.hold(fps * 2)
        return mv.encode(out)
