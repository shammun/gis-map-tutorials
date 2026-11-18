"""gismaps.styles.satellite: epoch-stacked footprints from satellite imagery (Sentinel-2 water masks).

The same picture as ``gismaps.styles.epochs`` (one colour per epoch, oldest at the bottom, newest on top), but the
footprints are **measured from satellite images**: for each year a water mask of a river in the dry season, from the
Modified Normalised Difference Water Index, MNDWI = (green - SWIR) / (green + SWIR), of Sentinel-2 L2A scenes.

Two ways to get the masks (both in ``tutorials/gis_styles/prepare_data.py``):
  ``sentinel``      no account: public STAC search (Element 84 Earth Search) + windowed reads of the cloud-optimised
                    GeoTIFFs with rasterio; per year the median of the 4 clearest scenes;
  ``sentinel_gee``  Google Earth Engine: per year the median of ALL clear scenes, computed on Google's side.

This map: the Jamuna (Brahmaputra) in Bangladesh, 2019-2026, a braided river that moves its channels every year.

API
---
``unpack(data)``          -> boolean masks (year, y, x) from the packed file
``RiverEpochs(data)``     -> ``.frame(...)`` / ``.poster(path)`` / ``.animate(path)``
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from . import common as C
from .epochs import epoch_colours, stack


def unpack(data: dict) -> np.ndarray:
    """Boolean river masks (year, y, x) from the bit-packed array of the reduced file."""
    shape = tuple(int(v) for v in data["shape"])
    return np.unpackbits(data["river"], count=int(np.prod(shape))).reshape(shape).astype(bool)


class RiverEpochs:
    """A river's water extent, one dry season per year, as stacked footprints beside the newest true-colour image."""

    W, H = 1920, 1080
    MH = 980                           # height of both pictures in the frame

    def __init__(self, data: dict, title: str = "The Jamuna never stays put",
                 place: str = "Jamuna (Brahmaputra) river, Bangladesh, 24.5–25.0°N 89.6–89.9°E",
                 method: str = "median of the 4 clearest scenes of 15 January to 5 March each year",
                 min_km2: float = 5.0):
        self.masks = unpack(data)
        self.years = [int(y) for y in data["year"]]
        self.res = float(data["res_m"])
        self.rgb = data["rgb_latest"].astype(np.float32) / 255.0
        self.rgb_date = str(data["rgb_date"]) if "rgb_date" in data else ""
        self.source = str(data["source"])
        self.title, self.place, self.method, self.min_km2 = title, place, method, min_km2
        if self.rgb_date:
            import datetime as _dt
            _d = _dt.date.fromisoformat(self.rgb_date)
            self.rgb_date = f"{_d.day} {_d:%B %Y}"
        n, h, w = self.masks.shape
        self.scale = self.MH / h
        self.mw = int(round(w * self.scale))
        self.area = self.masks.reshape(n, -1).sum(1) * self.res**2 / 1e6            # km2
        self.colours = epoch_colours(n)
        self.alphas = [ndi.gaussian_filter(m.astype(np.float32), 0.7) for m in self.masks]
        grey = (self.rgb @ np.array([0.30, 0.59, 0.11], np.float32))[..., None]
        self.base = np.clip(0.07 + 0.33 * (0.75 * grey + 0.25 * self.rgb), 0, 1) + C.rgb("#02040a")   # dim, nearly grey
        # how many of the years had water in each pixel (for the summary line)
        self.ever = self.masks.any(0).sum() * self.res**2 / 1e6
        self.always = self.masks.all(0).sum() * self.res**2 / 1e6

    def frame(self, path, upto: int | None = None, solo: int | None = None, scale: float = 1.0) -> Path:
        if path is None:                     # not this process's frame (see common.Movie)
            return None
        n = len(self.years)
        upto = n if upto is None else upto
        fig, ax = C.canvas(self.W, self.H, scale=scale)
        x1, x2, y0 = 560, 560 + self.mw + 22, 50
        C.show(ax, stack(self.base, self.alphas, self.colours, upto, solo=solo, opacity=0.96), x1, y0, self.mw, self.MH)
        C.show(ax, self.rgb, x2, y0, self.mw, self.MH)
        for x in (x1, x2):
            ax.plot([x, x + self.mw, x + self.mw, x, x], [y0, y0, y0 + self.MH, y0 + self.MH, y0], color="#2a3550", lw=1.0, zorder=4)
        C.halo(C.text(ax, x1 + 14, y0 + 30, "Water, year by year", 19, weight="bold", zorder=5), 3)
        C.halo(C.text(ax, x2 + 14, y0 + 30, f"True colour, {self.rgb_date}", 19, weight="bold", zorder=5), 3)
        km = 10
        bx, by = x1 + 20, y0 + self.MH - 24
        ax.plot([bx, bx + km * 1000 / self.res * self.scale], [by, by], color=C.INK, lw=2.4, zorder=6)
        C.halo(C.text(ax, bx + km * 1000 / self.res * self.scale + 10, by + 6, f"{km} km", 17, zorder=6), 3)
        ax.annotate("", xy=(x2 + self.mw - 34, y0 + self.MH - 70), xytext=(x2 + self.mw - 34, y0 + self.MH - 24),
                    arrowprops=dict(arrowstyle="-|>", color=C.INK, lw=2.0), zorder=6)
        C.halo(C.text(ax, x2 + self.mw - 34, y0 + self.MH - 78, "N", 19, weight="bold", ha="center", zorder=6), 3)
        # text column
        yy = C.title_block(ax, 44, 40, self.title, None, size=33)
        for i, line in enumerate(["Where the river's water was in each dry", "season, 2019–2026, from Sentinel-2.",
                                  "Newest on top: colours that still show are", "places the river has since left."]):
            C.text(ax, 44, yy + 40 + i * 26, line, 19.5, colour="#c9cfdd" if i < 2 else C.MUTED)
        ly = yy + 190
        C.text(ax, 44, ly, "DRY SEASON", 15, colour=C.MUTED, weight="bold")
        C.text(ax, 250, ly, "RIVER WATER", 15, colour=C.MUTED, weight="bold")
        top = self.area.max()
        for k in range(n):
            y = ly + 18 + k * 44
            a = 1.0 if k < upto else 0.22
            hi = solo == k
            ax.add_patch(plt.Rectangle((44, y), 44, 26, fc=self.colours[k], ec="none", alpha=a, zorder=5))
            C.text(ax, 104, y + 21, str(self.years[k]), 23, colour=C.AMBER if hi else C.INK, alpha=a, weight="bold" if hi else "normal")
            ax.add_patch(plt.Rectangle((250, y + 3), 150 * self.area[k] / top, 20, fc=self.colours[k], ec="none", alpha=a * 0.9, zorder=5))
            C.text(ax, 250 + 150 * self.area[k] / top + 10, y + 21, f"{self.area[k]:.0f} km²", 21,
                   colour=C.AMBER if hi else C.INK, alpha=a, weight="bold" if hi else "normal")
        y = ly + 18 + n * 44 + 26
        C.text(ax, 44, y, f"Water in at least one year: {self.ever:.0f} km²", 19, colour="#c9cfdd")
        C.text(ax, 44, y + 27, f"Water in every year: {self.always:.0f} km²", 19, colour="#c9cfdd")
        for i, line in enumerate([self.place, f"Water = MNDWI > 0.1, connected pieces of {self.min_km2:g} km2 or more;",
                                  self.method]):
            C.text(ax, 44, y + 78 + i * 22, line, 14.5, colour=C.MUTED)
        credit = ["Contains modified Copernicus Sentinel data 2019–2026"]
        credit.append("processed with Google Earth Engine" if "Earth Engine" in self.source
                      else "Sentinel-2 L2A from Element 84 Earth Search (AWS open data)")
        C.credit_block(ax, 44, self.H - 92, credit, size=15.5)
        return C.finish(fig, path)

    def poster(self, path, scale: float = 2.0) -> Path:
        return self.frame(path, scale=scale)

    def animate(self, out, frames_dir, fps: int = 30, solo_s: float = 1.2, build_s: float = 0.4, hold_s: float = 3.0) -> Path:
        mv = C.Movie(frames_dir, fps)
        n = len(self.years)
        self.frame(mv.path())                    # open on the finished picture
        mv.hold(int(1.5 * fps) - 1)
        for k in range(n):                       # each year alone
            self.frame(mv.path(), upto=k + 1, solo=k)
            mv.hold(int(solo_s * fps) - 1)
        for k in range(1, n + 1):                # then the stack, oldest first
            self.frame(mv.path(), upto=k)
            mv.hold(int(build_s * fps) - 1)
        mv.hold(int(hold_s * fps))
        return mv.encode(out)
