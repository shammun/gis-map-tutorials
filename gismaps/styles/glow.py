"""gismaps.styles.glow: glow-dot time-lapse with a mirrored season chart ("events that follow the calendar").

The picture: a dark map; every event is a small additive dot, so places with many events glow; a clock runs through
the calendar year and a mirrored area chart under the map (one group up, the other down) carries the playhead.

This map: 35 years of tropical-cyclone positions (IBTrACS), drawn on their calendar day. The Northern Hemisphere
season (June-November) and the Southern one (November-April) light up in turn.

API
---
``season_counts(doy, group, years)``  -> mean events per calendar day for the two groups (for the chart)
``mirror_chart(ax, ...)``             -> the mirrored area chart with a playhead
``glow_rgb(base, buf, cmap)``         -> lay a tone-mapped additive buffer on a base map
``StormSeason(data)``                 -> ``.frame(day)`` / ``.poster(path)`` / ``.animate(path)``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi

from . import common as C

#: ember ramp for additive glow: dark red -> orange -> amber -> near white (monotonic in lightness)
EMBER = LinearSegmentedColormap.from_list("ember", ["#3a0f05", "#a8380c", "#e8833a", "#ffc766", "#fff3d6"])
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
MONTH_START = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])


def season_counts(doy: np.ndarray, group: np.ndarray, n_years: int, per_day: float = 4.0, smooth: int = 7):
    """Mean number of events in progress on each calendar day, for group True and group False.

    ``doy`` is the 0-based day of year of each record, ``per_day`` the records one event makes per day (4 for
    6-hourly fixes), ``smooth`` a centred running mean in days (wraps round the year).
    """
    out = []
    for g in (True, False):
        n = np.bincount(np.clip(doy[group == g], 0, 364), minlength=365)[:365] / per_day / n_years
        out.append(ndi.uniform_filter1d(n, smooth, mode="wrap"))
    return out[0], out[1]


def mirror_chart(ax, x, y, w, h, up, down, day, up_colour=C.AMBER, down_colour="#e8603a", up_label="", down_label="",
                 unit="", size=17, playhead=True):
    """Mirrored area chart in frame pixels: ``up`` above the axis, ``down`` below, bright up to ``day`` (playhead)."""
    n = len(up)
    xs = x + w * (np.arange(n) + 0.5) / n
    top = max(up.max(), down.max())
    s = (h / 2) / top
    y0 = y + h / 2
    for arr, sign, col in ((up, -1, up_colour), (down, 1, down_colour)):
        ax.fill_between(xs, y0, y0 + sign * arr * s, color="#3a4257", lw=0, zorder=5)                 # the whole year, dim
        k = int(np.clip(np.floor(day), 0, n - 1)) + 1
        ax.fill_between(xs[:k], y0, y0 + sign * arr[:k] * s, color=col, lw=0, zorder=6)               # up to today, lit
    ax.plot([x, x + w], [y0, y0], color=C.MUTED, lw=0.8, zorder=7)
    px = x + w * (day + 0.5) / n
    if playhead:
        ax.plot([px, px], [y - 6, y + h + 6], color=C.INK, lw=1.6, zorder=8)
    for m in range(12):
        C.text(ax, x + w * (MONTH_START[m] + 15) / 365, y + h + 30, MONTHS[m][0], size, colour=C.MUTED, ha="center")
    C.text(ax, x, y - 14, up_label, size, colour=up_colour)
    C.text(ax, x, y + h + 58, down_label, size, colour=down_colour)
    # one scale mark so the chart can be read as numbers
    tick = float(np.floor(top))
    if tick >= 1:
        for sign in (-1, 1):
            ax.plot([x + w + 6, x + w + 14], [y0 + sign * tick * s] * 2, color=C.MUTED, lw=0.8, zorder=7)
        C.text(ax, x + w + 20, y0 - tick * s + 6, f"{tick:g} {unit}", size * 0.9, colour=C.MUTED)


def glow_rgb(base: np.ndarray, buf: np.ndarray, cmap=EMBER, gain: float = 1.0, alpha_gamma: float = 0.55) -> np.ndarray:
    """Base map + additive buffer -> RGB. Brightness = tone-mapped count; opacity rises faster than colour."""
    t = C.tone(buf, gain)
    return C.over(base, cmap(t)[..., :3].astype(np.float32), t**alpha_gamma)


class StormSeason:
    """Tropical cyclones on their calendar day (IBTrACS fixes, all basins)."""

    W, H = 1920, 1080
    MAP_X, MAP_Y, MAP_W = 40, 150, 1840

    def __init__(self, data: dict, tau_days: float = 5.0, title="The storm belts follow the sun",
                 credits=("Storm positions: IBTrACS v04r01, NOAA NCEI (public domain)", "Land: Natural Earth (public domain)")):
        self.lat, self.lon = data["lat"], data["lon"]
        self.wind, self.doy, self.year = data["wind_kt"].astype(float), np.clip(data["doy"], 0, 364), data["year"]
        self.y0, self.y1 = int(self.year.min()), int(self.year.max())
        self.tau, self.title, self.credits = tau_days, title, list(credits)
        # Equal Earth centred on 165 W: the cut falls on Africa (15 E), where there are no tropical cyclones
        self.grid = C.MapGrid.from_box("+proj=eqearth +lon_0=-165", 15.001, 374.999, -47, 53, self.MAP_W)
        g = self.grid
        land = g.land("110m", soft=0.7)
        base = np.broadcast_to(C.rgb("#080b14"), (g.h, g.w, 3)).copy()
        base = C.over(base, C.rgb("#161c2c"), land)
        base = C.over(base, C.rgb("#3a4560"), C.coast_alpha(land, 1.0) * 0.8)
        ex, ey = g.xy(np.linspace(15.01, 374.99, 400), np.zeros(400))
        self.equator = (ex, ey)
        self.base = base
        x, y = g.xy(self.lon, self.lat)
        self.xy = np.c_[x, y]
        self.weight = np.clip(self.wind / 64.0, 0.5, 2.2)          # hurricane-force fixes glow more
        self.north, self.south = season_counts(self.doy, self.lat > 0, self.y1 - self.y0 + 1)

    def buffer(self, day: float | None) -> np.ndarray:
        """Additive buffer for calendar ``day`` (0..365): today's fixes at full weight, earlier days fading with
        e-folding time ``tau``. ``None`` = the whole year at once."""
        if day is None:
            return C.glow(self.xy, self.weight * 0.16, (self.grid.h, self.grid.w), core=1.0, halo=4.0, halo_gain=0.35)
        age = (day - self.doy) % 365                                  # days since the fix, round the calendar
        sel = age < 5 * self.tau
        return C.glow(self.xy[sel], self.weight[sel] * np.exp(-age[sel] / self.tau), (self.grid.h, self.grid.w))

    def frame(self, day: float | None, path, scale: float = 1.0) -> Path:
        if path is None:                     # not this process's frame (see common.Movie)
            return None
        g = self.grid
        fig, ax = C.canvas(self.W, self.H, bg="#05070d", scale=scale)
        img = glow_rgb(self.base, self.buffer(day), gain=1.7 if day is not None else 1.0)
        C.show(ax, img, self.MAP_X, self.MAP_Y)
        ax.plot(self.MAP_X + self.equator[0], self.MAP_Y + self.equator[1], color="#8a93a8", lw=0.7, ls=(0, (2, 4)), zorder=3)
        C.text(ax, self.MAP_X + 12, self.MAP_Y + self.equator[1][0] - 8, "Equator", 17, colour="#8a93a8")
        C.title_block(ax, 60, 22, self.title,
                      f"Every tropical cyclone of {self.y0}–{self.y1}, drawn on its day of the year", size=50)
        C.text(ax, self.W - 60, 60, "Each dot is one 6-hourly position of a storm with winds of 34 knots or more.", 18,
               colour=C.MUTED, ha="right")
        C.text(ax, self.W - 60, 88, "Brighter = more storms passed there on these days; stronger storms glow more.", 18,
               colour=C.MUTED, ha="right")
        cy = self.MAP_Y + g.h + 48
        d = 364.0 if day is None else day
        label = "All year" if day is None else MONTHS[int(np.searchsorted(MONTH_START, int(d), side="right") - 1)]
        C.text(ax, 60, cy + 92, label, 62, family=C.SERIF)
        mirror_chart(ax, 500, cy + 16, 760, 120, self.north, self.south, d, up_label="Northern Hemisphere",
                     down_label="Southern Hemisphere", unit="storms at once (mean)", playhead=day is not None)
        C.credit_block(ax, 1400, cy + 62, self.credits, size=16)
        return C.finish(fig, path)

    def poster(self, path, scale: float = 2.0) -> Path:
        return self.frame(None, path, scale)

    def animate(self, out, frames_dir, fps: int = 30, days_per_frame: float = 0.5, hold_s: float = 2.0) -> Path:
        mv = C.Movie(frames_dir, fps)
        for d in np.arange(0, 365, days_per_frame):
            self.frame(float(d), mv.path())
        self.frame(None, mv.path())
        mv.hold(int(hold_s * fps))
        return mv.encode(out)

    def export_layers(self, folder, quality: int = 90) -> Path:
        """Map-only frames for a video editor or a web animation: ``map_000.jpg`` ... ``map_364.jpg``
        (one per calendar day), ``map_all.jpg`` and ``manifest.json`` (size, chart series, credits)."""
        import json
        from PIL import Image
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        def put(buf, gain, name):
            img = glow_rgb(self.base, buf, gain=gain)
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(folder / name, quality=quality)

        for d in range(365):
            put(self.buffer(float(d)), 1.7, f"map_{d:03d}.jpg")
        put(self.buffer(None), 1.0, "map_all.jpg")
        eq_y = float(self.equator[1][0])
        (folder / "manifest.json").write_text(json.dumps({
            "w": self.grid.w, "h": self.grid.h, "equator_y": round(eq_y, 1), "years": [self.y0, self.y1],
            "north": [round(float(v), 3) for v in self.north], "south": [round(float(v), 3) for v in self.south],
            "credits": self.credits}), encoding="utf8")
        return folder
