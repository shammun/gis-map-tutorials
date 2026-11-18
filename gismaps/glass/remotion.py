"""gismaps.glass.remotion: the pictures and numbers that the Remotion version of the glass-pane stack loads.

The Remotion project in ``remotion/glass_stack`` draws the globe and the panes with three.js. It loads, with
``staticFile()``, 42 files from its public folder; this module writes them::

    python -m gismaps.glass.remotion remotion/glass_stack/public          # from the shipped data
    python -m gismaps.glass.remotion remotion/glass_stack/public live     # from a live read (about 240 MB, ten minutes)

``relief.png``         the globe's skin: hill-shaded relief, latitude x longitude (row 0 = 90 N, column 0 = 180 W)
``<field>_<k>.png``    one pane picture per field (t2m, mslp, pwat, tcc, prate) and lead time (k = 0 ... 7): the
                       orthographic view of the hemisphere centred on 45 N 30 W, RGBA, clear outside the disc
``manifest.json``      the run, the lead times, the centre, the colour ramps and value ranges, and the position and
                       depth of the North Atlantic low at every lead time

Each pane is the orthographic projection along the pane normal (``gismaps.glass.ortho``), so a pane placed on that
axis lines up with the globe behind it from any camera angle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import gfs, ortho

N = 900                                                      # pane picture size in pixels
FIELDS = ["t2m", "mslp", "pwat", "tcc", "prate"]
#: one ramp per layer: temperature two-sided about 0 degC, the others one hue
RAMPS = {
    "t2m": ["#2b4f9e", "#5f8fd0", "#a9c6e8", "#e8ecf4", "#f1b38f", "#d9663f", "#9e2a2b"],
    "pwat": ["#173d47", "#1f6a6e", "#2aa198", "#6fd3c1", "#c9f4ea"],
    "tcc": ["#8b95a8", "#c3cad6", "#f4f6fa"],
    "prate": ["#9ecbff", "#4f8fe8", "#2f56c8", "#4b2fa0"],
}
RANGES = {"t2m": (-30.0, 30.0), "pwat": (0.0, 60.0), "tcc": (0.0, 100.0), "prate": (0.1, 10.0)}
UNITS = {"t2m": "°C", "mslp": "hPa", "pwat": "kg m⁻²", "tcc": "%", "prate": "mm h⁻¹"}


def _save(rgba: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray((np.clip(rgba, 0, 1) * 255).round().astype(np.uint8)).save(path)


def coast_from_mask(land: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """A coast outline, 2 px wide, where the land mask changes."""
    edge = np.zeros_like(land, bool)
    edge[:-1] |= land[:-1] != land[1:]
    edge[:, :-1] |= land[:, :-1] != land[:, 1:]
    edge[1:] |= edge[:-1].copy()
    edge[:, 1:] |= edge[:, :-1].copy()
    return edge & inside


def relief_picture(elev: dict) -> np.ndarray:
    """The globe's skin (RGB 0..1): light clay land and slate sea, lit from the north-west (heights x 20), with the
    coast drawn where the height crosses sea level."""
    z, lat = elev["z"].astype(np.float64), elev["lat"]
    dy = 6.371e6 * np.radians(abs(lat[1] - lat[0]))                    # metres per cell, north-south
    dx = dy * np.cos(np.radians(lat))[:, None]                         # ... and east-west, shrinking towards the poles
    gy, gx = np.gradient(z * 20.0)
    gy, gx = -gy / dy, gx / dx                                         # rows run north -> south
    az, alt = np.radians(315), np.radians(45)
    slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(gy, -gx)
    shade = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - np.pi / 2 - aspect), 0, 1)
    land = z > 0
    clay, sea, deep = np.array([0.93, 0.92, 0.89]), np.array([0.44, 0.49, 0.57]), np.array([0.30, 0.35, 0.44])
    t = np.clip(-z / 6000, 0, 1)[..., None]
    rgb = np.where(land[..., None], clay * (0.45 + 0.55 * shade)[..., None], (sea * (1 - t) + deep * t) * (0.85 + 0.15 * shade[..., None]))
    rgb[coast_from_mask(land, np.ones_like(land))] = [0.22, 0.25, 0.32]
    return rgb


def isobar_picture(p_hpa: np.ndarray, inside: np.ndarray, step: float = 4.0) -> np.ndarray:
    """Isobars every ``step`` hPa on a clear N x N picture, every 20 hPa heavier, over a faint navy tint in the disc."""
    n = p_hpa.shape[0]
    fig = plt.figure(figsize=(n / 100, n / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    tint = np.zeros((n, n, 4))
    tint[inside] = [0.09, 0.12, 0.2, 0.55]
    ax.imshow(tint, extent=(0, n, 0, n), origin="upper", interpolation="nearest")
    lv = np.arange(940, 1060 + step, step)
    z = np.where(inside, p_hpa, np.nan)
    ax.contour(np.arange(n), np.arange(n)[::-1], z, levels=[x for x in lv if x % 20], colors="#f4ecd2", linewidths=2.2, alpha=0.95)
    ax.contour(np.arange(n), np.arange(n)[::-1], z, levels=[x for x in lv if x % 20 == 0], colors="#f4ecd2", linewidths=4.2)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).astype(float) / 255.0
    plt.close(fig)
    return img


def pane_picture(field: str, x: np.ndarray, inside: np.ndarray, coast: np.ndarray) -> np.ndarray:
    """One hemisphere pane (RGBA 0..1) from the field already resampled onto the disc."""
    if field == "mslp":
        img = isobar_picture(x, inside)
    else:
        lo, hi = RANGES[field]
        if field == "prate":                                           # logarithmic; clear below 0.1 mm per hour
            t = np.clip((np.log(np.clip(x, 1e-6, None)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
            a = np.where(x >= lo, 0.55 + 0.45 * t, 0.0)
        else:
            t = np.clip((x - lo) / (hi - lo), 0, 1)
            a = {"tcc": 0.88 * t ** 1.2, "pwat": 0.45 + 0.5 * t}.get(field, np.full(x.shape, 0.82))
        img = LinearSegmentedColormap.from_list(field, RAMPS[field])(np.nan_to_num(t))
        img[..., 3] = np.nan_to_num(a)
    img = img.copy()
    img[coast, :3] = {"t2m": [0.12, 0.14, 0.2], "mslp": [0.49, 0.55, 0.65]}.get(field, [0.85, 0.88, 0.94])
    img[coast, 3] = np.maximum(img[coast, 3], 0.75)
    img[~inside] = 0.0
    return img


def lows(ds) -> list[dict]:
    """Depth and position of the deepest low north of 20 N between 60 W and 10 E, at every lead time."""
    sub = ds["mslp"].sel(latitude=slice(90, 20), longitude=slice(-60, 10))
    out = []
    for h in ds["lead"].values:
        m = sub.sel(lead=h)
        i, j = np.unravel_index(int(np.nanargmin(m.values)), m.shape)
        out.append(dict(lead_h=int(h), hPa=round(float(m.values[i, j]), 1), lat=float(m.latitude[i]), lon=float(m.longitude[j])))
    return out


def write_inputs(folder, source: str = "shipped", run: str | None = gfs.RUN, n: int = N, log=print) -> dict:
    """Write ``relief.png``, the 40 pane pictures and ``manifest.json`` into ``folder``; returns the manifest."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ds = gfs.hemisphere(source, run, log)
    elev = gfs.relief(log)
    lat0, lon0 = gfs.HEMI_CENTRE
    _save(relief_picture(elev), folder / "relief.png")
    lat, lon, inside, _ = ortho.disc(n, lat0, lon0, 90.0)
    land = ortho.sample(elev["z"], elev["lat"], elev["lon"], lat, lon) > 0
    coast = coast_from_mask(land, inside)
    glat, glon = ds["latitude"].values, ds["longitude"].values
    worst = 0.0
    for k in range(ds.sizes["lead"]):
        for f in FIELDS:
            x = ortho.sample(ds[f].values[k], glat, glon, lat, lon)
            worst = max(worst, float(np.isnan(x[inside]).mean()))
            _save(pane_picture(f, x, inside, coast), folder / f"{f}_{k}.png")
    if worst > 0.002:
        raise RuntimeError(f"{worst:.2%} of a disc has no data: the subset does not cover the hemisphere")
    man = dict(run=str(ds.attrs["run"])[:16] + ":00", leads_h=[int(h) for h in ds["lead"].values], centre=[lat0, lon0], n=n,
               ranges=RANGES, ramps=RAMPS, units=UNITS, isobar_step_hPa=4, atlantic_low=lows(ds),
               source="NOAA GFS forecast read from dynamical.org (CC BY 4.0)", credit=gfs.CREDIT,
               relief="ETOPO 2022, 10 arc-minutes (NOAA NCEI), hill shade x 20")
    (folder / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"wrote relief.png, {ds.sizes['lead'] * len(FIELDS)} pane pictures and manifest.json -> {folder}")
    return man


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m gismaps.glass.remotion <public folder> [live]")
    write_inputs(sys.argv[1], source="live" if "live" in sys.argv[2:] else "shipped")
