"""gismaps.glass.panes: one field -> one glass-pane picture, and the globe's skin.

A pane picture is an RGBA image (red, green, blue, opacity), n x n pixels, showing the orthographic view of one
field over a round cap of the Earth; everything outside the disc is clear. ``pane`` does four things: resample the
field onto the disc, colour it with a ramp, choose the opacity (constant, or growing with the value, or clear below
a threshold), and draw lines on top (isobars, the coastline).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi

from . import ortho

#: the palette of the frame: deep navy page, white ink, a warm and a cool accent
K = {"bg": "#122240", "bg2": "#0e1b36", "panel": "#1b2f57", "ink": "#f4f7fc", "muted": "#9db0cc", "rule": "#3a4f7a",
     "amber": "#ffb84d", "ice": "#a9dcff"}

#: one colour ramp per quantity, low -> high
RAMPS = {
    "t2m": ["#1d3f8f", "#4f86d0", "#a9c8ec", "#eef1f6", "#f3b792", "#dc6a45", "#9c2227"],   # two-sided, white at 0 degC
    "mslp": ["#1b2c4e", "#3c5a8a", "#8ea8cf"],                         # pressure: a quiet blue-grey (the isobars speak)
    "wind": ["#5b3a10", "#b8741c", "#f5a623", "#ffd27a", "#fff6da"],   # wind speed: amber
    "pwat": ["#221f5c", "#3f3a9c", "#7668d2", "#b9a4ee", "#f3e9ff"],   # precipitable water: violet
    "tcc": ["#7f8aa0", "#b9c2d2", "#f2f5fa"],                          # cloud: grey to white
    "rain": ["#2a56b0", "#3f8fe8", "#6cc4ff", "#c4eeff", "#ffffff"],   # rain rate: blue
}
ISOBARS = np.arange(900, 1080, 4)                                      # every 4 hPa


def ramp(name: str) -> LinearSegmentedColormap:
    """The colour map of one quantity."""
    return LinearSegmentedColormap.from_list(name, RAMPS[name])


def coast_lines(scale: str = "50m") -> list[np.ndarray]:
    """Natural Earth coastline as a list of (longitude, latitude) polylines (cartopy downloads it on first use)."""
    import cartopy.io.shapereader as shp

    out = []
    for geom in shp.Reader(shp.natural_earth(resolution=scale, category="physical", name="coastline")).geometries():
        for g in getattr(geom, "geoms", [geom]):
            out.append(np.asarray(g.coords)[:, :2])
    return out


def relief_texture(elev: dict, coast=None, width: int = 2880) -> np.ndarray:
    """The globe's skin: a shaded-relief picture in latitude x longitude (row 0 = 90 N, column 0 = 180 W), values 0..1.
    Light clay for land, slate blue for sea, lit from the north-west (shading exaggerated), the coastline drawn in."""
    from matplotlib.colors import LightSource
    from PIL import Image

    z = elev["z"]
    land = z > 0
    hs = LightSource(azdeg=315, altdeg=45).hillshade(np.where(land, z, z * 0.12), vert_exag=9.0, dx=18500.0, dy=18500.0)
    clay = np.array([0.86, 0.83, 0.77])
    high = np.clip(z / 5500.0, 0, 1)[..., None]                        # high ground a little whiter
    rgb_land = (clay * (1 - 0.35 * high) + 0.35 * high) * (0.42 + 0.80 * hs[..., None])
    deep = np.clip(-z / 6000.0, 0, 1)[..., None]                       # 0 at the coast, 1 in the deep ocean
    rgb_sea = (np.array([0.36, 0.48, 0.66]) * (1 - deep) + np.array([0.19, 0.28, 0.45]) * deep) * (0.80 + 0.28 * hs[..., None])
    rgb = np.clip(np.where(land[..., None], rgb_land, rgb_sea), 0, 1)
    img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((width, width // 2), Image.BICUBIC)
    if coast is None:
        return np.asarray(img).astype(np.float32) / 255.0
    fig = plt.figure(figsize=(width / 100, width / 200), dpi=100)      # draw the coastline into the skin
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, extent=(-180, 180, -90, 90), aspect="auto")
    for ln in coast:
        ax.plot(ln[:, 0], ln[:, 1], color="#2b3954", lw=0.9)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axis("off")
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
    plt.close(fig)
    return out


def fields(ds, lead: int) -> dict:
    """The six maps of one lead time (hours) from the dataset of ``gfs.region``, plus a smoothed pressure that is
    used ONLY to draw the isobars (stored pressure comes in steps of 0.64 hPa, which makes raw isobars stair-stepped
    in the tropics)."""
    F = {k: ds[k].sel(lead=int(lead)) for k in ("t2m", "mslp", "wind", "pwat", "tcc", "rain")}
    F["mslp_smooth"] = F["mslp"].copy(data=ndi.gaussian_filter(F["mslp"].values, 3.0))
    return F


def line_layer(n: int, draw) -> np.ndarray:
    """A transparent n x n picture on which ``draw(ax)`` paints lines; the axes run -1..1 both ways (pane coordinates)."""
    fig = plt.figure(figsize=(n / 100, n / 100), dpi=100)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    ax.patch.set_alpha(0.0)
    draw(ax)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32) / 255.0
    plt.close(fig)
    return rgba


def over(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Put the RGBA picture ``top`` over ``bottom`` (straight alpha, values 0..1)."""
    at, ab = top[..., 3:4], bottom[..., 3:4]
    a = at + ab * (1.0 - at)
    rgb = (top[..., :3] * at + bottom[..., :3] * ab * (1.0 - at)) / np.maximum(a, 1e-6)
    return np.concatenate([rgb, a], axis=-1)


def pane(field, cmap, vmin, vmax, title="", unit="", *, lat0, lon0, rad_deg, n=480, alpha=0.9, fade_below=None,
         alpha_by_value=False, log=False, lines=None, line_levels=None, line_color="#fff3dc", line_width=1.3,
         coast=None, coast_color="#0b1730", note="") -> dict:
    """One glass-pane picture: the orthographic view of ``field`` over the cap (lat0, lon0, rad_deg).

    field           2-D xarray map with latitude and longitude coordinates
    cmap            colour ramp, dark -> bright for amounts
    vmin, vmax      values at the two ends of the ramp
    alpha           opacity of the colours (0..1)
    fade_below      values below this are fully clear (rain below 0.2 mm/h)
    alpha_by_value  True: opacity grows with the value (cloud: clear sky is clear glass)
    log             True: logarithmic colour scale (rain rates)
    lines           2-D xarray map whose contour lines are drawn at ``line_levels`` (pressure)
    coast           list of (lon, lat) polylines to draw

    Returns a dictionary: the picture ("tex", n x n x 4, values 0..1) and what the legend needs.
    """
    s = np.sin(np.radians(rad_deg))                                    # disc radius of the cap (globe radius 1)
    lat, lon, inside, c = ortho.disc(n, lat0, lon0, rad_deg)           # the place each pixel shows
    tex = np.zeros((n, n, 4), dtype=np.float32)
    val = ortho.sample(field.values, field.latitude.values, field.longitude.values, lat, lon)
    if log:                                                            # position along the ramp, 0..1
        x = (np.log(np.clip(val, vmin, None)) - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
    else:
        x = (val - vmin) / (vmax - vmin)
    x = np.clip(x, 0.0, 1.0)
    tex[..., :3] = cmap(np.nan_to_num(x))[..., :3]
    a = np.full((n, n), float(alpha), dtype=np.float32)
    if alpha_by_value:
        a = a * x                                                      # more of the quantity = more opaque
    if fade_below is not None:                                         # clear below the threshold, soft edge above it
        a = a * np.clip((val - fade_below) / max(fade_below, 1e-9), 0.0, 1.0) ** 0.5
    tex[..., 3] = np.where(inside & np.isfinite(val), a, 0.0)

    def draw(ax):
        clip = plt.Circle((0, 0), 1.0, transform=ax.transData)         # nothing is drawn outside the disc
        if coast is not None:
            for ln in coast:
                uu, vv, ww = ortho.forward(ln[:, 1], ln[:, 0], lat0, lon0)
                if (ww > np.cos(np.radians(rad_deg)) - 0.05).any():    # at least partly inside the cap
                    uu = np.where(ww > 0, uu, np.nan)                  # break the line on the far side
                    (h,) = ax.plot(uu / s, vv / s, color=coast_color, lw=1.1 * max(1.0, n / 480), solid_capstyle="round")
                    h.set_clip_path(clip)
        if lines is not None:
            lv = ortho.sample(lines.values, lines.latitude.values, lines.longitude.values, lat, lon)
            cs = ax.contour(c, c, lv[::-1], levels=line_levels, colors=line_color, linewidths=line_width * max(1.0, n / 480))
            cs.set_clip_path(clip)

    if coast is not None or lines is not None:
        tex = over(line_layer(n, draw), tex)                           # lines on top of the colours
        tex[..., 3] *= inside
    return {"tex": tex, "title": title, "unit": unit, "cmap": cmap, "vmin": vmin, "vmax": vmax, "log": log, "note": note}


def six_panes(F: dict, cap: dict, coast, n: int = 480) -> list[dict]:
    """The six panes of one lead time, nearest the globe first. ``F`` comes from ``fields``; ``cap`` is
    ``dict(lat0=..., lon0=..., rad_deg=...)``."""
    kw = dict(coast=coast, n=n, **cap)                                 # the same cap and coastline on every pane
    return [
        # temperature: a TWO-sided ramp centred on 0 degC (blue below freezing, red above)
        pane(F["t2m"], ramp("t2m"), -30, 30, "Temperature at 2 m", "°C", alpha=0.88, **kw),
        # pressure: a quiet blue-grey fill; the isobars carry the information
        pane(F["mslp"], ramp("mslp"), 990, 1030, "Sea-level pressure", "hPa", alpha=0.80, lines=F["mslp_smooth"],
             line_levels=ISOBARS, coast_color="#5d7096", note="isobars every 4 hPa", **kw),
        # wind speed: one direction, dark -> bright amber
        pane(F["wind"], ramp("wind"), 0, 30, "Wind at 10 m", "m s⁻¹", alpha=0.88, coast_color="#f4f7fc", **kw),
        # precipitable water: one direction, dark -> bright violet
        pane(F["pwat"], ramp("pwat"), 0, 60, "Precipitable water", "kg m⁻²", alpha=0.88, coast_color="#f4f7fc", **kw),
        # cloud: opacity grows with cloud cover, so clear sky is clear glass
        pane(F["tcc"], ramp("tcc"), 0, 100, "Cloud cover", "%", alpha=0.92, alpha_by_value=True, coast_color="#c7d3e6", **kw),
        # rain rate: logarithmic scale, clear below 0.2 mm/h
        pane(F["rain"], ramp("rain"), 0.2, 20, "Rain and snow rate", "mm h⁻¹", alpha=0.95, fade_below=0.2, log=True,
             coast_color="#c7d3e6", **kw),
    ]
