"""gismaps.styles.common: the shared pieces of the seven map styles.

* ``MapGrid``    one projection + one pixel grid for every layer of a map (sample fields, project points, land mask)
* ``canvas``     a matplotlib figure whose data coordinates ARE output pixels (x right, y down)
* ``blend`` ...  small numpy compositing helpers (alpha-over, additive glow, drop shadow, hillshade)
* furniture      title block, credit block, colour bar, all in frame pixels
* ``Movie``      write numbered PNG frames and encode them with ffmpeg

The rule: rasters are composed in numpy on the map's own pixel grid (pixel-exact colours that match the
colour bar); matplotlib only draws lines and type on top. Never let matplotlib choose the extent.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:      # scripts draw off-screen; a notebook keeps its inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

# ---------------------------------------------------------------- the dark look
BG = "#0b1020"        # frame background (navy)
PANEL = "#111a30"     # ocean / map ground
LAND = "#1c2742"      # land on the dark ground
COAST = "#6c7da8"     # coast and border lines
INK = "#f1ede4"       # main type
MUTED = "#9aa3b8"     # secondary type
AMBER = "#f2b33d"     # the accent (numbers, playheads)


def _fonts() -> tuple[str, str]:
    """Georgia and Segoe UI where they are installed (Windows), else the DejaVu pair that ships with matplotlib
    (Linux, Google Colab), so no "font not found" warning is ever printed. ``GIS_FONTS=dejavu`` forces DejaVu."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    forced = os.environ.get("GIS_FONTS", "").lower() == "dejavu"
    return ("Georgia" if "Georgia" in have and not forced else "DejaVu Serif",
            "Segoe UI" if "Segoe UI" in have and not forced else "DejaVu Sans")


SERIF, SANS = _fonts()
#: type size factor: 1 for Georgia / Segoe UI; smaller for the wider DejaVu fonts, so every line still fits its place
FONT_SCALE = 0.88 if SANS.startswith("DejaVu") else 1.0
AUTHOR = "Map created by Shammunul Islam"


def rgb(c) -> np.ndarray:
    """Colour name or hex -> float32 RGB triple (0..1)."""
    return np.array(to_rgb(c), dtype=np.float32)


#: set by ``preview``: {"thin": keep every n-th frame, "scale": frame size factor, "fps": frames per second}
_PREVIEW: dict | None = None


# ---------------------------------------------------------------- canvas
def canvas(w: int, h: int, bg=BG, scale: float = 1.0):
    """Figure of w x h design pixels (saved at ``scale`` x that) with one pixel-coordinate axes (origin top-left)."""
    if _PREVIEW:                              # a preview run of an animation: smaller frames (see ``preview``)
        scale = scale * _PREVIEW["scale"]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100 * scale)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def show(ax, img: np.ndarray, x: float, y: float, w: float | None = None, h: float | None = None, z: float = 1):
    """Place an RGB(A) array with its top-left corner at (x, y) in frame pixels."""
    ih, iw = img.shape[:2]
    w = iw if w is None else w
    h = ih if h is None else h
    ax.imshow(np.clip(img, 0, 1), extent=(x, x + w, y + h, y), interpolation="lanczos" if (w, h) != (iw, ih) else "none",
              zorder=z, aspect="auto")


def clip_to(artists, ax, x, y, w, h):
    """Clip matplotlib artists (one or a list) to a rectangle given in frame pixels."""
    from matplotlib.patches import Rectangle
    r = Rectangle((x, y), w, h, transform=ax.transData)
    for a in (artists if isinstance(artists, (list, tuple)) else [artists]):
        a.set_clip_path(r)


def halo(t, width: float = 2.6, colour=BG):
    """Give a text a dark halo so it reads on any ground."""
    import matplotlib.patheffects as pe
    t.set_path_effects([pe.withStroke(linewidth=width, foreground=colour)])
    return t


def text(ax, x, y, s, size, colour=INK, family=None, weight="normal", style="normal", ha="left", va="baseline", **kw):
    """Type in frame pixels: ``size`` is the font size in design pixels."""
    return ax.text(x, y, s, fontsize=size * 0.72 * FONT_SCALE, color=colour, family=family or SANS, weight=weight, style=style,
                   ha=ha, va=va, **kw)


def title_block(ax, x, y, title, subtitle=None, note=None, size=54, width=None):
    """Serif title, italic subtitle, small muted note (top-left anchor)."""
    text(ax, x, y + size, title, size, family=SERIF, weight="bold")
    yy = y + size
    if subtitle:
        yy += size * 0.72
        text(ax, x, yy, subtitle, size * 0.48, family=SERIF, style="italic", colour="#d9d4c7")
    if note:
        for line in note.split("\n"):
            yy += size * 0.50
            text(ax, x, yy, line, size * 0.36, colour=MUTED)
    return yy


def credit_block(ax, x, y, data_lines, size=17, ha="left"):
    """The creator line, then the data-source lines (licence credits are required; keep them)."""
    text(ax, x, y, AUTHOR, size * 1.1, weight="bold", ha=ha)
    for i, line in enumerate(data_lines):
        text(ax, x, y + (i + 1) * size * 1.45, line, size, colour=MUTED, ha=ha)


def colourbar(ax, x, y, w, h, cmap, vmin, vmax, ticks, label, size=17, fmt="{:g}", norm=None, extend=None):
    """Horizontal colour bar in frame pixels. ``norm`` maps a value to 0..1 (default linear vmin..vmax)."""
    norm = norm or (lambda v: (np.asarray(v, float) - vmin) / (vmax - vmin))
    ramp = cmap(np.linspace(0, 1, 256))[None, :, :3]
    ax.imshow(ramp, extent=(x, x + w, y + h, y), aspect="auto", zorder=5)
    ax.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], color=MUTED, lw=0.6, zorder=6)
    for t in ticks:
        px = x + w * float(norm(t))
        ax.plot([px, px], [y + h, y + h + 5], color=MUTED, lw=0.8, zorder=6)
        text(ax, px, y + h + 8 + size, fmt.format(t), size, colour=INK, ha="center")
    text(ax, x, y - 8, label, size, colour=MUTED)


def finish(fig, path, scale: float | None = None):
    """Save (PNG) and close. The axes limits are reset to the frame first (imshow moves them)."""
    ax = fig.axes[0]
    w, h = fig.get_size_inches() * 100
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=fig.dpi if scale is None else 100 * scale, facecolor=fig.get_facecolor())
    plt.close(fig)
    return Path(path)


# ---------------------------------------------------------------- numpy compositing
def over(dst: np.ndarray, src_rgb, alpha) -> np.ndarray:
    """Alpha-over: lay ``src_rgb`` (h, w, 3 or a single colour) on ``dst`` with alpha (h, w) or scalar."""
    a = np.asarray(alpha, np.float32)
    a = a[..., None] if a.ndim == 2 else a
    return dst * (1 - a) + np.asarray(src_rgb, np.float32) * a


def colourise(field: np.ndarray, cmap, vmin: float, vmax: float, levels=None) -> np.ndarray:
    """Field -> RGB with a matplotlib colour map. ``levels`` quantises into filled bands (colour = band midpoint)."""
    if levels is not None:
        lv = np.asarray(levels, float)
        k = np.clip(np.digitize(field, lv) - 1, 0, len(lv) - 2)
        t = (k + 0.5) / (len(lv) - 1)
    else:
        t = (field - vmin) / (vmax - vmin)
    return cmap(np.clip(np.nan_to_num(t), 0, 1))[..., :3].astype(np.float32)


def hillshade(z: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0, zfactor: float = 1.0) -> np.ndarray:
    """Lambert shading 0..1 of a height array on a pixel grid (light from ``azimuth``, map north = up)."""
    gy, gx = np.gradient(z.astype(np.float32) * zfactor)
    az, alt = np.deg2rad(azimuth), np.deg2rad(altitude)
    lx, ly, lz = np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)   # light vector (east, north, up)
    n = np.sqrt(gx**2 + gy**2 + 1)
    # surface normal = (-dz/dx, -dz/dy_north, 1); the array's y runs down, so dz/dy_north = -gy
    return np.clip((-gx * lx + gy * ly + lz) / n, 0, 1)


def drop_shadow(mask: np.ndarray, dx: float = 10, dy: float = 14, blur: float = 12, strength: float = 0.55) -> np.ndarray:
    """Alpha (h, w) of a soft shadow cast by ``mask`` (1 inside), to lay in black UNDER the masked layer."""
    sh = ndi.shift(mask.astype(np.float32), (dy, dx), order=1, mode="constant")
    return np.clip(ndi.gaussian_filter(sh, blur) * strength, 0, 1)


def glow(points_xy: np.ndarray, weights: np.ndarray, shape: tuple[int, int], core: float = 1.1, halo: float = 5.0,
         halo_gain: float = 0.6) -> np.ndarray:
    """Additive glow buffer (h, w): points binned to pixels, blurred at two scales (a sharp core and a soft halo).

    Because the buffer is additive, dots that fall on the same place add up: density reads as brightness.
    """
    h, w = shape
    x, y = points_xy[:, 0], points_xy[:, 1]
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    buf = np.zeros((h, w), np.float32)
    np.add.at(buf, (y[ok].astype(int), x[ok].astype(int)), weights[ok].astype(np.float32))
    return ndi.gaussian_filter(buf, core) * (2 * np.pi * core**2) ** 0.5 + halo_gain * ndi.gaussian_filter(buf, halo) * halo


def tone(buf: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Soft tone-map of an additive buffer to 0..1 (1 - exp(-gain * v)): bright places saturate gently."""
    return 1.0 - np.exp(-gain * buf)


# ---------------------------------------------------------------- Natural Earth land raster (projection-agnostic)
@lru_cache(maxsize=8)
def land_raster(lon0: float = -180, lon1: float = 180, lat0: float = -90, lat1: float = 90, res: float = 0.1,
                scale: str = "50m") -> np.ndarray:
    """Boolean land mask on a plain lat/lon grid (row 0 = lat1, north), rasterised from Natural Earth land polygons.

    A lat/lon raster can be sampled through ANY projection (even interrupted ones) without clipping polygons.
    Natural Earth is public domain; cartopy downloads the shapefile on first use.
    """
    import cartopy.io.shapereader as shp
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch

    w, h = int(round((lon1 - lon0) / res)), int(round((lat1 - lat0) / res))
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.axis("off")
    fig.patch.set_facecolor("black")
    for geom in shp.Reader(shp.natural_earth(resolution=scale, category="physical", name="land")).geometries():
        for poly in getattr(geom, "geoms", [geom]):
            if poly.bounds[2] < lon0 or poly.bounds[0] > lon1 or poly.bounds[3] < lat0 or poly.bounds[1] > lat1:
                continue
            verts, codes = [], []
            for ring in [poly.exterior, *poly.interiors]:
                xy = np.asarray(ring.coords)[:, :2]
                verts.append(xy)
                codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(np.concatenate(verts), codes), fc="white", ec="none", antialiased=False))
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., 0] > 127
    plt.close(fig)
    return a


def border_lines(scale: str = "50m", kind: str = "admin_0_boundary_lines_land") -> list[np.ndarray]:
    """Natural Earth country border polylines as (n, 2) lon/lat arrays."""
    import cartopy.io.shapereader as shp
    out = []
    for geom in shp.Reader(shp.natural_earth(resolution=scale, category="cultural", name=kind)).geometries():
        for line in getattr(geom, "geoms", [geom]):
            out.append(np.asarray(line.coords)[:, :2])
    return out


# ---------------------------------------------------------------- the map grid
class MapGrid:
    """One projection and one pixel grid shared by every layer of a map.

    Parameters
    ----------
    proj : str
        A PROJ string (``"+proj=lcc +lat_1=45 +lat_2=58 +lat_0=50 +lon_0=-5"``, ``"+proj=eqearth +lon_0=-165"``,
        ``"+proj=spilhaus"`` ...).
    x0, x1, y0, y1 : float
        Extent in projected metres (left, right, bottom, top).
    w : int
        Width in pixels; the height follows from the extent (square pixels).
    """

    def __init__(self, proj: str, x0: float, x1: float, y0: float, y1: float, w: int):
        from pyproj import Transformer
        self.proj = proj
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.w = int(w)
        self.h = int(round(w * (y1 - y0) / (x1 - x0)))
        self.mpp = (x1 - x0) / self.w                       # metres per pixel
        self._fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
        self._inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)
        self._ll = None

    @classmethod
    def from_box(cls, proj: str, lon0: float, lon1: float, lat0: float, lat1: float, w: int, pad: float = 0.0):
        """Smallest projected rectangle that holds a lon/lat box (plus ``pad`` as a fraction)."""
        from pyproj import Transformer
        f = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
        lo = np.concatenate([np.linspace(lon0, lon1, 60), np.full(60, lon1), np.linspace(lon1, lon0, 60), np.full(60, lon0)])
        la = np.concatenate([np.full(60, lat0), np.linspace(lat0, lat1, 60), np.full(60, lat1), np.linspace(lat1, lat0, 60)])
        x, y = f.transform(lo, la)
        dx, dy = (x.max() - x.min()) * pad, (y.max() - y.min()) * pad
        return cls(proj, x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy, w)

    @classmethod
    def fit(cls, proj: str, cx: float, cy: float, width_m: float, w: int, h: int):
        """Grid of exactly w x h pixels centred on projected (cx, cy), ``width_m`` metres wide."""
        hm = width_m * h / w
        g = cls(proj, cx - width_m / 2, cx + width_m / 2, cy - hm / 2, cy + hm / 2, w)
        g.h = h
        return g

    # -- coordinates
    def xy(self, lon, lat):
        """lon/lat -> pixel coordinates (x right, y down); NaN where the projection fails."""
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        x = np.where(np.isfinite(x), x, np.nan)
        return (x - self.x0) / self.mpp, (self.y1 - np.where(np.isfinite(y), y, np.nan)) / self.mpp

    def lonlat(self):
        """lon, lat (h, w) of every pixel centre; NaN outside the projection's domain."""
        if self._ll is None:
            px = self.x0 + (np.arange(self.w) + 0.5) * self.mpp
            py = self.y1 - (np.arange(self.h) + 0.5) * self.mpp
            X, Y = np.meshgrid(px, py)
            lon, lat = self._inv.transform(X, Y)
            bad = ~np.isfinite(lon) | ~np.isfinite(lat)
            # an inverse can return a point for pixels outside the map: keep only pixels that round-trip
            xb, yb = self._fwd.transform(np.where(bad, 0, lon), np.where(bad, 0, lat))
            bad |= ~(np.hypot(xb - X, yb - Y) < 2 * self.mpp)
            self._ll = (np.where(bad, np.nan, lon).astype(np.float32), np.where(bad, np.nan, lat).astype(np.float32))
        return self._ll

    # -- sampling
    def sample(self, field: np.ndarray, lats: np.ndarray, lons: np.ndarray, order: int = 1, wrap: bool = False):
        """Field on a regular lat/lon grid -> (h, w) on the pixel grid (bilinear by default; NaN outside).

        ``lats`` and ``lons`` must be evenly spaced (either direction). ``wrap=True`` for global longitude grids.
        """
        lon, lat = self.lonlat()
        dlat, dlon = lats[1] - lats[0], lons[1] - lons[0]
        if wrap:
            lon = (lon - lons[0]) % 360 + lons[0]
        r = (lat - lats[0]) / dlat
        c = (lon - lons[0]) / dlon
        f = np.asarray(field, np.float32)
        if wrap:  # one extra column so the seam interpolates
            f = np.concatenate([f, f[:, :1]], axis=1)
        nanmask = ~np.isfinite(f)
        if nanmask.any():
            if order == 0:
                f = np.where(nanmask, 0, f)
            else:  # fill NaNs with the nearest valid value first so edges do not bleed zeros
                idx = ndi.distance_transform_edt(nanmask, return_distances=False, return_indices=True)
                f = f[tuple(idx)]
        bad = ~np.isfinite(r) | ~np.isfinite(c)
        out = ndi.map_coordinates(f, [np.where(bad, 0, r), np.where(bad, 0, c)], order=order, mode="nearest")
        if nanmask.any():
            m = ndi.map_coordinates(nanmask.astype(np.float32), [np.where(bad, 0, r), np.where(bad, 0, c)], order=min(order, 1),
                                    mode="nearest")
            out = np.where(m > 0.5, np.nan, out)
        outside = bad | (r < -0.5) | (r > f.shape[0] - 0.5) | (c < -0.5) | (c > f.shape[1] - 0.5)
        return np.where(outside, np.nan, out)

    def land(self, scale: str = "50m", res: float | None = None, soft: float = 0.8) -> np.ndarray:
        """Land mask 0..1 on the pixel grid (soft edge of ``soft`` px), from Natural Earth through a lat/lon raster."""
        lon, lat = self.lonlat()
        lo0, lo1 = np.floor(np.nanmin(lon)) - 1, np.ceil(np.nanmax(lon)) + 1
        la0, la1 = max(-90, np.floor(np.nanmin(lat)) - 1), min(90, np.ceil(np.nanmax(lat)) + 1)
        if lo1 - lo0 > 300:
            lo0, lo1 = -180.0, 180.0
        if res is None:  # about one raster cell per pixel, within sane bounds
            res = float(np.clip(np.rad2deg(self.mpp / 6.371e6), 0.02, 0.25))
            res = [0.02, 0.025, 0.05, 0.1, 0.125, 0.25][int(np.argmin(np.abs(np.array([0.02, 0.025, 0.05, 0.1, 0.125, 0.25]) - res)))]
        ras = land_raster(float(lo0), float(lo1), float(la0), float(la1), res, scale)
        lon = (lon - lo0) % 360 + lo0 if lo1 - lo0 >= 359 else lon
        r = (la1 - lat) / res - 0.5
        c = (lon - lo0) / res - 0.5
        bad = ~np.isfinite(r)
        m = ndi.map_coordinates(ras.astype(np.float32), [np.where(bad, 0, r), np.where(bad, 0, c)], order=1, mode="nearest")
        m = np.where(bad, 0, m)
        return np.clip(ndi.gaussian_filter(m, soft), 0, 1) if soft else m

    def valid(self) -> np.ndarray:
        """True where a pixel is inside the projection's domain."""
        return np.isfinite(self.lonlat()[0])

    def lines(self, polylines: list[np.ndarray], max_jump: float = 80.0) -> list[np.ndarray]:
        """Project lon/lat polylines to pixel polylines, split where they jump (projection seams) or leave the frame."""
        out = []
        for ll in polylines:
            x, y = self.xy(ll[:, 0], ll[:, 1])
            ok = np.isfinite(x) & np.isfinite(y) & (x > -50) & (x < self.w + 50) & (y > -50) & (y < self.h + 50)
            jump = np.r_[False, np.hypot(np.diff(x), np.diff(y)) > max_jump]
            start = 0
            for i in range(len(x) + 1):
                if i == len(x) or not ok[i] or jump[i]:
                    if i - start >= 2:
                        out.append(np.c_[x[start:i], y[start:i]])
                    start = i if (i < len(x) and ok[i]) else i + 1
        return out

    def ground(self, ocean=PANEL, land=LAND, scale: str = "50m") -> np.ndarray:
        """The muted dark base map (ocean + land) as RGB (h, w, 3)."""
        base = np.broadcast_to(rgb(ocean), (self.h, self.w, 3)).copy()
        return over(base, rgb(land), self.land(scale))


def coast_alpha(land: np.ndarray, width: float = 1.2) -> np.ndarray:
    """A thin coastline as alpha (h, w): the 0.5 crossing of a soft land mask."""
    edge = np.abs(land - 0.5)
    g = np.hypot(*np.gradient(land)) + 1e-6
    return np.clip(1 - (edge / g) / width, 0, 1) * (g > 0.02)


# ---------------------------------------------------------------- frames -> MP4
class Movie:
    """Write frames ``f_00000.png`` ... into a folder, then encode with ffmpeg (H.264, yuv420p).

    Use: ``thing.frame(..., mv.path())`` for every frame (``frame`` must return at once when the path is ``None``),
    ``mv.hold(n)`` to repeat the last frame n times, then ``mv.encode(out)``.

    Frames are independent, so a script renders them in parallel: the first process (the master) starts
    ``GIS_WORKERS`` copies of the same script (default 6; 1 = no parallel run; always 1 inside a notebook). Copy i of n
    renders only the frames whose number is i modulo n (``mv.path()`` gives it ``None`` for all others) and stops at
    ``encode``. The master waits, makes the held copies and encodes.

    Inside ``preview(...)`` the same calls make a short clip instead: only every ``thin``-th frame is drawn, at a
    smaller size, in this process, and holds are shortened to the preview's frame rate.
    """

    def __init__(self, folder, fps: int = 30):
        self.fps, self.n, self.holds = fps, 0, []
        self.calls, self.thin, self.hold_factor = 0, 1, 1.0
        if _PREVIEW:
            self.thin, self.hold_factor, self.fps = _PREVIEW["thin"], _PREVIEW["fps"] / fps, _PREVIEW["fps"]
            self.dir, self.role = Path(folder), "single"
            if self.dir.exists():
                shutil.rmtree(self.dir)
            self.dir.mkdir(parents=True)
            return
        part = os.environ.get("GIS_PART")
        if part:                                            # a worker: render my share into the master's folder
            self.i, self.parts = (int(v) for v in part.split("/"))
            self.dir = Path(os.environ["GIS_PART_DIR"])
            self.role = "worker"
            return
        self.dir = Path(folder)
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        workers = int(os.environ.get("GIS_WORKERS", "6"))
        script = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
        if workers > 1 and "ipykernel" not in sys.modules and script is not None and script.suffix == ".py" and script.exists():
            self.role = "master"
            procs = []
            for i in range(workers):
                env = dict(os.environ, GIS_PART=f"{i}/{workers}", GIS_PART_DIR=str(self.dir), GIS_MODE="FULL",
                           GIS_OUT=str(self.dir / f"_w{i}"))
                log = open(self.dir / f"_w{i}.log", "w")
                procs.append((subprocess.Popen([sys.executable, *sys.argv], env=env, stdout=log, stderr=subprocess.STDOUT), log))
            for p, log in procs:
                code = p.wait()
                log.close()
                if code:
                    raise RuntimeError(f"a frame worker failed (exit {code}); see {log.name}")
        else:
            self.role = "single"

    def path(self) -> Path | None:
        """Path of the next frame, or ``None`` if this process does not render it."""
        c = self.calls
        self.calls += 1
        if c % self.thin:                                   # a preview keeps every thin-th frame
            return None
        k = self.n
        self.n += 1
        if self.role == "master" or (self.role == "worker" and k % self.parts != self.i):
            return None
        return self.dir / f"f_{k:05d}.png"

    def hold(self, n: int) -> None:
        """Repeat the last frame n times (a held ending lets the eye read the final state)."""
        src = self.n - 1
        if self.hold_factor != 1.0:
            n = max(1, int(round(n * self.hold_factor)))
        for _ in range(n):
            self.holds.append((src, self.n))
            self.n += 1

    def encode(self, out, crf: int = 18, keep_frames: bool = False) -> Path | None:
        if self.role == "worker":
            return None
        for src, dst in self.holds:
            shutil.copyfile(self.dir / f"f_{src:05d}.png", self.dir / f"f_{dst:05d}.png")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg_exe(), "-v", "error", "-y", "-framerate", str(self.fps), "-i", str(self.dir / "f_%05d.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
               "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
        if not keep_frames:
            shutil.rmtree(self.dir, ignore_errors=True)
        return out


def ffmpeg_exe() -> str:
    """The ffmpeg program: the one on the PATH, else the one that ``pip install imageio-ffmpeg`` brings."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def preview(animate, out, frames_dir, thin: int = 8, scale: float = 0.5, fps: int = 12, **kw):
    """Run an ``animate`` method as a short preview clip: every ``thin``-th frame, ``scale`` times the size
    (0.5 -> 960 x 540 px), ``fps`` frames per second. A minute of work instead of many; the same code path.

        clip = preview(storm.animate, "out/glow_preview.mp4", "out/frames", thin=16)
    """
    global _PREVIEW
    _PREVIEW = {"thin": max(1, int(thin)), "scale": float(scale), "fps": int(fps)}
    try:
        return animate(out, frames_dir, **kw)
    finally:
        _PREVIEW = None


def ease(t: float) -> float:
    """Smooth 0..1 ease in-out (cubic)."""
    t = float(np.clip(t, 0, 1))
    return t * t * (3 - 2 * t)


def data_dir() -> Path:
    """Folder holding the reduced inputs of the seven style tutorials (see ``gismaps.data``)."""
    from .. import data as _data

    return _data.folder("gis_styles")
