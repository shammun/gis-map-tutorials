"""Lens map: animated versions of the posters, in pure Python (numpy + matplotlib + ffmpeg).

  city     the three lenses open one by one over the muted base map, the label blocks and the legend arrive, then the
           lenses trade places round the triangle (every piece of ground is seen three ways) and come home.
  weather  the three lenses open on Storm Ciaran at 12 UTC 1 November 2023, then one clock steps every field hour by
           hour to 12 UTC 2 November while the tracked low (L, central pressure, track) crosses the lenses.

How a frame is made (the same recipe for both):
  1. every map layer is drawn ONCE with the poster's own functions (``gismaps.lens.city.render_layers``,
     ``gismaps.lens.weather.hour_layers``) at poster size, then shrunk to the video size;
  2. every piece of furniture (title, label block, leader, legend, credits) is drawn ONCE on a transparent canvas with
     the poster's ``furniture`` function, shrunk, and cropped to a small "sprite";
  3. per frame only numpy works: ``gismaps.lens.common.composite`` (the poster's lens rule) with this frame's lens
     centres, radii and opacities, soft rims, then the sprites pasted with this frame's opacities;
  4. the frames are piped to ffmpeg (H.264, yuv420p) - no frame files on disk.
So the animation cannot drift from the poster: at full opacity and home positions it is the poster, shrunk.

Run:
  python -m gismaps.lens.anim city --city brisbane [--derived FILE] [--out FILE] [--gif]
  python -m gismaps.lens.anim weather [--reduced FILE] [--out FILE] [--gif]
Without --derived / --reduced the small copies shipped with the repository are used.
Style after Fan Mo, #30DayMapChallenge 2026 Day 14; the code and data are original.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import common as lc  # noqa: E402

PW, PH = 2000, 2828  # poster pixels


# ------------------------------------------------------------------------------------------------------ small tools
def ramp(t: float, a: float, b: float) -> float:
    """0 before a, 1 after b, a straight line between."""
    return float(np.clip((t - a) / (b - a), 0.0, 1.0))


def smooth(x: float) -> float:
    """Ease in and out (smoothstep): slow start, slow stop."""
    return x * x * (3 - 2 * x)


def ease_out(x: float) -> float:
    """Fast start, slow stop (a lens opening)."""
    return 1 - (1 - x) ** 3


def video_size(scale: float) -> tuple[int, int]:
    """Video width and height for a poster shrunk by ``scale``; both even (H.264 with yuv420p needs even sizes)."""
    return 2 * round(PW * scale / 2), 2 * round(PH * scale / 2)


def shrink(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a float image (h, w, 3 or 4) in 0..1 to ``size`` = (width, height) with a Lanczos filter.
    RGBA is resized with premultiplied alpha, so transparent pixels do not darken the edges of text."""
    u8 = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    if a.shape[2] == 4:
        im = Image.fromarray(u8, "RGBA").convert("RGBa").resize(size, Image.LANCZOS).convert("RGBA")
    else:
        im = Image.fromarray(u8, "RGB").resize(size, Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


@dataclass
class Sprite:
    """A small piece of a transparent overlay: colour, alpha and where its top-left corner goes on the frame."""
    rgb: np.ndarray  # (h, w, 3)
    a: np.ndarray  # (h, w, 1)
    x: int
    y: int


def sprite(rgba: np.ndarray, x0: int = 0, y0: int = 0) -> Sprite | None:
    """Crop an RGBA overlay to the box that holds its visible pixels (most of an overlay is empty)."""
    ys, xs = np.where(rgba[..., 3] > 0.004)
    if len(ys) == 0:
        return None
    a, b, c, d = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return Sprite(rgba[a:b, c:d, :3].copy(), rgba[a:b, c:d, 3:].copy(), int(c) + x0, int(a) + y0)


def paste(img: np.ndarray, sp: Sprite | None, alpha: float = 1.0, dx: float = 0, dy: float = 0) -> None:
    """Blend a sprite onto the frame (in place) at its own position moved by (dx, dy), with an overall opacity."""
    if sp is None or alpha <= 0:
        return
    h, w = sp.a.shape[:2]
    x, y = sp.x + int(round(dx)), sp.y + int(round(dy))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    a = sp.a[y0 - y:y1 - y, x0 - x:x1 - x] * alpha
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + sp.rgb[y0 - y:y1 - y, x0 - x:x1 - x] * a


def overlay(draw, size: tuple[int, int], w: int = PW, h: int = PH) -> Sprite | None:
    """Run ``draw(fig, ax)`` on a transparent poster-size canvas and return the result as a video-size sprite."""
    fig, ax = lc.canvas(w, h)
    draw(fig, ax)
    return sprite(shrink(lc.grab(fig), size))


def rim(img: np.ndarray, cx: float, cy: float, r: float, colour: str, lw: float, alpha: float = 1.0) -> None:
    """A lens rim drawn with numpy (in place): a soft glow (two wide faint rings) and a crisp ring, like ``lc.rims``."""
    if r <= 1 or alpha <= 0:
        return
    pad = int(lw * 3) + 3
    x0, x1 = max(int(cx - r) - pad, 0), min(int(cx + r) + pad + 1, img.shape[1])
    y0, y1 = max(int(cy - r) - pad, 0), min(int(cy + r) + pad + 1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.abs(np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) - r)  # distance of every pixel from the circle
    near = d < lw * 2.5 + 1  # only the thin band of pixels the rim can reach
    d = d[near]
    keep = np.ones_like(d)
    for width, al in ((lw * 5, 0.10), (lw * 3, 0.16), (lw, 1.0)):  # three rings of one colour = one blend
        keep *= 1 - np.clip(width / 2 + 0.5 - d, 0, 1) * al * alpha
    reg = img[y0:y1, x0:x1]
    reg[near] = reg[near] * keep[:, None] + lc.hexrgb(colour) * (1 - keep[:, None])


class Writer:
    """Frames (float RGB arrays in 0..1) -> an MP4, through a pipe to ffmpeg. No frame files are written."""

    def __init__(self, path: Path, size: tuple[int, int], fps: int = 30, crf: int = 18):
        self.path, self.size, self.n = Path(path), size, 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def add(self, img: np.ndarray, repeat: int = 1) -> None:
        buf = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(buf)
            self.n += 1

    def close(self) -> Path:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        return self.path


def ffmpeg_exe() -> str:
    """The ffmpeg program: the one on the PATH, else the one that ``pip install imageio-ffmpeg`` brings."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def to_gif(mp4: Path, gif: Path | None = None, width: int = 480, fps: int = 10) -> Path:
    """A small GIF of an MP4 (ffmpeg builds one colour palette for the whole clip, then maps every frame to it)."""
    gif = Path(gif) if gif else Path(mp4).with_suffix(".gif")
    vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, str(gif)], check=True)
    return gif


def _u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------------ city
# the storyboard, in seconds. Change the numbers, not the code.
CITY_T = dict(
    base=(0.0, 1.2),  # the muted base map fades in from black
    head=(0.3, 1.5),  # header and title
    open=((1.5, 3.1), (3.5, 5.1), (5.5, 7.1)),  # land use, public realm, morphology: one lens at a time
    legend=(7.3, 8.1),  # legend, north arrow, scale bar, credits
    leaders_out=(10.0, 10.5),  # 8.1 - 10.0 s: the finished poster stands still; then the leader lines let go
    legs=((10.6, 13.2), (14.1, 16.7), (17.6, 20.2)),  # three moves round the triangle, a rest after each
    leaders_in=(20.3, 20.9),  # home again: the leaders return
    end=24.0,  # hold the poster to the end
)


def animate_city(cfg: dict, L: dict, out: Path, scale: float = 0.72, fps: int = 30, stills_at=(), T: dict | None = None,
                 log=print, layers: dict | None = None) -> dict:
    """The animated city poster. ``cfg`` = a city configuration, ``L`` = its layers (``read_raw`` / ``read_derived``).

    Returns dict(path, frames, seconds, size, stills={second: uint8 RGB frame}). The layers must cover the convex hull
    of the three lenses (the reduced ``zone="hull"`` files do), because the lenses slide along its edges."""
    from . import city

    T = {**CITY_T, **(T or {})}
    size = video_size(scale)
    sx, sy = size[0] / PW, size[1] / PH
    t0 = time.time()
    counts = city.count(cfg, L)
    layers = {k: shrink(v, size) for k, v in (layers or city.render_layers(cfg, L)).items()}  # step 1: the four maps, once
    full = city.make_lenses(cfg, {k: None for k in ("use", "pub", "mor")})  # poster-pixel lens geometry for the labels
    home = [(Ln.cx * sx, Ln.cy * sy) for Ln in full]
    R = cfg["radius"] * sx
    keys = ("use", "pub", "mor")
    # step 2: the furniture, each part once
    sp_head = overlay(lambda f, a: city.furniture(f, a, cfg, full, counts, parts=("head",)), size)
    sp_rest = overlay(lambda f, a: city.furniture(f, a, cfg, full, counts, parts=("legend", "foot")), size)
    sp_block = [overlay(lambda f, a, k=k: city.furniture(f, a, cfg, full, counts, ("labels",), (k,), leaders=False), size) for k in keys]
    sp_lead = [overlay(lambda f, a, k=k: city.furniture(f, a, cfg, full, counts, ("labels",), (k,), blocks=False), size) for k in keys]
    log(f"layers and furniture ready in {time.time() - t0:.0f} s; video {size[0]} x {size[1]} px", flush=True)

    last: list = [None, None]  # (state, picture) of the previous frame

    def frame(t: float) -> np.ndarray:
        # everything that moves depends on these few numbers; if none changed, the previous picture is still right
        state = tuple(ramp(t, a, b) for a, b in (T["base"], T["head"], *T["open"], T["legend"], T["leaders_out"], *T["legs"], T["leaders_in"]))
        if state == last[0]:
            return last[1]
        # where each lens is: it has walked `walk` legs round the triangle (0 = home, 3 = home again)
        walk = sum(smooth(ramp(t, a, b)) for a, b in T["legs"])
        k, frac = int(min(walk, 2.999)), walk - int(min(walk, 2.999))
        lenses, opens = [], []
        for j in range(3):
            o = ramp(t, *T["open"][j])
            opens.append(o)
            if o <= 0:
                continue
            (xa, ya), (xb, yb) = home[(j + k) % 3], home[(j + k + 1) % 3]
            cx, cy = xa + (xb - xa) * frac, ya + (yb - ya) * frac
            # a lens opens by growing (fast, then slow) while its picture fades in over the first 40 %
            lenses.append(lc.Lens(layers[keys[j]], cx, cy, R * ease_out(o), city.RIMS[j], None, 0.5, alpha=min(1.0, o / 0.4)))
        img = lc.composite(layers["base"] * smooth(ramp(t, *T["base"])), lenses)  # step 3: the poster's own lens rule
        for Ln in lenses:
            rim(img, Ln.cx, Ln.cy, Ln.r, Ln.rim, 3.0 * 100 / 72 * sx, Ln.alpha)
        paste(img, sp_head, smooth(ramp(t, *T["head"])))
        lead = (1 - ramp(t, *T["leaders_out"])) + ramp(t, *T["leaders_in"])  # leaders hide while the lenses travel
        for j in range(3):
            paste(img, sp_block[j], smooth(ramp(opens[j], 0.3, 0.9)))
            paste(img, sp_lead[j], smooth(ramp(opens[j], 0.8, 1.0)) * lead)
        paste(img, sp_rest, smooth(ramp(t, *T["legend"])))
        last[:] = [state, img]
        return img

    return _run(frame, out, size, fps, T["end"], stills_at, log)


def _run(frame, out: Path, size, fps: int, seconds: float, stills_at, log) -> dict:
    """Call ``frame(t)`` for every frame time, pipe the frames to ffmpeg, keep the asked stills."""
    n = int(round(seconds * fps))
    want = {int(round(s * fps)): s for s in stills_at}
    stills, w, t0 = {}, Writer(out, size, fps), time.time()
    for i in range(n):
        img = frame(i / fps)
        w.add(img)
        if i in want:
            stills[want[i]] = _u8(img)
        if (i + 1) % (fps * 5) == 0:
            log(f"  {i + 1}/{n} frames, {time.time() - t0:.0f} s", flush=True)
    w.close()
    log(f"movie -> {out}  ({n} frames, {seconds:.1f} s, {Path(out).stat().st_size / 1e6:.2f} MB)", flush=True)
    return dict(path=Path(out), frames=n, seconds=seconds, size=size, stills=stills)


# --------------------------------------------------------------------------------------------------------- weather
WEATHER_T = dict(
    base=(0.0, 1.2), head=(0.3, 1.5),
    open=((1.5, 3.0), (3.3, 4.8), (5.1, 6.6)),  # temperature, moisture, pressure & wind
    rest=(6.8, 7.6),  # scale bar, note, credits
    run=8.6,  # the clock starts
    step=0.8,  # seconds per analysis hour
    fade=0.27,  # the last part of each hour: cross-fade to the next hour (lines would double in a long fade)
    tail=2.4,  # hold the last hour
)


def animate_weather(data, out: Path, scale: float = 0.72, fps: int = 30, hours=None, stills_at=(), T: dict | None = None,
                    log=print) -> dict:
    """The animated weather poster. ``data`` = (sfc, pl, times) from ``gismaps.lens.weather.load`` / ``load_reduced``.

    The map is the poster's own map (Lambert conformal about the low at 00 UTC 2 November) and the lenses stay where
    the poster has them; the fields step hour by hour and the tracked low moves across the lenses.
    ``hours`` = the analysis hours to show (default: every hour the 850-hPa data hold)."""
    from . import weather as lw

    T = {**WEATHER_T, **(T or {})}
    sfc, pl, _ = data
    size = video_size(scale)
    sx, sy = size[0] / PW, size[1] / PH
    t0 = time.time()
    ts = sfc.time.values
    hours = [np.datetime64(h, "ns") for h in (hours if hours is not None else pl.time.values)]
    trk_all = lw.track_low(sfc.mean_sea_level_pressure, int(np.where(ts == np.datetime64(lw.SEED_TIME))[0][0]))
    at = {t: trk_all[i] for i, t in enumerate(ts)}  # time -> (lon, lat, hPa) of the tracked low
    lon0, lat0, _p = at[np.datetime64("2023-11-02T00:00", "ns")]
    g = lw.poster_grid(lon0, lat0)  # the poster's grid: the 00 UTC frame of the movie IS the poster
    rings = lw.land_polys(g, pad_deg=10)
    base = shrink(lw.poster_base(g, rings), size)
    cx0, cy0 = (float(v) for v in g.ll2px(lon0, lat0))
    cen = lw.lens_centres(cx0, cy0)
    R = lw.LENS_R * sx
    keys = ("t", "q", "p")
    low_px = [tuple(float(v) for v in g.ll2px(at[h][0], at[h][1])) for h in hours]  # the low, poster pixels, per hour
    first, last = (str(h)[:13] for h in (hours[0], hours[-1]))
    span = f"{first[11:13]} UTC {int(first[8:10])} Nov – {last[11:13]} UTC {int(last[8:10])} Nov 2023"
    note = f"Lambert conformal map; north is up along {abs(lon0):.2f}°{'W' if lon0 < 0 else 'E'}"

    def F(parts, **kw):
        return overlay(lambda f, a: lw.furniture(f, a, g, cen, parts=parts, when=span, note=note, **kw), size)

    sp_head, sp_rest = F(("head",)), F(("scale", "foot"))
    sp_block = [F(("labels",), keys=(k,), leaders=False) for k in keys]
    sp_lead = [F(("labels",), keys=(k,), blocks=False) for k in keys]
    sp_stamp = [F(("stamp",), stamp=lw.stamp_text(h)) for h in hours]

    def marker(p):  # the L and its pressure, drawn at the middle of a small canvas -> a sprite centred on (0, 0)
        fig, ax = lc.canvas(400, 300)
        lw.low_marker(ax, 200, 120, p)
        sp = sprite(shrink(lc.grab(fig), (2 * round(200 * sx), 2 * round(150 * sy))))
        sp.x -= round(200 * sx)
        sp.y -= round(120 * sy)
        return sp

    sp_low = [marker(at[h][2]) for h in hours]

    def track(n):  # the path of the low up to hour n: a thin line and one dot per hour
        def draw(_f, ax):
            xy = np.array(low_px[:n + 1])
            ax.plot(xy[:, 0], xy[:, 1], color="#0b1020", lw=5.5, alpha=0.7, zorder=4, solid_capstyle="round")  # dark casing
            ax.plot(xy[:, 0], xy[:, 1], color="#f4ecd2", lw=2.6, zorder=5, solid_capstyle="round")
            ax.scatter(xy[:, 0], xy[:, 1], s=64, c="#f4ecd2", edgecolors="#0b1020", linewidths=1.6, zorder=6)
        return overlay(draw, size)

    sp_track = [track(n) for n in range(len(hours))]
    sp_key = overlay(lambda _f, ax: lc.text(ax, 1330, 2412, "L: the tracked low · line and dots: its path, one dot per hour", 22,
                                            lw.MUT, lc.SERIF, style="italic"), size)
    log(f"map and furniture ready in {time.time() - t0:.0f} s; video {size[0]} x {size[1]} px", flush=True)

    cache: dict[int, dict] = {}

    def hour(n: int) -> dict:  # the lens images of hour n, drawn when first needed; only two hours are kept
        if n not in cache:
            for old in [k for k in cache if k < n - 1]:
                del cache[old]
            lay = lw.hour_layers(g, rings, sfc, pl, hours[n])
            cache[n] = {k: shrink(v, size) for k, v in lay.items()}
            log(f"  fields of {str(hours[n])[:13]} UTC drawn ({time.time() - t0:.0f} s)", flush=True)
        return cache[n]

    def picture(n: int, opens, base_a: float) -> np.ndarray:
        lay = hour(n)
        lenses = []
        for j, Ln in enumerate(lw.make_lenses(lay, {k: (cen[k][0] * sx, cen[k][1] * sy) for k in keys}, R)):
            if opens[j] > 0:
                Ln.r, Ln.alpha = R * ease_out(opens[j]), min(1.0, opens[j] / 0.4)
                lenses.append(Ln)
        img = lc.composite(base * base_a, lenses)
        for Ln in lenses:
            rim(img, Ln.cx, Ln.cy, Ln.r, Ln.rim, 3.2 * 100 / 72 * sx, Ln.alpha)
        return img

    n_h = len(hours)
    seconds = T["run"] + (n_h - 1) * T["step"] + T["tail"]

    last: list = [None, None]  # (state, picture) of the previous frame

    def frame(t: float) -> np.ndarray:
        # one clock for everything: which hour is on screen (n) and how far the cross-fade to the next has got (m)
        c = np.clip((t - T["run"]) / T["step"], 0, n_h - 1)
        n = int(min(c, n_h - 1))
        m = 0.0 if n == n_h - 1 else smooth(ramp(c - n, 1 - T["fade"] / T["step"], 1.0))
        opens = [ramp(t, *T["open"][j]) for j in range(3)]
        state = (n, m, *opens, ramp(t, *T["base"]), ramp(t, *T["head"]), ramp(t, *T["rest"]))
        if state == last[0]:  # an hour that is being held: nothing moved, the previous picture is still right
            return last[1]
        base_a = smooth(ramp(t, *T["base"]))
        img = picture(n, opens, base_a)
        if m > 0:
            img = img * (1 - m) + picture(n + 1, opens, base_a) * m
        paste(img, sp_head, smooth(ramp(t, *T["head"])))
        for j in range(3):
            paste(img, sp_block[j], smooth(ramp(opens[j], 0.3, 0.9)))
            paste(img, sp_lead[j], smooth(ramp(opens[j], 0.8, 1.0)))
        rest = smooth(ramp(t, *T["rest"]))
        paste(img, sp_rest, rest)
        paste(img, sp_key, rest)
        show = smooth(ramp(t, *T["head"]))
        paste(img, sp_stamp[n], show * (1 - m))
        paste(img, sp_track[n], rest)
        low = smooth(ramp(opens[0], 0.6, 1.0))  # the L arrives with the first lens
        (xa, ya), (xb, yb) = low_px[n], low_px[min(n + 1, n_h - 1)]
        if m > 0:
            paste(img, sp_stamp[n + 1], show * m)
            paste(img, sp_low[n + 1], low * m, (xa + (xb - xa) * m) * sx, (ya + (yb - ya) * m) * sy)
        paste(img, sp_low[n], low * (1 - m), (xa + (xb - xa) * m) * sx, (ya + (yb - ya) * m) * sy)
        last[:] = [state, img]
        return img

    res = _run(frame, out, size, fps, seconds, stills_at, log)
    res["track"] = [(str(h)[:13], *at[h]) for h in hours]
    return res


# ------------------------------------------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Animated lens-map posters (MP4, optional GIF)")
    sub = ap.add_subparsers(dest="what", required=True)
    c = sub.add_parser("city")
    c.add_argument("--city", default="brisbane")
    c.add_argument("--derived", default="", help="reduced <city>_lens_layers.json.gz (default: the copy shipped with the repository)")
    w = sub.add_parser("weather")
    w.add_argument("--reduced", default="", help="the small 0.5-degree ERA5 file (default: the copy shipped with the repository)")
    for p in (c, w):
        p.add_argument("--out", default="")
        p.add_argument("--scale", type=float, default=0.72)
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--gif", action="store_true")
    a = ap.parse_args()
    if a.what == "city":
        from . import city
        from .cities import city as city_cfg

        cfg = city_cfg(a.city)
        pr = city.Proj(cfg)
        L = city.read_derived(cfg, pr, a.derived or None)
        res = animate_city(cfg, L, Path(a.out or city.OUT / f"lens_{a.city}_anim.mp4"), a.scale, a.fps)
    else:
        from . import weather as lw

        from .. import data as _data

        data = lw.load_reduced(a.reduced or _data.path("lens_maps", "era5_ciaran_lens_0p5deg.nc"))
        for name, var in (("ne_50m_land.zip", "LAND"), ("etopo2022_10min_natl.nc", "ETOPO")):
            if not Path(getattr(lw, var)).exists():
                if var == "LAND":
                    lw.fetch_land(lw.ROOT / "cache")
                else:
                    lw.ETOPO = _data.path("lens_maps", name)
        res = animate_weather(data, Path(a.out or lw.OUT / "lens_weather_anim.mp4"), a.scale, a.fps)
    if a.gif:
        print("gif ->", to_gif(res["path"]))


if __name__ == "__main__":
    main()
