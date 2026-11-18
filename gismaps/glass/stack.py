"""gismaps.glass.stack: the finished glass-pane stack (poster and animation).

``GlassStack`` holds the data of one forecast run, the coastline and the globe's skin. ``poster`` renders the fanned
stack at one lead time and adds the frame furniture: a top strap, a legend column (one numbered row per pane, the
same number on a chip at the pane's corner), the lead time with the valid time in UTC and local time, a timeline and
the credits. ``animate`` writes the 16-second film: the globe glides in, the panes fan out, one pane is lifted, the
forecast clock runs, and the panes close up on the globe.

All sizes are written for a frame of 1920 x 1080 and scale with the frame height.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..styles import common as C
from . import gfs, panes as P_, render as R
from .panes import K

DURATION = 16.0                                  # seconds; the keyframe times in ``film_state`` are written for this


# ---------------------------------------------------------------- small helpers
def valid_label(run: str, lead_h: float, utc_offset: float = 0.0, zone: str = "UTC") -> str:
    """Valid time as text, e.g. 'Thu 19 Nov · 12:00 UTC' (local time with ``utc_offset`` hours and a zone name)."""
    t = dt.datetime.fromisoformat(str(run)[:16]) + dt.timedelta(hours=float(lead_h) + utc_offset)
    return f"{t:%a} {t.day} {t:%b} · {t:%H:%M} {zone}"


def num(x: float, digits: int = 0) -> str:
    """A number as text with a true minus sign."""
    return f"{x:.{digits}f}".replace("-", "−")


def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def ease_in_out(x: float) -> float:
    """Cubic ease-in-out: slow start, fast middle, slow end."""
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x: float) -> float:
    """Cubic ease-out: fast start, gentle landing (for things that appear)."""
    return 1 - (1 - clamp01(x)) ** 3


def move(t: float, t0: float, t1: float) -> float:
    """Eased progress of a move that runs from time t0 to time t1 (seconds): 0 before, 1 after."""
    return ease_in_out((t - t0) / (t1 - t0))


def mix(a, b, w):
    """Blend two numbers or tuples: w = 0 gives a, w = 1 gives b."""
    if isinstance(a, tuple):
        return tuple(x + (y - x) * w for x, y in zip(a, b))
    return a + (b - a) * w


def cam_mix(A: dict, B: dict, w: float) -> dict:
    """Blend two camera descriptions, number by number."""
    return {k: mix(A[k], B[k], w) for k in A}


# ---------------------------------------------------------------- frame furniture (pixel coordinates, y down)
def canvas(img: np.ndarray):
    """A transparent figure of exactly the image's size with axes in pixel coordinates; ``to_image`` lays what was
    drawn on it over the image."""
    H, W = img.shape[:2]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.patch.set_alpha(0.0)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig._base = img
    return fig, ax


def to_image(fig) -> np.ndarray:
    """Render the figure's drawings and lay them over its base image. Returns RGB (0..255) and closes the figure."""
    ax = fig.axes[0]
    H, W = fig._base.shape[:2]
    ax.set_xlim(0, W)                                                  # (imshow moves the limits)
    ax.set_ylim(H, 0)
    fig.canvas.draw()
    top = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32)
    plt.close(fig)
    a = top[..., 3:4] / 255.0
    return (fig._base[..., :3] * (1.0 - a) + top[..., :3] * a).astype(np.uint8)


def _right_edge(ax, t) -> float:
    """The x pixel where a text ends."""
    return t.get_window_extent(renderer=ax.figure.canvas.get_renderer()).x1


def colour_bar(ax, x, y, w, h, P, k=1.0):
    """A horizontal colour bar with end values and unit for pane ``P``."""
    ax.imshow(np.linspace(0, 1, 128)[None, :], extent=(x, x + w, y + h, y), cmap=P["cmap"], aspect="auto", zorder=5)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=K["rule"], lw=1.2 * k, zorder=6))
    lo = num(P["vmin"], 1 if abs(P["vmin"]) < 1 and P["vmin"] != 0 else 0)
    if P["note"]:                                                      # a short note under the bar ("isobars every 4 hPa")
        C.text(ax, x, y + h + 22 * k, P["note"], 22 * k, K["muted"], va="center", zorder=6)
    C.text(ax, x - 10 * k, y + h / 2, lo, 26 * k, K["muted"], ha="right", va="center", zorder=6)
    C.text(ax, x + w + 10 * k, y + h / 2, f"{num(P['vmax'])} {P['unit']}", 26 * k, K["muted"], va="center", zorder=6)
    if P["vmin"] < 0 < P["vmax"]:                                      # a two-sided scale: mark the zero
        x0 = x + w * (0 - P["vmin"]) / (P["vmax"] - P["vmin"])
        ax.plot([x0, x0], [y - 4 * k, y + h + 4 * k], color=K["ink"], lw=2 * k, zorder=7)


def timeline(ax, x, y, w, run, lead, max_h=120, k=1.0):
    """A timeline from the start of the run to +max_h hours with a mark at every 00 UTC and a playhead at ``lead``."""
    ax.plot([x, x + w], [y, y], color=K["rule"], lw=6 * k, solid_capstyle="round", zorder=5)
    ax.plot([x, x + w * lead / max_h], [y, y], color=K["amber"], lw=6 * k, solid_capstyle="round", zorder=6)
    t0 = dt.datetime.fromisoformat(str(run)[:16])
    for h in range((24 - t0.hour) % 24, max_h + 1, 24):
        xx = x + w * h / max_h
        ax.plot([xx, xx], [y - 12 * k, y + 12 * k], color=K["muted"], lw=2 * k, zorder=6)
        day = t0 + dt.timedelta(hours=h)
        C.text(ax, xx + 8 * k, y + 30 * k, f"{day:%a} {day.day}", 26 * k, K["muted"], va="center", zorder=6)
    ax.plot([x + w * lead / max_h], [y], marker="o", ms=16 * k, color=K["amber"], mec=K["bg2"], mew=3 * k, zorder=7)


def chrome(ax, W, H, *, title, tag, run, lead, local=(5.5, "IST"), credit2="", max_h=120):
    """Top strap, bottom band with the clock, the timeline and the credits."""
    k = H / 1080.0
    ax.add_patch(plt.Rectangle((0, 0), W, 92 * k, fc=K["bg2"], ec="none", alpha=0.92, zorder=4))     # top strap
    ax.add_patch(plt.Rectangle((0, 88 * k), W, 4 * k, fc=K["amber"], ec="none", zorder=5))           # its amber rule
    t = C.text(ax, 48 * k, 46 * k, f" {tag} ", 34 * k, K["bg2"], weight="bold", va="center", zorder=6,
               bbox=dict(boxstyle="round,pad=0.25", fc=K["amber"], ec="none"))
    C.text(ax, _right_edge(ax, t) + 30 * k, 46 * k, title, 40 * k, K["ink"], weight="bold", va="center", zorder=6)
    C.text(ax, W - 48 * k, 46 * k, "MODEL GUIDANCE · not an official warning", 27 * k, K["muted"], weight="bold", va="center",
           ha="right", zorder=6)
    ax.add_patch(plt.Rectangle((0, 906 * k), W, 174 * k, fc=K["bg2"], ec="none", alpha=0.94, zorder=4))    # bottom band
    ax.add_patch(plt.Rectangle((0, 906 * k), W, 3 * k, fc=K["rule"], ec="none", zorder=5))
    C.text(ax, 48 * k, 950 * k, f"+{int(round(lead))} h", 54 * k, K["amber"], weight="bold", va="center", zorder=6)
    xc = 298 * k                                                       # a fixed column: the date does not jump as digits change
    C.text(ax, xc, 936 * k, valid_label(run, round(lead)), 36 * k, K["ink"], weight="bold", va="center", zorder=6)
    C.text(ax, xc, 978 * k, valid_label(run, round(lead), local[0], local[1]), 30 * k, K["ice"], va="center", zorder=6)
    timeline(ax, 800 * k, 946 * k, 1060 * k, run, lead, max_h=max_h, k=k)
    C.text(ax, 48 * k, 1028 * k, gfs.CREDIT, 24 * k, K["muted"], va="center", zorder=6)
    C.text(ax, W - 48 * k, 1028 * k, C.AUTHOR, 26 * k, K["ink"], weight="bold", va="center", ha="right", zorder=6)
    if credit2:
        c2 = C.text(ax, 48 * k, 1058 * k, credit2, 24 * k, K["muted"], va="center", zorder=6)
        width = _right_edge(ax, c2) - 48 * k
        if width > W - 96 * k:                                         # too long for the frame: a little smaller
            c2.set_fontsize(c2.get_fontsize() * (W - 96 * k) / width)


def legend_column(ax, cam, panes, W, H, *, rad_deg, gap, state=None, active=None, x0=1262, half=1.16, chips=True):
    """One row per pane (number, name, colour bar, unit) on a plate at the right; the same number sits on a chip at
    each pane's upper right corner (found with ``render.project``, so it stays attached when the camera moves)."""
    k = H / 1080.0
    s = np.sin(np.radians(rad_deg))
    z0 = R.first_pane_z(rad_deg)
    state = state or [{} for _ in panes]
    ax.add_patch(plt.Rectangle((1228 * k, 92 * k), W - 1228 * k, 814 * k, fc=K["bg2"], ec="none", alpha=0.80, zorder=4))
    row_h = min(122.0, 760.0 / max(len(panes), 1)) * k
    for i, P in enumerate(panes):
        st = state[i] if i < len(state) else {}
        y = 150 * k + i * row_h
        on = (active is None) or (active == i)                         # the pane being shown alone is brighter
        ax.add_patch(plt.Rectangle((x0 * k, y - 22 * k), 6 * k, 84 * k, fc=K["amber"] if active == i else K["rule"], ec="none", zorder=6))
        chip = dict(boxstyle="circle,pad=0.22", fc=K["amber"] if on else K["rule"], ec=K["bg2"], lw=1.5 * k)
        C.text(ax, (x0 + 44) * k, y, str(i + 1), 27 * k, K["bg2"], weight="bold", ha="center", va="center", zorder=7, bbox=chip)
        C.text(ax, (x0 + 84) * k, y, P["title"], 36 * k, K["ink"] if on else K["muted"], weight="bold", va="center", zorder=6)
        colour_bar(ax, (x0 + 84) * k, y + 34 * k, 250 * k, 20 * k, P, k)
        if chips and st.get("o", 1.0) > 0.25:                          # the same number on the pane's upper right corner
            px, py = R.project(cam, R.pane_point(cam, z0 + i * gap, st.get("lift", 0.0), 0.93 * half * s, 1.0 * half * s))
            C.text(ax, px, py, str(i + 1), 25 * k, K["bg2"], weight="bold", ha="center", va="center", zorder=7,
                   alpha=1.0 if on else 0.55, bbox=dict(chip, alpha=1.0 if on else 0.55))


# ---------------------------------------------------------------- the stack
class GlassStack:
    """The glass-pane stack of one forecast run over one round cap of the Earth.

    Parameters
    ----------
    ds : xarray.Dataset
        The six fields from ``gfs.region`` (dims lead, latitude, longitude; attribute ``run``).
    cap : dict
        ``dict(lat0=..., lon0=..., rad_deg=...)``: centre and angular radius of the cap every pane shows.
    place : str
        The name of the region, for the titles.
    local : tuple
        Local clock shown under UTC: (hours ahead of UTC, name).
    coast, elev, skin : optional
        The coastline (``panes.coast_lines``), the relief heights (``gfs.relief``) and the globe's skin
        (``panes.relief_texture``) if they are already made; otherwise they are made here.
    """

    def __init__(self, ds, cap=None, place="India and Bangladesh", local=(5.5, "IST"), coast=None, elev=None, skin=None, log=print):
        self.ds = ds
        self.run = str(ds.attrs["run"])[:16]
        self.leads = [int(h) for h in ds["lead"].values]
        self.cap = cap or dict(lat0=21.0, lon0=87.0, rad_deg=21.0)
        self.place, self.local = place, local
        self.coast = P_.coast_lines("50m") if coast is None else coast
        self.elev = gfs.relief(log) if elev is None else elev
        self.relief = P_.relief_texture(self.elev, self.coast) if skin is None else skin    # the globe's skin
        self.rad = float(np.sin(np.radians(self.cap["rad_deg"])))      # disc radius of the cap (globe radius = 1)
        self.gap = 0.9 * self.rad                                      # spacing of the fanned panes along the axis
        self.lifted = 3                                                # the pane the film lifts out (precipitable water)
        self._film = None
        self.credit2 = (f"{self.elev['credit']} (shading exaggerated) · coastlines: Natural Earth · "
                        "panes are not at physical heights")

    # -- the pieces
    def panes(self, lead: int, n: int = 480) -> list[dict]:
        """The six pane pictures of one lead time (hours), n x n pixels each."""
        return P_.six_panes(P_.fields(self.ds, lead), self.cap, self.coast, n=n)

    def fan_camera(self, size=(1920, 1080), az=-42.0, el=15.0, pp=(620, 510)) -> dict:
        """The fanned-stack view, seen from the west side so that the panes fan out to the right of the globe."""
        k = size[1] / 1080.0
        return R.camera(self.cap["lat0"], self.cap["lon0"], az=az, el=el, dist=14.0 * self.rad,
                        look=(0.6 * self.rad, 0.04, 1.10 + 2.3 * self.gap), fov=30.0, size=size, pp=(pp[0] * k, pp[1] * k))

    def render(self, panes, cam, gap=None, state=None, ss=2, glass_o=1.0) -> np.ndarray:
        """The bare 3-D picture: globe and panes, no labels."""
        return R.render_stack(panes, cam, self.relief, rad_deg=self.cap["rad_deg"], gap=self.gap if gap is None else gap,
                              state=state, ss=ss, glass_o=glass_o)

    # -- the poster
    def poster(self, path, lead: int = 66, scale: float = 1.0, title: str | None = None) -> Path:
        """The finished picture at 1920 x 1080 times ``scale``: the fanned stack at ``lead`` hours with legend, clock,
        timeline and credits."""
        from PIL import Image

        size = (int(round(1920 * scale)), int(round(1080 * scale)))
        panes = self.panes(lead, n=int(round(480 * scale)))
        cam = self.fan_camera(size)
        img = self.render(panes, cam, ss=2 if scale <= 1 else 1.5)
        fig, ax = canvas(img)
        legend_column(ax, cam, panes, *size, rad_deg=self.cap["rad_deg"], gap=self.gap)
        c = self.cap
        where = (f"Each pane: the cap of {c['rad_deg']:.0f}° around {abs(c['lat0']):.0f}°{'N' if c['lat0'] >= 0 else 'S'} "
                 f"{abs(c['lon0']):.0f}°{'E' if c['lon0'] >= 0 else 'W'}, seen along the stack axis · ")
        chrome(ax, *size, title=title or f"Six layers of one forecast · {self.place}", tag="THE LAYERS", run=self.run, lead=lead,
               local=self.local, credit2=f"{where}panes are not at physical heights · {self.elev['credit']} (shading exaggerated) · "
                                         "coastlines: Natural Earth")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(to_image(fig)).save(path)
        return path

    # -- the film
    def film_panes(self, lead: float) -> list[dict]:
        """The six panes at any lead time: the pictures are made every 6 hours (360 px) and cross-faded in between,
        so at +27 h a pane shows half of the +24 h picture and half of the +30 h one."""
        if self._film is None:
            self._film = [self.panes(h, n=360) for h in self.leads]
        x = float(np.clip((lead - self.leads[0]) / (self.leads[1] - self.leads[0]), 0, len(self._film) - 1))
        i = int(np.floor(x))
        j, w = min(i + 1, len(self._film) - 1), x - int(np.floor(x))
        return [dict(A, tex=A["tex"] * (1 - w) + B["tex"] * w) for A, B in zip(self._film[i], self._film[j])]

    def film_state(self, t: float) -> dict:
        """Everything one frame needs, as a function of time t (seconds): the camera, the gap between the panes, each
        pane's opacity, lift and glow, the lead time shown, the title. Each number is a blend between keyframe values."""
        rad, gap0, L = self.rad, self.gap, self.lifted
        first, last = float(self.leads[0]), float(self.leads[-1])
        look = (0.6 * rad, 0.04, 1.10 + 2.3 * gap0)
        far = dict(az=-8.0, el=10.0, dist=7.6, look=(0.0, 0.0, 0.0), pp=(640.0, 500.0))                      # the whole globe
        near = dict(az=-10.0, el=8.0, dist=2.2 + 3.4 * rad, look=(0.0, 0.0, 0.75), pp=(640.0, 500.0))        # the region on the globe
        fan_c = dict(az=-42.0, el=15.0, dist=14.0 * rad, look=look, pp=(620.0, 510.0))                       # the fanned stack
        lift_c = dict(az=-38.0, el=13.0, dist=14.0 * rad * 0.96, look=(look[0], look[1] + 0.16 * rad, look[2]), pp=(620.0, 560.0))
        orbit = dict(az=-32.0, el=13.0, dist=14.0 * rad, look=look, pp=(620.0, 510.0))                       # a slow swing
        axis = dict(az=0.0, el=0.0, dist=7.6 * rad + 0.9, look=(0.0, 0.0, 1.12), pp=(640.0, 500.0))          # straight down the axis

        locate = move(t, 0.3, 2.8)                                         # far -> near
        fan = move(t, 3.0, 5.2)                                            # near -> fanned stack
        lift = move(t, 5.7, 6.9) * (1 - move(t, 8.0, 9.0))                 # up, hold, down
        line = move(t, 14.5, 15.8)                                         # fan -> lined up on the axis
        cam = cam_mix(far, near, locate)
        cam = cam_mix(cam, fan_c, fan)
        cam = cam_mix(cam, lift_c, lift)
        cam = cam_mix(cam, orbit, move(t, 9.2, 14.3))
        cam = cam_mix(cam, axis, line)
        gap = mix(mix(0.02 * rad, gap0, fan), 0.004, line)                 # the panes spread, then close up
        lead = mix(first, last, (t - 9.2) / (14.3 - 9.2)) if 9.2 < t < 14.3 else (first if t <= 9.2 else last)   # constant speed: a clock
        state = []
        for k in range(6):
            appear = ease_out((t - 3.0 - 0.12 * k) / 0.9)                  # the panes appear one after the other
            dim = 1.0 - 0.72 * lift * (k != L)                             # the others dim while one is lifted
            keep = {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.55, 5: 1.0}[k]    # lined up: pressure, some cloud and rain stay
            state.append({"o": appear * dim * mix(1.0, keep, line), "lift": 0.85 * rad * lift if k == L else 0.0,
                          "glow": lift if k == L else 0.0})
        names = ["temperature", "sea-level pressure", "wind", "precipitable water", "cloud cover", "rain rate"]
        title = (f"Where we are: {self.place}" if t < 3.0 else "One forecast, six layers" if t < 5.6 else
                 f"One layer: {names[L]}" if t < 9.1 else "The layers move together" if t < 14.5 else "It is one system")
        return {"cam": cam, "gap": gap, "lead": lead, "state": state, "legend_o": fan, "glass_o": 1.0 - line,
                "active": L if lift > 0.5 else None, "title": title, "chips": line < 0.05 and fan > 0.95}

    def frame(self, t: float, path, size=(1280, 720), ss: float = 1.5):
        """Draw the frame of the film at time t (seconds) and save it. Returns at once when ``path`` is ``None``."""
        if path is None:
            return None
        from PIL import Image

        S = self.film_state(t)
        W, H = size
        k = H / 1080.0                                                     # camera pixel numbers are written for 1080 rows
        c = S["cam"]
        cam = R.camera(self.cap["lat0"], self.cap["lon0"], az=c["az"], el=c["el"], dist=c["dist"], look=c["look"], fov=30.0,
                       size=size, pp=(c["pp"][0] * k, c["pp"][1] * k))
        panes = self.film_panes(S["lead"])
        img = self.render(panes, cam, gap=S["gap"], state=S["state"], ss=ss, glass_o=S["glass_o"])
        kw = dict(rad_deg=self.cap["rad_deg"], gap=S["gap"], state=S["state"], active=S["active"])
        fig, ax = canvas(img)
        chrome(ax, W, H, title=S["title"], tag="THE FORECAST", run=self.run, lead=S["lead"], local=self.local, credit2=self.credit2)
        if S["legend_o"] >= 0.999:                                         # legend fully there: draw it in the same pass
            legend_column(ax, cam, panes, W, H, chips=S["chips"], **kw)
            out = to_image(fig)
        else:
            out = to_image(fig)
            if S["legend_o"] > 0.001:                                      # legend fading in: draw it on a copy and blend
                fig, ax = canvas(out)
                legend_column(ax, cam, panes, W, H, chips=False, **kw)
                out = (out * (1 - S["legend_o"]) + to_image(fig) * S["legend_o"]).astype(np.uint8)
        Image.fromarray(out).save(path, compress_level=1)
        return Path(path)

    def animate(self, out, frames_dir, fps: int = 24, size=(1280, 720), ss: float = 1.5) -> Path | None:
        """Write the whole film (16 s) as an MP4. Inside ``common.preview`` the same call makes a short, smaller clip."""
        if C._PREVIEW:                                                     # a preview: smaller frames, no supersampling
            size = tuple(int(round(v * C._PREVIEW["scale"] / 2)) * 2 for v in size)
            ss = 1
        mv = C.Movie(frames_dir, fps=fps)
        for i in range(int(round(DURATION * fps))):
            self.frame(i / fps, mv.path(), size=size, ss=ss)
        return mv.encode(out, crf=20)
