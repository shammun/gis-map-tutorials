"""Lens map: shared pieces of the city and weather posters.

* ``canvas`` / ``grab``   a matplotlib figure whose data coordinates ARE output pixels (x right, y down)
* ``composite``           numpy lens compositing (the lens rule: a lens is a mask)
* poster furniture        label blocks with a colour bar + leader, rims, rules (all in poster pixels)

Lens semantics:
  lens j is a disc (cx, cy, r). Its opaque ``fill`` layer is drawn at alpha 1 where it lies on the base map only and at
  ``overlap`` (0..1) where it lies on an earlier lens; its transparent ``over`` layer (line work) is drawn at full
  strength everywhere inside the disc. overlap = 0.5 gives the blended look of the city poster; overlap = 0 keeps the
  lower lens's colours exact (they still match their colour bar) and lays the upper lens's lines on top.

Style after Fan Mo, #30DayMapChallenge 2026 Day 14 ("Borgesian map"); the code and data are original.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIF = "Georgia"
SANS = "Segoe UI"
FONT_SCALE = 1.0  # text size factor: 1 for Georgia / Segoe UI; smaller for the wider DejaVu fonts, so lines still fit


def use_fonts(serif: str | None = None, sans: str | None = None) -> tuple[str, str]:
    """Choose the two poster fonts. Without arguments: Georgia and Segoe UI where they are installed (Windows), else
    DejaVu Serif and DejaVu Sans, which ship with matplotlib (Linux, Google Colab). ``LENS_FONTS=dejavu`` forces the
    DejaVu pair (to preview on Windows what a Colab run looks like). Returns (serif, sans)."""
    global SERIF, SANS, FONT_SCALE
    import os

    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    forced = os.environ.get("LENS_FONTS", "").lower() == "dejavu"
    SERIF = serif or ("Georgia" if "Georgia" in have and not forced else "DejaVu Serif")
    SANS = sans or ("Segoe UI" if "Segoe UI" in have and not forced else "DejaVu Sans")
    FONT_SCALE = 0.86 if SANS.startswith("DejaVu") else 1.0
    return SERIF, SANS


use_fonts()
AUTHOR = "Map created by Shammunul Islam"  # creator line of every poster credit block


def canvas(w: int, h: int, bg=(0, 0, 0, 0)):
    """Figure of exactly w x h px with one axes in pixel coordinates (origin top-left, y down)."""
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def grab(fig) -> np.ndarray:
    """RGBA float array (h, w, 4) in 0..1 of a figure, then close it."""
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32) / 255.0
    plt.close(fig)
    return a


@dataclass
class Lens:
    fill: np.ndarray  # (h, w, 3|4) opaque layer
    cx: float
    cy: float
    r: float
    rim: str
    over: np.ndarray | None = None  # (h, w, 4) line work, straight alpha
    overlap: float = 0.5
    alpha: float = 1.0  # < 1 while a lens fades in (animations); 1 = the poster


def disc(h: int, w: int, cx: float, cy: float, r: float, x0: int = 0, y0: int = 0) -> np.ndarray:
    """Antialiased disc mask (1 inside), 1-px soft edge. The mask covers the h x w pixels whose top-left pixel is
    (x0, y0) of the whole picture (default: the whole picture)."""
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32)
    d = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
    return np.clip(r - d + 0.5, 0, 1)


def composite(base: np.ndarray, lenses: list[Lens]) -> np.ndarray:
    """Base RGB + lenses in order -> RGB (h, w, 3).

    Each lens only touches the square that holds its disc: outside the disc its mask is exactly 0 and the picture
    stays as it was, so working on the square alone gives the same numbers as working on the whole picture, faster."""
    out = base[..., :3].astype(np.float32).copy()
    h, w = out.shape[:2]

    def box(L):  # the pixel square around a lens, clipped to the picture: rows y0..y1, columns x0..x1
        return (max(int(L.cy - L.r) - 2, 0), min(int(L.cy + L.r) + 3, h), max(int(L.cx - L.r) - 2, 0), min(int(L.cx + L.r) + 3, w))

    for j, L in enumerate(lenses):
        y0, y1, x0, x1 = box(L)
        if y1 <= y0 or x1 <= x0:
            continue
        a = disc(y1 - y0, x1 - x0, L.cx, L.cy, L.r, x0, y0)
        for i in range(j):  # where this lens lies on an earlier one, its fill is drawn at ``overlap`` only
            a *= 1 - (1 - L.overlap) * disc(y1 - y0, x1 - x0, lenses[i].cx, lenses[i].cy, lenses[i].r, x0, y0)
        if L.alpha != 1.0:
            a *= L.alpha
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a[..., None]) + L.fill[y0:y1, x0:x1, :3] * a[..., None]
    for j, L in enumerate(lenses):  # line work of every lens on top of every fill
        if L.over is not None:
            y0, y1, x0, x1 = box(L)
            if y1 <= y0 or x1 <= x0:
                continue
            a = (L.over[y0:y1, x0:x1, 3] * disc(y1 - y0, x1 - x0, L.cx, L.cy, L.r, x0, y0) * L.alpha)[..., None]
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + L.over[y0:y1, x0:x1, :3] * a
    return np.clip(out, 0, 1)


def rims(ax, lenses: list[Lens], lw: float = 3.0, glow: bool = True) -> None:
    for L in lenses:
        if glow:
            for k, al in ((5, 0.10), (3, 0.16)):
                ax.add_patch(plt.Circle((L.cx, L.cy), L.r, fill=False, ec=L.rim, lw=lw * k, alpha=al, zorder=20))
        ax.add_patch(plt.Circle((L.cx, L.cy), L.r, fill=False, ec=L.rim, lw=lw, zorder=21))


def label_block(ax, x: float, y: float, name: str, theme: str, lines: list[str], colour: str, s: float, ink="#f1ede4",
                sub="#c9c5bc") -> float:
    """Coloured vertical bar + CAPS name + italic theme + italic description lines. (x, y) = top-left of the bar.
    ``s`` = cap-name size in px. Returns the y of the block's bottom."""
    px = 72 / 100 * FONT_SCALE  # px -> pt at dpi 100
    y1 = y + s * 0.95
    y2 = y1 + s * 1.12
    yb = y2 + (len(lines) - 0.15) * s * 0.92
    ax.plot([x, x], [y, yb], color=colour, lw=s * 0.2, solid_capstyle="butt", zorder=30)
    tx = x + s * 0.42
    ax.text(tx, y1, name, color=ink, fontsize=s * px, fontfamily=SANS, fontweight="bold", va="baseline", zorder=30)
    ax.text(tx, y2, theme, color=ink, fontsize=s * 0.92 * px, fontfamily=SERIF, fontstyle="italic", va="baseline", zorder=30)
    for i, ln in enumerate(lines):
        ax.text(tx, y2 + (i + 1) * s * 0.92, ln, color=sub, fontsize=s * 0.74 * px, fontfamily=SERIF, fontstyle="italic",
                va="baseline", zorder=30)
    return yb


def leader(ax, p, q, colour="#f1ede4", lw=1.6) -> None:
    ax.plot([p[0], q[0]], [p[1], q[1]], color=colour, lw=lw, solid_capstyle="round", zorder=29)


def text(ax, x, y, t, s, colour="#f1ede4", family=SANS, weight="normal", style="normal", ha="left", va="baseline", **kw):
    return ax.text(x, y, t, color=colour, fontsize=s * 0.72 * FONT_SCALE, fontfamily=family, fontweight=weight, fontstyle=style, ha=ha,
                   va=va, zorder=30, **kw)


def credit_block(ax, x: float, lines: list[str], author: str | None = None, ink="#f1ede4", mut="#8d8a84") -> None:
    """Credit block of a poster footer: optional bold creator line, the DATA line in ink, the rest muted.
    Without ``author`` the lines sit exactly where the first posters had them."""
    if author:
        text(ax, x, 2630, author, 25, ink, weight="bold")
        ys = (2664, 2694, 2722, 2750)
    else:
        ys = (2646, 2678, 2708, 2738)
    for i, (y, ln) in enumerate(zip(ys, lines)):
        text(ax, x, y, ln, 22 if i == 0 else 20, ink if i == 0 else mut)


def hexrgb(c: str) -> np.ndarray:
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def save_rgb(path, a: np.ndarray) -> None:
    from PIL import Image

    arr = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr, "RGBA" if arr.shape[-1] == 4 else "RGB").save(path, optimize=True)
