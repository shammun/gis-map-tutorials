# Lens-map recipe 3 - the clip in which the lenses ride the storm.  Plain script; `tools/build_notebooks.py` turns it into the notebooks.
# Run:  python tutorials/lens_maps/03_lens_map_clip.py        (writes into ./out/)

# %% [markdown]
# # Lens map 3 · The lenses ride the storm (`lens_map.mp4`)
#
# **What this recipe makes.** The 41-second clip of the gallery shows Storm Ciarán on 1-2 November 2023 through three
# round lenses on one dark map of the north-east Atlantic: **temperature** about 1.5 km up, **moisture** there, and
# **sea-level pressure with the wind**. The lenses open one by one, sweep across the map, then follow the tracked low
# while one clock steps all three fields hour by hour.
#
# The clip is made in two parts:
#
# 1. **Python makes every picture** (this notebook, also on Google Colab): one PNG per field and hour on one pixel
#    grid, and a small `manifest.json` with the hours, the track of the low and a coarse wind grid.
# 2. **A video framework moves and masks them** ([Remotion](https://www.remotion.dev), in `remotion/lens_map/` of the
#    repository). **This part needs Node.js on your own machine; it does not run on Colab.** Step 5 gives the commands.
#
# So that the notebook also ends with a video, step 4 makes a **pure-Python approximation** of the clip from the same
# layers: the lenses follow the low hour by hour, without the sweep and without the moving wind streaks.
#
# **Data:** ERA5 reanalysis (Copernicus Climate Change Service / ECMWF), ETOPO 2022 relief (NOAA NCEI), Natural Earth
# coastlines. No key, no account.
# **Style** after Fan Mo, #30DayMapChallenge 2026 Day 14; the code and data here are original.
#
# **Time:** about 3 minutes.

# %% [markdown]
# ## Settings
# Everything you may want to change is in this one cell. The checks at the bottom **stop** the notebook with a plain
# message if a setting cannot work, before anything slow starts.

# %% CHANGE IF YOU LIKE - RUN :: Settings :: Preview or full size, and how many hours the approximation shows. Run it after every change.
import os

QUICK = os.environ.get("LENS_QUICK", "1") == "1"  # the approximation. True (default, right for Colab): 960 x 540 px. False = FULL: 1920 x 1080 px
LAST_HOUR = 15  # the approximation runs from hour 0 (12 UTC 1 Nov) to this hour (15 = 03 UTC 2 Nov, where the clip ends)
SECONDS_PER_HOUR = 1.0  # how long each hour is held


def stop_unless(ok: bool, message: str) -> None:
    """STOP check: halt here, with a plain message, when a setting cannot work."""
    if not ok:
        raise ValueError("STOP - " + message)


stop_unless(isinstance(LAST_HOUR, int) and 3 <= LAST_HOUR <= 15, "LAST_HOUR must be a whole number from 3 to 15: after hour 15 the three lenses no longer fit on the map")
stop_unless(0.3 <= SECONDS_PER_HOUR <= 5, "SECONDS_PER_HOUR must be between 0.3 and 5")
print(f"settings OK | {'preview 960 x 540' if QUICK else 'full size 1920 x 1080'} | hours 0 to {LAST_HOUR}, {SECONDS_PER_HOUR} s each")

# %% [markdown]
# ## 0 · Setup
# The code is this repository's `gismaps` package (`gismaps/lens/`); the small data files come from `data/lens_maps/`
# or are downloaded once from GitHub.

# %% RUN :: Setup :: Loads the modules and the output folder.
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # keep the page clean: library deprecation notes are not part of the recipe

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from gismaps import data as gdata, nb  # noqa: E402  finds the data files; notebook helpers
from gismaps.lens import anim as an  # noqa: E402  the ffmpeg writer and the GIF maker
from gismaps.lens import common as lc  # noqa: E402  the lens rule (composite), pixel canvas, text
from gismaps.lens import weather as lw  # noqa: E402  ERA5 on a Lambert pixel grid, the layers, the low tracker

OUT = nb.out_dir(os.environ.get("LENS_OUT"))  # /content/out on Google Colab, ./out elsewhere
LAYERS = OUT / "lens_layers"  # the pictures of step 2 go here; the Remotion project reads this folder
CACHE = Path("lens_cache")  # downloads are kept here
print("fonts:", lc.SERIF, "+", lc.SANS, "| ffmpeg:", Path(an.ffmpeg_exe()).name)

# %% [markdown]
# ## 1 · The data
# 25 hours of ERA5, 12 UTC 1 November to 12 UTC 2 November 2023: sea-level pressure and 10-m wind at the surface,
# temperature and specific humidity at 850 hPa. The repository ships them as one small file (every second grid point,
# 0.5°, 1.2 MB), because 25 hours of pressure-level fields straight from the public cloud store are about 5 GB of
# transfer. Coastlines (Natural Earth) and relief (ETOPO 2022) complete the base map.
# The clip of the gallery was made from the full 0.25° fields, so its contours are a little finer.

# %% RUN :: Load the data :: The reduced ERA5 file, the coast and the relief.
data = lw.load_reduced(gdata.path("lens_maps", "era5_ciaran_lens_0p5deg.nc"))  # (surface dataset, 850-hPa dataset, the 25 hours)
if not Path(lw.LAND).exists():
    lw.fetch_land(CACHE)  # Natural Earth 1:50m land, public domain (0.5 MB)
if not Path(lw.ETOPO).exists():
    lw.ETOPO = gdata.path("lens_maps", "etopo2022_10min_natl.nc")  # relief (ETOPO 2022, public domain)
times = data[2]
print(len(times), "hours:", str(times[0])[:13], "to", str(times[-1])[:13], "UTC")

# %% [markdown]
# ## 2 · One picture per field and hour
# `lw.build_movie` draws everything on **one pixel grid** (1140 × 625 px, Lambert conformal conic, 1.3 km per pixel
# at the centre), so any lens can show any layer anywhere and still line up with the base map:
#
# | File | What it is |
# |---|---|
# | `ctx.png` | the dark base map (land, coast, graticule) |
# | `p.png` | the plain chart ground of the pressure lens |
# | `t_NN.png`, `q_NN.png` | opaque fills of hour NN: temperature (blue-white-red, white at 0 °C) and humidity |
# | `ql_NN.png`, `pl_NN.png` | transparent line work of hour NN: moisture contours and isobars every 4 hPa |
# | `manifest.json` | the hours, the tracked low in pixels and hPa, a coarse wind grid, the colour ramps |
#
# The low is followed from a seed point hour by hour (the lowest pressure of the whole map would jump to another low).

# %% RUN :: Make the layers :: Writes about 100 small PNG files and manifest.json into out/lens_layers/ (about a minute).
lw.build_movie(data=data, out=LAYERS, log=lambda *a, **k: None)
man = json.loads((LAYERS / "manifest.json").read_text(encoding="utf-8"))
n_png = len(list(LAYERS.glob("*.png")))
print(f"{n_png} pictures of {man['w']} x {man['h']} px and manifest.json in {LAYERS}  ({sum(f.stat().st_size for f in LAYERS.iterdir()) / 1e6:.1f} MB)")
print("the low:", " | ".join(f"{t[8:10]}/{t[11:13]}h {p['hPa']:.0f}" for t, p in list(zip(man["times"], man["low"]))[::3]), "hPa")


def layer(name: str) -> np.ndarray:
    """One layer file as a float RGBA picture (0..1)."""
    return np.asarray(Image.open(LAYERS / name).convert("RGBA"), np.float32) / 255.0


nb.picture(np.concatenate([layer("ctx.png")[..., :3], layer("t_12.png")[..., :3], layer("q_12.png")[..., :3]], axis=1), 1100)

# %% [markdown]
# ## 3 · The lenses follow the low
# A lens is a **circular mask** on the common pixel grid. Three lenses of radius 180 px:
#
# * **pressure and wind** rides on the tracked low;
# * **temperature** sits 185 px to its west and a little north (the cold air behind the storm);
# * **moisture** swings round the low at a fixed distance, south-west of it and then due south (the moist air that
#   feeds the storm), as far as the bottom edge of the map allows.
#
# Where lenses overlap, `overlap = 0` (the **no-blend rule** for data fields): the lower lens keeps its exact colours,
# so they still match its colour bar, and the upper lens adds only its lines.

# %% RUN :: One hour :: Lens centres from the manifest, the lens rule, and the picture of one hour.
R = 180  # lens radius in map pixels
RIM = {"t": "#e8833a", "q": "#2aa198", "p": "#f4ecd2"}  # rim colours: orange, teal, warm white
MAP_H = man["h"]
base, ground = layer("ctx.png")[..., :3], layer("p.png")[..., :3]


def centres(i: int):
    """Centres (map pixels) of the temperature, moisture and pressure lenses at hour i."""
    px, py = man["low"][i]["x"], man["low"][i]["y"]  # the tracked low
    t = (px - 185.0, float(np.clip(py - 80.0, R + 6, MAP_H - R - 6)))
    dy = min(172.0, MAP_H - R - 6 - py)  # how far south of the low the moisture lens may sit
    q = (px - max(30.0**2, 176.0**2 - dy**2) ** 0.5, py + dy)
    return t, q, (px, py)


def map_picture(i: int, n_open: int = 3):
    """The map of hour i with the first ``n_open`` lenses, and those lenses."""
    t, q, p = centres(i)
    lenses = [lc.Lens(layer(f"t_{i:02d}.png"), *t, R, RIM["t"], overlap=0.0),
              lc.Lens(layer(f"q_{i:02d}.png"), *q, R, RIM["q"], over=layer(f"ql_{i:02d}.png"), overlap=0.0),
              lc.Lens(ground, *p, R, RIM["p"], over=layer(f"pl_{i:02d}.png"), overlap=0.0)][:n_open]
    return lc.composite(base, lenses), lenses


img, _ = map_picture(12)  # 00 UTC, 2 November
nb.picture(img, 900)

# %% [markdown]
# ## 4 · A pure-Python approximation of the clip
# Each hour becomes one 1920 × 1080 page: the map of step 3 with rims and the **L**, the title, the date and the
# central pressure, a key of the three lenses, and the credits. The pages are piped to ffmpeg; each is held for
# `SECONDS_PER_HOUR`. The three lenses open one after the other on the first hour.
# What this approximation leaves out, and the Remotion clip has: the sweep of the lenses across the map, the
# cross-fade between hours, and the wind drawn as moving streaks.

# %% RUN :: The approximation :: Writes lens_map_python.mp4 and a GIF, and plays the clip.
MX, MY = 60, 196  # where the map sits on the page
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
KEY = [("TEMPERATURE", "Warmth · 1.5 km up, −12 to +12 °C", RIM["t"]),
       ("PRESSURE & WIND", "Motion · isobars every 4 hPa", RIM["p"]),
       ("MOISTURE", "Fuel · vapour 1.5 km up, 0 to 10 g/kg", RIM["q"])]


def page(i: int, n_open: int = 3) -> np.ndarray:
    """The whole 1920 x 1080 page of hour i as a float RGB picture."""
    img, lenses = map_picture(i, n_open)
    fig, ax = lc.canvas(1920, 1080, "#171f33")
    ax.imshow(img, extent=(MX, MX + man["w"], MY + man["h"], MY), interpolation="nearest", zorder=1)
    for L in lenses:  # rims; the map sits at (MX, MY) on the page
        ax.add_patch(__import__("matplotlib").patches.Circle((MX + L.cx, MY + L.cy), L.r, fill=False, ec=L.rim, lw=3.2, zorder=21))
    low = man["low"][i]
    if n_open == 3:
        lc.text(ax, MX + low["x"], MY + low["y"] + 12, "L", 40, "#f4ecd2", weight="bold", ha="center")
    lc.text(ax, 60, 96, "One storm, many ", 76, "#f4ecd2", family=lc.SERIF, weight="bold")
    lc.text(ax, 60, 160, "realities.", 76, "#e8833a", family=lc.SERIF, weight="bold", style="italic")
    iso = man["times"][i]
    lc.text(ax, 1262, 90, "Storm Ciarán", 52, "#f4ecd2", weight="bold")
    lc.text(ax, 1262, 146, f"{iso[11:13]} UTC · {int(iso[8:10])} {MON[int(iso[5:7]) - 1]} {iso[:4]}", 44, "#d8b23a")
    lc.text(ax, 1262, 200, f"low centre {low['hPa']:.0f} hPa", 44, "#e8ecf4")
    for k, (name, what, colour) in enumerate(KEY[:max(n_open, 1)] if n_open < 3 else KEY):
        y = 300 + 170 * k
        ax.plot([1238, 1238], [y - 40, y + 70], color=colour, lw=6, solid_capstyle="butt", zorder=30)
        lc.text(ax, 1262, y, name, 44, "#f4ecd2", weight="bold")
        lc.text(ax, 1262, y + 54, what, 34, colour, family=lc.SERIF, style="italic")
    lc.text(ax, 60, 866, "Data: ERA5 (Copernicus C3S / ECMWF) · ETOPO 2022 · Natural Earth · lens style after Fan Mo, #30DayMapChallenge 2026 Day 14", 30, "#9aa6bd")
    lc.text(ax, 60, 912, lc.AUTHOR, 30, "#f1ede4", weight="bold")
    return lc.grab(fig)[..., :3]


FPS = 12
size = (960, 540) if QUICK else (1920, 1080)


def fit(a: np.ndarray) -> np.ndarray:
    """The page at the output size."""
    if a.shape[1] == size[0]:
        return a
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).resize(size, Image.LANCZOS), np.float32) / 255.0


clip = OUT / "lens_map_python.mp4"
writer = an.Writer(clip, size, fps=FPS)
for n_open in (0, 1, 2):  # the lenses open one after the other on the first hour
    writer.add(fit(page(0, n_open)), repeat=FPS // 2)
for i in range(LAST_HOUR + 1):  # then one page per hour
    writer.add(fit(page(i)), repeat=max(1, int(round(SECONDS_PER_HOUR * FPS))))
writer.add(fit(page(LAST_HOUR)), repeat=2 * FPS)  # hold the last hour
writer.close()
gif = an.to_gif(clip, width=480, fps=6)
print(f"{clip.name}: {writer.n} frames, {writer.n / FPS:.0f} s, {clip.stat().st_size / 1e6:.1f} MB | {gif.name}: {gif.stat().st_size / 1e6:.1f} MB")
nb.video(clip, width=960)

# %% [markdown]
# ## 5 · The clip of the gallery: render it with Remotion (needs Node.js on your own machine)
# The folder `remotion/lens_map/` of the repository is a small, self-contained Remotion project. It reads the folder
# of layers made in step 2 and renders the 41-second clip (1230 frames at 30 frames per second, 1920 × 1080 px):
#
# ```
# python tutorials/lens_maps/03_lens_map_clip.py            # steps 1-4: writes out/lens_layers/
# cd remotion/lens_map
# npm install                                               # remotion 4.0.537, react 19
# npx remotion render src/index.ts LensMapClip out/lens_map.mp4 --public-dir=../../out/lens_layers
# ```
#
# What the Remotion source does with the layers:
#
# * `src/kit/lens.tsx`: the same lens rule as `lc.composite`, written as circle masks: one element per lens with
#   `clip-path: circle()`, and a CSS `mask-image` of radial gradients that sets the opacity to `overlap` (here 0)
#   inside each earlier lens. The rims are SVG circles on top.
# * `src/LensMapClip.tsx`: the timeline. One frame count drives everything: the opening of each lens, the sweep, the
#   hour on the clock with a 10-frame cross-fade to the next hour, and the lens centres of step 3.
# * The wind is drawn as moving streaks: short streamlines integrated in each frame from the manifest's wind grid.
# * No text inside the lenses and no isobar labels: at 1920 × 1080 they would be too small to read.
#
# On Google Colab, download `out/lens_layers/` (the last cell zips the outputs) if you want to render on your machine.

# %% [markdown]
# ## Credits
# * ERA5: Hersbach et al. (2020), *Q. J. R. Meteorol. Soc.* 146, 1999-2049. Contains modified Copernicus Climate
#   Change Service information; neither the European Commission nor ECMWF is responsible for any use made of it.
# * Relief: ETOPO 2022, NOAA National Centers for Environmental Information (public domain). Coast: Natural Earth
#   (public domain).
# * Style after Fan Mo, #30DayMapChallenge 2026 Day 14. Map created by Shammunul Islam.

# %% [markdown]
# ## Verification
# A last look that the layers and the approximation are there and are what they should be.

# %% RUN :: Verification :: Checks the layers, the manifest, the approximation and the GIF.
checks = [
    ("4 layers for each of the 25 hours, plus the base map and the chart ground", n_png == 4 * 25 + 2),
    ("every layer is 1140 x 625 px", all(Image.open(f).size == (1140, 625) for f in list(LAYERS.glob("*.png"))[::10])),
    ("the manifest holds 25 hours, 25 positions of the low and 25 wind fields", len(man["times"]) == len(man["low"]) == len(man["wind"]["uv"]) == 25),
    ("the low deepens from hour 0 to hour 15", man["low"][15]["hPa"] < man["low"][0]["hPa"]),
    ("all three lenses stay on the map up to LAST_HOUR", all(R <= c[0] <= man["w"] - R and R <= c[1] <= man["h"] - R for i in range(LAST_HOUR + 1) for c in centres(i)[:2])),
    ("the approximation exists and has the planned length", clip.exists() and clip.stat().st_size > 100_000 and writer.n == 3 * (FPS // 2) + (LAST_HOUR + 1) * int(round(SECONDS_PER_HOUR * FPS)) + 2 * FPS),
    ("the GIF exists", gif.exists() and gif.stat().st_size > 50_000),
]
for text, good in checks:
    print("PASS " if good else "FAIL ", text)
stop_unless(all(good for _, good in checks), "a check failed; read the FAIL line above")
print(f"\nAll checks passed. Layers: {LAYERS} | approximation: {clip}")

# %% [markdown]
# ## Download (Google Colab)
# On Colab this zips the pictures and movies of the output folder and offers them as one download; anywhere else it
# only says where they are.

# %% RUN :: Download :: On Colab: one zip of the outputs. Elsewhere: the folder name.
import shutil  # noqa: E402

if nb.in_colab():
    shutil.make_archive(str(OUT / "lens_layers"), "zip", LAYERS)  # the layers for the Remotion render
    print("layers for the Remotion render:", OUT / "lens_layers.zip", "(Files panel on the left)")
nb.download(OUT)
