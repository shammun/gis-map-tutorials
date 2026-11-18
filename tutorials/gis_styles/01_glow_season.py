# %% [markdown]
# # 1 · Glow-dot time-lapse with a mirrored season chart
#
# **The idea.** Put every event on a dark map as a tiny dot of light and *add* the light where dots fall on top of
# each other. Places with many events glow; places with few stay dark. Then run a clock through the calendar and
# show only the events near "today", and put a small mirrored chart under the map (one group up, the other down)
# with a playhead. The map says *where*, the chart says *when*, and they move together.
#
# **This map.** Thirty-five years of tropical cyclones (IBTrACS, 1990-2024), each 6-hourly position drawn on
# its day of the year. The Northern Hemisphere season and the Southern one light up in turn, half a year apart.
#
# Steps: 1 data → 2 one map grid → 3 additive glow → 4 tone-mapping and colour → 5 the mirrored chart →
# 6 the finished frame → 7 the animation.

# %% [markdown]
# ## Settings
# Everything you may want to change is in this one cell. The checks at the bottom **stop** the notebook with a plain
# message if a setting cannot work, before anything slow starts.
#
# `MODE = "FAST"` (the default, one to three minutes; right for Google Colab) writes the poster at 1920 × 1080 px and a short
# preview clip of the animation at 960 × 540 px. `MODE = "FULL"` writes the poster at 3840 × 2160 px and the whole
# animation at 1920 × 1080 px (several minutes).

# %%
import os
import sys

MODE = "FULL" if "--full" in sys.argv else os.environ.get("GIS_MODE", "FAST")   # "FAST" (preview) or "FULL" (poster size, whole animation)
OUT_DIR = os.environ.get("GIS_OUT", "")          # output folder; "" = /content/out on Google Colab, ./out anywhere else
SAMPLE_DAY = 250                                 # the calendar day drawn in steps 3-4: 0 = 1 January ... 364; 250 = 8 September

from gismaps import nb                           # helpers: data loader, show(), video(), STOP checks, PASS / FAIL

nb.stop_unless(MODE in ("FAST", "FULL"), f"MODE = {MODE!r} must be FAST or FULL")
nb.stop_unless(0 <= SAMPLE_DAY <= 364, f"SAMPLE_DAY = {SAMPLE_DAY} must be a day of the year, 0 to 364")
POSTER_SCALE = 1.0 if MODE == "FAST" else 2.0    # the design size is 1920 x 1080; the poster is saved at this multiple
OUT = nb.out_dir(OUT_DIR)
print(f"settings OK | mode: {MODE} | output folder: {OUT} | poster {int(1920 * POSTER_SCALE)} x {int(1080 * POSTER_SCALE)} px")

# %% [markdown]
# ## 1 · The data
# One small file: every 6-hourly position of every tropical cyclone of 1990-2024 at tropical-storm strength or
# more (IBTrACS v04r01, NOAA NCEI, public domain). `nb.load` reads it from `data/gis_styles/`, downloads it if it is
# missing, and rebuilds it from NOAA's files if that fails (`gismaps/prepare_data.py` is the record of how).
# %%
import numpy as np
import matplotlib.pyplot as plt

from gismaps.styles import common as C, glow      # the map-style code of this repository

data = nb.load("cyclones")                  # lat, lon, wind_kt, doy (0-based day of year), year
print({k: (v.shape, v.dtype) for k, v in data.items() if v.ndim})
print("fixes:", len(data["lat"]), "| years", data["year"].min(), "to", data["year"].max())

# %% [markdown]
# ## 2 · One projection, one pixel grid
# Everything on the map (land, dots) is placed on the **same pixel grid**. `MapGrid` holds a PROJ string and an
# extent; `xy(lon, lat)` gives pixel positions. Equal Earth is an equal-area projection, so the same number of storms
# makes the same amount of light anywhere. We centre it on 165°W: the cut then falls on Africa (15°E), where
# there are no tropical cyclones, and no ocean basin is split.

# %%
grid = C.MapGrid.from_box("+proj=eqearth +lon_0=-165", 15.001, 374.999, -47, 53, w=1840)
x, y = grid.xy(data["lon"], data["lat"])
points = np.c_[x, y]
print("map is", grid.w, "x", grid.h, "px;", round(grid.mpp / 1000, 1), "km per pixel")

# %% [markdown]
# ## 3 · Additive glow
# `C.glow` bins the points into pixels and blurs the result twice: a sharp core (1 px) and a soft halo (5 px).
# Nothing is "drawn": the image is a **sum**, so density becomes brightness by itself.
#
# For one calendar day we give each fix a weight that fades with its age: today 1, five days ago 1/e. The fading
# tail turns separate 6-hourly fixes into short comet tracks.

# %%
day = SAMPLE_DAY                                         # 8 September with the default setting
age = (day - data["doy"]) % 365                         # days since each fix, round the calendar
near = age < 25
weight = np.clip(data["wind_kt"][near] / 64.0, 0.5, 2.2) * np.exp(-age[near] / 5.0)   # stronger storms glow more
buf = C.glow(points[near], weight, (grid.h, grid.w))
print("buffer max", round(float(buf.max()), 2), "| pixels lit", int((buf > 0.05).sum()))

# %% [markdown]
# ## 4 · Tone-mapping and colour
# The buffer has no upper limit, so it is squeezed into 0..1 with `1 - exp(-gain * v)` (bright places saturate
# gently instead of clipping) and coloured with a ramp that only gets lighter: dark red → orange → amber → white.
# Opacity rises faster than colour, so even a single faint dot is visible on the dark ground.

# %%
land = grid.land("110m")                                # Natural Earth land through a lat/lon raster
base = C.over(np.broadcast_to(C.rgb("#080b14"), (grid.h, grid.w, 3)).copy(), C.rgb("#161c2c"), land)
img = glow.glow_rgb(base, buf, gain=1.7)

fig, axs = plt.subplots(1, 2, figsize=(14, 2.9), facecolor="#05070d")
axs[0].imshow(C.tone(buf, 1.7), cmap="gray"); axs[0].set_title("tone-mapped buffer", color="w")
axs[1].imshow(img); axs[1].set_title("on the base map, 8 September", color="w")
for a in axs:
    a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5 · The mirrored season chart
# For the chart we count how many storms are in progress on each calendar day, on average: fixes per day ÷ 4
# (four fixes a day) ÷ 35 years, for the Northern Hemisphere (drawn upwards) and the Southern (downwards).

# %%
north, south = glow.season_counts(data["doy"], data["lat"] > 0, n_years=35)
print("north peaks on day", int(north.argmax()), f"({north.max():.1f} storms at once);",
      "south peaks on day", int(south.argmax()), f"({south.max():.1f})")

# %% [markdown]
# ## 6 · The finished frame, and the whole year as a poster
# `glow.StormSeason` wraps steps 2-5 with the map furniture (title, month, chart, credits).

# %%
storm = glow.StormSeason(data)
storm.frame(250.0, OUT / "01_glow_sep.png")
storm.frame(45.0, OUT / "01_glow_feb.png")
nb.strip([OUT / "01_glow_feb.png", OUT / "01_glow_sep.png"])

# %%
poster = storm.poster(OUT / "glow_season.png", scale=POSTER_SCALE)      # all days at once, 3840 x 2160
nb.show(poster)

# %% [markdown]
# ## 7 · The animation
# Half a day per frame at 30 frames a second: the year takes 24 seconds. `MODE = "FULL"` in the settings cell (or
# `--full` after the script name) renders all of it; `"FAST"` makes a short preview with the same code. Frames are written as PNG and encoded with ffmpeg.

# %%
if MODE == "FULL":                               # the whole animation, 1920 x 1080 px
    movie = storm.animate(OUT / "glow_season.mp4", OUT / "frames_01")
else:                                            # a short preview: every 16th frame, 960 x 540 px, 12 frames a second
    movie = C.preview(storm.animate, OUT / "glow_season_preview.mp4", OUT / "frames_01", thin=16)
nb.video(movie)

# %% [markdown]
# ## Check the outputs
# The last cell checks that every output exists and is what it should be, and prints PASS or FAIL per check.

# %%
size = (int(1920 * POSTER_SCALE), int(1080 * POSTER_SCALE))
checks = [
    ("the poster exists", poster.exists()),
    (f"the poster is {size[0]} x {size[1]} px", nb.image_size(poster) == size),
    ("two sample frames, 1920 x 1080 px", all(nb.image_size(OUT / f) == (1920, 1080) for f in ("01_glow_feb.png", "01_glow_sep.png"))),
    ("more than 50,000 storm positions were read", len(data["lat"]) > 50_000),
    ("the northern season peaks in August-October, the southern in January-March", 212 <= int(north.argmax()) <= 304 and int(south.argmax()) <= 90),
]
if not os.environ.get("GIS_PART"):               # (GIS_PART marks a frame worker of the parallel FULL render)
    checks.append(("the movie exists and is not empty", movie is not None and movie.exists() and movie.stat().st_size > 20_000))
nb.verify(checks)

# %% [markdown]
# ## Download (Google Colab)
# On Colab this zips the pictures and movies of the output folder and offers them as one download; anywhere else it
# only says where they are.

# %%
nb.download(OUT)
