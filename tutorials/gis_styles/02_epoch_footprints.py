# %% [markdown]
# # 2 · Epoch-stacked footprints
#
# **The idea.** Take the outline of one thing at several times and give each time one colour of an ordered ramp.
# First show each epoch alone. Then stack them, oldest at the bottom and newest on top: wherever only an *older*
# epoch reached, its colour stays visible as a crescent. One still picture then holds the whole history of a moving
# or shrinking shape.
#
# **This map.** Storm Ciarán, 1-2 November 2023: the area where mean-sea-level pressure was below 960 hPa,
# every 3 hours (ERA5).
#
# Steps: 1 data → 2 follow the low → 3 one footprint per epoch → 4 smooth edges on the map grid →
# 5 ordered colours and the stack → 6 the finished frames → 7 the animation.

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
THRESHOLD_HPA = 960.0                            # the footprint: sea-level pressure below this value (hPa)
STEP_HOURS = 3                                   # one footprint every this many hours

from gismaps import nb                           # helpers: data loader, show(), video(), STOP checks, PASS / FAIL

nb.stop_unless(MODE in ("FAST", "FULL"), f"MODE = {MODE!r} must be FAST or FULL")
nb.stop_unless(955 <= THRESHOLD_HPA <= 1000, f"THRESHOLD_HPA = {THRESHOLD_HPA}: Ciarán's centre fell to about 953 hPa; use 955 to 1000")
nb.stop_unless(STEP_HOURS in (1, 2, 3, 4, 6), "STEP_HOURS must be 1, 2, 3, 4 or 6")
POSTER_SCALE = 1.0 if MODE == "FAST" else 2.0    # the design size is 1920 x 1080; the poster is saved at this multiple
OUT = nb.out_dir(OUT_DIR)
print(f"settings OK | mode: {MODE} | output folder: {OUT} | poster {int(1920 * POSTER_SCALE)} x {int(1080 * POSTER_SCALE)} px")

# %% [markdown]
# ## 1 · The data
# Hourly ERA5 sea-level pressure and 10-m wind for 1-2 November 2023 over the north-east Atlantic and western
# Europe (Copernicus Climate Change Service / ECMWF), reduced to one small file.
# %%
import numpy as np
import matplotlib.pyplot as plt

from gismaps.styles import common as C, epochs      # the map-style code of this repository

data = nb.load("ciaran")                    # hourly ERA5: msl_hpa10 (hPa x 10), u10_cms, v10_cms on lat/lon
p = data["msl_hpa10"] / 10.0                    # hPa, shape (48 hours, lat, lon)
lat, lon, time = data["lat"], data["lon"], [str(t) for t in data["time"]]
print(p.shape, time[0], "to", time[-1], "| lowest pressure in the box:", p.min(), "hPa")

# %% [markdown]
# ## 2 · Follow the low (never "the minimum of the box")
# Two lows are in this box: Ciarán and its older, larger parent low near Iceland. The minimum of the whole box
# jumps between them. So we **seed** the tracker where Ciarán was at 12 UTC on 1 November (48°N, 17°W) and
# then, hour by hour, take the lowest pressure within 8 grid cells (2°) of the previous position.

# %%
i0 = time.index("2023-11-01T12")
seed = (int(np.argmin(np.abs(lat - 48.0))), int(np.argmin(np.abs(lon + 17.0))))
rc = epochs.track_minimum(p, seed, i0, radius=8)
for i in range(i0, 48, 6):
    print(time[i], f"{lat[rc[i, 0]]:5.2f}N {lon[rc[i, 1]]:6.2f}E  {p[i, rc[i, 0], rc[i, 1]]:.1f} hPa")

# %% [markdown]
# ## 3 · One footprint per epoch
# The footprint is the **connected** region below the threshold that contains the tracked centre
# (`scipy.ndimage.label`), so the parent low never leaks in. We pick 960 hPa: inside this box the parent low stayed above
# 964 hPa, and Ciarán was below 960 hPa from the evening of 1 November to the evening of 2 November.

# %%
steps = list(range(time.index("2023-11-01T21"), time.index("2023-11-02T21") + 1, STEP_HOURS))
masks = [epochs.footprint(p[i], THRESHOLD_HPA, tuple(rc[i])) for i in steps]
cell_km2 = (111.195 * 0.25) ** 2 * np.cos(np.deg2rad(lat))[:, None]        # area of one 0.25-degree cell
for i, m in zip(steps, masks):
    print(time[i], f"area below {THRESHOLD_HPA:.0f} hPa: {float((m * cell_km2).sum()) / 1e3:6.0f} thousand km2")

# %% [markdown]
# ## 4 · Smooth edges on the map grid
# A 0.25° mask drawn on a map has stairs. Instead we resample the **pressure** to the map's pixels (cubic) and
# apply the threshold there. The map is a Lambert conformal conic (standard parallels 45°N and 58°N), the usual
# choice for a mid-latitude weather map.

# %%
grid = C.MapGrid.fit("+proj=lcc +lat_1=45 +lat_2=58 +lat_0=52 +lon_0=-3.5", -60e3, 30e3, 2050e3, 1240, 1000)
i = steps[3]
p_px = grid.sample(p[i], lat, lon, order=3)
alpha = np.clip((THRESHOLD_HPA - p_px) / 0.15 + 0.5, 0, 1)        # 1 inside, 0 outside, a 1-pixel soft edge
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
axs[0].imshow(masks[3][::-1], cmap="gray"); axs[0].set_title("mask on the 0.25° grid")
axs[1].imshow(alpha, cmap="gray"); axs[1].set_title("threshold applied on the map pixels")
for a in axs:
    a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5 · Ordered colours, then the stack
# The colours come from one perceptually ordered ramp (`plasma`), first = oldest. An ordered ramp matters: the
# reader must be able to sort the epochs by colour alone. `epochs.stack` lays the footprints oldest first.

# %%
ciaran = epochs.CiaranFootprint(data, threshold=THRESHOLD_HPA, step=STEP_HOURS)           # does steps 2-4 for every epoch and builds the base map
cols = epochs.epoch_colours(len(ciaran.idx))
img = epochs.stack(ciaran.base, ciaran.alphas, cols, upto=len(cols))
plt.figure(figsize=(9, 7.3)); plt.imshow(img); plt.axis("off"); plt.show()

# %% [markdown]
# ## 6 · The finished frames
# Two kinds of frame: **solo** (one epoch filled, with that hour's isobars, earlier ones as outlines) and the
# **stack**. The legend is the clock: it lists every epoch with the lowest pressure at that time.

# %%
ciaran.frame(OUT / "02_solo.png", upto=4, solo=3)
poster = ciaran.poster(OUT / "epoch_footprints.png", scale=POSTER_SCALE)
nb.show(OUT / "02_solo.png")
nb.show(poster)

# %% [markdown]
# ## 7 · The animation
# Each epoch alone for about a second, then the stack builds, then a hold.

# %%
if MODE == "FULL":                               # the whole animation, 1920 x 1080 px
    movie = ciaran.animate(OUT / "epoch_footprints.mp4", OUT / "frames_02")
else:                                            # a short preview: every frame, 960 x 540 px, 12 frames a second
    movie = C.preview(ciaran.animate, OUT / "epoch_footprints_preview.mp4", OUT / "frames_02", thin=1)
nb.video(movie)

# %% [markdown]
# ## Check the outputs
# The last cell checks that every output exists and is what it should be, and prints PASS or FAIL per check.

# %%
size = (int(1920 * POSTER_SCALE), int(1080 * POSTER_SCALE))
checks = [
    ("the poster exists", poster.exists()),
    (f"the poster is {size[0]} x {size[1]} px", nb.image_size(poster) == size),
    ("the sample frame exists", (OUT / "02_solo.png").exists()),
    ("the tracked low deepens below the threshold", float(min(p[i, rc[i, 0], rc[i, 1]] for i in steps)) < THRESHOLD_HPA),
    ("at least three footprints are stacked", len(ciaran.idx) >= 3),
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
