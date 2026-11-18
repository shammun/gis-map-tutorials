# %% [markdown]
# # 4 · A data field as lit, terraced relief
#
# **The idea.** Treat the data as a height. Cut it into filled bands, light the bands from the north-west like a
# relief model, float the land above a dark sea on a soft shadow, and step a large counter through time. High
# values become real-looking peaks, so the eye finds them at once, and the bands keep the map readable against
# the colour bar.
#
# **This map.** Total column water vapour over South Asia, month by month (ERA5 means of 1991-2020): the
# water held in the whole column of air, as the depth of rain it would make. The summer monsoon rises over the
# plains like a mountain range; the Himalaya and Tibet stay a dry plateau.
#
# Steps: 1 data → 2 the map grid and the land mask → 3 bands → 4 terraces and light → 5 float it on the
# sea → 6 the finished frame → 7 the animation.

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
POSTER_MONTH = 6                                 # the month of the poster: 0 = January ... 11 = December; 6 = July

from gismaps import nb                           # helpers: data loader, show(), video(), STOP checks, PASS / FAIL

nb.stop_unless(MODE in ("FAST", "FULL"), f"MODE = {MODE!r} must be FAST or FULL")
nb.stop_unless(POSTER_MONTH in range(12), f"POSTER_MONTH = {POSTER_MONTH} must be 0 (January) to 11 (December)")
POSTER_SCALE = 1.0 if MODE == "FAST" else 2.0    # the design size is 1920 x 1080; the poster is saved at this multiple
OUT = nb.out_dir(OUT_DIR)
print(f"settings OK | mode: {MODE} | output folder: {OUT} | poster {int(1920 * POSTER_SCALE)} x {int(1080 * POSTER_SCALE)} px")

# %% [markdown]
# ## 1 · The data
# Monthly means 1991-2020 of total column water vapour over South Asia on a 1° grid (ERA5 monthly means,
# Copernicus Climate Change Service / ECMWF), reduced to one small file.
# %%
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from gismaps.styles import common as C, relief      # the map-style code of this repository

data = nb.load("tcwv_southasia")            # tcwv (12 months, lat, lon) in kg m-2 = mm of water
print(data["tcwv"].shape, "|", str(data["source"]))
print("July mean at 23.5N 90.5E (near Dhaka):",
      round(float(data["tcwv"][6, np.argmin(abs(data["lat"] - 23.5)), np.argmin(abs(data["lon"] - 90.5))]), 1), "mm")

# %% [markdown]
# ## 2 · The map grid and the land mask
# Lambert azimuthal equal-area, centred on India: equal areas matter because we will quote a mean over the
# land in view. The 1° field is resampled to the pixels with a cubic spline (`order=3`), which gives the smooth
# surface this style needs. The land mask comes from Natural Earth.

# %%
grid = C.MapGrid.fit("+proj=laea +lat_0=21.5 +lon_0=81", 0, 0, 4300e3, 1120, 1020)
land = grid.land("50m", soft=1.0)
july = grid.sample(data["tcwv"][6], data["lat"], data["lon"], order=3)
print("pixels:", grid.w, "x", grid.h, "| July range on land:", round(float(np.nanmin(july[land > 0.5])), 1), "to",
      round(float(np.nanmax(july[land > 0.5])), 1), "mm")

# %% [markdown]
# ## 3 · Bands
# Fifteen bands of 5 mm, from 0 to 75 mm (the wettest land pixel in July holds 71 mm). A banded map can be read against its colour bar (a smooth ramp cannot, once it is
# shaded), and the band edges will become the terraces. The colour map is `cmocean.haline`: dark blue for dry,
# through green, to pale yellow for moist; it only gets lighter, so it survives greyscale and colour-blind
# viewing.

# %%
levels = np.arange(0, 80, 5)
flat = C.colourise(july, cmocean.cm.haline, 0, 75, levels=levels)

# %% [markdown]
# ## 4 · Terraces and light
# `relief.terrace` mixes the smooth field with its band number, so each band edge is a small step; `C.hillshade`
# lights that surface from the north-west (azimuth 315°, 42° above the horizon). The colour is multiplied by
# `ambient + light × shade`, and a thin dark line marks every band edge.

# %%
height = relief.terrace(np.nan_to_num(july), levels)
shade = C.hillshade(height, 315, 42, zfactor=3.4)
rgb, alpha = relief.relief_rgb(july, levels, cmocean.cm.haline, land)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for a, im, t in zip(axs, (flat, shade, rgb), ("3 · flat bands", "4 · the light alone", "bands × light + edge lines")):
    a.imshow(im, cmap="gray" if im.ndim == 2 else None); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5 · Float it on the sea
# The lit layer is drawn only on land. Under it goes a soft shadow (the land mask, shifted 10 px right and 14 px
# down, blurred), and along the north-west shores a light rim: the land reads as a slab lying on the dark sea.

# %%
sea = np.broadcast_to(C.rgb("#0c2733"), (grid.h, grid.w, 3)).copy()
img = relief.float_on(sea, rgb, alpha * np.isfinite(july))
plt.figure(figsize=(8.5, 7.8)); plt.imshow(img); plt.axis("off"); plt.show()

# %% [markdown]
# ## 6 · The finished frame
# `VapourRelief` adds the big month, the mean over the land in view, the year as twelve bars with this month
# lit, city names with halos, the banded colour bar and the credits.

# %%
vap = relief.VapourRelief(data)
print("mean over the land in view, by month (mm):", vap.mean.round(0))
vap.frame(0.0, OUT / "04_jan.png")
poster = vap.poster(OUT / "relief_field.png", month=POSTER_MONTH, scale=POSTER_SCALE)      # July with the default setting
nb.strip([OUT / "04_jan.png"])
nb.show(poster)

# %% [markdown]
# ## 7 · The animation
# The twelve monthly means are blended linearly from one to the next (24 frames per month, with a short hold
# on each month so the counter can be read).

# %%
if MODE == "FULL":                               # the whole animation, 1920 x 1080 px
    movie = vap.animate(OUT / "relief_field.mp4", OUT / "frames_04")
else:                                            # a short preview: every 6th frame, 960 x 540 px, 12 frames a second
    movie = C.preview(vap.animate, OUT / "relief_field_preview.mp4", OUT / "frames_04", thin=6)
nb.video(movie)

# %% [markdown]
# ## Check the outputs
# The last cell checks that every output exists and is what it should be, and prints PASS or FAIL per check.

# %%
size = (int(1920 * POSTER_SCALE), int(1080 * POSTER_SCALE))
checks = [
    ("the poster exists", poster.exists()),
    (f"the poster is {size[0]} x {size[1]} px", nb.image_size(poster) == size),
    ("the January frame exists", (OUT / "04_jan.png").exists()),
    ("twelve monthly fields", data["tcwv"].shape[0] == 12),
    ("the monsoon shows: July holds more water vapour than January over the land in view", float(vap.mean[6]) > 1.5 * float(vap.mean[0])),
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
