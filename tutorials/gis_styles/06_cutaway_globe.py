# %% [markdown]
# # 6 · Cut-away globe: a map and three sections in one object
#
# **The idea.** A globe shows a field on its surface. Cut a wedge out of it (a range of longitudes, north of the
# Equator) and paint what lies *below* the surface on the three flat faces of the cut: two meridians and the
# equatorial plane. The reader sees a map and three vertical sections that meet at the right places, with no need
# to work out where each section lies.
#
# **This map.** Annual-mean ocean temperature (World Ocean Atlas 2023) from the surface to 1000 m in the
# Pacific, cut along 180°, 110°W and the Equator.
#
# Steps: 1 data → 2 one ray per pixel → 3 through the wedge to a face → 4 from the hit point to a value →
# 5 colour, light and lines → 6 the finished frame → 7 the animation.

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
LON_WEST, LON_EAST = 180.0, 250.0                # the two meridians of the cut (degrees east): 180 and 110°W

from gismaps import nb                           # helpers: data loader, show(), video(), STOP checks, PASS / FAIL

nb.stop_unless(MODE in ("FAST", "FULL"), f"MODE = {MODE!r} must be FAST or FULL")
nb.stop_unless(0 <= LON_WEST < LON_EAST <= 360 and 30 <= LON_EAST - LON_WEST <= 120,
               "LON_WEST and LON_EAST must be degrees east, west face first, 30 to 120 degrees apart")
POSTER_SCALE = 1.0 if MODE == "FAST" else 2.0    # the design size is 1920 x 1080; the poster is saved at this multiple
OUT = nb.out_dir(OUT_DIR)
print(f"settings OK | mode: {MODE} | output folder: {OUT} | poster {int(1920 * POSTER_SCALE)} x {int(1080 * POSTER_SCALE)} px")

# %% [markdown]
# ## 1 · The data
# Annual-mean ocean temperature from the surface to 2000 m on a 1° grid (World Ocean Atlas 2023, NOAA NCEI,
# public domain), reduced to one small file.
# %%
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from gismaps.styles import common as C, cutaway      # the map-style code of this repository

data = nb.load("woa23_temperature")         # t_c100 (depth, lat, lon) in 0.01 degrees C; -9999 = no water
print("grid:", data["t_c100"].shape, "| depths:", data["depth_m"][0], "to", data["depth_m"][-1], "m")

# %% [markdown]
# ## 2 · One ray per pixel
# The camera is far away (parallel rays). For a pixel at (x, y), in Earth radii from the centre of the disc, the
# ray meets the unit sphere at the point `P = x·right + y·up + sqrt(1 − x² − y²)·forward`. Its longitude and
# latitude follow from P. If that point is outside the wedge, the pixel shows the **surface**.

# %% [markdown]
# ## 3 · Through the wedge to a face
# If the point is inside the wedge, there is no surface there: the ray flies on, in a straight line away from the
# camera, until it leaves the wedge. The wedge is bounded by three planes (meridian A, meridian B, the Equator), so
# the exit point is the nearest of three ray-plane intersections. That point has a radius between 0 and 1 and
# lies on one of the three **faces**.
#
# `cutaway.raycast` does this for every pixel at once and returns what each pixel is and where its point lies.

# %%
rc = cutaway.raycast(600, lon_c=212, lat_c=30, lon_a=LON_WEST, lon_b=LON_EAST)
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
axs[0].imshow(rc["kind"], cmap="tab10", vmin=0, vmax=9); axs[0].set_title("what each pixel is: surface or one of three faces")
axs[1].imshow(rc["lat"], cmap="RdBu_r"); axs[1].set_title("latitude of the point seen")
axs[2].imshow(rc["rad"], cmap="viridis"); axs[2].set_title("radius of the point seen (1 = surface)")
for a in axs:
    a.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4 · From the hit point to a value
# A face pixel knows its longitude, latitude and radius. Radius is turned into depth with a **stated, linear**
# scale: radius 1 is the surface and radius 0.5 is 1000 m. That is about 3,200 times deeper than true scale
# (1000 m is 0.016 % of Earth's radius), so the figure says "not to scale" in words. The temperature is then
# read from the atlas at (depth, latitude, longitude) by linear interpolation; where there is no water (land, sea
# floor) the face is grey.

# %%
globe = cutaway.OceanCutaway(data, lon_a=LON_WEST, lon_b=LON_EAST)
exaggeration = (1 - globe.R_IN) * 6371e3 / globe.zmax
print(f"depth exaggeration: about {exaggeration:,.0f} times")
depths = np.array([0, 100, 200, 300, 500, 1000.0])
rad = 1 - depths / globe.zmax * (1 - globe.R_IN)
for name, lo in (("west face", LON_WEST), ("east face", LON_EAST)):
    t = globe.interior_t(np.zeros_like(rad), np.full_like(rad, lo), rad)
    print(f"Equator at {name:12s}:", "  ".join(f"{d:.0f} m {v:4.1f}" for d, v in zip(depths, t)), "(deg C)")

# %% [markdown]
# The numbers show what the equatorial face tells: at 180° the water is still about 28 °C at 100 m, while
# at 110°W it is about 15 °C at the same depth, some 13 degrees colder. The warm layer is deep in the west and
# shallow in the east.

# %% [markdown]
# ## 5 · Colour, light and lines
# Surface and faces share one colour bar (`cmocean.thermal`, −2 to 30 °C). The sphere is shaded as a ball;
# each face gets one brightness from its angle to the light, so the three faces can be told apart. Dark lines on
# the faces mark every 5 °C, white arcs every 250 m, and a light outline separates the parts.

# %%
img = globe.globe(700, lon_c=212, lat_c=30)
plt.figure(figsize=(7.5, 7.5), facecolor=C.BG); plt.imshow(img); plt.axis("off"); plt.show()

# %% [markdown]
# ## 6 · The finished frame

# %%
poster = globe.poster(OUT / "cutaway_globe.png", scale=POSTER_SCALE)
nb.show(poster)

# %% [markdown]
# ## 7 · The animation
# Three moves, the same every time this device is used: the whole globe turns to the place, the wedge opens
# (meridian B swings east from meridian A), then labels fade in while the camera sways a few degrees so that the
# faces separate in depth.

# %%
if MODE == "FULL":                               # the whole animation, 1920 x 1080 px
    movie = globe.animate(OUT / "cutaway_globe.mp4", OUT / "frames_06")
else:                                            # a short preview: every 10th frame, 960 x 540 px, 12 frames a second
    movie = C.preview(globe.animate, OUT / "cutaway_globe_preview.mp4", OUT / "frames_06", thin=10)
nb.video(movie)

# %% [markdown]
# ## Check the outputs
# The last cell checks that every output exists and is what it should be, and prints PASS or FAIL per check.

# %%
size = (int(1920 * POSTER_SCALE), int(1080 * POSTER_SCALE))
checks = [
    ("the poster exists", poster.exists()),
    (f"the poster is {size[0]} x {size[1]} px", nb.image_size(poster) == size),
    ("the ray-cast found the surface and all three faces", len(np.unique(rc["kind"])) >= 4),
    ("depth is exaggerated (and the figure says so)", exaggeration > 100),
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
