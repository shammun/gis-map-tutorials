# %% [markdown]
# # Glass-pane globe stack
#
# **The picture.** A relief globe with six glass panes standing in front of it. Every pane shows the same round
# patch of the Earth (the cap of 21° around 21°N 87°E: India, Bangladesh, the Bay of Bengal) at the same moment of
# one weather forecast, each with a different quantity: temperature, sea-level pressure, wind, precipitable water,
# cloud cover and rain rate. The panes are pulled apart along the line of sight to that patch, so each can be seen,
# and they line up with the globe behind them.
#
# **How to read it.** Left: the globe with the region turned towards you. In front of it, fanned to the right: the
# panes, numbered 1 (nearest the globe) to 6. Right: what each numbered pane shows, with its colour bar. Bottom: the
# lead time, the valid time in UTC and in Indian Standard Time, the place of that time on the five-day timeline, and
# the sources. The panes are **not** at physical heights: they are six maps of one patch, pulled apart.
#
# **The data.** NOAA GFS forecast, 0.25°, run of 16 November 2026 18 UTC, read from
# [dynamical.org](https://dynamical.org) (CC BY 4.0). Relief: ETOPO 2022 (NOAA NCEI). Coastlines: Natural Earth.
#
# Steps: 1 data → 2 the round view of one field → 3 six pane pictures → 4 the globe's skin → 5 globe and
# panes in 3-D → 6 the finished poster → 7 the animation → 8 the inputs of the Remotion version.

# %% [markdown]
# ## Settings
# Everything you may want to change is in this one cell. The checks at the bottom **stop** the notebook with a plain
# message if a setting cannot work, before anything slow starts.
#
# `MODE = "FAST"` (the default, about two minutes; right for Google Colab) writes the poster at 1920 × 1080 px and a
# short preview clip of the animation at 960 × 540 px. `MODE = "FULL"` writes the poster at 3840 × 2160 px and the
# whole animation (16 s, 1280 × 720 px, 24 frames a second; several minutes).
#
# `SOURCE = "shipped"` (the default) reads the small data file that comes with the repository: no network to
# dynamical.org is needed. `SOURCE = "live"` reads the run `RUN` straight from dynamical.org, anonymously (first
# `pip install icechunk zarr`; about 35 MB, about a minute); if that read fails for any reason, the shipped file is
# used and a line says so.

# %%
import os
import sys

MODE = "FULL" if "--full" in sys.argv else os.environ.get("GIS_MODE", "FAST")   # "FAST" (preview) or "FULL" (poster size, whole animation)
OUT_DIR = os.environ.get("GIS_OUT", "")          # output folder; "" = /content/out on Google Colab, ./out anywhere else
SOURCE = os.environ.get("GIS_SOURCE", "shipped")  # "shipped" (the file of the repository) or "live" (dynamical.org)
RUN = "2026-11-16T18:00"                         # with SOURCE = "live": the GFS run, hour 00, 06, 12 or 18 UTC; None = the newest complete run
LEAD = 66                                        # the lead time of the poster in hours: a multiple of 6 from 6 to 102
CAP = dict(lat0=21.0, lon0=87.0, rad_deg=21.0)   # the cap every pane shows: centre and angular radius in degrees

import datetime as dt

from gismaps import nb                           # helpers: show(), video(), STOP checks, PASS / FAIL

nb.stop_unless(MODE in ("FAST", "FULL"), f"MODE = {MODE!r} must be FAST or FULL")
nb.stop_unless(SOURCE in ("shipped", "live"), f"SOURCE = {SOURCE!r} must be shipped or live")
nb.stop_unless(LEAD in range(6, 103, 6), f"LEAD = {LEAD} must be a multiple of 6 from 6 to 102 (cloud and rain are averages: there is no value at 0)")
if SOURCE == "live" and RUN is not None:
    try:
        _run = dt.datetime.fromisoformat(RUN)
    except (TypeError, ValueError):
        _run = None
    nb.stop_unless(_run is not None and _run.hour in (0, 6, 12, 18) and _run.minute == 0,
                   f"RUN = {RUN!r} must look like '2026-11-16T18:00' with hour 00, 06, 12 or 18 (or be None for the newest run)")
WORKER = bool(os.environ.get("GIS_PART"))        # (set only in the frame workers of the parallel FULL render: they skip the extras)
POSTER_SCALE = 2.0 if MODE == "FULL" and not WORKER else 1.0   # the design size is 1920 x 1080; the poster is saved at this multiple
OUT = nb.out_dir(OUT_DIR)
print(f"settings OK | mode: {MODE} | data: {SOURCE} | output folder: {OUT} | poster {int(1920 * POSTER_SCALE)} x {int(1080 * POSTER_SCALE)} px")

# %% [markdown]
# ## 1 · The data
# Six fields of one forecast run on the box 0-44°N, 62-112°E at 0.25°, every 6 hours from +6 h to +102 h:
# temperature at 2 m, sea-level pressure, wind speed at 10 m, precipitable water (the water vapour of the whole
# column of air, as the depth of rain it would make), total cloud cover, and the precipitation rate averaged over
# the 6 hours before each lead time. The shipped file keeps them as 16-bit numbers (2 MB).

# %%
import numpy as np
import matplotlib.pyplot as plt

from gismaps.glass import gfs, ortho, panes, render        # the code of this recipe
from gismaps.glass.stack import GlassStack
from gismaps.styles import common as C

ds = gfs.region(SOURCE, RUN)                     # dims (lead, latitude, longitude)
RUN = ds.attrs["run"]
print("run:", RUN, "UTC |", dict(ds.sizes), "|", ds.attrs["credit"])
for k in ds.data_vars:
    a = ds[k].sel(lead=LEAD)
    print(f"  {k:5s} at +{LEAD} h: {float(a.min()):8.1f} to {float(a.max()):8.1f} {a.attrs['units']}")

# %% [markdown]
# ## 2 · The round view of one field
# Stand far away, straight above the centre of the cap: what you see is a disc, the **orthographic projection**.
# Each pane shows exactly that view, which is why the panes line up with the globe behind them. To make the
# picture we go backwards: for every pixel of the disc, `ortho.inverse` gives the place it shows, and
# `ortho.sample` reads the field there (bilinear). Going backwards leaves no holes. East is to the right and north
# is up, as seen from outside the globe.

# %%
F = panes.fields(ds, LEAD)                       # the six maps of the poster's lead time (+ a smoothed pressure for the isobars)
lat_px, lon_px, inside, _ = ortho.disc(360, **CAP)          # the place each pixel of a 360 x 360 pane shows
pw = F["pwat"]
disc = np.where(inside, ortho.sample(pw.values, pw.latitude.values, pw.longitude.values, lat_px, lon_px), np.nan)

th = np.linspace(0, 2 * np.pi, 200)              # the edge of the cap, drawn on the flat grid
edge_lat, edge_lon = ortho.inverse(np.cos(th), np.sin(th), CAP["lat0"], CAP["lon0"], np.sin(np.radians(CAP["rad_deg"])))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
axs[0].pcolormesh(pw.longitude, pw.latitude, pw, cmap=panes.ramp("pwat"), vmin=0, vmax=60, shading="nearest")
axs[0].plot(edge_lon, edge_lat, color=panes.K["amber"], lw=1.8)
axs[0].set_title("the grid as stored: latitude × longitude"); axs[0].set_xlabel("longitude (°E)"); axs[0].set_ylabel("latitude (°N)")
im = axs[1].imshow(disc, cmap=panes.ramp("pwat"), vmin=0, vmax=60, extent=(-1, 1, -1, 1))
axs[1].set_title("the same numbers on the disc (orthographic)"); axs[1].set_xlabel("u (east →)"); axs[1].set_ylabel("v (north ↑)")
fig.colorbar(im, ax=axs, shrink=0.85, pad=0.02, label="precipitable water (kg m⁻²)")
plt.show()

# %% [markdown]
# ## 3 · Six pane pictures
# A pane picture has four numbers per pixel: red, green, blue and opacity. `panes.pane` colours the disc with one
# ramp per quantity and chooses the opacity; then it draws lines on top (isobars every 4 hPa, the coastline), each
# projected with `ortho.forward`. The rules, nearest the globe first:
#
# 1. temperature: a two-sided ramp with white at 0 °C;
# 2. pressure: a quiet blue-grey fill, the isobars carry the information (drawn from a lightly smoothed copy,
#    because stored pressure comes in steps of 0.64 hPa);
# 3. wind speed and 4. precipitable water: one hue each, dark to bright;
# 5. cloud: the opacity grows with the cloud cover, so clear sky is clear glass;
# 6. rain rate: a logarithmic scale, fully clear below 0.2 mm per hour.

# %%
coast = panes.coast_lines("50m")                 # Natural Earth coastline (a small download the first time)
six = panes.six_panes(F, CAP, coast, n=480)      # six dictionaries: the picture "tex" (480 x 480 x 4) and what the legend needs
fig, axs = plt.subplots(2, 3, figsize=(11, 7.4))
for a, P in zip(axs.ravel(), six):
    a.set_facecolor(panes.K["bg"]); a.imshow(P["tex"], extent=(-1, 1, -1, 1))
    a.set_title(f"{P['title']} ({P['unit']})"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4 · The globe's skin
# The globe needs one picture that wraps around it: a map in plain latitude × longitude (row 0 = 90°N, column 0 =
# 180°W). `panes.relief_texture` builds it from ETOPO 2022 heights: light clay for land, slate blue for sea, shaded
# as if lit from the north-west so that mountains show (the shading is exaggerated), with the coastline drawn in.

# %%
elev = gfs.relief()                              # heights in metres on a 10 arc-minute grid
skin = panes.relief_texture(elev, coast)         # 1440 x 2880 x 3, values 0..1
plt.figure(figsize=(10, 5)); plt.imshow(skin, extent=(-180, 180, -90, 90)); plt.xlabel("longitude (°E)"); plt.ylabel("latitude (°N)")
plt.show()

# %% [markdown]
# ## 5 · Globe and panes in 3-D
# `render.render_stack` is a small ray caster. For every pixel it follows a straight line from the eye and asks what
# the line hits: the globe (a quadratic equation) and each pane (one division: where the line meets the pane's
# plane). The globe is painted first, then the panes from the nearest to the globe outwards, each laid over the
# picture with its opacity. Rendering at twice the size and shrinking (`ss=2`) smooths every edge.
#
# The camera sits 42° west of the stack axis and 15° above it, so the panes fan out to the right of the globe.
# The panes are spaced 0.9 × the disc radius apart, enough for a strip of every pane to be seen.

# %%
stack = GlassStack(ds, CAP, place="India and Bangladesh", local=(5.5, "IST"), coast=coast, elev=elev, skin=skin)
cam = stack.fan_camera((1920, 1080))             # the fanned view (az = -42, el = 15)
bare = stack.render(six, cam, ss=1)              # globe and six panes, no labels yet
plt.figure(figsize=(11, 6.2)); plt.imshow(bare); plt.axis("off"); plt.show()

# %% [markdown]
# ## 6 · The finished poster
# `stack.poster` renders the same view with smooth edges and adds what a picture of data needs: a top strap with
# the title, a legend column with one numbered row per pane (the same number sits on a chip at the pane's upper
# right corner, found with `render.project`), the lead time with the valid time in UTC and local time, a timeline,
# and the credits.
#
# What to look for: the closed isobars on pane 2 over the south-west Bay of Bengal are a **low** (a region of low
# pressure that air spirals into). On the panes in front of it you find its bright wind, its plume of precipitable
# water, its cloud shield and its rain: one system, six views.

# %%
poster = stack.poster(OUT / "glass_stack.png", lead=LEAD, scale=POSTER_SCALE)
nb.show(poster)

# %% [markdown]
# ## 7 · The animation
# The film is one function of time, `stack.film_state(t)`: it returns the camera, the gap between the panes, each
# pane's opacity and lift, and the lead time shown, each as an eased blend between keyframe values. In 16 seconds:
# the globe glides towards the region (0-3 s), the panes fan out (3-5 s), the precipitable-water pane is lifted
# with an amber edge while the others dim (6-9 s), the forecast clock runs from +6 h to +102 h (9-14 s; the pane
# pictures are made every 6 hours and cross-faded in between), and the panes close up on the globe (14-16 s).

# %%
if MODE == "FULL":                               # the whole animation: 384 frames, 1280 x 720 px
    movie = stack.animate(OUT / "glass_stack.mp4", OUT / "frames_glass")
else:                                            # a short preview: every 4th frame, 960 x 540 px, 8 frames a second
    movie = C.preview(stack.animate, OUT / "glass_stack_preview.mp4", OUT / "frames_glass", thin=4, scale=0.75, fps=8)
nb.video(movie)

# %% [markdown]
# ## 8 · The inputs of the Remotion version
# The second film of this recipe (`glass_stack_remotion.mp4`: five panes over the whole hemisphere centred on
# 45°N 30°W, drawn with three.js) is rendered by the [Remotion](https://www.remotion.dev) project in
# `remotion/glass_stack` of the repository. This cell writes everything that project loads: the globe's skin, 40
# pane pictures (5 fields × 8 lead times, 900 × 900 px) and a small manifest, from the shipped hemisphere file
# (0.5°, same run).
#
# **The render itself needs Node.js on your own computer; it does not run in this notebook or on Colab.** Copy the
# folder written here to `remotion/glass_stack/public`, then follow `remotion/glass_stack/README.md`
# (`npm install`, then one `npx remotion render` command).

# %%
from gismaps.glass import remotion

if WORKER:
    manifest = None
else:
    manifest = remotion.write_inputs(OUT / "remotion_public")      # always from the shipped file (a live read is 240 MB)
    print("lead times (h):", manifest["leads_h"], "| North Atlantic low at +36 h:", manifest["atlantic_low"][5])
    nb.strip([OUT / "remotion_public" / f"{f}_5.png" for f in remotion.FIELDS])

# %% [markdown]
# ## Check the outputs
# The last cell checks that every output exists and is what it should be, and prints PASS or FAIL per check.

# %%
size = (int(1920 * POSTER_SCALE), int(1080 * POSTER_SCALE))
low = float(ds["mslp"].sel(lead=66).sel(latitude=slice(20, 5), longitude=slice(78, 95)).min()) if RUN == gfs.RUN else None
checks = [
    ("the poster exists", poster.exists()),
    (f"the poster is {size[0]} x {size[1]} px", nb.image_size(poster) == size),
    ("six fields at 17 lead times", len(ds.data_vars) == 6 and ds.sizes["lead"] == 17),
    ("six pane pictures, clear outside the disc", len(six) == 6 and all(float(P["tex"][0, 0, 3]) == 0 for P in six)),
    ("the centre of the disc shows the centre of the cap", abs(float(lat_px[180, 180]) - CAP["lat0"]) < 0.2 and abs(float(lon_px[180, 180]) - CAP["lon0"]) < 0.2),
    ("forward and inverse projection agree (Dhaka)", np.allclose(ortho.inverse(*map(np.asarray, ortho.forward(23.81, 90.41, 21.0, 87.0)[:2]), 21.0, 87.0), [23.81, 90.41], atol=1e-9)),
]
if low is not None:
    checks.append(("the low over the Bay of Bengal is there at +66 h (below 1000 hPa)", low < 1000.0))
if not WORKER:                                   # (the frame workers of the parallel FULL render make neither)
    checks.append(("the movie exists and is not empty", movie is not None and movie.exists() and movie.stat().st_size > 20_000))
    checks.append(("the Remotion inputs: manifest, relief and 40 pane pictures", len(list((OUT / "remotion_public").glob("*.png"))) == 41 and (OUT / "remotion_public" / "manifest.json").exists()))
nb.verify(checks)

# %% [markdown]
# ## Download (Google Colab)
# On Colab this zips the pictures and movies of the output folder and offers them as one download; anywhere else it
# only says where they are.

# %%
nb.download(OUT)
