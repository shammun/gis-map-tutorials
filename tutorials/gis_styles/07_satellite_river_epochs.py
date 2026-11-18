# %% [markdown]
# # 7 · Epoch-stacked footprints from satellite images (Sentinel-2, with and without Earth Engine)
#
# **The idea.** Tutorial 2 stacked the footprints of a storm taken from a reanalysis. Here the footprints are
# **measured from satellite images**: the water of a river in each dry season, one colour per year, oldest at the
# bottom and newest on top. Colours that still show are places the river has left.
#
# **This map.** The Jamuna (the Brahmaputra in Bangladesh), a braided river that redraws its channels every
# monsoon, in the dry seasons of 2019 to 2026.
#
# **Two routes to the same masks, both shown:**
#
# | | (a) Public STAC, no account (default) | (b) Google Earth Engine (account) |
# |---|---|---|
# | Where the work happens | your machine reads small windows of cloud-optimised GeoTIFFs | Google's servers |
# | Per year | median of the 4 clearest scenes | median of all clear scenes |
# | Needs | `pystac-client`, `rasterio` | `earthengine-api`, a Cloud project id |
#
# Steps: 1 the data and the place → 2 find scenes (STAC) → 3 read a window, water index → 4 river mask → 5 the same
# with Earth Engine → 6 compare the two routes → 7 the picture → 8 the animation → 9 exporting from
# Earth Engine to Drive on an exact grid.

# %% [markdown]
# ## Settings
# Everything you may want to change is in this one cell. The checks at the bottom **stop** the notebook with a plain
# message if a setting cannot work, before anything slow starts.
#
# `MODE = "FAST"` (the default, one to three minutes; right for Google Colab) writes the poster at 1920 × 1080 px and a short
# preview clip of the animation at 960 × 540 px. `MODE = "FULL"` writes the poster at 3840 × 2160 px and the whole
# animation at 1920 × 1080 px.
#
# **Google Earth Engine is optional.** Leave `USE_GEE = False` and the notebook runs from start to end with no key
# and no account (public STAC route). Set it to `True` only if you have an Earth Engine account and a Google Cloud
# project of your own, and put that project's id in `PROJECT_ID`.

# %%
import os
import sys

MODE = "FULL" if "--full" in sys.argv else os.environ.get("GIS_MODE", "FAST")   # "FAST" (preview) or "FULL" (poster size, whole animation)
OUT_DIR = os.environ.get("GIS_OUT", "")          # output folder; "" = /content/out on Google Colab, ./out anywhere else

USE_GEE = False                                  # <- True = also run the Google Earth Engine route (needs your own account)
PROJECT_ID = "your-gcp-project"                  # <- your own Google Cloud project id (only used when USE_GEE = True)
DRIVE_PATH = "/content/drive/MyDrive"            # where Google Drive is mounted in Colab (only for the export cells)
DRIVE_FOLDER = "jamuna_mndwi_40m"                # Drive folder for the exports
FILENAME_PREFIX = "MNDWI_JAMUNA_40m"             # exported files are named <prefix>_<year>.tif

from gismaps import nb                           # helpers: data loader, show(), video(), STOP checks, PASS / FAIL

nb.stop_unless(MODE in ("FAST", "FULL"), f"MODE = {MODE!r} must be FAST or FULL")
nb.stop_unless(isinstance(USE_GEE, bool), "USE_GEE must be True or False")
nb.stop_unless(not USE_GEE or PROJECT_ID != "your-gcp-project" or bool(os.environ.get("EE_PROJECT")),
               "USE_GEE = True needs your own Cloud project id in PROJECT_ID (or in the environment variable EE_PROJECT)")
nb.stop_unless(FILENAME_PREFIX.replace("_", "").isalnum() and DRIVE_FOLDER.replace("_", "").isalnum(),
               "DRIVE_FOLDER and FILENAME_PREFIX may hold only letters, digits and underscores")
POSTER_SCALE = 1.0 if MODE == "FAST" else 2.0    # the design size is 1920 x 1080; the poster is saved at this multiple
OUT = nb.out_dir(OUT_DIR)
print(f"settings OK | mode: {MODE} | output folder: {OUT} | Earth Engine route: {'on' if USE_GEE else 'off (no account needed)'}")

# %% [markdown]
# ## 1 · The data and the place
# The place, the years, the pixel size and the water threshold are in `gismaps/prepare_data.py` (`S2_AOI`, `S2_YEARS`,
# `S2_RES`, `S2_WATER`, `S2_MIN_KM2`), so the data builder and this tutorial cannot disagree. The cell prints them
# and **stops** if the reduced file was built with other values (`python -m gismaps.prepare_data sentinel` rebuilds it).

# %%
import numpy as np
import matplotlib.pyplot as plt

from gismaps import prepare_data as P                 # how the reduced file is made (and the live STAC readers)
from gismaps.styles import common as C, satellite     # the map-style code of this repository

print("area (W, S, E, N):", P.S2_AOI)
print("years:", P.S2_YEARS[0], "to", P.S2_YEARS[-1], "| season: 15 January to 5 March | pixel:", P.S2_RES, "m")
print("water: MNDWI >", P.S2_WATER, "| smallest piece kept:", P.S2_MIN_KM2, "km2")

data = nb.load("jamuna_s2")                      # built by `python -m gismaps.prepare_data sentinel` (route a)
masks = satellite.unpack(data)
nb.stop_unless(list(data["year"]) == list(P.S2_YEARS) and float(data["res_m"]) == P.S2_RES,
               "jamuna_s2.npz was built with other settings - run `python -m gismaps.prepare_data sentinel` again")
print("masks:", masks.shape, "| grid:", str(data["crs"]), "| bounds (m):", data["bounds_utm"])

# %% [markdown]
# ## 2 · Route (a): find the scenes with a STAC search
# A STAC catalogue is a searchable index of scenes. Element 84's Earth Search is public: no key, no account.
# We ask for Sentinel-2 L2A (surface reflectance) scenes over the box, in the dry season, with under 5 % cloud,
# on one MGRS tile (so every year lands on the same pixel grid), and take the clearest ones.
#
# This cell runs when `pystac-client` and `rasterio` are installed (they are in Colab after the setup cell) and the
# catalogue answers; otherwise it is skipped and the reduced file is used, so the notebook always runs to the end.

# %%
try:
    import pystac_client, rasterio              # noqa: F401, E401
    LIVE = True
except ImportError:
    LIVE = False
    print("pystac-client / rasterio not installed here: using the reduced file (the live cells are skipped)")

if LIVE:
    try:
        items = P.s2_pick(P.S2_YEARS[-1], n=4)
        for it in items:
            print(it.id, it.datetime.date(), f"cloud {it.properties['eo:cloud_cover']:.2f} %")
    except Exception as err:                     # noqa: BLE001  (the public catalogue is busy or unreachable)
        LIVE = False
        print("Earth Search did not answer (", str(err)[:80], "): using the reduced file")

# %% [markdown]
# ## 3 · Read only a window, then the water index
# A Sentinel-2 tile is 110 km across at 10 m: hundreds of megabytes per band. A cloud-optimised GeoTIFF lets
# `rasterio` read **only our window, at a coarser overview** (here 40 m) over HTTP: a few hundred kilobytes.
#
# Water is dark in the short-wave infrared and brighter in green, so the Modified Normalised Difference Water
# Index, `MNDWI = (green − SWIR) / (green + SWIR)`, is positive on water and negative on land.
#
# **A trap worth knowing.** Since January 2022 ESA adds 1000 to every L2A value. Some catalogue copies have
# removed it again, some have not, and the metadata flag was wrong for one of our eight years. `s2_window`
# therefore decides from the data (is anything darker than 1000?) and subtracts the offset if it is there. Without
# that, the 2022 index is squeezed towards zero and the river nearly vanishes.

# %%
index = data["mndwi100_latest"] / 100.0          # the stored median index of the newest year (used if the live read fails)
if LIVE:
    try:
        d = P.s2_window(items[0])                # green, swir16, scl (scene classification) for the window
        index = P.mndwi(d["green"], d["swir16"])
        print("window:", index.shape, "px | offset subtracted:", d["dn_offset"],
              "| clouds, shadows, no data flagged by SCL:", f"{100 * np.isin(d['scl'], (0, 1, 3, 8, 9, 10)).mean():.2f} %")
    except Exception as err:                     # noqa: BLE001
        print("the scene could not be read (", str(err)[:80], "): showing the stored index")

fig, axs = plt.subplots(1, 2, figsize=(9, 6.6))
axs[0].imshow(data["rgb_latest"]); axs[0].set_title(f"true colour, {data['rgb_date']}")
im = axs[1].imshow(index, cmap="RdBu", vmin=-0.6, vmax=0.6); axs[1].set_title("MNDWI (blue = water)")
for a in axs:
    a.axis("off")
plt.colorbar(im, ax=axs[1], shrink=0.7); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4 · From index to "the river"
# Two decisions, both stated on the final figure:
# * **threshold** MNDWI > 0.1 (a little above 0 keeps wet sand and damp fields out);
# * **size**: keep only connected water of at least 5 km². In February the fields beside the river are
#   flooded for rice; they are water, but they are not the river.
#
# And one decision about time: **the median of several scenes, not one scene**. The river falls through the dry
# season, so single scenes a few weeks apart gave quite different water areas (96 to 248 km² when pieces down to 1 km² were kept). The median
# index of the four clearest scenes is much steadier.

# %%
mask = P.river_mask(index)
print(f"water by threshold alone: {(index > P.S2_WATER).sum() * P.S2_RES**2 / 1e6:.0f} km2;",
      f"river (connected pieces of 5 km2 or more): {mask.sum() * P.S2_RES**2 / 1e6:.0f} km2")
area = masks.reshape(len(masks), -1).sum(1) * P.S2_RES**2 / 1e6
for y, a, n in zip(data["year"], area, data["scenes"]):
    print(y, f"{a:5.0f} km2  (median of {n} scenes)")

# %% [markdown]
# ## 5 · Route (b): sign in to Google Earth Engine (optional)
# **Skip this part unless you set `USE_GEE = True` in the settings cell**: with `USE_GEE = False` the next four
# Earth Engine cells do nothing and the notebook goes on with the public route.
#
# Earth Engine holds the whole Sentinel-2 archive and does the compositing on its side. The recipe is the same:
# filter by place, season and cloud; mask clouds with the scene classification (SCL); take the median;
# `normalizedDifference(["B3", "B11"])`; download a small array.
#
# In **Colab**, the cell below mounts Drive (for the exports of step 9), signs you in (`ee.Authenticate()`) and
# starts Earth Engine with your project (`ee.Initialize(project=PROJECT_ID)`). On your own machine, run
# `earthengine authenticate` once and set the environment variable `EE_PROJECT`. Never write a real project id or
# a credential into a notebook that you share.

# %%
if USE_GEE:
    import ee
    try:                                         # Colab: mount Drive, sign in, then Earth Engine
        from google.colab import auth, drive
        drive.mount("/content/drive")
        os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
        auth.authenticate_user()
        try:
            ee.Authenticate(scopes=["https://www.googleapis.com/auth/cloud-platform",
                                    "https://www.googleapis.com/auth/earthengine",
                                    "https://www.googleapis.com/auth/devstorage.full_control"])
            ee.Initialize(project=PROJECT_ID)
            print("Earth Engine initialized successfully.")
        except Exception as e:
            print("Authentication failed:", e)
    except ImportError:                          # your own machine: credentials already stored by `earthengine authenticate`
        ee.Initialize(project=os.environ.get("EE_PROJECT") or PROJECT_ID)
        print("Earth Engine initialized with the stored credentials.")

# %%
def gee_periods(years=P.S2_YEARS, start="01-15", end="03-06"):
    """One (label, start, end) per dry season. For monthly work: one tuple per month instead."""
    return [(str(y), f"{y}-{start}", f"{y}-{end}") for y in years]


def gee_mndwi(region, start, end):
    """Median MNDWI x 100 (int8) of the clear Sentinel-2 scenes of one period, clouds masked with SCL."""
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)))
    clear = col.map(lambda im: im.updateMask(im.select("SCL").remap([0, 1, 3, 8, 9, 10], [0] * 6, 1)))
    return clear.median().normalizedDifference(["B3", "B11"]).multiply(100).round().toInt8().unmask(-128), col.size()


if USE_GEE:
    import io
    import urllib.request
    w, s, e, n = (float(v) for v in data["bounds_utm"])
    region = ee.Geometry.Rectangle([w, s, e, n], proj=str(data["crs"]), geodesic=False)
    gee_masks = []
    for label, start, end in gee_periods():
        img, count = gee_mndwi(region, start, end)
        # a small array straight to memory: a download URL takes the box and the pixel size (exports: see step 9)
        url = img.getDownloadURL({"region": region, "crs": str(data["crs"]), "scale": P.S2_RES, "format": "NPY"})
        raw = np.load(io.BytesIO(urllib.request.urlopen(url).read()))
        arr = raw[raw.dtype.names[0]] if raw.dtype.names else raw
        gee_masks.append(P.river_mask(arr / 100.0))
        print(label, "scenes:", count.getInfo(), "| river:", f"{gee_masks[-1].sum() * P.S2_RES**2 / 1e6:.0f} km2")
    gee_masks = np.array(gee_masks)

# %% [markdown]
# ## 6 · Do the two routes agree?
# They use the same satellite, season, index, threshold and grid; they differ in how many scenes go into the
# median. The repository holds the Earth Engine result made on 18 November 2026 (`data/jamuna_s2_gee.npz`), so
# the comparison below also runs without an account.

# %%
if USE_GEE:
    gee_scenes = ["all clear"] * len(gee_masks)
else:
    try:
        g = nb.load("jamuna_s2_gee")             # the stored Earth Engine result (downloaded if it is not on disk)
        gee_masks, gee_scenes = satellite.unpack(g), g["scenes"]
    except FileNotFoundError as err:
        gee_masks = None
        print("the stored Earth Engine result is not available:", err)
if gee_masks is not None:
    print(" year   STAC (4 scenes)   Earth Engine (scenes)   pixels that agree")
    for k, y in enumerate(data["year"]):
        a1, a2 = masks[k].sum() * P.S2_RES**2 / 1e6, gee_masks[k].sum() * P.S2_RES**2 / 1e6
        print(f" {y}      {a1:5.0f} km2         {a2:5.0f} km2 ({gee_scenes[k]})        {100 * (masks[k] == gee_masks[k]).mean():.1f} %")

# %% [markdown]
# ## 7 · The picture
# `satellite.RiverEpochs` stacks the yearly masks (oldest first, `plasma` colours, the same `epochs.stack` as
# tutorial 2) on a dimmed copy of the newest true-colour image, and puts that image beside it. The legend is the
# clock and carries the area of each year as a bar.

# %%
river = satellite.RiverEpochs(data)
river.frame(OUT / "07_solo.png", upto=4, solo=3)
poster = river.poster(OUT / "river_epochs.png", scale=POSTER_SCALE)
nb.show(OUT / "07_solo.png")
nb.show(poster)

# %% [markdown]
# ## 8 · The animation
# It opens on the finished stack, shows each year alone, then builds the stack again and holds.

# %%
if MODE == "FULL":                               # the whole animation, 1920 x 1080 px
    movie = river.animate(OUT / "river_epochs.mp4", OUT / "frames_07")
else:                                            # a short preview: every frame, 960 x 540 px, 12 frames a second
    movie = C.preview(river.animate, OUT / "river_epochs_preview.mp4", OUT / "frames_07", thin=1)
nb.video(movie)

# %% [markdown]
# ## 9 · Exporting the yearly index from Earth Engine to Drive on an exact grid (optional)
# A download URL is right for small arrays like ours. To keep the yearly index as GeoTIFFs, export them to Google
# Drive. Four habits make the exports dependable:
#
# 1. **Fix the grid**: pass `crs`, `crsTransform` and `dimensions`, and **no `region=`**. With `region=` Earth
#    Engine fits the pixels to the region and can stretch them; with the transform every file lands on exactly the
#    same pixels.
# 2. **Skip what exists**, so the cell can be run again after a time-out.
# 3. **Count** started, skipped and failed tasks.
# 4. **Check the files** when the Tasks page shows them finished: open each with `rasterio` and compare CRS,
#    transform, size and value range with what you asked for.
#
# The two cells below do that for the yearly MNDWI. They need `USE_GEE = True` and Colab with Drive mounted;
# with `USE_GEE = False` they do nothing.

# %%
if USE_GEE:                                      # DRIVE_PATH, DRIVE_FOLDER, FILENAME_PREFIX: see the settings cell
    w, s, e, n = (float(v) for v in data["bounds_utm"])
    height, width = masks.shape[1:]
    crs_transform = [P.S2_RES, 0, w, 0, -P.S2_RES, n]                # pixel size, 0, left edge, 0, -pixel size, top edge
    out_dir = f"{DRIVE_PATH}/{DRIVE_FOLDER}"
    started = skipped = failed = 0
    for label, start, end in gee_periods():
        name = f"{FILENAME_PREFIX}_{label}"
        if os.path.exists(f"{out_dir}/{name}.tif"):
            skipped += 1
            continue
        try:
            img, _ = gee_mndwi(ee.Geometry.Rectangle([w, s, e, n], proj=str(data["crs"]), geodesic=False), start, end)
            ee.batch.Export.image.toDrive(image=img, description=name, folder=DRIVE_FOLDER, fileNamePrefix=name,
                                          crs=str(data["crs"]), crsTransform=crs_transform,
                                          dimensions=f"{width}x{height}", maxPixels=1e9).start()   # no region=
            started += 1
        except Exception as err:
            failed += 1
            print("FAILED", name, err)
    print(f"started {started}, skipped (already on Drive) {skipped}, failed {failed}")

# %%
if USE_GEE:                                      # RUN AFTER THE EXPORTS FINISH (see the Tasks page of Earth Engine)
    import glob
    import rasterio
    files = sorted(glob.glob(f"{DRIVE_PATH}/{DRIVE_FOLDER}/{FILENAME_PREFIX}_*.tif"))
    print(f"{len(files)} file(s) found (expect {len(P.S2_YEARS)})")
    bad = 0
    for f in files:
        with rasterio.open(f) as src:
            on_grid = (str(src.crs) == str(data["crs"]) and (src.width, src.height) == (width, height)
                       and abs(src.transform.a - P.S2_RES) < 1e-6 and abs(src.transform.c - w) < 1e-6
                       and abs(src.transform.f - n) < 1e-6)
            a = src.read(1)
        v = a[a > -128]
        bad += not on_grid
        print(("on grid " if on_grid else "OFF GRID"), f.split("/")[-1], f"MNDWI x 100: min {v.min()}, median {np.median(v):.0f}, max {v.max()}")
    print(f"{len(files) - bad} of {len(files)} exactly on the grid")

# %% [markdown]
# ## Check the outputs
# The last cell checks that every output exists and is what it should be, and prints PASS or FAIL per check.

# %%
size = (int(1920 * POSTER_SCALE), int(1080 * POSTER_SCALE))
checks = [
    ("the poster exists", poster.exists()),
    (f"the poster is {size[0]} x {size[1]} px", nb.image_size(poster) == size),
    ("the sample frame exists", (OUT / "07_solo.png").exists()),
    ("one river mask per year", len(masks) == len(P.S2_YEARS)),
    ("every year has a river of 50 to 400 km2", bool(((area > 50) & (area < 400)).all())),
]
if gee_masks is not None:
    agree = float(np.mean([(masks[k] == gee_masks[k]).mean() for k in range(len(masks))]))
    checks.append((f"the two routes agree on {100 * agree:.1f} % of the pixels (more than 90 %)", agree > 0.90))
if not os.environ.get("GIS_PART"):               # (GIS_PART marks a frame worker of the parallel FULL render)
    checks.append(("the movie exists and is not empty", movie is not None and movie.exists() and movie.stat().st_size > 20_000))
nb.verify(checks)

# %% [markdown]
# ## Download (Google Colab)
# On Colab this zips the pictures and movies of the output folder and offers them as one download; anywhere else it
# only says where they are.

# %%
nb.download(OUT)
