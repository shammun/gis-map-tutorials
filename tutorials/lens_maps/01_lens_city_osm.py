# Lens-map tutorial 1 - a city poster from OpenStreetMap.  Plain script; `tools/build_notebooks.py` turns it into the notebooks.
# Run:  python tutorials/lens_maps/01_lens_city_osm.py        (writes into ./out/)

# %% [markdown]
# # Lens map 1 · One city, many realities
#
# **What you will make:** a poster of a city with three round *lenses*, and its 24-second video. Each lens draws the same ground another way:
#
# | Lens | What it shows | OpenStreetMap tags used |
# |---|---|---|
# | **LAND USE** | how the ground is used (homes, shops, industry, parks ...) | `landuse`, `leisure`, `amenity` polygons |
# | **PUBLIC REALM** | everyday life: parks and public toilets, taps, benches, playgrounds | park polygons + `amenity` points |
# | **CITY MORPHOLOGY** | the built form: every building footprint and street | `building`, `highway` |
#
# We draw **Brisbane** first, then **Dhaka** and **Richmond (Virginia)**.
#
# **Why a lens map?** No single map can show land use *and* amenities *and* buildings without becoming unreadable.
# A lens map keeps one quiet base map and opens small windows onto other layers of the *same place, on the same pixel
# grid*, so the windows line up with the map around them.
#
# **Data:** © OpenStreetMap contributors (ODbL), through the public Overpass API. No key, no account.
# **Style** after Fan Mo, #30DayMapChallenge 2026 Day 14 ("Borgesian map"); the code and data here are original.
#
# **Time:** about 2 minutes with the pre-extracted data (the default); about 35 minutes if you ask the public
# OpenStreetMap server for everything yourself (Section 3 shows how).

# %% [markdown]
# ## Settings
# Everything you may want to change is in this one cell. The checks at the bottom **stop** the notebook with a plain
# message if a setting cannot work, before anything slow starts.

# %% CHANGE IF YOU LIKE - RUN :: Settings :: The city drawn first and where the data come from. Run it after every change.
import os

CITY = "brisbane"  # the city drawn first, and the city of the video: "brisbane", "dhaka" or "richmond"
QUICK = os.environ.get("LENS_QUICK", "1") == "1"  # the video. True (default, right for Colab): a small preview, 720 x 1018 px at 15 frames a second. False = FULL: 1440 x 2036 px at 30 frames a second (about 6 minutes)
LIVE_OSM = False  # False = the shipped extract (fast). True = ask the public Overpass server for every layer (about 35 min)



def stop_unless(ok: bool, message: str) -> None:
    """STOP check: halt here, with a plain message, when a setting cannot work."""
    if not ok:
        raise ValueError("STOP - " + message)


stop_unless(CITY in ("brisbane", "dhaka", "richmond"), f"CITY = {CITY!r} is not one of brisbane, dhaka, richmond")
stop_unless(isinstance(LIVE_OSM, bool), "LIVE_OSM must be True or False")
print(f"settings OK | city: {CITY} | OpenStreetMap: {'live download' if LIVE_OSM else 'shipped extract'}")

# %% [markdown]
# ## 0 · Setup
# The poster code lives in this repository's small Python package, `gismaps` (folder `gismaps/lens/`): install it with
# `pip install -e .` in a clone, or `pip install git+https://github.com/shammun/gis-map-tutorials` anywhere else
# (the Colab notebook does that in its first cell). The small data files come from the repository's `data/lens_maps/`
# folder, or are downloaded once from GitHub. The posters and the tutorials run **the same functions**, so what you
# learn here is how the posters were made.

# %% RUN :: Setup :: Loads the poster modules and two small helpers.
import io
import os
import sys
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # keep the page clean: library deprecation notes are not part of the recipe

from gismaps import data as gdata, nb  # this repository's package: finds the data files; notebook helpers

OUT = nb.out_dir(os.environ.get("LENS_OUT"))  # posters are written here: /content/out on Google Colab, ./out elsewhere


def data_file(name: str) -> Path:
    """One of the small pre-extracted data files: from the repository's data/lens_maps/ folder, or downloaded once
    from the repository on GitHub when it is not on disk (Google Colab)."""
    return gdata.path("lens_maps", name)


show = nb.picture  # shows a picture (file or array) inline, as wide as the page allows; a plain script only names it


import numpy as np  # noqa: E402

from gismaps.lens import city  # noqa: E402  the poster: read, draw layers, furniture
from gismaps.lens import common as lc  # noqa: E402  the lens rule (composite), pixel canvas, label blocks
from gismaps.lens import osm  # noqa: E402  polite Overpass queries
from gismaps.lens.cities import CITIES, city as city_cfg  # noqa: E402  one small dict per city

print("fonts:", lc.SERIF, "+", lc.SANS, "| cities configured:", ", ".join(CITIES))

# %% [markdown]
# ## 1 · A city is one small dictionary
# Everything that changes from city to city is in one place. The important fields:
#
# * `lat0, lon0` - the point that sits at poster pixel (985, 1420), a little above the middle of the 2000 × 2828 px sheet;
# * `mpp` - metres per pixel (3.3 m/px shows 6.6 × 9.3 km);
# * `lenses` - where the three lens centres are, as pixel offsets from that point; `radius` in pixels;
# * `detail` - the box (south, west, north, east) that holds the three lenses: we fetch everything in it;
# * `context` - a larger box for the muted water and main roads outside the lenses.

# %% RUN :: The city dictionary :: Prints the fields that define the poster of the chosen city.
cfg = city_cfg(CITY)  # a copy of that city's entry, with its key added
for k in ("name", "country", "lat0", "lon0", "mpp", "lenses", "radius", "detail", "context", "tiles", "public"):
    print(f"{k:8s} = {cfg[k]}")
r_km = cfg["radius"] * cfg["mpp"] / 1000  # lens radius: pixels x metres per pixel
print(f"\nEach lens is {r_km:.2f} km in radius, so it shows {np.pi * r_km**2:.1f} km² of the city.")

# %% [markdown]
# ## 2 · From longitude and latitude to poster pixels
# A city is small, so the simplest honest map works: a **local equirectangular** map. One degree of latitude is
# about 110.574 km everywhere; one degree of longitude is 111.320 km × cos(latitude), because the meridians close in
# toward the poles.
#
# $$x = 985 + \frac{(\lambda-\lambda_0)\,111\,320\,\cos\varphi_0}{\text{mpp}},\qquad
#   y = 1420 - \frac{(\varphi-\varphi_0)\,110\,574}{\text{mpp}}$$
#
# ($\lambda$ = longitude, $\varphi$ = latitude, both in degrees; the minus sign is there because pixel rows count
# downward while latitude counts upward.) Over 10 km the error is below 0.1 %.

# %% RUN :: Longitude and latitude to pixels :: One point by hand, then the poster's function, then a check that they agree.
# --- from scratch: one point, one step per line ---
lon, lat = 153.0260, -27.4698  # a point in central Brisbane (degrees east, degrees north)
lat0, lon0, mpp = cfg["lat0"], cfg["lon0"], cfg["mpp"]  # the map's anchor point and scale
m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))  # metres in one degree of longitude at this latitude
m_per_deg_lat = 110_574.0  # metres in one degree of latitude
east_m = (lon - lon0) * m_per_deg_lon  # how far east of the anchor, in metres
north_m = (lat - lat0) * m_per_deg_lat  # how far north of the anchor, in metres
x = 985 + east_m / mpp  # pixels to the right of the left edge
y = 1420 - north_m / mpp  # pixels down from the top edge (north is up, so subtract)
print(f"{east_m:.0f} m east, {north_m:.0f} m north of the anchor -> pixel ({x:.1f}, {y:.1f})")

# --- the function the poster uses, and a check that both agree ---
pr = city.Proj(cfg)
np.testing.assert_allclose((x, y), pr.xy(lon, lat), rtol=1e-12)  # same formula, so they agree to rounding error
print("Proj.xy agrees. 1 km on the ground =", round(1000 / mpp), "pixels.")

# %% [markdown]
# ## 3 · Asking OpenStreetMap, politely
# The [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API) answers questions such as *"every way tagged
# `building` inside this box"*. It is a free, shared server, so we follow four rules (all built into
# `gismaps.lens.osm.fetch_city`):
#
# 1. **One query at a time**, with a pause (8 s) between queries, and a `User-Agent` that says what we are.
# 2. **Cache everything.** Every answer is saved as a file; a file on disk is never asked for again.
# 3. **Tile the heavy layer.** All buildings of a 7-km box in one query times out; 3 × 3 tiles do not (Dhaka needs 4 × 4).
#    A building on a tile edge comes back twice and is dropped by its OSM id.
# 4. **Expect "busy".** The server answers HTTP 429 / 502 / 504 when loaded; wait (30 s, 60 s ...) and try again.
#
# Here are the questions for Brisbane, in Overpass QL:

# %% RUN :: The Overpass questions :: Prints the queries; nothing is downloaded yet.
queries = osm.queries(cfg)  # layer name -> (Overpass QL text, a size guess)
print(len(queries), "queries:", ", ".join(queries))
print("\nThe roads query:\n ", queries["roads"][0])
print("\nOne of the nine building tiles:\n ", queries["buildings_4"][0])

# %% [markdown]
# **A small live test.** One tiny query (the main roads within about 300 m of the anchor) shows the route works
# from where you are. If the server is busy we say so and carry on: the rest of the tutorial does not depend on it.

# %% OPTIONAL :: Live test of the OpenStreetMap server :: One tiny query. If the server is busy the cell says so and the tutorial carries on.
south, west, north, east = lat0 - 0.003, lon0 - 0.003, lat0 + 0.003, lon0 + 0.003  # a box about 600 m across
tiny = f'[out:json][timeout:25];way["highway"~"^(primary|secondary|tertiary)$"]({south},{west},{north},{east});out geom;'
try:
    f = OUT / "overpass_live_test.json"
    osm.overpass(tiny, f, attempts=1, timeout=40, log=lambda *a, **k: None)  # one polite try, no retries here
    import json

    ways = json.loads(f.read_text(encoding="utf-8"))["elements"]
    print(f"Overpass answered: {len(ways)} road ways; the first is tagged {ways[0]['tags'].get('highway')!r} "
          f"with {len(ways[0]['geometry'])} points.")
except Exception as e:  # noqa: BLE001
    print("Overpass is busy right now (", str(e)[:70], ") - no problem, we use the pre-extracted file below.")

# %% [markdown]
# **Getting the whole city.** Two ways, same result:
#
# * `LIVE_OSM = False` (the default in the settings cell): load the **pre-extracted file** that ships with this tutorial. It is the reduced
#   result of exactly the queries above (coordinates kept to 6 decimals, clipped to the ground the lenses can reach).
# * `LIVE_OSM = True`: run all 14 queries yourself with `osm.fetch_city` (about 55 MB and 35 minutes for Brisbane on
#   the public server; the answers are cached in `lens_cache/`, so a second run is instant). If the server stays
#   busy, the cell falls back to the pre-extracted file and tells you.

# %% RUN :: Load the city layers :: Uses LIVE_OSM from the settings cell: the shipped extract (fast) or your own download (about 35 minutes).
def load_city(cfg: dict, live: bool = False) -> dict:
    """The layers of one city in poster pixels: water, land use, roads, rails, buildings, amenity points."""
    pr = city.Proj(cfg)
    if live:
        try:
            folder = Path("lens_cache") / f"OSM-overpass-{cfg['key']}"
            osm.fetch_city(cfg, folder, pause=8.0)  # polite, cached, tiled, retried; resumes if interrupted
            return city.read_raw(cfg, pr, folder)  # raw Overpass answers -> shapely polygons in pixels
        except Exception as e:  # noqa: BLE001
            print("Overpass did not finish (", str(e)[:80], ") - using the pre-extracted file instead.")
    return city.read_derived(cfg, pr, data_file(f"{cfg['key']}_lens_layers.json.gz"))


L = load_city(cfg, LIVE_OSM)
print({k: (len(v) if not isinstance(v, dict) else {a: len(b) for a, b in v.items()}) for k, v in L.items()})

# %% [markdown]
# ## 4 · From OSM elements to shapely polygons
# Overpass returns *elements*. Two kinds matter here:
#
# * a **way**: a list of points. If the first and last point are the same, it is an area (a building, a park);
# * a **relation** of type *multipolygon*: several ways with a role, `outer` or `inner`. A park with a lake in it is
#   an outer ring with an inner ring (a hole). The outer ways may be several pieces that only join end to end.
#
# `city.polys` turns either into [shapely](https://shapely.readthedocs.io) polygons: a closed way directly;
# a relation by **polygonizing** the outer pieces and **subtracting** the polygonized inner pieces.
# A hand-made example, in the exact shape Overpass uses:

# %% RUN :: OSM elements to shapely polygons :: A hand-made way and a multipolygon relation with a hole.
def pt(lon, lat):  # one Overpass point
    return {"lon": lon, "lat": lat}


square = [pt(153.020, -27.470), pt(153.022, -27.470), pt(153.022, -27.472), pt(153.020, -27.472), pt(153.020, -27.470)]
hole = [pt(153.0207, -27.4707), pt(153.0213, -27.4707), pt(153.0213, -27.4713), pt(153.0207, -27.4713), pt(153.0207, -27.4707)]
way = {"type": "way", "id": 1, "tags": {"building": "yes"}, "geometry": square}  # a closed way = one polygon
relation = {"type": "relation", "id": 2, "tags": {"leisure": "park", "type": "multipolygon"},  # a park ...
            "members": [{"type": "way", "role": "outer", "geometry": square[:3]},  # ... whose outer ring comes in two pieces
                        {"type": "way", "role": "outer", "geometry": square[2:]},
                        {"type": "way", "role": "inner", "geometry": hole}]}  # ... with a hole (a pond, say)
a = city.polys(way, pr)[0]  # shapely Polygon in poster pixels
b = city.polys(relation, pr)[0]
m2 = cfg["mpp"] ** 2  # one pixel covers mpp x mpp square metres
print(f"way      -> {a.geom_type}, {a.area * m2:,.0f} m²")
print(f"relation -> {b.geom_type}, {b.area * m2:,.0f} m², holes: {len(b.interiors)}")
assert b.area < a.area and len(b.interiors) == 1  # the hole was cut out

# %% [markdown]
# **Classifying land use.** OSM has hundreds of tag values; the poster has eleven classes. One table per tag key maps
# values to classes, tried in a fixed order: `landuse` → `leisure` → `amenity` → `natural`. A value that is in no
# table is **not drawn** and is printed, so that nothing lands in a wrong class by accident.

# %% RUN :: Land-use classes and feature counts :: The tag tables, and how many features each layer holds.
print("classes:", {k: v[0] for k, v in city.CLASSES.items()})
print("landuse  ->", city.LANDUSE)
print("leisure  ->", city.LEISURE)
print("amenity  ->", city.AMENITY)
counts = city.count(cfg, L)  # how many features each layer holds
print("\nBrisbane:", counts)

# %% [markdown]
# ## 5 · One extract, three drawings
# `render_layers` draws **four pictures of the whole sheet** with matplotlib, all on the same pixel grid
# (`lc.canvas` makes a figure whose data coordinates *are* output pixels, so nothing is ever rescaled):
#
# * `base` - what you see outside the lenses: water and main roads only, almost black, so it never competes;
# * `use`, `pub`, `mor` - the three lens pictures.
#
# Here are the three lens pictures whole, before any lens cuts them (the pre-extracted file only holds the ground the
# lenses can reach, which is why the corners are empty).

# %% RUN :: One extract, three drawings :: Draws the four full-sheet layers and shows the three lens pictures whole.
layers = city.render_layers(cfg, L)  # dict of float RGB arrays, 2828 x 2000 x 3, values 0..1
print({k: v.shape for k, v in layers.items()})
crop = (slice(560, 2360), slice(40, 1960))  # rows, columns of the part that holds the lenses
strip = np.concatenate([layers[k][crop][::3, ::3] for k in ("use", "pub", "mor")], axis=1)  # side by side, 1/3 size
show((strip * 255).astype(np.uint8), 1100)

# %% [markdown]
# ## 6 · A lens is a mask
# A lens is a disc. Inside the disc we show the lens picture, outside it the base map. With a mask $a$ that is 1
# inside and 0 outside (and in between on the one-pixel edge, so the edge is smooth):
#
# $$\text{out} = \text{base}\,(1-a) + \text{lens}\cdot a$$
#
# **Where two lenses overlap** there is a choice, one number called `overlap`:
#
# * `overlap = 0.5` - the upper lens is mixed half and half with the lower one. This is the poster look. It is fine
#   here because these are *categorical* pictures: nobody reads a number off a blended colour.
# * `overlap = 0` - the lower lens keeps its exact colours. Use this for any **data field with a colour bar**
#   (tutorial 2): a 50/50 mix of two colour maps matches neither colour bar.
#
# From scratch first, then the library function, then a check that they agree.

# %% RUN :: A lens is a mask :: The blend formula from scratch, the library call, and a check that they agree.
# --- from scratch ---
h, w = layers["base"].shape[:2]  # 2828 rows, 2000 columns
discs = city.lens_discs(cfg)  # [(centre x, centre y, radius)] for land use, public realm, morphology
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)  # the row and column number of every pixel


def disc_mask(cx, cy, r):
    d = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)  # distance of each pixel centre from the lens centre
    return np.clip(r - d + 0.5, 0, 1)  # 1 inside, 0 outside, a smooth ramp across the one-pixel edge


masks = [disc_mask(*d) for d in discs]
out = layers["base"].copy()  # start from the base map
overlap = 0.5  # the poster look for categorical maps
for j, key in enumerate(("use", "pub", "mor")):  # lenses in drawing order
    a = masks[j].copy()  # this lens's own disc
    for i in range(j):  # wherever an earlier lens already lies ...
        a *= 1 - (1 - overlap) * masks[i]  # ... draw this one at `overlap` strength only
    out = out * (1 - a[..., None]) + layers[key] * a[..., None]  # the blend formula, for all three colour channels

# --- the function the poster uses ---
lenses = city.make_lenses(cfg, layers)  # three lc.Lens objects (picture, centre, radius, rim colour, overlap 0.5)
img = lc.composite(layers["base"], lenses)
np.testing.assert_allclose(out, img, atol=1e-6)  # same arithmetic in float32, so equal to rounding error
print("scratch composite == lc.composite. Lens order:", [d[:2] for d in discs])
show((img[crop][::2, ::2] * 255).astype(np.uint8), 900)

# %% [markdown]
# ## 7 · Furniture: the title, three label blocks, legend, scale bar, north arrow, credits
# `city.furniture` adds everything that is not map; `city.poster` runs the whole chain
# (`count` → `render_layers` → `composite` → rims → `furniture`) and saves a 2000 × 2828 px PNG.
#
# The **label budget** is small on purpose: one block per lens (a bar in the rim colour, a CAPS name, an italic theme,
# two short lines) with one straight leader to a fixed point on its rim; **no place names on the map** (the header
# names the city). The legend prints how many of each amenity OSM holds, because the count is the finding.

# %% RUN :: The poster :: Writes lens_<city>.png (2000 x 2828 px) into the output folder.
out_png = OUT / f"lens_{CITY}.png"
counts = city.poster(cfg, L, out_png)  # also prints the counts
if CITY == "brisbane":  # the numbers printed on the gallery poster
    assert counts["points_detail_box"] == {"toilets": 164, "drinking_water": 516, "bench": 1553, "playground": 53}
show(out_png)

# %% [markdown]
# ## 8 · Why it looks the way it does
#
# | Choice | Reason |
# |---|---|
# | Black sheet, base map almost black | the lenses are the subject; the base only says "the city goes on" (the river shape orients the reader) |
# | Land use: light, flat, well-separated hues | a categorical map: classes must be told apart, not ranked. Residential (the most ground) gets the quietest colour |
# | "No land-use polygon in OSM" has its own dark swatch | honest emptiness: unmapped ground is shown as unmapped, not as some class |
# | Public realm: dark brown ground, green parks, small bright symbols | points must pop; each amenity gets its own shape *and* colour, so the map does not rely on colour alone |
# | Morphology: light buildings on dark green-grey, thin white streets | a figure-ground drawing: form only, no categories |
# | Rim colours pink / amber / green, repeated on the label bars | the rim colour is the link between a lens and its label, so no label has to sit inside a lens |
# | Lens order: land use, public realm, morphology | the overlaps mix neighbours; the most detailed drawing (buildings) goes on top, where it survives a 50 % mix |
# | `overlap = 0.5` | the poster look; allowed because no colour here is a measured value |
# | Scale bar and north arrow | a 10-km city map is true to scale and north-up everywhere, so both are honest here |
# | Credit block | the map's author, the data licence (© OpenStreetMap contributors, ODbL) and the style credit are always printed |

# %% [markdown]
# ## 9 · Another city: Dhaka and Richmond
# The other two posters need **no new code**, only their dictionary. What had to change tells you what to look for
# in any city:
#
# * **Dhaka** is far denser (132,000 buildings in the box), so buildings are fetched in 4 × 4 tiles and `mpp` is 3.6.
#   OSM holds only a handful of benches there, so the public-realm symbols are *places of worship* and *marketplaces*
#   first. An extra land-use class, "Military & border guard", keeps a large compound out of "Construction".
#   Much of the ground has no land-use polygon at all: that dark colour is the finding, not a defect.
# * **Richmond**: the two upper lenses are swapped, because OSM's toilets, taps and benches cluster in the west of
#   the city. Put each lens where its layer has something to say.
#
# Both count the amenities **inside the public-realm lens** (`count_area="lens"`): a number the reader can check by
# counting the symbols.

# %% RUN :: The other two cities :: Same code, another dictionary.
for key in [k for k in CITIES if k != CITY]:  # the cities not drawn yet
    c = city_cfg(key)
    Lc = load_city(c, LIVE_OSM)
    png = OUT / f"lens_{key}.png"
    n = city.poster(c, Lc, png)
    print(f"{c['name']}: {n['buildings']:,} buildings loaded, {n['buildings_in_morphology_lens']:,} in the morphology lens")
    show(png, 600)

# %% [markdown]
# ## 10 · What the map does not know
# A lens map of OSM is a map of *what volunteers have mapped*. Two numbers per city make that concrete: the share of
# the land-use lens with **no land-use polygon**, and the benches per km² in the public-realm lens. They differ a
# great deal, and that difference is about the map, not (only) about the cities.

# %% RUN :: What the map does not know :: Share of unmapped ground and amenities per km² in each city.
ground = lc.hexrgb("#2b2130")  # the colour of "no land-use polygon in OSM"
for key in ("brisbane", "dhaka", "richmond"):
    c = city_cfg(key)
    Lc = load_city(c)
    use = city.render_layers(c, Lc)["use"]  # the land-use picture
    cx, cy, r = city.lens_discs(c)[0]  # the land-use lens
    yy, xx = np.mgrid[0:use.shape[0], 0:use.shape[1]]
    inside = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) <= r  # pixels inside that lens
    empty = np.abs(use - ground).sum(-1) < 0.02  # pixels that still show the bare ground colour
    area_km2 = np.pi * (r * c["mpp"] / 1000) ** 2  # area of one lens
    n = city.count(c, Lc)["points_in_public_lens"]
    print(f"{c['name']:9s} no land-use polygon: {100 * (empty & inside).sum() / inside.sum():4.1f} % of the lens | "
          f"benches: {n.get('bench', 0) / area_km2:5.1f} per km² | toilets: {n.get('toilets', 0) / area_km2:4.1f} per km²")

# %% [markdown]
# ## 11 · The video
# In 24 seconds the lenses open one by one over the muted base map, the label blocks and the legend arrive, then the
# three lenses **trade places** round the triangle (every piece of ground is seen three ways) and come home.
#
# The video is made from the poster's own pieces, so it cannot drift from the poster:
#
# 1. every map layer is drawn **once** with the poster's functions, then shrunk to the video size;
# 2. every piece of furniture (title, label blocks, leaders, legend, credits) is drawn once on a transparent canvas
#    and cropped to a small "sprite";
# 3. per frame only numpy works: the lens rule (`lc.composite`) with this frame's lens centres, radii and opacities,
#    soft rims, then the sprites pasted with this frame's opacities;
# 4. the frames are piped straight to ffmpeg (H.264); no frame files are written.
#
# The video is made for the city named by `CITY` in the settings cell: set `CITY = "dhaka"` or `"richmond"` and run
# the notebook again for the other two. `QUICK = False` gives the full-size video of the gallery.

# %% RUN :: The video :: Writes lens_<city>_anim.mp4 for the city of the settings cell and plays it.
from gismaps.lens import anim as an  # noqa: E402  the animation code: sprites, rims, the ffmpeg writer, the storyboard

SCALE, FPS = (0.36, 15) if QUICK else (0.72, 30)  # poster shrink factor (0.72 -> 1440 x 2036 px) and frames per second
movie = an.animate_city(cfg, L, OUT / f"lens_{CITY}_anim.mp4", scale=SCALE, fps=FPS)  # cfg, L: the city loaded in step 3
print(f"{movie['path'].name}: {movie['frames']} frames, {movie['frames'] / FPS:.0f} s, {movie['path'].stat().st_size / 1e6:.1f} MB")
nb.video(movie["path"], width=420)

# %% RUN :: The GIF :: Converts the video to a small GIF (ffmpeg builds one colour palette for the whole clip).
gif = an.to_gif(movie["path"], width=360 if QUICK else 480, fps=10)
print(f"{gif.name}: {gif.stat().st_size / 1e6:.1f} MB")

# %% [markdown]
# ## 12 · The same poster for any place
# The three cities above are hand-tuned presets. `gismaps.lens.make_city` makes the same poster (and, if you ask,
# the video and the GIF) for **any place**: it looks the place up (or takes a latitude and longitude and finds the
# city's name), lays the three lenses out in the same triangle, downloads the OpenStreetMap layers from the public
# Overpass servers (cached, with retries), builds the legend from what is mapped there, and draws.
# A new place takes a few minutes, most of it the polite download.
#
# **The simplest way, also on a phone:** fill in the form of the next cell and run it. Give either a name in `PLACE`
# (a preset such as `Dhaka`, or any city: `Chittagong`, `Nairobi, Kenya`) or a point in `LAT_LON` (`23.72, 90.40`).
# With both empty the cell does nothing, so "Run all" stays quick. The same from a terminal:
#
# ```
# python -m gismaps.lens city --place "Chittagong" --outputs poster,video,gif
# python -m gismaps.lens city --lat 23.72 --lon 90.40 --outputs poster
# ```

# %% CHANGE FOR YOUR PLACE - RUN :: Any place :: Fill in PLACE or LAT_LON, tick the outputs you want, then run this cell.
PLACE = ""  #@param {type:"string"}
LAT_LON = ""  #@param {type:"string"}
MAKE_POSTER = True  #@param {type:"boolean"}
MAKE_VIDEO = False  #@param {type:"boolean"}
MAKE_GIF = False  #@param {type:"boolean"}
FULL_SIZE = False  #@param {type:"boolean"}

from gismaps.lens import make_city  # noqa: E402  one call: place -> poster / video / GIF

mine = None
if PLACE.strip() or LAT_LON.strip():
    wanted = tuple(k for k, on in (("poster", MAKE_POSTER), ("video", MAKE_VIDEO), ("gif", MAKE_GIF)) if on)
    stop_unless(len(wanted) > 0, "tick at least one of MAKE_POSTER, MAKE_VIDEO, MAKE_GIF")
    lat = lon = None
    if LAT_LON.strip():
        parts = LAT_LON.replace(";", ",").split(",")
        stop_unless(len(parts) == 2, "LAT_LON must look like 23.72, 90.40 (latitude first)")
        lat, lon = float(parts[0]), float(parts[1])
        stop_unless(-85 <= lat <= 85 and -180 <= lon <= 180, "LAT_LON must be latitude (-85 to 85), then longitude (-180 to 180)")
    mine = make_city(place=PLACE.strip() or None, lat=lat, lon=lon, outputs=wanted, fast=not FULL_SIZE, out_dir=OUT)
    print(f"{mine['name']}, {mine['country']} ({mine['lat']:.4f}, {mine['lon']:.4f})", *mine["notes"], sep="\n")
    if mine["poster"]:
        show(mine["poster"])
    if mine["video"]:
        nb.video(mine["video"], width=420)
else:
    print("PLACE and LAT_LON are empty: nothing to make. Fill one in and run this cell again.")

# %% [markdown]
# **Or choose with a list, a search box or a tap on a map.** The next two cells are optional. The first shows a map:
# tap a place and it prints `Selected: <city>, <country> (lat, lon)`; copy that latitude and longitude into `LAT_LON`
# above. The second shows a small panel (a list of about 28 world cities, a name box, a "lat, lon" box, tick boxes for
# poster / video / GIF and a Make button). Both work by touch.

# %% OPTIONAL :: Pick on a map :: Tap the map; the place is printed below it.
from gismaps.lens import picker  # noqa: E402

picker.pick_on_map()

# %% OPTIONAL :: The panel :: A city list, a name box, a lat-lon box, the outputs and a Make button.
picker.panel()

# %% [markdown]
# ## Credits
# * Map data © OpenStreetMap contributors, [ODbL 1.0](https://opendatacommons.org/licenses/odbl/). Print that line on
#   anything you make from it.
# * Style after Fan Mo, #30DayMapChallenge 2026 Day 14 ("Borgesian map"). The code and data here are original.
# * Posters: map created by Shammunul Islam.

# %% [markdown]
# ## Verification
# A last look that everything this notebook should have made is there and has the right size.

# %% RUN :: Verification :: Checks every poster file (exists, 2000 x 2828 px, not empty) and the video.
from PIL import Image  # noqa: E402

ok_all = True
for key in CITIES:
    f = OUT / f"lens_{key}.png"
    good = f.exists() and Image.open(f).size == (2000, 2828) and f.stat().st_size > 1_000_000  # there, right size, not blank
    ok_all &= good
    print(f"{'PASS' if good else 'FAIL'}  {f.name:26s} {f.stat().st_size / 1e6:5.2f} MB" if f.exists() else f"FAIL  {f.name} is missing")
np.testing.assert_allclose(out, img, atol=1e-6)  # the from-scratch lens rule still equals the library's
print("PASS  from-scratch composite equals lc.composite")
good = movie["path"].exists() and movie["path"].stat().st_size > 100_000 and abs(movie["frames"] / FPS - an.CITY_T["end"]) < 0.1
ok_all &= good
print(f"{'PASS' if good else 'FAIL'}  {movie['path'].name:26s} the video exists and is 24 s long")
good = gif.exists() and gif.stat().st_size > 50_000
ok_all &= good
print(f"{'PASS' if good else 'FAIL'}  {gif.name:26s} the GIF exists")
stop_unless(ok_all, "a poster or the video is missing or has the wrong size; run the cells above again")
print("\nAll checks passed. Posters and video are in:", OUT)

# %% [markdown]
# ## Download (Google Colab)
# On Colab this zips the pictures and movies of the output folder and offers them as one download; anywhere else it
# only says where they are.

# %% RUN :: Download :: On Colab: one zip of the outputs. Elsewhere: the folder name.
nb.download(OUT)
