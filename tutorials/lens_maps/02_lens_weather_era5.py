# Lens-map tutorial 2 - a weather poster from ERA5.  Plain script; `tools/build_notebooks.py` turns it into the notebooks.
# Run:  python tutorials/lens_maps/02_lens_weather_era5.py        (writes into ./out/)

# %% [markdown]
# # Lens map 2 · One storm, many realities
#
# **What you will make:** a poster of **Storm Ciarán** at 00 UTC on 2 November 2023, as it approached the English
# Channel. Three lenses look at the *same hour* of the *same storm*:
#
# | Lens | Field | Drawn as |
# |---|---|---|
# | **TEMPERATURE** | air temperature about 1.5 km up (the 850-hPa pressure level), °C | colour fill, blue-white-red, white at 0 °C |
# | **MOISTURE** | specific humidity at 850 hPa: grams of water vapour per kilogram of air | colour fill + dashed lines at 4, 6, 8 g/kg |
# | **PRESSURE & WIND** | sea-level pressure and the wind 10 m above the surface | isobars every 4 hPa + arrows |
#
# A few words first. **hPa** (hectopascal) is the unit of air pressure on weather maps; average sea-level pressure is
# about 1013 hPa. An **isobar** is a line of equal pressure. A **low** is a pressure minimum, the centre of a storm;
# in the Northern Hemisphere the wind blows anticlockwise around it (seen from above). **850 hPa** is the height
# where the pressure is 850 hPa, roughly 1.5 km up: high enough to be free of the ground's daily heating, so it shows
# the warm and cold air masses cleanly. **UTC** is the world's reference clock (Greenwich time, no summer time).
#
# **The new idea in this tutorial is the no-blend rule.** City maps could mix overlapping lenses half and half.
# Data fields must not: a mixed colour matches no colour bar. Here the overlaps become true *overlays* instead.
#
# **Data:** ERA5 reanalysis (Copernicus Climate Change Service / ECMWF) from the public ARCO-ERA5 store on Google
# Cloud, ETOPO 2022 relief (NOAA NCEI), Natural Earth coastlines. No key, no account.
# **Style** after Fan Mo, #30DayMapChallenge 2026 Day 14; the code and data here are original.
#
# **Time:** about 2 minutes, plus about 3 minutes of download when the ERA5 fields come straight from the cloud.

# %% [markdown]
# ## Settings
# Everything you may want to change is in this one cell. The checks at the bottom **stop** the notebook with a plain
# message if a setting cannot work, before anything slow starts.

# %% CHANGE IF YOU LIKE - RUN :: Settings :: The poster hour and where the ERA5 fields come from. Run it after every change.
import os

HOUR = "2023-11-02T00:00"  # the analysis hour of the poster (UTC), between 2023-11-01T12:00 and 2023-11-02T12:00
QUICK = os.environ.get("LENS_QUICK", "1") == "1"  # the video. True (default, right for Colab): a small preview, every third hour, 720 x 1018 px. False = FULL: all 25 hours, 1440 x 2036 px (about 10 minutes)
SOURCE = os.environ.get("LENS_ERA5", "reduced")  # "reduced" (the shipped 0.5° copy: fast, the default) or "arco" (the full 0.25° fields from the cloud store, about 230 MB of transfer)


def stop_unless(ok: bool, message: str) -> None:
    """STOP check: halt here, with a plain message, when a setting cannot work."""
    if not ok:
        raise ValueError("STOP - " + message)


for name, value in (("HOUR", HOUR),):
    stop_unless(len(value) == 16 and value.endswith(":00"), f"{name} must look like 2023-11-02T00:00 (a whole hour)")
    stop_unless("2023-11-01T12:00" <= value <= "2023-11-02T12:00", f"{name} = {value} is outside the 25 hours this tutorial covers")
stop_unless(SOURCE in ("arco", "reduced"), f"SOURCE = {SOURCE!r} must be arco or reduced")
print(f"settings OK | poster hour: {HOUR} UTC | ERA5 source: {SOURCE}")

# %% [markdown]
# ## 0 · Setup
# Same as in tutorial 1: the poster code is this repository's `gismaps` package (`gismaps/lens/`), and the small data
# files come from `data/lens_maps/` or are downloaded once from GitHub.

# %% RUN :: Setup :: Loads the poster modules and helpers.
import io
import os
import sys
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # keep the page clean: library deprecation notes are not part of the recipe

from gismaps import data as gdata, nb  # this repository's package: finds the data files; notebook helpers

OUT = nb.out_dir(os.environ.get("LENS_OUT"))  # posters are written here: /content/out on Google Colab, ./out elsewhere
CACHE = Path("lens_cache")  # downloads are kept here, so a second run needs no network


def data_file(name: str) -> Path:
    """One of the small pre-extracted data files: from the repository's data/lens_maps/ folder, or downloaded once
    from the repository on GitHub when it is not on disk (Google Colab)."""
    return gdata.path("lens_maps", name)


show = nb.picture  # shows a picture (file or array) inline, as wide as the page allows; a plain script only names it


def u8(a):
    """Float picture (0..1) -> 8-bit picture."""
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from gismaps.lens import common as lc  # noqa: E402  the lens rule (composite), pixel canvas, label blocks
from gismaps.lens import weather as lw  # noqa: E402  ERA5 on a Lambert pixel grid, contours, arrows, low tracker, poster

print("fonts:", lc.SERIF, "+", lc.SANS, "| poster hour:", HOUR, "UTC")

# %% [markdown]
# ## 1 · The data: ERA5 from a public cloud store
# **ERA5** is a *reanalysis*: a weather model run over the past, corrected every hour with the observations of that
# time, giving a complete, consistent picture of the atmosphere on a 0.25° grid (about 28 km), every hour since 1940.
#
# Google hosts a public copy, **ARCO-ERA5**, as one huge [zarr](https://zarr.dev) store. Zarr cuts each variable into
# *chunks* and a reader downloads only the chunks it needs. Here one chunk = one hour of one variable for the whole
# globe. That sets the cost:
#
# | What | Chunk size | For this poster |
# |---|---|---|
# | a surface field (pressure, 10-m wind) | about 3 MB | pressure for 7 hours + wind for 1 hour ≈ 27 MB |
# | a pressure-level field (temperature, humidity) | **80-120 MB** (all 37 levels are in one chunk) | 2 fields × 1 hour ≈ 200 MB |
#
# So: ask for upper-air fields at **few hours only**. (The video of step 9 needs all 25 hours of them, about
# 5 GB of transfer; that is why a small reduced copy ships with these tutorials.)
#
# Opening the store costs almost nothing: xarray reads only the description.

# %% OPTIONAL :: Look at the cloud store :: Opens ARCO-ERA5 lazily and prints what it holds. Nothing large is downloaded.
try:
    era5 = xr.open_zarr(lw.ARCO_URL, consolidated=True, chunks=None, zarr_format=2, storage_options={"token": "anon"})  # anonymous
    print(f"ARCO-ERA5: {len(era5.data_vars)} variables, {era5.sizes['time']:,} hours "
          f"(data valid {era5.attrs.get('valid_time_start', '?')} to {era5.attrs.get('valid_time_stop', '?')}), grid {era5.sizes['latitude']} x {era5.sizes['longitude']}")
    print("temperature:", dict(era5.temperature.sizes))  # lazy: nothing downloaded yet
except Exception as e:  # noqa: BLE001
    print("The cloud store is not reachable from here (", str(e)[:80], ") - the reduced copy will be used.")

# %% [markdown]
# **Two ways to get the fields**, both returning the same thing (`sfc` surface dataset, `pl` 850-hPa dataset):
#
# * `"reduced"` (the default) - the 1.2 MB copy that ships with the tutorial: every second grid point (0.5°), all
#   25 hours. Fast, and right for a first run or for Colab. Contours come out a little smoother than at full
#   resolution, and the low may sit one grid point away.
# * `"arco"` - straight from the cloud store with `lw.open_arco` (about 230 MB of transfer, about 4 MB kept in
#   `lens_cache/`; the second run needs no network; needs `pip install zarr gcsfs`). This is the full 0.25° data, the
#   same the gallery poster was made from. If the download fails, the notebook falls back to the reduced copy.

# %% RUN :: Get the fields :: ERA5 from the source chosen in the settings cell, plus coastlines and relief.
if SOURCE == "arco":
    try:
        data = lw.open_arco(HOUR, cache=CACHE)  # lazy selection: time, level, a latitude-longitude box
    except Exception as e:  # noqa: BLE001
        print("ARCO-ERA5 did not answer (", str(e)[:80], ") - using the reduced copy.")
        SOURCE = "reduced"
if SOURCE == "reduced":
    data = lw.load_reduced(data_file("era5_ciaran_lens_0p5deg.nc"))
sfc, pl, _ = data
step = float(sfc.latitude[1] - sfc.latitude[0])  # grid spacing in degrees
print(f"source: {SOURCE} | grid {step}° | surface hours {str(sfc.time.values[0])[:13]} .. {str(sfc.time.values[-1])[:13]} "
      f"| lat {float(sfc.latitude.min())}..{float(sfc.latitude.max())}, lon {float(sfc.longitude.min())}..{float(sfc.longitude.max())}")

# the two small background data sets: coastlines and relief
if not Path(lw.LAND).exists():
    lw.fetch_land(CACHE)  # Natural Earth 1:50m land, public domain (0.5 MB)
if not Path(lw.ETOPO).exists():
    try:
        lw.fetch_etopo(CACHE)  # every 10th point of ETOPO 2022 through OPeNDAP: only the asked points travel
        print("relief: ETOPO 2022 from NOAA NCEI (OPeNDAP)")
    except Exception as e:  # noqa: BLE001
        lw.ETOPO = data_file("etopo2022_10min_natl.nc")  # the same grid, shipped with the tutorial
        print("relief: the shipped ETOPO copy (NOAA's server did not answer:", str(e)[:50], ")")

# %% [markdown]
# ## 2 · Finding the storm: a seeded low tracker
# The poster is centred on the low. Finding it sounds easy ("the lowest pressure on the map") and is a classic trap:
# a large map usually holds *several* lows, and the deepest one is not always the one you mean.
#
# The fix is to **follow one low**: start in a small box where and when you know the storm is (the *seed*: the
# western English Channel at 06 UTC on 2 November), then step hour by hour, each time taking the minimum only within
# 2.5° of the last position.

# %% RUN :: Track the low :: One tracking step from scratch, then the whole track against the whole-map minimum.
msl = sfc.mean_sea_level_pressure  # pascals; dimensions (time, latitude, longitude)
ts = sfc.time.values
seed_i = int(np.where(ts == np.datetime64(lw.SEED_TIME))[0][0])  # index of the seed hour
track = lw.track_low(msl, seed_i)  # [(lon, lat, hPa)] for every hour of `sfc`
i = int(np.where(ts == np.datetime64(HOUR))[0][0])  # index of the poster hour
lon0, lat0, p0 = track[i]

# --- from scratch: one tracking step, from the hour after the poster hour back to the poster hour ---
lon_next, lat_next, _ = track[i + 1]  # where the low is one hour later (already known)
near = msl.isel(time=i).sel(longitude=slice(lon_next - 3.75, lon_next + 3.75),  # 2.5° x 1.5 east-west (degrees of longitude are shorter)
                            latitude=slice(lat_next - 2.5, lat_next + 2.5))  # 2.5° north-south
k = np.unravel_index(int(near.values.argmin()), near.shape)  # row and column of the smallest value in that window
scratch = (float(near.longitude[k[1]]), float(near.latitude[k[0]]), float(near.values[k]) / 100.0)  # Pa -> hPa
assert scratch == (lon0, lat0, p0)  # the same grid point: this is exactly what track_low does at each step

# --- the trap: the minimum of the whole map, hour by hour ---
print("hour (UTC)     tracked low                whole-map minimum")
for j in range(0, len(ts), max(1, len(ts) // 8)):
    a = msl.isel(time=j)
    kk = np.unravel_index(int(a.values.argmin()), a.shape)
    naive = (float(a.longitude[kk[1]]), float(a.latitude[kk[0]]), float(a.values[kk]) / 100.0)
    flag = "" if abs(naive[0] - track[j][0]) < 3 and abs(naive[1] - track[j][1]) < 3 else "   <- a different low"
    print(f"{str(ts[j])[5:13]}   {track[j][0]:7.2f}°E {track[j][1]:5.2f}°N {track[j][2]:6.1f} hPa   "
          f"{naive[0]:7.2f}°E {naive[1]:5.2f}°N {naive[2]:6.1f} hPa{flag}")
print(f"\nPoster hour: the low is at {abs(lon0):.2f}°{'W' if lon0 < 0 else 'E'}, {lat0:.2f}°N, {p0:.0f} hPa.")

# %% [markdown]
# ## 3 · The map: a Lambert conformal pixel grid
# This map is 2,900 km wide. At that size the choice of projection matters. We use the **Lambert conformal conic**
# projection, the usual one for mid-latitude weather maps: it is *conformal* (angles and shapes are kept: a circular
# storm stays circular) and distances are true along two chosen "standard" parallels (45°N and 58°N here) and close
# to true between them.
#
# `lw.Grid` ties the projection to output pixels: pixel → metres on the projection plane → longitude and latitude.
# Every layer is then computed **at the pixel centres of this one grid**; matplotlib never chooses an extent.
#
# One consequence of a conic map: **north is not "up" everywhere.** The meridians fan out, so north tilts to the
# left on the left side of the sheet and to the right on the right side. That is why this poster has a faint
# graticule and a note instead of a north arrow.

# %% RUN :: The Lambert pixel grid :: Longitude and latitude to pixels, and how far north turns across the sheet.
g = lw.poster_grid(lon0, lat0)  # 2000 x 2828 px, 1.45 km per pixel, the low a little below and right of the middle
cx, cy = (float(v) for v in g.ll2px(lon0, lat0))  # where the low falls on the sheet
print(f"{g.mpp / 1000:.2f} km per pixel -> the sheet shows {g.w * g.mpp / 1e6:.2f} x {g.h * g.mpp / 1e6:.2f} thousand km; the low is at pixel ({cx:.0f}, {cy:.0f})")

# --- from scratch: longitude/latitude -> pixel, one step per line ---
x_m, y_m = g.proj(lon0 + 10.0, lat0)  # a point 10° east of the low, in metres on the projection plane (pyproj)
px = g.cx + x_m / g.mpp  # metres -> pixels, counted from the pixel of the projection centre
py = g.cy - y_m / g.mpp  # pixel rows count downward, so subtract
np.testing.assert_allclose((px, py), g.ll2px(lon0 + 10.0, lat0), rtol=1e-12)  # same two lines as Grid.ll2px

# --- how far does "north" turn across the sheet? ---
for name, xpix in (("left edge", 0.0), ("low", cx), ("right edge", float(g.w))):
    lon_a, lat_a = g.px2ll(xpix, cy)  # the place at this pixel
    x1, y1 = g.ll2px(lon_a, lat_a + 0.1)  # a point 0.1° further north
    tilt = np.degrees(np.arctan2(x1 - xpix, cy - y1))  # angle of "north" from straight up; + = leaning right
    print(f"north at the {name:10s} ({float(lon_a):6.1f}°E): {tilt:+5.1f}° from straight up")

# %% [markdown]
# **Putting a field on the grid.** ERA5 lives on a latitude-longitude grid; the poster needs a value at every pixel
# centre. `Grid.sample` does a **bilinear interpolation**: find the four grid points around the pixel, weight them by
# closeness. From scratch for one pixel, then the whole sheet at once:

# %% RUN :: Put the fields on the grid :: Bilinear interpolation for one pixel by hand, then the whole sheet.
T_da = pl.temperature.sel(time=np.datetime64(HOUR))  # kelvin, (latitude, longitude)
lon_p, lat_p = (float(v) for v in g.px2ll(cx - 300.5, cy - 200.5))  # the place under one pixel centre, up-left of the low
lats, lons = T_da.latitude.values, T_da.longitude.values  # both increasing
ia = int(np.searchsorted(lats, lat_p) - 1)  # index of the grid latitude just south of the place
io_ = int(np.searchsorted(lons, lon_p) - 1)  # index of the grid longitude just west of it
fy = (lat_p - lats[ia]) / (lats[ia + 1] - lats[ia])  # 0 at the southern row, 1 at the northern row
fx = (lon_p - lons[io_]) / (lons[io_ + 1] - lons[io_])  # 0 at the western column, 1 at the eastern column
v = T_da.values
south = v[ia, io_] * (1 - fx) + v[ia, io_ + 1] * fx  # blend west-east along the southern row
north = v[ia + 1, io_] * (1 - fx) + v[ia + 1, io_ + 1] * fx  # the same along the northern row
T_scratch = south * (1 - fy) + north * fy  # then blend south-north

T = g.sample(T_da, strict=False) - 273.15  # the whole sheet at once, kelvin -> °C (NaN outside the data box)
np.testing.assert_allclose(T_scratch - 273.15, T[int(cy - 201), int(cx - 301)], atol=2e-3)  # float32 rounding only
q = g.sample(pl.specific_humidity.sel(time=np.datetime64(HOUR)), strict=False) * 1000.0  # kg/kg -> g/kg
p = g.sample(msl.sel(time=np.datetime64(HOUR)), strict=False) / 100.0  # Pa -> hPa
print(f"T850 at that pixel: {T_scratch - 273.15:.2f} °C | on the sheet: T {np.nanmin(T):.1f}..{np.nanmax(T):.1f} °C, "
      f"q {np.nanmin(q):.1f}..{np.nanmax(q):.1f} g/kg, p {np.nanmin(p):.0f}..{np.nanmax(p):.0f} hPa")

# %% [markdown]
# ## 4 · Colour: two fills, each with a colour bar you can trust
# * **Temperature** uses `cmocean.balance`, a *diverging* map: two hues that meet in white. A diverging map needs a
#   meaningful middle, and here it has one: **0 °C, freezing**. The range is symmetric, −12 to +12 °C, so white is
#   exactly 0 °C and equal steps either side look equally strong.
# * **Moisture** uses `cmocean.deep` reversed, a *sequential* map (dark = dry, light green = moist), from 0 to 10 g/kg.
#   Humidity has no natural middle, so one hue ramp is right.
# * **No rainbow / jet.** Rainbow maps have bright bands that invent edges in smooth data, and they fail for
#   colour-blind readers. The cmocean maps are perceptually even.
#
# The colours are applied **with numpy, pixel by pixel** (`fill_layer`): value → 0..1 → colour. No resampling, so a
# colour on the poster is exactly the colour-bar colour of that pixel's value.

# %% RUN :: Colour the fills :: One temperature to its colour by hand, then both fill layers.
rings = lw.land_polys(g, pad_deg=10)  # Natural Earth coast as pixel polygons
# --- from scratch: the colour of one temperature ---
vmin, vmax = lw.T_RANGE  # -12, +12 °C
T_px = float(T[int(cy - 201), int(cx - 301)])  # the pixel sampled above
frac = np.clip((T_px - vmin) / (vmax - vmin), 0, 1)  # where the value sits in the range, 0..1
rgb_scratch = np.array(lw.T_CMAP(frac)[:3])  # look that fraction up in the colour map
t_fill = lw.fill_layer(g, T, lw.T_CMAP, *lw.T_RANGE, rings, "#0b1020", 0.6)  # the whole layer (+ a thin coast line)
np.testing.assert_allclose(rgb_scratch, t_fill[int(cy - 201), int(cx - 301)], atol=1e-6)
print(f"{T_px:.2f} °C -> fraction {frac:.3f} -> RGB {np.round(rgb_scratch, 3)}; 0 °C -> {np.round(lw.T_CMAP(0.5)[:3], 2)} (white)")
q_fill = lw.fill_layer(g, q, lw.Q_CMAP, *lw.Q_RANGE, rings, "#9fb4c8", 0.55)
show(u8(np.concatenate([t_fill[::4, ::4], q_fill[::4, ::4]], axis=1)), 900)

# %% [markdown]
# ## 5 · Line work: isobars, moisture contours, wind arrows
# Lines are what make overlaps useful (next section), so every lens that can sit on top of another needs a
# *line-work version* of its field, drawn on a transparent layer:
#
# * **Isobars** every 4 hPa (the standard spacing on surface charts), every 20 hPa bolder, labelled every 8 hPa.
#   Tightly packed isobars mean strong wind.
# * **Moisture contours** at 4, 6, 8 g/kg, dashed, on a lightly smoothed field so the lines are calm.
# * **Wind arrows**, one every 46 px (67 km), 2.1 px per m/s. ERA5 gives the wind as *u* (toward east) and *v*
#   (toward north). On this map east is not "right" away from the central meridian, so each arrow is turned by the
#   map's own east and north directions at its place. Below, from scratch for one arrow.

# %% RUN :: Line work :: One wind arrow turned to the map by hand, then isobars, moisture lines and arrows.
u_da = sfc["10m_u_component_of_wind"].sel(time=np.datetime64(HOUR))  # m/s toward east
v_da = sfc["10m_v_component_of_wind"].sel(time=np.datetime64(HOUR))  # m/s toward north
gx, gy = cx - 600.0, cy + 200.0  # one arrow position, 870 km west of the low (pixels)
wlon, wlat = (float(a) for a in g.px2ll(gx, gy))  # the place there
u1 = float(g.sample(u_da, np.array([wlon]), np.array([wlat]), strict=False)[0])  # wind toward east there
v1 = float(g.sample(v_da, np.array([wlon]), np.array([wlat]), strict=False)[0])  # wind toward north there
east = np.array(g.ll2px(wlon + 0.01, wlat)) - (gx, gy)  # the pixel step you make going a little east
east /= np.linalg.norm(east)  # ... as a unit vector (x right, y down)
north = np.array(g.ll2px(wlon, wlat + 0.01)) - (gx, gy)  # the same going a little north
north /= np.linalg.norm(north)
arrow = u1 * east + v1 * north  # the wind as a pixel vector: u along local east, v along local north
print(f"at {wlon:.1f}°E {wlat:.1f}°N: u = {u1:.1f}, v = {v1:.1f} m/s ({np.hypot(u1, v1):.1f} m/s). "
      f"Local east on the sheet = ({east[0]:.3f}, {east[1]:.3f}), so the arrow is ({arrow[0]:.1f}, {arrow[1]:.1f}) px-units, "
      f"not ({u1:.1f}, {-v1:.1f}).")
assert abs(east @ north) < 1e-3  # local east and north are at right angles on the sheet: the map is conformal
assert abs(np.hypot(*arrow) - np.hypot(u1, v1)) < 0.01  # so turning the arrow does not change its length

layers = lw.hour_layers(g, rings, sfc, pl, HOUR)  # t, q fills; ql moisture lines; pg chart ground; po isobars + arrows
print({k: v.shape for k, v in layers.items()})
ground = layers["pg"]  # the plain dark-blue chart ground of the pressure lens
a = layers["po"][..., 3:]  # the transparency of the line-work layer
chart = ground * (1 - a) + layers["po"][..., :3] * a  # lines over ground: what the pressure lens shows on its own
win = (slice(int(cy - 520), int(cy + 520)), slice(int(cx - 700), int(cx + 700)))  # a window round the low
show(u8(chart[win][::2, ::2]), 800)

# %% [markdown]
# ## 6 · The no-blend rule
# Now the three lenses, and the question from tutorial 1: what happens where they overlap?
#
# **If we blend (overlap = 0.5):** the temperature colour and the moisture colour are averaged. The result is a
# colour that is on *neither* colour bar. The cell below takes one pixel in the overlap and asks: "if a reader looks
# this blended colour up on the temperature bar, what temperature do they read?"
#
# **The rule for data fields (overlap = 0):** the lower lens **keeps its exact colours**; an upper lens contributes
# only its **line work** there. The overlap becomes a true overlay. Where all three lenses meet you see temperature
# as colour, moisture as dashed lines, pressure as isobars and the wind as arrows: a complete weather chart.
#
# That also fixes the **order**: the richest fill (temperature) lies lowest because it is the colour you see in every
# overlap; the pure line-work lens (pressure and wind) lies on top.

# %% RUN :: The no-blend rule :: Blended against exact overlaps, with the temperature a reader would read off each.
cen = lw.lens_centres(cx, cy)  # lens centres relative to the low
base = lw.poster_base(g, rings)  # the muted map outside the lenses: black sea, dark land, faint relief and graticule
exact = lw.make_lenses(layers, cen)  # overlap = 0 on every lens
mixed = lw.make_lenses(layers, cen)
for Ln in mixed:
    Ln.overlap = 0.5  # the city-poster look, WRONG for data
img_exact, img_mixed = lc.composite(base, exact), lc.composite(base, mixed)

# one pixel inside both the temperature lens and the moisture lens, away from any line
qx, qy = int(cen["t"][0] + 60), int(cen["q"][1] - 380)
T_true = float(T[qy, qx])
bar = np.array([lw.T_CMAP(f)[:3] for f in np.linspace(0, 1, 481)])  # the temperature colour bar in 0.05 °C steps
bar_T = np.linspace(*lw.T_RANGE, 481)


def read_off(rgb):  # the temperature whose bar colour is closest to a given colour
    return float(bar_T[np.argmin(((bar - rgb) ** 2).sum(1))])


print(f"true temperature at that pixel:        {T_true:+.1f} °C")
print(f"read off the exact (overlap 0) colour: {read_off(img_exact[qy, qx]):+.1f} °C")
print(f"read off the blended (0.5) colour:     {read_off(img_mixed[qy, qx]):+.1f} °C   <- the blend misleads the reader")
assert abs(read_off(img_exact[qy, qx]) - T_true) < 0.1  # exact colours still match the bar

mid = (slice(int(cy - 420), int(cy + 420)), slice(int(cx - 560), int(cx + 560)))  # where the three lenses meet
show(u8(np.concatenate([img_mixed[mid], np.zeros((840, 24, 3), np.float32), img_exact[mid]], axis=1)[::2, ::2]), 1000)
print("left: blended overlaps (colours match no colour bar) | right: the no-blend rule (fill below, lines above)")

# %% [markdown]
# ## 7 · The poster
# `lw.build_poster` runs the chain you have just walked through: track the low → grid → base map → the hour's layers
# → three lenses with `overlap = 0` → rims → the **L** with the central pressure → furniture (title, three label
# blocks each with its colour bar or key and a leader, scale bar, projection note, credits).

# %% RUN :: The poster :: Writes lens_weather.png (2000 x 2828 px) into the output folder.
png = lw.build_poster(HOUR, data=data, out=OUT / "lens_weather.png")
show(png)

# %% [markdown]
# ## 8 · Why it looks the way it does
#
# | Choice | Reason |
# |---|---|
# | Lenses ride on the tracked low | all three look at the storm; the offsets put the warm sector in the temperature lens, the moist inflow in the moisture lens and the strongest pressure gradient in the pressure lens |
# | `overlap = 0` on every lens | the no-blend rule: a colour must match its colour bar everywhere |
# | Order: temperature, moisture, pressure & wind | lowest = the fill you see in overlaps; top = pure line work |
# | Diverging map centred on 0 °C; sequential map for humidity | the colour map follows the meaning of the numbers |
# | A colour bar or key inside every label block | each lens explains itself; no separate legend to hunt for |
# | Isobar labels on the poster only | at poster size 24-px labels are readable; in a video they would be too small, so the label block says "every 4 hPa" |
# | Scale bar with a note, graticule, no north arrow | on a conic map scale is true only near the standard parallels and north turns across the sheet |
# | "850 hPa" always explained as "1.5 km up" | the reader may not know pressure levels |
# | Base map nearly black with faint relief | it only has to say "this is the North Atlantic and Europe" |
#
# Two limits of this picture: 850-hPa values are extrapolated below ground over high mountains, and ERA5's 28-km
# grid smooths the sharpest gusts.

# %% [markdown]
# ## 9 · The video
# The three lenses open on the storm at 12 UTC on 1 November 2023; then **one clock** steps all three fields hour by
# hour to 12 UTC on 2 November, while the tracked low (the **L**, its central pressure and its track) crosses the
# lenses. Each hour is held and only the last third of a second cross-fades to the next: a long cross-fade of contour
# lines would show both hours' lines at once.
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
# The video needs all 25 hours, so it always uses the reduced 0.5° copy (from the cloud store that would be about 5 GB
# of transfer). The video of the gallery was made from the full 0.25° fields with `QUICK = False`.

# %% RUN :: The video and the GIF :: Writes lens_weather_anim.mp4 and .gif, plays the video and prints the track of the low.
from gismaps.lens import anim as an  # noqa: E402  the animation code: sprites, rims, the ffmpeg writer, the storyboard

SCALE, FPS = (0.36, 15) if QUICK else (0.72, 30)  # poster shrink factor (0.72 -> 1440 x 2036 px) and frames per second
movie_data = data if SOURCE == "reduced" else lw.load_reduced(data_file("era5_ciaran_lens_0p5deg.nc"))  # all 25 hours
hours = movie_data[1].time.values  # the 25 analysis hours
if QUICK:
    hours = hours[::3]  # the preview shows every third hour (each hour's fields take about 15 s to draw)
movie = an.animate_weather(movie_data, OUT / "lens_weather_anim.mp4", scale=SCALE, fps=FPS, hours=hours,
                           T={"step": 1.6} if QUICK else None)  # fewer hours -> hold each one longer
print("the tracked low:  hour (UTC), longitude, latitude, central pressure (hPa)")
for when, lon_, lat_, hpa in movie["track"]:
    print(f"  {when[5:]}  {lon_:7.2f}  {lat_:6.2f}  {hpa:6.1f}")
nb.video(movie["path"], width=420)
gif = an.to_gif(movie["path"], width=360 if QUICK else 480, fps=10)  # a small GIF of the same clip
print(f"{gif.name}: {gif.stat().st_size / 1e6:.1f} MB")

# %% [markdown]
# ## Credits
# * ERA5: Hersbach et al. (2020), *Q. J. R. Meteorol. Soc.* 146, 1999-2049. Contains modified Copernicus Climate
#   Change Service information; neither the European Commission nor ECMWF is responsible for any use made of it.
#   ARCO-ERA5: Carver and Merose (2023), Google Research.
# * Relief: ETOPO 2022, NOAA National Centers for Environmental Information (public domain). Coast: Natural Earth
#   (public domain).
# * Style after Fan Mo, #30DayMapChallenge 2026 Day 14. Poster: map created by Shammunul Islam.

# %% [markdown]
# ## Verification
# A last look that the poster is there, has the right size, and that the storm is where it should be.

# %% RUN :: Verification :: Checks the poster file, the tracked low, the no-blend rule and the video.
from PIL import Image  # noqa: E402

f = OUT / "lens_weather.png"
checks = [
    ("poster file exists, 2000 x 2828 px", f.exists() and Image.open(f).size == (2000, 2828)),
    ("the tracked low is a deep low (below 980 hPa)", p0 < 980.0),
    ("the low is between Ireland and the North Sea (20°W-5°E, 46-54°N)", -20 <= lon0 <= 5 and 46 <= lat0 <= 54),
    ("the low sits at the centre pixel of the grid", abs(cx - 1060) < 1 and abs(cy - 1554) < 1),
    ("exact overlap colours match the temperature colour bar", abs(read_off(img_exact[qy, qx]) - T_true) < 0.1),
    ("every lens uses overlap = 0", all(Ln.overlap == 0.0 for Ln in exact)),
    ("the video exists and shows every chosen hour", movie["path"].exists() and movie["path"].stat().st_size > 100_000 and len(movie["track"]) == len(hours)),
    ("the low deepens along its track in the video", movie["track"][-1][3] < movie["track"][0][3]),
    ("the GIF exists", gif.exists() and gif.stat().st_size > 50_000),
]
for text, good in checks:
    print("PASS " if good else "FAIL ", text)
stop_unless(all(good for _, good in checks), "a check failed; read the FAIL line above")
print(f"\nAll checks passed. Low at {HOUR} UTC: {abs(lon0):.2f}°{'W' if lon0 < 0 else 'E'}, {lat0:.2f}°N, {p0:.1f} hPa (source: {SOURCE}).")

# %% [markdown]
# ## Download (Google Colab)
# On Colab this zips the pictures and movies of the output folder and offers them as one download; anywhere else it
# only says where they are.

# %% RUN :: Download :: On Colab: one zip of the outputs. Elsewhere: the folder name.
nb.download(OUT)
