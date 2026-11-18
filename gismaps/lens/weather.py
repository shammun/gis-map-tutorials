"""Lens map, weather version: "One storm, many realities" - Storm Ciaran, 1-2 November 2023.

Three lenses over ONE moment of ONE storm, each a different field of the same ERA5 analysis hour:
  TEMPERATURE      air temperature at 850 hPa [deg C], cmocean balance centred on 0 deg C (freezing)
  MOISTURE         specific humidity at 850 hPa [g/kg], cmocean deep (reversed: dry = dark)
  PRESSURE & WIND  sea-level pressure isobars every 4 hPa + 10-m wind

Outputs
  layers  : <work folder>/video_layers/{ctx,p}.png, {t,q,ql,pl}_NN.png (25 hours), manifest.json  (map 1140 x 625 px;
            one picture per field and hour, for a video editor or a web animation that moves and masks them)
  poster  : <out>/lens_weather.png (2000 x 2828 px, 00 UTC 2 Nov 2023)

Inputs: ERA5 hourly (ARCO-ERA5, Copernicus C3S/ECMWF),
ETOPO 2022 10-arc-min (NOAA NCEI), Natural Earth 50 m land (public domain).
Projection: Lambert conformal conic (standard parallels 45 and 58 N), north up on the central meridian.
Style after Fan Mo, #30DayMapChallenge 2026 Day 14; the code and data are original.

Run:  python -m gismaps.lens.weather [movie] [poster]

The poster is built in steps that the tutorials (``tutorials/lens_maps/``) and the animation (``gismaps.lens.anim``)
call one by one:  ``poster_grid`` -> ``context_layer`` + ``hour_layers`` (images on that pixel grid) ->
``make_lenses`` + ``lc.composite`` -> ``low_marker`` + ``furniture``.  ``build_poster`` runs them all.
``open_arco`` reads the ERA5 selection straight from the public ARCO-ERA5 store (anonymous); ``load_reduced`` reads
the small 0.5-degree copy that ships with the tutorials; ``load`` reads full-resolution netCDF subsets you saved
yourself under the work folder (``GISMAPS_WORK``, default ``./gismaps_work``). ``ETOPO`` and ``LAND`` are module variables: point them at your own copies before drawing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cmocean
import geopandas as gpd
import matplotlib.patheffects as pe
import numpy as np
import pyproj
import shapely
import xarray as xr
from matplotlib.collections import LineCollection, PolyCollection
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from . import common as lc  # noqa: E402

ROOT = Path(os.environ.get("GISMAPS_WORK", "gismaps_work"))
RAW = ROOT / "raw"
OUT = ROOT / "out"
PUB = ROOT / "video_layers"
SFC = RAW / "ERA5-ARCO/era5_ciaran_2023_u10-v10-msl-t2m_2023-11-01T00_2023-11-02T23_1h.nc"
PL = RAW / "ERA5-ARCO/era5_ciaran_850_t-q-u-v_850_2023-11-01T12_2023-11-02T12_1h.nc"
ETOPO = RAW / "ETOPO2022-60s/etopo2022_10min_global.nc"
LAND = RAW / "NE-50m-land/ne_50m_land.zip"

T_RANGE = (-12.0, 12.0)  # deg C, centred on 0 (freezing)
Q_RANGE = (0.0, 10.0)  # g/kg
T_CMAP = cmocean.cm.balance
Q_CMAP = cmocean.cm.deep_r
RIM = {"t": "#e8833a", "q": "#2aa198", "p": "#f4ecd2"}  # orange, teal, warm white


# ----------------------------------------------------------------------------------------------------------- data
def _fix(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude").sortby("latitude")
    return ds


def load(sfc_path=None, pl_path=None):
    """Two full-resolution ERA5 subsets saved as netCDF (default paths ``SFC`` and ``PL``) -> (surface dataset,
    850-hPa dataset, the 25 hourly times they share). Latitude south -> north, longitude -180 .. 180."""
    sfc = _fix(xr.open_dataset(sfc_path or SFC))
    pl = _fix(xr.open_dataset(pl_path or PL)).squeeze("level")
    times = pl.time.values
    sfc = sfc.sel(time=times)
    return sfc, pl, times


ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
SEED_TIME = "2023-11-02T06:00"  # the hour the low tracker starts from (Ciaran over the western Channel)
_SFC_VARS = ["mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind"]
_PL_VARS = ["temperature", "specific_humidity"]


def load_reduced(path):
    """The small copy that ships with the tutorials (every second grid point = 0.5 degrees, 16-bit packed; made by
    ``tutorials/lens_maps/make_data.py``) -> the same (sfc, pl, times) as :func:`load`."""
    ds = xr.open_dataset(path).load()
    sfc = _fix(ds[_SFC_VARS])
    pl = _fix(ds[_PL_VARS].rename(latitude_pl="latitude", longitude_pl="longitude"))
    return sfc, pl, ds.time.values


def open_arco(hour: str = "2023-11-02T00:00", lat=(30.0, 70.0), lon=(-50.0, 15.0), level: int = 850, cache=None, log=print):
    """ERA5 for the poster of one hour, straight from the public ARCO-ERA5 zarr store on Google Cloud (anonymous: no
    key, no account) -> the same (sfc, pl, times) as :func:`load`, with only what one poster needs:

      * sea-level pressure for every hour from ``hour`` to the tracker's seed hour (the low is followed hour by hour);
      * 10-m wind, 850-hPa temperature and 850-hPa specific humidity at ``hour`` only.

    What it costs: the store keeps one whole global field per chunk. A surface chunk is about 3 MB, a pressure-level
    chunk holds all 37 levels (80-120 MB), so one poster hour transfers roughly 230 MB and keeps about 4 MB.
    ``cache`` (a folder) keeps those 4 MB as netCDF, so the second run needs no network."""
    t0, t1 = sorted([np.datetime64(hour, "ns"), np.datetime64(SEED_TIME, "ns")])
    hours = np.arange(t0, t1 + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    tag = str(np.datetime64(hour, "h")).replace(":", "")
    f_sfc = Path(cache) / f"era5_arco_ciaran_sfc_{tag}.nc" if cache else None
    f_pl = Path(cache) / f"era5_arco_ciaran_{level}_{tag}.nc" if cache else None
    if f_sfc and f_sfc.exists() and f_pl.exists():
        sfc, pl = xr.open_dataset(f_sfc).load(), xr.open_dataset(f_pl).load()
    else:
        # zarr_format=2 + consolidated metadata: one small request describes the whole store; chunks=None: no dask
        ds = xr.open_zarr(ARCO_URL, consolidated=True, chunks=None, zarr_format=2, storage_options={"token": "anon"})
        lons = np.arange(lon[0], lon[1] + 0.125, 0.25) % 360.0  # the store counts longitude 0 .. 359.75 east
        box = dict(latitude=slice(lat[1], lat[0]), longitude=lons)  # the store runs north -> south
        one = [np.datetime64(hour, "ns")]
        log(f"ARCO-ERA5: sea-level pressure, {len(hours)} hours ...", flush=True)
        msl = ds[_SFC_VARS[:1]].sel(time=hours, **box).load()
        log("ARCO-ERA5: 10-m wind, 1 hour ...", flush=True)
        uv = ds[_SFC_VARS[1:]].sel(time=one, **box).load()
        log(f"ARCO-ERA5: temperature and humidity at {level} hPa, 1 hour (two chunks of about 100 MB) ...", flush=True)
        pl = ds[_PL_VARS].sel(time=one, level=level, **box).load()
        sfc = xr.merge([msl, uv.reindex(time=hours)], compat="no_conflicts")
        for d in (sfc, pl):
            d.attrs = {"source": ARCO_URL, "licence": "Contains modified Copernicus Climate Change Service information (ERA5)"}
        if cache:
            Path(cache).mkdir(parents=True, exist_ok=True)
            sfc.to_netcdf(f_sfc)
            pl.to_netcdf(f_pl)
    return _fix(sfc), _fix(pl.drop_vars("level", errors="ignore")), sfc.time.values


LAND_URLS = ("https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_land.zip",
             "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip")
ETOPO_URL = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"


def fetch_land(folder) -> Path:
    """Natural Earth 1:50m land polygons (public domain, 0.5 MB zip) from its public server -> ``folder``; sets ``LAND``."""
    global LAND
    import urllib.request

    f = Path(folder) / "ne_50m_land.zip"
    err = None
    for url in LAND_URLS:
        if f.exists():
            break
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "gismaps-tutorials/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                f.write_bytes(resp.read())
        except Exception as e:  # noqa: BLE001  (try the mirror)
            err = e
    if not f.exists():
        raise RuntimeError(f"Natural Earth land not reachable: {err}")
    LAND = f
    return f


def fetch_etopo(folder, lat=(15.0, 85.0), lon=(-85.0, 60.0), stride: int = 10) -> Path:
    """ETOPO 2022 relief (NOAA NCEI, public domain) for the map's background: every ``stride``-th point of the
    60-arc-second grid inside a box, read through OPeNDAP so that only the asked points travel (about 1.5 MB instead of
    the 0.5 GB file) -> ``folder``; sets ``ETOPO``. Raises if the server does not answer (then use the small copy that
    ships with the tutorials)."""
    global ETOPO
    f = Path(folder) / f"etopo2022_{stride}min_box.nc"
    if not f.exists():
        err = None
        for engine in ("pydap", "netcdf4"):  # pydap works on Windows; either works on Linux / Colab
            try:
                z = xr.open_dataset(ETOPO_URL, engine=engine).z.isel(lat=slice(0, None, stride), lon=slice(0, None, stride))
                z = z.sel(lat=slice(*lat), lon=slice(*lon)).load()
                f.parent.mkdir(parents=True, exist_ok=True)
                xr.Dataset({"z": z.astype("float32")}, attrs={"source": ETOPO_URL, "licence": "NOAA NCEI, public domain"}).to_netcdf(f)
                break
            except Exception as e:  # noqa: BLE001
                err = e
        if not f.exists():
            raise RuntimeError(f"ETOPO OPeNDAP server not reachable: {err}")
    ETOPO = f
    return f


class Grid:
    """Lambert conformal pixel grid: W x H px, centred on (clon, clat), half-height ``half_h_km``."""

    def __init__(self, clon: float, clat: float, half_h_km: float, w: int, h: int, x0: float = 0.0, y0: float = 0.0):
        self.proj = pyproj.Proj(proj="lcc", lat_1=45, lat_2=58, lat_0=clat, lon_0=clon, R=6371229.0)
        self.w, self.h = w, h
        self.mpp = 2 * half_h_km * 1000.0 / h  # metres per pixel (true on the standard parallels)
        self.cx, self.cy = w / 2 + x0, h / 2 + y0
        xx, yy = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        self.lon, self.lat = self.px2ll(xx, yy)

    def ll2px(self, lon, lat):
        x, y = self.proj(lon, lat)
        return self.cx + np.asarray(x) / self.mpp, self.cy - np.asarray(y) / self.mpp

    def px2ll(self, px, py):
        return self.proj((np.asarray(px) - self.cx) * self.mpp, (self.cy - np.asarray(py)) * self.mpp, inverse=True)

    def sample(self, da: xr.DataArray, lon=None, lat=None, strict: bool = True) -> np.ndarray:
        f = RegularGridInterpolator((da.latitude.values, da.longitude.values), da.values, bounds_error=strict, fill_value=np.nan)
        lon = self.lon if lon is None else lon
        lat = self.lat if lat is None else lat
        return f(np.stack([lat, lon], -1)).astype(np.float32)


def land_polys(g: Grid, pad_deg: float = 6.0):
    """Natural Earth land rings in pixel coordinates (list of (n, 2) arrays), clipped to the grid's lon/lat box."""
    lo0, lo1 = float(np.nanmin(g.lon)) - pad_deg, float(np.nanmax(g.lon)) + pad_deg
    la0, la1 = float(np.nanmin(g.lat)) - pad_deg, float(np.nanmax(g.lat)) + pad_deg
    gdf = gpd.read_file(f"zip://{Path(LAND).as_posix()}")
    box = shapely.box(lo0, la0, lo1, min(la1, 89))
    rings = []
    for geom in gdf.geometry:
        c = geom.intersection(box)
        if c.is_empty:
            continue
        for poly in getattr(c, "geoms", [c]):
            if poly.geom_type != "Polygon":
                continue
            ex = shapely.segmentize(poly, 0.25).exterior
            xy = np.asarray(ex.coords)
            px, py = g.ll2px(xy[:, 0], xy[:, 1])
            rings.append(np.column_stack([px, py]))
    return rings


def graticule(g: Grid, step: int = 10):
    lines = []
    for lon in range(-80, 60, step):
        la = np.linspace(20, 85, 200)
        lines.append(np.column_stack(g.ll2px(np.full_like(la, lon), la)))
    for lat in range(20, 86, step):
        lo = np.linspace(-90, 60, 400)
        lines.append(np.column_stack(g.ll2px(lo, np.full_like(lo, lat))))
    return lines


def hillshade(g: Grid) -> tuple[np.ndarray, np.ndarray]:
    """(shade 0..1 around 0.5, land mask) from ETOPO 10-arc-min sampled on the pixel grid."""
    z = xr.open_dataset(ETOPO).z
    f = RegularGridInterpolator((z.lat.values, z.lon.values), z.values, bounds_error=False, fill_value=0.0)
    zz = f(np.stack([g.lat, g.lon], -1)).astype(np.float32)
    zz = gaussian_filter(zz, 1.2)
    gy, gx = np.gradient(zz, g.mpp)
    sh = np.clip(0.5 + 18.0 * (-gx - gy) / np.sqrt(2), 0, 1)  # light from the north-west, vertical x ~18
    return sh, zz > 0


# ------------------------------------------------------------------------------------------------------- layers
def context_layer(g: Grid, rings, bg: str, land: str, coast: str, grat: str, fade: int = 0) -> np.ndarray:
    """Muted dark context map: flat land + faint relief + coast + faint graticule. RGB."""
    fig, ax = lc.canvas(g.w, g.h, bg)
    ax.add_collection(PolyCollection(rings, facecolors=land, edgecolors="none"))
    a = lc.grab(fig)[..., :3]
    sh, _ = hillshade(g)
    bgc = lc.hexrgb(bg)
    is_land = np.abs(a - bgc).sum(-1) > 0.02
    a = np.where(is_land[..., None], np.clip(a * (0.72 + 0.56 * sh[..., None]), 0, 1), a)
    fig, ax = lc.canvas(g.w, g.h)
    ax.add_collection(LineCollection(graticule(g), colors=grat, linewidths=1.0))
    ax.add_collection(LineCollection(rings, colors=coast, linewidths=1.3))
    o = lc.grab(fig)
    a = a * (1 - o[..., 3:]) + o[..., :3] * o[..., 3:]
    if fade:
        yy, xx = np.mgrid[0:g.h, 0:g.w]
        d = np.minimum.reduce([xx, yy, g.w - 1 - xx, g.h - 1 - yy]).astype(np.float32)
        k = np.clip(d / fade, 0, 1)[..., None]
        a = bgc * (1 - k) + a * k
    return a


def fill_layer(g: Grid, field: np.ndarray, cmap, vmin: float, vmax: float, rings, coast: str, coast_a: float,
               nan_colour: str = "#000000") -> np.ndarray:
    """Opaque colour-mapped field + thin coast line. RGB. NaN (outside the data) -> nan_colour."""
    rgb = cmap(np.clip((field - vmin) / (vmax - vmin), 0, 1))[..., :3].astype(np.float32)
    rgb = np.where(np.isnan(field)[..., None], lc.hexrgb(nan_colour), rgb)
    fig, ax = lc.canvas(g.w, g.h)
    ax.add_collection(LineCollection(rings, colors=coast, linewidths=1.4, alpha=coast_a))
    o = lc.grab(fig)
    return rgb * (1 - o[..., 3:]) + o[..., :3] * o[..., 3:]


def chart_ground(g: Grid, rings, ocean="#0e1524", land="#1d2942", coast="#6c7da8") -> np.ndarray:
    fig, ax = lc.canvas(g.w, g.h, ocean)
    ax.add_collection(PolyCollection(rings, facecolors=land, edgecolors=coast, linewidths=1.3))
    return lc.grab(fig)[..., :3]


def _contours(g: Grid, field: np.ndarray, levels, colour: str, lw: float, ls="solid", casing="#0b1020", bold=None, labels=None,
              label_px: float = 0) -> np.ndarray:
    fig, ax = lc.canvas(g.w, g.h)
    X, Y = np.meshgrid(np.arange(g.w) + 0.5, np.arange(g.h) + 0.5)
    f = np.ma.masked_invalid(field)
    lws = [lw * (1.75 if (bold and bold(v)) else 1.0) for v in levels]
    cs = ax.contour(X, Y, f, levels=levels, colors=colour, linewidths=lws, linestyles=ls)
    cs.set_path_effects([pe.Stroke(linewidth=lw + 2.6, foreground=casing, alpha=0.55), pe.Normal()])
    if labels and label_px:
        tl = ax.clabel(cs, levels=[v for v in levels if labels(v)], fmt="%d", fontsize=label_px * 0.72, inline=True, inline_spacing=6)
        for t in tl:
            t.set_fontfamily(lc.SANS)
            t.set_fontweight("bold")
            t.set_path_effects([pe.Stroke(linewidth=3, foreground=casing, alpha=0.8), pe.Normal()])
    return lc.grab(fig)


def isobar_layer(g: Grid, msl_hpa: np.ndarray, lw: float = 2.0, label_px: float = 0) -> np.ndarray:
    lev = np.arange(940, 1052, 4)
    return _contours(g, msl_hpa, lev, "#f4ecd2", lw, bold=lambda v: v % 20 == 0, labels=lambda v: v % 8 == 0, label_px=label_px)


def moist_lines(g: Grid, q: np.ndarray, lw: float = 1.8, label_px: float = 0) -> np.ndarray:
    qs = gaussian_filter(np.nan_to_num(q, nan=0.0), 4.0)
    qs = np.where(np.isnan(q), np.nan, qs)
    return _contours(g, qs, [4.0, 6.0, 8.0], "#bff7ea", lw, ls="dashed", labels=lambda v: True, label_px=label_px)


def cmap_stops(cmap, n: int = 11) -> list[str]:
    return ["#%02x%02x%02x" % tuple(int(round(255 * c)) for c in cmap(i / (n - 1))[:3]) for i in range(n)]


# -------------------------------------------------------------------------------------------------------- track
def track_low(msl: xr.DataArray, seed_i: int, seed_box=(-8.0, 2.0, 48.0, 53.0), radius_deg: float = 2.5):
    """Follow ONE low: seed = the minimum inside a small box at hour seed_i (Ciaran over the western Channel at 06 UTC
    2 Nov, 954 hPa), then step backwards and forwards taking the local minimum within radius_deg of the last position
    (never the minimum of a big box, which jumps to the parent low near Iceland)."""
    lat, lon = msl.latitude.values, msl.longitude.values
    n = msl.shape[0]
    out = [None] * n

    def local_min(i, lo0, lo1, la0, la1):
        a = msl.isel(time=i).sel(longitude=slice(lo0, lo1), latitude=slice(la0, la1))
        k = np.unravel_index(int(a.values.argmin()), a.shape)
        return float(a.longitude[k[1]]), float(a.latitude[k[0]]), float(a.values[k]) / 100.0

    out[seed_i] = local_min(seed_i, seed_box[0], seed_box[1], seed_box[2], seed_box[3])
    for rng in (range(seed_i + 1, n), range(seed_i - 1, -1, -1)):
        prev = out[seed_i]
        for i in rng:
            r = radius_deg
            prev = local_min(i, max(lon[0], prev[0] - r * 1.5), min(lon[-1], prev[0] + r * 1.5), max(lat[0], prev[1] - r),
                             min(lat[-1], prev[1] + r))
            out[i] = prev
    return out


# -------------------------------------------------------------------------------------------------------- movie
MOVIE = dict(clon=-15.0, clat=49.3, half_h_km=740.0, w=1140, h=625)


def build_movie(only_ctx: bool = False, data=None, out=None, log=print) -> Path:
    """The map layers of the clip in which the lenses ride the storm: ``ctx.png`` (base map), ``p.png`` (chart ground)
    and, per hour, ``t_NN.png``, ``q_NN.png`` (fills), ``ql_NN.png``, ``pl_NN.png`` (transparent line work), all on one
    1140 x 625 px Lambert grid, plus ``manifest.json`` (hours, tracked low in pixels, a coarse wind grid, colour ramps).
    ``data`` = (sfc, pl, times) from ``load`` / ``load_reduced`` (default: ``load()``); ``out`` = the folder."""
    global PUB
    if out is not None:
        PUB = Path(out)
    PUB.mkdir(parents=True, exist_ok=True)
    sfc, pl, times = data if data is not None else load()
    g = Grid(**MOVIE)
    rings = land_polys(g)
    lc.save_rgb(PUB / "ctx.png", context_layer(g, rings, bg="#171f33", land="#2b3654", coast="#5a6892", grat="#27314c", fade=46))
    lc.save_rgb(PUB / "p.png", chart_ground(g, rings))
    if only_ctx:
        return PUB
    seed_i = int(np.where(times == np.datetime64("2023-11-02T06:00"))[0][0])
    trk = track_low(sfc.mean_sea_level_pressure, seed_i)
    # wind grid in pixel space (every WG px): components along +x (right) and +y (down), m/s x 4 as integers
    WG = 19
    gx, gy = np.meshgrid(np.arange(WG / 2, g.w, WG), np.arange(WG / 2, g.h, WG))
    wlon, wlat = g.px2ll(gx, gy)
    ex = np.stack(g.ll2px(wlon + 0.01, wlat), -1) - np.stack([gx, gy], -1)
    ex /= np.linalg.norm(ex, axis=-1, keepdims=True)
    nn = np.stack(g.ll2px(wlon, wlat + 0.01), -1) - np.stack([gx, gy], -1)
    nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
    wind, low = [], []
    for i, t in enumerate(times):
        T = g.sample(pl.temperature.isel(time=i)) - 273.15
        q = g.sample(pl.specific_humidity.isel(time=i)) * 1000.0
        p = g.sample(sfc.mean_sea_level_pressure.isel(time=i)) / 100.0
        lc.save_rgb(PUB / f"t_{i:02d}.png", fill_layer(g, T, T_CMAP, *T_RANGE, rings, "#0b1020", 0.6))
        lc.save_rgb(PUB / f"q_{i:02d}.png", fill_layer(g, q, Q_CMAP, *Q_RANGE, rings, "#9fb4c8", 0.55))
        lc.save_rgb(PUB / f"ql_{i:02d}.png", moist_lines(g, q))
        lc.save_rgb(PUB / f"pl_{i:02d}.png", isobar_layer(g, p, lw=2.2))
        u = g.sample(sfc["10m_u_component_of_wind"].isel(time=i), wlon, wlat)
        v = g.sample(sfc["10m_v_component_of_wind"].isel(time=i), wlon, wlat)
        vx = u * ex[..., 0] + v * nn[..., 0]
        vy = u * ex[..., 1] + v * nn[..., 1]
        wind.append(np.round(np.stack([vx, vy], -1) * 4).astype(int).reshape(-1).tolist())
        lx, ly = g.ll2px(trk[i][0], trk[i][1])
        low.append({"x": round(float(lx), 1), "y": round(float(ly), 1), "hPa": round(trk[i][2], 1), "lon": trk[i][0], "lat": trk[i][1]})
        log(f"{i:2d}  {str(t)[:13]}  low {trk[i][0]:6.2f} {trk[i][1]:5.2f} {trk[i][2]:6.1f} hPa", flush=True)
    man = {
        "w": g.w, "h": g.h, "km_per_px": g.mpp / 1000.0, "times": [str(t)[:13] for t in times], "low": low,
        "wind": {"step": WG, "nx": int(gx.shape[1]), "ny": int(gx.shape[0]), "scale": 4, "uv": wind},
        "ramps": {"t": cmap_stops(T_CMAP), "q": cmap_stops(Q_CMAP)}, "ranges": {"t": T_RANGE, "q": Q_RANGE}, "rim": RIM,
        "projection": "Lambert conformal conic, standard parallels 45N and 58N, centre 49.3N 15W",
    }
    (PUB / "manifest.json").write_text(json.dumps(man, separators=(",", ":")), encoding="utf-8")
    log(f"layers -> {PUB}  ({sum(f.stat().st_size for f in PUB.iterdir()) / 1e6:.1f} MB)")
    return PUB


# ------------------------------------------------------------------------------------------------------- poster
POSTER_W, POSTER_H = 2000, 2828
LENS_R = 500.0  # lens radius in poster pixels (725 km at 1.45 km per pixel)
LENS_OFF = {"t": (-330, -230), "q": (-150, 330), "p": (330, -60)}  # lens centres relative to the low, poster pixels
INK, SUB, MUT = "#f1ede4", "#c9c5bc", "#8d8a84"  # text colours: main, secondary, muted
BG = "#000000"
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")


def poster_grid(lon0: float, lat0: float) -> Grid:
    """The poster's pixel grid: Lambert conformal about (lon0, lat0), 1.45 km per pixel; the centre sits 60 px right
    of and 140 px below the middle of the sheet, which leaves room for the title."""
    return Grid(clon=lon0, clat=lat0, half_h_km=1414 * 1.45, w=POSTER_W, h=POSTER_H, x0=60, y0=140)  # 1.45 km per px


def lens_centres(cx: float, cy: float) -> dict:
    """Lens centres for a low at poster pixel (cx, cy)."""
    return {k: (cx + dx, cy + dy) for k, (dx, dy) in LENS_OFF.items()}


def wind_arrows(g: Grid, u_da: xr.DataArray, v_da: xr.DataArray, st: int = 46) -> np.ndarray:
    """10-m wind arrows on the pixel grid (RGBA, transparent ground): one arrow every ``st`` px, 2.1 px per m/s.

    ERA5 gives the wind as (u toward east, v toward north). On a Lambert map east is not "right" except on the central
    meridian, so each arrow is turned with the map's own east and north directions at that point."""
    fig, ax = lc.canvas(g.w, g.h)
    gx, gy = np.meshgrid(np.arange(st / 2, g.w, st), np.arange(st / 2, g.h, st))
    wlon, wlat = g.px2ll(gx, gy)
    u = g.sample(u_da, wlon, wlat, strict=False)
    v = g.sample(v_da, wlon, wlat, strict=False)
    ex = np.stack(g.ll2px(wlon + 0.01, wlat), -1) - np.stack([gx, gy], -1)
    ex /= np.linalg.norm(ex, axis=-1, keepdims=True)
    nn = np.stack(g.ll2px(wlon, wlat + 0.01), -1) - np.stack([gx, gy], -1)
    nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
    vx, vy = u * ex[..., 0] + v * nn[..., 0], u * ex[..., 1] + v * nn[..., 1]
    ax.quiver(gx, gy, vx, vy, angles="xy", scale_units="xy", scale=1 / 2.1, color="#8fd0ff", width=0.0011, headwidth=4.2,
              headlength=5, alpha=0.95, edgecolor="#0b1020", linewidth=0.5)  # 2.1 px per m/s: 20 m/s = 42 px
    return lc.grab(fig)


def hour_layers(g: Grid, rings, sfc: xr.Dataset, pl: xr.Dataset, hour) -> dict:
    """Every image the three lenses need for one analysis hour, on the grid ``g``:
    ``t`` / ``q`` opaque colour-mapped fills, ``ql`` moisture contours, ``pg`` the plain chart ground of the pressure
    lens, ``po`` its line work (isobars over wind arrows). ``ql`` and ``po`` are transparent RGBA."""
    hour = np.datetime64(hour, "ns")
    T = g.sample(pl.temperature.sel(time=hour), strict=False) - 273.15
    q = g.sample(pl.specific_humidity.sel(time=hour), strict=False) * 1000.0
    p = g.sample(sfc.mean_sea_level_pressure.sel(time=hour), strict=False) / 100.0
    # wind arrows (10-m wind) for the pressure lens
    arrows = wind_arrows(g, sfc["10m_u_component_of_wind"].sel(time=hour), sfc["10m_v_component_of_wind"].sel(time=hour))
    iso = isobar_layer(g, p, lw=2.4, label_px=24)
    a = arrows[..., 3:] * (1 - iso[..., 3:])
    p_over = np.concatenate([(iso[..., :3] * iso[..., 3:] + arrows[..., :3] * a) / np.maximum(iso[..., 3:] + a, 1e-6), iso[..., 3:] + a], -1)
    return dict(t=fill_layer(g, T, T_CMAP, *T_RANGE, rings, "#0b1020", 0.6),
                q=fill_layer(g, q, Q_CMAP, *Q_RANGE, rings, "#9fb4c8", 0.55), ql=moist_lines(g, q, 2.2, 24),
                pg=chart_ground(g, rings), po=p_over)


def make_lenses(layers: dict, cen: dict, R: float = LENS_R) -> list:
    """The three lenses in drawing order. ``overlap = 0`` on every lens (the no-blend rule for data fields): the lowest
    lens keeps its exact colours, so they still match its colour bar, and each upper lens adds only its line work.
    Order = meaning: the richest fill (temperature) lies lowest, the pure line-work lens (isobars + wind) on top."""
    return [
        lc.Lens(layers["t"], *cen["t"], R, RIM["t"], None, 0.0),
        lc.Lens(layers["q"], *cen["q"], R, RIM["q"], layers["ql"], 0.0),
        lc.Lens(layers["pg"], *cen["p"], R, RIM["p"], layers["po"], 0.0),
    ]


def low_marker(ax, cx: float, cy: float, p0: float) -> None:
    """The low: L + central pressure (the storm every lens is looking at)."""
    ax.text(cx, cy, "L", color="#f4ecd2", fontsize=46, fontfamily=lc.SANS, fontweight="bold", ha="center", va="center", zorder=25,
            path_effects=[pe.Stroke(linewidth=6, foreground="#0b1020"), pe.Normal()])
    ax.text(cx, cy + 62, f"{p0:.0f} hPa", color="#f4ecd2", fontsize=20, fontfamily=lc.SANS, fontweight="bold", ha="center", va="center",
            zorder=25, path_effects=[pe.Stroke(linewidth=5, foreground="#0b1020"), pe.Normal()])


def stamp_text(hour) -> str:
    """'2 NOVEMBER 2023, 00 UTC' for an analysis hour."""
    t = str(np.datetime64(hour, "h"))
    return f"{int(t[8:10])} {MONTHS[int(t[5:7]) - 1].upper()} {t[:4]}, {t[11:13]} UTC"


def _bar(ax, x, y, w, h, cmap, lo, hi, unit, mid=None):
    """A colour bar drawn by hand: the same colour map as the lens fill, end values, optional middle tick."""
    import matplotlib.pyplot as plt

    grad = np.linspace(0, 1, 256)[None, :]
    ax.imshow(grad, extent=(x, x + w, y + h, y), cmap=cmap, aspect="auto", zorder=30)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="#8d8a84", lw=1, zorder=31))
    lc.text(ax, x, y + h + 30, lo, 24, SUB)
    lc.text(ax, x + w, y + h + 30, f"{hi} {unit}", 24, SUB, ha="right")
    if mid:
        lc.text(ax, x + w / 2, y + h + 30, mid, 24, SUB, ha="center")
        ax.plot([x + w / 2, x + w / 2], [y, y + h], color="#1a1a1a", lw=1, zorder=31)


PARTS = ("head", "stamp", "labels", "scale", "foot")


def furniture(fig, ax, g: Grid, cen: dict, R: float = LENS_R, parts=PARTS, stamp: str = "2 NOVEMBER 2023, 00 UTC",
              when: str = "00 UTC 2 November 2023", leaders: bool = True, blocks: bool = True, keys=("t", "q", "p"),
              note: str = "Lambert conformal map; north is up along the meridian of the low") -> None:
    """Everything on the weather poster that is not map: ``head`` (header, title, subtitle), ``stamp`` (date and hour,
    top right), ``labels`` (one block per lens: name, theme, two lines, colour bar or key, leader to the rim),
    ``scale`` (scale bar and projection note) and ``foot`` (closing sentence and credit block). Poster pixels.
    ``keys`` / ``leaders`` / ``blocks`` choose which lenses' label blocks and leader lines are drawn (the animation
    draws them one by one on transparent canvases)."""
    W = POSTER_W
    ink, sub, mut = INK, SUB, MUT
    # header and title
    if "head" in parts:
        lc.text(ax, 60, 62, "LENS MAP · WEATHER STUDY", 24, mut)
    if "stamp" in parts:
        lc.text(ax, W - 60, 62, f"STORM CIARÁN · {stamp}", 24, mut, ha="right")
    if "head" in parts:
        ax.plot([520, W - 760], [54, 54], color="#55534f", lw=1.2, zorder=30)
        lc.text(ax, 84, 250, "ONE STORM,", 104, ink, lc.SERIF, "bold")
        t1 = lc.text(ax, 84, 372, "MANY ", 104, ink, lc.SERIF, "bold")
        fig.canvas.draw()
        x1 = t1.get_window_extent().x1
        lc.text(ax, x1 + 4, 372, "REALITIES.", 104, "#e8a25c", lc.SERIF, "bold", "italic")
        lc.text(ax, 86, 436, "Each field shows part of the storm. Where the lenses overlap, the parts fit together.", 32, sub, lc.SERIF,
                style="italic")
    if "labels" in parts:
        # label blocks (bar + caps + italic theme + description) with leaders to the rims
        s = 40
        blocks_all, leaders_all = blocks, leaders
        blocks, leaders = blocks_all and "t" in keys, leaders_all and "t" in keys
        if blocks:
            yb = lc.label_block(ax, 84, 600, "TEMPERATURE", "Warmth", ["Air temperature 1.5 km up", "(850 hPa): warm sector, cold air"], RIM["t"], s)
            _bar(ax, 100, yb + 22, 330, 20, T_CMAP, "−12", "+12", "°C", "0")
        else:
            yb = 600 + s * 0.95 + s * 1.12 + (2 - 0.15) * s * 0.92  # the y that label_block returns for two lines
        if leaders:
            lc.leader(ax, (270, yb + 96), (cen["t"][0] - R * 0.72, cen["t"][1] - R * 0.42))
        yq = 2280
        blocks, leaders = blocks_all and "q" in keys, leaders_all and "q" in keys
        if blocks:
            yb = lc.label_block(ax, 84, yq, "MOISTURE", "Fuel", ["Water vapour 1.5 km up (850 hPa);", "dashed lines 4, 6, 8 g/kg"], RIM["q"], s)
            _bar(ax, 100, yb + 22, 330, 20, Q_CMAP, "0", "10", "g/kg")
        if leaders:
            lc.leader(ax, (330, yq - 14), (cen["q"][0] - R * 0.66, cen["q"][1] + R * 0.5))
        yp = 600
        blocks, leaders = blocks_all and "p" in keys, leaders_all and "p" in keys
        if blocks:
            yb = lc.label_block(ax, 1440, yp, "PRESSURE & WIND", "Motion", ["Sea-level isobars every 4 hPa;", "arrows: wind 10 m above the surface"],
                                RIM["p"], s)
            ax.annotate("", xy=(1456 + 42, yb + 40), xytext=(1456, yb + 40), arrowprops=dict(arrowstyle="-|>", color="#8fd0ff", lw=2.2), zorder=30)
            lc.text(ax, 1516, yb + 49, "20 m/s", 24, sub)
            ax.plot([1640, 1720], [yb + 40, yb + 40], color="#f4ecd2", lw=2.4, zorder=30)
            lc.text(ax, 1736, yb + 49, "isobar", 24, sub)
        else:
            yb = yp + s * 0.95 + s * 1.12 + (2 - 0.15) * s * 0.92
        if leaders:
            lc.leader(ax, (1560, yb + 80), (cen["p"][0] + R * 0.5, cen["p"][1] - R * 0.866))
    if "scale" in parts:
        # scale bar (true near 45 N and 58 N; within ~2 % across the lenses)
        kmpp = g.mpp / 1000.0
        sx, sy, L = 1330, 2330, 500 / kmpp
        ax.plot([sx, sx, sx + L, sx + L], [sy - 12, sy, sy, sy - 12], color=ink, lw=1.6, zorder=30)
        ax.plot([sx + L / 2, sx + L / 2], [sy - 8, sy], color=ink, lw=1.6, zorder=30)
        for k, lab in ((0, "0"), (0.5, "250"), (1, "500 km")):
            lc.text(ax, sx + L * k, sy - 24, lab, 24, ink, ha="center")
        lc.text(ax, sx, sy + 38, note, 22, mut, lc.SERIF, style="italic")
    if "foot" in parts:
        # footer
        ax.plot([84, W - 84], [2580, 2580], color="#55534f", lw=1.2, zorder=30)
        lc.text(ax, 84, 2660, "One storm, three fields of the same hour. Temperature, moisture and pressure", 30, sub, lc.SERIF, style="italic")
        lc.text(ax, 84, 2702, "with wind each show a different part of the same weather system.", 30, sub, lc.SERIF, style="italic")
        lc.credit_block(ax, 1330, ["DATA: ERA5 hourly reanalysis (Copernicus C3S / ECMWF)",
                                   "Relief: ETOPO 2022 (NOAA NCEI) · coast: Natural Earth",
                                   f"Storm Ciarán · {when}",
                                   "Style after Fan Mo, #30DayMapChallenge 2026 Day 14"], lc.AUTHOR, ink, mut)


def poster_base(g: Grid, rings) -> np.ndarray:
    """The muted dark map seen outside the lenses (black ground, dark land with faint relief, coast, graticule)."""
    return context_layer(g, rings, bg=BG, land="#1b1c20", coast="#34363d", grat="#101114")


def build_poster(hour: str = "2023-11-02T00:00", data=None, out=None) -> Path:
    """Draw the weather poster for one analysis hour. ``data`` = (sfc, pl, times) from :func:`load` (default),
    :func:`open_arco` or :func:`load_reduced`; ``out`` = the PNG to write."""
    import matplotlib.pyplot as plt

    out = Path(out) if out else OUT / "lens_weather.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = POSTER_W, POSTER_H
    sfc, pl, _times = data or load()
    ts = sfc.time.values
    i = int(np.where(ts == np.datetime64(hour))[0][0])
    trk = track_low(sfc.mean_sea_level_pressure, int(np.where(ts == np.datetime64(SEED_TIME))[0][0]))
    lon0, lat0, p0 = trk[i]
    g = poster_grid(lon0, lat0)
    rings = land_polys(g, pad_deg=10)
    base = poster_base(g, rings)
    R = LENS_R
    cx, cy = g.ll2px(lon0, lat0)
    cx, cy = float(cx), float(cy)
    cen = lens_centres(cx, cy)
    lenses = make_lenses(hour_layers(g, rings, sfc, pl, hour), cen, R)
    img = lc.composite(base, lenses)
    fig, ax = lc.canvas(W, H, BG)
    ax.imshow(img, extent=(0, W, H, 0), interpolation="nearest", zorder=1)
    lc.rims(ax, lenses, 3.2)
    low_marker(ax, cx, cy, p0)
    t = str(np.datetime64(hour, "h"))
    furniture(fig, ax, g, cen, R, stamp=stamp_text(hour), when=f"{t[11:13]} UTC {int(t[8:10])} {MONTHS[int(t[5:7]) - 1]} {t[:4]}")
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    print("poster ->", out, "low", lon0, lat0, p0, "km/px", g.mpp / 1000.0)
    return out


if __name__ == "__main__":
    what = sys.argv[1:] or ["movie", "poster"]
    if "movie" in what or "ctx" in what:
        build_movie(only_ctx="ctx" in what)
    if "poster" in what:
        build_poster()
