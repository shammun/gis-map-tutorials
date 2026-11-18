"""gismaps.glass.gfs: the data of the glass-pane stack.

Source. NOAA GFS forecast, 0.25 degrees, as published by dynamical.org (https://dynamical.org, catalogue
https://stac.dynamical.org/noaa-gfs-forecast/collection.json): an Icechunk / Zarr store on public cloud storage, read
anonymously. Licence CC BY 4.0; credit "dynamical.org / NOAA GFS". Relief: ETOPO 2022 (NOAA NCEI, public domain).

Three small files are shipped in ``data/glass_stack/`` so nothing has to be downloaded from dynamical.org:

``gfs_south_asia_20261116T18.nc``   the region of the poster and the animation (0-44 N, 62-112 E, 0.25 degrees):
                                    six fields at 17 lead times (+6 ... +102 h, every 6 h) of the run of
                                    16 Nov 2026 18 UTC
``gfs_hemisphere_20261116T18.nc``   the hemisphere of the Remotion version (centred on 45 N 30 W, 0.5 degrees):
                                    five fields at 8 lead times (+6 ... +48 h) of the same run
``etopo2022_10min.nc``              heights of land and sea floor on a 10 arc-minute grid

``region(source="shipped")`` and ``hemisphere(source="shipped")`` read them. With ``source="live"`` the same fields
are read from dynamical.org (needs ``pip install icechunk zarr``; the region is about 35 MB of chunks, the hemisphere
about 240 MB) and kept in the work folder (``GISMAPS_WORK``, default ``./gismaps_work``); if the live read fails for
any reason, the shipped file is used and a line says so.

The shipped files were written by this module::

    python -m gismaps.glass.gfs region        # live read of the run RUN -> gfs_south_asia_<run>.nc
    python -m gismaps.glass.gfs hemisphere    # live read            -> gfs_hemisphere_<run>.nc
    python -m gismaps.glass.gfs relief        # ETOPO 2022 through OPeNDAP -> etopo2022_10min.nc
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

from .. import data as _data

GROUP = "glass_stack"
RUN = "2026-11-16T18:00"                                     # the run of the shipped files
STAC = "https://stac.dynamical.org/noaa-gfs-forecast/collection.json"
CREDIT = "Data: dynamical.org / NOAA GFS (CC BY 4.0)"
RELIEF_CREDIT = "relief: ETOPO 2022 (NOAA NCEI)"
ETOPO_DAP = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
WORK = Path(os.environ.get("GISMAPS_WORK", "gismaps_work")) / GROUP

#: the region of the poster: (south, north), (west, east); it sits inside four storage chunks of the store
REGION_BOX = dict(lat=(0.0, 44.0), lon=(62.0, 112.0))
REGION_LEADS = list(range(6, 103, 6))                        # +6 ... +102 h (cloud and rain are averages: none at 0 h)
#: the hemisphere of the Remotion version, as two boxes: 90-30 N all longitudes, and 30 N-50 S from 160 W to 100 E
HEMI_BOXES = [dict(lat=(30.0, 90.0), lon=None), dict(lat=(-50.0, 29.5), lon=(-160.0, 100.0))]
HEMI_LEADS = [6, 12, 18, 24, 30, 36, 42, 48]
HEMI_CENTRE = (45.0, -30.0)                                  # the hemisphere every pane of the Remotion version shows

#: short name -> (name in the store, unit after conversion, step the shipped file keeps)
VARS = {
    "t2m": ("temperature_2m", "degC", 0.25),
    "mslp": ("pressure_reduced_to_mean_sea_level", "hPa", 0.64),
    "u10": ("wind_u_10m", "m s-1", 0.1),
    "v10": ("wind_v_10m", "m s-1", 0.1),
    "wind": ("wind speed at 10 m", "m s-1", 0.1),
    "pwat": ("precipitable_water_atmosphere", "kg m-2", 0.1),
    "tcc": ("total_cloud_cover_atmosphere", "percent", 1.0),
    "prate": ("precipitation_surface", "mm h-1", 0.01),
    "rain": ("precipitation rate, mean of the 6 h before the lead time", "mm h-1", 0.01),
}


def run_id(run) -> str:
    """'2026-11-16T18:00' -> '20261116T18' (part of the file names)."""
    return str(run)[:13].replace("-", "").replace(":", "")


# ---------------------------------------------------------------- the shipped files
def _shipped(name: str, log=print) -> xr.Dataset:
    with xr.open_dataset(_data.path(GROUP, name, log=log)) as ds:
        return ds.load()


def relief(log=print) -> dict:
    """Heights of land and sea floor in metres on a 10 arc-minute grid, north -> south: {"z", "lat", "lon", "credit"}."""
    ds = _shipped("etopo2022_10min.nc", log)
    return {"z": ds["z"].values.astype(np.float32), "lat": ds["lat"].values, "lon": ds["lon"].values, "credit": RELIEF_CREDIT}


def region(source: str = "shipped", run: str | None = RUN, log=print) -> xr.Dataset:
    """The six fields of the region, dims (lead, latitude, longitude), in display units:
    t2m [degC], mslp [hPa], wind [m s-1], pwat [kg m-2], tcc [percent], rain [mm h-1, mean of the 6 h before].

    ``source="shipped"`` reads the file of the repository. ``source="live"`` reads run ``run`` (``None`` = the newest
    complete run) from dynamical.org and falls back to the shipped file if that fails.
    """
    return _load("region", source, run, log)


def hemisphere(source: str = "shipped", run: str | None = RUN, log=print) -> xr.Dataset:
    """The five fields of the hemisphere centred on 45 N 30 W at 0.5 degrees (NaN elsewhere), dims (lead, latitude,
    longitude): t2m [degC], mslp [hPa], pwat [kg m-2], tcc [percent], prate [mm h-1]."""
    return _load("hemisphere", source, run, log)


def _load(kind: str, source: str, run, log) -> xr.Dataset:
    stem = {"region": "gfs_south_asia", "hemisphere": "gfs_hemisphere"}[kind]
    if source == "live":
        try:
            cached = WORK / f"{stem}_{run_id(run)}.nc" if run else None
            if cached is not None and cached.exists():
                log(f"live data already in the work folder: {cached}")
                with xr.open_dataset(cached) as ds:
                    return ds.load()
            ds = (live_region if kind == "region" else live_hemisphere)(run, log)
            p = write(ds, WORK / f"{stem}_{run_id(ds.attrs['run'])}.nc")
            log(f"kept in the work folder: {p} ({p.stat().st_size / 1e6:.1f} MB)")
            with xr.open_dataset(p) as ds:                    # read back, so live and shipped data pass the same rounding
                return ds.load()
        except Exception as err:  # noqa: BLE001  (no network, package missing, run no longer in the store ...)
            log(f"live read from dynamical.org failed ({type(err).__name__}: {str(err)[:200]})\n"
                f"-> using the shipped file of run {RUN} instead")
    return _shipped(f"{stem}_{run_id(RUN)}.nc", log)


def write(ds: xr.Dataset, path) -> Path:
    """Write a dataset as a small netCDF file: 16-bit integers with a step per variable (``VARS``), compressed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {}
    for k in ds.data_vars:
        step = VARS[k][2]
        enc[k] = {"dtype": "int16", "scale_factor": step, "add_offset": 999.68 if k == "mslp" else 0.0, "_FillValue": -32768,
                  "zlib": True, "complevel": 9, "shuffle": True}
    tmp = path.with_name(path.name + ".part")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(path)
    return path


# ---------------------------------------------------------------- the live route: dynamical.org
_STORE: dict = {}


def open_store(log=print) -> xr.Dataset:
    """Open the GFS forecast store LAZILY (names, sizes and coordinates only), anonymously."""
    import icechunk                                              # the reader of dynamical.org's storage format

    if "ds" not in _STORE:
        req = urllib.request.Request(STAC, headers={"User-Agent": _data.USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:       # the catalogue card says where the store lives
            href = json.loads(r.read().decode("utf-8"))["assets"]["icechunk-https"]["href"]
        repo = icechunk.Repository.open(icechunk.http_storage(href))
        store = repo.readonly_session("main").store              # the newest published state
        _STORE["ds"] = xr.open_zarr(store, chunks=None, decode_timedelta=True, consolidated=False)
        _STORE["href"] = href
        log(f"opened {href}")
    return _STORE["ds"]


def newest_complete_run(lead_h: int = 102, log=print) -> str:
    """The newest run whose forecast is already written out to ``lead_h`` hours (reads a few single values)."""
    ds = open_store(log)
    a = ds[VARS["mslp"][0]]
    for i in range(ds.sizes["init_time"] - 1, ds.sizes["init_time"] - 9, -1):
        v = a.isel(init_time=i, latitude=360, longitude=720).sel(lead_time=np.timedelta64(lead_h, "h")).values
        if np.isfinite(v):
            return str(ds["init_time"].values[i])[:16]
    raise RuntimeError("no complete run among the newest eight")


def _lon_positions(lon, west, east):
    """Column numbers of the longitudes from west to east; works across the date line."""
    off = (lon - west) % 360.0
    width = (east - west) % 360.0 or 360.0
    idx = np.nonzero(off <= width + 1e-6)[0]
    return idx[np.argsort(off[idx], kind="stable")]


def read_box(var: str, run: str, leads, lat=None, lon=None, stride: int = 1, log=print) -> xr.DataArray:
    """One variable of one run for some lead hours and a box (``lat`` = (south, north), ``lon`` = (west, east) or
    ``None`` for all), every ``stride``-th grid point. Only the chunks the box touches travel over the network."""
    import pandas as pd

    ds = open_store(log)
    a = ds[VARS[var][0]]
    i_run = int(ds.indexes["init_time"].get_loc(pd.Timestamp(run)))
    i_lead = ds.indexes["lead_time"].get_indexer(pd.to_timedelta(list(leads), unit="h"))
    if (i_lead < 0).any():
        raise KeyError(f"some lead hours are not in the store: {list(leads)}")
    la, lo = ds["latitude"].values, ds["longitude"].values       # north -> south; -180 ... 179.75
    i_lat = np.arange(la.size) if lat is None else np.nonzero((la >= min(lat)) & (la <= max(lat)))[0]
    i_lon = np.arange(lo.size) if lon is None else _lon_positions(lo, *lon)
    t0 = time.perf_counter()
    sel = a.isel(init_time=i_run, lead_time=i_lead, latitude=i_lat[::stride], longitude=i_lon[::stride])
    for attempt in range(4):                                     # the network can hiccup: try up to four times
        try:
            got = sel.load()
            break
        except Exception as err:  # noqa: BLE001
            if attempt == 3:
                raise
            log(f"  retry {attempt + 1} after {type(err).__name__}")
            time.sleep(5 * (attempt + 1))
    log(f"  read {var} ({VARS[var][0]}): {got.shape} in {time.perf_counter() - t0:.0f} s")
    got = got.reset_coords(drop=True).astype("float32")
    return got.assign_coords(lead_time=(got["lead_time"].values / np.timedelta64(1, "h")).astype("int32")).rename(lead_time="lead")


def _attrs(run: str, what: str) -> dict:
    return {"run": str(run)[:16], "source": "NOAA GFS forecast, 0.25 degrees, read from dynamical.org (Icechunk store)",
            "store": _STORE.get("href", ""), "catalogue": STAC, "licence": "CC BY 4.0", "credit": CREDIT, "subset": what}


def live_region(run: str | None = None, log=print) -> xr.Dataset:
    """Read the six fields of the region from dynamical.org (about 35 MB of chunks, about a minute)."""
    run = run or newest_complete_run(REGION_LEADS[-1], log)
    log(f"reading GFS run {run} UTC from dynamical.org ...")
    six = {k: read_box(k, run, REGION_LEADS, **REGION_BOX, log=log) for k in ("t2m", "mslp", "u10", "v10", "pwat", "tcc")}
    hourly = read_box("prate", run, range(1, REGION_LEADS[-1] + 1), **REGION_BOX, log=log)      # kg m-2 s-1, mean of each hour
    h = hourly["lead"].values
    rain = np.stack([hourly.values[(h > lead - 6) & (h <= lead)].mean(axis=0) * 3600.0 for lead in REGION_LEADS])
    ds = xr.Dataset({
        "t2m": six["t2m"],
        "mslp": six["mslp"] / 100.0,                             # Pa -> hPa
        "wind": np.hypot(six["u10"], six["v10"]),                # speed = length of the (u, v) arrow
        "pwat": six["pwat"],
        "tcc": six["tcc"],
        "rain": (six["t2m"].dims, rain.astype("float32")),       # kg m-2 s-1 -> mm per hour, mean of the 6 h before
    })
    return _finish(ds, run, "0-44 N, 62-112 E at 0.25 degrees; leads +6 ... +102 h every 6 h")


def live_hemisphere(run: str | None = None, log=print) -> xr.Dataset:
    """Read the five fields of the hemisphere at 0.5 degrees from dynamical.org (about 240 MB of chunks, about ten
    minutes) and place them on a global 0.5-degree grid (NaN outside the two boxes)."""
    run = run or newest_complete_run(HEMI_LEADS[-1], log)
    log(f"reading GFS run {run} UTC from dynamical.org (two boxes, every 2nd grid point) ...")
    ds0 = open_store(log)
    lat = ds0["latitude"].values[::2].astype("float32")          # 90 ... -90
    lon = ds0["longitude"].values[::2].astype("float32")         # -180 ... 179.5
    out = {}
    for k in ("t2m", "mslp", "pwat", "tcc", "prate"):
        full = np.full((len(HEMI_LEADS), lat.size, lon.size), np.nan, np.float32)
        for box in HEMI_BOXES:
            a = read_box(k, run, HEMI_LEADS, stride=2, log=log, **box)
            i = np.searchsorted(-lat, -a["latitude"].values.astype("float32"))
            j = np.searchsorted(lon, a["longitude"].values.astype("float32"))
            full[:, i[:, None], j[None, :]] = a.values
        out[k] = (("lead", "latitude", "longitude"), full * {"mslp": 0.01, "prate": 3600.0}.get(k, 1.0))
    ds = xr.Dataset(out, coords={"lead": np.array(HEMI_LEADS, "int32"), "latitude": lat, "longitude": lon})
    from . import ortho

    la, lo = np.meshgrid(lat, lon, indexing="ij")               # keep the hemisphere and a margin of 2 degrees
    ds = ds.where(xr.DataArray(ortho.forward(la, lo, *HEMI_CENTRE)[2] > -0.03, dims=("latitude", "longitude")))
    return _finish(ds, run, "every 2nd grid point (0.5 degrees); the hemisphere centred on 45 N 30 W (plus a margin of 2 degrees), "
                            "NaN elsewhere; leads +6 ... +48 h")


def _finish(ds: xr.Dataset, run: str, what: str) -> xr.Dataset:
    for k in ds.data_vars:
        ds[k].attrs = {"long_name": VARS[k][0], "units": VARS[k][1]}
        ds[k].encoding = {}
    for c in ds.coords:
        ds[c].attrs, ds[c].encoding = {}, {}
    ds["lead"].attrs = {"long_name": "forecast lead time", "units": "hours"}
    ds["latitude"].attrs, ds["longitude"].attrs = {"units": "degrees_north"}, {"units": "degrees_east"}
    ds.attrs = _attrs(run, what)
    return ds


def build_relief(log=print) -> Path:
    """ETOPO 2022 (1 arc-minute, through OPeNDAP) -> every 10th point, heights rounded to 5 m."""
    try:
        z = xr.open_dataset(ETOPO_DAP, engine="pydap")["z"]
    except Exception:  # noqa: BLE001
        z = xr.open_dataset(ETOPO_DAP)["z"]
    z = z.isel(lat=slice(0, None, 10), lon=slice(0, None, 10)).load()
    z = z.isel(lat=slice(None, None, -1))                        # north -> south, like the forecast grids
    ds = xr.Dataset({"z": (("lat", "lon"), z.values.astype("float32"), {"units": "m", "long_name": "height of land and sea floor"})},
                    coords={"lat": z["lat"].values, "lon": z["lon"].values},
                    attrs={"source": "ETOPO 2022 60 arc-second surface elevation (NOAA NCEI), every 10th point", "licence": "public domain"})
    path = _data.write_folder(GROUP) / "etopo2022_10min.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding={"z": {"dtype": "int16", "scale_factor": 5.0, "_FillValue": -32768, "zlib": True, "complevel": 9, "shuffle": True}})
    return path


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "relief":
        p = build_relief()
    elif what in ("region", "hemisphere"):
        d = (live_region if what == "region" else live_hemisphere)(sys.argv[2] if len(sys.argv) > 2 else RUN)
        p = write(d, _data.write_folder(GROUP) / f"{'gfs_south_asia' if what == 'region' else 'gfs_hemisphere'}_{run_id(d.attrs['run'])}.nc")
    else:
        raise SystemExit("usage: python -m gismaps.glass.gfs region|hemisphere|relief [run, e.g. 2026-11-16T18:00]")
    print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")
