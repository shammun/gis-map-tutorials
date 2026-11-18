"""Build the small reduced inputs of the seven style tutorials from public sources.

Every style tutorial reads one small ``.npz`` file from ``data/gis_styles/``. This module is the record of how each
file is made, and it can rebuild any of them:

    python -m gismaps.prepare_data cyclones      # IBTrACS v04r01 (NOAA NCEI, public domain)           -> cyclones.npz
    python -m gismaps.prepare_data ciaran        # ERA5 hourly msl, u10, v10 (Copernicus C3S / ECMWF)  -> ciaran.npz
    python -m gismaps.prepare_data terrain       # ETOPO 2022 (NOAA NCEI, public domain)               -> terrain_nweurope.npz
    python -m gismaps.prepare_data tcwv          # monthly total column water vapour                   -> tcwv_southasia.npz
    python -m gismaps.prepare_data oisst         # NOAA OISST v2.1 daily SST and sea ice               -> oisst_2025.npz
    python -m gismaps.prepare_data woa           # World Ocean Atlas 2023 annual temperature           -> woa23_temperature.npz
    python -m gismaps.prepare_data sentinel      # Sentinel-2 L2A via public STAC (no account)         -> jamuna_s2.npz
    python -m gismaps.prepare_data sentinel_gee  # the same through Google Earth Engine (account)      -> jamuna_s2_gee.npz
    python -m gismaps.prepare_data all           # everything except sentinel_gee

Every source is public and needs no key (``sentinel_gee`` needs your own Earth Engine account). Downloads go to
``./gismaps_work/raw/`` (``GISMAPS_WORK`` changes the folder) and are reused on the next run; the reduced file is
written to ``$GISMAPS_DATA/gis_styles/`` if that variable is set, else to ``data/gis_styles/`` of a clone, else to
``./gismaps_data/gis_styles/``.

ERA5 note. The keyless public route to ERA5 is ARCO-ERA5 on Google Cloud (``gs://gcp-public-data-arco-era5``), read
with xarray + gcsfs; it transfers whole global chunks (about 2 MB per surface variable and hour). Monthly ERA5 means
have no keyless route: the shipped ``tcwv_southasia.npz`` was made from ERA5 monthly means downloaded with a free
Copernicus CDS account (pass that file to ``tcwv(era5_file=...)``); without it the builder uses the NCEP/NCAR
Reanalysis 1 precipitable-water climatology from NOAA PSL (2.5 degrees instead of 1 degree; the map is smoother, the
seasonal cycle is the same).
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

from . import data as _data

OUT = _data.write_folder("gis_styles")                              # where the reduced files are written
RAW = Path(os.environ.get("GISMAPS_WORK", "gismaps_work")) / "raw"  # downloads, kept for the next run

IBTRACS = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/"
           "access/netcdf/IBTrACS.{basin}.v04r01.nc")
OISST = ("https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/"
         "{d:%Y%m}/oisst-avhrr-v02r01.{d:%Y%m%d}.nc")
WOA = "https://www.ncei.noaa.gov/thredds-ocean/fileServer/woa23/DATA/temperature/netcdf/decav/1.00/woa23_decav_t00_01.nc"
ETOPO_DAP = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
SEAICE = "https://noaadata.apps.nsidc.org/NOAA/G02135/{h}/daily/data/{H}_seaice_extent_daily_v4.0.csv"
NCEP_PW = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface/pr_wtr.eatm.mon.ltm.1991-2020.nc"


def get(url: str) -> Path:
    """Download one public file into the raw folder (only the first time)."""
    name = url.rsplit("/", 1)[-1]
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / name
    if not p.exists():
        print("download", url)
        tmp = p.with_name(name + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(p)
    return p


def save(name: str, **arrays) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    np.savez_compressed(p, **arrays)
    print(f"wrote {p.name}  {p.stat().st_size / 1e6:.2f} MB")
    return p


# ---------------------------------------------------------------- 1 cyclones
def cyclones(y0: int = 1990, y1: int = 2024) -> Path:
    """Six-hourly best-track fixes of every tropical cyclone, all basins, y0..y1, at tropical-storm strength or more."""
    import xarray as xr
    lat, lon, wind, doy, year = [], [], [], [], []
    for basin in ("NA", "EP", "WP", "NI", "SI", "SP", "SA"):
        ds = xr.open_dataset(get(IBTRACS.format(basin=basin)))
        t = ds["time"].values                      # (storm, date_time) datetime64, NaT where no fix
        w = ds["usa_wind"].values                  # knots, 1-minute sustained (NHC / JTWC)
        la, lo = ds["lat"].values, ds["lon"].values
        ok = ~np.isnat(t) & np.isfinite(w) & (w >= 34)
        tt = t[ok].astype("datetime64[s]")
        hours = (tt - tt.astype("datetime64[D]")).astype(int) // 3600
        yrs = tt.astype("datetime64[Y]").astype(int) + 1970
        keep = (hours % 6 == 0) & ((tt - tt.astype("datetime64[h]")).astype(int) == 0) & (yrs >= y0) & (yrs <= y1)
        d = (tt.astype("datetime64[D]") - tt.astype("datetime64[Y]")).astype(int)   # 0-based day of year
        lat.append(la[ok][keep]); lon.append(lo[ok][keep]); wind.append(w[ok][keep])
        doy.append(d[keep]); year.append(yrs[keep])
        ds.close()
    lat, lon, wind, doy, year = map(np.concatenate, (lat, lon, wind, doy, year))
    # a storm that crosses a basin boundary is in two basin files: drop exact duplicates
    key = np.round(lat * 10).astype(np.int64) * 10**9 + np.round((lon % 360) * 10).astype(np.int64) * 10**5 \
        + year.astype(np.int64) * 0 + doy
    _, first = np.unique(np.stack([key, year, wind.astype(np.int64)]), axis=1, return_index=True)
    first.sort()
    return save("cyclones.npz", lat=lat[first].astype(np.float32), lon=((lon[first] + 180) % 360 - 180).astype(np.float32),
                wind_kt=wind[first].astype(np.int16), doy=doy[first].astype(np.int16), year=year[first].astype(np.int16),
                source="IBTrACS v04r01 (Knapp et al. 2010), NOAA NCEI, public domain; usa_wind >= 34 kt, 6-hourly fixes")


# ---------------------------------------------------------------- 2-3 Ciaran (ERA5)
def ciaran() -> Path:
    """Hourly mean-sea-level pressure and 10-m wind, 1 Nov 00 UTC - 2 Nov 2023 23 UTC, 40-62 N, 25 W-10 E, 0.25 deg."""
    import xarray as xr
    names = {"msl": "mean_sea_level_pressure", "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind"}
    ds = xr.open_zarr(ARCO, storage_options={"token": "anon"}, chunks=None)[list(names.values())]   # anonymous, lazy
    ds = ds.sel(time=slice("2023-11-01T00", "2023-11-02T23"))
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360 - 180)).sortby("longitude").sortby("latitude")
    ds = ds.sel(latitude=slice(40, 62), longitude=slice(-25, 10)).load()
    return save("ciaran.npz", lat=ds.latitude.values.astype(np.float32), lon=ds.longitude.values.astype(np.float32),
                time=ds.time.values.astype("datetime64[h]").astype(str),
                msl_hpa10=np.round(ds[names["msl"]].values / 10).astype(np.int16),          # hPa x 10
                u10_cms=np.round(ds[names["u10"]].values * 100).astype(np.int16),          # cm/s
                v10_cms=np.round(ds[names["v10"]].values * 100).astype(np.int16),
                source="ERA5 hourly single levels, Copernicus Climate Change Service (C3S) / ECMWF")


def terrain() -> Path:
    """ETOPO 2022 surface elevation, 40-58 N, 12 W-10 E, thinned to 2 arc-minutes."""
    import xarray as xr
    try:                                         # OPeNDAP: only the asked box travels (pydap also works on Windows)
        z = xr.open_dataset(ETOPO_DAP, engine="pydap")["z"]
    except Exception:                            # noqa: BLE001
        z = xr.open_dataset(ETOPO_DAP)["z"]
    z = z.sel(lat=slice(40, 58), lon=slice(-12, 12))
    z = z.sel(lon=slice(-12, 10)).coarsen(lat=2, lon=2, boundary="trim").mean().load()
    return save("terrain_nweurope.npz", lat=z.lat.values.astype(np.float32), lon=z.lon.values.astype(np.float32),
                z_m=np.round(z.values).astype(np.int16), source="ETOPO 2022 (NOAA NCEI), public domain")


# ---------------------------------------------------------------- 4 water vapour
def tcwv(era5_file=None) -> Path:
    """Monthly climatology 1991-2020 of total column water vapour, 0-45 N, 55-105 E (kg m-2 = mm of water).

    ``era5_file``: ERA5 monthly means of ``tcwv`` on 1 degree, downloaded from the Copernicus Climate Data Store (free
    account); also read from the environment variable ``GISMAPS_ERA5_TCWV``. Without it: NCEP/NCAR Reanalysis 1.
    """
    import xarray as xr
    local = era5_file or os.environ.get("GISMAPS_ERA5_TCWV")
    if local and Path(local).exists():
        da = xr.open_dataset(local)["tcwv"].rename(valid_time="time", latitude="lat", longitude="lon")
        da = da.sel(time=slice("1991-01", "2020-12")).groupby("time.month").mean().sortby("lat")
        src = "ERA5 monthly means 1991-2020, Copernicus Climate Change Service (C3S) / ECMWF"
    else:
        da = xr.open_dataset(get(NCEP_PW), decode_times=False)["pr_wtr"].sortby("lat")
        da = da.rename(time="month").assign_coords(month=np.arange(1, 13))
        src = "NCEP/NCAR Reanalysis 1 precipitable water, 1991-2020 means, NOAA PSL"
    da = da.sel(lat=slice(0, 45), lon=slice(55, 105)).load()
    return save("tcwv_southasia.npz", lat=da.lat.values.astype(np.float32), lon=da.lon.values.astype(np.float32),
                tcwv=da.values.astype(np.float32), source=src)


# ---------------------------------------------------------------- 5 OISST
def _oisst_bundle(days, path: Path) -> None:
    """Download the daily global files of ``days`` and write their sst and ice as ONE netCDF (0.25 degrees)."""
    import tempfile
    import xarray as xr
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for d in days:
            f = Path(tmp) / f"{d}.nc"
            urllib.request.urlretrieve(OISST.format(d=d), f)
            with xr.open_dataset(f) as ds:
                parts.append(ds[["sst", "ice"]].isel(zlev=0, drop=True).load())
    xr.concat(parts, "time").to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in ("sst", "ice")})


def seaice_index(year: int) -> tuple[np.ndarray, np.ndarray]:
    """Daily sea-ice extent of ``year`` (million km2), north and south: NSIDC Sea Ice Index v4 (G02135).

    This is the reference series for extent. (OISST carries an ice field for its own SST analysis; summing it gives
    extents up to 3 million km2 too large in the southern summer, with spurious ice near 52 S, so it is not used
    for the chart.)
    """
    out = []
    for h, H in (("north", "N"), ("south", "S")):
        rows = np.genfromtxt(get(SEAICE.format(h=h, H=H)), delimiter=",", skip_header=2,
                             usecols=(0, 1, 2, 3))
        rows = rows[rows[:, 0] == year]
        doy = np.array([dt.date(int(y), int(m), int(d)).timetuple().tm_yday - 1 for y, m, d, _ in rows])
        out.append(np.interp(np.arange(365), doy, rows[:, 3]).astype(np.float32))
    return out[0], out[1]


def oisst(year: int = 2025) -> Path:
    """SST and sea-ice concentration on the 1st and 16th of each month, 0.5 deg; ice extent of both hemispheres."""
    import xarray as xr
    days = [dt.date(year, m, d) for m in range(1, 13) for d in (1, 16)]
    name = f"oisst_v21_sst-ice_{year}_1st-16th_24days.nc"
    RAW.mkdir(parents=True, exist_ok=True)
    bundle = RAW / name                                            # the 24 daily global files as one netCDF (45 MB)
    if not bundle.exists():
        _oisst_bundle(days, bundle)
    ice_n, ice_s = seaice_index(year)
    ds = xr.open_dataset(bundle)
    lat = ds["lat"].values
    cell = (111.195 * 0.25) ** 2 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, ds.sizes["lon"]))   # km2 per cell
    sst, ice, ext_n, ext_s = [], [], [], []
    for k in range(len(days)):
        s, c = ds["sst"].values[k], ds["ice"].values[k]            # (720, 1440); ice is a 0..1 fraction, NaN = none
        icy = np.nan_to_num(c) >= 0.15
        ext_n.append((cell * icy)[lat > 0].sum() / 1e6)             # million km2 (extent: cells with >= 15 % ice)
        ext_s.append((cell * icy)[lat < 0].sum() / 1e6)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s2 = np.nanmean(s.reshape(360, 2, 720, 2), axis=(1, 3))     # 0.5-degree means (NaN = land)
        c2 = np.mean(np.nan_to_num(c).reshape(360, 2, 720, 2), axis=(1, 3))
        sst.append(np.where(np.isfinite(s2), np.round(s2 * 100), -9999).astype(np.int16))
        ice.append(np.round(c2 * 100).astype(np.uint8))
    return save("oisst_2025.npz", lat=np.arange(-89.75, 90, 0.5, dtype=np.float32),
                lon=np.arange(0.25, 360, 0.5, dtype=np.float32), date=np.array([str(d) for d in days]),
                sst_c100=np.array(sst), ice_pct=np.array(ice), oisst_extent_n=np.array(ext_n, np.float32),
                oisst_extent_s=np.array(ext_s, np.float32), extent_n=ice_n, extent_s=ice_s,
                extent_source="Sea Ice Index v4 (G02135), NSIDC / NOAA (Fetterer et al. 2017, updated daily)", source="NOAA OISST v2.1 (Huang et al. 2021), NOAA NCEI, public domain")


# ---------------------------------------------------------------- 6 WOA23
def woa() -> Path:
    """Annual-mean temperature 1991-2020 ("decav" = all decades) on 1 degree, surface to 2000 m."""
    import xarray as xr
    p = get(WOA)                                                    # 86 MB
    t = xr.open_dataset(p, decode_times=False)["t_an"].isel(time=0).sel(depth=slice(0, 2000)).load()
    t = t.isel(depth=slice(None, None, 2))                       # every second standard level
    v = t.values
    return save("woa23_temperature.npz", lat=t.lat.values.astype(np.float32), lon=t.lon.values.astype(np.float32),
                depth_m=t.depth.values.astype(np.float32),
                t_c100=np.where(np.isfinite(v), np.round(v * 100), -9999).astype(np.int16),
                source="World Ocean Atlas 2023 (Locarnini et al. 2024), NOAA NCEI, public domain")


# ---------------------------------------------------------------- 7 Sentinel-2 (satellite route)
S2_STAC = "https://earth-search.aws.element84.com/v1"          # Element 84 Earth Search: public, no account
S2_AOI = (89.55, 24.47, 89.90, 24.95)                           # the Jamuna (Brahmaputra) in Bangladesh: W, S, E, N
S2_YEARS = tuple(range(2019, 2027))                             # one dry-season scene per year
S2_RES = 40.0                                                   # metres per pixel of the reduced product


def s2_pick(year: int, aoi=S2_AOI, tile: str = "MGRS-45RYH", max_cloud: float = 5.0, n: int = 1):
    """The ``n`` least cloudy Sentinel-2 L2A scenes of the dry season (15 Jan - 5 Mar) of ``year`` on one MGRS tile
    (one item if ``n`` is 1, else a list)."""
    import pystac_client
    cat = pystac_client.Client.open(S2_STAC)
    items = [i for i in cat.search(collections=["sentinel-2-l2a"], bbox=list(aoi), datetime=f"{year}-01-15/{year}-03-05",
                                   query={"eo:cloud_cover": {"lt": max_cloud}}).items()
             if i.properties.get("grid:code") == tile]
    if not items:
        raise RuntimeError(f"no scene with < {max_cloud} % cloud for {year}")
    items = sorted(items, key=lambda i: (i.properties["eo:cloud_cover"], i.id))
    return items[0] if n == 1 else items[:n]


def s2_dn_offset(bands: dict) -> int:
    """Digital-number offset to subtract before using reflectances, decided from the data.

    Since processing baseline 04.00 (January 2022) ESA adds 1000 to every L2A value. Earth Search removed that offset
    again in most items, but not in all, and its metadata flag (``earthsearch:boa_offset_applied``) is not a safe guide
    (checked 2026-11-18: the 2022 scene here still carries the offset; the 2025 scene has the flag false but no
    offset). The data are: without the offset the darkest half per cent of a scene with open water is far below
    1000 in every band; with it, nothing is below 1000. Ignore the offset and a normalised difference is squeezed
    towards 0.
    """
    dark = min(float(np.percentile(a[a > 0], 0.5)) for k, a in bands.items() if k != "scl" and (a > 0).any())
    return 1000 if dark >= 1000 else 0


def s2_window(item, bands=("green", "swir16", "scl"), aoi=S2_AOI, res: float = S2_RES) -> dict:
    """Read only the window of ``aoi`` from the scene's Cloud-Optimised GeoTIFFs, resampled to ``res`` metres.

    rasterio reads the matching overview level over HTTP (a few hundred kB per band), never the whole 100-km tile.
    Returns the bands as arrays (row 0 = north) and the window's UTM bounds.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    out = {}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
        for b in bands:
            with rasterio.open(item.assets[b].href) as src:
                w, s, e, n = transform_bounds("EPSG:4326", src.crs, *aoi, densify_pts=21)
                w, s, e, n = (round(v / res) * res for v in (w, s, e, n))          # snap so every year has the same grid
                shape = (int(round((n - s) / res)), int(round((e - w) / res)))
                rs = Resampling.nearest if b == "scl" else Resampling.average
                a = src.read(1, window=from_bounds(w, s, e, n, src.transform), out_shape=shape, resampling=rs)
                out[b] = a
                bounds, crs = (w, s, e, n), str(src.crs)
    off = s2_dn_offset(out)
    for b in out:
        if b != "scl":
            out[b] = np.clip(out[b].astype(np.int32) - off, 0, None).astype(np.uint16)
    out.update(bounds=bounds, crs=crs, dn_offset=off)
    return out


def mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Modified Normalised Difference Water Index (Xu 2006): (green - SWIR) / (green + SWIR); water > 0."""
    g, s = green.astype(np.float32), swir.astype(np.float32)
    return (g - s) / np.maximum(g + s, 1)


S2_WATER = 0.10          # MNDWI above this is water
S2_MIN_KM2 = 5.0         # connected water smaller than this (ponds, flooded rice fields) is not "the river"


def river_mask(index: np.ndarray, res: float = S2_RES, threshold: float = S2_WATER, min_km2: float = S2_MIN_KM2) -> np.ndarray:
    """Water mask of the river: index > threshold, keeping only connected pieces of at least ``min_km2``."""
    from scipy import ndimage as ndi
    lab, _ = ndi.label(index > threshold)
    size = np.bincount(lab.ravel())
    keep = size >= min_km2 * 1e6 / res**2
    keep[0] = False
    return keep[lab]


def sentinel(n_scenes: int = 4) -> Path:
    """Keyless route: per year, the median MNDWI of the ``n_scenes`` clearest dry-season Sentinel-2 scenes, at 40 m.

    One scene is not enough: the river falls through the dry season and fields are flooded for rice, so single
    scenes a few weeks apart gave water areas from 96 to 248 km2. The median of several scenes is steadier.
    """
    import warnings
    ids, counts, idx = [], [], []
    rgb = None
    RAW.mkdir(parents=True, exist_ok=True)
    for y in S2_YEARS:
        items = s2_pick(y, n=n_scenes)
        cache = RAW / f"jamuna_mndwi_median_{y}_n{len(items)}.npy"      # so thresholds can change without new reads
        d = None
        if cache.exists():
            med = np.load(cache).astype(np.float32) / 100.0
        else:
            layers = []
            for it in items:
                d = s2_window(it)
                m = mndwi(d["green"], d["swir16"])
                m[np.isin(d["scl"], (0, 1, 3, 8, 9, 10))] = np.nan   # SCL: no data, saturated, shadow, cloud, cirrus
                layers.append(m)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = np.nan_to_num(np.nanmedian(np.array(layers), axis=0), nan=-1.0)
            np.save(cache, np.round(np.clip(med, -1, 1) * 100).astype(np.int8))
        idx.append(river_mask(med))
        last_index = np.round(np.clip(med, -1, 1) * 100).astype(np.int8)
        ids.append(",".join(i.id for i in items))
        counts.append(len(items))
        print(y, len(items), "scenes", [str(i.datetime.date()) for i in items], f"river {idx[-1].sum() * S2_RES**2 / 1e6:.0f} km2")
        if y == S2_YEARS[-1]:
            c = s2_window(items[0], bands=("red", "green", "blue"))
            rgb = np.dstack([c["red"], c["green"], c["blue"]])
            rgb = np.clip(rgb / 3000.0, 0, 1) ** 0.8                 # reflectance x 10000 -> a plain true-colour stretch
            rgb = (rgb * 255).astype(np.uint8)
            bounds, crs, rgb_date = c["bounds"], c["crs"], str(items[0].datetime.date())
    return save("jamuna_s2.npz", river=np.packbits(np.array(idx), axis=None), mndwi100_latest=last_index,
                shape=np.array(np.array(idx).shape), rgb_latest=rgb, rgb_date=rgb_date, bounds_utm=np.array(bounds), crs=crs,
                res_m=S2_RES, year=np.array(S2_YEARS), scenes=np.array(counts), scene_ids=np.array(ids),
                source="Contains modified Copernicus Sentinel data 2019-2026 (Sentinel-2 L2A, ESA), "
                       "read from Element 84 Earth Search (AWS open data)")


def sentinel_gee(project: str | None = None) -> Path:
    """Earth Engine route: the median of ALL clear dry-season scenes of each year (not one scene), MNDWI at 40 m.

    Needs an Earth Engine account: ``ee.Authenticate()`` once, then a Cloud project id (argument, or the environment
    variable ``EE_PROJECT``). Only a small array per year is downloaded.
    """
    import io
    import ee
    project = project or os.environ.get("EE_PROJECT")
    ee.Initialize(project=project) if project else ee.Initialize()
    ref = _data.load_npz("jamuna_s2")                      # same grid as the keyless product
    w, s, e, n = (float(v) for v in ref["bounds_utm"])
    region = ee.Geometry.Rectangle([w, s, e, n], proj=str(ref["crs"]), geodesic=False)
    out, count = [], []
    for y in S2_YEARS:
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
               .filterDate(f"{y}-01-15", f"{y}-03-06").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)))
        clear = col.map(lambda im: im.updateMask(im.select("SCL").remap([0, 1, 3, 8, 9, 10], [0] * 6, 1)))
        img = clear.median().normalizedDifference(["B3", "B11"]).multiply(100).round().toInt8().unmask(-128)
        url = img.getDownloadURL({"region": region, "crs": str(ref["crs"]), "scale": S2_RES, "format": "NPY"})
        raw = np.load(io.BytesIO(urllib.request.urlopen(url).read()))
        a = raw[raw.dtype.names[0]] if raw.dtype.names else raw
        last_index = a.astype(np.int8)
        out.append(river_mask(a / 100.0))
        count.append(int(col.size().getInfo()))
        print(y, "scenes", count[-1], "shape", a.shape, f"river {out[-1].sum() * S2_RES**2 / 1e6:.0f} km2")
    return save("jamuna_s2_gee.npz", river=np.packbits(np.array(out), axis=None), shape=np.array(np.array(out).shape),
                mndwi100_latest=last_index, year=np.array(S2_YEARS), scenes=np.array(count),
                bounds_utm=ref["bounds_utm"], crs=ref["crs"], res_m=S2_RES,
                source="Contains modified Copernicus Sentinel data 2019-2026 (Sentinel-2 L2A harmonised, ESA), "
                       "median composites processed with Google Earth Engine")


BUILDERS = {"cyclones": cyclones, "ciaran": ciaran, "terrain": terrain, "tcwv": tcwv, "oisst": oisst, "woa": woa,
            "sentinel": sentinel, "sentinel_gee": sentinel_gee}

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for k in ([b for b in BUILDERS if b != "sentinel_gee"] if which == ["all"] else which):
        BUILDERS[k]()
