"""Lens map: the city poster, its video and its GIF for ANY place, in one call.

    from gismaps.lens import make_city
    make_city("Chittagong")                                   # a place name (geocoded with Nominatim)
    make_city(lat=-1.286, lon=36.817, outputs=("poster",))    # a point (the city and country are looked up)
    make_city(preset="brisbane")                              # a hand-tuned city, from the data shipped with the repository

    python -m gismaps.lens city --place "Chittagong" --outputs poster,video,gif

What happens for a place that is not one of the hand-tuned cities (``gismaps.lens.cities.CITIES``):
  1. the name becomes a centre point (Nominatim search), or the point becomes a name (Nominatim reverse);
  2. ``cities.place_config`` works out the map scale, the three lenses and the download boxes from that centre;
  3. ``osm.fetch_city`` downloads the OpenStreetMap layers from a public Overpass server, one polite query at a time;
     every answer is kept in ``<out_dir>/../lens_cache/``, so a stopped run resumes and a second run downloads nothing;
  4. the legend rows are chosen from the point kinds the place really holds, and the poster, the video and the GIF
     are drawn by the same functions that draw the hand-tuned cities.
FAST (the default) maps a smaller piece of ground and writes a small video, so that a large dense city fits a free
Google Colab runtime; ``fast=False`` (``--full``) is the full size.
Data (c) OpenStreetMap contributors (ODbL). Style after Fan Mo, #30DayMapChallenge 2026 Day 14.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from . import cities, osm

OUTPUTS = ("poster", "video", "gif")
VIDEO = {True: (0.36, 15), False: (0.72, 30)}  # fast -> (video scale, frames per second)
# other spellings of the hand-tuned cities (their keys and names always match)
ALIASES = {"richmond": ("richmond (virginia)", "richmond, virginia", "richmond virginia", "richmond, va", "richmond va")}


def preset_key(place) -> str | None:
    """The key of the hand-tuned city a name means ("Dhaka", "brisbane", "Richmond (Virginia)"), else ``None``."""
    t = " ".join(str(place or "").lower().split())
    for k, c in cities.CITIES.items():
        if t in (k, c["name"].lower(), f"{c['name']}, {c['country']}".lower(), *ALIASES.get(k, ())):
            return k
    return None


def parse_lat_lon(text) -> tuple[float, float] | None:
    """ "23.72, 90.40" (comma, semicolon or space between the two numbers) -> (23.72, 90.40); anything else -> None."""
    m = re.fullmatch(r"\s*\(?\s*(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)\s*\)?\s*", str(text or ""))
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    return (lat, lon) if abs(lat) <= 90 and abs(lon) <= 180 else None


def lat_lon_text(lat: float, lon: float) -> str:
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"


def _tidy(place: str) -> str:
    """The name printed on the poster for a typed place: the part before the first comma, capitalised if it was
    typed in lower case ("chittagong, bangladesh" -> "Chittagong")."""
    t = " ".join(str(place).split(",")[0].split())
    return t.title() if t == t.lower() else t


def _preset(key: str) -> dict:
    c = cities.CITIES[key]
    return dict(name=c["name"], country=c["country"], lat=c["lat0"], lon=c["lon0"], preset=key, notes=[])


def locate(place=None, lat=None, lon=None, preset=None, cache: Path | None = None) -> dict:
    """Work out what place is meant: dict(name, country, lat, lon, preset, notes). ``preset`` is the key of a
    hand-tuned city or ``None``.

    Order: ``preset=`` wins; a point (``lat`` and ``lon``, or a "lat, lon" text as ``place``) is used as given and
    its city and country are looked up (a hand-tuned city is still recognised when ``place`` names it and the point is
    within 0.05 degrees of its centre); a name that means a hand-tuned city gives that city; any other name is geocoded.
    """
    if preset:
        key = preset_key(preset)
        if not key:
            raise ValueError(f"unknown preset {preset!r}; known: {', '.join(cities.CITIES)}")
        return _preset(key)
    if lat is None and lon is None and parse_lat_lon(place):
        (lat, lon), place = parse_lat_lon(place), None
    if (lat is None) != (lon is None):
        raise ValueError("give both lat and lon (or neither)")
    place = str(place).strip() if place else ""
    key = preset_key(place) if place else None
    if lat is None:
        if not place:
            raise ValueError("give a place name, a point (lat and lon) or preset=")
        if key:
            return _preset(key)
        try:
            hit = osm.lookup(place, cache)
        except LookupError as err:
            raise ValueError(f"{err}. Check the spelling, add the country, or give lat and lon.") from err
        except OSError as err:
            raise RuntimeError(f"the place-name service (Nominatim) did not answer ({err}). Try again in a minute, "
                               "or give lat and lon.") from err
        name = _tidy(place)
        country = hit["country"] if hit["country"].lower() != name.lower() else ""
        return dict(name=name, country=country, lat=round(hit["lat"], 5), lon=round(hit["lon"], 5), preset=None,
                    notes=[f"Nominatim: {hit['display_name']}"])
    lat, lon = float(lat), float(lon)
    if abs(lat) > 90 or abs(lon) > 180:
        raise ValueError(f"lat must be within -90..90 and lon within -180..180 (got {lat}, {lon})")
    if key and abs(lat - cities.CITIES[key]["lat0"]) <= 0.05 and abs(lon - cities.CITIES[key]["lon0"]) <= 0.05:
        return _preset(key)
    notes = []
    try:
        hit = osm.reverse(lat, lon, cache)
        name, country = (_tidy(place) if place else hit["name"]), hit["country"]
        notes.append(f"Nominatim: {hit['display_name']}")
    except Exception as err:  # noqa: BLE001  (no named place here, or the service did not answer: carry on)
        name, country = (_tidy(place) if place else lat_lon_text(lat, lon)), ""
        notes.append(f"no place name from Nominatim ({type(err).__name__}: {err}); the poster shows "
                     f"{'the name as typed' if place else 'the coordinates'}")
    if country.lower() == name.lower():
        country = ""
    return dict(name=name, country=country, lat=round(lat, 5), lon=round(lon, 5), preset=None, notes=notes)


def _outputs(outputs) -> tuple[str, ...]:
    want = [o.strip().lower() for o in (outputs.split(",") if isinstance(outputs, str) else outputs) if str(o).strip()]
    bad = [o for o in want if o not in OUTPUTS]
    if bad or not want:
        raise ValueError(f"outputs must be one or more of {', '.join(OUTPUTS)} (got {outputs!r})")
    return tuple(o for o in OUTPUTS if o in want)


def make_city(place=None, lat=None, lon=None, outputs=("poster", "video", "gif"), radius_km=None, fast=True,
              out_dir="out", preset=None, log=print) -> dict:
    """Make the lens-map poster, video and/or GIF of any place.

    Parameters
    ----------
    place      a place name ("Chittagong", "Nairobi, Kenya"). "brisbane", "dhaka" and "richmond" are the hand-tuned
               cities: they are drawn from the reduced data shipped with the repository, with no download
    lat, lon   a centre point instead of (or together with) a name; the city and country come from a reverse lookup
    outputs    any of "poster", "video", "gif" (a tuple, or one text such as "poster,gif")
    radius_km  half the ground width of the sheet in km (default 3.3; FAST: 2.0). Ignored for a hand-tuned city
    fast       True: smaller ground, fewer building tiles, video 720 x 1018 px at 15 frames a second;
               False: the full sheet, video 1440 x 2036 px at 30 frames a second
    out_dir    where ``lens_<key>.png``, ``lens_<key>_anim.mp4`` and ``lens_<key>_anim.gif`` are written; the downloads
               are cached next to it in ``<out_dir>/../lens_cache/``
    preset     the key of a hand-tuned city (same as naming it in ``place``)
    log        where progress lines go (default ``print``)

    Returns dict(key, name, country, lat, lon, preset (bool), poster, video, gif (paths or None), counts, notes).
    """
    from . import anim, city  # (matplotlib and shapely load here, not when this module is imported)

    def say(*a, **_k):
        log(" ".join(str(x) for x in a))

    t_start = time.time()
    want = _outputs(outputs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir.resolve().parent / "lens_cache"
    where = locate(place, lat, lon, preset, cache)
    notes = list(where["notes"])
    say(f"{where['name']}{', ' + where['country'] if where['country'] else ''} ({where['lat']:.4f}, {where['lon']:.4f})"
        + (" · hand-tuned city, shipped data" if where["preset"] else f" · {'FAST' if fast else 'FULL'}"))

    if where["preset"]:
        cfg = cities.city(where["preset"])
        pr = city.Proj(cfg)
        L = city.read_derived(cfg, pr)
        if radius_km:
            notes.append("radius_km is ignored for a hand-tuned city")
    else:
        if radius_km is not None and not 0.3 <= float(radius_km) <= 12:
            raise ValueError(f"radius_km must be between 0.3 and 12 (got {radius_km})")
        cfg = cities.place_config(where["name"], where["country"], where["lat"], where["lon"], radius_km, fast)
        pr = city.Proj(cfg)
        raw = cache / f"OSM-overpass-{cfg['key']}" / f"{cfg['lat0']:.4f}_{cfg['lon0']:.4f}_r{cfg['mpp']:g}_t{cfg['tiles']}"
        say(f"sheet {2 * cfg['mpp']:.1f} km x {2.828 * cfg['mpp']:.1f} km ({cfg['mpp']:g} m per pixel); "
            f"OpenStreetMap layers -> {raw}")
        t0 = time.time()
        fresh = not (raw / f"{cfg['key']}_manifest.json").exists()
        try:
            manifest = osm.fetch_city(cfg, raw, pause=5.0, attempts=8, log=say, timeout=240)
        except RuntimeError as err:
            raise RuntimeError(f"{err}. The public Overpass servers are busy: run the same call again in a few "
                               f"minutes; the layers already downloaded are kept in {raw}.") from err
        mb = json.loads(manifest.read_text(encoding="utf-8"))["total_bytes"] / 1e6
        notes.append(f"OpenStreetMap extract: {mb:.1f} MB " + (f"downloaded in {time.time() - t0:.0f} s" if fresh else "(from the cache)"))
        say(notes[-1])
        cfg["extract"] = time.strftime("%B %Y", time.localtime(manifest.stat().st_mtime))
        L = city.read_raw(cfg, pr, raw)
        first = city.count(cfg, L)
        cities.choose_public(cfg, first["points_in_public_lens"], first["points_detail_box"])
        if not any(first["landuse"].values()):
            cfg["use_lines"] = ("no OSM land-use", "polygons here")
            notes.append("no OSM land-use polygons here")
        if not first["buildings"]:
            cfg["mor_lines"] = ("no OSM buildings here",) if first["roads"] else ("no OSM buildings", "or streets here")
            notes.append("no OSM buildings here" if first["roads"] else "no OSM buildings or streets here")
        if not any(first["points_detail_box"].values()):
            notes.append("no public amenity points (toilets, taps, benches ...) in OSM here")
        if not (first["water"] or first["roads"] or first["buildings"] or any(first["landuse"].values())):
            notes.append("OSM holds nothing here (open sea or empty land): the lenses are blank")

    counts = city.count(cfg, L)
    key = cfg["key"]
    res = dict(key=key, name=cfg["name"], country=cfg["country"], lat=cfg["lat0"], lon=cfg["lon0"],
               preset=bool(where["preset"]), poster=None, video=None, gif=None, counts=counts, notes=notes)
    layers = city.render_layers(cfg, L)  # the four maps, drawn once for the poster and for the video
    if "poster" in want:
        res["poster"] = out_dir / f"lens_{key}.png"
        city.poster(cfg, L, res["poster"], layers=layers)
    if "video" in want or "gif" in want:
        scale, fps = VIDEO[bool(fast)]
        mp4 = out_dir / (f"lens_{key}_anim.mp4" if "video" in want else f"lens_{key}_anim.tmp.mp4")
        anim.animate_city(cfg, L, mp4, scale, fps, log=say, layers=layers)
        if "gif" in want:
            res["gif"] = anim.to_gif(mp4, out_dir / f"lens_{key}_anim.gif")
            say(f"gif -> {res['gif']}  ({res['gif'].stat().st_size / 1e6:.2f} MB)")
        if "video" in want:
            res["video"] = mp4
        else:
            mp4.unlink(missing_ok=True)
    for n in notes:
        say("note:", n)
    say(f"done in {time.time() - t_start:.0f} s")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m gismaps.lens", description="Lens maps of any place")
    sub = ap.add_subparsers(dest="what", required=True)
    c = sub.add_parser("city", help="the city poster, video and GIF of any place",
                       description="Give ONE of --place, --lat with --lon, or --preset.")
    c.add_argument("--place", default=None, help='a place name, for example "Dhaka" or "Nairobi, Kenya"')
    c.add_argument("--lat", type=float, default=None)
    c.add_argument("--lon", type=float, default=None)
    c.add_argument("--preset", default=None, choices=sorted(cities.CITIES), help="a hand-tuned city (no download)")
    c.add_argument("--outputs", default="poster,video,gif", help="any of poster,video,gif (default: all three)")
    c.add_argument("--radius-km", type=float, default=None, help="half the ground width of the sheet (default 3.3; FAST 2.0)")
    c.add_argument("--full", action="store_true", help="full size (default: FAST, a smaller sheet and a small video)")
    c.add_argument("--out", default="out", help="output folder (default ./out)")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # place names can hold letters a Windows console cannot print
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    if not (a.place or a.preset or (a.lat is not None and a.lon is not None)):
        c.error("give --place, --lat with --lon, or --preset")
    try:
        res = make_city(a.place, a.lat, a.lon, a.outputs, a.radius_km, not a.full, a.out, a.preset,
                        log=lambda *x: print(*x, flush=True))
    except (ValueError, RuntimeError) as err:
        raise SystemExit(f"STOP - {err}") from err
    for k in OUTPUTS:
        if res[k]:
            print(f"{k:6s} -> {res[k]}")


if __name__ == "__main__":
    main()
