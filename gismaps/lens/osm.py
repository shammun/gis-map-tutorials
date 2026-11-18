"""Lens map: OpenStreetMap extracts of one city through the public Overpass API (no key, no account).

Pure standard library + curl (or ``urllib`` where curl is missing), so it runs the same on a laptop and on Google
Colab. The tutorials in ``tutorials/lens_maps/`` call :func:`fetch_city` directly.

Polite use of shared, free servers:
  * one query at a time, a pause between queries (default 8 s), a User-Agent that says what the client is;
  * every answer is cached as a file; a file already on disk is never fetched again, so a stopped run resumes;
  * buildings are asked for in n x n tiles (one query for a whole 7-km box times out);
  * on HTTP 429 / 502 / 504 or a timeout the query goes to the next public server after a growing pause
    (5 s, 10 s, 20 s ... up to 90 s);
  * place names come from Nominatim: at most one request per second, every answer cached in a small file.
Data (c) OpenStreetMap contributors, ODbL 1.0. Print that credit on every map made from it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

# public Overpass servers, tried in turn: the main one answers HTTP 429 / 502 / 504 under load and recovers after a
# pause; the others are independent mirrors. The server that answered last is asked first the next time.
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.openstreetmap.fr/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter", "https://overpass.private.coffee/api/interpreter"]
NOMINATIM = "https://nominatim.openstreetmap.org"
USER_AGENT = "gismaps-tutorial/1.0"
_first = [0]  # index in OVERPASS of the server that answered last
_health: dict[str, tuple[bool, float]] = {}  # server -> (did it answer the tiny test query, when it was asked)
_last_lookup = [0.0]  # time of the last Nominatim request (at most one per second)

_RE_LEISURE = "park|pitch|garden|golf_course|sports_centre|stadium|playground|recreation_ground|nature_reserve|track"
_RE_AMEN_POLY = "school|university|college|hospital|clinic|place_of_worship|parking|kindergarten"
_RE_ROAD = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|pedestrian|service|busway"


def _bb(b: tuple[float, float, float, float]) -> str:
    """An Overpass bounding box: (south, west, north, east)."""
    return f"({b[0]},{b[1]},{b[2]},{b[3]})"


def _tiles(b, n):
    """Cut the box b into n x n equal tiles: yields (index, (south, west, north, east))."""
    for i in range(n):
        for j in range(n):
            yield (i * n + j, (b[0] + (b[2] - b[0]) * i / n, b[1] + (b[3] - b[1]) * j / n,
                               b[0] + (b[2] - b[0]) * (i + 1) / n, b[1] + (b[3] - b[1]) * (j + 1) / n))


def _buildings_ql(box) -> str:
    return f'[out:json][timeout:170];(way["building"]{_bb(box)};relation["building"]{_bb(box)};);out geom;'


def queries(cfg: dict) -> dict[str, tuple[str, int]]:
    """name -> (Overpass QL, expected bytes for the disk guard) for one city configuration."""
    detail, context = cfg["detail"], cfg["context"]
    q: dict[str, tuple[str, int]] = {
        "water": (f'[out:json][timeout:180];(way["natural"="water"]{_bb(context)};relation["natural"="water"]{_bb(context)};'
                  f'way["waterway"="riverbank"]{_bb(context)};relation["waterway"="riverbank"]{_bb(context)};'
                  f'way["natural"="coastline"]{_bb(context)};);out geom;', 30_000_000),
        "landuse": (f'[out:json][timeout:180];(way["landuse"]{_bb(detail)};relation["landuse"]{_bb(detail)};'
                    f'way["leisure"~"^({_RE_LEISURE})$"]{_bb(detail)};relation["leisure"~"^({_RE_LEISURE})$"]{_bb(detail)};'
                    f'way["amenity"~"^({_RE_AMEN_POLY})$"]{_bb(detail)};relation["amenity"~"^({_RE_AMEN_POLY})$"]{_bb(detail)};'
                    f'way["natural"~"^(wood|scrub|grassland)$"]{_bb(detail)};);out geom;', 30_000_000),
        "roads": (f'[out:json][timeout:180];(way["highway"~"^({_RE_ROAD})(_link)?$"]{_bb(detail)};'
                  f'way["railway"~"^(rail|light_rail)$"]{_bb(detail)};);out geom;', 40_000_000),
        "roads_context": (f'[out:json][timeout:180];way["highway"~"^(motorway|trunk|primary)$"]{_bb(context)};out geom;', 20_000_000),
    }
    a = cfg["amenity_re"]
    a_way = "|".join(v for v in a.split("|") if v not in ("bench", "bbq"))  # benches and barbecues are nodes
    q["amenities"] = (f'[out:json][timeout:120];(node["amenity"~"^({a})$"]{_bb(detail)};'
                      f'way["amenity"~"^({a_way})$"]{_bb(detail)};'
                      f'node["leisure"="playground"]{_bb(detail)};);out center;', 5_000_000)
    # buildings in n x n tiles (one query for the whole box times out on the public server); a way on a tile edge comes
    # back in both tiles and is de-duplicated by OSM id in the builder
    for k, t in _tiles(detail, cfg["tiles"]):
        q[f"buildings_{k}"] = (_buildings_ql(t), 25_000_000)
    return q


def _post(ql: str, path: Path, url: str, timeout: int) -> str:
    """POST one query, write the answer to ``path``; returns "200" or a short description of what went wrong."""
    if shutil.which("curl"):  # python-requests hung on the larger answers on 2026-11-17; curl did not
        r = subprocess.run(["curl", "-s", "--connect-timeout", "20", "-m", str(timeout), "-A", USER_AGENT, "-o", str(path),
                            "-w", "%{http_code}", "--data-urlencode", f"data={ql}", url], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else f"curl rc {r.returncode}, HTTP {r.stdout.strip()}"
    try:  # no curl on this machine: the standard library does the same POST
        req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": ql}).encode(), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            path.write_bytes(resp.read())
            return str(resp.status)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def _answers(url: str, path: Path, log=print) -> bool:
    """Does this server answer at all? One tiny query (a single node, 20 s at most) the first time a server is used;
    a "yes" is kept for the session and a "no" for three minutes, so a mirror that accepts connections but never
    replies costs 20 s now and then instead of a full query timeout every time."""
    ok, when = _health.get(url, (False, -1e9))
    if not ok and time.time() - when > 180:  # a "no" is asked again after three minutes
        probe = path.with_name(path.name + ".probe")
        code = _post("[out:json][timeout:10];node(1);out;", probe, url, 20)
        ok = code == "200" and probe.exists() and probe.read_bytes()[:1] == b"{"
        probe.unlink(missing_ok=True)
        _health[url] = (ok, time.time())
        if not ok:
            log(f"  {url.split('/')[2]} does not answer now ({code if code.startswith('curl') else 'HTTP ' + code}); skipped")
    return ok


def overpass(ql: str, path: Path, attempts: int = 8, timeout: int = 300, log=print) -> None:
    """Run one Overpass query and save the JSON answer to ``path``. A failure (HTTP 429 / 502 / 504, a timeout, a
    broken answer) moves on to the next server of ``OVERPASS`` that answers a tiny test query, after a growing pause
    (5 s, 10 s, 20 s ... up to 90 s); raises after ``attempts`` failures. A server-side timeout comes back as HTTP 200
    with a "remark" and counts as a failure."""
    last = tried = None
    for attempt in range(attempts):
        turn = [OVERPASS[(_first[0] + k) % len(OVERPASS)] for k in range(len(OVERPASS))]  # last good server first
        if tried in turn:  # after a failure: start with the server after the one that failed
            turn = turn[turn.index(tried) + 1:] + turn[:turn.index(tried) + 1]
        url = tried = next((u for u in turn if _answers(u, path, log)), None)  # (stops at the first that answers)
        if url is None:  # nobody answered the test query: forget the verdicts, wait, test again
            _health.clear()
            last = "no Overpass server answered the test query"
            if attempt + 1 < attempts:
                log("  retry:", last)
                time.sleep(min(5 * 2 ** attempt, 90))
            continue
        code = _post(ql, path, url, timeout)
        if code == "200" and path.exists() and path.read_bytes()[:1] == b"{":
            tail = path.read_bytes()[-400:]
            if b'"remark"' not in tail:
                _first[0] = OVERPASS.index(url)
                return
            last = "Overpass remark: " + tail.decode("utf-8", "replace").strip()[-200:]
        else:
            last = f"{code if code.startswith('curl') else 'HTTP ' + code} ({url.split('/')[2]})"
            if not code.isdigit():  # no HTTP answer at all (a timeout, a refused connection): leave it out for a while
                _health[url] = (False, time.time())
        path.unlink(missing_ok=True)
        if attempt + 1 < attempts:
            log("  retry:", last)
            time.sleep(min(5 * 2 ** attempt, 90))
    raise RuntimeError(f"Overpass failed: {last}")


def fetch_city(cfg: dict, folder: Path, pause: float = 8.0, attempts: int = 8, log=print, timeout: int = 300) -> Path:
    """All layers of one city, one query at a time with a pause, then a manifest. Returns the manifest path.

    ``folder`` gets ``<key>_<layer>.json`` per layer and ``<key>_manifest.json`` (bytes, sha256, fetch time and the
    exact Overpass QL of every layer). A layer file already in the folder is kept, so an interrupted run resumes.
    A building tile that no server can answer (too many buildings for one query) is cut into 2 x 2 smaller tiles
    (``<key>_buildings_<k>_<j>.json``) and asked again."""
    c = cfg["key"]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    layers = {}

    def get(name: str, ql: str) -> None:
        f = folder / f"{c}_{name}.json"
        if not f.exists():
            t0 = time.time()
            part = f.with_name(f.name + ".part")
            overpass(ql, part, attempts=attempts, timeout=timeout, log=log)
            part.replace(f)
            log(f"  {name}: {f.stat().st_size / 1e6:.2f} MB in {time.time() - t0:.0f} s")
            time.sleep(pause)
        layers[name] = {"file": f.name, "bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                        "fetched": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(f.stat().st_mtime)), "query": ql}

    boxes = {f"buildings_{k}": t for k, t in _tiles(cfg["detail"], cfg["tiles"])}
    for name, (ql, _n) in queries(cfg).items():
        if not (name in boxes and any(folder.glob(f"{c}_{name}_*.json"))):  # (an earlier run already cut this tile)
            try:
                get(name, ql)
                continue
            except RuntimeError:
                if name not in boxes:
                    raise
                log(f"  {name}: no answer for the whole tile, asking for it in 2 x 2 smaller tiles")
        for j, t in _tiles(boxes[name], 2):
            get(f"{name}_{j}", _buildings_ql(t))
    manifest = folder / f"{c}_manifest.json"
    manifest.write_text(json.dumps({
        "city": c, "servers_tried_in_order": OVERPASS, "licence": "(c) OpenStreetMap contributors, ODbL 1.0",
        "detail_bbox_swne": cfg["detail"], "context_bbox_swne": cfg["context"], "building_tiles": cfg["tiles"],
        "total_bytes": sum(v["bytes"] for v in layers.values()), "layers": layers}, indent=1), encoding="utf-8")
    return manifest


# ------------------------------------------------------------------------------------------------ place names
_NAME_KEYS = ("city", "town", "municipality", "village", "hamlet", "city_district", "county", "state_district", "state",
              "region", "island", "archipelago")


def _nominatim(endpoint: str, params: dict, cache: Path | None = None, timeout: int = 20):
    """One Nominatim request (public, no key): at most one per second, a User-Agent that names the client, and a
    small cache file (``<cache>/nominatim.json``, default ``./lens_cache``) so a place is looked up only once."""
    params = {**params, "format": "jsonv2", "addressdetails": 1, "accept-language": "en"}
    url = f"{NOMINATIM}/{endpoint}?" + urllib.parse.urlencode(params)
    f = Path(cache if cache else "lens_cache") / "nominatim.json"
    try:
        known = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    if url in known:
        return known[url]
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):  # a busy moment (HTTP 503, a timeout) is asked again after 3 s, then 6 s
        time.sleep(max(0.0, 1.1 + 3 * attempt - (time.time() - _last_lookup[0])))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                hit = json.loads(resp.read().decode("utf-8"))
            break
        except OSError:  # (urllib's HTTP and network errors are all OSError)
            if attempt == 2:
                raise
        finally:
            _last_lookup[0] = time.time()
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        known[url] = hit
        f.write_text(json.dumps(known, ensure_ascii=False, indent=0), encoding="utf-8")
    except OSError:
        pass  # a folder that cannot be written only means no cache
    return hit


def lookup(place: str, cache: Path | None = None, timeout: int = 20) -> dict:
    """A place name -> dict(lat, lon, name, country, display_name) from OSM's Nominatim search."""
    hit = _nominatim("search", {"q": place, "limit": 1}, cache, timeout)
    if not hit:
        raise LookupError(f"Nominatim found nothing for {place!r}")
    h = hit[0]
    ad = h.get("address", {})
    name = h.get("name") or next((ad[k] for k in _NAME_KEYS if ad.get(k)), place)
    return dict(lat=float(h["lat"]), lon=float(h["lon"]), name=name, country=ad.get("country", ""),
                display_name=h.get("display_name", name))


def reverse(lat: float, lon: float, cache: Path | None = None, zoom: int = 10, timeout: int = 20) -> dict:
    """A point -> dict(name, country, display_name) of the city (outside cities: the district or region) it lies in,
    from OSM's Nominatim reverse search. Raises LookupError where OSM has no named place (open ocean)."""
    h = _nominatim("reverse", {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}", "zoom": zoom}, cache, timeout)
    ad = h.get("address") if isinstance(h, dict) else None
    if not ad:
        raise LookupError(f"Nominatim has no named place at {lat:.4f}, {lon:.4f}")
    name = next((ad[k] for k in _NAME_KEYS if ad.get(k)), None) or h.get("name") or ad.get("country", "")
    return dict(name=name, country=ad.get("country", ""), display_name=h.get("display_name", name))


def geocode(place: str, timeout: int = 20, cache: Path | None = None) -> tuple[float, float, str]:
    """(lat, lon, display name) of a place name from OSM's Nominatim search (public, no key).
    Usage policy: at most one request per second and a User-Agent; answers are cached (see :func:`lookup`)."""
    h = lookup(place, cache, timeout)
    return h["lat"], h["lon"], h["display_name"]
