"""Lens map: per-city configuration of the "One city, many realities" poster.

One entry per city; ``gismaps.lens.osm.fetch_city`` downloads it and ``python -m gismaps.lens.city --city <key>`` draws it.
A new city needs only a new entry here; ``city_config`` builds one from a centre point, and ``place_config`` builds
the automatic one that ``gismaps.lens.make_city`` uses for any other place.

Fields
------
name, country   printed in the header and the credit block (usual English spelling)
lat0, lon0      the point that sits at poster pixel (cx, cy) = (985, 1420); the local equirectangular map is built on it
mpp             metres per poster pixel (the poster is 2000 x 2828 px)
detail          (south, west, north, east) box that holds the three lenses: land use, roads, buildings, amenities
context         (south, west, north, east) box of the muted water and main roads outside the lenses (>= the poster)
tiles           buildings are fetched in tiles x tiles boxes (dense cities need more, smaller tiles)
lenses          poster-pixel offsets (dx, dy) of the three lens centres from (cx, cy): land use, public realm, morphology
radius          lens radius in poster pixels
water_kinds     values of the OSM ``water=`` tag drawn as water (None = untagged natural=water)
water_min_m2    smallest water polygon drawn
amenity_re      amenity values fetched as points (nodes, and ways as centres)
public          the point symbols of the PUBLIC REALM lens, in legend order: (OSM value, legend label)
public_lines    the two italic description lines of the PUBLIC REALM label block
labels          label-block anchors (x, y) and leaders (from, lens-relative end as fractions of the radius)
count_area      "detail" = legend counts every mapped point of the detail box (the first Brisbane study);
                "lens" = only the points inside the PUBLIC REALM lens (what the reader can see and count)
legend_xy       top-left of the legend (default (1560, 1790)); raise it when ``public`` has more than four rows
north_xy        foot of the north arrow (default (1050, 2440)); the scale bar sits 100 px to its right
playground_areas  True = playgrounds mapped as areas (leisure=playground polygons) are added to the playground
                points at their centroids (OSM maps most playgrounds as areas; the Brisbane study counted nodes only)
legend_dy       row pitch of the legend in px (default 37)
extra_landuse   {OSM landuse value: (class key, legend label, colour)}: classes this city needs beyond the common eleven
scale_m         length of the scale bar in metres
author          creator line of the credit block
sea             True = the sea is built from the OSM coastlines (any place; the hand-tuned cities use water polygons)
use_lines, mor_lines  description lines of the LAND USE / CITY MORPHOLOGY label blocks (default: the usual ones)
extract         month and year of the Overpass extract, printed in the credit block
"""
from __future__ import annotations

from .common import AUTHOR  # "Map created by ..." (one place for every poster)

_AMEN_WIDE = "toilets|drinking_water|bench|fountain|shelter|place_of_worship|marketplace|library|community_centre|bus_station"

# label blocks of the Brisbane study; a city overrides only what it must
LABELS = {
    "use": {"xy": (84, 560), "from": (330, 640), "to": (-0.5, -0.6)},
    "pub": {"xy": (1500, 500), "from": (1560, 700), "to": (0.25, -0.72)},
    "mor": {"xy": (84, 2080), "from": (400, 2060), "to": (-0.6, 0.55)},
}

CITIES: dict[str, dict] = {
    "brisbane": dict(
        name="Brisbane", country="Australia", lat0=-27.4705, lon0=153.0235, mpp=3.3,
        detail=(-27.504, 152.985, -27.436, 153.063), context=(-27.57, 152.92, -27.37, 153.13), tiles=3,
        lenses=((-370, -250), (395, -215), (10, 330)), radius=530.0,
        water_kinds=("river", "bay", "harbour", None), water_min_m2=6000.0,
        amenity_re="toilets|drinking_water|bench|bbq|fountain|shelter",
        public=(("toilets", "Public toilet"), ("drinking_water", "Drinking water"), ("bench", "Bench"),
                ("playground", "Playground")),
        public_lines=("Parks, toilets, taps,", "benches, playgrounds"),
        labels=LABELS, scale_m=1000, author=AUTHOR,
    ),
    # Dhaka: Dhaka University / New Market / Dhanmondi (land use), Ramna - Motijheel - Kamalapur (public realm),
    # Old Dhaka on the Buriganga with Sadarghat and Keraniganj across the river (morphology)
    "dhaka": dict(
        name="Dhaka", country="Bangladesh", lat0=23.7233, lon0=90.4020, mpp=3.6,
        detail=(23.690, 90.366, 23.752, 90.438), context=(23.670, 90.345, 23.775, 90.460), tiles=4,
        lenses=((-370, -250), (395, -215), (10, 330)), radius=530.0,
        water_kinds=("river", "lake", "pond", "canal", "reservoir", "basin", None), water_min_m2=3000.0,
        amenity_re=_AMEN_WIDE,
        public=(("place_of_worship", "Place of worship"), ("marketplace", "Marketplace"), ("toilets", "Public toilet"),
                ("drinking_water", "Drinking water"), ("playground", "Playground"), ("bench", "Bench")),
        public_lines=("Parks, mosques, markets,", "toilets, taps, playgrounds"),
        count_area="lens", playground_areas=True, legend_xy=(1560, 1768), legend_dy=34,
        extra_landuse={"military": ("mil", "Military & border guard", "#c4ad7c")},  # Pilkhana (BGB headquarters)
        labels=LABELS, scale_m=1000, author=AUTHOR,
    ),
    # Richmond, Virginia. The lens roles are swapped left-right against Brisbane, because OSM's public amenities here
    # cluster in the west: PUBLIC REALM (left) = Byrd Park, Hollywood Cemetery, Oregon Hill, Monroe Park / VCU and the
    # Fan; LAND USE (right) = downtown, Capitol Square, Jackson Ward, Shockoe and Church Hill; MORPHOLOGY (bottom) =
    # the James River with Belle Isle, and Manchester on the south bank
    "richmond": dict(
        name="Richmond", country="Virginia, USA", lat0=37.5350, lon0=-77.4460, mpp=3.3,
        detail=(37.508, -77.486, 37.566, -77.406), context=(37.490, -77.510, 37.585, -77.385), tiles=3,
        lenses=((395, -215), (-370, -250), (10, 330)), radius=530.0,  # land use RIGHT, public realm LEFT
        water_kinds=("river", "canal", "lake", "reservoir", "pond", "basin", None), water_min_m2=4000.0,
        amenity_re=_AMEN_WIDE,
        public=(("toilets", "Public toilet"), ("drinking_water", "Drinking water"), ("bench", "Bench"),
                ("playground", "Playground")),
        public_lines=("Parks, toilets, taps,", "benches, playgrounds"),
        count_area="lens", playground_areas=True,
        labels={"use": {"xy": (1500, 500), "from": (1560, 700), "to": (0.25, -0.72)},
                "pub": {"xy": (84, 488), "from": (236, 684), "to": (-0.5, -0.6)}, "mor": LABELS["mor"]},
        scale_m=1000, author=AUTHOR,
    ),
}


def city(key: str) -> dict:
    """The configuration of one city, with its key added."""
    if key not in CITIES:
        raise SystemExit(f"unknown city {key!r}; known: {', '.join(CITIES)}")
    return {"key": key, **CITIES[key]}


def city_config(key: str, name: str, country: str, lat0: float, lon0: float, mpp: float = 3.3, **over) -> dict:
    """A configuration for a city that is not in ``CITIES``: the boxes are worked out from the lens geometry.

    (lat0, lon0) is the point at poster pixel (985, 1420). ``detail`` = the box around the three lenses plus 300 m,
    ``context`` = the whole 2000 x 2828 px poster plus 1 km. Everything else starts as the Richmond entry without its
    lens swap; pass any field as a keyword to change it (``tiles=4`` for a very dense city, ``lenses=...``,
    ``public=...``, ``water_kinds=...``)."""
    import math

    cfg = {**CITIES["richmond"], "lenses": CITIES["brisbane"]["lenses"], "labels": LABELS,
           "name": name, "country": country, "lat0": lat0, "lon0": lon0, "mpp": mpp, **over, "key": key}
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110574.0  # metres per degree of longitude / latitude here
    r = cfg["radius"]
    xs = [985 + dx + s * r for dx, _ in cfg["lenses"] for s in (-1, 1)]  # left and right edge of every lens (px)
    ys = [1420 + dy + s * r for _, dy in cfg["lenses"] for s in (-1, 1)]  # top and bottom edge of every lens (px)

    def box(x0, y0, x1, y1, pad_m):  # poster-pixel rectangle -> (south, west, north, east), padded by pad_m metres
        return (round(lat0 - ((y1 - 1420) * mpp + pad_m) / ky, 4), round(lon0 + ((x0 - 985) * mpp - pad_m) / kx, 4),
                round(lat0 - ((y0 - 1420) * mpp - pad_m) / ky, 4), round(lon0 + ((x1 - 985) * mpp + pad_m) / kx, 4))

    if "detail" not in over:
        cfg["detail"] = box(min(xs), min(ys), max(xs), max(ys), 300.0)
    if "context" not in over:
        cfg["context"] = box(0, 0, 2000, 2828, 1000.0)
    return cfg


# ---------------------------------------------------------------------------------------------- any place
# every point symbol the poster can draw (``gismaps.lens.city.SYMBOLS``): (OSM value, legend label, word in the label)
SUPPORTED = (("toilets", "Public toilet", "toilets"), ("drinking_water", "Drinking water", "taps"),
             ("bench", "Bench", "benches"), ("playground", "Playground", "playgrounds"),
             ("place_of_worship", "Place of worship", "places of worship"), ("marketplace", "Marketplace", "markets"),
             ("fountain", "Fountain", "fountains"), ("shelter", "Shelter", "shelters"), ("library", "Library", "libraries"))
RADIUS_KM = 3.3  # half the ground width of a full-size sheet: 6.6 km x 9.3 km, the scale of the Brisbane poster
FAST_FACTOR = 0.6  # a FAST sheet covers 60 % of that ground size (4.0 km x 5.6 km)


def slug(text: str) -> str:
    """A lower-case file-name key of a place name: "São Paulo" -> "sao_paulo", "Washington DC" -> "washington_dc"."""
    import re
    import unicodedata

    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", plain.lower()).strip("_") or "place"


def place_config(name: str, country: str, lat: float, lon: float, radius_km: float | None = None, fast: bool = True,
                 key: str | None = None) -> dict:
    """The configuration of any place from its centre alone (what ``gismaps.lens.make_city`` uses for a place that
    is not hand-tuned).

    ``radius_km`` = half the ground width of the 2000 px wide sheet, so the map scale is ``mpp = radius_km`` metres per
    pixel (default 3.3 km; with ``fast=True`` 60 % of it, 2.0 km). The three lenses sit in the Brisbane triangle; the
    boxes follow from the lens geometry (:func:`city_config`); buildings are asked for in tiles of about 2 km; the
    scale bar is the longest round length that stays clear of the legend. Every supported point symbol is fetched; the
    legend rows (``public``) are chosen later from what the place really holds (:func:`choose_public`)."""
    import math

    radius_km = float(radius_km) if radius_km else RADIUS_KM * (FAST_FACTOR if fast else 1.0)
    mpp = round(radius_km, 3)  # 1000 px from the middle of the sheet to its side
    cfg = city_config(
        key or slug(name), name, country, round(float(lat), 5), round(float(lon), 5), mpp,
        water_kinds=("river", "canal", "lake", "reservoir", "pond", "basin", "bay", "harbour", "lagoon", "oxbow",
                     "stream", "moat", "lock", "fishpond", "reflecting_pool", None),
        water_min_m2=3000.0 * (mpp / 3.3) ** 2, sea=True,
        amenity_re="|".join(k for k, _, _ in SUPPORTED if k != "playground"),
        public=tuple((k, lab) for k, lab, _ in SUPPORTED), count_area="lens", playground_areas=True,
        scale_m=max(s for s in (50, 100, 200, 500, 1000, 2000, 5000) if s / mpp <= 330 or s == 50),
    )
    s, w, n, e = cfg["detail"]
    side_km = max((n - s) * 110.574, (e - w) * 111.32 * math.cos(math.radians(lat)))
    cfg["tiles"] = max(1, min(6, math.ceil(side_km / 2.15)))  # FAST: 2 x 2 tiles of 2.1 km, FULL: 4 x 4 of 1.7 km
    return cfg


def choose_public(cfg: dict, n_lens: dict, n_box: dict) -> dict:
    """Set the legend rows of the PUBLIC REALM lens from the data: the four to six most numerous supported point
    kinds inside the lens (ties: the detail box), most numerous first; kinds with no point at all are left out unless
    fewer than four are present. Also writes the label lines and moves the legend up when it has more than four rows."""
    order = {k: i for i, (k, _, _) in enumerate(SUPPORTED)}
    ranked = sorted(order, key=lambda k: (-n_lens.get(k, 0), -n_box.get(k, 0), order[k]))
    keep = [k for k in ranked if n_lens.get(k, 0) > 0][:6]
    keep += [k for k in ranked if k not in keep][:max(0, 4 - len(keep))]
    lab = {k: (label, word) for k, label, word in SUPPORTED}
    cfg["public"] = tuple((k, lab[k][0]) for k in keep)
    words = ["Parks", *[lab[k][1] for k in keep[:4]]]
    # two lines of about the same length
    cut = min(range(1, len(words)), key=lambda i: max(len(", ".join(words[:i])), len(", ".join(words[i:]))))
    cfg["public_lines"] = (", ".join(words[:cut]) + ",", ", ".join(words[cut:]))
    if len(keep) > 4:
        cfg["legend_xy"], cfg["legend_dy"] = (1560, 1768), 34
    return cfg
