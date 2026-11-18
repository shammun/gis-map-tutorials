"""Lens map: the city poster "One city, many realities" for any configured city.

Three round lenses over one place, each drawing the same OpenStreetMap extract another way:
  LAND USE         categorical fills (OSM landuse / leisure / amenity polygons)
  PUBLIC REALM     parks and the point amenities OSM holds (toilets, drinking water, benches, playgrounds ...)
  CITY MORPHOLOGY  building footprints and street segments
The muted water continues outside the lenses; where lenses overlap the two maps blend 50/50 (the poster look).

Cities live in ``gismaps.lens.cities`` (one small dict each). Raw data comes from ``gismaps.lens.osm.fetch_city``.
Original code and data. Style after Fan Mo, #30DayMapChallenge 2026 Day 14 ("Borgesian map"); nothing traced or copied.
Data (c) OpenStreetMap contributors (ODbL).
Projection: local equirectangular about (lat0, lon0) (x = dlon * cos(lat0), north up); error < 0.1 % over 10 km.

Run:  python -m gismaps.lens.city --city dhaka|richmond|brisbane [--export] [--from-derived]
  --export        also write the reduced layers (lon/lat, clipped to the ground the lenses can reach) to <work folder>/derived/
  --from-derived  draw from that reduced file instead of the raw Overpass answers (what a notebook does)
Any other place:  python -m gismaps.lens city --place "Chittagong"   (``gismaps.lens.make.make_city``)

The poster is built in three steps that the tutorials (``tutorials/lens_maps/``) and the animation
(``gismaps.lens.anim``) call one by one:  ``render_layers`` (four full-poster images) -> ``lc.composite`` (the lenses)
-> ``furniture`` (title, label blocks, legend, north arrow, scale bar, credits).  ``poster`` runs all three.
The work folder is ``GISMAPS_WORK`` (default ``./gismaps_work``: raw Overpass answers in ``raw/``, posters in ``out/``);
every reader also takes an explicit path, and ``read_derived`` falls back to the copy shipped with the repository.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import shapely
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from . import common as lc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from .cities import CITIES, city as city_cfg  # noqa: E402

ROOT = Path(os.environ.get("GISMAPS_WORK", "gismaps_work"))
OUT = ROOT / "out"
DERIVED = ROOT / "derived"
W, H = 2000, 2828
CX, CY = 985.0, 1420.0  # where (lat0, lon0) sits on the poster

# land-use classes: (label, colour); OSM tag -> class
CLASSES = {
    "res": ("Residential", "#fff3a6"), "com": ("Commercial & retail", "#f58d86"), "ind": ("Industrial", "#b9a5dc"),
    "edu": ("Education", "#f3cf4d"), "med": ("Medical", "#f7bfe2"), "park": ("Park & garden", "#a7d77c"),
    "rec": ("Sport & recreation", "#f4a558"), "tra": ("Transport & parking", "#86d0c6"), "rel": ("Religious", "#cfc8ef"),
    "con": ("Construction & brownfield", "#d8d8d8"), "nat": ("Wood, grass & cemetery", "#7fae8a"),
}
LANDUSE = {"residential": "res", "commercial": "com", "retail": "com", "industrial": "ind", "education": "edu", "railway": "tra",
           "construction": "con", "brownfield": "con", "greenfield": "con", "religious": "rel", "cemetery": "nat", "grass": "nat",
           "forest": "nat", "meadow": "nat", "recreation_ground": "rec", "village_green": "park", "military": "con",
           "flowerbed": "park", "garages": "tra", "institutional": "com"}
LEISURE = {"park": "park", "garden": "park", "playground": "park", "nature_reserve": "nat", "pitch": "rec", "golf_course": "rec",
           "sports_centre": "rec", "stadium": "rec", "recreation_ground": "rec", "track": "rec"}
AMENITY = {"school": "edu", "university": "edu", "college": "edu", "kindergarten": "edu", "hospital": "med", "clinic": "med",
           "place_of_worship": "rel", "parking": "tra"}
NATURAL = {"wood": "nat", "scrub": "nat", "grassland": "nat"}
ORDER = ["res", "com", "ind", "con", "tra", "edu", "med", "rel", "nat", "park", "rec"]  # draw order (small on top)
ROAD_W = {"motorway": 5.0, "trunk": 4.4, "primary": 3.8, "secondary": 3.2, "tertiary": 2.6, "busway": 2.2, "residential": 1.7,
          "unclassified": 1.7, "living_street": 1.5, "pedestrian": 1.5, "service": 0.9}


def drop_marker():
    """A water-drop marker (point up)."""
    t = np.linspace(0, np.pi, 24)
    arc = np.column_stack([0.62 * np.cos(-t), -0.35 + 0.62 * np.sin(-t)])  # lower half circle, left <- right
    v = np.vstack([[0.0, 1.0], arc[0], arc, [0.0, 1.0]])
    return MPath(v)


# point symbols of the PUBLIC REALM lens: OSM value -> (style on the map, z-order, style in the legend)
SYMBOLS = {
    "toilets": (dict(s=52, marker="o", facecolors="none", edgecolors="#ff5a4f", linewidths=2.4), 9,
                dict(s=52, marker="o", facecolors="none", edgecolors="#ff5a4f", linewidths=2.4)),
    "drinking_water": (dict(s=58, marker=drop_marker(), c="#7fd8ff", edgecolors="#0b2a3a", linewidths=0.6), 10,
                       dict(s=58, marker=drop_marker(), c="#7fd8ff", linewidths=0)),
    "bench": (dict(s=13, marker="^", c="#3d8bff", linewidths=0), 7, dict(s=22, marker="^", c="#3d8bff", linewidths=0)),
    "playground": (dict(s=46, marker="s", c="#ffd166", edgecolors="#33241a", linewidths=0.8), 8,
                   dict(s=46, marker="s", c="#ffd166", linewidths=0)),
    "place_of_worship": (dict(s=34, marker="D", c="#d9b8ff", edgecolors="#33241a", linewidths=0.6), 8,
                         dict(s=34, marker="D", c="#d9b8ff", linewidths=0)),
    "marketplace": (dict(s=66, marker="p", c="#ff9f43", edgecolors="#33241a", linewidths=0.6), 8,
                    dict(s=66, marker="p", c="#ff9f43", linewidths=0)),
    "fountain": (dict(s=70, marker="*", c="#9be7ff", edgecolors="#0b2a3a", linewidths=0.5), 9,
                 dict(s=70, marker="*", c="#9be7ff", linewidths=0)),
    "shelter": (dict(s=40, marker="h", c="#e9e2d0", edgecolors="#33241a", linewidths=0.6), 8,
                dict(s=40, marker="h", c="#e9e2d0", linewidths=0)),
    "library": (dict(s=44, marker="P", c="#f4ecd2", edgecolors="#33241a", linewidths=0.5), 8,
                dict(s=44, marker="P", c="#f4ecd2", linewidths=0)),
}


class Proj:
    """Local equirectangular map of one city: (lon, lat) <-> poster pixels."""

    def __init__(self, cfg: dict):
        self.lat0, self.lon0, self.mpp = cfg["lat0"], cfg["lon0"], cfg["mpp"]
        self.kx, self.ky = 111320.0 * math.cos(math.radians(self.lat0)), 110574.0

    def xy(self, lon, lat):
        return CX + (lon - self.lon0) * self.kx / self.mpp, CY - (lat - self.lat0) * self.ky / self.mpp

    def px(self, geom) -> np.ndarray:
        """Overpass geometry (list of {lon, lat}) -> (n, 2) poster pixels."""
        a = np.array([(p["lon"], p["lat"]) for p in geom], dtype=np.float64)
        return np.column_stack(self.xy(a[:, 0], a[:, 1]))

    def lonlat(self, xy) -> np.ndarray:
        a = np.asarray(xy, dtype=np.float64)
        return np.column_stack([self.lon0 + (a[:, 0] - CX) * self.mpp / self.kx, self.lat0 - (a[:, 1] - CY) * self.mpp / self.ky])

    def from_lonlat(self, ll) -> np.ndarray:
        a = np.asarray(ll, dtype=np.float64).reshape(-1, 2)
        return np.column_stack(self.xy(a[:, 0], a[:, 1]))


def polys(el: dict, pr: Proj) -> list[Polygon]:
    """Polygon(s) of a closed way or a multipolygon relation, in poster pixels."""
    try:
        if el["type"] == "way":
            g = el.get("geometry") or []
            if len(g) >= 4 and g[0] == g[-1]:
                p = Polygon(pr.px(g))
                return [p if p.is_valid else p.buffer(0)]
            return []
        outer, inner = [], []
        for m in el.get("members", []):
            if m.get("type") == "way" and m.get("geometry") and len(m["geometry"]) >= 2:
                (inner if m.get("role") == "inner" else outer).append(LineString(pr.px(m["geometry"])))
        if not outer:
            return []
        po = unary_union(list(polygonize(unary_union(outer))))
        if inner:
            pi = list(polygonize(unary_union(inner)))
            if pi:
                po = po.difference(unary_union(pi))
        return [g for g in getattr(po, "geoms", [po]) if g.geom_type == "Polygon" and not g.is_empty]
    except Exception:  # noqa: BLE001  (a broken ring in the extract is skipped, not fatal)
        return []


def patches(pp: list[Polygon]) -> list[PathPatch]:
    out = []
    for p in pp:
        verts, codes = [], []
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            verts.append(xy)
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
        out.append(PathPatch(MPath(np.concatenate(verts), codes)))
    return out


def add_polys(ax, pp, fc, ec="none", lw=0.0, alpha=1.0, z=2):
    if pp:
        ax.add_collection(PatchCollection(patches(pp), facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=z))


def box_clip(pp: list[Polygon]) -> list[Polygon]:
    b = shapely.box(-50, -50, W + 50, H + 50)
    out = []
    for p in pp:
        if p.intersects(b):
            c = p.intersection(b)
            out += [g for g in getattr(c, "geoms", [c]) if g.geom_type == "Polygon"]
    return out


def lens_discs(cfg: dict) -> list[tuple[float, float, float]]:
    return [(CX + dx, CY + dy, cfg["radius"]) for dx, dy in cfg["lenses"]]


def sea_polys(elements: list[dict], pr: Proj) -> list[Polygon]:
    """The sea inside the poster, from the OSM coastlines (ways tagged natural=coastline, which OSM draws with the land
    on their left and the water on their right). The coastlines and the poster frame cut the sheet into faces; a face
    is sea when more points just right of a coastline fall in it than points just left of one."""
    ways = [pr.px(e["geometry"]) for e in elements if e.get("type") == "way"
            and e.get("tags", {}).get("natural") == "coastline" and len(e.get("geometry") or []) >= 2]
    if not ways:
        return []
    try:
        frame = shapely.box(-50, -50, W + 50, H + 50)
        cut = unary_union([*[LineString(xy) for xy in ways], frame.boundary])
        faces = [f for f in polygonize(cut) if f.representative_point().within(frame)]
        if not faces:
            return []
        tree, votes = shapely.STRtree(faces), np.zeros((len(faces), 2))
        for xy in ways:
            d = np.diff(xy, axis=0)
            n = np.hypot(d[:, 0], d[:, 1])
            ok = n > 1e-6
            mid = ((xy[:-1] + xy[1:]) / 2)[ok]
            right = np.column_stack([-d[:, 1], d[:, 0]])[ok] / n[ok, None]  # y runs down the poster, so this is the right
            for side, col in ((2.0, 0), (-2.0, 1)):  # 2 px to the right (sea) and to the left (land) of every segment
                hit = tree.query(shapely.points(mid + side * right), predicate="within")
                np.add.at(votes[:, col], hit[1], 1)
        return [f for f, (sea, land) in zip(faces, votes) if sea > land]
    except Exception:  # noqa: BLE001  (a broken coastline in the extract only means no sea is drawn)
        return []


# ------------------------------------------------------------------------------------------------ read
def read_raw(cfg: dict, pr: Proj, raw: Path | None = None) -> dict:
    """The Overpass answers of one city (folder ``raw``, default <data root>/raw/OSM-overpass-<key>) -> layers in
    poster pixels."""
    c = cfg["key"]
    raw = Path(raw) if raw else ROOT / "raw" / f"OSM-overpass-{c}"
    mpp = cfg["mpp"]

    def load(name: str) -> list[dict]:
        return json.loads((raw / f"{c}_{name}.json").read_text(encoding="utf-8"))["elements"]

    kinds = cfg["water_kinds"]
    water = [p for e in load("water") for p in polys(e, pr)
             if (e.get("tags", {}).get("water") in kinds or e.get("tags", {}).get("waterway"))]
    water = box_clip([p for p in water if p.area * mpp * mpp > cfg["water_min_m2"]])
    if cfg.get("sea"):
        water += sea_polys(load("water"), pr)
    river = unary_union(water)
    water = [g for g in getattr(river, "geoms", [river]) if g.geom_type == "Polygon"]
    extra = {tag: k for tag, (k, _lab, _col) in cfg.get("extra_landuse", {}).items()}  # landuse value -> extra class
    lu: dict[str, list[Polygon]] = {k: [] for k in [*CLASSES, *extra.values()]}
    skipped = Counter()
    for e in load("landuse"):
        t = e.get("tags", {})
        k = extra.get(t.get("landuse", "")) or LANDUSE.get(t.get("landuse", "")) or LEISURE.get(t.get("leisure", "")) or AMENITY.get(t.get("amenity", "")) or NATURAL.get(t.get("natural", ""))
        if k:
            lu[k] += polys(e, pr)
        elif t.get("landuse"):
            skipped[t["landuse"]] += 1
    roads, rails = [], []
    for e in load("roads"):
        t = e.get("tags", {})
        if "geometry" not in e:
            continue
        if t.get("railway"):
            rails.append(pr.px(e["geometry"]))
        else:
            roads.append((t.get("highway", "").replace("_link", ""), pr.px(e["geometry"])))
    ctx_roads = [pr.px(e["geometry"]) for e in load("roads_context") if "geometry" in e]
    seen, bld = set(), []
    for f in sorted(raw.glob(f"{c}_buildings_*.json")):
        for e in json.loads(f.read_text(encoding="utf-8"))["elements"]:
            if (e["type"], e["id"]) not in seen:
                seen.add((e["type"], e["id"]))
                bld += polys(e, pr)
    want = [k for k, _ in cfg["public"]]
    pts: dict[str, list] = {k: [] for k in want}
    other = Counter()
    for e in load("amenities"):
        t = e.get("tags", {})
        k = t.get("amenity") if t.get("amenity") in pts else ("playground" if t.get("leisure") == "playground" else None)
        if k not in pts:
            other[t.get("amenity") or t.get("leisure")] += 1
            continue
        lat = e.get("lat") or e.get("center", {}).get("lat")
        lon = e.get("lon") or e.get("center", {}).get("lon")
        if lat is None:
            continue
        pts[k].append(pr.xy(lon, lat))
    if cfg.get("playground_areas") and "playground" in pts:  # playgrounds mapped as areas count too (centroids)
        for e in load("landuse"):
            if e.get("tags", {}).get("leisure") == "playground":
                pts["playground"] += [(q.centroid.x, q.centroid.y) for q in polys(e, pr)[:1]]
    if skipped:
        print("landuse values not drawn:", dict(skipped.most_common(12)))
    if other:
        print("amenity points fetched but not drawn:", dict(other))
    return dict(water=water, lu=lu, roads=roads, rails=rails, ctx_roads=ctx_roads, bld=bld, pts=pts)


# ------------------------------------------------------------------------------------------------ reduced file
def _rings(p: Polygon, pr: Proj) -> list:
    return [np.round(pr.lonlat(np.asarray(r.coords)), 6).tolist() for r in [p.exterior, *p.interiors]]


def export(cfg: dict, pr: Proj, L: dict, counts: dict, out_dir: Path | None = None, zone_kind: str = "hull") -> Path:
    """Write the layers the poster needs (lon/lat, 6 decimals) as one gzip JSON.

    ``zone_kind="hull"`` (default) keeps every layer inside the convex hull of the three lenses + 25 px: all the ground
    a lens can cross when it slides from one lens position to another, so the same file feeds the poster and the
    animation. ``zone_kind="lens"`` keeps each layer only under its own lens (the first, smaller files of 2026-11-17:
    enough for the static poster)."""
    discs = lens_discs(cfg)
    if zone_kind == "hull":
        zone = shapely.MultiPoint([(x, y) for x, y, _ in discs]).convex_hull.buffer(discs[0][2] + 25, 64)
        use_z = pub_z = mor_z = zone
    else:
        zone = unary_union([Point(x, y).buffer(r + 25, 64) for x, y, r in discs])
        use_z, pub_z, mor_z = (Point(x, y).buffer(r + 25, 64) for x, y, r in discs)

    def keep(pp, z):
        return [p for p in pp if p.intersects(z)]

    def keep_l(ls, z):
        return [xy for xy in ls if LineString(xy).intersects(z)]

    out = {
        "city": {k: v for k, v in cfg.items() if k not in ("labels", "debug")},
        "zone": zone_kind,
        "crs": "EPSG:4326 lon/lat, 6 decimals; polygons = [exterior, hole, ...]",
        "licence": "(c) OpenStreetMap contributors, ODbL 1.0; Overpass API extract",
        "counts": counts,
        "water": [_rings(p, pr) for p in L["water"]],
        "landuse": {k: [_rings(p, pr) for p in keep(v, zone if k in ("park", "rec", "nat") else use_z)] for k, v in L["lu"].items()},
        "roads": [[k, np.round(pr.lonlat(xy), 6).tolist()] for k, xy in L["roads"] if LineString(xy).intersects(zone)],
        "rails": [np.round(pr.lonlat(xy), 6).tolist() for xy in keep_l(L["rails"], zone)],
        "ctx_roads": [np.round(pr.lonlat(xy), 5).tolist() for xy in L["ctx_roads"]],
        "buildings": [_rings(p, pr) for p in keep(L["bld"], mor_z)],
        "points": {k: np.round(pr.lonlat(v), 6).tolist() if v else [] for k, v in L["pts"].items()},
    }
    out_dir = Path(out_dir) if out_dir else DERIVED
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{cfg['key']}_lens_layers.json.gz"
    with gzip.open(f, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(out, fh, separators=(",", ":"))
    print("derived ->", f, f"{f.stat().st_size / 1e6:.2f} MB")
    return f


def read_derived(cfg: dict, pr: Proj, path: Path | None = None) -> dict:
    """The reduced file (``path``, default: the copy shipped with the repository, <key>_lens_layers.json.gz) -> the same
    layers as :func:`read_raw`, inside the zone the file was clipped to."""
    if not path:
        path = DERIVED / f"{cfg['key']}_lens_layers.json.gz"
        if not path.exists():  # the reduced copy shipped with the repository (downloaded once if it is not on disk)
            from .. import data as _data

            path = _data.path("lens_maps", path.name)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        d = json.load(fh)

    def P(rr):
        return Polygon(pr.from_lonlat(rr[0]), [pr.from_lonlat(h) for h in rr[1:]])

    return dict(water=[P(r) for r in d["water"]], lu={k: [P(r) for r in v] for k, v in d["landuse"].items()},
                roads=[(k, pr.from_lonlat(xy)) for k, xy in d["roads"]], rails=[pr.from_lonlat(xy) for xy in d["rails"]],
                ctx_roads=[pr.from_lonlat(xy) for xy in d["ctx_roads"]], bld=[P(r) for r in d["buildings"]],
                pts={k: [tuple(q) for q in pr.from_lonlat(v)] if v else [] for k, v in d["points"].items()})


# ------------------------------------------------------------------------------------------------ poster
RIMS = ("#f2a6d8", "#f0b565", "#a9e5a2")  # rim colours: land use (pink), public realm (amber), morphology (green)
BG = "#000000"


def count(cfg: dict, L: dict) -> dict:
    """Feature counts of one city: per layer, in the detail box and inside the lenses (printed in the legend)."""
    R = cfg["radius"]
    water, lu, roads, rails, bld, pts = (L[k] for k in ("water", "lu", "roads", "rails", "bld", "pts"))
    discs = lens_discs(cfg)
    # counts: all mapped points of the detail box, and those inside the PUBLIC REALM lens
    px_, py_, pr_ = discs[1]
    n_all = {k: len(v) for k, v in pts.items()}
    n_lens = {k: int(sum(math.hypot(x - px_, y - py_) <= pr_ for x, y in v)) for k, v in pts.items()}
    mx, my, _ = discs[2]
    n_bld = int(sum(math.hypot(p.centroid.x - mx, p.centroid.y - my) <= R for p in bld))
    counts = dict(water=len(water), landuse={k: len(v) for k, v in lu.items()}, roads=len(roads), rails=len(rails),
                  buildings=len(bld), buildings_in_morphology_lens=n_bld, points_detail_box=n_all, points_in_public_lens=n_lens)
    return counts


def _classes(cfg: dict) -> tuple[dict, list]:
    """(all land-use classes of this city, the keys of its extra classes)."""
    classes = {**CLASSES, **{k: (label, col) for k, label, col in cfg.get("extra_landuse", {}).values()}}
    return classes, [k for k in classes if k not in CLASSES]


def render_layers(cfg: dict, L: dict) -> dict:
    """The four full-poster images (RGB float arrays, 2828 x 2000): ``base`` (muted water and main roads, seen outside
    the lenses), ``use`` (LAND USE), ``pub`` (PUBLIC REALM), ``mor`` (CITY MORPHOLOGY). All four are drawn on the same
    pixel grid, so a lens can sit anywhere and still line up with the map around it."""
    water, lu, roads, rails, ctx_roads, bld, pts = (L[k] for k in ("water", "lu", "roads", "rails", "ctx_roads", "bld", "pts"))
    classes, extra_k = _classes(cfg)
    parks = lu["park"] + lu["rec"]
    by_w: dict[float, list] = {}
    for k, xy in roads:
        by_w.setdefault(ROAD_W.get(k, 1.4), []).append(xy)

    def road_lines(ax, colour, scale=1.0, z=5, alpha=1.0):
        for w, ls in sorted(by_w.items()):
            ax.add_collection(LineCollection(ls, colors=colour, linewidths=w * scale, capstyle="round", joinstyle="round", zorder=z, alpha=alpha))

    # ---------------------------------------------------------------- layers (poster pixels)
    bg = BG
    fig, ax = lc.canvas(W, H, bg)  # context: the water and the main roads, very muted
    add_polys(ax, water, "#26272b")
    ax.add_collection(LineCollection(ctx_roads, colors="#151619", linewidths=1.6, zorder=3))
    base = lc.grab(fig)[..., :3]

    fig, ax = lc.canvas(W, H, "#2b2130")  # LAND USE (ground = no land-use polygon in OSM)
    for k in [*ORDER[:4], *extra_k, *ORDER[4:]]:
        add_polys(ax, lu.get(k, []), classes[k][1], z=2)
    add_polys(ax, water, "#33383f", z=3)
    road_lines(ax, "#1d1522", 1.25, z=5)
    ax.add_collection(LineCollection(rails, colors="#1d1522", linewidths=1.6, zorder=5))
    L_use = lc.grab(fig)[..., :3]

    fig, ax = lc.canvas(W, H, "#33241a")  # PUBLIC REALM
    add_polys(ax, parks + lu["nat"], "#3f7d26", z=2)
    add_polys(ax, water, "#55504b", z=3)
    road_lines(ax, "#120b06", 1.0, z=5)
    for k, _lab in cfg["public"]:
        st, z, _ = SYMBOLS[k]
        a = np.array(pts[k]) if pts[k] else np.zeros((0, 2))
        ax.scatter(a[:, 0], a[:, 1], zorder=z, **st)
    L_pub = lc.grab(fig)[..., :3]

    fig, ax = lc.canvas(W, H, "#2c382d")  # CITY MORPHOLOGY
    add_polys(ax, parks, "#22301f", z=1)
    add_polys(ax, water, "#252b2a", z=3)
    road_lines(ax, "#f2f2ee", 0.55, z=4, alpha=0.9)
    ax.add_collection(LineCollection(rails, colors="#f2f2ee", linewidths=0.8, linestyles=(0, (4, 3)), zorder=4, alpha=0.7))
    add_polys(ax, bld, "#b9bcb8", ec="#5c625c", lw=0.35, z=6)
    L_mor = lc.grab(fig)[..., :3]
    return dict(base=base, use=L_use, pub=L_pub, mor=L_mor)


def make_lenses(cfg: dict, layers: dict) -> list:
    """The three lenses at their poster positions, in drawing order (land use, public realm, morphology).
    ``overlap = 0.5``: where two lenses meet, the upper map is mixed half and half with the lower one. That is the
    poster look and it is fine for these categorical city maps; never use it for a data field with a colour bar."""
    return [lc.Lens(layers[k], x, y, r, col, None, 0.5) for k, (x, y, r), col in zip(("use", "pub", "mor"), lens_discs(cfg), RIMS)]


INK, SUB, MUT = "#f1ede4", "#c9c5bc", "#8d8a84"  # text colours: main, secondary, muted
PARTS = ("head", "labels", "legend", "foot")


def furniture(fig, ax, cfg: dict, lenses: list, counts: dict, parts=PARTS, label_keys=("use", "pub", "mor"), leaders=True,
              blocks=True) -> None:
    """Everything on the poster that is not map: ``head`` (header line, title, subtitle), ``labels`` (one label block
    per lens with a leader line to its rim), ``legend`` (right column), ``foot`` (north arrow, scale bar, closing
    sentence, credit block). All positions are poster pixels. The animation draws the parts on separate transparent
    canvases so that each can fade in on its own."""
    place = f"{cfg['name']}, {cfg['country']}" if cfg.get("country") else cfg["name"]
    classes, extra_k = _classes(cfg)
    am = counts["points_in_public_lens"] if cfg.get("count_area", "detail") == "lens" else counts["points_detail_box"]
    if "head" in parts:
        _head(fig, ax, place)
    if "labels" in parts:
        _labels(ax, cfg, lenses, label_keys, leaders, blocks)
    if "legend" in parts:
        _legend(ax, cfg, am, classes, extra_k)
    if "foot" in parts:
        _foot(ax, cfg, place)


def _head(fig, ax, place: str) -> None:
    ink, sub, mut = INK, SUB, MUT
    lc.text(ax, 60, 62, "LENS MAP · STYLE STUDY", 24, mut)
    th = lc.text(ax, W - 60, 62, place.upper(), 24, mut, ha="right")
    lc.text(ax, 84, 250, "ONE CITY,", 104, ink, lc.SERIF, "bold")
    t1 = lc.text(ax, 84, 372, "MANY ", 104, ink, lc.SERIF, "bold")
    fig.canvas.draw()
    ax.plot([440, min(W - 410, th.get_window_extent().x0 - 60)], [54, 54], color="#55534f", lw=1.2, zorder=30)
    lc.text(ax, t1.get_window_extent().x1 + 4, 372, "REALITIES.", 104, "#e8a25c", lc.SERIF, "bold", "italic")
    lc.text(ax, 86, 436, "The same streets, drawn three ways. No single map shows all of it.", 32, sub, lc.SERIF, style="italic")


def _labels(ax, cfg: dict, lenses: list, keys=("use", "pub", "mor"), leaders: bool = True, blocks_on: bool = True) -> None:
    """Label budget: three blocks, one per lens, each with a bar in the rim colour, a CAPS name, an italic theme and
    two short lines; one straight leader from the block to a fixed point on the rim. No place names on the map."""
    R = cfg["radius"]
    s = 40
    lab = cfg["labels"]
    blocks = (("use", "LAND USE", "Function", list(cfg.get("use_lines", ("How the ground", "is used")))),
              ("pub", "PUBLIC REALM", "Everyday life", list(cfg["public_lines"])),
              ("mor", "CITY MORPHOLOGY", "Form", list(cfg.get("mor_lines", ("Buildings and streets:", "how the city is shaped")))))
    for (key, name, theme, lines), Ln in zip(blocks, lenses):
        if key not in keys:
            continue
        b = lab[key]
        if blocks_on:
            lc.label_block(ax, b["xy"][0], b["xy"][1], name, theme, lines, Ln.rim, s)
        if leaders:
            lc.leader(ax, b["from"], (Ln.cx + R * b["to"][0], Ln.cy + R * b["to"][1]))


def _legend(ax, cfg: dict, am: dict, classes: dict, extra_k: list) -> None:
    ink, sub = INK, SUB
    # legend (right column, below the public-realm lens)
    (lx, ly), dy = cfg.get("legend_xy", (1560, 1790)), cfg.get("legend_dy", 37)

    def sw(y, colour, label, kind="box"):
        if kind == "box":
            ax.add_patch(plt.Rectangle((lx, y - 13), 40, 24, fc=colour, ec="none", zorder=30))
        elif kind == "line":
            ax.plot([lx, lx + 40], [y, y], color=colour, lw=1.6, zorder=30)
        else:
            ax.scatter([lx + 20], [y], zorder=31, **kind)
        lc.text(ax, lx + 58, y + 8, label, 23, sub)

    sw(ly, "#b9bcb8", "Building footprint")
    sw(ly + dy, "#f2f2ee", "Street segment", "line")
    sw(ly + 2 * dy, "#3f7d26", "Park, sports ground, wood")
    n = len(cfg["public"])
    for i, (k, label) in enumerate(cfg["public"]):
        sw(ly + (3 + i) * dy, None, f"{label} ({am[k]:,})" if am[k] >= 10000 else f"{label} ({am[k]})", SYMBOLS[k][2])
    lc.text(ax, lx, ly + (3 + n) * dy + 30, "Land use", 32, ink)
    y0 = ly + (4 + n) * dy + 28
    rows = ["res", "com", "ind", "edu", "med", "rel", "park", "rec", "nat", "tra", "con", *extra_k]
    for i, k in enumerate(rows):
        sw(y0 + i * dy, classes[k][1], classes[k][0])
    yn = y0 + len(rows) * dy
    sw(yn, "#2b2130", "No land-use polygon in OSM")
    ax.add_patch(plt.Rectangle((lx, yn - 13), 40, 24, fc="none", ec="#6b6470", lw=1, zorder=31))


def _foot(ax, cfg: dict, place: str) -> None:
    ink, sub, mut = INK, SUB, MUT
    mpp = cfg["mpp"]
    # north arrow and scale bar
    nx, ny = cfg.get("north_xy", (1050, 2440))
    ax.add_patch(plt.Polygon([(nx, ny - 60), (nx - 17, ny), (nx, ny - 13), (nx + 17, ny)], closed=True, fc="none", ec=ink, lw=1.6, zorder=30))
    lc.text(ax, nx, ny + 32, "N", 26, ink, weight="bold", ha="center")
    sm = cfg["scale_m"]
    sx, sy, Lb = nx + 100, ny, sm / mpp
    ax.plot([sx, sx, sx + Lb, sx + Lb], [sy - 12, sy, sy, sy - 12], color=ink, lw=1.6, zorder=30)
    ax.plot([sx + Lb / 2, sx + Lb / 2], [sy - 8, sy], color=ink, lw=1.6, zorder=30)
    for k, t in ((0, "0"), (0.5, f"{sm // 2:,}"), (1, f"{sm:,} m")):
        lc.text(ax, sx + Lb * k, sy - 24, t, 24, ink, ha="center")
    # footer
    ax.plot([84, W - 84], [2580, 2580], color="#55534f", lw=1.2, zorder=30)
    lc.text(ax, 84, 2660, "One city, three maps of the same ground. Land use, public amenities and built", 30, sub, lc.SERIF, style="italic")
    lc.text(ax, 84, 2702, "form each show something the others leave out.", 30, sub, lc.SERIF, style="italic")
    credit = ["DATA: © OpenStreetMap contributors (ODbL)",
              f"Overpass API extract, {cfg.get('extract', 'November 2026')} · Python: shapely, matplotlib",
              f"{place} · north up · local equirectangular map" if len(place) <= 26 else f"{place[:48]} · north up",
              "Style after Fan Mo, #30DayMapChallenge 2026 Day 14"]
    lc.credit_block(ax, 1330, credit, cfg.get("author"), ink, mut)


def poster(cfg: dict, L: dict, out: Path, layers: dict | None = None) -> dict:
    """Draw the poster of one city to ``out`` (2000 x 2828 px PNG); returns the feature counts.
    ``layers`` = the four images of :func:`render_layers` if they are already drawn (else they are drawn here)."""
    c = cfg["key"]
    counts = count(cfg, L)
    print(c, json.dumps(counts))
    layers = layers or render_layers(cfg, L)
    if cfg.get("debug"):  # the three whole layers, to choose where the lenses go (not a deliverable)
        dbg = Path(out).parent / "cities"
        dbg.mkdir(exist_ok=True)
        for nm in ("use", "pub", "mor"):
            lc.save_rgb(dbg / f"{c}_layer_{nm}.png", layers[nm][::2, ::2])
    lenses = make_lenses(cfg, layers)
    img = lc.composite(layers["base"], lenses)
    fig, ax = lc.canvas(W, H, BG)
    ax.imshow(img, extent=(0, W, H, 0), interpolation="nearest", zorder=1)
    lc.rims(ax, lenses, 3.0)
    furniture(fig, ax, cfg, lenses, counts)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
    print("poster ->", out)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--city", default="brisbane", choices=sorted(CITIES))
    ap.add_argument("--export", action="store_true", help="also write the reduced layers to <work folder>/derived/")
    ap.add_argument("--from-derived", action="store_true", help="draw from the reduced file instead of the raw answers")
    ap.add_argument("--no-author", action="store_true", help="leave out the creator line (the first Brisbane study)")
    ap.add_argument("--debug-layers", action="store_true", help="also save the three whole layers (half size) for tuning")
    ap.add_argument("--out", default="", help="output PNG (default: <work folder>/out/lens_<city>.png)")
    a = ap.parse_args()
    cfg = city_cfg(a.city)
    if a.no_author:
        cfg["author"] = None
    cfg["debug"] = a.debug_layers
    pr = Proj(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    L = read_derived(cfg, pr) if a.from_derived else read_raw(cfg, pr)
    counts = poster(cfg, L, Path(a.out) if a.out else OUT / f"lens_{a.city}.png")
    if a.export and not a.from_derived:
        export(cfg, pr, L, counts)


if __name__ == "__main__":
    main()
