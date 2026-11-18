"""gismaps.lens: lens maps. ``common`` (the lens rule, canvas, label and credit blocks), ``cities`` (one dict per city),
``osm`` (polite Overpass downloads, place names), ``city`` (the city poster), ``weather`` (the ERA5 storm poster),
``anim`` (the animated posters), ``make`` (``make_city``: poster, video and GIF of any place in one call), ``picker``
(choose a place in a notebook: list, text box or a tap on a map).
Style after Fan Mo, #30DayMapChallenge 2026 Day 14 ("Borgesian map"); the code and data are original."""


def make_city(place=None, lat=None, lon=None, outputs=("poster", "video", "gif"), radius_km=None, fast=True,
              out_dir="out", preset=None, log=print) -> dict:
    """The lens-map poster, video and/or GIF of any place: see :func:`gismaps.lens.make.make_city`.
    (The drawing modules load when this is called, so ``import gismaps.lens`` stays light.)"""
    from .make import make_city as _make_city

    return _make_city(place, lat, lon, outputs, radius_km, fast, out_dir, preset, log)
