"""gismaps.data: find the small input files of the tutorials, and download them when they are not on disk.

The repository carries a reduced copy of every input (``data/gis_styles/*.npz``, ``data/lens_maps/*``; about 21 MB in
all, each with its source and licence in the README). This module looks for a file in this order:

1. ``$GISMAPS_DATA/<group>/``                 a folder you chose;
2. ``<repository>/data/<group>/``             a clone of the repository (``pip install -e .``);
3. ``./data/<group>/``, ``../../data/<group>/``  the current folder is the repository root or a tutorial folder;
4. ``./gismaps_data/<group>/``                the download folder: the file is fetched once from the repository on
   GitHub (``$GISMAPS_DATA_BASE`` overrides the address; a ``file://`` address works too). This is what happens on Google Colab after
   ``pip install git+https://github.com/shammun/gis-map-tutorials``.

If the download fails, the seven style inputs are rebuilt from their original public sources
(``gismaps.prepare_data``); the lens-map tutorials have their own live routes (Overpass, ARCO-ERA5).
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

REPO = "shammun/gis-map-tutorials"
RAW_URL = os.environ.get("GISMAPS_DATA_BASE", f"https://raw.githubusercontent.com/{REPO}/main/data")
USER_AGENT = "gismaps-tutorials/0.1"

#: name of each style input -> the builder in ``gismaps.prepare_data`` that makes it from the public source
BUILDERS = {"cyclones": "cyclones", "ciaran": "ciaran", "terrain_nweurope": "terrain", "tcwv_southasia": "tcwv",
            "oisst_2025": "oisst", "woa23_temperature": "woa", "jamuna_s2": "sentinel"}


def _candidates(group: str) -> list[Path]:
    here = Path.cwd()
    out = []
    if os.environ.get("GISMAPS_DATA"):
        out.append(Path(os.environ["GISMAPS_DATA"]) / group)
    out += [Path(__file__).resolve().parents[1] / "data" / group, here / "data" / group,
            here.parent.parent / "data" / group]
    return out


def download_folder(group: str) -> Path:
    """Where downloaded or rebuilt files of ``group`` go (``$GISMAPS_DATA`` if set, else ``./gismaps_data``)."""
    root = Path(os.environ["GISMAPS_DATA"]) if os.environ.get("GISMAPS_DATA") else Path.cwd() / "gismaps_data"
    return root / group


def write_folder(group: str) -> Path:
    """Where a rebuilt input is written: ``$GISMAPS_DATA`` if set, else the ``data/<group>`` folder of a clone of the
    repository, else ``./gismaps_data``."""
    repo = Path(__file__).resolve().parents[1] / "data" / group
    if not os.environ.get("GISMAPS_DATA") and repo.is_dir():
        return repo
    return download_folder(group)


def folder(group: str) -> Path:
    """The first folder that already holds files of ``group`` ("gis_styles" or "lens_maps"), else the download folder."""
    for c in [*_candidates(group), download_folder(group)]:
        if c.is_dir() and any(c.iterdir()):
            return c
    return download_folder(group)


def find(group: str, name: str) -> Path | None:
    """The file if it is on disk in one of the known places, else ``None``."""
    for c in [*_candidates(group), download_folder(group)]:
        if (c / name).is_file():
            return c / name
    return None


def path(group: str, name: str, log=print) -> Path:
    """Path of one input file; downloaded once from the repository on GitHub if it is not on disk."""
    p = find(group, name)
    if p is not None:
        return p
    p = download_folder(group) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_URL}/{group}/{name}"
    tmp = p.with_name(p.name + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(p)
    except Exception as err:  # noqa: BLE001  (offline, or the file is not published)
        tmp.unlink(missing_ok=True)
        raise FileNotFoundError(f"{name} is not on disk and could not be downloaded from {url} ({err})") from err
    log(f"downloaded {name} ({p.stat().st_size / 1e6:.1f} MB) -> {p.parent}")
    return p


def load_npz(name: str, log=print) -> dict:
    """One reduced input of the style tutorials (``<name>.npz``) as a dict of arrays.

    On disk -> read it. Not on disk -> download it from the repository. Download fails -> rebuild it from the
    original public source with ``gismaps.prepare_data`` (minutes; see that module for what each builder transfers).
    """
    import numpy as np

    try:
        p = path("gis_styles", f"{name}.npz", log=log)
    except FileNotFoundError as err:
        if name not in BUILDERS:
            raise
        log(f"{err}\nrebuilding {name}.npz from the public source ...")
        from . import prepare_data

        p = prepare_data.BUILDERS[BUILDERS[name]]()
    return dict(np.load(p))
