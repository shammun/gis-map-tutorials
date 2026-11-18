"""gismaps.glass.ortho: the orthographic projection, written out by hand.

Stand far away from the Earth, straight above the point (lat0, lon0). What you see is a disc: the orthographic
projection. Each glass pane shows exactly that view, which is why the panes line up with the globe behind them.

Axes (globe radius 1): y points to the North Pole, z points out of the Earth at 0 N 0 E, x points out at 0 N 90 E.
At the centre point there are three unit vectors: ``d`` straight out of the globe (towards the viewer), ``e`` towards
the east and ``n`` towards the north. A place ``p`` lands on the pane at u = p . e (right), v = p . n (up); it is on
the visible side when w = p . d is positive.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def xyz(lat, lon) -> np.ndarray:
    """Unit vector(s) of the place(s) at latitude ``lat``, longitude ``lon`` (degrees)."""
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)], axis=-1)


def basis(lat0: float, lon0: float):
    """The three unit vectors at the centre point: d (outwards), e (east), n (north)."""
    la, lo = np.radians(lat0), np.radians(lon0)
    d = np.array([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)])
    e = np.array([np.cos(lo), 0.0, -np.sin(lo)])
    n = np.array([-np.sin(la) * np.sin(lo), np.cos(la), -np.sin(la) * np.cos(lo)])
    return d, e, n


def forward(lat, lon, lat0: float, lon0: float):
    """Place -> (u, v, w): right, up, towards the viewer. Visible when w > 0."""
    d, e, n = basis(lat0, lon0)
    p = xyz(np.asarray(lat, float), np.asarray(lon, float))
    return p @ e, p @ n, p @ d


def inverse(u, v, lat0: float, lon0: float, s: float = 1.0):
    """Pane coordinates (u, v in -1..1) -> latitude, longitude. ``s`` = sin(cap radius); s = 1 is a whole hemisphere."""
    d, e, n = basis(lat0, lon0)
    w = np.sqrt(np.clip(1.0 - s * s * (u * u + v * v), 0.0, None))      # how far towards the viewer the point is
    p = s * u[..., None] * e + s * v[..., None] * n + w[..., None] * d  # the point on the globe
    return np.degrees(np.arcsin(np.clip(p[..., 1], -1.0, 1.0))), np.degrees(np.arctan2(p[..., 0], p[..., 2]))


def disc(n: int, lat0: float, lon0: float, rad_deg: float):
    """The pixel grid of an n x n pane picture: latitude, longitude of every pixel, the mask of the disc, and the
    pixel-centre coordinates -1..1 (row 0 is the top = north)."""
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    U, V = np.meshgrid(c, -c)
    lat, lon = inverse(U, V, lat0, lon0, float(np.sin(np.radians(rad_deg))))
    return lat, lon, U * U + V * V <= 1.0, c


def sample(field, glat, glon, lat, lon) -> np.ndarray:
    """Read a regular lat/lon grid at arbitrary places by bilinear interpolation (NaN outside the grid and where a
    neighbour is missing). ``glat`` may run north -> south or south -> north; a global grid wraps in longitude."""
    field = np.asarray(field, dtype=np.float32)
    dlat, dlon = glat[1] - glat[0], glon[1] - glon[0]
    fi = (lat - glat[0]) / dlat                                        # fractional row of each place
    fj = ((lon - glon[0]) % 360.0) / dlon                              # fractional column, measured eastward
    if glon.size * dlon > 359.9:                                       # a global grid: repeat the first column at the end
        field = np.concatenate([field, field[:, :1]], axis=1)
    out = map_coordinates(field, [fi, fj], order=1, mode="nearest")    # NaN neighbours give NaN
    outside = (fi < -0.5) | (fi > glat.size - 0.5) | (fj > field.shape[1] - 0.5)
    return np.where(outside, np.nan, out)
