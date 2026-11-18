"""gismaps.glass.render: a small ray caster for the globe and the glass panes.

For every pixel of the picture, follow a straight line (a ray) from the eye through that pixel and ask what it hits.
Two shapes are all we need.

A sphere (the globe, radius 1, centre at the origin). A point on the ray is o + t D (eye position plus t steps along
the direction). It lies on the sphere when its length is 1:

    (D.D) t^2 + 2 (o.D) t + (o.o - 1) = 0

a quadratic in t. A real solution means the ray hits the globe; the smaller t is the near side.

A flat pane, part of a plane through a centre point C with normal d (the stack axis). The ray meets the plane at
t = (C - o).d / (D.d); the hit point q = o + t D has pane coordinates u = (q - C).e, v = (q - C).n, which are the
coordinates of ``gismaps.glass.ortho``, so the pane picture is read there.

Order: the eye is outside the stack, so the globe is painted first, then the panes from the one nearest the globe
outwards, each laid over the picture with its opacity: new = old x (1 - alpha) + colour x alpha.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .ortho import basis


def camera(lat0, lon0, az=-33.0, el=16.0, dist=4.0, look=(0.0, 0.0, 0.0), fov=30.0, size=(1920, 1080), pp=None) -> dict:
    """A perspective camera in the frame of the stack.

    az, el   degrees from the stack axis d towards east e / towards north n (a negative az puts the camera on the
             west side, so the panes fan out to the right of the globe)
    dist     distance from the target point (globe radius = 1)
    look     the target point as (along e, along n, along d)
    fov      vertical field of view in degrees (30 = a mild telephoto lens, little distortion)
    pp       pixel where the target point appears (default: the middle of the image)
    """
    d, e, n = basis(lat0, lon0)
    W, H = size
    target = look[0] * e + look[1] * n + look[2] * d
    a, b = np.radians(az), np.radians(el)
    back = np.cos(b) * np.cos(a) * d + np.cos(b) * np.sin(a) * e + np.sin(b) * n     # from the target to the eye
    fwd = -back
    right = np.cross(fwd, n)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    focal = (H / 2.0) / np.tan(np.radians(fov) / 2.0)                  # focal length in pixels
    return {"pos": target + dist * back, "fwd": fwd, "right": right, "up": up, "F": focal, "W": W, "H": H,
            "pp": (W / 2.0, H / 2.0) if pp is None else pp, "basis": (d, e, n)}


def project(cam: dict, P):
    """3-D point(s) -> pixel (x, y): the inverse of casting a ray (used to place the numbered chips)."""
    rel = np.asarray(P, float) - cam["pos"]
    depth = rel @ cam["fwd"]
    return cam["pp"][0] + cam["F"] * (rel @ cam["right"]) / depth, cam["pp"][1] - cam["F"] * (rel @ cam["up"]) / depth


def pane_point(cam: dict, z, lift, su, sv):
    """A point on a pane in 3-D: z along the axis, lift along north, (su, sv) across the pane (globe radius = 1)."""
    d, e, n = cam["basis"]
    return (z + 0.25 * lift) * d + (sv + lift) * n + su * e            # a lifted pane also comes a little forward


def _read(tex, fx, fy):
    """Read a picture (h x w x channels) at fractional pixel positions, channel by channel (bilinear)."""
    return np.stack([map_coordinates(tex[..., k], [fy, fx], order=1, mode="nearest", output=np.float32)
                     for k in range(tex.shape[-1])], axis=-1)


def first_pane_z(rad_deg: float) -> float:
    """Distance of the first pane from the globe's centre: just clear of the surface."""
    return 1.25 if np.sin(np.radians(rad_deg)) > 0.9 else 1.10


def render_stack(panes, cam, relief, *, rad_deg, gap, z0=None, state=None, ss=2, globe_o=1.0, glass_o=1.0, half=1.16):
    """Ray-cast the globe and the panes. Returns an RGB image H x W x 3 with values 0..255.

    panes    list of pane dictionaries (nearest the globe first)
    relief   the globe's wrap-around picture (rows: 90 N -> 90 S, columns: 180 W -> 180 E)
    gap      distance between neighbouring panes along the axis (globe radius = 1)
    z0       distance of the first pane from the globe's centre
    state    per pane: {"o": opacity 0..1, "lift": lift along north, "glow": amber edge 0..1}
    ss       supersampling: render ss times larger, then shrink (smooth edges); 1 = fastest, 2 = best
    """
    f32 = np.float32
    d, e, n = (np.asarray(v, f32) for v in cam["basis"])
    fwd, right, up, o = (np.asarray(cam[k], f32) for k in ("fwd", "right", "up", "pos"))
    W, H = cam["W"], cam["H"]
    Ws, Hs = int(round(W * ss)), int(round(H * ss))
    s = np.sin(np.radians(rad_deg))                                    # disc radius of the cap
    z0 = first_pane_z(rad_deg) if z0 is None else z0
    state = state or [{} for _ in panes]
    xs = ((np.arange(Ws) + 0.5) / ss).astype(f32)                      # pixel-centre coordinates, in final pixels
    ys = ((np.arange(Hs) + 0.5) / ss).astype(f32)
    # the page: deep navy with a soft lighter centre
    gx, gy = (xs[None, :] / W - 0.45) / 0.75, (ys[:, None] / H - 0.45) / 0.70
    t = np.clip(np.sqrt(gx * gx + gy * gy) / 0.62, 0.0, 1.0)[..., None]
    c0, c1 = np.array([0x1d, 0x34, 0x63], f32) / 255.0, np.array([0x12, 0x22, 0x40], f32) / 255.0
    img = c0 * (1.0 - t) + c1 * t

    def rays(x0, x1, y0, y1):
        """Ray directions for the block of pixels [x0, x1) x [y0, y1) (supersampled pixels)."""
        dx = (xs[x0:x1] - f32(cam["pp"][0]))[None, :, None]
        dy = (ys[y0:y1] - f32(cam["pp"][1]))[:, None, None]
        return f32(cam["F"]) * fwd + dx * right - dy * up              # forward + sideways + up/down

    def block(points, pad=3):
        """The pixel block that contains the projected points."""
        px, py = project(cam, points)
        return (int(np.clip(np.floor(px.min() - pad) * ss, 0, Ws)), int(np.clip(np.ceil(px.max() + pad) * ss, 0, Ws)),
                int(np.clip(np.floor(py.min() - pad) * ss, 0, Hs)), int(np.clip(np.ceil(py.max() + pad) * ss, 0, Hs)))

    # ---- the globe: solve the quadratic for every pixel that could see it
    if globe_o > 0:
        cx, cy = project(cam, np.zeros(3))
        r_px = cam["F"] / np.sqrt(max(float(o @ o) - 1.0, 1e-6)) * 1.25    # generous radius of the globe's disc in pixels
        x0, x1 = int(np.clip((cx - r_px) * ss, 0, Ws)), int(np.clip((cx + r_px) * ss, 0, Ws))
        y0, y1 = int(np.clip((cy - r_px) * ss, 0, Hs)), int(np.clip((cy + r_px) * ss, 0, Hs))
        light = -0.55 * fwd + 0.50 * up - 0.60 * right                 # light from the upper left of the camera
        light /= np.linalg.norm(light)
        h, w = relief.shape[:2]
        for ya in range(y0, y1, 256):                                  # strips of 256 rows keep the memory small
            yb = min(ya + 256, y1)
            D = rays(x0, x1, ya, yb)
            A, B, C = (D * D).sum(-1), D @ o, float(o @ o) - 1.0       # the quadratic's coefficients
            disc = B * B - A * C                                       # positive = the ray hits the globe
            hit = (disc > 0) & (B < 0)                                 # ... and the globe is in front of the eye
            if not hit.any():
                continue
            tt = (-B[hit] - np.sqrt(disc[hit])) / A[hit]               # the nearer of the two solutions
            q = o + tt[:, None] * D[hit]                               # the points on the globe
            lat = np.degrees(np.arcsin(np.clip(q[:, 1], -1, 1)))
            lon = np.degrees(np.arctan2(q[:, 0], q[:, 2]))
            col = _read(relief, (lon + 180.0) / 360.0 * w - 0.5, (90.0 - lat) / 180.0 * h - 0.5)     # the skin's colour there
            shade = 0.55 + 0.50 * np.clip(q @ light, 0.0, 1.0)         # brighter where the surface faces the light
            blk = img[ya:yb, x0:x1]
            blk[hit] = blk[hit] * (1 - globe_o) + np.clip(col * shade[:, None], 0, 1) * globe_o

    # ---- the panes, from the one nearest the globe outwards
    order = []
    for k, P in enumerate(panes):
        st = state[k] if k < len(state) else {}
        if P is None or st.get("o", 1.0) <= 0.003:
            continue
        lift = st.get("lift", 0.0)
        order.append((z0 + k * gap + 0.25 * lift, k, P, st, lift))
    hs = half * s                                                      # half-size of the square glass slab
    rc = 0.07 * s                                                      # corner radius of the slab
    tint = np.array([0xc9, 0xd8, 0xea], f32) / 255.0                   # the colour of the glass
    for z, k, P, st, lift in sorted(order, key=lambda r: r[0]):
        op, glow = f32(st.get("o", 1.0)), f32(st.get("glow", 0.0))
        if float(o @ d) <= z:                                          # the eye must be in front of every pane
            continue
        C = z * d + f32(lift) * n                                      # the pane's centre
        corners = np.array([C + sx * hs * e + sy * hs * n for sx in (-1, 1) for sy in (-1, 1)])
        x0, x1, y0, y1 = block(corners)
        if x1 <= x0 or y1 <= y0:
            continue
        D = rays(x0, x1, y0, y1)
        tt = f32((C - o) @ d) / (D @ d)                                # where each ray meets the pane's plane
        q = tt[..., None] * D + (o - C)                                # hit point relative to the pane's centre
        u, v = q @ e, q @ n                                            # pane coordinates
        # rounded square: signed distance to its edge (negative inside)
        qx, qy = np.abs(u) - (hs - rc), np.abs(v) - (hs - rc)
        sd = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rc
        slab = sd < 0
        sheen = np.clip(0.5 + 0.4 * (v - u) / hs, 0, 1)                # glass is a little brighter towards the upper left
        a = (slab * (glass_o * op) * (0.09 + 0.08 * sheen))[..., None].astype(f32)    # the glass itself: a faint cool tint
        blk = img[y0:y1, x0:x1] * (1 - a) + tint * a
        in_disc = (u * u + v * v) <= s * s                             # pixels that see the round data picture
        n_tex = P["tex"].shape[0]
        pm = P["tex"].astype(f32)
        pm[..., :3] *= pm[..., 3:4]                                    # colour x alpha (no dark fringes when reading)
        got = _read(pm, (u[in_disc] / s + 1.0) / 2.0 * n_tex - 0.5, (1.0 - v[in_disc] / s) / 2.0 * n_tex - 0.5)
        blk[in_disc] = blk[in_disc] * (1 - got[:, 3:4] * op) + got[:, :3] * op
        rim_w = 0.012 * s * (1 + 2.0 * glow)                           # the thin bright edge of the slab
        a = ((slab & (sd > -rim_w)) * ((0.55 * glass_o + 0.45 * glow) * op))[..., None].astype(f32)
        rim = np.array([0xee, 0xf4, 0xff], f32) / 255.0 * (1 - glow) + np.array([0xff, 0xb8, 0x4d], f32) / 255.0 * glow
        img[y0:y1, x0:x1] = blk * (1 - a) + rim * a
    out = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if ss != 1:                                                        # shrink the supersampled picture
        from PIL import Image

        out = np.asarray(Image.fromarray(out).resize((W, H), Image.LANCZOS))
    return out
