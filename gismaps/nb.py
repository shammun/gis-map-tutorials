"""gismaps.nb: the small helpers every tutorial uses (output folder, pictures and videos in a notebook, STOP checks,
PASS/FAIL verification)."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

from . import data as _data


def in_colab() -> bool:
    return "google.colab" in sys.modules


def out_dir(folder=None) -> Path:
    """Create and return the output folder: ``folder`` if given, else ``/content/out`` on Google Colab and ``./out``
    anywhere else."""
    out = Path(folder) if folder else Path("/content/out" if in_colab() else "out")
    out.mkdir(parents=True, exist_ok=True)
    return out


def load(name: str) -> dict:
    """One reduced input (``data/gis_styles/<name>.npz``): from disk, else downloaded, else rebuilt from its source."""
    return _data.load_npz(name)


def in_notebook() -> bool:
    return "ipykernel" in sys.modules


def stop_unless(ok: bool, message: str) -> None:
    """STOP check: halt here, with a plain message, when a setting cannot work."""
    if not ok:
        raise ValueError("STOP - " + message)


def _jpeg(im, width: int, quality: int = 84):
    """Show a PIL image inline as a JPEG that fills the width of the page up to ``width`` px (so it also fits a phone)."""
    import base64

    from IPython.display import HTML, display

    im.thumbnail((width * 2, width * 2))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    display(HTML(f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;max-width:{width}px;height:auto" alt="output picture">'))


def show(path, width: int = 960):
    """Display a PNG in a notebook (as a JPEG copy so the notebook stays small); a plain script only names the file."""
    if not in_notebook():
        print("wrote", path)
        return None
    from PIL import Image as PImage

    _jpeg(PImage.open(path).convert("RGB"), width, 86)


def picture(img, width: int = 760) -> None:
    """Show a picture given as a file or as an array (floats 0..1 or 8-bit); a plain script only names it."""
    import numpy as np
    from PIL import Image as PImage

    if isinstance(img, (str, Path)):
        im = PImage.open(img)
    else:
        a = np.asarray(img)
        im = PImage.fromarray(a if a.dtype == np.uint8 else (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
    if in_notebook():
        _jpeg(im.convert("RGB"), width, 78)
    else:
        print("picture:", img if isinstance(img, (str, Path)) else f"array {im.size}")


def strip(paths, width: int = 960):
    """Several frames side by side in a notebook."""
    if not in_notebook():
        return None
    from PIL import Image as PImage

    ims = [PImage.open(p).convert("RGB") for p in paths]
    sheet = PImage.new("RGB", (sum(i.width for i in ims), ims[0].height))
    x = 0
    for i in ims:
        sheet.paste(i, (x, 0))
        x += i.width
    _jpeg(sheet, width)


def video(path, width: int = 720, max_mb: float = 8.0):
    """Play an MP4 inline in a notebook (HTML5 video, the file embedded as base64, as wide as the page up to ``width``
    px). Files above ``max_mb`` are only named, so a notebook never grows by tens of megabytes. A plain script only
    names the file."""
    if path is None:
        return None
    path = Path(path)
    mb = path.stat().st_size / 1e6
    if not in_notebook() or mb > max_mb:
        print(f"movie: {path} ({mb:.1f} MB)")
        return None
    import base64

    from IPython.display import HTML, display

    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    display(HTML(f'<video controls muted loop playsinline style="width:100%;max-width:{width}px;height:auto" '
                 f'src="data:video/mp4;base64,{b64}"></video>'))


def download(folder, name: str = "outputs.zip") -> None:
    """On Google Colab: zip the pictures and movies of ``folder`` and offer the zip as one download. Anywhere else
    (your own machine, a plain script) it only says where the files are."""
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".mp4", ".gif"))
    if not in_colab():
        print(f"{len(files)} output file(s) in the folder {folder.as_posix()}/")
        return
    import zipfile

    from google.colab import files as colab_files

    target = folder / name
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as z:
        for p in files:
            z.write(p, p.name)
    print(f"{len(files)} files, {target.stat().st_size / 1e6:.1f} MB: the download starts in your browser")
    colab_files.download(str(target))


def image_size(path) -> tuple[int, int]:
    from PIL import Image as PImage

    with PImage.open(path) as im:
        return im.size


def verify(checks) -> None:
    """Print PASS or FAIL for every ``(what, ok)`` pair and raise if any failed (the last cell of every notebook)."""
    bad = 0
    for what, ok in checks:
        print("PASS" if ok else "FAIL", "-", what)
        bad += not ok
    if bad:
        raise AssertionError(f"{bad} check(s) failed")
    print(f"\nAll {len(list(checks))} checks passed.")
