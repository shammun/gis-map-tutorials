"""Check the built web page.

* Links: every relative link, image, poster and video of docs/*.html must exist; every Colab / GitHub link must point at
  a real file of this repository (case included). The Colab links of the README and of the notebooks are checked too.
* First load: the bytes a phone loads before the reader does anything must stay under 2 MB.
* With ``--shots <folder>``: the gallery is opened in a headless browser (playwright) at 360 x 800, 390 x 844,
  768 x 1024 and 1440 x 900. At each size: screenshots, no horizontal scrolling, no broken image, every video has a
  poster and loads its metadata on demand, tap targets are at least 44 px high, body text is at least 16 px.

    python tools/check_site.py [--shots <folder>]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog as cat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SIZES = ((360, 800), (390, 844), (768, 1024), (1440, 900))


def exact(p: Path) -> bool:
    """The file exists with exactly this spelling (GitHub is case-sensitive)."""
    return p.is_file() and p.name in [q.name for q in p.parent.iterdir()]


def check_links() -> int:
    bad = 0
    for page in [DOCS / "index.html", *sorted((DOCS / "tutorials").glob("*.html"))]:
        s = page.read_text(encoding="utf-8")
        refs = set(re.findall(r'(?:href|src|poster)="([^"#][^"]*)"', s))
        for part in re.findall(r'srcset="([^"]+)"', s):
            refs |= {x.strip().split(" ")[0] for x in part.split(",")}
        n_rel = n_ext = 0
        for r in sorted(refs):
            if r.startswith(("data:", "mailto:", "javascript:")):
                continue
            if r.startswith("http"):
                n_ext += 1
                for base in (cat.COLAB + "/", cat.BLOB + "/", cat.TREE + "/"):
                    if r.startswith(base):
                        p = ROOT / r[len(base):]
                        if not (exact(p) or (base == cat.TREE + "/" and p.is_dir())):
                            print(f"BAD repository link in {page.name}: {r}")
                            bad += 1
                continue
            n_rel += 1
            if not exact(page.parent / r.split("#")[0].split("?")[0]):
                print(f"MISSING in {page.name}: {r}")
                bad += 1
        print(f"{page.relative_to(DOCS)}: {n_rel} relative and {n_ext} external references checked")
    n = 0
    for t in [ROOT / "README.md", *sorted((ROOT / "tutorials").glob("*/*.ipynb"))]:
        for r in set(re.findall(re.escape(cat.COLAB) + r'/([\w/\.-]+\.ipynb)', t.read_text(encoding="utf-8"))):
            n += 1
            if not exact(ROOT / r) or not r.endswith("_colab.ipynb"):
                print(f"BAD Colab link in {t.name}: {r}")
                bad += 1
    print(f"README + notebooks: {n} Colab links checked")
    return bad


def first_load() -> int:
    """Bytes loaded before the reader does anything: the page, the video posters (they load with the page) and the
    images that are not lazy. Lazy images, the videos and the map of the city section load only on demand."""
    s = (DOCS / "index.html").read_text(encoding="utf-8")
    page = len(s.encode("utf-8"))
    eager = set(re.findall(r'poster="([^"]+)"', s)) | set(re.findall(r'<img(?![^>]*loading="lazy")[^>]*\ssrc="([^"]+)"', s))
    total = page + sum((DOCS / r).stat().st_size for r in eager)
    lazy = len(re.findall(r'<img[^>]*loading="lazy"', s))
    auto = len(re.findall(r"<video[^>]*autoplay", s))
    print(f"first load: {total / 1e6:.2f} MB (page {page / 1e3:.0f} kB + {len(eager)} poster images); {lazy} lazy images; "
          f"{auto} autoplaying videos")
    return int(total > 2_000_000) + auto


def shots(folder: Path) -> int:
    from playwright.sync_api import sync_playwright

    folder.mkdir(parents=True, exist_ok=True)
    bad = 0
    with sync_playwright() as p:
        browser = None
        for channel in ("msedge", "chrome", None):          # an installed browser first, then playwright's own
            try:
                browser = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
                break
            except Exception:                               # noqa: BLE001
                continue
        if browser is None:
            raise SystemExit("no browser found: python -m playwright install chromium")
        for width, height in SIZES:
            page = browser.new_page(viewport={"width": width, "height": height}, has_touch=width < 1000, is_mobile=width < 1000)
            failed = []
            # (a browser aborts the rest of an MP4 once it has the metadata; map tiles are outside this check)
            page.on("requestfailed", lambda r: None if r.url.endswith(".mp4") or "openstreetmap" in r.url else failed.append(r.url))
            page.on("response", lambda r: failed.append(f"{r.status} {r.url}") if r.status >= 400 and "openstreetmap" not in r.url else None)
            page.goto((DOCS / "index.html").resolve().as_uri())
            page.wait_for_load_state("load")
            page.screenshot(path=str(folder / f"index_{width}_top.png"))
            page.evaluate("document.querySelectorAll('img[loading=lazy]').forEach(i => i.loading = 'eager')")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2500)
            page.evaluate("window.scrollTo(0, 0)")
            broken = page.evaluate("[...document.images].filter(i => !(i.complete && i.naturalWidth > 0)).map(i => i.src)")
            overflow = page.evaluate("Math.max(document.documentElement.scrollWidth, window.innerWidth)") - width
            if overflow > 0:
                print("   too wide:", page.evaluate("""[...document.querySelectorAll('body *')].filter(e =>
                    e.getBoundingClientRect().right > %d + 1 && !e.closest("pre, .tablewrap, nav .links")).slice(0, 8)
                    .map(e => e.tagName + '.' + e.className + ' ' + Math.round(e.getBoundingClientRect().right))""" % width))
            videos = page.evaluate("""async () => {
                const out = [];
                for (const v of document.querySelectorAll('video')) {
                    const ok = v.controls && v.muted && v.playsInline && v.getAttribute('preload') === 'none' && !!v.poster && !v.autoplay;
                    v.preload = 'metadata'; v.load();
                    await new Promise(r => { v.onloadedmetadata = r; v.onerror = r; setTimeout(r, 8000); });
                    out.push([v.currentSrc.split('/').pop(), v.videoWidth, Math.round(v.duration || 0), ok]);
                }
                return out; }""")
            small = page.evaluate("""[...document.querySelectorAll('a.btn, button, nav a, summary, select, input[type=text], .under a, .tut .l a, .strip a, .checks label')]
                .filter(e => e.offsetParent !== null && e.getBoundingClientRect().height < 43.5)
                .map(e => e.tagName + ' "' + (e.textContent || '').trim().slice(0, 24) + '" ' + Math.round(e.getBoundingClientRect().height))""")
            body_px = page.evaluate("parseFloat(getComputedStyle(document.body).fontSize)")
            gutter = page.evaluate("Math.round(document.querySelector('main .wrap').getBoundingClientRect().left + parseFloat(getComputedStyle(document.querySelector('main .wrap')).paddingLeft))")
            cols = page.evaluate("getComputedStyle(document.querySelector('.card')).gridTemplateColumns.split(' ').length")
            leaflet = page.evaluate("!!window.L && document.querySelectorAll('#citymap .leaflet-container, #citymap.leaflet-container').length > 0")
            page.screenshot(path=str(folder / f"index_{width}.png"), full_page=True)
            bad_videos = [v for v in videos if not v[1] or not v[3]]
            print(f"{width} x {height}: page wider than the screen by {overflow} px | {len(videos)} videos (bad: {bad_videos}) | broken images "
                  f"{broken} | failed requests {failed} | tap targets under 44 px: {small[:5]} | body {body_px} px | side gutter {gutter} px | "
                  f"card columns {cols} | city map ready: {leaflet}")
            bad += len(broken) + len(failed) + (overflow > 0) + len(bad_videos) + len(small) + (body_px < 16) + (width < 720 and cols != 1)
            page.close()
        browser.close()
    return bad


if __name__ == "__main__":
    bad = check_links() + first_load()
    if "--shots" in sys.argv:
        bad += shots(Path(sys.argv[sys.argv.index("--shots") + 1]))
    print("OK" if not bad else f"{bad} problem(s)")
    sys.exit(1 if bad else 0)
