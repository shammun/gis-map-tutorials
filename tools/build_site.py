"""Build the web page (GitHub Pages, served from ``docs/``). Mobile first.

* ``docs/index.html``              the gallery: one card per map / video, from ``tools/catalog.py``. Every card shows
  what the map shows, the data (source, licence), the numbered steps, the command, the code of the recipe (collapsed)
  and the links (Colab, page, notebook, .py). A "make your own city" section builds the Colab form values and the
  command line for any place picked on a map.
* ``docs/tutorials/<name>.html``   every executed notebook as a page (nbconvert, dark theme, a link bar on top)

    python tools/build_site.py             # both
    python tools/build_site.py --index     # only the gallery page

The media (``docs/media/``: posters as PNG + JPG, MP4, GIF) are made by running the recipes at full size and are
committed; this script adds the phone-size JPGs (800 px) and thumbnails and checks that every file a card needs is there.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog as cat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MEDIA = DOCS / "media"
LEAFLET = "https://unpkg.com/leaflet@1.9.4/dist/leaflet"

CSS = """
:root{--bg:#0b1020;--bg2:#0e1528;--panel:#111a30;--line:#24304f;--ink:#f1ede4;--soft:#d9d4c7;--muted:#9aa3b8;
--amber:#f2b33d;--orange:#e8833a;--teal:#2aa198;--serif:Georgia,"DejaVu Serif","Times New Roman",serif;
--sans:"Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,"DejaVu Sans",sans-serif;
--mono:Consolas,"SFMono-Regular",Menlo,"DejaVu Sans Mono",monospace}
*{box-sizing:border-box}
html{scroll-behavior:smooth;-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.6;
-webkit-font-smoothing:antialiased;overflow-wrap:break-word}
img,video{max-width:100%;height:auto}
a{color:var(--amber);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1240px;margin:0 auto;padding:0 16px}
nav{position:sticky;top:0;z-index:1000;background:rgba(11,16,32,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;align-items:center;gap:6px;min-height:48px}
nav .brand{font-family:var(--serif);font-weight:bold;font-size:17px;color:var(--ink);white-space:nowrap;padding:12px 8px 12px 0}
nav .links{display:flex;gap:2px;margin-left:auto;min-width:0;font-size:15px;overflow-x:auto;white-space:nowrap;scrollbar-width:none}
nav .links::-webkit-scrollbar{display:none}
nav .links a{color:var(--muted);padding:13px 10px;display:inline-block}
nav .links a:hover{color:var(--ink);text-decoration:none}
header.hero{padding:44px 0 36px;border-bottom:1px solid var(--line);
background:radial-gradient(1100px 420px at 78% -10%,rgba(242,179,61,.13),transparent 60%),
radial-gradient(900px 380px at 8% 0%,rgba(42,161,152,.10),transparent 60%)}
.eyebrow{font-size:12.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 14px}
h1{font-family:var(--serif);font-size:clamp(34px,6.2vw,72px);line-height:1.06;margin:0 0 18px;font-weight:bold;letter-spacing:-.01em}
h1 em{color:var(--orange);font-style:italic}
.lede{font-family:var(--serif);font-style:italic;font-size:clamp(18px,2.1vw,24px);color:var(--soft);max-width:860px;margin:0 0 16px}
.intro{max-width:820px;color:var(--soft);margin:0 0 12px}
.stats{display:flex;flex-wrap:wrap;gap:12px 26px;margin:24px 0 0;padding:0;list-style:none;color:var(--muted);font-size:15px}
.stats b{display:block;font-family:var(--serif);font-size:28px;color:var(--amber);font-weight:bold;line-height:1.1}
section{padding:44px 0 10px}
section h2{font-family:var(--serif);font-size:clamp(26px,3.6vw,42px);margin:0 0 10px;line-height:1.15}
section .sub{color:var(--soft);max-width:840px;margin:0 0 12px}
.strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(104px,1fr));gap:8px;margin:20px 0 4px}
.strip a{display:block;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--panel);aspect-ratio:16/10;min-height:44px}
.strip img{width:100%;height:100%;object-fit:cover;display:block}
.card{display:grid;grid-template-columns:minmax(0,1fr);gap:20px;padding:30px 0;border-top:1px solid var(--line)}
.card>div{min-width:0}
.media figure{margin:0 0 14px}
.media img.poster,.media video{width:100%;display:block;border-radius:10px;border:1px solid var(--line);background:#05070d}
.media figcaption{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 6px}
.media .under{display:flex;flex-wrap:wrap;gap:0 4px;font-size:14.5px}
.media .under a{display:inline-flex;align-items:center;min-height:44px;padding:0 10px 0 0}
.card.portrait .media{max-width:520px}
.kicker{font-size:12.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--amber);margin:2px 0 8px}
.card h3{font-family:var(--serif);font-size:clamp(24px,2.7vw,34px);line-height:1.15;margin:0 0 12px}
.card p.what{color:var(--soft);margin:0 0 16px}
.card h4{font-size:12.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:18px 0 8px;font-weight:600}
ul.data{margin:0;padding:0;list-style:none;font-size:15px;color:var(--soft)}
ul.data li{padding:0 0 8px}
ul.data .lic{display:block;color:var(--muted);font-size:14px}
ol.steps{margin:0;padding:0 0 0 22px;color:var(--soft);font-size:15.5px}
ol.steps li{padding:0 0 7px}
ol.steps li::marker{color:var(--teal);font-weight:700}
p.tools{font-size:15px;color:var(--soft);margin:0}
p.note{font-size:15px;color:var(--ink);background:#1b2440;border-left:4px solid var(--orange);padding:10px 12px;border-radius:4px;margin:14px 0 0}
ul.credits{margin:0;padding:0;list-style:none;font-size:14px;color:var(--muted)}
ul.credits li.author{color:var(--ink);font-weight:600;font-size:15px}
pre{background:#070b17;border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow-x:auto;font-size:13.5px;
line-height:1.5;color:#dfe6f5;font-family:var(--mono);margin:8px 0 0;max-width:100%;-webkit-overflow-scrolling:touch}
code{font-family:var(--mono);font-size:.92em;color:#e9dfc6;overflow-wrap:anywhere}
pre code{overflow-wrap:normal}
details{border:1px solid var(--line);border-radius:8px;margin:14px 0 0;background:var(--panel)}
details summary{cursor:pointer;min-height:44px;display:flex;flex-wrap:wrap;align-items:center;padding:8px 14px;font-size:15px;color:var(--ink);list-style:none}
details summary::-webkit-details-marker{display:none}
details summary::before{content:"▸";color:var(--amber);margin-right:10px}
details[open] summary::before{content:"▾"}
details pre{margin:0;border:0;border-top:1px solid var(--line);border-radius:0 0 8px 8px;max-height:70vh;overflow:auto}
.btns{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}
.btn{display:inline-flex;align-items:center;justify-content:center;min-height:46px;padding:0 18px;border-radius:999px;
border:1px solid var(--line);color:var(--ink);font-size:15px;background:var(--panel);font-family:var(--sans);cursor:pointer}
.btn:hover{border-color:var(--amber);text-decoration:none;background:#16213d}
.btn.primary{background:var(--amber);border-color:var(--amber);color:#1a1405;font-weight:700;font-size:16px;width:100%;min-height:52px}
.btn.primary:hover{background:#ffc655}
.tut{display:grid;grid-template-columns:minmax(0,1fr);gap:12px;margin:20px 0 0}
.tut article{border:1px solid var(--line);border-radius:10px;background:var(--panel);padding:16px 16px 6px}
.tut .n{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--amber)}
.tut h3{font-family:var(--serif);font-size:20px;margin:4px 0 6px;line-height:1.25}
.tut p{font-size:15px;color:var(--soft);margin:0 0 4px}
.tut .l{display:flex;flex-wrap:wrap;gap:0 6px;font-size:15px}
.tut .l a{display:inline-flex;align-items:center;min-height:44px;padding:0 12px 0 0}
.two{display:grid;grid-template-columns:minmax(0,1fr);gap:20px}
.two>div{min-width:0}
.two h3{font-family:var(--serif);font-size:21px;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-size:14.5px;margin-top:8px;min-width:560px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12.5px;letter-spacing:.1em;text-transform:uppercase}
td{color:var(--soft)}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
#citymap{width:100%;height:60vh;min-height:320px;border-radius:10px;border:1px solid var(--line);background:#0e1528;margin:12px 0;z-index:1}
.form{display:grid;grid-template-columns:minmax(0,1fr);gap:10px;margin:14px 0 0}
.form label{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);display:block;margin:0 0 4px}
.form select,.form input[type=text]{width:100%;min-height:46px;border-radius:8px;border:1px solid var(--line);background:#070b17;
color:var(--ink);font-size:16px;padding:0 12px;font-family:var(--sans)}
.row{display:flex;gap:8px}
.row input{flex:1;min-width:0}
.checks{display:flex;flex-wrap:wrap;gap:6px 18px}
.checks label{display:inline-flex;align-items:center;gap:10px;min-height:44px;font-size:16px;text-transform:none;letter-spacing:0;color:var(--ink);margin:0}
.checks input{width:22px;height:22px;accent-color:var(--amber)}
#chosen{font-size:17px;color:var(--ink);margin:10px 0 0;font-weight:600}
footer{margin-top:44px;border-top:1px solid var(--line);background:var(--bg2);padding:32px 0 48px;color:var(--muted);font-size:15px}
footer p{max-width:900px;margin:0 0 10px}
footer strong{color:var(--ink)}
@media (min-width:720px){
 .wrap{padding:0 28px}
 body{font-size:17px}
 header.hero{padding:76px 0 52px}
 section{padding:64px 0 20px}
 .card{grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:36px;padding:42px 0;align-items:start}
 .card.portrait{grid-template-columns:minmax(0,.8fr) minmax(0,1fr)}
 .card.portrait .media{max-width:none}
 .media{position:sticky;top:64px}
 .btn.primary{width:auto}
 .tut{grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}
 .two{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
 .strip{grid-template-columns:repeat(auto-fill,minmax(150px,1fr))}
 .form{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
 .form .wide{grid-column:1 / -1}
 table{min-width:0}
}
"""

E = html.escape


def web_images(stem: str) -> None:
    """Make the phone-size JPG (800 px wide) and the thumbnail (640 px) of ``<stem>.jpg`` if they are missing."""
    from PIL import Image

    src = MEDIA / f"{stem}.jpg"
    if not src.is_file():
        raise SystemExit(f"missing media file: docs/media/{stem}.jpg")
    for suffix, width, q in (("_800", 800, 82), ("_thumb", 640, 80)):
        out = MEDIA / f"{stem}{suffix}.jpg"
        if not out.is_file():
            im = Image.open(src).convert("RGB")
            im.thumbnail((width, 20 * width), Image.LANCZOS)
            im.save(out, "JPEG", quality=q, optimize=True, progressive=True)


def need(name: str) -> str:
    if not (MEDIA / name).is_file():
        raise SystemExit(f"missing media file: docs/media/{name}")
    return f"media/{name}"


def size_of(name: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(MEDIA / name) as im:
        return im.size


def links(stem: str) -> str:
    name = Path(stem).name
    return (f'<a class="btn primary" href="{cat.colab_url(stem)}" target="_blank" rel="noopener">Open in Colab</a>'
            f'<a class="btn" href="tutorials/{name}.html">Read the recipe</a>'
            f'<a class="btn" href="{cat.BLOB}/tutorials/{stem}.ipynb" target="_blank" rel="noopener">Notebook</a>'
            f'<a class="btn" href="{cat.BLOB}/tutorials/{stem}.py" target="_blank" rel="noopener">Python code</a>')


def code_block(rel: str, label: str) -> str:
    src = (ROOT / rel).read_text(encoding="utf-8")
    return (f'<details><summary>{E(label)}: <code style="margin-left:6px;min-width:0;word-break:break-all">{E(rel)}</code></summary>'
            f'<pre><code>{E(src)}</code></pre></details>')


def card(c: dict) -> str:
    m = c["media"]
    web_images(m)
    w, h = size_of(f"{m}_800.jpg")
    under = []
    figs = ""
    if not c.get("video_only"):
        figs += (f'<figure><figcaption>The map</figcaption><img class="poster" loading="lazy" width="{w}" height="{h}" '
                 f'src="{need(m + "_800.jpg")}" srcset="{need(m + "_800.jpg")} 800w, {need(m + ".jpg")} 1600w" '
                 f'sizes="(max-width: 720px) 100vw, 55vw" alt="{E(c["title"])}: the map"></figure>')
        under.append(f'<a href="{need(m + ".jpg")}">Map (JPG)</a>')
        if c.get("png", True):
            under.append(f'<a href="{need(m + ".png")}">Full size (PNG)</a>')
    videos = c.get("videos") or [(c["video"], "The video")]
    for stem, label in videos:
        figs += (f'<figure><figcaption>{E(label)}</figcaption><video controls muted playsinline preload="none" '
                 f'poster="{need(m + "_800.jpg")}" aria-label="{E(c["title"])}: {E(label.lower())}">'
                 f'<source src="{need(stem + ".mp4")}" type="video/mp4"></video></figure>')
        under.append(f'<a href="media/{stem}.mp4">{"Video" if len(videos) == 1 else E(label)} (MP4)</a>')
        if (MEDIA / f"{stem}.gif").is_file():
            under.append(f'<a href="media/{stem}.gif">GIF</a>')
    data = "".join(f'<li><a href="{E(u)}" target="_blank" rel="noopener">{E(n)}</a><span class="lic">Licence / credit: {E(lic)}</span></li>'
                   for n, u, lic in c["data"])
    steps = "".join(f"<li>{E(s.format(city=c.get('city', '')))}.</li>" for s in c["steps"])
    credits = [f'<li class="author">{E(cat.AUTHOR)}</li>'] + [f"<li>{E(n)}</li>" for n, _u, _l in c["data"]]
    if c.get("fan_mo"):
        credits.append(f"<li>{E(cat.FAN_MO)}</li>")
    code = code_block(f"tutorials/{c['tutorial']}.py", "The code of this recipe")
    note = ""
    if c.get("remotion"):
        note = ('<p class="note">The final render of this clip needs Node.js on your own machine (Remotion). Google Colab covers the '
                'data, every map layer and a pure-Python version of the clip.</p>')
        code += (f'<div class="btns" style="margin-top:10px"><a class="btn" href="{cat.TREE}/{c["remotion"]}" target="_blank" '
                 f'rel="noopener">Remotion source ({E(c["remotion"])}/)</a></div>')
        for rel, label in c.get("more_code", []):
            code += code_block(rel, label)
    if c.get("city"):
        note = '<p class="note">For any other place, see <a href="#your-city">Make your own city</a> below.</p>'
    return f'''
<article class="card{' portrait' if c.get('portrait') else ''}" id="{m}">
  <div class="media">{figs}<div class="under">{''.join(under)}</div></div>
  <div class="text">
    <p class="kicker">{E(c['kicker'])}</p>
    <h3>{E(c['title'])}</h3>
    <p class="what">{E(c['what'])}</p>
    <div class="btns" style="margin-top:0">{links(c['tutorial'])}</div>
    <h4>Data</h4>
    <ul class="data">{data}</ul>
    <h4>Steps to make it</h4>
    <ol class="steps">{steps}</ol>
    <h4>Run it on your own machine</h4>
    <pre>{E(cat.command(c))}</pre>
    {note}
    {code}
    <h4>Tools</h4>
    <p class="tools">{E(c['tools'])}</p>
    <h4>Credits</h4>
    <ul class="credits">{''.join(credits)}</ul>
  </div>
</article>'''


def strip(section: str) -> str:
    out = []
    for c in cat.CARDS:
        if c["section"] == section:
            web_images(c["media"])
            out.append(f'<a href="#{c["media"]}" title="{E(c["title"])}"><img loading="lazy" width="640" height="400" '
                       f'src="{need(c["media"] + "_thumb.jpg")}" alt="{E(c["title"])}"></a>')
    return f'<div class="strip">{"".join(out)}</div>'


def tutorial_list() -> str:
    out = []
    for stem, t in cat.TUTORIALS.items():
        name = Path(stem).name
        out.append(f'''<article><div class="n">{E(t["n"])}</div><h3>{E(t["title"])}</h3><p>{E(t["one"])}</p>
<div class="l"><a href="{cat.colab_url(stem)}" target="_blank" rel="noopener">Colab</a><a href="tutorials/{name}.html">Web page</a>
<a href="{cat.BLOB}/tutorials/{stem}.ipynb" target="_blank" rel="noopener">Notebook</a>
<a href="{cat.BLOB}/tutorials/{stem}.py" target="_blank" rel="noopener">.py</a></div></article>''')
    return f'<div class="tut">{"".join(out)}</div>'


def city_presets() -> dict:
    """The preset list of the notebook picker (``gismaps.lens.picker.PRESETS``), so the page and the notebook agree."""
    sys.path.insert(0, str(ROOT))
    try:
        from gismaps.lens import picker

        return {k: [round(v[0], 4), round(v[1], 4)] for k, v in picker.PRESETS.items()}
    except Exception as err:  # noqa: BLE001
        print("picker presets not available:", err)
        return {"Brisbane": [-27.47, 153.025], "Dhaka": [23.7233, 90.402], "Richmond (Virginia)": [37.535, -77.446]}


CITY_JS = r"""
(function(){
  var PRESETS = __PRESETS__;
  var state = {name:"Dhaka", lat:PRESETS["Dhaka"][0], lon:PRESETS["Dhaka"][1], byName:true};
  var $ = function(id){ return document.getElementById(id); };
  var sel = $("cityselect");
  Object.keys(PRESETS).forEach(function(k){ var o=document.createElement("option"); o.value=k; o.textContent=k; sel.appendChild(o); });
  sel.value = "Dhaka";
  var map = null, marker = null;
  function ensureMap(){
    if (map || !window.L) return;
    map = L.map("citymap", {scrollWheelZoom:false, tap:true}).setView([state.lat, state.lon], 11);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {maxZoom:18,
      attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
    marker = L.marker([state.lat, state.lon]).addTo(map);
    map.on("click", function(e){ pick(e.latlng.lat, e.latlng.lng); });
  }
  var last = 0;
  function nominatim(url){            // at most one request per second, as the service asks
    var wait = Math.max(0, 1100 - (Date.now() - last)); last = Date.now() + wait;
    return new Promise(function(res){ setTimeout(res, wait); }).then(function(){
      return fetch(url, {headers:{"Accept":"application/json"}}); }).then(function(r){ return r.json(); });
  }
  function show(){
    if (marker){ marker.setLatLng([state.lat, state.lon]); }
    $("chosen").textContent = "Selected: " + state.name + " (" + state.lat.toFixed(4) + ", " + state.lon.toFixed(4) + ")";
    var outs = ["poster","video","gif"].filter(function(k){ return $("o_"+k).checked; });
    if (!outs.length){ $("o_poster").checked = true; outs = ["poster"]; }
    var full = $("o_full").checked, short = state.name.split(",")[0];
    var preset = /^(Brisbane|Dhaka|Richmond)/.test(state.name) && state.byName;
    var place = state.byName ? short : "";
    var ll = state.byName ? "" : state.lat.toFixed(4) + ", " + state.lon.toFixed(4);
    $("colabvals").textContent = 'PLACE = "' + place + '"\nLAT_LON = "' + ll + '"\nMAKE_POSTER = ' + (outs.indexOf("poster")>=0?"True":"False") +
      "\nMAKE_VIDEO = " + (outs.indexOf("video")>=0?"True":"False") + "\nMAKE_GIF = " + (outs.indexOf("gif")>=0?"True":"False") +
      "\nFULL_SIZE = " + (full?"True":"False");
    $("clicmd").textContent = "python -m gismaps.lens city " + (preset ? "--preset " + short.split(" ")[0].toLowerCase() :
      (state.byName ? '--place "' + short + '"' : "--lat " + state.lat.toFixed(4) + " --lon " + state.lon.toFixed(4))) +
      " --outputs " + outs.join(",") + (full ? " --full" : "");
  }
  function pick(lat, lon){
    state.lat = lat; state.lon = lon; state.byName = false; state.name = "finding the place name ...";
    show();
    nominatim("https://nominatim.openstreetmap.org/reverse?format=jsonv2&zoom=10&lat=" + lat + "&lon=" + lon)
      .then(function(j){ var a = j.address || {};
        state.name = [(a.city || a.town || a.village || a.municipality || a.county || j.name || "this place"), a.country].filter(Boolean).join(", "); show(); })
      .catch(function(){ state.name = "this place"; show(); });
  }
  sel.addEventListener("change", function(){ var p = PRESETS[sel.value]; state = {name:sel.value, lat:p[0], lon:p[1], byName:true};
    ensureMap(); if (map) map.setView(p, 11); show(); });
  $("searchbtn").addEventListener("click", function(){
    var q = $("search").value.trim(); if (!q) return;
    $("chosen").textContent = "Searching ...";
    nominatim("https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=" + encodeURIComponent(q)).then(function(j){
      if (!j.length){ $("chosen").textContent = 'Nothing found for "' + q + '".'; return; }
      state = {name:j[0].display_name.split(",").slice(0,1).concat(j[0].display_name.split(",").slice(-1)).map(function(s){return s.trim();}).join(", "),
               lat:parseFloat(j[0].lat), lon:parseFloat(j[0].lon), byName:false};
      ensureMap(); if (map) map.setView([state.lat, state.lon], 11); show(); })
      .catch(function(){ $("chosen").textContent = "The search service did not answer; tap the map instead."; });
  });
  $("search").addEventListener("keydown", function(e){ if (e.key === "Enter"){ e.preventDefault(); $("searchbtn").click(); } });
  ["o_poster","o_video","o_gif","o_full"].forEach(function(id){ $(id).addEventListener("change", show); });
  document.querySelectorAll("[data-copy]").forEach(function(b){ b.addEventListener("click", function(){
    var t = $(b.getAttribute("data-copy")).textContent;
    (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(function(){ b.textContent = "Copied"; },
      function(){ var r = document.createRange(); r.selectNodeContents($(b.getAttribute("data-copy"))); var s = getSelection(); s.removeAllRanges(); s.addRange(r); b.textContent = "Selected: copy it"; });
    setTimeout(function(){ b.textContent = "Copy"; }, 2500); }); });
  // the map (and its tiles) load only when the section comes near the screen: nothing is fetched on the first screen
  var holder = $("citymap");
  function load(){
    if (window.L){ ensureMap(); return; }
    var css = document.createElement("link"); css.rel = "stylesheet"; css.href = "__LEAFLET__.css"; document.head.appendChild(css);
    var js = document.createElement("script"); js.src = "__LEAFLET__.js"; js.onload = ensureMap; document.head.appendChild(js);
  }
  if ("IntersectionObserver" in window){
    var io = new IntersectionObserver(function(es){ if (es.some(function(e){ return e.isIntersecting; })){ io.disconnect(); load(); } }, {rootMargin:"600px"});
    io.observe(holder);
  } else { load(); }
  show();
})();
"""


def your_city() -> str:
    stem = "lens_maps/01_lens_city_osm"
    return f'''
<section id="your-city"><div class="wrap">
<h2>Make your own city</h2>
<p class="sub">The lens-map recipe works for any place. Choose one here (a city from the list, a search, or a tap on the map), tick what you want, and this page writes the values for the Colab form and the command for your own machine. The map itself is drawn by the recipe, in Colab or on your machine: this page only helps you fill it in.</p>
<div class="form">
  <div><label for="cityselect">A city from the list</label><select id="cityselect"></select></div>
  <div><label for="search">Or search a place name</label><div class="row"><input type="text" id="search" placeholder="e.g. Chittagong" autocomplete="off"><button class="btn" id="searchbtn" type="button">Search</button></div></div>
</div>
<p class="sub" style="margin-top:12px">Or tap the map: the point you tap becomes the centre of the poster.</p>
<div id="citymap" aria-label="Map to choose the centre of the poster"></div>
<p id="chosen" aria-live="polite"></p>
<div class="form">
  <div class="wide"><label>Outputs</label><div class="checks">
    <label><input type="checkbox" id="o_poster" checked> Poster</label>
    <label><input type="checkbox" id="o_video"> Video</label>
    <label><input type="checkbox" id="o_gif"> GIF</label>
    <label><input type="checkbox" id="o_full"> Full size (slower)</label></div></div>
</div>
<h4 style="font-size:12.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:18px 0 0">1 · For Google Colab</h4>
<p class="sub" style="margin:6px 0 0">Colab cannot take values from a link. Open the notebook, find the form cell of step 12 ("The same poster for any place"), fill it with these values (or paste these lines over its first six lines), then Runtime ▸ Run all.</p>
<pre id="colabvals"></pre>
<div class="btns"><button class="btn" type="button" data-copy="colabvals">Copy</button>
<a class="btn primary" href="{cat.colab_url(stem)}" target="_blank" rel="noopener">Open in Colab</a></div>
<h4 style="font-size:12.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:22px 0 0">2 · Or on your own machine</h4>
<pre id="clicmd"></pre>
<div class="btns"><button class="btn" type="button" data-copy="clicmd">Copy</button></div>
<p class="sub" style="margin-top:14px;font-size:14.5px;color:var(--muted)">Place names and the map come from OpenStreetMap (© OpenStreetMap contributors, ODbL) through its Nominatim service and tile server; a new place takes a few minutes because the recipe downloads the map data politely, one query at a time.</p>
</div></section>
<script>{CITY_JS.replace("__PRESETS__", json.dumps(city_presets(), ensure_ascii=False)).replace("__LEAFLET__", LEAFLET)}</script>'''


def index() -> str:
    styles = "".join(card(c) for c in cat.CARDS if c["section"] == "styles")
    lens = "".join(card(c) for c in cat.CARDS if c["section"] == "lens")
    globe = "".join(card(c) for c in cat.CARDS if c["section"] == "globe")
    n_maps = sum(1 for c in cat.CARDS if not c.get("video_only"))
    n_videos = sum(len(c.get("videos") or [1]) for c in cat.CARDS)
    globe_section = "" if not globe else f'''
<section id="globe"><div class="wrap">
<h2>Glass-pane globe stack</h2>
<p class="sub">A globe with flat panes of glass stacked above one region, one weather field per pane, so several fields of the same place and time can be read together.</p>
{globe}
</div></section>'''
    globe_nav = '<a href="#globe">Glass stack</a>' if globe else ""
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GIS map tutorials · Shammunul Islam</title>
<meta name="description" content="Maps and videos of storms, oceans, rivers and cities, each with the recipe that makes it: data, steps and Python code, as a web page, a notebook and a Google Colab notebook. Maps created by Shammunul Islam.">
<meta name="author" content="Shammunul Islam">
<meta property="og:title" content="GIS map tutorials">
<meta property="og:description" content="Maps and videos of storms, oceans, rivers and cities, each with the recipe that makes it.">
<meta property="og:image" content="{cat.PAGES}/media/glow_season.jpg">
<style>{CSS}</style>
</head>
<body>
<nav><div class="wrap"><a class="brand" href="#top">GIS map tutorials</a>
<div class="links"><a href="#styles">Maps</a><a href="#lens">Lens maps</a><a href="#your-city">Your city</a>{globe_nav}<a href="#tutorials">Recipes</a><a href="#run">Run</a><a href="{cat.GITHUB}" target="_blank" rel="noopener">GitHub</a></div></div></nav>

<header class="hero" id="top"><div class="wrap">
<p class="eyebrow">Maps created by Shammunul Islam · Python · open data</p>
<h1>Maps that <em>explain</em>, <br>and how to make them.</h1>
<p class="lede">{n_maps} maps and {n_videos} videos of storms, oceans, rivers and cities. Each one comes with the recipe that makes it: the data, the steps and the code.</p>
<p class="intro">Under every map you find what it shows, the data with its source and licence, the numbered steps, the command to run, the code, and a button that opens the recipe in Google Colab, where it runs with no key and no account, also from a phone.</p>
<ul class="stats"><li><b>{n_maps}</b>maps</li><li><b>{n_videos}</b>videos</li><li><b>{len(cat.TUTORIALS)}</b>recipes</li><li><b>0</b>keys needed</li></ul>
</div></header>

<main>
<section id="styles"><div class="wrap">
<h2>Seven maps of weather, ocean and land</h2>
<p class="sub">Each is a poster and a short video made by one recipe. Tap a picture below to jump to it.</p>
{strip("styles")}
{styles}
</div></section>

<section id="lens"><div class="wrap">
<h2>Lens maps</h2>
<p class="sub">One quiet, dark base map with round windows onto other layers of the same place, drawn on the same pixel grid, so each lens lines up with the map around it. For categories (land use), overlapping lenses may mix; for data fields with a colour bar they never mix.</p>
{strip("lens")}
{lens}
</div></section>
{your_city()}
{globe_section}
<section id="tutorials"><div class="wrap">
<h2>The recipes</h2>
<p class="sub">Each recipe is one Python script, built into an executed notebook, a Colab notebook and a web page. Every notebook has one settings cell with checks that stop early with a plain message, a coloured banner on every step, a last cell that verifies the outputs (PASS or FAIL), and on Colab a download of the results.</p>
{tutorial_list()}
</div></section>

<section id="run"><div class="wrap">
<h2>Run it yourself</h2>
<div class="two">
<div><h3>On Google Colab (also on a phone)</h3>
<p class="sub">Tap "Open in Colab" under any map, then Runtime ▸ Run all. The first cell installs the package; the data are downloaded from the repository. The default mode is quick (a small poster and a short preview clip, a few minutes); the settings cell has the switch for full size.</p>
<pre>%pip install -q git+{cat.GITHUB}</pre></div>
<div><h3>On your own machine</h3>
<p class="sub">Python 3.10 or newer. The videos use the ffmpeg on your PATH, or the one the package brings.</p>
<pre>git clone {cat.GITHUB}.git
cd gis-map-tutorials
python -m venv .venv
.venv/bin/pip install -e .      # Windows: .venv\\Scripts\\pip
.venv/bin/python tutorials/gis_styles/01_glow_season.py --full</pre></div>
</div>
</div></section>

<section id="data"><div class="wrap">
<h2>Data and credits</h2>
<p class="sub">Small reduced copies of every input ship with the repository, so the recipes run in minutes; each recipe also shows the route to the original public source. Please keep the credit lines when you reuse a map.</p>
<div class="tablewrap"><table>
<tr><th>Data</th><th>Source</th><th>Credit / licence</th></tr>
<tr><td>Tropical cyclone tracks</td><td>IBTrACS v04r01, NOAA NCEI</td><td>Public domain; Knapp et al. (2010)</td></tr>
<tr><td>ERA5 reanalysis</td><td>Copernicus Climate Change Service (C3S) / ECMWF; hourly fields through the public ARCO-ERA5 store</td><td>Contains modified Copernicus Climate Change Service information; Hersbach et al. (2020)</td></tr>
<tr><td>Relief</td><td>ETOPO 2022, NOAA NCEI</td><td>Public domain</td></tr>
<tr><td>Sea-surface temperature and sea ice</td><td>NOAA OISST v2.1, NOAA NCEI; Sea Ice Index v4, NSIDC</td><td>Public domain; Huang et al. (2021); Fetterer et al. (2017)</td></tr>
<tr><td>Ocean temperature at depth</td><td>World Ocean Atlas 2023, NOAA NCEI</td><td>Public domain; Locarnini et al. (2024)</td></tr>
<tr><td>Sentinel-2 imagery</td><td>ESA / Copernicus, read from Element 84 Earth Search (AWS open data)</td><td>Contains modified Copernicus Sentinel data 2019-2026</td></tr>
<tr><td>City layers</td><td>OpenStreetMap through the Overpass API</td><td>© OpenStreetMap contributors, Open Database Licence (ODbL) 1.0</td></tr>
{cat.EXTRA_DATA_ROWS if hasattr(cat, "EXTRA_DATA_ROWS") else ""}
<tr><td>Land, coast and borders</td><td>Natural Earth</td><td>Public domain</td></tr>
<tr><td>Colour maps</td><td>cmocean</td><td>Thyng et al. (2016)</td></tr>
</table></div>
</div></section>
</main>

<footer id="about"><div class="wrap">
<p><strong>{E(cat.AUTHOR)}.</strong> All maps, videos and code on this page are my own work on open data. The lens maps follow a style by Fan Mo ({E(cat.FAN_MO.replace("Style after Fan Mo, ", ""))}, "Borgesian map"): the style was reproduced with my own data and code, and nothing was traced or copied.</p>
<p>Code: MIT licence. Data: each source keeps its own licence (table above); OpenStreetMap-derived files are under the ODbL. Maps and videos may be reused with the credit lines they carry.</p>
<p><a href="{cat.GITHUB}" style="display:inline-flex;align-items:center;min-height:44px">github.com/{cat.REPO}</a></p>
</div></footer>
</body>
</html>
'''


BAR = '''<div style="position:sticky;top:0;z-index:1000;background:#0b1020;border-bottom:1px solid #24304f;padding:4px 12px;
font:15px 'Segoe UI',system-ui,sans-serif;display:flex;flex-wrap:wrap;gap:0 14px;align-items:center">
<a href="../index.html" style="color:#f1ede4;font-family:Georgia,serif;font-weight:bold;text-decoration:none;display:inline-flex;align-items:center;min-height:44px">← GIS map tutorials</a>
<span style="margin-left:auto;display:flex;flex-wrap:wrap;gap:0 16px">
<a style="color:#f2b33d;display:inline-flex;align-items:center;min-height:44px;font-weight:700" href="{colab}">Open in Colab</a><a style="color:#f2b33d;display:inline-flex;align-items:center;min-height:44px" href="{nb}">Notebook</a><a style="color:#f2b33d;display:inline-flex;align-items:center;min-height:44px" href="{py}">Python code</a></span></div>
<style>img,video{{max-width:100%;height:auto}} pre{{overflow-x:auto;max-width:100%}} .jp-Cell{{max-width:100vw}}
@media (max-width:720px){{ body{{font-size:16px}} .jp-InputPrompt,.jp-OutputPrompt{{display:none}} .jp-Cell{{padding:4px 8px}}
 .jp-RenderedHTMLCommon table{{display:block;overflow-x:auto}} }}</style>'''


def tutorial_pages() -> None:
    out = DOCS / "tutorials"
    out.mkdir(parents=True, exist_ok=True)
    for stem, t in cat.TUTORIALS.items():
        nb = ROOT / "tutorials" / f"{stem}.ipynb"
        name = Path(stem).name
        subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", "--template", "lab", "--theme", "dark",
                        "--output-dir", str(out), "--output", f"{name}.html", str(nb)], check=True)
        f = out / f"{name}.html"
        s = f.read_text(encoding="utf-8")
        # no external scripts: the pages hold no formulas (MathJax) and no widgets (require.js)
        s = re.sub(r'<script src="https?://[^"]+"[^>]*>\s*</script>', "", s)
        s = re.sub(r"/\*[^*]*auto-generated\s+by[^*]*\*/", "", s)       # comments in the page template's bundled styles
        s = re.sub(r"is generated\s+by Mermaid", "comes from Mermaid", s)
        s = re.sub(r"<title>.*?</title>", f"<title>{E(t['n'])} · {E(t['title'])} · GIS map tutorials</title>", s, count=1, flags=re.S)
        if 'name="viewport"' not in s:
            s = s.replace("<head>", '<head><meta name="viewport" content="width=device-width, initial-scale=1">', 1)
        bar = BAR.format(colab=cat.colab_url(stem), nb=f"{cat.BLOB}/tutorials/{stem}.ipynb", py=f"{cat.BLOB}/tutorials/{stem}.py")
        s = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + "\n" + bar, s, count=1)
        f.write_text(s, encoding="utf-8", newline="\n")
        print(f"docs/tutorials/{name}.html  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(index(), encoding="utf-8", newline="\n")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print("docs/index.html", f"{(DOCS / 'index.html').stat().st_size / 1e3:.0f} kB")
    if "--index" not in sys.argv:
        tutorial_pages()
