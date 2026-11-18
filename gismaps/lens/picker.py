"""Lens map: choose the place of your poster in a notebook, with a finger or a mouse.

Three ways, each of which ends with one line ``Selected: <city>, <country> (lat, lon)``:

    from gismaps.lens import picker
    picker.resolve("Tokyo")                    # a name from picker.PRESETS, or any city name you type
    picker.resolve("", "48.8566, 2.3522")      # a point as "lat, lon"
    picker.pick_on_map()                       # tap a map; the choice lands in picker.SELECTED
    picker.panel()                             # list + text boxes + check boxes + a Make button (needs ipywidgets)

``resolve`` needs nothing but Python. ``pick_on_map`` shows a Leaflet map (OpenStreetMap tiles): on Google Colab a tap
is sent to Python; elsewhere it uses ipyleaflet when that is installed, and otherwise the tap shows "lat, lon" to copy
into the text box. Every control is a tap target: nothing here needs a mouse pointer to hover.
Place names: (c) OpenStreetMap contributors, through Nominatim (one request per second, cached).
"""
from __future__ import annotations

import json

from . import make as _make

# name -> (lat, lon) of the city centre. The first three are the hand-tuned cities (drawn from the data shipped with
# the repository, no download); every other place is downloaded from OpenStreetMap when its poster is made.
PRESETS: dict[str, tuple[float, float]] = {
    "Brisbane": (-27.4705, 153.0235), "Dhaka": (23.7233, 90.4020), "Richmond (Virginia)": (37.5350, -77.4460),
    "Chittagong": (22.3384, 91.8317), "Kolkata": (22.5726, 88.3639), "Delhi": (28.6328, 77.2197),
    "Mumbai": (18.9388, 72.8354), "Karachi": (24.8607, 67.0011), "Tokyo": (35.6812, 139.7671),
    "Seoul": (37.5665, 126.9780), "Singapore": (1.2903, 103.8519), "Jakarta": (-6.1754, 106.8272),
    "Istanbul": (41.0082, 28.9784), "London": (51.5074, -0.1278), "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050), "Rome": (41.8933, 12.4829), "Cairo": (30.0444, 31.2357),
    "Nairobi": (-1.2864, 36.8172), "Lagos": (6.4550, 3.3941), "Cape Town": (-33.9249, 18.4241),
    "New York": (40.7128, -74.0060), "Washington DC": (38.8951, -77.0364), "Toronto": (43.6532, -79.3832),
    "Mexico City": (19.4326, -99.1332), "São Paulo": (-23.5505, -46.6333), "Buenos Aires": (-34.6037, -58.3816),
    "Sydney": (-33.8688, 151.2093), "Melbourne": (-37.8136, 144.9631),
}
SELECTED: dict | None = None  # the last choice: dict(name, country, lat, lon, preset)
_PANEL: dict = {}  # the widgets of the last panel (a tap on the map fills its "lat, lon" box)
_CALLBACK = "gismaps.lens.pick"


def _select(where: dict) -> dict:
    """Keep a located place as the current choice and print the ``Selected:`` line."""
    global SELECTED
    SELECTED = {k: where[k] for k in ("name", "country", "lat", "lon", "preset")}
    print(_line(SELECTED))
    return SELECTED


def _line(sel: dict) -> str:
    place = f"{sel['name']}, {sel['country']}" if sel["country"] else sel["name"]
    return f"Selected: {place} ({sel['lat']:.4f}, {sel['lon']:.4f})"


def resolve(place: str = "", lat_lon: str = "") -> dict:
    """The no-widget way to choose a place: a name from ``PRESETS``, any city name, or a "lat, lon" text (the text
    wins when both are given). Prints ``Selected: <city>, <country> (lat, lon)`` and returns
    dict(name, country, lat, lon, preset); ``preset`` is the key of a hand-tuned city ("brisbane", "dhaka",
    "richmond") or ``None``. Pass the result on with ``make_city(place=sel["name"], lat=sel["lat"], lon=sel["lon"])``."""
    place, lat_lon = str(place or "").strip(), str(lat_lon or "").strip()
    if lat_lon:
        point = _make.parse_lat_lon(lat_lon)
        if not point:
            raise ValueError(f'could not read {lat_lon!r} as "lat, lon"; write it like 23.72, 90.40')
        return _select(_make.locate(lat=point[0], lon=point[1]))
    if not place:
        raise ValueError("give a city name or a 'lat, lon' text")
    known = {k.lower(): k for k in PRESETS}
    if place.lower() in known:
        name = known[place.lower()]
        if _make.preset_key(name):
            return _select(_make.locate(preset=name))
        return _select(_make.locate(place=name, lat=PRESETS[name][0], lon=PRESETS[name][1]))
    return _select(_make.locate(place=place))


# ---------------------------------------------------------------------------------------------------- the map
_MAP_HTML = """
<style>
#__ID__ .leaflet-bar a{width:44px;height:44px;line-height:44px;font-size:24px}
#__ID___msg{font:16px/1.5 system-ui,sans-serif;padding:10px 4px;min-height:24px}
</style>
<div id="__ID__" style="width:100%;height:60vh;min-height:340px;border-radius:8px"></div>
<div id="__ID___msg">Tap the map to choose the centre of your poster. Pinch or use + and - to zoom.</div>
<script>
(function () {
  var id = "__ID__", start = [__LAT__, __LON__], callback = "__CALLBACK__";
  function init() {
    var map = L.map(id, {zoomControl: true, touchZoom: true, tap: true}).setView(start, 11);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
    var marker = null, msg = document.getElementById(id + "_msg");
    map.on("click", function (e) {
      var lat = e.latlng.lat.toFixed(5), lon = e.latlng.lng.toFixed(5), text = lat + ", " + lon;
      if (marker) { map.removeLayer(marker); }
      marker = L.marker(e.latlng).addTo(map).bindPopup(text).openPopup();
      var colab = callback && window.google && google.colab && google.colab.kernel;
      if (!colab) { msg.textContent = 'Copy this into the "lat, lon" box: ' + text; return; }
      msg.textContent = "Looking up " + text + " ...";
      google.colab.kernel.invokeFunction(callback, [parseFloat(lat), parseFloat(lon)], {}).then(function (r) {
        var j = r && r.data && r.data["application/json"];
        var line = (j && j.text) || ("Selected: " + text);
        msg.textContent = line;
        marker.setPopupContent(line);
      }).catch(function (err) {
        msg.textContent = 'Copy this into the "lat, lon" box: ' + text + " (" + err + ")";
      });
    });
    setTimeout(function () { map.invalidateSize(); }, 300);
  }
  var css = document.createElement("link");
  css.rel = "stylesheet";
  css.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
  document.head.appendChild(css);
  if (window.L && window.L.map) { init(); return; }
  var js = document.createElement("script");
  js.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
  js.onload = init;
  document.head.appendChild(js);
})();
</script>
"""
_count = [0]


def map_html(start=(23.72, 90.40), callback: str = "") -> str:
    """The Leaflet map as one piece of HTML (Leaflet 1.9.4 from unpkg, OpenStreetMap tiles with their credit, full
    width, 60 % of the window height, pinch zoom). A tap puts a marker with "lat, lon" on the map; with ``callback``
    (the name of a function registered on Google Colab) the point is also sent to Python."""
    _count[0] += 1
    return (_MAP_HTML.replace("__ID__", f"lensmap{_count[0]}").replace("__LAT__", f"{float(start[0]):.5f}")
            .replace("__LON__", f"{float(start[1]):.5f}").replace("__CALLBACK__", callback))


def _picked(lat: float, lon: float) -> dict:
    """A tap on the map: look the point up, keep it in ``SELECTED``, print the line, fill the panel's box."""
    sel = _select(_make.locate(lat=float(lat), lon=float(lon)))
    if _PANEL.get("lat_lon") is not None:
        _PANEL["lat_lon"].value = f"{sel['lat']:.5f}, {sel['lon']:.5f}"
    return sel


def pick_on_map(start=(23.72, 90.40)) -> None:
    """Show a map; a tap (or click) chooses the centre of the poster.

    Google Colab: the tap calls Python, which stores the choice in ``picker.SELECTED``, looks up the city and country
    and prints ``Selected: <city>, <country> (lat, lon)`` (the line also appears under the map).
    Elsewhere: an ipyleaflet map that does the same when ipyleaflet is installed; otherwise the plain map, where a
    tap shows "lat, lon" to copy into the text box of ``panel()`` or into ``resolve("", "lat, lon")``."""
    from .. import nb

    if not nb.in_notebook() and not nb.in_colab():
        print('pick_on_map needs a notebook. In a plain script use picker.resolve("city name") or '
              'picker.resolve("", "lat, lon").')
        return
    from IPython.display import HTML, JSON, display

    if nb.in_colab():
        from google.colab import output

        def on_tap(lat, lon):
            return JSON({"text": _line(_picked(lat, lon))})

        output.register_callback(_CALLBACK, on_tap)
        display(HTML(map_html(start, _CALLBACK)))
        return
    try:
        import ipyleaflet
        import ipywidgets as widgets
    except ImportError:
        display(HTML(map_html(start)))
        return
    m = ipyleaflet.Map(center=tuple(start), zoom=11, scroll_wheel_zoom=True, layout=widgets.Layout(width="100%", height="60vh", min_height="340px"))
    out = widgets.Output()
    marker = ipyleaflet.Marker(location=tuple(start), draggable=False)

    def on_tap(**event):
        if event.get("type") != "click":
            return
        lat, lon = event["coordinates"]
        marker.location = (lat, lon)
        if marker not in m.layers:
            m.add(marker)
        with out:
            out.clear_output()
            _picked(lat, lon)

    m.on_interaction(on_tap)
    with out:
        print("Tap the map to choose the centre of your poster.")
    display(widgets.VBox([m, out]))


# -------------------------------------------------------------------------------------------------- the panel
def panel(out_dir=None) -> None:
    """A small form for a notebook: a list of well-known cities, a box for any city name, a "lat, lon" box (a tap on
    ``pick_on_map()`` fills it), check boxes for poster / video / GIF, FAST or FULL, and a Make button that runs
    ``make_city`` and shows the results below. What is typed wins over the list; "lat, lon" wins over the name.
    ``out_dir`` = where the files go (default ``/content/out`` on Google Colab, ``./out`` elsewhere)."""
    from .. import nb

    try:
        import ipywidgets as widgets
        from IPython.display import HTML, display
    except ImportError:
        print("The panel needs ipywidgets (pip install ipywidgets). Without it, two lines do the same:\n"
              '    sel = picker.resolve("Tokyo")            # or picker.resolve("", "35.68, 139.77")\n'
              '    res = make_city(place=sel["name"], lat=sel["lat"], lon=sel["lon"], outputs=("poster",))')
        return
    wide = widgets.Layout(width="100%", max_width="560px")
    style = {"description_width": "96px"}
    city = widgets.Dropdown(options=list(PRESETS), value="Dhaka", description="City list", layout=wide, style=style)
    name = widgets.Text(placeholder="or type any city, e.g. Kathmandu, Nepal", description="City name", layout=wide, style=style)
    lat_lon = widgets.Text(placeholder="or a point, e.g. 23.72, 90.40", description="lat, lon", layout=wide, style=style)
    boxes = {k: widgets.Checkbox(value=(k == "poster"), description=label, indent=False, layout=widgets.Layout(width="auto"))
             for k, label in (("poster", "Poster (PNG)"), ("video", "Video (MP4)"), ("gif", "GIF"))}
    size = widgets.ToggleButtons(options=["FAST", "FULL"], value="FAST", description="Size", style=style,
                                 tooltips=["smaller ground, small video: fits a free Colab runtime", "the full sheet and video"])
    button = widgets.Button(description="Make", button_style="primary", layout=widgets.Layout(width="100%", max_width="560px", height="48px"))
    out = widgets.Output()
    _PANEL.update(lat_lon=lat_lon, name=name, city=city)

    def run(_button) -> None:
        from . import make_city

        want = tuple(k for k, b in boxes.items() if b.value)
        button.disabled = True
        button.description = "Working ..."
        with out:
            out.clear_output()
            try:
                nb.stop_unless(bool(want), "tick at least one of Poster, Video, GIF")
                sel = resolve(name.value or city.value, lat_lon.value)
                res = make_city(place=sel["name"], lat=sel["lat"], lon=sel["lon"], outputs=want, fast=size.value == "FAST",
                                out_dir=nb.out_dir(out_dir), preset=sel["preset"])
                if res["poster"]:
                    nb.picture(res["poster"])
                if res["video"]:
                    nb.video(res["video"])
                if res["gif"]:
                    _show_gif(res["gif"])
                for k in ("poster", "video", "gif"):
                    if res[k]:
                        print(f"{k}: {res[k]}")
            except Exception as err:  # noqa: BLE001  (show the reason in the panel, keep the button usable)
                print(f"{type(err).__name__}: {err}")
            finally:
                button.disabled = False
                button.description = "Make"

    button.on_click(run)
    help_ = widgets.HTML('<div style="font-size:15px;line-height:1.5">Choose from the list, or type a city, or give a point. '
                         "Brisbane, Dhaka and Richmond are ready-made (no download); any other place is downloaded from "
                         "OpenStreetMap, which takes a few minutes.</div>")
    display(widgets.VBox([help_, city, name, lat_lon, widgets.HBox(list(boxes.values()), layout=widgets.Layout(flex_flow="row wrap")),
                          size, button, out], layout=widgets.Layout(width="100%")))


def _show_gif(path, max_mb: float = 8.0) -> None:
    import base64
    from pathlib import Path

    from IPython.display import HTML, display

    path = Path(path)
    if path.stat().st_size / 1e6 > max_mb:
        return
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    display(HTML(f'<img src="data:image/gif;base64,{b64}" style="width:100%;max-width:480px;height:auto" alt="animated lens map">'))


def _check_presets(tol: float = 0.02) -> list[str]:
    """Compare every ``PRESETS`` point with the Nominatim search result for its name (a maintenance check; about one
    request per second). Returns one line per city: the offset in degrees, and where the point itself lies."""
    from . import osm

    rows = []
    for name, (lat, lon) in PRESETS.items():
        hit = osm.lookup(name if name != "Richmond (Virginia)" else "Richmond, Virginia")
        here = osm.reverse(lat, lon, zoom=14)
        d = max(abs(hit["lat"] - lat), abs(hit["lon"] - lon))
        rows.append(f"{'ok  ' if d <= tol else 'far '} {name}: {d:.3f} deg from Nominatim ({hit['lat']:.4f}, {hit['lon']:.4f}); "
                    f"the point is in {json.dumps(here['display_name'], ensure_ascii=True)}")
    return rows
