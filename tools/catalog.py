"""The catalogue: the recipes (tutorials) and the gallery cards. One place for the titles, explanations, data sources,
steps and credits that the notebooks (first cell), the README and the web page (docs/index.html) share."""
from __future__ import annotations

REPO = "shammun/gis-map-tutorials"
GITHUB = f"https://github.com/{REPO}"
BLOB = f"{GITHUB}/blob/main"
TREE = f"{GITHUB}/tree/main"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main"
PAGES = "https://shammun.github.io/gis-map-tutorials"
AUTHOR = "Map created by Shammunul Islam"
FAN_MO = "Style after Fan Mo, #30DayMapChallenge 2026 Day 14"

# ---------------------------------------------------------------- data sources: (name, link, licence / credit to print)
IBTRACS = ("IBTrACS v04r01 tropical-cyclone best tracks, NOAA NCEI", "https://www.ncei.noaa.gov/products/international-best-track-archive",
           "public domain; cite Knapp et al. (2010)")
ERA5_H = ("ERA5 hourly reanalysis (Copernicus Climate Change Service / ECMWF), read from the public ARCO-ERA5 store",
          "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels",
          "Copernicus licence (CC BY 4.0): \"Contains modified Copernicus Climate Change Service information\"; cite Hersbach et al. (2020)")
ERA5_M = ("ERA5 monthly means of total column water vapour, 1991-2020 (Copernicus Climate Change Service / ECMWF)",
          "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means",
          "Copernicus licence (CC BY 4.0): \"Contains modified Copernicus Climate Change Service information\"")
ETOPO = ("ETOPO 2022 global relief, NOAA NCEI", "https://www.ncei.noaa.gov/products/etopo-global-relief-model", "public domain")
OISST = ("NOAA OISST v2.1 daily sea-surface temperature and sea ice, NOAA NCEI", "https://www.ncei.noaa.gov/products/optimum-interpolation-sst",
         "public domain; cite Huang et al. (2021)")
NSIDC = ("Sea Ice Index v4 (daily sea-ice extent), NSIDC / NOAA", "https://nsidc.org/data/g02135", "free to use; cite Fetterer et al. (2017)")
WOA = ("World Ocean Atlas 2023, annual-mean temperature, NOAA NCEI", "https://www.ncei.noaa.gov/products/world-ocean-atlas",
       "public domain; cite Locarnini et al. (2024)")
S2 = ("Sentinel-2 L2A (ESA / Copernicus), read from Element 84 Earth Search on AWS open data", "https://earth-search.aws.element84.com/v1",
      "free and open: \"Contains modified Copernicus Sentinel data 2019-2026\"")
OSM = ("OpenStreetMap, through the Overpass API", "https://www.openstreetmap.org/copyright",
       "© OpenStreetMap contributors, Open Database Licence (ODbL) 1.0")
NE = ("Natural Earth land, coast and borders", "https://www.naturalearthdata.com", "public domain")

#: stem -> one recipe. ``pip`` = extra packages of the Colab setup cell.
TUTORIALS = {
    "gis_styles/01_glow_season": dict(
        n="Recipe 1", title="The storm belts follow the sun",
        one="Makes the poster and the video of 35 years of tropical cyclones as dots of light on their calendar day, with a mirrored season chart.",
        pip=""),
    "gis_styles/02_epoch_footprints": dict(
        n="Recipe 2", title="Storm Ciarán's footprint",
        one="Makes the poster and the video of where Storm Ciarán's core below 960 hPa was every three hours, stacked on one map.",
        pip=""),
    "gis_styles/03_block_diagram": dict(
        n="Recipe 3", title="Storm Ciarán crosses north-west Europe",
        one="Makes the poster and the video of Storm Ciarán's 10-m wind on a 3-D block of north-west Europe, with a wind chart for Brest and London.",
        pip=""),
    "gis_styles/04_relief_field": dict(
        n="Recipe 4", title="Where the water vapour sits",
        one="Makes the poster and the video of the water vapour above South Asia, month by month, drawn as lit terraces.",
        pip=""),
    "gis_styles/05_spilhaus_ocean": dict(
        n="Recipe 5", title="One ocean",
        one="Makes the poster and the video of sea-surface temperature and sea ice through 2025 on a map cut through the continents.",
        pip=""),
    "gis_styles/06_cutaway_globe": dict(
        n="Recipe 6", title="Inside the Pacific",
        one="Makes the poster and the video of a globe with a wedge cut out of the Pacific, showing the temperature down to 1000 m.",
        pip=""),
    "gis_styles/07_satellite_river_epochs": dict(
        n="Recipe 7", title="The Jamuna never stays put",
        one="Makes the video of the Jamuna river's water in eight dry seasons, from Sentinel-2 with no key (Google Earth Engine is an optional section).",
        pip="pystac-client rasterio earthengine-api"),
    "lens_maps/01_lens_city_osm": dict(
        n="Recipe 8", title="One city, many realities",
        one="Makes the lens-map poster, video and GIF of Brisbane, Dhaka or Richmond from OpenStreetMap, and of any other place you choose.",
        pip="ipywidgets"),
    "lens_maps/02_lens_weather_era5": dict(
        n="Recipe 9", title="One storm, many realities",
        one="Makes the lens-map poster, video and GIF of Storm Ciarán (temperature, moisture, pressure and wind) from ERA5.",
        pip="zarr gcsfs"),
    "lens_maps/03_lens_map_clip": dict(
        n="Recipe 10", title="The lenses ride the storm",
        one="Makes every map layer of the clip in which three lenses follow Storm Ciarán, and a pure-Python version of the clip; the final render uses Remotion on your own machine.",
        pip=""),
    "glass_stack/01_glass_stack": dict(
        n="Recipe 11", title="Glass-pane globe stack",
        one="Makes the poster and the video of a relief globe with six glass panes, each a different field of one weather forecast, and the inputs of its Remotion version.",
        pip=""),
}

GFS = ("NOAA GFS forecast, 0.25°, run of 16 November 2026 18 UTC, as published by dynamical.org (anonymous Icechunk / Zarr store)",
       "https://dynamical.org", "CC BY 4.0; credit \"dynamical.org / NOAA GFS\"")
EXTRA_DATA_ROWS = ("<tr><td>Weather forecast</td><td>NOAA GFS, as published by dynamical.org</td>"
                   "<td>CC BY 4.0; credit \"dynamical.org / NOAA GFS\"</td></tr>")

LENS_CITY_STEPS = [
    "Install the package; the reduced OpenStreetMap extract of the city ships with the repository (LIVE_OSM = True asks the public Overpass server for every layer instead: about 35 minutes, tiled and cached)",
    "Set CITY = \"{city}\" in the settings cell",
    "Project longitude and latitude to poster pixels with a local equirectangular map about the city's anchor point",
    "Turn the OpenStreetMap ways and relations into shapely polygons and sort land use into classes",
    "Draw the same extract three times at full size on one pixel grid: land use, public realm, city morphology",
    "Cut the lenses: a lens is a circular mask on that grid; overlapping lenses mix half and half (allowed for categories)",
    "Add the title, the label blocks, the legend with its counts, the scale bar, north arrow and credits, and save the 2000 × 2828 px poster",
    "Make the video (layers and labels are drawn once, then each frame only moves the masks; frames are piped to ffmpeg) and convert it to the GIF",
]
LENS_TOOLS = "shapely · numpy · matplotlib · pillow · ffmpeg"

#: the gallery cards, in page order. ``media`` = poster stem in docs/media/, ``video`` = MP4 stem.
CARDS = [
    dict(section="styles", media="glow_season", video="glow_season", tutorial="gis_styles/01_glow_season",
         title="The storm belts follow the sun", kicker="Glow-dot time-lapse with a mirrored season chart",
         what="Every 6-hourly position of every tropical cyclone of 1990-2024 is one small dot of light, drawn on its day of the year. "
              "Where many storms passed, the dots add up and the map glows. Read it like a long-exposure photograph: brightness "
              "is mostly how often storms passed there (stronger storms glow a little more). The chart under the map counts the "
              "storms in progress, Northern Hemisphere upwards and Southern downwards; in the video its playhead shows the date.",
         data=[IBTRACS, NE],
         steps=["Load the storm positions (34 knots or more, 6-hourly, all basins): data/gis_styles/cyclones.npz, or rebuild it from NOAA's files",
                "Build one Equal Earth pixel grid centred on 165°W, so the cut falls on Africa and no storm basin is split",
                "Bin the positions into pixels and blur them twice (a sharp core and a soft halo); weight each by its wind and by its age",
                "Tone-map the sum with 1 − exp(−gain · v) and colour it from dark red to white on the dark base map",
                "Count the storms in progress per calendar day for the two hemispheres and draw the mirrored chart",
                "Add the title, the month and the credits; save the poster (all days at once)",
                "For the video, draw one frame per half day with a five-day fading tail, and encode the frames with ffmpeg"],
         tools="numpy · scipy · pyproj · cartopy · matplotlib · ffmpeg"),
    dict(section="styles", media="epoch_footprints", video="epoch_footprints", tutorial="gis_styles/02_epoch_footprints",
         title="Storm Ciarán's footprint", kicker="Epoch-stacked footprints",
         what="Each coloured shape is the area where sea-level pressure was below 960 hPa at one moment, every three hours on "
              "1-2 November 2023. The shapes are stacked with the newest on top, so the crescents left showing are places the "
              "deep core of the storm has already left. Follow the colours from purple to yellow to follow time; the white line "
              "is the path of the lowest pressure, and the legend gives that pressure at each step.",
         data=[ERA5_H, NE],
         steps=["Load the hourly sea-level pressure of 1-2 November 2023: data/gis_styles/ciaran.npz, or rebuild it from ARCO-ERA5",
                "Follow the low from a seed point, hour by hour (the minimum of the whole box would jump to another low)",
                "Every three hours, keep the connected area below 960 hPa that holds the tracked centre",
                "Resample the pressure to a Lambert conformal pixel grid and apply the threshold there, for smooth edges",
                "Give the epochs ordered plasma colours and stack them, oldest first, on the dark base map",
                "Add the track, the legend with the lowest pressure of each step, the scale bar and the credits; save the poster",
                "For the video, show each epoch alone with its outline, then build the stack, and encode with ffmpeg"],
         tools="numpy · scipy · pyproj · cartopy · matplotlib · ffmpeg"),
    dict(section="styles", media="block_diagram", video="block_diagram", tutorial="gis_styles/03_block_diagram",
         title="Storm Ciarán crosses north-west Europe", kicker="3-D block diagram with a draped field and a linked chart",
         what="A block of north-west Europe is cut out, its relief is exaggerated 40 times, and the wind 10 m above the ground is "
              "laid on it in colour. Places calmer than 8 m/s keep the bare grey relief, so the gale is the only colour. The "
              "chart at the top right gives the wind at Brest and at London for the same hours, with a playhead at the hour "
              "shown. The relief is not to scale and the far side of the block is foreshortened: use the chart for numbers.",
         data=[ERA5_H, ETOPO],
         steps=["Load the hourly 10-m wind (data/gis_styles/ciaran.npz) and the relief at 2 arc-minutes (terrain_nweurope.npz)",
                "Set an oblique parallel camera: it looks towards 24° from north, 35° above the horizon; relief ×40",
                "Compute once, in numpy, which ground point each screen pixel sees (a lookup table pixel → latitude, longitude)",
                "Shade the relief from the north-west, add the walls of the slab and its shadow",
                "Sample the wind speed of one hour through the lookup and lay it on the ground, transparent below 8 m/s",
                "Add the chart of the grid cells nearest Brest and London, the clock, place names, north arrow and credits; save the poster (03 UTC, 2 November)",
                "For the video, repeat the sampling for 47 hours, four frames per hour, and encode with ffmpeg"],
         tools="numpy · scipy · matplotlib · ffmpeg (no 3-D library)"),
    dict(section="styles", media="relief_field", video="relief_field", tutorial="gis_styles/04_relief_field",
         title="Where the water vapour sits", kicker="A data field as lit, terraced relief",
         what="Height and colour both show the same thing: the water vapour in the whole column of air above each place, as the "
              "depth of water it would make if it all fell as rain (mm). The field is cut into bands of 5 mm and lit from the "
              "north-west, so each band is a terrace. In July the monsoon stands as a high plateau over the Ganges plain and the "
              "Bay of Bengal, and the dry Tibetan Plateau is a deep blue floor. The hills are data, not mountains.",
         data=[ERA5_M, NE],
         steps=["Load the twelve monthly means 1991-2020: data/gis_styles/tcwv_southasia.npz",
                "Resample a month to a Lambert azimuthal equal-area pixel grid over South Asia and make the land mask",
                "Cut the field into filled bands of 5 mm (cmocean haline colours)",
                "Turn the bands into terraces and light them from the north-west; draw a fine line on every band edge",
                "Float the land on the dark sea with a soft drop shadow",
                "Add the month, the mean over the land, the bar chart of the twelve months, the banded colour bar, city names and credits; save the poster (July)",
                "For the video, blend from month to month with an ease and a short hold on each month, and encode with ffmpeg"],
         tools="numpy · scipy · pyproj · cmocean · cartopy · matplotlib · ffmpeg"),
    dict(section="styles", media="spilhaus_ocean", video="spilhaus_ocean", tutorial="gis_styles/05_spilhaus_ocean",
         title="One ocean", kicker="Ocean-first map (Spilhaus projection) with a linked chart",
         what="Most world maps cut the oceans at their edges. This one is cut through the continents instead, so all the oceans "
              "show as one connected body of water, with Antarctica in the middle and the Arctic at the top left. Colour is the "
              "temperature of the sea surface; white is sea ice. The chart follows the ice of both poles through 2025: when one "
              "grows the other shrinks. Shapes and areas are strongly distorted, so read areas from the chart, never from the map.",
         data=[OISST, NSIDC, NE],
         steps=["Load the 24 snapshots of 2025 (1st and 16th of each month) and the daily ice extent: data/gis_styles/oisst_2025.npz",
                "Build the Spilhaus square (needs PROJ 9.6 or newer) and turn every pixel back into longitude and latitude",
                "Sample temperature and ice at those points; drop stray ice where the sea is warmer than 1.5 °C",
                "Colour the temperature (cmocean thermal), lay the ice and the land on top",
                "Take the chart's numbers from the NSIDC Sea Ice Index, not from the mapped ice field",
                "Add the date, the two colour bars, the extent chart with its playhead, ocean names and credits; save the poster (16 September)",
                "For the video, blend the snapshots in time, 1.5 days per frame, and encode with ffmpeg"],
         tools="numpy · scipy · pyproj ≥ 3.7.1 · cmocean · cartopy · matplotlib · ffmpeg"),
    dict(section="styles", media="cutaway_globe", video="cutaway_globe", tutorial="gis_styles/06_cutaway_globe",
         title="Inside the Pacific", kicker="Cut-away globe",
         what="The globe shows the temperature of the sea surface. A wedge is cut out of the Pacific north of the Equator, and its "
              "three faces show the temperature below, down to 1000 m. Warm water turns out to be a thin skin on a cold ocean, "
              "and along the Equator that warm layer is deep in the west and shallow in the east. Depth is not to scale: 1000 m is "
              "drawn about 3,200 times too deep, and the figure says so.",
         data=[WOA, NE],
         steps=["Load the annual-mean temperature from the surface to 2000 m: data/gis_styles/woa23_temperature.npz",
                "Cast one ray per pixel at a unit sphere with the wedge 180°-110°W, north of the Equator, removed",
                "For each ray find what it hits: the surface, or one of the three flat faces of the wedge",
                "Turn the hit point into latitude, longitude and depth (1000 m is drawn as half of the radius) and interpolate the temperature there",
                "Colour it (cmocean thermal), add isotherms every 5 °C, depth arcs every 250 m, land and light",
                "Add the labels, the colour bar, the note on the depth exaggeration and the credits; save the poster",
                "For the video, turn the globe into place, open the wedge eastwards, then sway a few degrees; encode with ffmpeg"],
         tools="numpy · scipy · cmocean · cartopy · matplotlib · ffmpeg (no 3-D library)"),
    dict(section="styles", media="river_epochs", video="river_epochs", tutorial="gis_styles/07_satellite_river_epochs", png=False,
         video_only=True,
         title="The Jamuna never stays put", kicker="Epoch stack from satellite images · video",
         what="The left panel shows where the water of the Jamuna (the Brahmaputra in Bangladesh) was in each dry season from 2019 "
              "to 2026, one colour per year, newest on top. Colours that still show are places the river has since left. The right "
              "panel is the newest true-colour satellite image of the same ground. The bars in the legend give the water area of "
              "each year; only 8 km² were water in all eight years. The video shows each year alone, then builds the stack.",
         data=[S2],
         steps=["Find the Sentinel-2 L2A scenes of each dry season (15 January to 5 March, under 5 % cloud, one MGRS tile) with a public STAC search: no key, no account",
                "Read only the window of the river from the cloud-optimised GeoTIFFs, at 40 m (a few hundred kilobytes per band)",
                "Compute the water index MNDWI = (green − SWIR) / (green + SWIR); mask clouds and shadows with the scene classification",
                "Per year take the median of the four clearest scenes; water is MNDWI above 0.1 in connected pieces of 5 km² or more",
                "Optional section inside the same notebook: the same masks with Google Earth Engine (USE_GEE = True and your own PROJECT_ID), and exports on an exact grid",
                "Stack the yearly masks in plasma colours on a dimmed image, beside the newest true-colour image; add the legend with the areas and the credits",
                "For the video, show the finished stack, each year alone, then the build-up; encode with ffmpeg"],
         tools="pystac-client · rasterio · numpy · scipy · matplotlib · ffmpeg · optional: earthengine-api"),
    # ---------------------------------------------------------------- lens maps
    dict(section="lens", media="lens_brisbane", video="lens_brisbane_anim", tutorial="lens_maps/01_lens_city_osm", fan_mo=True,
         portrait=True, city="brisbane",
         title="One city, many realities · Brisbane", kicker="Lens map · city poster from OpenStreetMap",
         what="One quiet, dark base map of central Brisbane with three round lenses. Each lens draws the same ground another way: "
              "how the land is used, where everyday public life happens (parks, toilets, taps, benches, playgrounds), and the built "
              "form (every building and street). Where two lenses overlap, the two maps mix, and you see how they relate. In the "
              "video the lenses trade places, so every piece of ground is seen all three ways.",
         data=[OSM], steps=LENS_CITY_STEPS, tools=LENS_TOOLS),
    dict(section="lens", media="lens_dhaka", video="lens_dhaka_anim", tutorial="lens_maps/01_lens_city_osm", fan_mo=True,
         portrait=True, city="dhaka",
         title="One city, many realities · Dhaka", kicker="Lens map · city poster from OpenStreetMap",
         what="The same three lenses over Dhaka, on the Buriganga river: land use, public realm and city morphology. The lenses are "
              "adapted to what OpenStreetMap holds there: the public-realm lens counts places of worship, marketplaces, toilets, "
              "drinking water and playgrounds. Dark ground inside the land-use lens means that nobody has mapped a land use there "
              "yet, which is a fact about the map and not about the city.",
         data=[OSM], steps=LENS_CITY_STEPS, tools=LENS_TOOLS),
    dict(section="lens", media="lens_richmond", video="lens_richmond_anim", tutorial="lens_maps/01_lens_city_osm", fan_mo=True,
         portrait=True, city="richmond",
         title="One city, many realities · Richmond, Virginia", kicker="Lens map · city poster from OpenStreetMap",
         what="Richmond on the James River, through the same three lenses. Here the land-use lens sits over downtown, Capitol "
              "Square and Church Hill, the public-realm lens over Byrd Park, Monroe Park and the Fan, and the morphology lens over "
              "the river with Belle Isle and Manchester on the south bank. The muted river runs on outside the lenses, so each "
              "lens stays tied to the base map around it.",
         data=[OSM], steps=LENS_CITY_STEPS, tools=LENS_TOOLS),
    dict(section="lens", media="lens_weather", video="lens_weather_anim", tutorial="lens_maps/02_lens_weather_era5", fan_mo=True,
         portrait=True,
         title="One storm, many realities · Storm Ciarán", kicker="Lens map · weather poster from ERA5",
         what="Three fields of one storm at one hour (00 UTC, 2 November 2023): temperature about 1.5 km up, the moisture there, "
              "and sea-level pressure with the wind near the ground. Each lens keeps its exact colours where lenses overlap, and "
              "the upper lens adds only its lines, so every colour still matches its colour bar. Where all three meet you see a "
              "complete weather chart: the low (L, 955 hPa) with warm, moist air wrapping into it. In the video one clock steps "
              "all three fields hour by hour while the low crosses the lenses.",
         data=[ERA5_H, ETOPO, NE],
         steps=["Get the ERA5 fields: the reduced 0.5° copy shipped with the repository (default), or the full 0.25° fields from the anonymous ARCO-ERA5 store (SOURCE = \"arco\", about 230 MB of transfer)",
                "Find the storm: follow the low from a seed point hour by hour and centre the map on it",
                "Build a Lambert conformal pixel grid and sample every field on it",
                "Colour temperature (diverging, white at 0 °C) and humidity (sequential), each with its colour bar",
                "Draw the line work: isobars every 4 hPa, moisture contours, wind arrows turned to the map's north",
                "Cut the lenses with overlap = 0 (the no-blend rule): the lower lens keeps its colours, the upper adds only its lines",
                "Add the title, the label blocks with their colour bars, the scale bar and credits; save the 2000 × 2828 px poster",
                "Make the video (the layers of each hour are drawn once, one clock steps all three fields, frames are piped to ffmpeg) and convert it to the GIF"],
         tools="xarray · pyproj · geopandas · cmocean · scipy · numpy · matplotlib · ffmpeg · optional: zarr, gcsfs"),
    dict(section="lens", media="lens_map", video="lens_map", tutorial="lens_maps/03_lens_map_clip", fan_mo=True, png=False,
         video_only=True, remotion="remotion/lens_map",
         remotion_cmd="cd remotion/lens_map\nnpm install\nnpx remotion render src/index.ts LensMapClip out/lens_map.mp4 --public-dir=../../out/lens_layers",
         title="The lenses ride the storm", kicker="Lens map · 41-second clip (Python layers + Remotion)",
         what="The three weather lenses open one by one over the north-east Atlantic, sweep across the map, then follow the tracked "
              "low of Storm Ciarán while one clock steps the hours from 12 UTC on 1 November to 03 UTC on 2 November 2023. Inside "
              "the pressure lens the wind is drawn as moving streaks. Colours never mix where lenses overlap, so each still "
              "matches its colour bar. The final render needs Node.js on your own machine; Colab makes the data, every map "
              "layer and a pure-Python version of the clip.",
         data=[ERA5_H, ETOPO, NE],
         steps=["Load 25 hours of ERA5 (the reduced 0.5° copy shipped with the repository), the coast and the relief",
                "Follow the low from a seed point hour by hour",
                "Draw one picture per field and hour on one 1140 × 625 px Lambert grid: base map, temperature and humidity fills, moisture contours, isobars; write manifest.json (hours, the track in pixels, a coarse wind grid)",
                "Place the lenses from the track: pressure on the low, temperature to its west, moisture to its south-west",
                "In the notebook: compose each hour with the lens rule in numpy and pipe the pages to ffmpeg (a pure-Python version of the clip, plus a GIF)",
                "On your own machine (Node.js 18 or newer): install the small Remotion project in remotion/lens_map/ and render; it masks the same layers with circles, cross-fades the hours, sweeps the lenses and adds the wind streaks"],
         tools="xarray · pyproj · geopandas · cmocean · numpy · matplotlib · ffmpeg · Remotion 4.0.537 (Node.js) for the final render"),
    # ---------------------------------------------------------------- glass-pane globe stack
    dict(section="globe", media="glass_stack", video="glass_stack", tutorial="glass_stack/01_glass_stack",
         videos=[("glass_stack", "The video (pure Python)"), ("glass_stack_remotion", "The Remotion version")],
         remotion="remotion/glass_stack",
         cmd=("pip install -e .\n"
              "python tutorials/glass_stack/01_glass_stack.py --full     # poster 3840 x 2160 + the whole video, into ./out/\n"
              "python tutorials/glass_stack/01_glass_stack.py            # quick: poster 1920 x 1080 + a preview clip\n"
              "# the Remotion version (needs Node.js):\n"
              "python -m gismaps.glass.remotion remotion/glass_stack/public\n"
              "cd remotion/glass_stack\nnpm install\n"
              "npx remotion render src/index.ts GlassStack out/glass_stack_remotion.mp4 --public-dir=public"),
         title="Glass-pane globe stack", kicker="A globe with six panes of one forecast",
         what="A relief globe with six glass panes in front of it. Each pane shows the same round patch of the Earth (India, "
              "Bangladesh and the Bay of Bengal) at the same moment of one weather forecast, each with a different quantity: "
              "temperature, sea-level pressure with isobars, wind, precipitable water, cloud cover and rain rate. Read it from "
              "the left: the globe with the region facing you, then the panes numbered 1 to 6, then one colour bar per number. "
              "The panes are not at physical heights: they are six maps of one patch, pulled apart. The second video is the "
              "Remotion version: five panes of the whole hemisphere over the North Atlantic.",
         data=[GFS, ETOPO, NE],
         steps=["Load six fields of one GFS run: the small file shipped with the repository, or live and anonymous from dynamical.org (SOURCE = \"live\", needs icechunk and zarr; falls back to the shipped file)",
                "Make the round view of one field: an orthographic projection written by hand, read back onto the data grid",
                "Draw the six pane pictures: one colour ramp and one opacity rule per quantity, isobars every 4 hPa, the coastline",
                "Make the globe's skin: shaded ETOPO 2022 relief with the coast drawn in",
                "Put globe and panes in 3-D with a small ray caster in numpy",
                "Add the legend with numbered chips, the clock, the timeline and the credits; save the poster",
                "For the video, move the camera and the panes along a few keyframes while the forecast steps, and encode with ffmpeg",
                "For the Remotion version (Node.js on your own machine): write the relief, 40 pane pictures and a manifest from Python, then render the project in remotion/glass_stack/"],
         tools="xarray · numpy · scipy · cartopy · matplotlib · ffmpeg · optional: icechunk, zarr · Remotion 4.0.537 + three.js (Node.js) for the second video"),
]


def colab_url(stem: str) -> str:
    return f"{COLAB}/tutorials/{stem}_colab.ipynb"


def badge(stem: str) -> str:
    return f'<a href="{colab_url(stem)}" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'


def command(card: dict) -> str:
    """The exact commands that regenerate one card's outputs on your own machine (from the repository root)."""
    if card.get("cmd"):
        return card["cmd"]
    stem = card["tutorial"]
    if stem.startswith("gis_styles"):
        return (f"pip install -e .\n"
                f"python tutorials/{stem}.py --full     # poster 3840 x 2160 + the whole video, into ./out/\n"
                f"python tutorials/{stem}.py            # quick: poster 1920 x 1080 + a preview clip")
    if card.get("city"):
        c = card["city"]
        return (f"pip install -e .\n"
                f"python -m gismaps.lens city --preset {c} --outputs poster,video,gif --full\n"
                f"python tutorials/{stem}.py            # the step-by-step recipe (all three posters + a preview video)")
    if stem.endswith("02_lens_weather_era5"):
        return (f"pip install -e .\n"
                f"python tutorials/{stem}.py                  # the poster + a preview video and GIF\n"
                f"LENS_QUICK=0 python tutorials/{stem}.py     # the full-size video (about 10 minutes)\n"
                f"LENS_ERA5=arco python tutorials/{stem}.py   # the poster from the full 0.25° fields (pip install zarr gcsfs)")
    return (f"pip install -e .\n"
            f"python tutorials/{stem}.py                  # the layers + the pure-Python clip, into ./out/\n"
            + card.get("remotion_cmd", ""))
