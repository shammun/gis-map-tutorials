# Glass-pane globe stack: the Remotion version

The source of `glass_stack_remotion.mp4`: 26 seconds, 1920 × 1080 px, 30 frames a second. A relief globe and five
glass panes (temperature at 2 m, sea-level pressure as isobars, precipitable water, cloud cover, precipitation rate)
of one NOAA GFS forecast. Every pane shows the whole hemisphere centred on 45°N 30°W. The panes fan out, the isobar
pane lifts out and the low over the North Atlantic is marked, while the forecast steps from +6 h to +48 h.

It is drawn with [Remotion](https://www.remotion.dev) 4.0.537 and three.js 0.186.1. **It needs Node.js on your own computer** (made
with Node 24); it does not run in a notebook or on Google Colab. Remotion downloads a headless
browser (about 120 MB) the first time.

## 1 · Write the pictures it loads (Python)

The film loads 42 files from the folder `public/` with `staticFile()`:

| File | What it is |
|---|---|
| `relief.png` | the globe's skin: hill-shaded ETOPO 2022 relief, latitude × longitude, 2160 × 1080 px |
| `t2m_0.png` … `prate_7.png` | 40 pane pictures: 5 fields × 8 lead times, each the orthographic view of the hemisphere, 900 × 900 px, clear outside the disc |
| `manifest.json` | the run, the lead times, the centre, colour ramps and value ranges, and the position and depth of the North Atlantic low at every lead time |

They are not kept in the repository (26 MB). Write them from the repository root, after `pip install -e .`:

```
python -m gismaps.glass.remotion remotion/glass_stack/public
```

This reads the shipped file `data/glass_stack/gfs_hemisphere_20261116T18.nc` (NOAA GFS run of 16 November 2026
18 UTC at 0.5°, read from dynamical.org) and `data/glass_stack/etopo2022_10min.nc`; it takes about 20 seconds and
needs no network. The last step of the notebook `tutorials/glass_stack/01_glass_stack` writes the same folder
(`out/remotion_public`; copy it here as `public`). To read the same fields fresh from dynamical.org instead (about
240 MB of chunks, about ten minutes; `pip install icechunk zarr`):

```
python -m gismaps.glass.remotion remotion/glass_stack/public live
```

## 2 · Render (Node.js)

```
cd remotion/glass_stack
npm install
npx remotion render src/index.ts GlassStack out/glass_stack_remotion.mp4 --public-dir=public
```

About one minute on a laptop with a graphics card. One frame as a picture, to check before the full render:

```
npx remotion still src/index.ts GlassStack out/glass_stack_remotion.png --frame=480 --public-dir=public
```

`npx remotion studio src/index.ts --public-dir=public` opens the film in the browser with a timeline.
On Linux without a graphics card add `--gl=swangle` to the render command.

## The files

| File | What it does |
|---|---|
| `src/index.ts`, `src/Root.tsx` | register the composition `GlassStack` (780 frames) |
| `src/GlassStack.tsx` | the film: the choreography as functions of the frame number, the legend with leader lines, the low marker, the credits |
| `src/kit/glass.tsx` | the globe and the panes in three.js; two pane pictures are cross-faded in a shader between lead times |
| `src/kit/view3d.tsx` | the three.js canvas with a camera set from numbers on every frame |
| `src/kit/cam.ts`, `src/kit/globe.ts` | camera maths and globe coordinates; the same camera places the SVG labels on the 3-D objects |
| `src/kit/draw.tsx` | colours, type, SVG text |
| `remotion.config.ts` | render settings (H.264, CRF 23) |

## Credits

Map created by Shammunul Islam. Data: dynamical.org / NOAA GFS (CC BY 4.0). Relief: ETOPO 2022 (NOAA NCEI).
