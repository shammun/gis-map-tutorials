# The clip in which the lenses ride the storm (`lens_map.mp4`)

A 41-second, 1920 × 1080 clip: three lenses (temperature, moisture, pressure and wind) open over the north-east
Atlantic and follow the tracked low of Storm Ciarán while one clock steps the hours. The pictures inside the lenses are
made in Python; this small [Remotion](https://www.remotion.dev) project only moves and masks them. Rendering needs
Node.js (18 or newer) on your own machine; it does not run on Google Colab.

## 1 · Make the layers (Python)

From the repository root (after `pip install -e .`):

```
python tutorials/lens_maps/03_lens_map_clip.py
```

This writes `out/lens_layers/`: `ctx.png` (base map), `p.png` (chart ground), and for each of the 25 hours `t_NN.png`,
`q_NN.png` (fills), `ql_NN.png`, `pl_NN.png` (line work), plus `manifest.json` (the hours, the tracked low in pixels, a
coarse wind grid, colour ramps). It also writes a pure-Python approximation of the clip. The notebook
`03_lens_map_clip.ipynb` does the same, also on Google Colab.

## 2 · Render (Node.js)

```
cd remotion/lens_map
npm install
npx remotion render src/index.ts LensMapClip out/lens_map.mp4 --public-dir=../../out/lens_layers
```

`npx remotion studio src/index.ts --public-dir=../../out/lens_layers` opens the clip in a browser to scrub through it.

## Files

| File | What it is |
|---|---|
| `src/LensMapClip.tsx` | the clip: layout, timeline, the lens centres that follow the low, wind streaks, labels |
| `src/kit/lens.tsx` | the lens rule as circle masks (`clip-path` + CSS masks); the same rule as `gismaps.lens.common.composite` |
| `src/kit/util.tsx` | colours, the font, a smooth ramp, the pulsing ring |
| `src/Root.tsx`, `src/index.ts` | the composition (1230 frames at 30 frames per second) and the entry point |

Data: ERA5 (Copernicus Climate Change Service / ECMWF; contains modified Copernicus Climate Change Service information),
ETOPO 2022 (NOAA NCEI), Natural Earth. Style after Fan Mo, #30DayMapChallenge 2026 Day 14. Map created by Shammunul Islam.
