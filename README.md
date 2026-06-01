# Pixel Art Lab

Local browser tool for converting AI-generated or faux pixel art into a strict pixel grid while preserving rare colors, silhouettes, and one-pixel details.

It is designed for agent/operator work: an agent starts the local service, the operator imports one image, tunes the live preview, and exports the result.

## Features

- Hidden-grid transfer for GPT/imagegen "mixel" art.
- Palette builders that keep rare hues, accents, shadows, and outlines.
- Output dimensions locked to the source aspect ratio, with source-height capped at 1024 px.
- Zoom at cursor, clamped panning, and `Z` hover compare against the imported original.
- Topbar `Hold Before` button that draws the imported original over the whole output only while pressed.
- Designer-styled local tooltips for controls and palette swatches; browser-native `title` tooltips are not used.
- Local-only processing. Images are uploaded only to the local Python process.
- Custom browser presets for every conversion setting, including disabled conditional controls.

## Requirements

- Python 3.11+
- A local browser
- Python packages from `requirements.txt`

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```sh
python3 pixel_art_lab.py --host 127.0.0.1 --port 8767 --no-browser
```

Open the printed URL:

```txt
http://127.0.0.1:8767/
```

If `8767` is busy, the service automatically tries the next 49 ports.

## Basic Workflow

1. Import one image in `Start`.
2. Set width or height in `Output & Dither`. With `lock source aspect` enabled, the other side updates after Enter or field blur.
3. Choose a palette strategy in `Palette Builder`.
4. Enable `Grid Snap` in `Hidden Grid` for generated art that contains enlarged pseudo-pixels.
5. Wheel zooms at the cursor. Drag pans. Hold `Hold Before` to overlay the full imported original, or hold `Z` over the preview to compare a local original/output region.
6. Export the PNG from `Export` or save your complete current controls as a JSON settings file.

## Examples

- [Example: GPT Character Generation, Mixels](docs/examples/README.md)

## Starting Settings

For GPT/imagegen pixel art with visible mixels:

```txt
Grid Snap: on
auto size from detected mixels: on
quantize before grid vote: off
Cell reducer: nearest center sample
Palette strategy: projected-rare
Color distance: weighted RGB
Colors: 64
Dither: ordered Bayer if smooth backgrounds need texture, otherwise none
Dither scope: adaptive smooth areas
```

For larger environment concepts where color richness matters:

```txt
Grid Snap: off unless the source has obvious mixels
Palette strategy: projected-mass or projected-rare
Colors: 128-256
Resample: box or lanczos
preserve luma: on if brightness drifts
preserve saturation: on if accents wash out
```

## CLI Example

The GUI uses `pixel_art_grid.py` internally. The converter can also run directly:

```sh
python3 pixel_art_grid.py input.png \
  --output output-1024.png \
  --size 1024x576 \
  --colors 64 \
  --palette-strategy projected-rare \
  --dither none \
  --preview-scale 2
```

## Files

- `pixel_art_lab.py` - local web GUI and API server.
- `pixel_art_grid.py` - conversion pipeline and command-line converter.
- `requirements.txt` - Python dependencies.
- `docs/pixel-art-lab.md` - detailed notes on grid snap, palettes, dithering, caching, and performance rules.

## Agent Runbook

Use this exact command when launching the service for an operator:

```sh
python3 pixel_art_lab.py --host 127.0.0.1 --port 8767 --no-browser
```

Then send the operator the printed URL and keep the process running.

## Publish Separately

This folder is self-contained. To publish it as its own repository:

```sh
cd pixel-art-lab
git init
git add .
git commit -m "Initial Pixel Art Lab"
```

## Performance

The heavy image stages use NumPy and Numba. The first render after enabling a Numba-backed feature can include JIT compilation time; later renders reuse compiled kernels and cached stages. New hot-path algorithms should be implemented in NumPy or Numba before being exposed in the GUI.
