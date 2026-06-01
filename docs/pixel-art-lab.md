# Pixel Art Lab

`pixel_art_lab.py` is a local browser GUI for exploring the pixel-art conversion pipeline from `pixel_art_grid.py`.

Run it from the Pixel Art Lab repository root:

```sh
python3 pixel_art_lab.py --port 8767
```

If the requested port is busy, the tool searches the next available port. Use `--no-browser` when you want to open the URL manually.

## Workflow

1. Load one source image with the file picker.
2. Change output size, palette strategy, edge filtering, cleanup, and color controls.
3. The output canvas re-renders after each setting change with a short debounce.
4. Use the mouse wheel to zoom at the cursor. Zoom is clamped to 100% minimum.
5. Drag to pan. Panning is clamped so the output cannot be moved beyond its visible bounds.
6. Hold `Hold Before` in the topbar to draw the imported original over the whole output using the same crop, pan, and zoom; release it to immediately return to the rendered result.
7. Hold `Z` while hovering the output to show a split tooltip. The left side samples the imported original image at its own resolution, mapped to the same normalized region as the output; the right side is the output at the exact same logical coordinates.
8. Export the current PNG or settings JSON from the Export panel.

## Output Size

After import, the GUI derives the output aspect ratio from the source image. The default output height is the source height capped at 1024 pixels, with width computed from the same source ratio. With `lock source aspect` enabled, editing width updates height and editing height updates width, so the conversion keeps the original image proportions instead of introducing a crop target.

## Grid Snap

`Grid Snap` is for GPT/image-generation pseudo-pixel art where the model drew enlarged "mixels" instead of a true logical pixel grid. When enabled, the normal resize stage is replaced by a hidden-grid transfer stage:

- `auto size from detected mixels` estimates likely source-pixel cell sizes from horizontal and vertical edge profiles, then exposes the best candidate logical resolutions through the `Auto variant` slider. Turning this off keeps manual `Width` and `Height` fully in control.
- `quantize before grid vote` is disabled by default. When enabled, it limits the source colors with the active Palette Builder before cell voting, so tiny intended colors survive as palette colors instead of being replaced by a separate hidden median-cut palette. Dithering is disabled while this is on because every voted cell has already been snapped into the active palette before the final palette stage.
- `cell mode` chooses the most common quantized color inside each detected output cell.
- `nearest center sample` takes the nearest source color at the center of each output cell and avoids averaging.
- `dark-stroke bias` starts from cell mode, then preserves narrow dark high-contrast strokes inside a cell, which helps one-mixel black separators on bones, fingers, ribs, teeth, and other line-art details.

For a large environment concept, start with `Grid Snap` enabled, `auto size from detected mixels` enabled, `nearest center sample`, and `quantize before grid vote` off; then scrub `Auto variant` around the detected candidates and compare stone edges, cracks, moss, and fog gradients with the `Z` tooltip. Turn quantize-first on only when rare source colors are being averaged away and dithering is not needed.

## Included Algorithm Families

Palette strategies from previous experiments are available:

- `median-cut`
- `interesting`
- `hue-mass`
- `spectrum-peaks`
- `shadow-spectrum`
- `projected-mass`
- `projected-rare`
- `projected-edge`
- `projected-islands`
- `projected-anchors`
- `projected-frontier`
- `projected-graft`

The projected strategies use the imported image as the palette donor. `projected-mass` and `projected-rare` are the two accepted baselines from the 64-color tests. The newer projected variants keep that family but redistribute palette slots by contour votes, local color islands, tonal anchor cells, or mixed frontier scoring.

## Performance Rule

New conversion strategies must put expensive per-pixel, per-tile, per-unique-color mapping, dithering, cleanup, and scoring work in NumPy or Numba first. If a step needs an explicit per-pixel loop, implement it as a Numba kernel before exposing it in the GUI. Python should only orchestrate small palette-slot allocation steps and keep a pure-Python fallback for missing optional dependencies. Do not add new Python pixel loops to the hot render path.

Current accelerated coverage includes projected mass/rare palette projection, projected edge/anchor/frontier/graft color mapping, projected island/frontier tile scoring, grid-snap center/mode/dark-stroke cell transfer, rare-color guarded palette mapping, ordered and Floyd-Steinberg dithering, bilateral smoothing, edge sharpen blending, luma/saturation preservation, low-detail snapping, and single-pixel mixel cleanup.

## Edge Modes

The GUI can send different edge masks into the existing palette and cleanup pipeline:

- `Sobel`: the original edge mask from the CLI.
- `Laplacian`: a 3x3 high-frequency kernel.
- `High-pass`: grayscale difference from a blurred copy.
- `Pillow contour`: Pillow's contour filter normalized into a mask.
- `none`: disables edge-weighted palette votes and edge-dependent cleanup.

## Dithering

Ordered Bayer dithering and Floyd-Steinberg error diffusion both use the active palette strategy from the source image. They do not rebuild a separate Pillow palette, so rare protected colors from `projected-*`, hue protection, and interesting-color slots remain available when dithering is enabled. Dithering is available after normal resizing or after `Grid Snap` with `quantize before grid vote` turned off. With quantize-first grid voting, dithering is intentionally disabled because the grid vote already maps source cells into palette colors and leaves no useful quantization error to distribute. The `adaptive smooth areas` scope applies to ordered dithering and suppresses it on strong edge-mask pixels, high local-luma-detail pixels, and pixels whose nearest-palette error is already low. This keeps dithering available for smooth backgrounds, fog, sky, water, and large gradients without chewing up silhouettes and detailed object interiors.

## Notes

- The UI is local-only and has no dependency on the original Electron/React game runtime.
- Realtime behavior depends on output resolution and strategy. The heavy per-pixel stages use NumPy and Numba when installed, and the GUI caches resized bases, edge masks, prepared images, preview PNGs, and recently used full renders. The first render after enabling a Numba-backed option can include JIT compilation time; later renders reuse the compiled code.
- The original image is not shown as a separate viewport. It is only used for the `Z` comparison tooltip and optional palette donor modes.
- Dither modes are limited to 256 colors because indexed palette dithering is 8-bit.
