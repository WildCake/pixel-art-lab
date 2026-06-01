#!/usr/bin/env python3
"""Regression check for rare warm/purple colors in mixel grid transfer."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pixel_art_grid as pag
import pixel_art_lab as lab


def hsv(color: tuple[int, int, int]) -> tuple[float, float, float]:
    red, green, blue = [channel / 255.0 for channel in color]
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    chroma = maximum - minimum
    if chroma <= 0:
        hue = 0.0
    elif maximum == red:
        hue = (60.0 * ((green - blue) / chroma) + 360.0) % 360.0
    elif maximum == green:
        hue = 60.0 * ((blue - red) / chroma + 2.0)
    else:
        hue = 60.0 * ((red - green) / chroma + 4.0)
    saturation = 0.0 if maximum <= 0 else chroma / maximum
    return hue, saturation, maximum


def hue_in_ranges(hue: float, ranges: tuple[tuple[float, float], ...]) -> bool:
    for start, end in ranges:
        if start <= end:
            if start <= hue <= end:
                return True
        elif hue >= start or hue <= end:
            return True
    return False


def count_hue_pixels(
    image: Image.Image,
    ranges: tuple[tuple[float, float], ...],
    min_saturation: float = 0.35,
    min_value: float = 0.12,
) -> int:
    total = 0
    for color in image.convert("RGB").getdata():
        hue, saturation, value = hsv(color)
        if hue_in_ranges(hue, ranges) and saturation >= min_saturation and value >= min_value:
            total += 1
    return total


def count_hue_palette(
    palette: list[str],
    ranges: tuple[tuple[float, float], ...],
    min_saturation: float = 0.35,
    min_value: float = 0.12,
) -> int:
    total = 0
    for item in palette:
        color = tuple(int(item[index : index + 2], 16) for index in (1, 3, 5))
        hue, saturation, value = hsv(color)
        if hue_in_ranges(hue, ranges) and saturation >= min_saturation and value >= min_value:
            total += 1
    return total


def main() -> None:
    source = Image.open(ROOT / "docs/examples/gpt-character-generation-mixels.png").convert("RGB")
    settings = {
        "targetHeight": 256,
        "aspectDriver": "height",
        "aspectLock": True,
        "colors": 64,
        "dither": "none",
        "resample": "box",
        "colorDistance": "oklab",
        "edgeMode": "sobel",
        "edgeThreshold": 0.04,
        "edgePaletteWeight": 0.45,
        "accentPaletteWeight": 0.8,
        "hueRarityWeight": 1.6,
        "hueMatchWeight": 0.35,
        "interestingMinSaturation": 0.07,
        "interestingMinValue": 0.05,
        "gridSnap": True,
        "gridAutoSize": False,
        "gridQuantizeFirst": True,
        "gridSnapMethod": "dark-stroke",
        "paletteInput": "prepared",
        "preserveLuma": False,
        "preserveSaturation": False,
        "flatRegionPaletteColors": 0,
        "flatRegionChannelStep": 0,
        "mixelCleanupPasses": 0,
        "paletteStrategy": "projected-rare",
    }
    warm_ranges = ((345.0, 360.0), (0.0, 55.0))
    purple_ranges = ((245.0, 315.0),)

    cache = {}
    median_result = lab.convert_in_memory(
        source,
        {**settings, "paletteStrategy": "median-cut"},
        cache=cache,
        version=1,
    )
    median_output = Image.open(
        io.BytesIO(base64.b64decode(median_result["output"].split(",", 1)[1]))
    ).convert("RGB")
    assert (
        count_hue_pixels(median_output, purple_ranges) < 100
    ), "median-cut fixture unexpectedly preserved the purple accent"

    result = lab.convert_in_memory(source, settings, cache=cache, version=1)
    output = Image.open(io.BytesIO(base64.b64decode(result["output"].split(",", 1)[1]))).convert("RGB")
    palette = result["palette"]

    warm_pixels = count_hue_pixels(output, warm_ranges)
    purple_pixels = count_hue_pixels(output, purple_ranges)
    warm_palette = count_hue_palette(palette, warm_ranges)
    saturation = pag.luma_weighted_saturation_mean(output)

    assert warm_palette >= 16, f"warm palette collapsed: {warm_palette}"
    assert warm_pixels >= 2000, f"warm output pixels collapsed: {warm_pixels}"
    assert purple_pixels >= 500, f"purple output pixels collapsed: {purple_pixels}"
    assert saturation >= 0.39, f"output saturation drifted low: {saturation:.4f}"

    print(
        "palette preservation passed: "
        f"warm_palette={warm_palette}, warm_pixels={warm_pixels}, "
        f"purple_pixels={purple_pixels}, saturation={saturation:.4f}"
    )


if __name__ == "__main__":
    main()
