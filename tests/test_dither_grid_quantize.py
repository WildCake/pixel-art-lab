#!/usr/bin/env python3
"""Regression checks for dither compatibility with grid quantize-first voting."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pixel_art_lab as lab


def output_image(result: dict) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(result["output"].split(",", 1)[1]))).convert("RGB")


def changed_pixels(left: Image.Image, right: Image.Image) -> int:
    diff = ImageChops.difference(left, right)
    return sum(1 for pixel in diff.getdata() if pixel != (0, 0, 0))


def render(source: Image.Image, settings: dict) -> tuple[Image.Image, dict]:
    result = lab.convert_in_memory(source, settings, cache={}, version=7)
    return output_image(result), result


def main() -> None:
    source = Image.open(ROOT / "docs/examples/gpt-character-generation-mixels.png").convert("RGB")
    settings = {
        "targetHeight": 256,
        "aspectDriver": "height",
        "aspectLock": True,
        "colors": 64,
        "dither": "none",
        "ditherStrength": 64,
        "ditherScope": "adaptive",
        "ditherErrorThreshold": 3,
        "ditherEdgeThreshold": 0.28,
        "ditherLumaRange": 45,
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
        "gridSnapMethod": "center",
        "paletteInput": "prepared",
        "preserveLuma": False,
        "preserveSaturation": False,
        "flatRegionPaletteColors": 0,
        "flatRegionChannelStep": 0,
        "mixelCleanupPasses": 0,
        "paletteStrategy": "projected-rare",
    }

    none_quantized, _none_result = render(
        source,
        {**settings, "gridQuantizeFirst": True, "dither": "none"},
    )
    ordered_quantized, ordered_result = render(
        source,
        {**settings, "gridQuantizeFirst": True, "dither": "ordered"},
    )
    floyd_quantized, floyd_result = render(
        source,
        {**settings, "gridQuantizeFirst": True, "dither": "floyd"},
    )

    assert changed_pixels(none_quantized, ordered_quantized) == 0
    assert changed_pixels(none_quantized, floyd_quantized) == 0
    assert ordered_result["stats"]["dither"] == "none"
    assert ordered_result["stats"]["ditherRequested"] == "ordered"
    assert ordered_result["stats"]["ditherDisabledReason"] == "gridQuantizeFirst"
    assert floyd_result["stats"]["dither"] == "none"
    assert floyd_result["stats"]["ditherRequested"] == "floyd"
    assert floyd_result["stats"]["ditherDisabledReason"] == "gridQuantizeFirst"

    none_raw, _ = render(source, {**settings, "gridQuantizeFirst": False, "dither": "none"})
    ordered_raw, ordered_raw_result = render(
        source,
        {**settings, "gridQuantizeFirst": False, "dither": "ordered"},
    )
    floyd_raw, floyd_raw_result = render(
        source,
        {**settings, "gridQuantizeFirst": False, "dither": "floyd"},
    )

    ordered_delta = changed_pixels(none_raw, ordered_raw)
    floyd_delta = changed_pixels(none_raw, floyd_raw)
    assert ordered_delta > 0, "ordered dither should affect raw grid vote output"
    assert floyd_delta > 0, "Floyd-Steinberg dither should affect raw grid vote output"
    assert ordered_raw_result["stats"]["dither"] == "ordered"
    assert ordered_raw_result["stats"]["ditherDisabledReason"] is None
    assert floyd_raw_result["stats"]["dither"] == "floyd"
    assert floyd_raw_result["stats"]["ditherDisabledReason"] is None

    print(
        "dither grid quantize passed: "
        f"ordered_delta={ordered_delta}, floyd_delta={floyd_delta}"
    )


if __name__ == "__main__":
    main()
