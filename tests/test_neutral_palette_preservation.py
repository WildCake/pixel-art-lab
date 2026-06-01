#!/usr/bin/env python3
"""Regression check for rare pale/neutral colors in projected palettes."""

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


def decode_output(data_url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")


def is_pale_neutral(color: tuple[int, int, int]) -> bool:
    return pag.rgb_saturation(color) <= 0.42 and max(color) >= 95 and pag.srgb_luma(color) >= 90


def count_pale_neutral_pixels(image: Image.Image) -> int:
    return sum(1 for color in image.convert("RGB").getdata() if is_pale_neutral(color))


def count_pale_neutral_palette(palette: list[str]) -> int:
    total = 0
    for item in palette:
        color = tuple(int(item[index : index + 2], 16) for index in (1, 3, 5))
        if is_pale_neutral(color):
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
        "colorDistance": "rgb",
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
        "gridQuantizeFirst": False,
        "gridSnapMethod": "center",
        "paletteInput": "prepared",
        "preserveLuma": False,
        "preserveSaturation": False,
        "flatRegionPaletteColors": 0,
        "flatRegionChannelStep": 0,
        "mixelCleanupPasses": 0,
        "paletteStrategy": "projected-islands",
    }

    result = lab.convert_in_memory(source, settings, cache={}, version=1)
    output = decode_output(result["output"])

    mask_crop = output.crop((72, 32, 114, 82))
    pale_palette = count_pale_neutral_palette(result["palette"])
    pale_crop_pixels = count_pale_neutral_pixels(mask_crop)

    assert pale_palette >= 3, f"pale neutral palette collapsed: {pale_palette}"
    assert pale_crop_pixels >= 80, f"pale neutral mask pixels collapsed: {pale_crop_pixels}"

    print(
        "neutral palette preservation passed: "
        f"pale_palette={pale_palette}, pale_crop_pixels={pale_crop_pixels}"
    )


if __name__ == "__main__":
    main()
