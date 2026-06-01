#!/usr/bin/env python3
"""Regression check for contour-safe bilateral smoothing."""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

from PIL import Image
from PIL import ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pixel_art_grid as pag
import pixel_art_lab as lab


def channel_variance(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    pixels = list(image.crop(box).convert("RGB").getdata())
    return sum(statistics.pvariance(pixel[index] for pixel in pixels) for index in range(3)) / 3.0


def mean_luma(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    pixels = list(image.crop(box).convert("RGB").getdata())
    return sum(pag.srgb_luma(pixel) for pixel in pixels) / len(pixels)


def changed_pixels(left: Image.Image, right: Image.Image) -> int:
    diff = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
    return sum(1 for pixel in diff.getdata() if pixel != (0, 0, 0))


def build_fixture() -> Image.Image:
    random.seed(42)
    image = Image.new("RGB", (80, 80), (0, 0, 0))
    pixels = image.load()
    for y in range(80):
        for x in range(80):
            if 38 <= x <= 41 or y == 44:
                pixels[x, y] = (0, 0, 0)
                continue
            jitter = random.randint(-14, 14)
            if x < 38:
                pixels[x, y] = (
                    max(0, min(255, 126 + jitter)),
                    max(0, min(255, 48 + jitter // 2)),
                    max(0, min(255, 72 - jitter // 3)),
                )
            else:
                pixels[x, y] = (
                    max(0, min(255, 40 + jitter // 3)),
                    max(0, min(255, 74 + jitter // 2)),
                    max(0, min(255, 126 + jitter)),
                )
    return image


def main() -> None:
    source = build_fixture()
    edge_mask = pag.build_sobel_edge_mask(source, threshold=0.02)

    safe_config = lab.config_from_settings({"bilateralRadius": 1, "bilateralSafeEdges": True}, source_size=source.size)
    plain_config = lab.config_from_settings({"bilateralRadius": 1, "bilateralSafeEdges": False}, source_size=source.size)
    legacy_config = lab.config_from_settings({"bilateralRadius": 1, "bilateralMode": "standard"}, source_size=source.size)
    assert safe_config.bilateral_mode == "edge-safe"
    assert plain_config.bilateral_mode == "standard"
    assert legacy_config.bilateral_mode == "standard"

    moderate_edge_safe = pag.bilateral_smooth(
        source,
        radius=2,
        sigma_color=18.0,
        sigma_space=1.4,
        mode="edge-safe",
        edge_mask=edge_mask,
    )
    assert changed_pixels(source, moderate_edge_safe) >= 4000, (
        "edge-safe mode collapsed into no-op smoothing"
    )
    assert mean_luma(moderate_edge_safe, (38, 0, 42, 80)) <= 1.0, (
        "moderate smoothing blurred the dark contour"
    )

    standard = pag.bilateral_smooth(
        source,
        radius=3,
        sigma_color=80.0,
        sigma_space=2.0,
        mode="standard",
        edge_mask=edge_mask,
    )
    edge_safe = pag.bilateral_smooth(
        source,
        radius=3,
        sigma_color=80.0,
        sigma_space=2.0,
        mode="edge-safe",
        edge_mask=edge_mask,
    )

    source_variance = channel_variance(source, (8, 8, 30, 30))
    safe_variance = channel_variance(edge_safe, (8, 8, 30, 30))
    assert safe_variance < source_variance * 0.75, (
        f"flat-area smoothing too weak: source={source_variance:.2f}, safe={safe_variance:.2f}"
    )

    safe_contour_luma = mean_luma(edge_safe, (38, 0, 42, 80))
    standard_contour_luma = mean_luma(standard, (38, 0, 42, 80))
    assert safe_contour_luma <= 1.0, f"dark contour was blurred: luma={safe_contour_luma:.2f}"
    assert standard_contour_luma > safe_contour_luma + 4.0, (
        f"fixture did not expose standard blur: standard={standard_contour_luma:.2f}, "
        f"safe={safe_contour_luma:.2f}"
    )

    boundary_source_luma = mean_luma(source, (36, 0, 38, 80))
    boundary_safe_luma = mean_luma(edge_safe, (36, 0, 38, 80))
    assert abs(boundary_safe_luma - boundary_source_luma) <= 3.0, (
        f"edge-safe mode changed pixels next to the contour: "
        f"source={boundary_source_luma:.2f}, safe={boundary_safe_luma:.2f}"
    )

    print(
        "edge-safe bilateral passed: "
        f"flat_variance {source_variance:.2f}->{safe_variance:.2f}, "
        f"contour_luma standard={standard_contour_luma:.2f}, safe={safe_contour_luma:.2f}"
    )


if __name__ == "__main__":
    main()
