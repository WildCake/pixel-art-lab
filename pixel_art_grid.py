#!/usr/bin/env python3
"""Convert generated/faux pixel art into a strict low-resolution pixel grid.

The tool is intentionally small and deterministic:
- crop to a target aspect ratio;
- resize to the logical pixel canvas;
- lightly sharpen/color-grade before palette reduction;
- quantize to a strict palette;
- optionally apply ordered Bayer dithering;
- export a nearest-neighbor preview for review.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing dependency hint.
    raise SystemExit(
        "Missing dependency: Pillow. Install with: python3 -m pip install -r requirements.txt"
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - the pure Python path remains supported.
    np = None

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - optional accelerator, keep pure Python fallback.
    njit = None
    prange = range


BAYER_4X4 = (
    (0, 8, 2, 10),
    (12, 4, 14, 6),
    (3, 11, 1, 9),
    (15, 7, 13, 5),
)
BAYER_4X4_ARRAY = np.asarray(BAYER_4X4, dtype=np.float64) if np is not None else None
EDGE_SAFE_BILATERAL_EDGE_THRESHOLD = 0.16
EDGE_SAFE_BILATERAL_DARK_LUMA = 58.0
EDGE_SAFE_BILATERAL_DARK_CONTRAST = 24.0
EDGE_SAFE_BILATERAL_MAX_BLEND = 0.92


@dataclass(frozen=True)
class PixelArtConfig:
    target_width: int
    target_height: int
    colors: int
    preview_scale: int
    dither: str
    dither_strength: float
    dither_scope: str
    dither_edge_threshold: float
    dither_luma_range: float
    dither_error_threshold: float
    saturation: float
    contrast: float
    sharpness: float
    autocontrast_cutoff: float
    resample: str
    grid_snap_enabled: bool
    grid_snap_method: str
    grid_snap_quantize_first: bool
    grid_snap_dark_threshold: float
    preserve_luma: bool
    preserve_saturation: bool
    palette_source: Path | None
    bilateral_radius: int
    bilateral_mode: str
    bilateral_sigma_color: float
    bilateral_sigma_space: float
    edge_palette_weight: float
    edge_sharpen: float
    edge_threshold: float
    palette_strategy: str
    color_distance: str
    accent_palette_weight: float
    hue_rarity_weight: float
    interesting_color_slots: int
    interesting_min_saturation: float
    interesting_min_value: float
    protected_hue_ranges: tuple[tuple[float, float], ...]
    protected_hue_weight: float
    protected_hue_slots: int
    protected_hue_min_saturation: float
    hue_match_weight: float
    flat_region_palette_colors: int
    flat_region_channel_step: int
    flat_region_max_saturation: float
    flat_region_edge_threshold: float
    flat_region_luma_range: float
    mixel_cleanup_passes: int
    mixel_cleanup_min_neighbors: int
    mixel_cleanup_distance: float
    mixel_cleanup_max_saturation: float


@dataclass
class ColorBox:
    colors: list[tuple[tuple[int, int, int], float]]
    weight: float
    min_rgb: tuple[int, int, int]
    max_rgb: tuple[int, int, int]

    @classmethod
    def from_colors(cls, colors: list[tuple[tuple[int, int, int], float]]) -> "ColorBox":
        weight = sum(count for _color, count in colors)
        reds = [color[0] for color, _count in colors]
        greens = [color[1] for color, _count in colors]
        blues = [color[2] for color, _count in colors]
        return cls(
            colors=colors,
            weight=weight,
            min_rgb=(min(reds), min(greens), min(blues)),
            max_rgb=(max(reds), max(greens), max(blues)),
        )

    def range_for_channel(self, channel: int) -> int:
        return self.max_rgb[channel] - self.min_rgb[channel]

    def largest_range(self) -> int:
        return max(self.range_for_channel(0), self.range_for_channel(1), self.range_for_channel(2))


def parse_size(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT, for example 320x180")
    left, right = value.lower().split("x", 1)
    try:
        width = int(left)
        height = int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("width and height must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def parse_hue_ranges(value: str) -> tuple[tuple[float, float], ...]:
    if not value.strip():
        return ()
    ranges: list[tuple[float, float]] = []
    for chunk in value.split(","):
        if "-" not in chunk:
            raise argparse.ArgumentTypeError("hue ranges must look like START-END, separated by commas")
        start, end = chunk.split("-", 1)
        try:
            ranges.append((float(start), float(end)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("hue range values must be numbers") from exc
    return tuple(ranges)


@lru_cache(maxsize=262144)
def srgb_luma(color: tuple[int, int, int]) -> float:
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


@lru_cache(maxsize=262144)
def rgb_saturation(color: tuple[int, int, int]) -> float:
    value = max(color)
    if value <= 0:
        return 0.0
    return (value - min(color)) / value


@lru_cache(maxsize=262144)
def rgb_hue(color: tuple[int, int, int]) -> float:
    red, green, blue = [channel / 255.0 for channel in color]
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    chroma = maximum - minimum
    if chroma <= 0:
        return 0.0
    if math.isclose(maximum, red):
        hue = ((green - blue) / chroma) % 6
    elif math.isclose(maximum, green):
        hue = ((blue - red) / chroma) + 2
    else:
        hue = ((red - green) / chroma) + 4
    return hue * 60.0


def rgb_array(image: Image.Image, dtype=None):
    if np is None:
        return None
    return np.asarray(image.convert("RGB"), dtype=dtype if dtype is not None else np.float64)


def saturation_value_arrays(rgb):
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.zeros_like(maximum, dtype=np.float64)
    np.divide(
        maximum - minimum,
        maximum,
        out=saturation,
        where=maximum > 0,
    )
    return saturation, maximum / 255.0


def hue_array(rgb):
    red = rgb[:, :, 0] / 255.0
    green = rgb[:, :, 1] / 255.0
    blue = rgb[:, :, 2] / 255.0
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    chroma = maximum - minimum
    hue = np.zeros_like(maximum, dtype=np.float64)
    mask = chroma > 0

    red_mask = mask & (maximum == red)
    green_mask = mask & ~red_mask & (maximum == green)
    blue_mask = mask & ~red_mask & ~green_mask

    hue[red_mask] = np.mod((green[red_mask] - blue[red_mask]) / chroma[red_mask], 6.0)
    hue[green_mask] = ((blue[green_mask] - red[green_mask]) / chroma[green_mask]) + 2.0
    hue[blue_mask] = ((red[blue_mask] - green[blue_mask]) / chroma[blue_mask]) + 4.0
    return hue * 60.0


def hue_ranges_mask_array(hue, ranges: tuple[tuple[float, float], ...]):
    mask = np.zeros(hue.shape, dtype=bool)
    for start, end in ranges:
        start %= 360.0
        end %= 360.0
        if start <= end:
            mask |= (hue >= start) & (hue <= end)
        else:
            mask |= (hue >= start) | (hue <= end)
    return mask


def pack_rgb_array(rgb):
    packed = (
        (rgb[:, :, 0].astype(np.uint32) << 16)
        | (rgb[:, :, 1].astype(np.uint32) << 8)
        | rgb[:, :, 2].astype(np.uint32)
    )
    return packed


def unpack_packed_rgb_channels(values, dtype=None):
    packed = values.astype(np.uint32)
    target_dtype = dtype if dtype is not None else np.float64
    return (
        ((packed >> 16) & 255).astype(target_dtype),
        ((packed >> 8) & 255).astype(target_dtype),
        (packed & 255).astype(target_dtype),
    )


def saturation_value_from_channels(red, green, blue):
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    saturation = np.zeros_like(maximum, dtype=np.float64)
    np.divide(
        maximum - minimum,
        maximum,
        out=saturation,
        where=maximum > 0,
    )
    return saturation, maximum / 255.0


def hue_from_channels(red, green, blue):
    red_normalized = red / 255.0
    green_normalized = green / 255.0
    blue_normalized = blue / 255.0
    maximum = np.maximum(np.maximum(red_normalized, green_normalized), blue_normalized)
    minimum = np.minimum(np.minimum(red_normalized, green_normalized), blue_normalized)
    chroma = maximum - minimum
    hue = np.zeros_like(maximum, dtype=np.float64)
    mask = chroma > 0

    red_mask = mask & (maximum == red_normalized)
    green_mask = mask & ~red_mask & (maximum == green_normalized)
    blue_mask = mask & ~red_mask & ~green_mask

    hue[red_mask] = np.mod(
        (green_normalized[red_mask] - blue_normalized[red_mask]) / chroma[red_mask],
        6.0,
    )
    hue[green_mask] = (
        (blue_normalized[green_mask] - red_normalized[green_mask]) / chroma[green_mask]
    ) + 2.0
    hue[blue_mask] = (
        (red_normalized[blue_mask] - green_normalized[blue_mask]) / chroma[blue_mask]
    ) + 4.0
    return hue * 60.0


def unpack_rgb_value(value: int) -> tuple[int, int, int]:
    return (
        int((value >> 16) & 255),
        int((value >> 8) & 255),
        int(value & 255),
    )


def hue_in_ranges(hue: float, ranges: tuple[tuple[float, float], ...]) -> bool:
    for start, end in ranges:
        start %= 360.0
        end %= 360.0
        if start <= end:
            if start <= hue <= end:
                return True
        elif hue >= start or hue <= end:
            return True
    return False


def hue_distance_degrees(left: float, right: float) -> float:
    distance = abs((left - right) % 360.0)
    return min(distance, 360.0 - distance)


def color_distance_squared(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    return (
        0.30 * (left[0] - right[0]) * (left[0] - right[0])
        + 0.59 * (left[1] - right[1]) * (left[1] - right[1])
        + 0.11 * (left[2] - right[2]) * (left[2] - right[2])
    )


def srgb_to_linear(channel: int) -> float:
    value = channel / 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


@lru_cache(maxsize=65536)
def oklab_color(color: tuple[int, int, int]) -> tuple[float, float, float]:
    red = srgb_to_linear(color[0])
    green = srgb_to_linear(color[1])
    blue = srgb_to_linear(color[2])

    long = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    short = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue

    long_root = math.copysign(abs(long) ** (1 / 3), long)
    medium_root = math.copysign(abs(medium) ** (1 / 3), medium)
    short_root = math.copysign(abs(short) ** (1 / 3), short)

    lightness = 0.2104542553 * long_root + 0.7936177850 * medium_root - 0.0040720468 * short_root
    green_red = 1.9779984951 * long_root - 2.4285922050 * medium_root + 0.4505937099 * short_root
    blue_yellow = 0.0259040371 * long_root + 0.7827717662 * medium_root - 0.8086757660 * short_root
    return lightness, green_red, blue_yellow


def oklab_distance_squared(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    left_lab = oklab_color(left)
    right_lab = oklab_color(right)
    lightness = left_lab[0] - right_lab[0]
    green_red = left_lab[1] - right_lab[1]
    blue_yellow = left_lab[2] - right_lab[2]
    return 65025.0 * (
        lightness * lightness
        + 1.8 * green_red * green_red
        + 1.8 * blue_yellow * blue_yellow
    )


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if math.isclose(edge0, edge1):
        return 1.0 if value >= edge1 else 0.0
    x = max(0.0, min(1.0, (value - edge0) / (edge1 - edge0)))
    return x * x * (3 - 2 * x)


if njit is not None and np is not None:

    @njit(cache=True)
    def _numba_clamp_channel(value: float) -> int:
        rounded = round(value)
        if rounded < 0:
            return 0
        if rounded > 255:
            return 255
        return int(rounded)


    @njit(cache=True)
    def _numba_rgb_saturation(red: int, green: int, blue: int) -> float:
        maximum = red
        if green > maximum:
            maximum = green
        if blue > maximum:
            maximum = blue
        if maximum <= 0:
            return 0.0

        minimum = red
        if green < minimum:
            minimum = green
        if blue < minimum:
            minimum = blue
        return (maximum - minimum) / maximum


    @njit(cache=True)
    def _numba_rgb_hue(red: int, green: int, blue: int) -> float:
        red_float = red / 255.0
        green_float = green / 255.0
        blue_float = blue / 255.0
        maximum = red_float
        if green_float > maximum:
            maximum = green_float
        if blue_float > maximum:
            maximum = blue_float

        minimum = red_float
        if green_float < minimum:
            minimum = green_float
        if blue_float < minimum:
            minimum = blue_float

        chroma = maximum - minimum
        if chroma <= 0:
            return 0.0
        if maximum == red_float:
            hue = ((green_float - blue_float) / chroma) % 6.0
        elif maximum == green_float:
            hue = ((blue_float - red_float) / chroma) + 2.0
        else:
            hue = ((red_float - green_float) / chroma) + 4.0
        return hue * 60.0


    @njit(cache=True)
    def _numba_hue_in_ranges(hue: float, ranges) -> bool:
        for index in range(ranges.shape[0]):
            start = ranges[index, 0] % 360.0
            end = ranges[index, 1] % 360.0
            if start <= end:
                if start <= hue <= end:
                    return True
            elif hue >= start or hue <= end:
                return True
        return False


    @njit(cache=True)
    def _numba_is_protected_hue(
        red: int,
        green: int,
        blue: int,
        ranges,
        min_saturation: float,
    ) -> bool:
        if ranges.shape[0] == 0:
            return False
        saturation = _numba_rgb_saturation(red, green, blue)
        if saturation < min_saturation:
            return False
        return _numba_hue_in_ranges(_numba_rgb_hue(red, green, blue), ranges)


    @njit(cache=True)
    def _numba_hue_distance_degrees(left: float, right: float) -> float:
        distance = abs((left - right) % 360.0)
        if distance < 360.0 - distance:
            return distance
        return 360.0 - distance


    @njit(cache=True)
    def _numba_smoothstep(edge0: float, edge1: float, value: float) -> float:
        if edge0 == edge1:
            if value >= edge1:
                return 1.0
            return 0.0
        x = (value - edge0) / (edge1 - edge0)
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        return x * x * (3.0 - 2.0 * x)


    @njit(cache=True)
    def _numba_srgb_luma(red: int, green: int, blue: int) -> float:
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue


    @njit(cache=True)
    def _numba_local_luma_range(source, x: int, y: int) -> float:
        height = source.shape[0]
        width = source.shape[1]
        minimum = 255.0
        maximum = 0.0
        for dy in range(-1, 2):
            yy = y + dy
            if yy < 0:
                yy = 0
            elif yy >= height:
                yy = height - 1
            for dx in range(-1, 2):
                xx = x + dx
                if xx < 0:
                    xx = 0
                elif xx >= width:
                    xx = width - 1
                luma = _numba_srgb_luma(
                    np.int64(source[yy, xx, 0]),
                    np.int64(source[yy, xx, 1]),
                    np.int64(source[yy, xx, 2]),
                )
                if luma < minimum:
                    minimum = luma
                if luma > maximum:
                    maximum = luma
        return maximum - minimum


    @njit(cache=True)
    def _numba_color_distance_squared(
        left_red: int,
        left_green: int,
        left_blue: int,
        right_red: int,
        right_green: int,
        right_blue: int,
    ) -> float:
        return (
            0.30 * (left_red - right_red) * (left_red - right_red)
            + 0.59 * (left_green - right_green) * (left_green - right_green)
            + 0.11 * (left_blue - right_blue) * (left_blue - right_blue)
        )


    @njit(cache=True)
    def _numba_srgb_to_linear(channel: int) -> float:
        value = channel / 255.0
        if value <= 0.04045:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4


    @njit(cache=True)
    def _numba_oklab_distance_squared(
        left_red: int,
        left_green: int,
        left_blue: int,
        right_red: int,
        right_green: int,
        right_blue: int,
    ) -> float:
        left_red_linear = _numba_srgb_to_linear(left_red)
        left_green_linear = _numba_srgb_to_linear(left_green)
        left_blue_linear = _numba_srgb_to_linear(left_blue)
        right_red_linear = _numba_srgb_to_linear(right_red)
        right_green_linear = _numba_srgb_to_linear(right_green)
        right_blue_linear = _numba_srgb_to_linear(right_blue)

        left_long = (
            0.4122214708 * left_red_linear
            + 0.5363325363 * left_green_linear
            + 0.0514459929 * left_blue_linear
        )
        left_medium = (
            0.2119034982 * left_red_linear
            + 0.6806995451 * left_green_linear
            + 0.1073969566 * left_blue_linear
        )
        left_short = (
            0.0883024619 * left_red_linear
            + 0.2817188376 * left_green_linear
            + 0.6299787005 * left_blue_linear
        )
        right_long = (
            0.4122214708 * right_red_linear
            + 0.5363325363 * right_green_linear
            + 0.0514459929 * right_blue_linear
        )
        right_medium = (
            0.2119034982 * right_red_linear
            + 0.6806995451 * right_green_linear
            + 0.1073969566 * right_blue_linear
        )
        right_short = (
            0.0883024619 * right_red_linear
            + 0.2817188376 * right_green_linear
            + 0.6299787005 * right_blue_linear
        )

        left_long_root = math.copysign(abs(left_long) ** (1.0 / 3.0), left_long)
        left_medium_root = math.copysign(abs(left_medium) ** (1.0 / 3.0), left_medium)
        left_short_root = math.copysign(abs(left_short) ** (1.0 / 3.0), left_short)
        right_long_root = math.copysign(abs(right_long) ** (1.0 / 3.0), right_long)
        right_medium_root = math.copysign(abs(right_medium) ** (1.0 / 3.0), right_medium)
        right_short_root = math.copysign(abs(right_short) ** (1.0 / 3.0), right_short)

        left_lightness = (
            0.2104542553 * left_long_root
            + 0.7936177850 * left_medium_root
            - 0.0040720468 * left_short_root
        )
        left_green_red = (
            1.9779984951 * left_long_root
            - 2.4285922050 * left_medium_root
            + 0.4505937099 * left_short_root
        )
        left_blue_yellow = (
            0.0259040371 * left_long_root
            + 0.7827717662 * left_medium_root
            - 0.8086757660 * left_short_root
        )
        right_lightness = (
            0.2104542553 * right_long_root
            + 0.7936177850 * right_medium_root
            - 0.0040720468 * right_short_root
        )
        right_green_red = (
            1.9779984951 * right_long_root
            - 2.4285922050 * right_medium_root
            + 0.4505937099 * right_short_root
        )
        right_blue_yellow = (
            0.0259040371 * right_long_root
            + 0.7827717662 * right_medium_root
            - 0.8086757660 * right_short_root
        )

        lightness = left_lightness - right_lightness
        green_red = left_green_red - right_green_red
        blue_yellow = left_blue_yellow - right_blue_yellow
        return 65025.0 * (
            lightness * lightness
            + 1.8 * green_red * green_red
            + 1.8 * blue_yellow * blue_yellow
        )


    @njit(cache=True)
    def _numba_nearest_palette_index(
        red: int,
        green: int,
        blue: int,
        palette,
        hue_match_weight: float,
        color_distance_mode: int,
    ) -> int:
        source_saturation = _numba_rgb_saturation(red, green, blue)
        source_hue = 0.0
        if hue_match_weight > 0 and source_saturation >= 0.08:
            source_hue = _numba_rgb_hue(red, green, blue)

        best_index = 0
        best_score = 1.0e308
        for index in range(palette.shape[0]):
            item_red = np.int64(palette[index, 0])
            item_green = np.int64(palette[index, 1])
            item_blue = np.int64(palette[index, 2])
            if color_distance_mode == 1:
                rgb_distance = _numba_oklab_distance_squared(
                    red,
                    green,
                    blue,
                    item_red,
                    item_green,
                    item_blue,
                )
            else:
                rgb_distance = _numba_color_distance_squared(
                    red,
                    green,
                    blue,
                    item_red,
                    item_green,
                    item_blue,
                )

            score = rgb_distance
            if hue_match_weight > 0 and source_saturation >= 0.08:
                item_saturation = _numba_rgb_saturation(item_red, item_green, item_blue)
                if item_saturation >= 0.08:
                    hue_distance = _numba_hue_distance_degrees(
                        source_hue,
                        _numba_rgb_hue(item_red, item_green, item_blue),
                    ) / 180.0
                    saturation_distance = abs(source_saturation - item_saturation)
                    score = rgb_distance + hue_match_weight * 65025.0 * (
                        hue_distance * hue_distance * source_saturation * item_saturation
                        + 0.2 * saturation_distance * saturation_distance
                    )

            if score < best_score:
                best_score = score
                best_index = index
        return best_index


    @njit(cache=True, parallel=True)
    def _nearest_palette_map_numba(colors, palette, hue_match_weight: float, color_distance_mode: int):
        mapped = np.empty((colors.shape[0], 3), dtype=np.uint8)
        for index in prange(colors.shape[0]):
            packed = np.uint32(colors[index])
            red = np.int64((packed >> 16) & 255)
            green = np.int64((packed >> 8) & 255)
            blue = np.int64(packed & 255)
            palette_index = _numba_nearest_palette_index(
                red,
                green,
                blue,
                palette,
                hue_match_weight,
                color_distance_mode,
            )
            mapped[index, 0] = palette[palette_index, 0]
            mapped[index, 1] = palette[palette_index, 1]
            mapped[index, 2] = palette[palette_index, 2]
        return mapped


    @njit(cache=True, parallel=True)
    def _nearest_palette_indices_numba(
        colors,
        palette,
        hue_match_weight: float,
        color_distance_mode: int,
    ):
        mapped_indices = np.empty(colors.shape[0], dtype=np.int64)
        for index in prange(colors.shape[0]):
            packed = np.uint32(colors[index])
            red = np.int64((packed >> 16) & 255)
            green = np.int64((packed >> 8) & 255)
            blue = np.int64(packed & 255)
            mapped_indices[index] = _numba_nearest_palette_index(
                red,
                green,
                blue,
                palette,
                hue_match_weight,
                color_distance_mode,
            )
        return mapped_indices


    @njit(cache=True, parallel=True)
    def _nearest_palette_map_rare_guard_numba(
        colors,
        palette,
        rare_seeds,
        hue_match_weight: float,
        color_distance_mode: int,
    ):
        mapped = np.empty((colors.shape[0], 3), dtype=np.uint8)
        for index in prange(colors.shape[0]):
            packed = np.uint32(colors[index])
            red = np.int64((packed >> 16) & 255)
            green = np.int64((packed >> 8) & 255)
            blue = np.int64(packed & 255)
            palette_index = _numba_nearest_palette_index(
                red,
                green,
                blue,
                palette,
                hue_match_weight,
                color_distance_mode,
            )
            mapped_red = np.int64(palette[palette_index, 0])
            mapped_green = np.int64(palette[palette_index, 1])
            mapped_blue = np.int64(palette[palette_index, 2])

            source_saturation = _numba_rgb_saturation(red, green, blue)
            if source_saturation >= 0.035 and rare_seeds.shape[0] > 0:
                source_hue = _numba_rgb_hue(red, green, blue)
                best_seed_index = 0
                best_seed_score = 1.0e308
                for seed_index in range(rare_seeds.shape[0]):
                    seed_red = np.int64(rare_seeds[seed_index, 0])
                    seed_green = np.int64(rare_seeds[seed_index, 1])
                    seed_blue = np.int64(rare_seeds[seed_index, 2])
                    hue_distance = _numba_hue_distance_degrees(
                        source_hue,
                        _numba_rgb_hue(seed_red, seed_green, seed_blue),
                    ) / 180.0
                    score = _numba_oklab_distance_squared(
                        red,
                        green,
                        blue,
                        seed_red,
                        seed_green,
                        seed_blue,
                    ) + 65025.0 * hue_distance * hue_distance * max(source_saturation, 0.08)
                    if score < best_seed_score:
                        best_seed_score = score
                        best_seed_index = seed_index

                best_seed_red = np.int64(rare_seeds[best_seed_index, 0])
                best_seed_green = np.int64(rare_seeds[best_seed_index, 1])
                best_seed_blue = np.int64(rare_seeds[best_seed_index, 2])
                hue_distance = _numba_hue_distance_degrees(
                    source_hue,
                    _numba_rgb_hue(best_seed_red, best_seed_green, best_seed_blue),
                )
                if hue_distance <= 42.0:
                    if color_distance_mode == 1:
                        normal_distance = _numba_oklab_distance_squared(
                            red,
                            green,
                            blue,
                            mapped_red,
                            mapped_green,
                            mapped_blue,
                        )
                        seed_distance = _numba_oklab_distance_squared(
                            red,
                            green,
                            blue,
                            best_seed_red,
                            best_seed_green,
                            best_seed_blue,
                        )
                    else:
                        normal_distance = _numba_color_distance_squared(
                            red,
                            green,
                            blue,
                            mapped_red,
                            mapped_green,
                            mapped_blue,
                        )
                        seed_distance = _numba_color_distance_squared(
                            red,
                            green,
                            blue,
                            best_seed_red,
                            best_seed_green,
                            best_seed_blue,
                        )
                    if seed_distance <= normal_distance * 1.55 + 420.0:
                        mapped_red = best_seed_red
                        mapped_green = best_seed_green
                        mapped_blue = best_seed_blue

            mapped[index, 0] = mapped_red
            mapped[index, 1] = mapped_green
            mapped[index, 2] = mapped_blue
        return mapped


    @njit(cache=True, parallel=True)
    def _project_palette_mass_numba(
        colors,
        counts,
        palette,
        hue_match_weight: float,
        color_distance_mode: int,
    ):
        mapped_indices = np.empty(colors.shape[0], dtype=np.int64)
        for index in prange(colors.shape[0]):
            packed = np.uint32(colors[index])
            red = np.int64((packed >> 16) & 255)
            green = np.int64((packed >> 8) & 255)
            blue = np.int64(packed & 255)
            mapped_indices[index] = _numba_nearest_palette_index(
                red,
                green,
                blue,
                palette,
                hue_match_weight,
                color_distance_mode,
            )

        projected = np.zeros(palette.shape[0], dtype=np.float64)
        for index in range(colors.shape[0]):
            projected[mapped_indices[index]] += counts[index]
        return projected


    @njit(cache=True, parallel=True)
    def _project_island_scores_numba(
        source,
        palette,
        tile_size: int,
        hue_match_weight: float,
        color_distance_mode: int,
    ):
        height = source.shape[0]
        width = source.shape[1]
        candidate_count = palette.shape[0]
        tile_rows = (height + tile_size - 1) // tile_size
        tile_cols = (width + tile_size - 1) // tile_size
        tile_count = tile_rows * tile_cols
        tile_counts = np.zeros((tile_count, candidate_count), dtype=np.float64)

        for tile_index in prange(tile_count):
            tile_row = tile_index // tile_cols
            tile_col = tile_index - tile_row * tile_cols
            tile_top = tile_row * tile_size
            tile_left = tile_col * tile_size
            y_end = tile_top + tile_size
            x_end = tile_left + tile_size
            if y_end > height:
                y_end = height
            if x_end > width:
                x_end = width

            local_counts = np.zeros(candidate_count, dtype=np.float64)
            for y in range(tile_top, y_end):
                for x in range(tile_left, x_end):
                    red = np.int64(source[y, x, 0])
                    green = np.int64(source[y, x, 1])
                    blue = np.int64(source[y, x, 2])
                    palette_index = _numba_nearest_palette_index(
                        red,
                        green,
                        blue,
                        palette,
                        hue_match_weight,
                        color_distance_mode,
                    )
                    local_counts[palette_index] += 1.0

            for index in range(candidate_count):
                tile_counts[tile_index, index] = local_counts[index]

        mass = np.zeros(candidate_count, dtype=np.float64)
        island_score = np.zeros(candidate_count, dtype=np.float64)
        for tile_index in range(tile_count):
            tile_total = 0.0
            for index in range(candidate_count):
                tile_total += tile_counts[tile_index, index]
            if tile_total <= 0.0:
                continue

            for index in range(candidate_count):
                count = tile_counts[tile_index, index]
                if count <= 0.0:
                    continue
                red = np.int64(palette[index, 0])
                green = np.int64(palette[index, 1])
                blue = np.int64(palette[index, 2])
                saturation = _numba_rgb_saturation(red, green, blue)
                value = red
                if green > value:
                    value = green
                if blue > value:
                    value = blue
                value_float = value / 255.0
                local_fraction = count / tile_total
                mass[index] += count
                island_score[index] += (
                    math.sqrt(count)
                    * ((1.0 - local_fraction) ** 0.35)
                    * ((0.35 + saturation) ** 1.25)
                    * (0.35 + value_float)
                )

        return mass, island_score


    @njit(cache=True, parallel=True)
    def _project_island_scores_from_indices_numba(mapped_indices, palette, tile_size: int):
        height = mapped_indices.shape[0]
        width = mapped_indices.shape[1]
        candidate_count = palette.shape[0]
        tile_rows = (height + tile_size - 1) // tile_size
        tile_cols = (width + tile_size - 1) // tile_size
        tile_count = tile_rows * tile_cols
        tile_counts = np.zeros((tile_count, candidate_count), dtype=np.float64)

        for tile_index in prange(tile_count):
            tile_row = tile_index // tile_cols
            tile_col = tile_index - tile_row * tile_cols
            tile_top = tile_row * tile_size
            tile_left = tile_col * tile_size
            y_end = tile_top + tile_size
            x_end = tile_left + tile_size
            if y_end > height:
                y_end = height
            if x_end > width:
                x_end = width

            local_counts = np.zeros(candidate_count, dtype=np.float64)
            for y in range(tile_top, y_end):
                for x in range(tile_left, x_end):
                    local_counts[mapped_indices[y, x]] += 1.0

            for index in range(candidate_count):
                tile_counts[tile_index, index] = local_counts[index]

        mass = np.zeros(candidate_count, dtype=np.float64)
        island_score = np.zeros(candidate_count, dtype=np.float64)
        for tile_index in range(tile_count):
            tile_total = 0.0
            for index in range(candidate_count):
                tile_total += tile_counts[tile_index, index]
            if tile_total <= 0.0:
                continue

            for index in range(candidate_count):
                count = tile_counts[tile_index, index]
                if count <= 0.0:
                    continue
                red = np.int64(palette[index, 0])
                green = np.int64(palette[index, 1])
                blue = np.int64(palette[index, 2])
                saturation = _numba_rgb_saturation(red, green, blue)
                value = red
                if green > value:
                    value = green
                if blue > value:
                    value = blue
                value_float = value / 255.0
                local_fraction = count / tile_total
                mass[index] += count
                island_score[index] += (
                    math.sqrt(count)
                    * ((1.0 - local_fraction) ** 0.35)
                    * ((0.35 + saturation) ** 1.25)
                    * (0.35 + value_float)
                )

        return mass, island_score


    @njit(cache=True, parallel=True)
    def _project_frontier_scores_from_indices_numba(mapped_indices, source, edge_mask, palette, tile_size: int):
        height = mapped_indices.shape[0]
        width = mapped_indices.shape[1]
        candidate_count = palette.shape[0]
        tile_rows = (height + tile_size - 1) // tile_size
        tile_cols = (width + tile_size - 1) // tile_size
        tile_count = tile_rows * tile_cols
        tile_mass = np.zeros((tile_count, candidate_count), dtype=np.float64)
        tile_contour = np.zeros((tile_count, candidate_count), dtype=np.float64)
        tile_counts = np.zeros((tile_count, candidate_count), dtype=np.float64)

        for tile_index in prange(tile_count):
            tile_row = tile_index // tile_cols
            tile_col = tile_index - tile_row * tile_cols
            tile_top = tile_row * tile_size
            tile_left = tile_col * tile_size
            y_end = tile_top + tile_size
            x_end = tile_left + tile_size
            if y_end > height:
                y_end = height
            if x_end > width:
                x_end = width

            local_mass = np.zeros(candidate_count, dtype=np.float64)
            local_contour = np.zeros(candidate_count, dtype=np.float64)
            local_counts = np.zeros(candidate_count, dtype=np.float64)
            for y in range(tile_top, y_end):
                for x in range(tile_left, x_end):
                    palette_index = mapped_indices[y, x]
                    red = np.int64(source[y, x, 0])
                    green = np.int64(source[y, x, 1])
                    blue = np.int64(source[y, x, 2])
                    saturation = _numba_rgb_saturation(red, green, blue)
                    value = red
                    if green > value:
                        value = green
                    if blue > value:
                        value = blue
                    value_float = value / 255.0
                    edge = edge_mask[y, x] / 255.0
                    local_mass[palette_index] += 1.0
                    local_contour[palette_index] += 1.0 + 4.0 * edge + 1.4 * edge * saturation
                    local_counts[palette_index] += 1.0 + 0.25 * saturation * value_float

            for index in range(candidate_count):
                tile_mass[tile_index, index] = local_mass[index]
                tile_contour[tile_index, index] = local_contour[index]
                tile_counts[tile_index, index] = local_counts[index]

        mass = np.zeros(candidate_count, dtype=np.float64)
        contour = np.zeros(candidate_count, dtype=np.float64)
        island = np.zeros(candidate_count, dtype=np.float64)
        for tile_index in range(tile_count):
            tile_total = 0.0
            for index in range(candidate_count):
                tile_total += tile_counts[tile_index, index]
            if tile_total <= 0.0:
                continue

            for index in range(candidate_count):
                count = tile_counts[tile_index, index]
                mass[index] += tile_mass[tile_index, index]
                contour[index] += tile_contour[tile_index, index]
                if count > 0.0:
                    island[index] += math.sqrt(count) * (1.0 - min(0.92, count / tile_total))

        return mass, contour, island


    @njit(cache=True, parallel=True)
    def _ordered_dither_numba(
        source,
        palette,
        bayer,
        strength: float,
        adaptive: int,
        edge_mask,
        edge_threshold: float,
        luma_range_threshold: float,
        error_threshold: float,
        hue_match_weight: float,
        color_distance_mode: int,
    ):
        height = source.shape[0]
        width = source.shape[1]
        output = np.empty_like(source)
        error_threshold_squared = error_threshold * error_threshold

        for y in prange(height):
            for x in range(width):
                red = np.int64(source[y, x, 0])
                green = np.int64(source[y, x, 1])
                blue = np.int64(source[y, x, 2])
                scale = 1.0

                if adaptive != 0:
                    edge = edge_mask[y, x] / 255.0
                    if edge_threshold <= 0:
                        edge_factor = 1.0 if edge <= 0 else 0.0
                    else:
                        edge_factor = 1.0 - _numba_smoothstep(
                            edge_threshold * 0.5,
                            edge_threshold,
                            edge,
                        )
                    local_range = _numba_local_luma_range(source, x, y)
                    if luma_range_threshold <= 0:
                        detail_factor = 1.0 if local_range <= 0 else 0.0
                    else:
                        detail_factor = 1.0 - _numba_smoothstep(
                            luma_range_threshold * 0.65,
                            luma_range_threshold,
                            local_range,
                        )
                    nearest_index = _numba_nearest_palette_index(
                        red,
                        green,
                        blue,
                        palette,
                        hue_match_weight,
                        color_distance_mode,
                    )
                    nearest_red = np.int64(palette[nearest_index, 0])
                    nearest_green = np.int64(palette[nearest_index, 1])
                    nearest_blue = np.int64(palette[nearest_index, 2])
                    quant_error = _numba_color_distance_squared(
                        red,
                        green,
                        blue,
                        nearest_red,
                        nearest_green,
                        nearest_blue,
                    )
                    if error_threshold <= 0:
                        error_factor = 1.0
                    else:
                        error_factor = _numba_smoothstep(
                            error_threshold_squared,
                            error_threshold_squared * 4.0,
                            quant_error,
                        )
                    scale = edge_factor * detail_factor * error_factor

                threshold = ((bayer[y % 4, x % 4] + 0.5) / 16.0 - 0.5) * strength * scale
                shifted_red = _numba_clamp_channel(red + threshold)
                shifted_green = _numba_clamp_channel(green + threshold)
                shifted_blue = _numba_clamp_channel(blue + threshold)
                palette_index = _numba_nearest_palette_index(
                    shifted_red,
                    shifted_green,
                    shifted_blue,
                    palette,
                    hue_match_weight,
                    color_distance_mode,
                )
                output[y, x, 0] = palette[palette_index, 0]
                output[y, x, 1] = palette[palette_index, 1]
                output[y, x, 2] = palette[palette_index, 2]

        return output


    @njit(cache=True)
    def _numba_add_diffused_error(
        work,
        y: int,
        x: int,
        err_red: float,
        err_green: float,
        err_blue: float,
        weight: float,
    ) -> None:
        work[y, x, 0] = min(255.0, max(0.0, work[y, x, 0] + err_red * weight))
        work[y, x, 1] = min(255.0, max(0.0, work[y, x, 1] + err_green * weight))
        work[y, x, 2] = min(255.0, max(0.0, work[y, x, 2] + err_blue * weight))


    @njit(cache=True)
    def _floyd_steinberg_dither_numba(
        source,
        palette,
        hue_match_weight: float,
        color_distance_mode: int,
    ):
        height = source.shape[0]
        width = source.shape[1]
        work = np.empty((height, width, 3), dtype=np.float64)
        output = np.empty_like(source)

        for y in range(height):
            for x in range(width):
                work[y, x, 0] = float(source[y, x, 0])
                work[y, x, 1] = float(source[y, x, 1])
                work[y, x, 2] = float(source[y, x, 2])

        for y in range(height):
            for x in range(width):
                red = _numba_clamp_channel(work[y, x, 0])
                green = _numba_clamp_channel(work[y, x, 1])
                blue = _numba_clamp_channel(work[y, x, 2])
                palette_index = _numba_nearest_palette_index(
                    red,
                    green,
                    blue,
                    palette,
                    hue_match_weight,
                    color_distance_mode,
                )
                new_red = np.int64(palette[palette_index, 0])
                new_green = np.int64(palette[palette_index, 1])
                new_blue = np.int64(palette[palette_index, 2])

                output[y, x, 0] = new_red
                output[y, x, 1] = new_green
                output[y, x, 2] = new_blue

                err_red = float(red - new_red)
                err_green = float(green - new_green)
                err_blue = float(blue - new_blue)

                if x + 1 < width:
                    _numba_add_diffused_error(
                        work,
                        y,
                        x + 1,
                        err_red,
                        err_green,
                        err_blue,
                        7.0 / 16.0,
                    )
                if y + 1 < height:
                    if x > 0:
                        _numba_add_diffused_error(
                            work,
                            y + 1,
                            x - 1,
                            err_red,
                            err_green,
                            err_blue,
                            3.0 / 16.0,
                        )
                    _numba_add_diffused_error(
                        work,
                        y + 1,
                        x,
                        err_red,
                        err_green,
                        err_blue,
                        5.0 / 16.0,
                    )
                    if x + 1 < width:
                        _numba_add_diffused_error(
                            work,
                            y + 1,
                            x + 1,
                            err_red,
                            err_green,
                            err_blue,
                            1.0 / 16.0,
                        )

        return output


    @njit(cache=True, parallel=True)
    def _bilateral_smooth_numba(source, radius: int, sigma_color2: float, spatial):
        height = source.shape[0]
        width = source.shape[1]
        out = np.empty_like(source)

        for y in prange(height):
            for x in range(width):
                center_red = float(source[y, x, 0])
                center_green = float(source[y, x, 1])
                center_blue = float(source[y, x, 2])
                weighted_red = 0.0
                weighted_green = 0.0
                weighted_blue = 0.0
                total_weight = 0.0

                for dy in range(-radius, radius + 1):
                    yy = y + dy
                    if yy < 0:
                        yy = 0
                    elif yy >= height:
                        yy = height - 1

                    for dx in range(-radius, radius + 1):
                        xx = x + dx
                        if xx < 0:
                            xx = 0
                        elif xx >= width:
                            xx = width - 1

                        red = float(source[yy, xx, 0])
                        green = float(source[yy, xx, 1])
                        blue = float(source[yy, xx, 2])
                        diff_red = red - center_red
                        diff_green = green - center_green
                        diff_blue = blue - center_blue
                        range_weight = math.exp(
                            -(
                                diff_red * diff_red
                                + diff_green * diff_green
                                + diff_blue * diff_blue
                            )
                            / sigma_color2
                        )
                        weight = spatial[dy + radius, dx + radius] * range_weight
                        weighted_red += red * weight
                        weighted_green += green * weight
                        weighted_blue += blue * weight
                        total_weight += weight

                out[y, x, 0] = _numba_clamp_channel(weighted_red / total_weight)
                out[y, x, 1] = _numba_clamp_channel(weighted_green / total_weight)
                out[y, x, 2] = _numba_clamp_channel(weighted_blue / total_weight)

        return out


    @njit(cache=True, parallel=True)
    def _edge_safe_bilateral_smooth_numba(
        source,
        edge_mask,
        radius: int,
        sigma_color2: float,
        spatial,
        edge_threshold: float,
        luma_gate: float,
        dark_luma: float,
        dark_contrast: float,
        boundary_distance2: float,
    ):
        height = source.shape[0]
        width = source.shape[1]
        out = np.empty_like(source)

        for y in prange(height):
            for x in range(width):
                center_red = float(source[y, x, 0])
                center_green = float(source[y, x, 1])
                center_blue = float(source[y, x, 2])
                center_luma = _numba_srgb_luma(
                    np.int64(source[y, x, 0]),
                    np.int64(source[y, x, 1]),
                    np.int64(source[y, x, 2]),
                )
                center_edge = float(edge_mask[y, x]) / 255.0
                local_range = _numba_local_luma_range(source, x, y)
                center_is_dark = center_luma <= dark_luma

                if center_is_dark and local_range >= dark_contrast:
                    out[y, x, 0] = source[y, x, 0]
                    out[y, x, 1] = source[y, x, 1]
                    out[y, x, 2] = source[y, x, 2]
                    continue

                weighted_red = 0.0
                weighted_green = 0.0
                weighted_blue = 0.0
                total_weight = 0.0

                for dy in range(-radius, radius + 1):
                    yy = y + dy
                    if yy < 0:
                        yy = 0
                    elif yy >= height:
                        yy = height - 1

                    for dx in range(-radius, radius + 1):
                        xx = x + dx
                        if xx < 0:
                            xx = 0
                        elif xx >= width:
                            xx = width - 1

                        red = float(source[yy, xx, 0])
                        green = float(source[yy, xx, 1])
                        blue = float(source[yy, xx, 2])
                        neighbor_luma = _numba_srgb_luma(
                            np.int64(source[yy, xx, 0]),
                            np.int64(source[yy, xx, 1]),
                            np.int64(source[yy, xx, 2]),
                        )
                        luma_delta = neighbor_luma - center_luma
                        if luma_delta < 0.0:
                            luma_delta = -luma_delta
                        if luma_delta > luma_gate:
                            continue

                        neighbor_is_dark = neighbor_luma <= dark_luma
                        if center_is_dark != neighbor_is_dark and luma_delta >= dark_contrast:
                            continue

                        color_distance = _numba_color_distance_squared(
                            np.int64(source[y, x, 0]),
                            np.int64(source[y, x, 1]),
                            np.int64(source[y, x, 2]),
                            np.int64(source[yy, xx, 0]),
                            np.int64(source[yy, xx, 1]),
                            np.int64(source[yy, xx, 2]),
                        )
                        if color_distance > boundary_distance2:
                            continue

                        diff_red = red - center_red
                        diff_green = green - center_green
                        diff_blue = blue - center_blue
                        range_weight = math.exp(
                            -(
                                diff_red * diff_red
                                + diff_green * diff_green
                                + diff_blue * diff_blue
                            )
                            / sigma_color2
                        )
                        neighbor_edge = float(edge_mask[yy, xx]) / 255.0
                        edge_value = center_edge if center_edge > neighbor_edge else neighbor_edge
                        edge_factor = 1.0 - 0.55 * _numba_smoothstep(
                            edge_threshold,
                            0.9,
                            edge_value,
                        )
                        luma_factor = 1.0 - 0.35 * _numba_smoothstep(luma_gate * 0.65, luma_gate, luma_delta)
                        weight = (
                            spatial[dy + radius, dx + radius]
                            * range_weight
                            * edge_factor
                            * luma_factor
                        )
                        weighted_red += red * weight
                        weighted_green += green * weight
                        weighted_blue += blue * weight
                        total_weight += weight

                if total_weight <= 1e-6:
                    out[y, x, 0] = source[y, x, 0]
                    out[y, x, 1] = source[y, x, 1]
                    out[y, x, 2] = source[y, x, 2]
                    continue

                smoothed_red = weighted_red / total_weight
                smoothed_green = weighted_green / total_weight
                smoothed_blue = weighted_blue / total_weight
                detail_factor = 1.0 - _numba_smoothstep(
                    luma_gate * 0.9,
                    luma_gate * 2.4,
                    local_range,
                ) * 0.45
                edge_factor = 1.0 - 0.55 * _numba_smoothstep(edge_threshold, 0.9, center_edge)
                blend = detail_factor * edge_factor
                if blend > EDGE_SAFE_BILATERAL_MAX_BLEND:
                    blend = EDGE_SAFE_BILATERAL_MAX_BLEND

                out[y, x, 0] = _numba_clamp_channel(center_red + (smoothed_red - center_red) * blend)
                out[y, x, 1] = _numba_clamp_channel(center_green + (smoothed_green - center_green) * blend)
                out[y, x, 2] = _numba_clamp_channel(center_blue + (smoothed_blue - center_blue) * blend)

        return out


    @njit(cache=True, parallel=True)
    def _cleanup_single_pixel_mixels_numba(
        source_image,
        passes: int,
        min_neighbors: int,
        max_distance: float,
        max_saturation: float,
        protected_ranges,
        protected_min_saturation: float,
    ):
        output = source_image.copy()
        height = output.shape[0]
        width = output.shape[1]

        for _pass in range(passes):
            source = output
            next_image = source.copy()
            changes = 0

            for y in prange(height):
                for x in range(width):
                    color_red = np.int64(source[y, x, 0])
                    color_green = np.int64(source[y, x, 1])
                    color_blue = np.int64(source[y, x, 2])

                    if _numba_is_protected_hue(
                        color_red,
                        color_green,
                        color_blue,
                        protected_ranges,
                        protected_min_saturation,
                    ):
                        continue
                    if _numba_rgb_saturation(color_red, color_green, color_blue) > max_saturation:
                        continue

                    unique = np.empty((9, 3), dtype=np.uint8)
                    counts = np.zeros(9, dtype=np.int64)
                    unique_count = 0

                    for dy in range(-1, 2):
                        yy = y + dy
                        if yy < 0:
                            yy = 0
                        elif yy >= height:
                            yy = height - 1

                        for dx in range(-1, 2):
                            xx = x + dx
                            if xx < 0:
                                xx = 0
                            elif xx >= width:
                                xx = width - 1

                            neighbor_red = source[yy, xx, 0]
                            neighbor_green = source[yy, xx, 1]
                            neighbor_blue = source[yy, xx, 2]
                            found = -1
                            for index in range(unique_count):
                                if (
                                    unique[index, 0] == neighbor_red
                                    and unique[index, 1] == neighbor_green
                                    and unique[index, 2] == neighbor_blue
                                ):
                                    found = index
                                    break
                            if found >= 0:
                                counts[found] += 1
                            else:
                                unique[unique_count, 0] = neighbor_red
                                unique[unique_count, 1] = neighbor_green
                                unique[unique_count, 2] = neighbor_blue
                                counts[unique_count] = 1
                                unique_count += 1

                    current_count = 0
                    replacement_index = 0
                    replacement_count = counts[0]
                    for index in range(unique_count):
                        if (
                            unique[index, 0] == color_red
                            and unique[index, 1] == color_green
                            and unique[index, 2] == color_blue
                        ):
                            current_count = counts[index]
                        if counts[index] > replacement_count:
                            replacement_count = counts[index]
                            replacement_index = index

                    replacement_red = np.int64(unique[replacement_index, 0])
                    replacement_green = np.int64(unique[replacement_index, 1])
                    replacement_blue = np.int64(unique[replacement_index, 2])
                    diff_red = float(color_red - replacement_red)
                    diff_green = float(color_green - replacement_green)
                    diff_blue = float(color_blue - replacement_blue)
                    distance = (
                        0.30 * diff_red * diff_red
                        + 0.59 * diff_green * diff_green
                        + 0.11 * diff_blue * diff_blue
                    )

                    if (
                        current_count <= 1
                        and replacement_count >= min_neighbors
                        and distance <= max_distance
                    ):
                        next_image[y, x, 0] = replacement_red
                        next_image[y, x, 1] = replacement_green
                        next_image[y, x, 2] = replacement_blue
                        changes += 1

            output = next_image
            if changes == 0:
                break

        return output


    @njit(cache=True, parallel=True)
    def _grid_snap_center_numba(
        source,
        output_width: int,
        output_height: int,
        origin_x: float,
        origin_y: float,
        cell_w: float,
        cell_h: float,
    ):
        source_height = source.shape[0]
        source_width = source.shape[1]
        output = np.empty((output_height, output_width, 3), dtype=np.uint8)

        for y in prange(output_height):
            sample_y = int(math.floor(origin_y + (y + 0.5) * cell_h))
            if sample_y < 0:
                sample_y = 0
            elif sample_y >= source_height:
                sample_y = source_height - 1
            for x in range(output_width):
                sample_x = int(math.floor(origin_x + (x + 0.5) * cell_w))
                if sample_x < 0:
                    sample_x = 0
                elif sample_x >= source_width:
                    sample_x = source_width - 1
                output[y, x, 0] = source[sample_y, sample_x, 0]
                output[y, x, 1] = source[sample_y, sample_x, 1]
                output[y, x, 2] = source[sample_y, sample_x, 2]

        return output


    @njit(cache=True, parallel=True)
    def _grid_snap_vote_numba(
        vote_source,
        detail_source,
        output_width: int,
        output_height: int,
        origin_x: float,
        origin_y: float,
        cell_w: float,
        cell_h: float,
        dark_threshold: float,
        dark_bias: int,
    ):
        source_height = vote_source.shape[0]
        source_width = vote_source.shape[1]
        output = np.empty((output_height, output_width, 3), dtype=np.uint8)
        max_unique = 256

        for y in prange(output_height):
            for x in range(output_width):
                x0 = int(math.floor(origin_x + x * cell_w))
                y0 = int(math.floor(origin_y + y * cell_h))
                x1 = int(math.ceil(origin_x + (x + 1) * cell_w))
                y1 = int(math.ceil(origin_y + (y + 1) * cell_h))

                if x0 < 0:
                    x0 = 0
                elif x0 >= source_width:
                    x0 = source_width - 1
                if y0 < 0:
                    y0 = 0
                elif y0 >= source_height:
                    y0 = source_height - 1
                if x1 <= x0:
                    x1 = x0 + 1
                if y1 <= y0:
                    y1 = y0 + 1
                if x1 > source_width:
                    x1 = source_width
                if y1 > source_height:
                    y1 = source_height

                unique = np.empty(max_unique, dtype=np.uint32)
                counts = np.zeros(max_unique, dtype=np.int64)
                unique_count = 0
                min_luma = 1.0e9
                total_pixels = 0

                for yy in range(y0, y1):
                    for xx in range(x0, x1):
                        red = np.uint32(vote_source[yy, xx, 0])
                        green = np.uint32(vote_source[yy, xx, 1])
                        blue = np.uint32(vote_source[yy, xx, 2])
                        packed = (red << 16) | (green << 8) | blue
                        found = -1
                        for index in range(unique_count):
                            if unique[index] == packed:
                                found = index
                                break
                        if found >= 0:
                            counts[found] += 1
                        elif unique_count < max_unique:
                            unique[unique_count] = packed
                            counts[unique_count] = 1
                            unique_count += 1

                        detail_red = float(detail_source[yy, xx, 0])
                        detail_green = float(detail_source[yy, xx, 1])
                        detail_blue = float(detail_source[yy, xx, 2])
                        luma = 0.2126 * detail_red + 0.7152 * detail_green + 0.0722 * detail_blue
                        if luma < min_luma:
                            min_luma = luma
                        total_pixels += 1

                best_index = 0
                best_count = counts[0]
                for index in range(1, unique_count):
                    if counts[index] > best_count:
                        best_count = counts[index]
                        best_index = index

                packed_mode = unique[best_index]
                mode_red = np.int64((packed_mode >> 16) & 255)
                mode_green = np.int64((packed_mode >> 8) & 255)
                mode_blue = np.int64(packed_mode & 255)
                mode_luma = 0.2126 * mode_red + 0.7152 * mode_green + 0.0722 * mode_blue

                if dark_bias == 1 and mode_luma - min_luma >= dark_threshold and total_pixels > 0:
                    dark_unique = np.empty(max_unique, dtype=np.uint32)
                    dark_counts = np.zeros(max_unique, dtype=np.int64)
                    dark_unique_count = 0
                    dark_total = 0
                    dark_limit = min_luma + max(8.0, dark_threshold * 0.35)

                    for yy in range(y0, y1):
                        for xx in range(x0, x1):
                            detail_red = float(detail_source[yy, xx, 0])
                            detail_green = float(detail_source[yy, xx, 1])
                            detail_blue = float(detail_source[yy, xx, 2])
                            luma = 0.2126 * detail_red + 0.7152 * detail_green + 0.0722 * detail_blue
                            if luma > dark_limit:
                                continue
                            red = np.uint32(vote_source[yy, xx, 0])
                            green = np.uint32(vote_source[yy, xx, 1])
                            blue = np.uint32(vote_source[yy, xx, 2])
                            packed = (red << 16) | (green << 8) | blue
                            found = -1
                            for index in range(dark_unique_count):
                                if dark_unique[index] == packed:
                                    found = index
                                    break
                            if found >= 0:
                                dark_counts[found] += 1
                            elif dark_unique_count < max_unique:
                                dark_unique[dark_unique_count] = packed
                                dark_counts[dark_unique_count] = 1
                                dark_unique_count += 1
                            dark_total += 1

                    if dark_unique_count > 0 and dark_total <= max(1, int(math.ceil(total_pixels * 0.42))):
                        dark_best_index = 0
                        dark_best_count = dark_counts[0]
                        for index in range(1, dark_unique_count):
                            if dark_counts[index] > dark_best_count:
                                dark_best_count = dark_counts[index]
                                dark_best_index = index
                        packed_mode = dark_unique[dark_best_index]

                output[y, x, 0] = np.uint8((packed_mode >> 16) & 255)
                output[y, x, 1] = np.uint8((packed_mode >> 8) & 255)
                output[y, x, 2] = np.uint8(packed_mode & 255)

        return output

else:
    _nearest_palette_map_numba = None
    _nearest_palette_indices_numba = None
    _nearest_palette_map_rare_guard_numba = None
    _project_palette_mass_numba = None
    _project_island_scores_numba = None
    _project_island_scores_from_indices_numba = None
    _project_frontier_scores_from_indices_numba = None
    _ordered_dither_numba = None
    _floyd_steinberg_dither_numba = None
    _bilateral_smooth_numba = None
    _edge_safe_bilateral_smooth_numba = None
    _cleanup_single_pixel_mixels_numba = None
    _grid_snap_center_numba = None
    _grid_snap_vote_numba = None


def center_crop_to_aspect(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    source_width, source_height = image.size
    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if math.isclose(source_aspect, target_aspect, rel_tol=1e-6):
        return image

    if source_aspect > target_aspect:
        crop_width = round(source_height * target_aspect)
        if source_width - crop_width <= 1:
            return image
        left = (source_width - crop_width) // 2
        box = (left, 0, left + crop_width, source_height)
    else:
        crop_height = round(source_width / target_aspect)
        if source_height - crop_height <= 1:
            return image
        top = (source_height - crop_height) // 2
        box = (0, top, source_width, top + crop_height)
    return image.crop(box)


def resize_filter(name: str) -> int:
    filters = {
        "box": Image.Resampling.BOX,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return filters[name]


def prepare_base_image(image: Image.Image, config: PixelArtConfig) -> Image.Image:
    image = center_crop_to_aspect(image.convert("RGB"), config.target_width, config.target_height)
    if config.grid_snap_enabled:
        return grid_snap_image(image, config)
    return image.resize(
        (config.target_width, config.target_height),
        resample=resize_filter(config.resample),
    )


def quantize_grid_source(image: Image.Image, config: PixelArtConfig) -> Image.Image:
    if config.colors <= 0:
        return image.convert("RGB")

    quantized, _palette = quantize_median_cut_rgb(
        image.convert("RGB"),
        max(2, int(config.colors)),
        edge_mask=None,
        palette_image=None,
        palette_edge_mask=None,
        edge_palette_weight=0.0,
        palette_strategy=config.palette_strategy,
        color_distance=config.color_distance,
        accent_palette_weight=config.accent_palette_weight,
        hue_rarity_weight=config.hue_rarity_weight,
        interesting_color_slots=config.interesting_color_slots,
        interesting_min_saturation=config.interesting_min_saturation,
        interesting_min_value=config.interesting_min_value,
        protected_hue_ranges=config.protected_hue_ranges,
        protected_hue_weight=config.protected_hue_weight,
        protected_hue_slots=config.protected_hue_slots,
        protected_hue_min_saturation=config.protected_hue_min_saturation,
        hue_match_weight=config.hue_match_weight,
    )
    return quantized.convert("RGB")


def grid_edge_profiles(image: Image.Image) -> tuple[object, object]:
    if np is None:
        return [], []
    rgb = rgb_array(image, dtype=np.float64)
    diff_x = rgb[:, 1:, :] - rgb[:, :-1, :]
    diff_y = rgb[1:, :, :] - rgb[:-1, :, :]
    profile_x = np.sqrt(np.sum(diff_x * diff_x, axis=2)).mean(axis=0)
    profile_y = np.sqrt(np.sum(diff_y * diff_y, axis=2)).mean(axis=1)
    return profile_x, profile_y


def grid_axis_score_and_origin(profile, period: float) -> tuple[float, float]:
    if np is None or len(profile) < 4 or period < 1.5:
        return 0.0, 0.0

    profile_array = np.asarray(profile, dtype=np.float64)
    spread = float(profile_array.std())
    if spread <= 1e-6:
        return 0.0, 0.0

    length = profile_array.shape[0]
    best_score = -1.0e9
    best_origin = 0.0
    phase_count = 12
    xs = np.arange(length, dtype=np.float64)
    for phase_index in range(phase_count):
        origin = ((phase_index / phase_count) - 0.5) * period
        positions = origin + np.arange(1, int((length - origin) / period) + 1, dtype=np.float64) * period
        positions = positions[(positions >= 0) & (positions <= length - 1)]
        if positions.size < 4:
            continue
        boundary = np.interp(positions, xs, profile_array)
        interior_positions = positions + period * 0.5
        interior_positions = interior_positions[interior_positions <= length - 1]
        if interior_positions.size < 4:
            continue
        interior = np.interp(interior_positions, xs, profile_array)
        boundary_mean = float(boundary.mean())
        interior_mean = float(interior.mean())
        boundary_peak = float(np.percentile(boundary, 78))
        score = ((0.65 * boundary_mean + 0.35 * boundary_peak) - interior_mean) / spread
        score *= math.log1p(float(positions.size)) ** 0.35
        if score > best_score:
            best_score = score
            best_origin = origin

    return max(0.0, best_score), best_origin


def detect_mixel_grid_variants(
    image: Image.Image,
    max_output_width: int = 4096,
    max_output_height: int = 1024,
    min_output_size: int = 16,
    max_variants: int = 9,
) -> list[dict[str, float | int]]:
    if np is None:
        return []

    source = image.convert("RGB")
    profile_x, profile_y = grid_edge_profiles(source)
    source_width, source_height = source.size
    min_height = max(min_output_size, int(math.ceil(source_height / 24.0)))
    max_height = min(max_output_height, max(min_output_size, int(math.floor(source_height / 2.0))))
    scored: list[dict[str, float | int]] = []

    for height in range(min_height, max_height + 1):
        cell_size = source_height / height
        if cell_size < 1.75 or cell_size > 32.0:
            continue
        width = max(min_output_size, int(round(source_width / cell_size)))
        if width > max_output_width:
            continue
        cell_x = source_width / width
        score_y, origin_y = grid_axis_score_and_origin(profile_y, cell_size)
        score_x, origin_x = grid_axis_score_and_origin(profile_x, cell_x)
        square_penalty = abs(cell_x - cell_size) / max(cell_size, 1e-6)
        score = score_y * 0.58 + score_x * 0.42 - square_penalty * 0.6
        if score <= 0:
            continue
        scored.append(
            {
                "width": width,
                "height": height,
                "cellSize": round((cell_size + cell_x) * 0.5, 3),
                "score": round(score, 4),
                "originX": round(origin_x, 3),
                "originY": round(origin_y, 3),
            }
        )

    ranked = sorted(scored, key=lambda item: float(item["score"]), reverse=True)
    variants: list[dict[str, float | int]] = []
    for item in ranked:
        if any(abs(int(item["height"]) - int(existing["height"])) < 4 for existing in variants):
            continue
        variants.append(item)
        if len(variants) >= max_variants:
            break
    return variants


def grid_snap_image(image: Image.Image, config: PixelArtConfig) -> Image.Image:
    source = image.convert("RGB")
    if np is None or _grid_snap_center_numba is None or _grid_snap_vote_numba is None:
        return source.resize(
            (config.target_width, config.target_height),
            resample=Image.Resampling.BOX,
        )

    detail_array = np.asarray(source, dtype=np.uint8)
    if config.grid_snap_quantize_first:
        vote_source = quantize_grid_source(source, config)
    else:
        vote_source = source
    vote_array = np.asarray(vote_source, dtype=np.uint8)

    profile_x, profile_y = grid_edge_profiles(source)
    cell_w = source.width / config.target_width
    cell_h = source.height / config.target_height
    _score_x, origin_x = grid_axis_score_and_origin(profile_x, cell_w)
    _score_y, origin_y = grid_axis_score_and_origin(profile_y, cell_h)

    if config.grid_snap_method == "center":
        output = _grid_snap_center_numba(
            vote_array,
            config.target_width,
            config.target_height,
            origin_x,
            origin_y,
            cell_w,
            cell_h,
        )
    else:
        output = _grid_snap_vote_numba(
            vote_array,
            detail_array,
            config.target_width,
            config.target_height,
            origin_x,
            origin_y,
            cell_w,
            cell_h,
            config.grid_snap_dark_threshold,
            1 if config.grid_snap_method == "dark-stroke" else 0,
        )
    return Image.fromarray(output, mode="RGB")


def build_sobel_edge_mask(image: Image.Image, threshold: float = 0.0) -> Image.Image:
    if np is not None:
        rgb = rgb_array(image, dtype=np.float64)
        luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        padded = np.pad(luma, 1, mode="edge")
        gx = (
            -padded[:-2, :-2]
            + padded[:-2, 2:]
            - 2.0 * padded[1:-1, :-2]
            + 2.0 * padded[1:-1, 2:]
            - padded[2:, :-2]
            + padded[2:, 2:]
        )
        gy = (
            -padded[:-2, :-2]
            - 2.0 * padded[:-2, 1:-1]
            - padded[:-2, 2:]
            + padded[2:, :-2]
            + 2.0 * padded[2:, 1:-1]
            + padded[2:, 2:]
        )
        magnitude = np.sqrt(gx * gx + gy * gy)
        max_magnitude = float(magnitude.max())
        if max_magnitude <= 0:
            return Image.new("L", image.size, 0)

        threshold_value = max_magnitude * threshold
        normalized = np.maximum(0.0, (magnitude - threshold_value) / max_magnitude)
        mask_data = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
        mask = Image.fromarray(mask_data, mode="L")
        return mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(radius=0.4))

    source = image.convert("RGB")
    width, height = source.size
    luma = [[0.0 for _x in range(width)] for _y in range(height)]
    pixels = source.load()

    for y in range(height):
        for x in range(width):
            luma[y][x] = srgb_luma(pixels[x, y])

    raw = [[0.0 for _x in range(width)] for _y in range(height)]
    max_magnitude = 0.0
    for y in range(height):
        ym = max(0, y - 1)
        yp = min(height - 1, y + 1)
        for x in range(width):
            xm = max(0, x - 1)
            xp = min(width - 1, x + 1)
            gx = (
                -luma[ym][xm]
                + luma[ym][xp]
                - 2 * luma[y][xm]
                + 2 * luma[y][xp]
                - luma[yp][xm]
                + luma[yp][xp]
            )
            gy = (
                -luma[ym][xm]
                - 2 * luma[ym][x]
                - luma[ym][xp]
                + luma[yp][xm]
                + 2 * luma[yp][x]
                + luma[yp][xp]
            )
            magnitude = math.sqrt(gx * gx + gy * gy)
            raw[y][x] = magnitude
            max_magnitude = max(max_magnitude, magnitude)

    if max_magnitude <= 0:
        return Image.new("L", image.size, 0)

    mask = Image.new("L", image.size)
    out = mask.load()
    threshold_value = max_magnitude * threshold
    for y in range(height):
        for x in range(width):
            normalized = max(0.0, (raw[y][x] - threshold_value) / max_magnitude)
            out[x, y] = clamp_channel(normalized * 255)

    return mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(radius=0.4))


def bilateral_smooth(
    image: Image.Image,
    radius: int,
    sigma_color: float,
    sigma_space: float,
    mode: str = "standard",
    edge_mask: Image.Image | None = None,
) -> Image.Image:
    if radius <= 0:
        return image

    source = image.convert("RGB")
    mode = mode if mode in {"standard", "edge-safe"} else "standard"
    if _bilateral_smooth_numba is not None:
        sigma_color2 = max(1e-6, 2 * sigma_color * sigma_color)
        sigma_space2 = max(1e-6, 2 * sigma_space * sigma_space)
        spatial = np.empty((radius * 2 + 1, radius * 2 + 1), dtype=np.float64)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                spatial[dy + radius, dx + radius] = math.exp(-(dx * dx + dy * dy) / sigma_space2)
        source_array = np.asarray(source, dtype=np.uint8)
        if mode == "edge-safe" and _edge_safe_bilateral_smooth_numba is not None:
            if edge_mask is None:
                edge_array = np.zeros((source.height, source.width), dtype=np.uint8)
            else:
                edge_array = np.asarray(
                    edge_mask.convert("L").resize(source.size, Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
            luma_gate = max(22.0, min(96.0, sigma_color * 2.4))
            boundary_distance = max(34.0, min(132.0, sigma_color * 3.4))
            output = _edge_safe_bilateral_smooth_numba(
                source_array,
                edge_array,
                radius,
                sigma_color2,
                spatial,
                EDGE_SAFE_BILATERAL_EDGE_THRESHOLD,
                luma_gate,
                EDGE_SAFE_BILATERAL_DARK_LUMA,
                EDGE_SAFE_BILATERAL_DARK_CONTRAST,
                boundary_distance * boundary_distance,
            )
        else:
            output = _bilateral_smooth_numba(
                source_array,
                radius,
                sigma_color2,
                spatial,
            )
        return Image.fromarray(output, mode="RGB")

    width, height = source.size
    src = source.load()
    out = Image.new("RGB", source.size)
    dst = out.load()
    edge_pixels = (
        edge_mask.convert("L").resize(source.size, Image.Resampling.NEAREST).load()
        if edge_mask is not None
        else None
    )
    sigma_color2 = max(1e-6, 2 * sigma_color * sigma_color)
    sigma_space2 = max(1e-6, 2 * sigma_space * sigma_space)
    spatial: dict[tuple[int, int], float] = {}

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial[(dx, dy)] = math.exp(-(dx * dx + dy * dy) / sigma_space2)

    for y in range(height):
        for x in range(width):
            center = src[x, y]
            if mode == "edge-safe":
                center_luma = srgb_luma(center)
                local_values = []
                for local_dy in (-1, 0, 1):
                    yy = min(height - 1, max(0, y + local_dy))
                    for local_dx in (-1, 0, 1):
                        xx = min(width - 1, max(0, x + local_dx))
                        local_values.append(srgb_luma(src[xx, yy]))
                local_range = max(local_values) - min(local_values)
                center_edge = 0.0
                if edge_pixels is not None:
                    center_edge = edge_pixels[x, y] / 255.0
                if center_luma <= EDGE_SAFE_BILATERAL_DARK_LUMA and local_range >= EDGE_SAFE_BILATERAL_DARK_CONTRAST:
                    dst[x, y] = center
                    continue

            weighted_red = weighted_green = weighted_blue = total_weight = 0.0
            for dy in range(-radius, radius + 1):
                yy = min(height - 1, max(0, y + dy))
                for dx in range(-radius, radius + 1):
                    xx = min(width - 1, max(0, x + dx))
                    color = src[xx, yy]
                    if mode == "edge-safe":
                        neighbor_luma = srgb_luma(color)
                        if abs(neighbor_luma - center_luma) > max(22.0, sigma_color * 2.4):
                            continue
                        dark_crossing = (
                            (center_luma <= EDGE_SAFE_BILATERAL_DARK_LUMA)
                            != (neighbor_luma <= EDGE_SAFE_BILATERAL_DARK_LUMA)
                        )
                        if dark_crossing and abs(neighbor_luma - center_luma) >= EDGE_SAFE_BILATERAL_DARK_CONTRAST:
                            continue
                        if color_distance_squared(center, color) > max(34.0, sigma_color * 3.4) ** 2:
                            continue
                    dr = color[0] - center[0]
                    dg = color[1] - center[1]
                    db = color[2] - center[2]
                    range_weight = math.exp(-(dr * dr + dg * dg + db * db) / sigma_color2)
                    weight = spatial[(dx, dy)] * range_weight
                    weighted_red += color[0] * weight
                    weighted_green += color[1] * weight
                    weighted_blue += color[2] * weight
                    total_weight += weight
            if total_weight <= 1e-6:
                dst[x, y] = center
            elif mode == "edge-safe":
                blend = min(
                    EDGE_SAFE_BILATERAL_MAX_BLEND,
                    (
                        1.0
                        - 0.55 * smoothstep(EDGE_SAFE_BILATERAL_EDGE_THRESHOLD, 0.9, center_edge)
                    )
                    * (
                        1.0
                        - 0.45
                        * smoothstep(max(19.8, sigma_color * 2.16), max(52.8, sigma_color * 5.76), local_range)
                    ),
                )
                dst[x, y] = (
                    clamp_channel(center[0] + (weighted_red / total_weight - center[0]) * blend),
                    clamp_channel(center[1] + (weighted_green / total_weight - center[1]) * blend),
                    clamp_channel(center[2] + (weighted_blue / total_weight - center[2]) * blend),
                )
            else:
                dst[x, y] = (
                    clamp_channel(weighted_red / total_weight),
                    clamp_channel(weighted_green / total_weight),
                    clamp_channel(weighted_blue / total_weight),
                )

    return out


def selective_edge_sharpen(image: Image.Image, edge_mask: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return image

    sharp = image.filter(ImageFilter.UnsharpMask(radius=0.7, percent=180, threshold=2)).convert("RGB")
    base = image.convert("RGB")
    if np is not None:
        base_array = np.asarray(base, dtype=np.float64)
        sharp_array = np.asarray(sharp, dtype=np.float64)
        blend = np.minimum(
            1.0,
            (np.asarray(edge_mask.convert("L"), dtype=np.float64) / 255.0) * strength,
        )
        output = base_array + (sharp_array - base_array) * blend[:, :, None]
        return Image.fromarray(np.clip(np.rint(output), 0, 255).astype(np.uint8), mode="RGB")

    mask = edge_mask.convert("L").load()
    source = base.load()
    sharpened = sharp.load()
    out = Image.new("RGB", image.size)
    target = out.load()

    for y in range(image.height):
        for x in range(image.width):
            blend = min(1.0, (mask[x, y] / 255.0) * strength)
            original = source[x, y]
            edged = sharpened[x, y]
            target[x, y] = (
                clamp_channel(original[0] + (edged[0] - original[0]) * blend),
                clamp_channel(original[1] + (edged[1] - original[1]) * blend),
                clamp_channel(original[2] + (edged[2] - original[2]) * blend),
            )

    return out


def grade_image(image: Image.Image, config: PixelArtConfig) -> Image.Image:
    if config.autocontrast_cutoff > 0:
        image = ImageOps.autocontrast(image, cutoff=config.autocontrast_cutoff)
    if not math.isclose(config.saturation, 1.0):
        image = ImageEnhance.Color(image).enhance(config.saturation)
    if not math.isclose(config.contrast, 1.0):
        image = ImageEnhance.Contrast(image).enhance(config.contrast)
    if config.sharpness > 0:
        image = image.filter(
            ImageFilter.UnsharpMask(radius=0.6, percent=int(config.sharpness), threshold=3)
        )

    return image


def luma_mean(image: Image.Image) -> float:
    if np is not None:
        rgb = rgb_array(image, dtype=np.float64)
        luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        return float(luma.mean())

    total = 0.0
    for red, green, blue in image.convert("RGB").getdata():
        total += srgb_luma((red, green, blue))
    return total / (image.width * image.height)


def luma_weighted_saturation_mean(image: Image.Image) -> float:
    if np is not None:
        rgb = rgb_array(image, dtype=np.float64)
        luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        saturation, _value = saturation_value_arrays(rgb)
        luma_total = float(luma.sum())
        if luma_total <= 0:
            return 0.0
        return float((saturation * luma).sum() / luma_total)

    weighted_saturation = 0.0
    luma_total = 0.0
    for red, green, blue in image.convert("RGB").getdata():
        color = (red, green, blue)
        luma = srgb_luma(color)
        weighted_saturation += rgb_saturation(color) * luma
        luma_total += luma
    if luma_total <= 0:
        return 0.0
    return weighted_saturation / luma_total


def match_luma_mean(image: Image.Image, target_mean: float) -> Image.Image:
    current_mean = luma_mean(image)
    if current_mean <= 0 or math.isclose(current_mean, target_mean, abs_tol=0.05):
        return image

    scale = target_mean / current_mean
    if np is not None:
        rgb = rgb_array(image, dtype=np.float64)
        output = np.clip(np.rint(rgb * scale), 0, 255).astype(np.uint8)
        return Image.fromarray(output, mode="RGB")

    out = Image.new("RGB", image.size)
    source = image.load()
    target = out.load()
    cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    for y in range(image.height):
        for x in range(image.width):
            color = source[x, y]
            adjusted = cache.get(color)
            if adjusted is None:
                adjusted = (
                    clamp_channel(color[0] * scale),
                    clamp_channel(color[1] * scale),
                    clamp_channel(color[2] * scale),
                )
                cache[color] = adjusted
            target[x, y] = adjusted

    return out


def adjust_color_saturation(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    luma = srgb_luma(color)
    return (
        clamp_channel(luma + (color[0] - luma) * factor),
        clamp_channel(luma + (color[1] - luma) * factor),
        clamp_channel(luma + (color[2] - luma) * factor),
    )


def apply_saturation_factor(image: Image.Image, factor: float) -> Image.Image:
    if np is not None:
        rgb = rgb_array(image, dtype=np.float64)
        luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        output = luma[:, :, None] + (rgb - luma[:, :, None]) * factor
        return Image.fromarray(np.clip(np.rint(output), 0, 255).astype(np.uint8), mode="RGB")

    out = Image.new("RGB", image.size)
    source = image.load()
    target = out.load()
    cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    for y in range(image.height):
        for x in range(image.width):
            color = source[x, y]
            adjusted = cache.get(color)
            if adjusted is None:
                adjusted = adjust_color_saturation(color, factor)
                cache[color] = adjusted
            target[x, y] = adjusted

    return out


def match_luma_weighted_saturation(image: Image.Image, target_mean: float) -> Image.Image:
    current_mean = luma_weighted_saturation_mean(image)
    if current_mean <= 0 or math.isclose(current_mean, target_mean, abs_tol=0.002):
        return image

    if current_mean < target_mean:
        low = 1.0
        high = 1.25
        best = image
        for _step in range(8):
            candidate = apply_saturation_factor(image, high)
            if luma_weighted_saturation_mean(candidate) >= target_mean or high >= 8:
                break
            low = high
            high *= 1.5
    else:
        low = 0.0
        high = 1.0
        best = image

    for _step in range(14):
        factor = (low + high) / 2
        candidate = apply_saturation_factor(image, factor)
        candidate_mean = luma_weighted_saturation_mean(candidate)
        best = candidate
        if math.isclose(candidate_mean, target_mean, abs_tol=0.002):
            return candidate
        if candidate_mean < target_mean:
            low = factor
        else:
            high = factor

    return best


def extract_palette(image: Image.Image, colors: int) -> list[tuple[int, int, int]]:
    quantized = image.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette = quantized.getpalette() or []
    used = sorted(quantized.getcolors(image.width * image.height) or [], reverse=True)

    result: list[tuple[int, int, int]] = []
    for _count, index in used:
        offset = index * 3
        if offset + 2 >= len(palette):
            continue
        color = (palette[offset], palette[offset + 1], palette[offset + 2])
        if color not in result:
            result.append(color)

    return sorted(result, key=srgb_luma)


def split_color_box(box: ColorBox) -> tuple[ColorBox, ColorBox] | None:
    if len(box.colors) < 2:
        return None

    channel = max(range(3), key=box.range_for_channel)
    colors = sorted(box.colors, key=lambda item: item[0][channel])
    half_weight = box.weight / 2
    running_weight = 0
    split_at = 1

    for index, (_color, count) in enumerate(colors):
        running_weight += count
        if running_weight >= half_weight:
            split_at = max(1, min(len(colors) - 1, index + 1))
            break

    return ColorBox.from_colors(colors[:split_at]), ColorBox.from_colors(colors[split_at:])


def weighted_average_color(colors: list[tuple[tuple[int, int, int], float]]) -> tuple[int, int, int]:
    total = sum(count for _color, count in colors)
    if total <= 0:
        return (0, 0, 0)

    red = sum(color[0] * count for color, count in colors) / total
    green = sum(color[1] * count for color, count in colors) / total
    blue = sum(color[2] * count for color, count in colors) / total
    return (round(red), round(green), round(blue))


def collect_weighted_colors(
    image: Image.Image,
    edge_mask: Image.Image | None,
    edge_palette_weight: float,
    accent_palette_weight: float = 0.0,
    hue_rarity_weight: float = 0.0,
    protected_hue_ranges: tuple[tuple[float, float], ...] = (),
    protected_hue_weight: float = 0.0,
    protected_hue_min_saturation: float = 0.08,
) -> list[tuple[tuple[int, int, int], float]]:
    source = image.convert("RGB")
    use_color_weights = (
        accent_palette_weight > 0
        or hue_rarity_weight > 0
        or (protected_hue_ranges and protected_hue_weight > 0)
    )
    if (edge_mask is None or edge_palette_weight <= 0) and not use_color_weights:
        color_counts = source.getcolors(maxcolors=source.width * source.height)
        if not color_counts:
            raise RuntimeError("could not collect image colors")
        return [(color, float(count)) for count, color in color_counts]

    if np is not None:
        rgb = rgb_array(source, dtype=np.uint8)
        rgb_float = rgb.astype(np.float64)
        saturation, value = saturation_value_arrays(rgb_float)
        weights = np.ones((source.height, source.width), dtype=np.float64)

        if edge_mask is not None and edge_palette_weight > 0:
            edge = np.asarray(edge_mask.convert("L"), dtype=np.float64) / 255.0
            weights += edge_palette_weight * edge

        if accent_palette_weight > 0:
            ramp = np.clip((saturation - 0.08) / 0.20, 0.0, 1.0)
            ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            weights += accent_palette_weight * saturation * ramp * value

        hue = None
        if hue_rarity_weight > 0:
            hue = hue_array(rgb_float)
            hue_bins_index = np.minimum(35, np.floor(hue / 10.0).astype(np.int64))
            saturated = saturation >= 0.08
            hue_bins = np.bincount(
                hue_bins_index[saturated].ravel(),
                weights=saturation[saturated].ravel(),
                minlength=36,
            ).astype(np.float64)
            hue_total = float(hue_bins.sum())
            if hue_total > 0:
                hue_bins /= hue_total
            rarity = 1.0 - np.minimum(1.0, hue_bins[hue_bins_index] * 36.0)
            weights += np.where(
                saturated,
                hue_rarity_weight * rarity * saturation * value,
                0.0,
            )

        if protected_hue_ranges and protected_hue_weight > 0:
            if hue is None:
                hue = hue_array(rgb_float)
            protected = hue_ranges_mask_array(hue, protected_hue_ranges)
            protected &= saturation >= protected_hue_min_saturation
            weights += np.where(
                protected,
                protected_hue_weight * saturation * value,
                0.0,
            )

        packed = pack_rgb_array(rgb)
        unique, first_index, inverse = np.unique(
            packed.reshape(-1),
            return_index=True,
            return_inverse=True,
        )
        weight_sums = np.bincount(inverse, weights=weights.reshape(-1))
        scan_order = np.argsort(first_index)
        return [
            (unpack_rgb_value(int(color)), float(weight))
            for color, weight in zip(unique[scan_order], weight_sums[scan_order], strict=True)
        ]

    hue_bins = [0.0 for _bin in range(36)]
    if hue_rarity_weight > 0:
        for red, green, blue in source.getdata():
            color = (red, green, blue)
            saturation = rgb_saturation(color)
            if saturation < 0.08:
                continue
            hue_bins[min(35, int(rgb_hue(color) // 10))] += saturation
        total_hue = sum(hue_bins)
        hue_bins = [count / total_hue if total_hue > 0 else 0.0 for count in hue_bins]

    weights: dict[tuple[int, int, int], float] = {}
    mask = edge_mask.convert("L").load() if edge_mask is not None else None
    pixels = source.load()
    for y in range(source.height):
        for x in range(source.width):
            color = pixels[x, y]
            saturation = rgb_saturation(color)
            value = max(color) / 255.0
            edge = mask[x, y] / 255.0 if mask is not None else 0.0
            weight = 1.0 + edge_palette_weight * edge
            weight += accent_palette_weight * saturation * smoothstep(0.08, 0.28, saturation) * value
            if hue_rarity_weight > 0 and saturation >= 0.08:
                bin_index = min(35, int(rgb_hue(color) // 10))
                rarity = 1.0 - min(1.0, hue_bins[bin_index] * 36.0)
                weight += hue_rarity_weight * rarity * saturation * value
            if (
                protected_hue_ranges
                and protected_hue_weight > 0
                and saturation >= protected_hue_min_saturation
            ):
                if hue_in_ranges(rgb_hue(color), protected_hue_ranges):
                    weight += protected_hue_weight * saturation * value
            weights[color] = weights.get(color, 0.0) + weight

    return list(weights.items())


def quantize_median_cut_rgb(
    image: Image.Image,
    colors: int,
    edge_mask: Image.Image | None = None,
    palette_image: Image.Image | None = None,
    palette_edge_mask: Image.Image | None = None,
    edge_palette_weight: float = 0.0,
    palette_strategy: str = "median-cut",
    color_distance: str = "rgb",
    accent_palette_weight: float = 0.0,
    hue_rarity_weight: float = 0.0,
    interesting_color_slots: int = 0,
    interesting_min_saturation: float = 0.12,
    interesting_min_value: float = 0.06,
    protected_hue_ranges: tuple[tuple[float, float], ...] = (),
    protected_hue_weight: float = 0.0,
    protected_hue_slots: int = 0,
    protected_hue_min_saturation: float = 0.08,
    hue_match_weight: float = 0.0,
) -> tuple[Image.Image, list[tuple[int, int, int]]]:
    palette_input = palette_image if palette_image is not None else image
    palette_mask = palette_edge_mask if palette_image is not None else edge_mask
    weighted_colors = collect_weighted_colors(
        palette_input,
        palette_mask,
        edge_palette_weight,
        accent_palette_weight=accent_palette_weight,
        hue_rarity_weight=hue_rarity_weight,
        protected_hue_ranges=protected_hue_ranges,
        protected_hue_weight=protected_hue_weight,
        protected_hue_min_saturation=protected_hue_min_saturation,
    )

    reserved_palette: list[tuple[int, int, int]] = []
    if protected_hue_ranges and protected_hue_slots > 0:
        protected_colors = [
            (color, weight)
            for color, weight in weighted_colors
            if (
                rgb_saturation(color) >= protected_hue_min_saturation
                and hue_in_ranges(rgb_hue(color), protected_hue_ranges)
            )
        ]
        if protected_colors:
            reserved_palette = merge_palette_slots(
                reserved_palette,
                median_cut_palette(protected_colors, min(protected_hue_slots, colors - 1)),
                colors,
            )

    if palette_strategy == "hue-mass":
        palette = hue_mass_palette(
            weighted_colors,
            colors,
            min_saturation=interesting_min_saturation,
            min_value=interesting_min_value,
        )
        return (
            map_to_palette(
                image,
                palette,
                hue_match_weight=hue_match_weight,
                color_distance=color_distance,
            ),
            palette,
        )

    if palette_strategy == "spectrum-peaks":
        palette = spectrum_peak_palette(
            weighted_colors,
            colors,
            min_saturation=interesting_min_saturation,
            min_value=interesting_min_value,
        )
        return (
            map_to_palette(
                image,
                palette,
                hue_match_weight=hue_match_weight,
                color_distance=color_distance,
            ),
            palette,
        )

    if palette_strategy == "shadow-spectrum":
        palette = shadow_spectrum_palette(
            weighted_colors,
            colors,
            min_saturation=interesting_min_saturation,
            min_value=interesting_min_value,
        )
        return (
            map_to_palette(
                image,
                palette,
                hue_match_weight=hue_match_weight,
                color_distance=color_distance,
            ),
            palette,
        )

    if palette_strategy == "projected-mass":
        palette = projected_source_palette(
            image,
            weighted_colors,
            colors,
            rare_boost=False,
            min_saturation=interesting_min_saturation,
            min_value=interesting_min_value,
            color_distance=color_distance,
        )
        return (
            map_to_palette(
                image,
                palette,
                hue_match_weight=hue_match_weight,
                color_distance=color_distance,
            ),
            palette,
        )

    if palette_strategy == "projected-rare":
        palette = projected_source_palette(
            image,
            weighted_colors,
            colors,
            rare_boost=True,
            min_saturation=interesting_min_saturation,
            min_value=interesting_min_value,
            color_distance=color_distance,
        )
        return (
            map_to_palette(
                image,
                palette,
                hue_match_weight=hue_match_weight,
                color_distance=color_distance,
            ),
            palette,
        )

    if palette_strategy == "projected-edge":
        palette = projected_edge_palette(
            image,
            weighted_colors,
            colors,
            min_saturation=interesting_min_saturation,
            min_value=interesting_min_value,
            color_distance=color_distance,
        )
        return (
            map_to_palette_with_rare_guard(
                image,
                palette,
                hue_match_weight=hue_match_weight,
                color_distance=color_distance,
                min_value=interesting_min_value,
            ),
            palette,
        )

    if palette_strategy == "projected-islands":
        palette = projected_island_palette(
            image,
            weighted_colors,
            colors,
            min_saturation=interesting_min_saturation,
            min_value=interesting_min_value,
            color_distance=color_distance,
        )
        return (
            map_to_palette_with_rare_guard(
                image,
                palette,
                hue_match_weight=hue_match_weight,
                color_distance=color_distance,
                min_value=interesting_min_value,
            ),
            palette,
        )

    if palette_strategy == "projected-anchors":
        palette = projected_anchor_palette(
            image,
            weighted_colors,
            colors,
            min_saturation=interesting_min_saturation,
            min_value=interesting_min_value,
            color_distance=color_distance,
        )
        return (
            map_to_palette_with_rare_guard(
                image,
                palette,
                hue_match_weight=hue_match_weight,
                color_distance=color_distance,
                min_value=interesting_min_value,
            ),
            palette,
        )

    if palette_strategy == "projected-frontier":
        palette = projected_frontier_palette(
            image,
            weighted_colors,
            colors,
            min_saturation=interesting_min_saturation,
            min_value=interesting_min_value,
            color_distance=color_distance,
        )
        return (
            map_to_palette_with_rare_guard(
                image,
                palette,
                hue_match_weight=hue_match_weight,
                color_distance=color_distance,
                min_value=interesting_min_value,
            ),
            palette,
        )

    if palette_strategy == "projected-graft":
        palette = projected_graft_palette(
            image,
            weighted_colors,
            colors,
            min_saturation=interesting_min_saturation,
            min_value=interesting_min_value,
            color_distance=color_distance,
        )
        return (
            map_to_palette(
                image,
                palette,
                hue_match_weight=hue_match_weight,
                color_distance=color_distance,
            ),
            palette,
        )

    if palette_strategy == "interesting" and interesting_color_slots > 0:
        remaining_reserved_slots = max(0, colors - len(reserved_palette) - 1)
        reserved_palette = merge_palette_slots(
            reserved_palette,
            select_interesting_palette(
                weighted_colors,
                min(interesting_color_slots, remaining_reserved_slots),
                min_saturation=interesting_min_saturation,
                min_value=interesting_min_value,
            ),
            colors,
        )

    remaining_colors = max(1, colors - len(reserved_palette))
    palette = merge_palette_slots(
        reserved_palette,
        median_cut_palette(weighted_colors, remaining_colors),
        colors,
    )
    return (
        map_to_palette(
            image,
            palette,
            hue_match_weight=hue_match_weight,
            color_distance=color_distance,
        ),
        palette,
    )


def merge_palette_slots(
    reserved_palette: list[tuple[int, int, int]],
    base_palette: list[tuple[int, int, int]],
    colors: int,
) -> list[tuple[int, int, int]]:
    palette: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for color in reserved_palette + base_palette:
        if color in seen:
            continue
        palette.append(color)
        seen.add(color)
        if len(palette) >= colors:
            break
    return palette


def select_interesting_palette(
    weighted_colors: list[tuple[tuple[int, int, int], float]],
    slots: int,
    min_saturation: float,
    min_value: float,
) -> list[tuple[int, int, int]]:
    if slots <= 0:
        return []

    hue_totals = [0.0 for _bin in range(48)]
    total_weight = 0.0
    for color, weight in weighted_colors:
        saturation = rgb_saturation(color)
        value = max(color) / 255.0
        if saturation < min_saturation or value < min_value:
            continue
        hue_totals[min(47, int(rgb_hue(color) // 7.5))] += weight
        total_weight += weight

    if total_weight <= 0:
        return []

    groups: dict[tuple[int, int, int], tuple[tuple[int, int, int], float]] = {}
    for color, weight in weighted_colors:
        saturation = rgb_saturation(color)
        value = max(color) / 255.0
        if saturation < min_saturation or value < min_value:
            continue

        hue = rgb_hue(color)
        hue_bin = min(47, int(hue // 7.5))
        value_bin = min(5, int(value * 6))
        saturation_bin = min(4, int(saturation * 5))
        hue_fraction = hue_totals[hue_bin] / total_weight
        hue_rarity = min(3.0, 1.0 / math.sqrt(max(hue_fraction * 48.0, 0.05)))
        score = (
            (saturation ** 1.65)
            * ((0.25 + value) ** 1.15)
            * (0.5 + hue_rarity)
            * math.log1p(weight)
        )
        group = (hue_bin, value_bin, saturation_bin)
        if group not in groups or score > groups[group][1]:
            groups[group] = (color, score)

    ranked = sorted(groups.values(), key=lambda item: (-item[1], srgb_luma(item[0])))
    return [color for color, _score in ranked[:slots]]


def allocate_slots_by_score(
    scores: list[tuple[object, float]],
    slots: int,
    minimum_score_fraction: float = 0.0,
) -> dict[object, int]:
    active = [(key, max(0.0, score)) for key, score in scores if score > 0]
    if not active or slots <= 0:
        return {}

    total_score = sum(score for _key, score in active)
    if total_score <= 0:
        return {}

    if minimum_score_fraction > 0:
        active = [
            (key, score)
            for key, score in active
            if score / total_score >= minimum_score_fraction
        ] or active
        total_score = sum(score for _key, score in active)

    if len(active) >= slots:
        return {key: 1 for key, _score in sorted(active, key=lambda item: item[1], reverse=True)[:slots]}

    allocation = {key: 1 for key, _score in active}
    remaining = slots - len(active)
    weights = [(key, math.sqrt(score)) for key, score in active]
    weight_total = sum(weight for _key, weight in weights)
    fractions: list[tuple[float, object]] = []
    for key, weight in weights:
        exact = remaining * weight / weight_total if weight_total > 0 else 0.0
        whole = math.floor(exact)
        allocation[key] += whole
        fractions.append((exact - whole, key))

    used = sum(allocation.values())
    for _fraction, key in sorted(fractions, reverse=True):
        if used >= slots:
            break
        allocation[key] += 1
        used += 1

    return allocation


def representative_peak_color(
    weighted_colors: list[tuple[tuple[int, int, int], float]],
    chroma_bias: float = 0.0,
    shadow_bias: float = 0.0,
) -> tuple[int, int, int]:
    return max(
        weighted_colors,
        key=lambda item: (
            math.log1p(item[1])
            * ((0.25 + rgb_saturation(item[0])) ** chroma_bias)
            * ((0.25 + (1.0 - max(item[0]) / 255.0)) ** shadow_bias)
            * (0.45 + max(item[0]) / 255.0)
        ),
    )[0]


def representative_tonal_rare_color(
    weighted_colors: list[tuple[tuple[int, int, int], float]],
) -> tuple[int, int, int]:
    return max(
        weighted_colors,
        key=lambda item: (
            math.log1p(item[1])
            * ((0.42 + srgb_luma(item[0]) / 255.0) ** 1.1)
            * (0.80 + rgb_saturation(item[0]))
        ),
    )[0]


def neutral_rare_palette(
    weighted_colors: list[tuple[tuple[int, int, int], float]],
    slots: int,
    min_value: float,
    max_saturation: float = 0.42,
) -> list[tuple[int, int, int]]:
    if slots <= 0 or not weighted_colors:
        return []

    groups: dict[tuple[int, int, int], list[tuple[tuple[int, int, int], float]]] = {}
    cell_weights: dict[tuple[int, int, int], float] = {}
    total_weight = 0.0
    for color, weight in weighted_colors:
        value = max(color) / 255.0
        saturation = rgb_saturation(color)
        if value < min_value or saturation > max_saturation:
            continue
        warm_axis = (color[0] - color[2]) / 255.0
        green_axis = (color[1] - ((color[0] + color[2]) * 0.5)) / 255.0
        if abs(warm_axis) < 0.025 and abs(green_axis) < 0.025:
            tint_bin = 0
        elif abs(warm_axis) >= abs(green_axis):
            tint_bin = 1 if warm_axis > 0 else 2
        else:
            tint_bin = 3 if green_axis > 0 else 4
        key = (
            min(11, int(srgb_luma(color) / 256.0 * 12)),
            min(5, int(saturation / max(max_saturation, 1e-6) * 6)),
            tint_bin,
        )
        groups.setdefault(key, []).append((color, weight))
        cell_weights[key] = cell_weights.get(key, 0.0) + weight
        total_weight += weight

    if not groups or total_weight <= 0:
        return []

    cell_count = max(1, len(groups))
    scored_cells: list[tuple[tuple[int, int, int], float]] = []
    for key, group in groups.items():
        representative = representative_tonal_rare_color(group)
        cell_weight = cell_weights[key]
        rarity = min(5.0, 1.0 / math.sqrt(max(cell_weight / total_weight * cell_count, 0.015)))
        value = max(representative) / 255.0
        score = (0.45 * math.log1p(cell_weight) + 1.35 * rarity) * (0.5 + value)
        scored_cells.append((key, score))

    palette: list[tuple[int, int, int]] = []
    for key, _score in sorted(scored_cells, key=lambda item: item[1], reverse=True):
        palette.append(representative_tonal_rare_color(groups[key]))
        if len(palette) >= slots:
            break
    return merge_palette_slots([], palette, slots)


def tonal_rare_palette(
    weighted_colors: list[tuple[tuple[int, int, int], float]],
    slots: int,
    min_value: float,
) -> list[tuple[int, int, int]]:
    if slots <= 0 or not weighted_colors:
        return []

    groups: dict[tuple[int, int, int], list[tuple[tuple[int, int, int], float]]] = {}
    cell_weights: dict[tuple[int, int, int], float] = {}
    total_weight = 0.0
    for color, weight in weighted_colors:
        value = max(color) / 255.0
        if value < min_value:
            continue
        saturation = rgb_saturation(color)
        luma_bin = min(11, int(srgb_luma(color) / 256.0 * 12))
        chroma_bin = min(7, int(saturation * 16.0))
        if saturation < 0.14:
            warm_axis = (color[0] - color[2]) / 255.0
            green_axis = (color[1] - ((color[0] + color[2]) * 0.5)) / 255.0
            if abs(warm_axis) < 0.025 and abs(green_axis) < 0.025:
                tint_bin = 0
            elif abs(warm_axis) >= abs(green_axis):
                tint_bin = 1 if warm_axis > 0 else 2
            else:
                tint_bin = 3 if green_axis > 0 else 4
        else:
            tint_bin = 5 + min(23, int(rgb_hue(color) / 15.0))
        key = (luma_bin, chroma_bin, tint_bin)
        groups.setdefault(key, []).append((color, weight))
        cell_weights[key] = cell_weights.get(key, 0.0) + weight
        total_weight += weight

    if not groups or total_weight <= 0:
        return []

    cell_count = max(1, len(groups))
    scored_cells: list[tuple[tuple[int, int, int], float]] = []
    for key, group in groups.items():
        cell_weight = cell_weights[key]
        representative = representative_tonal_rare_color(group)
        saturation = rgb_saturation(representative)
        value = max(representative) / 255.0
        rarity = min(5.0, 1.0 / math.sqrt(max(cell_weight / total_weight * cell_count, 0.015)))
        neutral_bonus = 1.25 if saturation < 0.18 else 1.0
        score = (
            (0.55 * math.log1p(cell_weight) + 1.20 * rarity)
            * (0.45 + value)
            * (0.75 + saturation)
            * neutral_bonus
        )
        scored_cells.append((key, score))

    palette: list[tuple[int, int, int]] = []
    for key, _score in sorted(scored_cells, key=lambda item: item[1], reverse=True):
        palette.append(representative_tonal_rare_color(groups[key]))
        if len(palette) >= slots:
            break
    return merge_palette_slots([], palette, slots)


def value_band_palette(
    weighted_colors: list[tuple[tuple[int, int, int], float]],
    slots: int,
    bands: int = 10,
    chroma_bias: float = 0.0,
    shadow_bias: float = 0.0,
) -> list[tuple[int, int, int]]:
    if slots <= 0 or not weighted_colors:
        return []

    groups: dict[int, list[tuple[tuple[int, int, int], float]]] = {}
    scores = [0.0 for _band in range(bands)]
    for color, weight in weighted_colors:
        value = max(color) / 255.0
        band = min(bands - 1, int(value * bands))
        groups.setdefault(band, []).append((color, weight))
        scores[band] += weight

    allocation = allocate_slots_by_score(list(enumerate(scores)), slots, minimum_score_fraction=0.0)
    palette: list[tuple[int, int, int]] = []
    for band, count in sorted(allocation.items()):
        group = groups.get(int(band), [])
        if not group:
            continue
        if count <= 1:
            palette.append(
                representative_peak_color(
                    group,
                    chroma_bias=chroma_bias,
                    shadow_bias=shadow_bias,
                )
            )
        else:
            palette.extend(
                band_peak_palette(
                    group,
                    count,
                    hue_bins=24,
                    value_bins=max(2, min(5, count)),
                    chroma_bias=chroma_bias,
                    shadow_bias=shadow_bias,
                )
            )

    if len(palette) < slots:
        palette = merge_palette_slots(
            palette,
            median_cut_palette(weighted_colors, slots - len(palette)),
            slots,
        )
    return palette[:slots]


def band_peak_palette(
    weighted_colors: list[tuple[tuple[int, int, int], float]],
    slots: int,
    hue_bins: int,
    value_bins: int,
    chroma_bias: float,
    shadow_bias: float,
) -> list[tuple[int, int, int]]:
    if slots <= 0 or not weighted_colors:
        return []

    groups: dict[tuple[int, int], list[tuple[tuple[int, int, int], float]]] = {}
    scores: dict[tuple[int, int], float] = {}
    for color, weight in weighted_colors:
        saturation = rgb_saturation(color)
        value = max(color) / 255.0
        hue = rgb_hue(color)
        key = (min(hue_bins - 1, int(hue / 360.0 * hue_bins)), min(value_bins - 1, int(value * value_bins)))
        groups.setdefault(key, []).append((color, weight))
        scores[key] = scores.get(key, 0.0) + weight * (0.25 + saturation) * (0.35 + value)

    allocation = allocate_slots_by_score(list(scores.items()), slots, minimum_score_fraction=0.0)
    palette: list[tuple[int, int, int]] = []
    for key, count in sorted(allocation.items(), key=lambda item: (item[0][1], item[0][0])):
        group = groups.get(key, [])
        if not group:
            continue
        if count <= 1:
            palette.append(
                representative_peak_color(
                    group,
                    chroma_bias=chroma_bias,
                    shadow_bias=shadow_bias,
                )
            )
        else:
            palette.extend(median_cut_palette(group, count))
    return palette[:slots]


def hue_peak_palette(
    weighted_colors: list[tuple[tuple[int, int, int], float]],
    slots: int,
    min_saturation: float,
    min_value: float,
    hue_bins: int = 56,
    value_bins: int = 8,
) -> list[tuple[int, int, int]]:
    if slots <= 0:
        return []

    hue_groups: dict[int, list[tuple[tuple[int, int, int], float]]] = {}
    cell_groups: dict[tuple[int, int], list[tuple[tuple[int, int, int], float]]] = {}
    cell_scores: dict[tuple[int, int], float] = {}
    for color, weight in weighted_colors:
        saturation = rgb_saturation(color)
        value = max(color) / 255.0
        if saturation < min_saturation or value < min_value:
            continue
        hue_bin = min(hue_bins - 1, int(rgb_hue(color) / 360.0 * hue_bins))
        value_bin = min(value_bins - 1, int(value * value_bins))
        hue_groups.setdefault(hue_bin, []).append((color, weight))
        cell_groups.setdefault((hue_bin, value_bin), []).append((color, weight))
        cell_scores[(hue_bin, value_bin)] = cell_scores.get((hue_bin, value_bin), 0.0) + (
            weight * (saturation ** 1.7) * ((0.08 + value) ** 1.35)
        )

    if not hue_groups:
        return []

    hue_scores: list[tuple[int, float]] = []
    for hue_bin, group in hue_groups.items():
        mass_score = sum(
            weight * (rgb_saturation(color) ** 1.35) * (0.25 + max(color) / 255.0)
            for color, weight in group
        )
        peak_score = max(score for (candidate_hue, _value_bin), score in cell_scores.items() if candidate_hue == hue_bin)
        # Log compression keeps huge blue-gray/yellow masses from burying tiny but
        # visually important hue islands such as purple flowers and ember reds.
        hue_scores.append((hue_bin, 0.8 * math.log1p(mass_score) + 1.7 * math.log1p(peak_score)))

    selected_hues = [
        hue_bin
        for hue_bin, _score in sorted(hue_scores, key=lambda item: item[1], reverse=True)[:slots]
    ]

    palette: list[tuple[int, int, int]] = []
    for hue_bin in selected_hues:
        hue_cells = [
            key
            for key in cell_groups
            if key[0] == hue_bin
        ]
        best_cell = max(
            hue_cells,
            key=lambda key: cell_scores[key] * (0.35 + key[1] / max(1, value_bins - 1)),
        )
        palette.append(
            representative_peak_color(
                cell_groups[best_cell],
                chroma_bias=2.2,
                shadow_bias=0.0,
            )
        )
    return palette[:slots]


def hue_mass_palette(
    weighted_colors: list[tuple[tuple[int, int, int], float]],
    colors: int,
    min_saturation: float,
    min_value: float,
) -> list[tuple[int, int, int]]:
    neutral: list[tuple[tuple[int, int, int], float]] = []
    chroma: list[tuple[tuple[int, int, int], float]] = []
    for color, weight in weighted_colors:
        saturation = rgb_saturation(color)
        value = max(color) / 255.0
        if saturation < min_saturation or value < min_value:
            neutral.append((color, weight))
        else:
            chroma.append((color, weight))

    neutral_weight = sum(weight for _color, weight in neutral)
    chroma_weight = sum(weight for _color, weight in chroma)
    total_weight = max(neutral_weight + chroma_weight, 1.0)
    neutral_slots = round(colors * (0.22 + 0.20 * math.sqrt(neutral_weight / total_weight)))
    neutral_slots = max(12, min(colors - 20, neutral_slots))
    chroma_slots = colors - neutral_slots

    palette = value_band_palette(neutral or weighted_colors, neutral_slots, bands=12, shadow_bias=1.0)

    hue_bins = 48
    groups: dict[int, list[tuple[tuple[int, int, int], float]]] = {}
    scores = [0.0 for _bin in range(hue_bins)]
    for color, weight in chroma:
        saturation = rgb_saturation(color)
        value = max(color) / 255.0
        hue_bin = min(hue_bins - 1, int(rgb_hue(color) / 360.0 * hue_bins))
        groups.setdefault(hue_bin, []).append((color, weight))
        scores[hue_bin] += weight * math.sqrt(max(saturation, 0.001)) * (0.35 + value)

    guaranteed_hue_slots = min(chroma_slots, max(34, round(colors * 0.60)))
    chroma_palette = hue_peak_palette(
        chroma,
        guaranteed_hue_slots,
        min_saturation=min_saturation,
        min_value=min_value,
    )
    remaining_chroma_slots = chroma_slots - len(chroma_palette)

    allocation = allocate_slots_by_score(
        list(enumerate(scores)),
        remaining_chroma_slots,
        minimum_score_fraction=0.0015,
    )
    for hue_bin, count in sorted(allocation.items()):
        group = groups.get(int(hue_bin), [])
        if not group:
            continue
        chroma_palette.extend(
            band_peak_palette(
                group,
                count,
                hue_bins=1,
                value_bins=max(2, min(7, count)),
                chroma_bias=1.2,
                shadow_bias=0.2,
            )
        )

    palette = merge_palette_slots(palette, chroma_palette, colors)
    if len(palette) < colors:
        palette = merge_palette_slots(
            palette,
            median_cut_palette(weighted_colors, colors - len(palette)),
            colors,
        )
    return sorted(palette[:colors], key=srgb_luma)


def spectrum_peak_palette(
    weighted_colors: list[tuple[tuple[int, int, int], float]],
    colors: int,
    min_saturation: float,
    min_value: float,
) -> list[tuple[int, int, int]]:
    hue_bins = 56
    value_bins = 9
    neutral = [
        (color, weight)
        for color, weight in weighted_colors
        if rgb_saturation(color) < min_saturation or max(color) / 255.0 < min_value
    ]
    chroma = [
        (color, weight)
        for color, weight in weighted_colors
        if rgb_saturation(color) >= min_saturation and max(color) / 255.0 >= min_value
    ]
    neutral_slots = max(14, min(22, round(colors * 0.28)))
    peak_slots = colors - neutral_slots
    palette = value_band_palette(neutral or weighted_colors, neutral_slots, bands=12, shadow_bias=1.1)
    hue_palette = hue_peak_palette(
        chroma,
        min(peak_slots, max(40, round(colors * 0.66))),
        min_saturation=min_saturation,
        min_value=min_value,
    )
    palette = merge_palette_slots(palette, hue_palette, colors)
    peak_slots = colors - len(palette)

    groups: dict[tuple[int, int], list[tuple[tuple[int, int, int], float]]] = {}
    scores: dict[tuple[int, int], float] = {}
    for color, weight in chroma:
        saturation = rgb_saturation(color)
        value = max(color) / 255.0
        key = (
            min(hue_bins - 1, int(rgb_hue(color) / 360.0 * hue_bins)),
            min(value_bins - 1, int(value * value_bins)),
        )
        groups.setdefault(key, []).append((color, weight))
        scores[key] = scores.get(key, 0.0) + weight * (saturation ** 1.25) * ((0.2 + value) ** 0.75)

    top_cells = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected: list[tuple[int, int]] = []
    for key, _score in top_cells:
        hue, value = key
        too_close = False
        for selected_hue, selected_value in selected:
            hue_distance = min(abs(hue - selected_hue), hue_bins - abs(hue - selected_hue))
            if hue_distance <= 1 and abs(value - selected_value) <= 1:
                too_close = True
                break
        if too_close:
            continue
        selected.append(key)
        if len(selected) >= peak_slots:
            break

    if len(selected) < peak_slots:
        for key, _score in top_cells:
            if key not in selected:
                selected.append(key)
            if len(selected) >= peak_slots:
                break

    peak_palette = [
        representative_peak_color(groups[key], chroma_bias=1.65, shadow_bias=0.1)
        for key in selected
        if groups.get(key)
    ]
    palette = merge_palette_slots(palette, peak_palette, colors)
    if len(palette) < colors:
        palette = merge_palette_slots(
            palette,
            median_cut_palette(weighted_colors, colors - len(palette)),
            colors,
        )
    return sorted(palette[:colors], key=srgb_luma)


def shadow_spectrum_palette(
    weighted_colors: list[tuple[tuple[int, int, int], float]],
    colors: int,
    min_saturation: float,
    min_value: float,
) -> list[tuple[int, int, int]]:
    value_layers = (
        (0.00, 0.20, 18, 0.4, 1.4),
        (0.20, 0.42, 18, 0.8, 0.7),
        (0.42, 0.66, 16, 1.2, 0.2),
        (0.66, 1.01, 12, 1.5, 0.0),
    )
    palette = hue_peak_palette(
        weighted_colors,
        max(32, round(colors * 0.50)),
        min_saturation=min_saturation,
        min_value=min_value,
    )
    remaining = colors - len(palette)
    for layer_index, (low, high, desired, chroma_bias, shadow_bias) in enumerate(value_layers):
        if remaining <= 0:
            break
        layer = [
            (color, weight)
            for color, weight in weighted_colors
            if low <= max(color) / 255.0 < high
        ]
        if not layer:
            continue
        slots = min(desired, remaining)
        if layer_index == 0:
            layer_palette = value_band_palette(
                layer,
                slots,
                bands=8,
                chroma_bias=chroma_bias,
                shadow_bias=shadow_bias,
            )
        else:
            saturated = [
                (color, weight)
                for color, weight in layer
                if rgb_saturation(color) >= min_saturation and max(color) / 255.0 >= min_value
            ]
            layer_palette = band_peak_palette(
                saturated or layer,
                slots,
                hue_bins=36,
                value_bins=4,
                chroma_bias=chroma_bias,
                shadow_bias=shadow_bias,
            )
        palette = merge_palette_slots(palette, layer_palette, colors)
        remaining = colors - len(palette)

    if len(palette) < colors:
        palette = merge_palette_slots(
            palette,
            median_cut_palette(weighted_colors, colors - len(palette)),
            colors,
        )
    return sorted(palette[:colors], key=srgb_luma)


def projected_candidate_bank(
    source_weighted_colors: list[tuple[tuple[int, int, int], float]],
    colors: int,
    min_saturation: float,
    min_value: float,
) -> list[tuple[int, int, int]]:
    candidate_limit = max(192, colors * 4)
    candidates = merge_palette_slots(
        neutral_rare_palette(
            source_weighted_colors,
            min(max(8, round(colors * 0.14)), candidate_limit // 8),
            min_value=min_value,
        ),
        merge_palette_slots(
            tonal_rare_palette(
                source_weighted_colors,
                min(max(12, round(colors * 0.22)), candidate_limit // 4),
                min_value=min_value,
            ),
            merge_palette_slots(
                hue_peak_palette(
                    source_weighted_colors,
                    min(96, candidate_limit // 2),
                    min_saturation=min_saturation,
                    min_value=min_value,
                ),
                shadow_spectrum_palette(
                    source_weighted_colors,
                    min(candidate_limit // 2, 128),
                    min_saturation=min_saturation,
                    min_value=min_value,
                ),
                candidate_limit,
            ),
            candidate_limit,
        ),
        candidate_limit,
    )
    return merge_palette_slots(
        candidates,
        median_cut_palette(source_weighted_colors, candidate_limit - len(candidates)),
        candidate_limit,
    )


def nearest_candidate_cache(
    color: tuple[int, int, int],
    candidates: list[tuple[int, int, int]],
    cache: dict[tuple[int, int, int], tuple[int, int, int]],
    hue_match_weight: float = 0.0,
    color_distance: str = "rgb",
) -> tuple[int, int, int]:
    mapped = cache.get(color)
    if mapped is None:
        mapped = nearest_palette_color(
            color,
            candidates,
            hue_match_weight=hue_match_weight,
            color_distance=color_distance,
        )
        cache[color] = mapped
    return mapped


def rare_seed_conflict(
    color: tuple[int, int, int],
    rare_seed: list[tuple[int, int, int]],
    min_value: float,
) -> bool:
    color_saturation = rgb_saturation(color)
    for seed in rare_seed:
        seed_saturation = rgb_saturation(seed)
        if seed_saturation < 0.16 or max(seed) / 255.0 < min_value:
            continue
        hue_distance = hue_distance_degrees(rgb_hue(color), rgb_hue(seed))
        if hue_distance <= 35:
            continue
        if oklab_distance_squared(color, seed) < 900 and color_saturation < seed_saturation + 0.22:
            return True
    return False


def merge_preserving_rare_seed(
    palette: list[tuple[int, int, int]],
    additions: list[tuple[int, int, int]],
    colors: int,
    rare_seed: list[tuple[int, int, int]],
    min_value: float,
) -> list[tuple[int, int, int]]:
    filtered = [
        color
        for color in additions
        if color in palette or not rare_seed_conflict(color, rare_seed, min_value)
    ]
    merged = merge_palette_slots(palette, filtered, colors)
    if len(merged) < colors:
        merged = merge_palette_slots(merged, additions, colors)
    return merged


def projected_source_palette(
    target_image: Image.Image,
    source_weighted_colors: list[tuple[tuple[int, int, int], float]],
    colors: int,
    rare_boost: bool,
    min_saturation: float,
    min_value: float,
    color_distance: str = "rgb",
) -> list[tuple[int, int, int]]:
    candidate_limit = max(192, colors * 4)
    candidates = merge_palette_slots(
        hue_peak_palette(
            source_weighted_colors,
            min(96, candidate_limit // 2),
            min_saturation=min_saturation,
            min_value=min_value,
        ),
        shadow_spectrum_palette(
            source_weighted_colors,
            min(candidate_limit // 2, 128),
            min_saturation=min_saturation,
            min_value=min_value,
        ),
        candidate_limit,
    )
    candidates = merge_palette_slots(
        candidates,
        median_cut_palette(source_weighted_colors, candidate_limit - len(candidates)),
        candidate_limit,
    )

    source_interest: dict[tuple[int, int, int], float] = {}
    for color, weight in source_weighted_colors:
        saturation = rgb_saturation(color)
        value = max(color) / 255.0
        source_interest[color] = max(
            source_interest.get(color, 0.0),
            math.log1p(weight) * (saturation ** 1.4) * (0.2 + value),
        )

    projected: dict[tuple[int, int, int], float] = {candidate: 0.0 for candidate in candidates}
    if np is not None and _project_palette_mass_numba is not None:
        rgb = rgb_array(target_image.convert("RGB"), dtype=np.uint8)
        packed_colors, counts = np.unique(pack_rgb_array(rgb).reshape(-1), return_counts=True)
        projected_masses = _project_palette_mass_numba(
            packed_colors.astype(np.uint32),
            counts.astype(np.float64),
            np.asarray(candidates, dtype=np.uint8),
            0.05 if rare_boost else 0.0,
            1 if color_distance == "oklab" else 0,
        )
        for candidate, mass in zip(candidates, projected_masses, strict=True):
            projected[candidate] = projected.get(candidate, 0.0) + float(mass)
    else:
        target_counts = count_colors(target_image)
        cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for color, count in target_counts.items():
            mapped = cache.get(color)
            if mapped is None:
                mapped = nearest_palette_color(
                    color,
                    candidates,
                    hue_match_weight=0.05 if rare_boost else 0.0,
                    color_distance=color_distance,
                )
                cache[color] = mapped
            projected[mapped] = projected.get(mapped, 0.0) + count

    projected_items: list[tuple[tuple[int, int, int], float]] = []
    for candidate, mass in projected.items():
        if mass <= 0:
            continue
        saturation = rgb_saturation(candidate)
        value = max(candidate) / 255.0
        interest = source_interest.get(candidate, saturation * value)
        if rare_boost:
            mass = (mass ** 0.82) * (1.0 + 0.75 * saturation + 0.35 * math.log1p(interest))
        projected_items.append((candidate, mass))

    if not projected_items:
        return median_cut_palette(source_weighted_colors, colors)

    if rare_boost:
        accent_slots = max(24, round(colors * 0.48))
        palette = merge_palette_slots(
            neutral_rare_palette(
                projected_items,
                max(4, round(colors * 0.10)),
                min_value=min_value,
            ),
            merge_palette_slots(
                tonal_rare_palette(
                    projected_items,
                    max(6, round(colors * 0.14)),
                    min_value=min_value,
                ),
                hue_peak_palette(
                    projected_items,
                    accent_slots,
                    min_saturation=min_saturation,
                    min_value=min_value,
                ),
                colors,
            ),
            colors,
        )
        palette = merge_palette_slots(
            palette,
            value_band_palette(
                projected_items,
                colors - len(palette),
                bands=12,
                chroma_bias=0.5,
                shadow_bias=0.8,
            ),
            colors,
        )
    else:
        palette = shadow_spectrum_palette(
            projected_items,
            colors,
            min_saturation=min_saturation,
            min_value=min_value,
        )

    if len(palette) < colors:
        palette = merge_palette_slots(
            palette,
            sorted(
                (color for color, _mass in projected_items),
                key=lambda color: projected[color],
                reverse=True,
            ),
            colors,
        )
    return sorted(palette[:colors], key=srgb_luma)


def projected_edge_palette(
    target_image: Image.Image,
    source_weighted_colors: list[tuple[tuple[int, int, int], float]],
    colors: int,
    min_saturation: float,
    min_value: float,
    color_distance: str = "rgb",
) -> list[tuple[int, int, int]]:
    candidates = projected_candidate_bank(
        source_weighted_colors,
        colors,
        min_saturation=min_saturation,
        min_value=min_value,
    )
    projected: dict[tuple[int, int, int], float] = {candidate: 0.0 for candidate in candidates}
    contour_votes: dict[tuple[int, int, int], float] = {candidate: 0.0 for candidate in candidates}
    edge_mask = build_sobel_edge_mask(target_image, threshold=0.025)
    target = target_image.convert("RGB")

    if np is not None and _nearest_palette_indices_numba is not None:
        rgb = rgb_array(target, dtype=np.uint8)
        packed = pack_rgb_array(rgb)
        unique, inverse, counts = np.unique(
            packed.reshape(-1),
            return_inverse=True,
            return_counts=True,
        )
        edge = np.asarray(edge_mask.convert("L"), dtype=np.float64).reshape(-1) / 255.0
        edge_sums = np.bincount(inverse, weights=edge)
        candidate_array = np.asarray(candidates, dtype=np.uint8)
        mapped_indices = _nearest_palette_indices_numba(
            unique.astype(np.uint32),
            candidate_array,
            0.02,
            1 if color_distance == "oklab" else 0,
        )
        red, green, blue = unpack_packed_rgb_channels(unique)
        saturation, value = saturation_value_from_channels(red, green, blue)
        count_weights = counts.astype(np.float64)
        contour_weights = (
            count_weights * (1.0 + 0.65 * saturation * value)
            + edge_sums * (5.5 + 2.0 * saturation)
        )
        mass_array = np.bincount(mapped_indices, weights=count_weights, minlength=len(candidates))
        contour_array = np.bincount(mapped_indices, weights=contour_weights, minlength=len(candidates))
        for index, candidate in enumerate(candidates):
            projected[candidate] = float(mass_array[index])
            contour_votes[candidate] = float(contour_array[index])
    else:
        pixels = target.load()
        edges = edge_mask.convert("L").load()
        cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}

        for y in range(target.height):
            for x in range(target.width):
                color = pixels[x, y]
                mapped = nearest_candidate_cache(
                    color,
                    candidates,
                    cache,
                    hue_match_weight=0.02,
                    color_distance=color_distance,
                )
                saturation = rgb_saturation(color)
                value = max(color) / 255.0
                edge = edges[x, y] / 255.0
                projected[mapped] = projected.get(mapped, 0.0) + 1.0
                contour_votes[mapped] = contour_votes.get(mapped, 0.0) + (
                    1.0 + 5.5 * edge + 2.0 * edge * saturation + 0.65 * saturation * value
                )

    mass_items = [(color, mass) for color, mass in projected.items() if mass > 0]
    contour_items = [
        (
            color,
            (vote ** 0.86) * (0.55 + rgb_saturation(color)) * (0.45 + max(color) / 255.0),
        )
        for color, vote in contour_votes.items()
        if vote > 0
    ]
    if not mass_items:
        return median_cut_palette(source_weighted_colors, colors)

    rare_seed = projected_source_palette(
        target_image,
        source_weighted_colors,
        max(26, round(colors * 0.40)),
        rare_boost=True,
        min_saturation=min_saturation,
        min_value=min_value,
        color_distance=color_distance,
    )
    contour_slots = max(26, round(colors * 0.46))
    palette = merge_preserving_rare_seed(
        rare_seed,
        hue_peak_palette(
            contour_items,
            contour_slots,
            min_saturation=min_saturation,
            min_value=min_value,
        ),
        colors,
        rare_seed,
        min_value,
    )
    palette = merge_preserving_rare_seed(
        palette,
        shadow_spectrum_palette(
            mass_items,
            colors - len(palette),
            min_saturation=min_saturation,
            min_value=min_value,
        ),
        colors,
        rare_seed,
        min_value,
    )
    if len(palette) < colors:
        palette = merge_preserving_rare_seed(
            palette,
            sorted((color for color, _mass in mass_items), key=lambda color: projected[color], reverse=True),
            colors,
            rare_seed,
            min_value,
        )
    return sorted(palette[:colors], key=srgb_luma)


def projected_island_palette(
    target_image: Image.Image,
    source_weighted_colors: list[tuple[tuple[int, int, int], float]],
    colors: int,
    min_saturation: float,
    min_value: float,
    color_distance: str = "rgb",
) -> list[tuple[int, int, int]]:
    candidates = projected_candidate_bank(
        source_weighted_colors,
        colors,
        min_saturation=min_saturation,
        min_value=min_value,
    )
    target = target_image.convert("RGB")
    mass: dict[tuple[int, int, int], float] = {candidate: 0.0 for candidate in candidates}
    island_score: dict[tuple[int, int, int], float] = {candidate: 0.0 for candidate in candidates}
    tile_size = max(16, round(min(target.size) / 18))

    if _nearest_palette_indices_numba is not None and _project_island_scores_from_indices_numba is not None:
        rgb = rgb_array(target, dtype=np.uint8)
        packed = pack_rgb_array(rgb)
        unique, inverse = np.unique(packed.reshape(-1), return_inverse=True)
        candidate_array = np.asarray(candidates, dtype=np.uint8)
        mapped_unique = _nearest_palette_indices_numba(
            unique.astype(np.uint32),
            candidate_array,
            0.03,
            1 if color_distance == "oklab" else 0,
        )
        mapped_indices = mapped_unique[inverse].reshape((target.height, target.width))
        mass_array, island_score_array = _project_island_scores_from_indices_numba(
            mapped_indices,
            candidate_array,
            tile_size,
        )
        for index, candidate in enumerate(candidates):
            mass[candidate] = float(mass_array[index])
            island_score[candidate] = float(island_score_array[index])
    elif _project_island_scores_numba is not None:
        mass_array, island_score_array = _project_island_scores_numba(
            np.asarray(target, dtype=np.uint8),
            np.asarray(candidates, dtype=np.uint8),
            tile_size,
            0.03,
            1 if color_distance == "oklab" else 0,
        )
        for index, candidate in enumerate(candidates):
            mass[candidate] = float(mass_array[index])
            island_score[candidate] = float(island_score_array[index])
    else:
        pixels = target.load()
        cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}

        for tile_top in range(0, target.height, tile_size):
            for tile_left in range(0, target.width, tile_size):
                tile_counts: dict[tuple[int, int, int], float] = {}
                y_end = min(target.height, tile_top + tile_size)
                x_end = min(target.width, tile_left + tile_size)
                for y in range(tile_top, y_end):
                    for x in range(tile_left, x_end):
                        color = pixels[x, y]
                        mapped = nearest_candidate_cache(
                            color,
                            candidates,
                            cache,
                            hue_match_weight=0.03,
                            color_distance=color_distance,
                        )
                        tile_counts[mapped] = tile_counts.get(mapped, 0.0) + 1.0
                        mass[mapped] = mass.get(mapped, 0.0) + 1.0
                tile_total = max(1.0, sum(tile_counts.values()))
                for mapped, count in tile_counts.items():
                    local_fraction = count / tile_total
                    saturation = rgb_saturation(mapped)
                    value = max(mapped) / 255.0
                    island_score[mapped] = island_score.get(mapped, 0.0) + (
                        math.sqrt(count)
                        * ((1.0 - local_fraction) ** 0.35)
                        * (0.35 + saturation) ** 1.25
                        * (0.35 + value)
                    )

    mass_items = [(color, count) for color, count in mass.items() if count > 0]
    island_items = [
        (
            color,
            score + (mass.get(color, 0.0) ** 0.58) * (0.2 + rgb_saturation(color)),
        )
        for color, score in island_score.items()
        if score > 0
    ]
    if not mass_items:
        return median_cut_palette(source_weighted_colors, colors)

    rare_seed = projected_source_palette(
        target_image,
        source_weighted_colors,
        max(28, round(colors * 0.44)),
        rare_boost=True,
        min_saturation=min_saturation,
        min_value=min_value,
        color_distance=color_distance,
    )
    island_slots = max(30, round(colors * 0.52))
    palette = merge_preserving_rare_seed(
        rare_seed,
        hue_peak_palette(
            island_items,
            island_slots,
            min_saturation=min_saturation,
            min_value=min_value,
        ),
        colors,
        rare_seed,
        min_value,
    )
    palette = merge_preserving_rare_seed(
        palette,
        value_band_palette(
            mass_items,
            colors - len(palette),
            bands=12,
            chroma_bias=0.35,
            shadow_bias=0.8,
        ),
        colors,
        rare_seed,
        min_value,
    )
    if len(palette) < colors:
        palette = merge_preserving_rare_seed(
            palette,
            sorted((color for color, _mass in mass_items), key=lambda color: mass[color], reverse=True),
            colors,
            rare_seed,
            min_value,
        )
    return sorted(palette[:colors], key=srgb_luma)


def projected_anchor_palette(
    target_image: Image.Image,
    source_weighted_colors: list[tuple[tuple[int, int, int], float]],
    colors: int,
    min_saturation: float,
    min_value: float,
    color_distance: str = "rgb",
) -> list[tuple[int, int, int]]:
    candidates = projected_candidate_bank(
        source_weighted_colors,
        colors,
        min_saturation=min_saturation,
        min_value=min_value,
    )
    mass: dict[tuple[int, int, int], float] = {candidate: 0.0 for candidate in candidates}
    cell_scores: dict[tuple[int, int, int], float] = {}
    cell_candidates: dict[tuple[int, int, int], dict[tuple[int, int, int], float]] = {}

    if np is not None and _nearest_palette_indices_numba is not None:
        target = target_image.convert("RGB")
        rgb = rgb_array(target, dtype=np.uint8)
        packed = pack_rgb_array(rgb)
        unique, counts = np.unique(packed.reshape(-1), return_counts=True)
        candidate_array = np.asarray(candidates, dtype=np.uint8)
        mapped_indices = _nearest_palette_indices_numba(
            unique.astype(np.uint32),
            candidate_array,
            0.025,
            1 if color_distance == "oklab" else 0,
        )
        red, green, blue = unpack_packed_rgb_channels(unique)
        saturation, value = saturation_value_from_channels(red, green, blue)
        hue = hue_from_channels(red, green, blue)
        hue_bin = np.where(
            saturation < min_saturation,
            12,
            np.minimum(11, np.floor(hue / 30.0).astype(np.int64)),
        ).astype(np.int64)
        value_bin = np.minimum(5, np.floor(value * 6.0).astype(np.int64))
        chroma_bin = np.minimum(2, np.floor(np.maximum(0.0, saturation - min_saturation) * 4.0).astype(np.int64))
        cell_ids = value_bin * 39 + chroma_bin * 13 + hue_bin
        count_weights = counts.astype(np.float64)
        cell_weights = count_weights * (0.65 + saturation) * (0.35 + value)
        mass_array = np.bincount(mapped_indices, weights=count_weights, minlength=len(candidates))
        cell_score_array = np.bincount(cell_ids, weights=cell_weights, minlength=234)
        cell_candidate_weights = np.zeros((234, len(candidates)), dtype=np.float64)
        np.add.at(cell_candidate_weights, (cell_ids, mapped_indices), cell_weights)

        for index, candidate in enumerate(candidates):
            mass[candidate] = float(mass_array[index])

        for cell_id in np.nonzero(cell_score_array > 0)[0]:
            value_cell = int(cell_id // 39)
            remainder = int(cell_id % 39)
            chroma_cell = int(remainder // 13)
            hue_cell = int(remainder % 13)
            cell = (value_cell, chroma_cell, hue_cell)
            cell_scores[cell] = float(cell_score_array[cell_id])
            row = cell_candidate_weights[cell_id]
            active_indices = np.nonzero(row > 0)[0]
            cell_candidates[cell] = {
                candidates[int(index)]: float(row[index])
                for index in active_indices
            }
    else:
        target_counts = count_colors(target_image)
        cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for color, count in target_counts.items():
            mapped = nearest_candidate_cache(
                color,
                candidates,
                cache,
                hue_match_weight=0.025,
                color_distance=color_distance,
            )
            saturation = rgb_saturation(color)
            value = max(color) / 255.0
            hue_bin = 12 if saturation < min_saturation else min(11, int(rgb_hue(color) / 30.0))
            value_bin = min(5, int(value * 6))
            chroma_bin = min(2, int(max(0.0, saturation - min_saturation) * 4))
            cell = (value_bin, chroma_bin, hue_bin)
            cell_weight = count * (0.65 + saturation) * (0.35 + value)
            mass[mapped] = mass.get(mapped, 0.0) + count
            cell_scores[cell] = cell_scores.get(cell, 0.0) + cell_weight
            bucket = cell_candidates.setdefault(cell, {})
            bucket[mapped] = bucket.get(mapped, 0.0) + cell_weight

    allocation = allocate_slots_by_score(
        list(cell_scores.items()),
        max(38, round(colors * 0.66)),
        minimum_score_fraction=0.0008,
    )
    rare_seed = projected_source_palette(
        target_image,
        source_weighted_colors,
        max(24, round(colors * 0.38)),
        rare_boost=True,
        min_saturation=min_saturation,
        min_value=min_value,
        color_distance=color_distance,
    )
    palette = rare_seed[:]
    for cell, slots in sorted(allocation.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        bucket = cell_candidates.get(cell, {})
        if not bucket:
            continue
        ranked = sorted(
            bucket.items(),
            key=lambda item: item[1] * (0.4 + rgb_saturation(item[0])) * (0.35 + max(item[0]) / 255.0),
            reverse=True,
        )
        palette = merge_preserving_rare_seed(
            palette,
            [color for color, _score in ranked[:slots]],
            colors,
            rare_seed,
            min_value,
        )

    mass_items = [(color, count) for color, count in mass.items() if count > 0]
    palette = merge_preserving_rare_seed(
        palette,
        shadow_spectrum_palette(
            mass_items,
            colors - len(palette),
            min_saturation=min_saturation,
            min_value=min_value,
        ),
        colors,
        rare_seed,
        min_value,
    )
    if len(palette) < colors:
        palette = merge_preserving_rare_seed(
            palette,
            sorted((color for color, _mass in mass_items), key=lambda color: mass[color], reverse=True),
            colors,
            rare_seed,
            min_value,
        )
    return sorted(palette[:colors], key=srgb_luma)


def projected_frontier_palette(
    target_image: Image.Image,
    source_weighted_colors: list[tuple[tuple[int, int, int], float]],
    colors: int,
    min_saturation: float,
    min_value: float,
    color_distance: str = "rgb",
) -> list[tuple[int, int, int]]:
    candidates = projected_candidate_bank(
        source_weighted_colors,
        colors,
        min_saturation=min_saturation,
        min_value=min_value,
    )
    target = target_image.convert("RGB")
    edge_mask = build_sobel_edge_mask(target, threshold=0.025)
    mass: dict[tuple[int, int, int], float] = {candidate: 0.0 for candidate in candidates}
    contour: dict[tuple[int, int, int], float] = {candidate: 0.0 for candidate in candidates}
    island: dict[tuple[int, int, int], float] = {candidate: 0.0 for candidate in candidates}
    tile_size = max(16, round(min(target.size) / 20))

    if (
        np is not None
        and _nearest_palette_indices_numba is not None
        and _project_frontier_scores_from_indices_numba is not None
    ):
        rgb = rgb_array(target, dtype=np.uint8)
        packed = pack_rgb_array(rgb)
        unique, inverse = np.unique(packed.reshape(-1), return_inverse=True)
        candidate_array = np.asarray(candidates, dtype=np.uint8)
        mapped_unique = _nearest_palette_indices_numba(
            unique.astype(np.uint32),
            candidate_array,
            0.035,
            1 if color_distance == "oklab" else 0,
        )
        mapped_indices = mapped_unique[inverse].reshape((target.height, target.width))
        edge_array = np.asarray(edge_mask.convert("L"), dtype=np.uint8)
        mass_array, contour_array, island_array = _project_frontier_scores_from_indices_numba(
            mapped_indices,
            rgb,
            edge_array,
            candidate_array,
            tile_size,
        )
        for index, candidate in enumerate(candidates):
            mass[candidate] = float(mass_array[index])
            contour[candidate] = float(contour_array[index])
            island[candidate] = float(island_array[index])
    else:
        pixels = target.load()
        edges = edge_mask.convert("L").load()
        cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for tile_top in range(0, target.height, tile_size):
            for tile_left in range(0, target.width, tile_size):
                tile_counts: dict[tuple[int, int, int], float] = {}
                y_end = min(target.height, tile_top + tile_size)
                x_end = min(target.width, tile_left + tile_size)
                for y in range(tile_top, y_end):
                    for x in range(tile_left, x_end):
                        color = pixels[x, y]
                        mapped = nearest_candidate_cache(
                            color,
                            candidates,
                            cache,
                            hue_match_weight=0.035,
                            color_distance=color_distance,
                        )
                        saturation = rgb_saturation(color)
                        value = max(color) / 255.0
                        edge = edges[x, y] / 255.0
                        mass[mapped] = mass.get(mapped, 0.0) + 1.0
                        contour[mapped] = contour.get(mapped, 0.0) + 1.0 + 4.0 * edge + 1.4 * edge * saturation
                        tile_counts[mapped] = tile_counts.get(mapped, 0.0) + 1.0 + 0.25 * saturation * value
                tile_total = max(1.0, sum(tile_counts.values()))
                for mapped, count in tile_counts.items():
                    island[mapped] = island.get(mapped, 0.0) + math.sqrt(count) * (
                        1.0 - min(0.92, count / tile_total)
                    )

    active = [color for color, count in mass.items() if count > 0]
    if not active:
        return median_cut_palette(source_weighted_colors, colors)

    max_mass = max(mass[color] for color in active)
    max_contour = max(contour[color] for color in active)
    max_island = max(island[color] for color in active)
    scored: list[tuple[tuple[int, int, int], float]] = []
    for color in active:
        saturation = rgb_saturation(color)
        value = max(color) / 255.0
        mass_score = math.log1p(mass[color]) / math.log1p(max_mass)
        contour_score = math.log1p(contour[color]) / math.log1p(max_contour)
        island_score = math.log1p(island[color]) / math.log1p(max_island) if max_island > 0 else 0.0
        chroma_score = saturation * (0.35 + value)
        score = (
            0.95 * mass_score
            + 0.90 * contour_score
            + 1.25 * island_score
            + 0.70 * chroma_score
        )
        scored.append((color, score))

    cells: dict[tuple[int, int], list[tuple[tuple[int, int, int], float]]] = {}
    cell_scores: dict[tuple[int, int], float] = {}
    for color, score in scored:
        saturation = rgb_saturation(color)
        hue_bin = 16 if saturation < min_saturation else min(15, int(rgb_hue(color) / 22.5))
        value_bin = min(7, int(max(color) / 255.0 * 8))
        cell = (value_bin, hue_bin)
        cells.setdefault(cell, []).append((color, score))
        cell_scores[cell] = max(cell_scores.get(cell, 0.0), score)

    allocation = allocate_slots_by_score(
        list(cell_scores.items()),
        max(44, round(colors * 0.72)),
        minimum_score_fraction=0.0005,
    )
    rare_seed = projected_source_palette(
        target_image,
        source_weighted_colors,
        max(28, round(colors * 0.44)),
        rare_boost=True,
        min_saturation=min_saturation,
        min_value=min_value,
        color_distance=color_distance,
    )
    palette = rare_seed[:]
    for cell, slots in sorted(allocation.items(), key=lambda item: (item[0][0], item[0][1])):
        ranked = sorted(cells.get(cell, []), key=lambda item: item[1], reverse=True)
        palette = merge_preserving_rare_seed(
            palette,
            [color for color, _score in ranked[:slots]],
            colors,
            rare_seed,
            min_value,
        )

    palette = merge_preserving_rare_seed(
        palette,
        [color for color, _score in sorted(scored, key=lambda item: item[1], reverse=True)],
        colors,
        rare_seed,
        min_value,
    )
    return sorted(palette[:colors], key=srgb_luma)


def target_hue_rarity_bins(target_image: Image.Image) -> list[float]:
    if np is not None:
        rgb = rgb_array(target_image.convert("RGB"), dtype=np.float64)
        saturation, _value = saturation_value_arrays(rgb)
        hue = hue_array(rgb)
        active = saturation >= 0.06
        if not np.any(active):
            return [0.0 for _bin in range(36)]
        hue_indices = np.minimum(35, np.floor(hue / 10.0).astype(np.int64))
        hue_bins_array = np.bincount(
            hue_indices[active].reshape(-1),
            weights=saturation[active].reshape(-1),
            minlength=36,
        )
        total = float(hue_bins_array.sum())
        if total <= 0:
            return [0.0 for _bin in range(36)]
        average = total / 36.0
        return [1.0 - min(1.0, float(hue_mass) / max(average, 1e-6)) for hue_mass in hue_bins_array]

    hue_bins = [0.0 for _bin in range(36)]
    for red, green, blue in target_image.convert("RGB").getdata():
        color = (red, green, blue)
        saturation = rgb_saturation(color)
        if saturation < 0.06:
            continue
        hue_bins[min(35, int(rgb_hue(color) // 10))] += saturation
    total = sum(hue_bins)
    if total <= 0:
        return [0.0 for _bin in range(36)]
    average = total / len(hue_bins)
    return [1.0 - min(1.0, hue_mass / max(average, 1e-6)) for hue_mass in hue_bins]


def projected_graft_palette(
    target_image: Image.Image,
    source_weighted_colors: list[tuple[tuple[int, int, int], float]],
    colors: int,
    min_saturation: float,
    min_value: float,
    color_distance: str = "rgb",
) -> list[tuple[int, int, int]]:
    base = projected_source_palette(
        target_image,
        source_weighted_colors,
        colors,
        rare_boost=True,
        min_saturation=min_saturation,
        min_value=min_value,
        color_distance=color_distance,
    )
    candidates = projected_candidate_bank(
        source_weighted_colors,
        colors,
        min_saturation=min_saturation,
        min_value=min_value,
    )
    hue_rarity = target_hue_rarity_bins(target_image)
    candidate_mass: dict[tuple[int, int, int], float] = {candidate: 0.0 for candidate in candidates}
    base_usage: dict[tuple[int, int, int], float] = {color: 0.0 for color in base}

    if np is not None and _nearest_palette_indices_numba is not None:
        target = target_image.convert("RGB")
        rgb = rgb_array(target, dtype=np.uint8)
        packed = pack_rgb_array(rgb)
        unique, counts = np.unique(packed.reshape(-1), return_counts=True)
        candidate_indices = _nearest_palette_indices_numba(
            unique.astype(np.uint32),
            np.asarray(candidates, dtype=np.uint8),
            0.04,
            1 if color_distance == "oklab" else 0,
        )
        base_indices = _nearest_palette_indices_numba(
            unique.astype(np.uint32),
            np.asarray(base, dtype=np.uint8),
            0.04,
            1 if color_distance == "oklab" else 0,
        )
        red, green, blue = unpack_packed_rgb_channels(unique)
        saturation, value = saturation_value_from_channels(red, green, blue)
        votes = counts.astype(np.float64) * (0.55 + saturation) * (0.4 + value)
        candidate_mass_array = np.bincount(candidate_indices, weights=votes, minlength=len(candidates))
        base_usage_array = np.bincount(base_indices, weights=votes, minlength=len(base))
        for index, candidate in enumerate(candidates):
            candidate_mass[candidate] = float(candidate_mass_array[index])
        for index, color in enumerate(base):
            base_usage[color] = float(base_usage_array[index])
    else:
        target_counts = count_colors(target_image)
        cache_candidates: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        cache_base: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for color, count in target_counts.items():
            candidate = nearest_candidate_cache(
                color,
                candidates,
                cache_candidates,
                hue_match_weight=0.04,
                color_distance=color_distance,
            )
            base_color = nearest_candidate_cache(
                color,
                base,
                cache_base,
                hue_match_weight=0.04,
                color_distance=color_distance,
            )
            saturation = rgb_saturation(color)
            value = max(color) / 255.0
            vote = count * (0.55 + saturation) * (0.4 + value)
            candidate_mass[candidate] = candidate_mass.get(candidate, 0.0) + vote
            base_usage[base_color] = base_usage.get(base_color, 0.0) + vote

    def rarity(color: tuple[int, int, int]) -> float:
        if rgb_saturation(color) < min_saturation:
            return 0.0
        return hue_rarity[min(35, int(rgb_hue(color) // 10))]

    def protected(color: tuple[int, int, int]) -> bool:
        return (
            rgb_saturation(color) >= 0.12
            and max(color) / 255.0 >= min_value
            and rarity(color) >= 0.22
        )

    def candidate_score(color: tuple[int, int, int]) -> float:
        saturation = rgb_saturation(color)
        value = max(color) / 255.0
        return (candidate_mass.get(color, 0.0) ** 0.82) * (0.45 + saturation) * (0.55 + rarity(color)) * (0.35 + value)

    def removal_score(color: tuple[int, int, int]) -> float:
        guard = 1_000_000.0 if protected(color) else 0.0
        saturation = rgb_saturation(color)
        return guard + (base_usage.get(color, 0.0) ** 0.9) * (0.45 + saturation) * (0.45 + rarity(color))

    palette = base[:]
    ranked_candidates = sorted(
        [color for color in candidates if candidate_mass.get(color, 0.0) > 0],
        key=candidate_score,
        reverse=True,
    )
    replacements = 0
    max_replacements = max(8, round(colors * 0.16))
    for candidate in ranked_candidates:
        if replacements >= max_replacements:
            break
        if candidate in palette or rare_seed_conflict(candidate, base, min_value):
            continue
        removable = [color for color in palette if not protected(color)]
        if not removable:
            break
        weakest = min(removable, key=removal_score)
        palette.remove(weakest)
        palette.append(candidate)
        replacements += 1

    return sorted(palette[:colors], key=srgb_luma)


def median_cut_palette(
    weighted_colors: list[tuple[tuple[int, int, int], float]],
    colors: int,
) -> list[tuple[int, int, int]]:
    if len(weighted_colors) <= colors:
        return sorted((color for color, _count in weighted_colors), key=srgb_luma)

    boxes = [ColorBox.from_colors(weighted_colors)]
    while len(boxes) < colors:
        splittable = [box for box in boxes if len(box.colors) > 1 and box.largest_range() > 0]
        if not splittable:
            break

        box = max(splittable, key=lambda item: item.weight * max(1, item.largest_range()))
        boxes.remove(box)
        split = split_color_box(box)
        if split is None:
            boxes.append(box)
            break
        boxes.extend(split)

    palette: list[tuple[int, int, int]] = []
    for box in boxes:
        average = weighted_average_color(box.colors)
        palette.append(average)

    return sorted(set(palette), key=srgb_luma)


def map_to_palette(
    image: Image.Image,
    palette: list[tuple[int, int, int]],
    hue_match_weight: float = 0.0,
    color_distance: str = "rgb",
) -> Image.Image:
    if not palette:
        raise RuntimeError("cannot map image to an empty palette")

    if np is not None:
        rgb = rgb_array(image, dtype=np.uint8)
        packed = pack_rgb_array(rgb)
        unique, inverse = np.unique(packed.reshape(-1), return_inverse=True)
        if _nearest_palette_map_numba is not None:
            mapped = _nearest_palette_map_numba(
                unique.astype(np.uint32),
                np.asarray(palette, dtype=np.uint8),
                hue_match_weight,
                1 if color_distance == "oklab" else 0,
            )
        else:
            mapped = np.asarray(
                [
                    nearest_palette_color(
                        unpack_rgb_value(int(color)),
                        palette,
                        hue_match_weight=hue_match_weight,
                        color_distance=color_distance,
                    )
                    for color in unique
                ],
                dtype=np.uint8,
            )
        output = mapped[inverse].reshape((image.height, image.width, 3))
        return Image.fromarray(output, mode="RGB")

    out = Image.new("RGB", image.size)
    source = image.load()
    target = out.load()
    cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for y in range(image.height):
        for x in range(image.width):
            color = source[x, y]
            mapped = cache.get(color)
            if mapped is None:
                mapped = nearest_palette_color(
                    color,
                    palette,
                    hue_match_weight=hue_match_weight,
                    color_distance=color_distance,
                )
                cache[color] = mapped
            target[x, y] = mapped

    return out


def nearest_palette_color(
    color: tuple[int, int, int],
    palette: Iterable[tuple[int, int, int]],
    hue_match_weight: float = 0.0,
    color_distance: str = "rgb",
) -> tuple[int, int, int]:
    red, green, blue = color
    source_saturation = rgb_saturation(color)
    source_hue = rgb_hue(color) if hue_match_weight > 0 and source_saturation >= 0.08 else 0.0

    def score(item: tuple[int, int, int]) -> float:
        if color_distance == "oklab":
            rgb_distance = oklab_distance_squared(color, item)
        else:
            rgb_distance = (
                0.30 * (red - item[0]) * (red - item[0])
                + 0.59 * (green - item[1]) * (green - item[1])
                + 0.11 * (blue - item[2]) * (blue - item[2])
            )
        if hue_match_weight <= 0 or source_saturation < 0.08:
            return rgb_distance

        item_saturation = rgb_saturation(item)
        if item_saturation < 0.08:
            return rgb_distance

        hue_distance = hue_distance_degrees(source_hue, rgb_hue(item)) / 180.0
        saturation_distance = abs(source_saturation - item_saturation)
        return rgb_distance + hue_match_weight * 65025.0 * (
            hue_distance * hue_distance * source_saturation * item_saturation
            + 0.2 * saturation_distance * saturation_distance
        )

    return min(
        palette,
        key=score,
    )


def map_to_palette_with_rare_guard(
    image: Image.Image,
    palette: list[tuple[int, int, int]],
    hue_match_weight: float = 0.0,
    color_distance: str = "rgb",
    min_value: float = 0.05,
) -> Image.Image:
    rare_seeds = [
        color
        for color in palette
        if rgb_saturation(color) >= 0.16 and max(color) / 255.0 >= min_value
    ]
    if not rare_seeds:
        return map_to_palette(
            image,
            palette,
            hue_match_weight=hue_match_weight,
            color_distance=color_distance,
        )

    if np is not None:
        rgb = rgb_array(image, dtype=np.uint8)
        packed = pack_rgb_array(rgb)
        unique, inverse = np.unique(packed.reshape(-1), return_inverse=True)
        if _nearest_palette_map_rare_guard_numba is not None:
            mapped = _nearest_palette_map_rare_guard_numba(
                unique.astype(np.uint32),
                np.asarray(palette, dtype=np.uint8),
                np.asarray(rare_seeds, dtype=np.uint8),
                hue_match_weight,
                1 if color_distance == "oklab" else 0,
            )
        else:
            mapped_colors: list[tuple[int, int, int]] = []
            for packed_color in unique:
                color = unpack_rgb_value(int(packed_color))
                mapped_color = nearest_palette_color(
                    color,
                    palette,
                    hue_match_weight=hue_match_weight,
                    color_distance=color_distance,
                )
                source_saturation = rgb_saturation(color)
                if source_saturation >= 0.035:
                    source_hue = rgb_hue(color)
                    best_seed = min(
                        rare_seeds,
                        key=lambda seed: (
                            oklab_distance_squared(color, seed)
                            + 65025.0
                            * (hue_distance_degrees(source_hue, rgb_hue(seed)) / 180.0)
                            ** 2
                            * max(source_saturation, 0.08)
                        ),
                    )
                    hue_distance = hue_distance_degrees(source_hue, rgb_hue(best_seed))
                    if hue_distance <= 42:
                        normal_distance = (
                            oklab_distance_squared(color, mapped_color)
                            if color_distance == "oklab"
                            else color_distance_squared(color, mapped_color)
                        )
                        seed_distance = (
                            oklab_distance_squared(color, best_seed)
                            if color_distance == "oklab"
                            else color_distance_squared(color, best_seed)
                        )
                        if seed_distance <= normal_distance * 1.55 + 420:
                            mapped_color = best_seed
                mapped_colors.append(mapped_color)
            mapped = np.asarray(mapped_colors, dtype=np.uint8)
        output = mapped[inverse].reshape((image.height, image.width, 3))
        return Image.fromarray(output, mode="RGB")

    out = Image.new("RGB", image.size)
    source = image.load()
    target = out.load()
    cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for y in range(image.height):
        for x in range(image.width):
            color = source[x, y]
            mapped = cache.get(color)
            if mapped is None:
                mapped = nearest_palette_color(
                    color,
                    palette,
                    hue_match_weight=hue_match_weight,
                    color_distance=color_distance,
                )
                source_saturation = rgb_saturation(color)
                if source_saturation >= 0.035:
                    source_hue = rgb_hue(color)
                    best_seed = min(
                        rare_seeds,
                        key=lambda seed: (
                            oklab_distance_squared(color, seed)
                            + 65025.0
                            * (hue_distance_degrees(source_hue, rgb_hue(seed)) / 180.0)
                            ** 2
                            * max(source_saturation, 0.08)
                        ),
                    )
                    hue_distance = hue_distance_degrees(source_hue, rgb_hue(best_seed))
                    if hue_distance <= 42:
                        normal_distance = (
                            oklab_distance_squared(color, mapped)
                            if color_distance == "oklab"
                            else color_distance_squared(color, mapped)
                        )
                        seed_distance = (
                            oklab_distance_squared(color, best_seed)
                            if color_distance == "oklab"
                            else color_distance_squared(color, best_seed)
                        )
                        if seed_distance <= normal_distance * 1.55 + 420:
                            mapped = best_seed
                cache[color] = mapped
            target[x, y] = mapped

    return out


def clamp_channel(value: float) -> int:
    return max(0, min(255, round(value)))


def snap_channel(value: int, step: int) -> int:
    if step <= 1:
        return value
    return clamp_channel(round(value / step) * step)


def is_protected_hue(color: tuple[int, int, int], config: PixelArtConfig) -> bool:
    if not config.protected_hue_ranges:
        return False
    saturation = rgb_saturation(color)
    return saturation >= config.protected_hue_min_saturation and hue_in_ranges(
        rgb_hue(color),
        config.protected_hue_ranges,
    )


def local_luma_range(
    pixels,
    width: int,
    height: int,
    x: int,
    y: int,
) -> float:
    minimum = 255.0
    maximum = 0.0
    for dy in (-1, 0, 1):
        yy = min(height - 1, max(0, y + dy))
        for dx in (-1, 0, 1):
            xx = min(width - 1, max(0, x + dx))
            luma = srgb_luma(pixels[xx, yy])
            minimum = min(minimum, luma)
            maximum = max(maximum, luma)
    return maximum - minimum


def snap_low_detail_regions(
    image: Image.Image,
    edge_mask: Image.Image,
    config: PixelArtConfig,
) -> Image.Image:
    if config.flat_region_channel_step <= 1:
        if config.flat_region_palette_colors <= 0:
            return image

    source = image.convert("RGB")
    if np is not None:
        rgb = rgb_array(source, dtype=np.uint8)
        rgb_float = rgb.astype(np.float64)
        saturation, _value = saturation_value_arrays(rgb_float)
        edge = np.asarray(edge_mask.convert("L"), dtype=np.float64) / 255.0
        luma = 0.2126 * rgb_float[:, :, 0] + 0.7152 * rgb_float[:, :, 1] + 0.0722 * rgb_float[:, :, 2]
        padded = np.pad(luma, 1, mode="edge")
        neighbors = [
            padded[dy : dy + source.height, dx : dx + source.width]
            for dy in range(3)
            for dx in range(3)
        ]
        local_range = np.maximum.reduce(neighbors) - np.minimum.reduce(neighbors)
        protected = np.zeros((source.height, source.width), dtype=bool)
        if config.protected_hue_ranges:
            hue = hue_array(rgb_float)
            protected = hue_ranges_mask_array(hue, config.protected_hue_ranges)
            protected &= saturation >= config.protected_hue_min_saturation

        low_detail = (
            (edge <= config.flat_region_edge_threshold)
            & (local_range <= config.flat_region_luma_range)
            & (saturation <= config.flat_region_max_saturation)
            & ~protected
        )
        if not np.any(low_detail):
            return image

        output = rgb.copy()
        if config.flat_region_palette_colors > 0:
            packed = pack_rgb_array(rgb)
            unique, first_index, inverse = np.unique(
                packed[low_detail],
                return_index=True,
                return_inverse=True,
            )
            counts = np.bincount(inverse).astype(np.float64)
            scan_order = np.argsort(first_index)
            weighted_colors = [
                (unpack_rgb_value(int(color)), float(count))
                for color, count in zip(unique[scan_order], counts[scan_order], strict=True)
            ]
            flat_palette = median_cut_palette(
                weighted_colors,
                min(config.flat_region_palette_colors, len(weighted_colors)),
            )
            low_pixels = rgb[low_detail]
            low_packed = (
                (low_pixels[:, 0].astype(np.uint32) << 16)
                | (low_pixels[:, 1].astype(np.uint32) << 8)
                | low_pixels[:, 2].astype(np.uint32)
            )
            unique_low, inverse_low = np.unique(low_packed, return_inverse=True)
            if _nearest_palette_map_numba is not None:
                mapped = _nearest_palette_map_numba(
                    unique_low.astype(np.uint32),
                    np.asarray(flat_palette, dtype=np.uint8),
                    config.hue_match_weight,
                    1 if config.color_distance == "oklab" else 0,
                )
            else:
                mapped = np.asarray(
                    [
                        nearest_palette_color(
                            unpack_rgb_value(int(color)),
                            flat_palette,
                            hue_match_weight=config.hue_match_weight,
                            color_distance=config.color_distance,
                        )
                        for color in unique_low
                    ],
                    dtype=np.uint8,
                )
            output[low_detail] = mapped[inverse_low]
        else:
            step = config.flat_region_channel_step
            snapped = np.clip(np.rint(rgb[low_detail].astype(np.float64) / step) * step, 0, 255)
            output[low_detail] = snapped.astype(np.uint8)

        return Image.fromarray(output, mode="RGB")

    mask = edge_mask.convert("L").load()
    pixels = source.load()

    low_detail: list[tuple[int, int]] = []
    weighted_colors: dict[tuple[int, int, int], float] = {}
    for y in range(source.height):
        for x in range(source.width):
            color = pixels[x, y]
            edge = mask[x, y] / 255.0
            if (
                edge <= config.flat_region_edge_threshold
                and local_luma_range(pixels, source.width, source.height, x, y)
                <= config.flat_region_luma_range
                and rgb_saturation(color) <= config.flat_region_max_saturation
                and not is_protected_hue(color, config)
            ):
                low_detail.append((x, y))
                weighted_colors[color] = weighted_colors.get(color, 0.0) + 1.0

    if not low_detail:
        return image

    flat_palette: list[tuple[int, int, int]] = []
    if config.flat_region_palette_colors > 0 and weighted_colors:
        flat_palette = median_cut_palette(
            list(weighted_colors.items()),
            min(config.flat_region_palette_colors, len(weighted_colors)),
        )

    out = Image.new("RGB", source.size)
    target = out.load()
    cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    low_detail_set = set(low_detail)

    for y in range(source.height):
        for x in range(source.width):
            color = pixels[x, y]
            if (x, y) in low_detail_set:
                snapped = cache.get(color)
                if snapped is None:
                    if flat_palette:
                        snapped = nearest_palette_color(
                            color,
                            flat_palette,
                            hue_match_weight=config.hue_match_weight,
                            color_distance=config.color_distance,
                        )
                    else:
                        snapped = (
                            snap_channel(color[0], config.flat_region_channel_step),
                            snap_channel(color[1], config.flat_region_channel_step),
                            snap_channel(color[2], config.flat_region_channel_step),
                        )
                    cache[color] = snapped
                target[x, y] = snapped
            else:
                target[x, y] = color

    return out


def cleanup_single_pixel_mixels(image: Image.Image, config: PixelArtConfig) -> Image.Image:
    if config.mixel_cleanup_passes <= 0:
        return image

    output = image.convert("RGB")
    if _cleanup_single_pixel_mixels_numba is not None:
        protected_ranges = np.asarray(config.protected_hue_ranges, dtype=np.float64)
        if protected_ranges.size == 0:
            protected_ranges = np.empty((0, 2), dtype=np.float64)
        else:
            protected_ranges = protected_ranges.reshape((-1, 2))
        output_array = _cleanup_single_pixel_mixels_numba(
            np.asarray(output, dtype=np.uint8),
            config.mixel_cleanup_passes,
            config.mixel_cleanup_min_neighbors,
            config.mixel_cleanup_distance * config.mixel_cleanup_distance,
            config.mixel_cleanup_max_saturation,
            protected_ranges,
            config.protected_hue_min_saturation,
        )
        return Image.fromarray(output_array, mode="RGB")

    max_distance = config.mixel_cleanup_distance * config.mixel_cleanup_distance
    for _pass in range(config.mixel_cleanup_passes):
        source = output
        pixels = source.load()
        next_image = source.copy()
        target = next_image.load()
        changes = 0

        for y in range(source.height):
            for x in range(source.width):
                color = pixels[x, y]
                if is_protected_hue(color, config) or rgb_saturation(color) > config.mixel_cleanup_max_saturation:
                    continue

                counts: dict[tuple[int, int, int], int] = {}
                for dy in (-1, 0, 1):
                    yy = min(source.height - 1, max(0, y + dy))
                    for dx in (-1, 0, 1):
                        xx = min(source.width - 1, max(0, x + dx))
                        neighbor = pixels[xx, yy]
                        counts[neighbor] = counts.get(neighbor, 0) + 1

                current_count = counts.get(color, 0)
                replacement, replacement_count = max(counts.items(), key=lambda item: item[1])
                if (
                    current_count <= 1
                    and replacement_count >= config.mixel_cleanup_min_neighbors
                    and color_distance_squared(color, replacement) <= max_distance
                ):
                    target[x, y] = replacement
                    changes += 1

        output = next_image
        if changes == 0:
            break

    return output


def apply_ordered_dither(
    image: Image.Image,
    palette: list[tuple[int, int, int]],
    strength: float,
    scope: str = "global",
    edge_mask: Image.Image | None = None,
    edge_threshold: float = 0.28,
    luma_range_threshold: float = 45.0,
    error_threshold: float = 3.0,
    hue_match_weight: float = 0.0,
    color_distance: str = "rgb",
) -> Image.Image:
    if _ordered_dither_numba is not None:
        source_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if edge_mask is None:
            edge_array = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            edge_array = np.asarray(edge_mask.convert("L"), dtype=np.uint8)
        output = _ordered_dither_numba(
            source_array,
            np.asarray(palette, dtype=np.uint8),
            BAYER_4X4_ARRAY,
            strength,
            1 if scope == "adaptive" else 0,
            edge_array,
            edge_threshold,
            luma_range_threshold,
            error_threshold,
            hue_match_weight,
            1 if color_distance == "oklab" else 0,
        )
        return Image.fromarray(output, mode="RGB")

    out = Image.new("RGB", image.size)
    source = image.load()
    target = out.load()
    mask = edge_mask.convert("L").load() if edge_mask is not None else None

    for y in range(image.height):
        for x in range(image.width):
            red, green, blue = source[x, y]
            scale = 1.0
            if scope == "adaptive":
                edge = mask[x, y] / 255.0 if mask is not None else 0.0
                if edge_threshold <= 0:
                    edge_factor = 1.0 if edge <= 0 else 0.0
                else:
                    edge_factor = 1.0 - smoothstep(edge_threshold * 0.5, edge_threshold, edge)
                local_range = local_luma_range(source, image.width, image.height, x, y)
                if luma_range_threshold <= 0:
                    detail_factor = 1.0 if local_range <= 0 else 0.0
                else:
                    detail_factor = 1.0 - smoothstep(
                        luma_range_threshold * 0.65,
                        luma_range_threshold,
                        local_range,
                    )
                nearest = nearest_palette_color(
                    (red, green, blue),
                    palette,
                    hue_match_weight=hue_match_weight,
                    color_distance=color_distance,
                )
                quant_error = color_distance_squared((red, green, blue), nearest)
                if error_threshold <= 0:
                    error_factor = 1.0
                else:
                    error2 = error_threshold * error_threshold
                    error_factor = smoothstep(error2, error2 * 4.0, quant_error)
                scale = edge_factor * detail_factor * error_factor

            threshold = ((BAYER_4X4[y % 4][x % 4] + 0.5) / 16.0 - 0.5) * strength * scale
            shifted = (
                clamp_channel(red + threshold),
                clamp_channel(green + threshold),
                clamp_channel(blue + threshold),
            )
            target[x, y] = nearest_palette_color(
                shifted,
                palette,
                hue_match_weight=hue_match_weight,
                color_distance=color_distance,
            )

    return out


def apply_floyd_steinberg_dither(
    image: Image.Image,
    palette: list[tuple[int, int, int]],
    hue_match_weight: float = 0.0,
    color_distance: str = "rgb",
) -> Image.Image:
    if _floyd_steinberg_dither_numba is not None:
        source_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        output = _floyd_steinberg_dither_numba(
            source_array,
            np.asarray(palette, dtype=np.uint8),
            hue_match_weight,
            1 if color_distance == "oklab" else 0,
        )
        return Image.fromarray(output, mode="RGB")

    width, height = image.size
    work = [
        [[float(channel) for channel in image.getpixel((x, y))] for x in range(width)]
        for y in range(height)
    ]
    out = Image.new("RGB", image.size)
    target = out.load()

    def add_error(x: int, y: int, error: tuple[float, float, float], weight: float) -> None:
        if x < 0 or x >= width or y < 0 or y >= height:
            return
        work[y][x][0] = min(255.0, max(0.0, work[y][x][0] + error[0] * weight))
        work[y][x][1] = min(255.0, max(0.0, work[y][x][1] + error[1] * weight))
        work[y][x][2] = min(255.0, max(0.0, work[y][x][2] + error[2] * weight))

    for y in range(height):
        for x in range(width):
            old = (
                clamp_channel(work[y][x][0]),
                clamp_channel(work[y][x][1]),
                clamp_channel(work[y][x][2]),
            )
            new = nearest_palette_color(
                old,
                palette,
                hue_match_weight=hue_match_weight,
                color_distance=color_distance,
            )
            target[x, y] = new
            error = (
                float(old[0] - new[0]),
                float(old[1] - new[1]),
                float(old[2] - new[2]),
            )
            add_error(x + 1, y, error, 7.0 / 16.0)
            add_error(x - 1, y + 1, error, 3.0 / 16.0)
            add_error(x, y + 1, error, 5.0 / 16.0)
            add_error(x + 1, y + 1, error, 1.0 / 16.0)

    return out


def effective_dither_mode(config: PixelArtConfig) -> str:
    if config.grid_snap_enabled and config.grid_snap_quantize_first:
        return "none"
    return config.dither


def quantize_to_palette(
    image: Image.Image,
    config: PixelArtConfig,
    edge_mask: Image.Image | None = None,
    palette_image: Image.Image | None = None,
    palette_edge_mask: Image.Image | None = None,
) -> tuple[Image.Image, list[str]]:
    dither = effective_dither_mode(config)
    if dither != "none" and config.colors > 256:
        raise ValueError("--colors above 256 currently supports --dither none only")

    output, palette_rgb = quantize_median_cut_rgb(
        image,
        config.colors,
        edge_mask=edge_mask,
        palette_image=palette_image,
        palette_edge_mask=palette_edge_mask,
        edge_palette_weight=config.edge_palette_weight,
        palette_strategy=config.palette_strategy,
        color_distance=config.color_distance,
        accent_palette_weight=config.accent_palette_weight,
        hue_rarity_weight=config.hue_rarity_weight,
        interesting_color_slots=config.interesting_color_slots,
        interesting_min_saturation=config.interesting_min_saturation,
        interesting_min_value=config.interesting_min_value,
        protected_hue_ranges=config.protected_hue_ranges,
        protected_hue_weight=config.protected_hue_weight,
        protected_hue_slots=config.protected_hue_slots,
        protected_hue_min_saturation=config.protected_hue_min_saturation,
        hue_match_weight=config.hue_match_weight,
    )
    if dither == "none":
        return output, [f"#{red:02x}{green:02x}{blue:02x}" for red, green, blue in palette_rgb]

    if dither == "ordered":
        output = apply_ordered_dither(
            image,
            palette_rgb,
            config.dither_strength,
            scope=config.dither_scope,
            edge_mask=edge_mask,
            edge_threshold=config.dither_edge_threshold,
            luma_range_threshold=config.dither_luma_range,
            error_threshold=config.dither_error_threshold,
            hue_match_weight=config.hue_match_weight,
            color_distance=config.color_distance,
        )
    elif dither == "floyd":
        output = apply_floyd_steinberg_dither(
            image,
            palette_rgb,
            hue_match_weight=config.hue_match_weight,
            color_distance=config.color_distance,
        )
    else:
        raise ValueError(f"unsupported dither mode: {dither}")

    return output, [f"#{red:02x}{green:02x}{blue:02x}" for red, green, blue in palette_rgb]


def count_colors(image: Image.Image) -> dict[tuple[int, int, int], float]:
    if np is not None:
        rgb = rgb_array(image, dtype=np.uint8)
        packed = pack_rgb_array(rgb)
        unique, counts = np.unique(packed.reshape(-1), return_counts=True)
        return {
            unpack_rgb_value(int(color)): float(count)
            for color, count in zip(unique, counts, strict=True)
        }

    color_counts = image.convert("RGB").getcolors(maxcolors=image.width * image.height)
    if not color_counts:
        raise RuntimeError("could not collect image colors")
    return {color: float(count) for count, color in color_counts}


def clamp_to_color_limit(image: Image.Image, colors: int, config: PixelArtConfig) -> tuple[Image.Image, list[str]]:
    counts = count_colors(image)
    if len(counts) <= colors:
        palette = sorted(counts.keys(), key=srgb_luma)
        return image, [f"#{red:02x}{green:02x}{blue:02x}" for red, green, blue in palette]

    weighted_colors = list(counts.items())
    reserved_palette: list[tuple[int, int, int]] = []
    if config.protected_hue_ranges and config.protected_hue_slots > 0:
        protected_items = [
            (color, count)
            for color, count in counts.items()
            if is_protected_hue(color, config)
        ]
        if protected_items:
            reserved_palette = merge_palette_slots(
                reserved_palette,
                median_cut_palette(
                    protected_items,
                    min(config.protected_hue_slots, colors - 1),
                ),
                colors,
            )

    if config.palette_strategy == "interesting" and config.interesting_color_slots > 0:
        remaining_reserved_slots = max(0, colors - len(reserved_palette) - 1)
        reserved_palette = merge_palette_slots(
            reserved_palette,
            select_interesting_palette(
                weighted_colors,
                min(config.interesting_color_slots, remaining_reserved_slots),
                min_saturation=config.interesting_min_saturation,
                min_value=config.interesting_min_value,
            ),
            colors,
        )

    remaining_colors = max(1, colors - len(reserved_palette))
    base_items = weighted_colors
    palette_rgb = median_cut_palette(base_items, remaining_colors)
    palette_rgb = merge_palette_slots(reserved_palette, palette_rgb, colors)
    output = map_to_palette(
        image,
        palette_rgb,
        hue_match_weight=config.hue_match_weight,
        color_distance=config.color_distance,
    )
    palette = sorted(set(output.getdata()), key=srgb_luma)
    return output, [f"#{red:02x}{green:02x}{blue:02x}" for red, green, blue in palette]


def write_palette(path: Path, palette: list[str]) -> None:
    path.write_text("\n".join(palette) + "\n", encoding="utf-8")


def build_default_preview_path(output: Path, scale: int) -> Path:
    return output.with_name(f"{output.stem}@{scale}x{output.suffix}")


def prepare_palette_source(config: PixelArtConfig) -> tuple[Image.Image, Image.Image] | tuple[None, None]:
    if config.palette_source is None:
        return None, None

    palette_base = prepare_base_image(Image.open(config.palette_source), config)
    palette_edge_mask = build_sobel_edge_mask(palette_base, threshold=config.edge_threshold)
    palette_processed = bilateral_smooth(
        palette_base,
        radius=config.bilateral_radius,
        sigma_color=config.bilateral_sigma_color,
        sigma_space=config.bilateral_sigma_space,
        mode=config.bilateral_mode,
        edge_mask=palette_edge_mask,
    )
    palette_prepared = grade_image(palette_processed, config)
    palette_prepared = selective_edge_sharpen(
        palette_prepared,
        palette_edge_mask,
        config.edge_sharpen,
    )
    return palette_prepared, palette_edge_mask


def convert(input_path: Path, output_path: Path, preview_path: Path | None, config: PixelArtConfig) -> dict:
    image = Image.open(input_path)
    base = prepare_base_image(image, config)
    edge_mask = build_sobel_edge_mask(base, threshold=config.edge_threshold)
    processed = bilateral_smooth(
        base,
        radius=config.bilateral_radius,
        sigma_color=config.bilateral_sigma_color,
        sigma_space=config.bilateral_sigma_space,
        mode=config.bilateral_mode,
        edge_mask=edge_mask,
    )
    prepared = grade_image(processed, config)
    prepared = selective_edge_sharpen(prepared, edge_mask, config.edge_sharpen)
    palette_image, palette_edge_mask = prepare_palette_source(config)
    pixel_art, palette = quantize_to_palette(
        prepared,
        config,
        edge_mask=edge_mask,
        palette_image=palette_image,
        palette_edge_mask=palette_edge_mask,
    )
    source_luma = luma_mean(base)
    source_saturation = luma_weighted_saturation_mean(base)
    quantized_luma = luma_mean(pixel_art)
    quantized_saturation = luma_weighted_saturation_mean(pixel_art)

    if config.preserve_luma:
        pixel_art = match_luma_mean(pixel_art, source_luma)

    if config.preserve_saturation:
        pixel_art = match_luma_weighted_saturation(pixel_art, source_saturation)

    if config.preserve_luma:
        pixel_art = match_luma_mean(pixel_art, source_luma)

    pixel_art = snap_low_detail_regions(pixel_art, edge_mask, config)
    pixel_art = cleanup_single_pixel_mixels(pixel_art, config)

    pixel_art, palette = clamp_to_color_limit(pixel_art, config.colors, config)

    if config.preserve_luma or config.preserve_saturation:
        palette = [
            f"#{red:02x}{green:02x}{blue:02x}"
            for red, green, blue in sorted(set(pixel_art.getdata()), key=srgb_luma)
        ]

    output_luma = luma_mean(pixel_art)
    output_saturation = luma_weighted_saturation_mean(pixel_art)
    dither = effective_dither_mode(config)
    dither_disabled_reason = (
        "grid_quantize_first"
        if config.dither != "none" and dither == "none" and config.grid_snap_enabled and config.grid_snap_quantize_first
        else None
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixel_art.save(output_path)

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "source_size": list(image.size),
        "logical_size": [config.target_width, config.target_height],
        "colors_requested": config.colors,
        "colors_written": len(palette),
        "dither": dither,
        "dither_requested": config.dither,
        "dither_disabled_reason": dither_disabled_reason,
        "dither_strength": config.dither_strength if dither == "ordered" else None,
        "dither_scope": config.dither_scope if dither == "ordered" else None,
        "dither_edge_threshold": config.dither_edge_threshold if dither == "ordered" else None,
        "dither_luma_range": config.dither_luma_range if dither == "ordered" else None,
        "dither_error_threshold": config.dither_error_threshold if dither == "ordered" else None,
        "resample": config.resample,
        "grid_snap_enabled": config.grid_snap_enabled,
        "grid_snap_method": config.grid_snap_method if config.grid_snap_enabled else None,
        "grid_snap_quantize_first": config.grid_snap_quantize_first if config.grid_snap_enabled else None,
        "grid_snap_dark_threshold": config.grid_snap_dark_threshold if config.grid_snap_enabled else None,
        "preview_scale": config.preview_scale,
        "preserve_luma": config.preserve_luma,
        "preserve_saturation": config.preserve_saturation,
        "palette_source": str(config.palette_source) if config.palette_source else None,
        "bilateral_radius": config.bilateral_radius,
        "bilateral_mode": config.bilateral_mode,
        "bilateral_sigma_color": config.bilateral_sigma_color,
        "bilateral_sigma_space": config.bilateral_sigma_space,
        "edge_palette_weight": config.edge_palette_weight,
        "edge_sharpen": config.edge_sharpen,
        "edge_threshold": config.edge_threshold,
        "palette_strategy": config.palette_strategy,
        "color_distance": config.color_distance,
        "accent_palette_weight": config.accent_palette_weight,
        "hue_rarity_weight": config.hue_rarity_weight,
        "interesting_color_slots": config.interesting_color_slots,
        "interesting_min_saturation": config.interesting_min_saturation,
        "interesting_min_value": config.interesting_min_value,
        "protected_hue_ranges": list(config.protected_hue_ranges),
        "protected_hue_weight": config.protected_hue_weight,
        "protected_hue_slots": config.protected_hue_slots,
        "protected_hue_min_saturation": config.protected_hue_min_saturation,
        "hue_match_weight": config.hue_match_weight,
        "flat_region_palette_colors": config.flat_region_palette_colors,
        "flat_region_channel_step": config.flat_region_channel_step,
        "flat_region_max_saturation": config.flat_region_max_saturation,
        "flat_region_edge_threshold": config.flat_region_edge_threshold,
        "flat_region_luma_range": config.flat_region_luma_range,
        "mixel_cleanup_passes": config.mixel_cleanup_passes,
        "mixel_cleanup_min_neighbors": config.mixel_cleanup_min_neighbors,
        "mixel_cleanup_distance": config.mixel_cleanup_distance,
        "mixel_cleanup_max_saturation": config.mixel_cleanup_max_saturation,
        "source_luma_mean": round(source_luma, 3),
        "quantized_luma_mean_before_match": round(quantized_luma, 3),
        "output_luma_mean": round(output_luma, 3),
        "source_luma_weighted_saturation": round(source_saturation, 4),
        "quantized_luma_weighted_saturation_before_match": round(quantized_saturation, 4),
        "output_luma_weighted_saturation": round(output_saturation, 4),
    }

    palette_path = output_path.with_suffix(".palette.txt")
    write_palette(palette_path, palette)
    manifest["palette"] = str(palette_path)

    if preview_path:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview = pixel_art.resize(
            (config.target_width * config.preview_scale, config.target_height * config.preview_scale),
            resample=Image.Resampling.NEAREST,
        )
        preview.save(preview_path)
        manifest["preview"] = str(preview_path)

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Snap generated art into a strict low-resolution pixel-art grid."
    )
    parser.add_argument("input", type=Path, help="Source PNG/JPG/WebP image.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Logical low-res PNG output.")
    parser.add_argument(
        "--size",
        type=parse_size,
        default=(1024, 576),
        help="Logical pixel resolution, default: 1024x576.",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=512,
        help="Target palette color count, default: 512.",
    )
    parser.add_argument(
        "--dither",
        choices=("ordered", "floyd", "none"),
        default="none",
        help="Dithering mode after palette reduction, default: none. Ignored when --grid-snap uses --grid-quantize-first.",
    )
    parser.add_argument(
        "--dither-strength",
        type=float,
        default=14.0,
        help="Ordered dither channel offset strength, default: 14.",
    )
    parser.add_argument(
        "--dither-scope",
        choices=("global", "adaptive"),
        default="global",
        help="Ordered dither placement: global or adaptive smooth non-edge regions, default: global.",
    )
    parser.add_argument(
        "--dither-edge-threshold",
        type=float,
        default=0.28,
        help="Adaptive ordered dither maximum edge-mask value, default: 0.28.",
    )
    parser.add_argument(
        "--dither-luma-range",
        type=float,
        default=45.0,
        help="Adaptive ordered dither maximum 3x3 luma range, default: 45.",
    )
    parser.add_argument(
        "--dither-error-threshold",
        type=float,
        default=3.0,
        help="Adaptive ordered dither minimum weighted RGB quantization error, default: 3.",
    )
    parser.add_argument(
        "--resample",
        choices=("box", "bicubic", "lanczos"),
        default="lanczos",
        help="Downsample filter before palette snapping, default: lanczos.",
    )
    parser.add_argument(
        "--grid-snap",
        action="store_true",
        help="Replace resize with hidden-grid snapping for AI pseudo-pixel art.",
    )
    parser.add_argument(
        "--grid-snap-method",
        choices=("cell-mode", "center", "dark-stroke"),
        default="center",
        help="Grid snap cell reducer, default: center.",
    )
    parser.add_argument(
        "--grid-quantize-first",
        action="store_true",
        help="Quantize source colors before grid cell voting. Disabled by default so dither can be tested with --grid-snap.",
    )
    parser.add_argument(
        "--grid-dark-threshold",
        type=float,
        default=38.0,
        help="Minimum luma contrast for dark-stroke grid snap, default: 38.",
    )
    parser.add_argument(
        "--palette-source",
        type=Path,
        default=None,
        help="Optional image used only for palette extraction; output is still mapped from INPUT.",
    )
    parser.add_argument("--saturation", type=float, default=1.0, help="Color multiplier, default: 1.0.")
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast multiplier, default: 1.0.")
    parser.add_argument(
        "--sharpness",
        type=float,
        default=125.0,
        help="Unsharp-mask percent before quantization, default: 125.",
    )
    parser.add_argument(
        "--autocontrast-cutoff",
        type=float,
        default=0.0,
        help="Autocontrast cutoff percentage, default: 0.",
    )
    parser.add_argument("--preview-scale", type=int, default=1, help="Nearest preview scale, default: 1.")
    parser.add_argument(
        "--bilateral-radius",
        type=int,
        default=0,
        help="Apply edge-preserving bilateral smoothing before quantization, default: 0.",
    )
    parser.add_argument(
        "--bilateral-mode",
        choices=("standard", "edge-safe"),
        default="edge-safe",
        help="Bilateral algorithm: standard smoothing or contour-safe guarded smoothing, default: edge-safe.",
    )
    parser.add_argument(
        "--bilateral-sigma-color",
        type=float,
        default=18.0,
        help="Bilateral range sigma; lower values preserve stronger edges, default: 18.",
    )
    parser.add_argument(
        "--bilateral-sigma-space",
        type=float,
        default=1.4,
        help="Bilateral spatial sigma, default: 1.4.",
    )
    parser.add_argument(
        "--edge-palette-weight",
        type=float,
        default=0.0,
        help="Extra palette weight for Sobel edge pixels, default: 0.",
    )
    parser.add_argument(
        "--edge-sharpen",
        type=float,
        default=0.0,
        help="Selective Sobel-edge sharpening strength, default: 0.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.04,
        help="Relative Sobel threshold for edge masks, default: 0.04.",
    )
    parser.add_argument(
        "--palette-strategy",
        choices=(
            "median-cut",
            "interesting",
            "hue-mass",
            "spectrum-peaks",
            "shadow-spectrum",
            "projected-mass",
            "projected-rare",
            "projected-edge",
            "projected-islands",
            "projected-anchors",
            "projected-frontier",
            "projected-graft",
        ),
        default="median-cut",
        help="Palette builder: median-cut, interesting, hue-mass, spectrum-peaks, shadow-spectrum, or projected variants.",
    )
    parser.add_argument(
        "--color-distance",
        choices=("rgb", "oklab"),
        default="rgb",
        help="Nearest-palette color distance, default: rgb.",
    )
    parser.add_argument(
        "--accent-palette-weight",
        type=float,
        default=0.0,
        help="Extra palette weight for saturated accent colors, default: 0.",
    )
    parser.add_argument(
        "--hue-rarity-weight",
        type=float,
        default=0.0,
        help="Extra palette weight for rare saturated hue bins, default: 0.",
    )
    parser.add_argument(
        "--interesting-color-slots",
        type=int,
        default=0,
        help="Reserve slots for saturated rare/accent colors when using --palette-strategy interesting.",
    )
    parser.add_argument(
        "--interesting-min-saturation",
        type=float,
        default=0.12,
        help="Minimum saturation for interesting palette slots, default: 0.12.",
    )
    parser.add_argument(
        "--interesting-min-value",
        type=float,
        default=0.06,
        help="Minimum value/brightness for interesting palette slots, default: 0.06.",
    )
    parser.add_argument(
        "--protected-hue-ranges",
        type=parse_hue_ranges,
        default=(),
        help="Protected hue ranges like 250-330 or 330-20, separated by commas.",
    )
    parser.add_argument(
        "--protected-hue-weight",
        type=float,
        default=0.0,
        help="Extra palette weight for protected hue ranges, default: 0.",
    )
    parser.add_argument(
        "--protected-hue-slots",
        type=int,
        default=0,
        help="Reserve palette slots for protected hue ranges, default: 0.",
    )
    parser.add_argument(
        "--protected-hue-min-saturation",
        type=float,
        default=0.08,
        help="Minimum saturation allowed into protected hue slots, default: 0.08.",
    )
    parser.add_argument(
        "--hue-match-weight",
        type=float,
        default=0.0,
        help="Nearest-palette hue/chroma penalty for saturated colors, default: 0.",
    )
    parser.add_argument(
        "--flat-region-channel-step",
        type=int,
        default=0,
        help="Snap low-detail non-edge regions to RGB channel steps, default: 0 disabled.",
    )
    parser.add_argument(
        "--flat-region-palette-colors",
        type=int,
        default=0,
        help="Map low-detail non-edge regions to a local image-derived palette, default: 0 disabled.",
    )
    parser.add_argument(
        "--flat-region-max-saturation",
        type=float,
        default=0.35,
        help="Maximum saturation eligible for low-detail region flattening, default: 0.35.",
    )
    parser.add_argument(
        "--flat-region-edge-threshold",
        type=float,
        default=0.18,
        help="Maximum Sobel edge-mask value for low-detail snapping, default: 0.18.",
    )
    parser.add_argument(
        "--flat-region-luma-range",
        type=float,
        default=10.0,
        help="Maximum 3x3 luma range for low-detail snapping, default: 10.",
    )
    parser.add_argument(
        "--mixel-cleanup-passes",
        type=int,
        default=0,
        help="Replace isolated near-color pixels by local neighbors, default: 0.",
    )
    parser.add_argument(
        "--mixel-cleanup-min-neighbors",
        type=int,
        default=3,
        help="Minimum matching 3x3 neighbors needed for mixel cleanup, default: 3.",
    )
    parser.add_argument(
        "--mixel-cleanup-distance",
        type=float,
        default=18.0,
        help="Maximum weighted RGB distance for mixel replacement, default: 18.",
    )
    parser.add_argument(
        "--mixel-cleanup-max-saturation",
        type=float,
        default=0.45,
        help="Maximum saturation eligible for mixel replacement, default: 0.45.",
    )
    parser.add_argument(
        "--no-preserve-luma",
        action="store_true",
        help="Disable mean-luma matching against the cropped/resized source.",
    )
    parser.add_argument(
        "--no-preserve-saturation",
        action="store_true",
        help="Disable luma-weighted saturation matching against the cropped/resized source.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="Preview PNG path. Defaults to OUTPUT_STEM@SCALE.png when --preview-scale is above 1.",
    )
    parser.add_argument("--no-preview", action="store_true", help="Do not write a scaled preview.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.colors < 2 or args.colors > 1024:
        parser.error("--colors must be between 2 and 1024")
    dither_disabled_by_grid_vote = args.grid_snap and args.grid_quantize_first and args.dither != "none"
    if args.dither != "none" and args.colors > 256 and not dither_disabled_by_grid_vote:
        parser.error("--dither modes support at most 256 colors")
    if args.dither_strength < 0:
        parser.error("--dither-strength must be at least 0")
    if args.dither_edge_threshold < 0 or args.dither_edge_threshold > 1:
        parser.error("--dither-edge-threshold must be between 0 and 1")
    if args.dither_luma_range < 0:
        parser.error("--dither-luma-range must be at least 0")
    if args.dither_error_threshold < 0:
        parser.error("--dither-error-threshold must be at least 0")
    if args.preview_scale < 1:
        parser.error("--preview-scale must be at least 1")
    if args.grid_dark_threshold < 0:
        parser.error("--grid-dark-threshold must be at least 0")
    if args.palette_source is not None and not args.palette_source.exists():
        parser.error("--palette-source does not exist")
    if args.bilateral_radius < 0:
        parser.error("--bilateral-radius must be at least 0")
    if args.protected_hue_slots < 0:
        parser.error("--protected-hue-slots must be at least 0")
    if args.protected_hue_min_saturation < 0 or args.protected_hue_min_saturation > 1:
        parser.error("--protected-hue-min-saturation must be between 0 and 1")
    if args.hue_match_weight < 0:
        parser.error("--hue-match-weight must be at least 0")
    if args.interesting_color_slots < 0:
        parser.error("--interesting-color-slots must be at least 0")
    if args.interesting_min_saturation < 0 or args.interesting_min_saturation > 1:
        parser.error("--interesting-min-saturation must be between 0 and 1")
    if args.interesting_min_value < 0 or args.interesting_min_value > 1:
        parser.error("--interesting-min-value must be between 0 and 1")
    if args.flat_region_channel_step < 0:
        parser.error("--flat-region-channel-step must be at least 0")
    if args.flat_region_palette_colors < 0:
        parser.error("--flat-region-palette-colors must be at least 0")
    if args.flat_region_max_saturation < 0 or args.flat_region_max_saturation > 1:
        parser.error("--flat-region-max-saturation must be between 0 and 1")
    if args.flat_region_edge_threshold < 0 or args.flat_region_edge_threshold > 1:
        parser.error("--flat-region-edge-threshold must be between 0 and 1")
    if args.flat_region_luma_range < 0:
        parser.error("--flat-region-luma-range must be at least 0")
    if args.mixel_cleanup_passes < 0:
        parser.error("--mixel-cleanup-passes must be at least 0")
    if args.mixel_cleanup_min_neighbors < 2 or args.mixel_cleanup_min_neighbors > 9:
        parser.error("--mixel-cleanup-min-neighbors must be between 2 and 9")
    if args.mixel_cleanup_distance < 0:
        parser.error("--mixel-cleanup-distance must be at least 0")
    if args.mixel_cleanup_max_saturation < 0 or args.mixel_cleanup_max_saturation > 1:
        parser.error("--mixel-cleanup-max-saturation must be between 0 and 1")

    target_width, target_height = args.size
    config = PixelArtConfig(
        target_width=target_width,
        target_height=target_height,
        colors=args.colors,
        preview_scale=args.preview_scale,
        dither=args.dither,
        dither_strength=args.dither_strength,
        dither_scope=args.dither_scope,
        dither_edge_threshold=args.dither_edge_threshold,
        dither_luma_range=args.dither_luma_range,
        dither_error_threshold=args.dither_error_threshold,
        saturation=args.saturation,
        contrast=args.contrast,
        sharpness=args.sharpness,
        autocontrast_cutoff=args.autocontrast_cutoff,
        resample=args.resample,
        grid_snap_enabled=args.grid_snap,
        grid_snap_method=args.grid_snap_method,
        grid_snap_quantize_first=args.grid_quantize_first,
        grid_snap_dark_threshold=args.grid_dark_threshold,
        preserve_luma=not args.no_preserve_luma,
        preserve_saturation=not args.no_preserve_saturation,
        palette_source=args.palette_source,
        bilateral_radius=args.bilateral_radius,
        bilateral_mode=args.bilateral_mode,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
        edge_palette_weight=args.edge_palette_weight,
        edge_sharpen=args.edge_sharpen,
        edge_threshold=args.edge_threshold,
        palette_strategy=args.palette_strategy,
        color_distance=args.color_distance,
        accent_palette_weight=args.accent_palette_weight,
        hue_rarity_weight=args.hue_rarity_weight,
        interesting_color_slots=args.interesting_color_slots,
        interesting_min_saturation=args.interesting_min_saturation,
        interesting_min_value=args.interesting_min_value,
        protected_hue_ranges=args.protected_hue_ranges,
        protected_hue_weight=args.protected_hue_weight,
        protected_hue_slots=args.protected_hue_slots,
        protected_hue_min_saturation=args.protected_hue_min_saturation,
        hue_match_weight=args.hue_match_weight,
        flat_region_palette_colors=args.flat_region_palette_colors,
        flat_region_channel_step=args.flat_region_channel_step,
        flat_region_max_saturation=args.flat_region_max_saturation,
        flat_region_edge_threshold=args.flat_region_edge_threshold,
        flat_region_luma_range=args.flat_region_luma_range,
        mixel_cleanup_passes=args.mixel_cleanup_passes,
        mixel_cleanup_min_neighbors=args.mixel_cleanup_min_neighbors,
        mixel_cleanup_distance=args.mixel_cleanup_distance,
        mixel_cleanup_max_saturation=args.mixel_cleanup_max_saturation,
    )

    preview_path = None
    if not args.no_preview:
        if args.preview:
            preview_path = args.preview
        elif args.preview_scale > 1:
            preview_path = build_default_preview_path(args.output, args.preview_scale)

    manifest = convert(args.input, args.output, preview_path, config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
