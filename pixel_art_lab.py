#!/usr/bin/env python3
"""Local browser lab for the pixel-art palette conversion pipeline."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import socket
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from PIL import Image, ImageChops, ImageFilter, ImageOps
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing dependency hint.
    raise SystemExit(
        "Missing dependency: Pillow. Install with: python3 -m pip install -r requirements.txt"
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pixel_art_grid as pag


MAX_UPLOAD_BYTES = 64 * 1024 * 1024
DEFAULT_PORT = 8767
MIN_OUTPUT_SIZE = 16
MAX_OUTPUT_HEIGHT = 1024
MAX_OUTPUT_WIDTH = 4096
MAX_STAGE_CACHE_ENTRIES = 64
MAX_RENDER_CACHE_ENTRIES = 8
PALETTE_STRATEGIES = (
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
)


@dataclass
class LabState:
    image: Image.Image | None = None
    name: str = ""
    uploaded_at: float = 0.0
    version: int = 0
    cache: dict[str, OrderedDict[tuple[Any, ...], Any]] = field(default_factory=dict)


STATE = LabState()
STATE_LOCK = threading.Lock()
RENDER_LOCK = threading.Lock()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def as_int(settings: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(settings.get(key, default))
    except (TypeError, ValueError):
        value = default
    return int(clamp(value, minimum, maximum))


def as_float(settings: dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(settings.get(key, default))
    except (TypeError, ValueError):
        value = default
    return clamp(value, minimum, maximum)


def as_bool(settings: dict[str, Any], key: str, default: bool = False) -> bool:
    value = settings.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def as_choice(settings: dict[str, Any], key: str, default: str, choices: tuple[str, ...]) -> str:
    value = str(settings.get(key, default))
    return value if value in choices else default


def data_url_to_image(data_url: str) -> Image.Image:
    if "," in data_url:
        _header, data_url = data_url.split(",", 1)
    raw = base64.b64decode(data_url, validate=False)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("image upload is too large")
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image.convert("RGB")


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def normalize_edge_mask(mask: Image.Image, threshold: float) -> Image.Image:
    gray = mask.convert("L")
    _low, high = gray.getextrema()
    if high <= 0:
        return Image.new("L", gray.size, 0)

    threshold_value = high * threshold
    source = gray.load()
    out = Image.new("L", gray.size)
    target = out.load()
    for y in range(gray.height):
        for x in range(gray.width):
            normalized = max(0.0, (source[x, y] - threshold_value) / max(1.0, high - threshold_value))
            target[x, y] = pag.clamp_channel(normalized * 255)
    return out.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(radius=0.35))


def build_lab_edge_mask(image: Image.Image, mode: str, threshold: float) -> Image.Image | None:
    if mode == "none":
        return None
    if mode == "sobel":
        return pag.build_sobel_edge_mask(image, threshold=threshold)

    gray = ImageOps.grayscale(image)
    if mode == "laplacian":
        mask = gray.filter(
            ImageFilter.Kernel(
                (3, 3),
                (-1, -1, -1, -1, 8, -1, -1, -1, -1),
                scale=1,
                offset=0,
            )
        )
    elif mode == "highpass":
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2.0))
        mask = ImageChops.difference(gray, blurred)
    elif mode == "contour":
        mask = gray.filter(ImageFilter.CONTOUR)
    else:
        mask = pag.build_sobel_edge_mask(image, threshold=threshold)
    return normalize_edge_mask(mask, threshold)


def parse_ranges(value: str) -> tuple[tuple[float, float], ...]:
    if not value.strip():
        return ()
    return pag.parse_hue_ranges(value)


def source_aspect_dimensions(
    source_width: int,
    source_height: int,
    driver: str = "height",
    requested_width: int | None = None,
    requested_height: int | None = None,
) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        return 1024, 576

    aspect = source_width / source_height
    default_height = min(source_height, MAX_OUTPUT_HEIGHT)
    default_width = round(default_height * aspect)

    if driver == "width":
        width = requested_width if requested_width is not None else default_width
        width = int(clamp(width, MIN_OUTPUT_SIZE, MAX_OUTPUT_WIDTH))
        height = max(MIN_OUTPUT_SIZE, round(width / aspect))
    else:
        height = requested_height if requested_height is not None else default_height
        height = int(clamp(height, MIN_OUTPUT_SIZE, MAX_OUTPUT_HEIGHT))
        width = max(MIN_OUTPUT_SIZE, round(height * aspect))

    if height > MAX_OUTPUT_HEIGHT:
        height = MAX_OUTPUT_HEIGHT
        width = max(MIN_OUTPUT_SIZE, round(height * aspect))
    if width > MAX_OUTPUT_WIDTH:
        width = MAX_OUTPUT_WIDTH
        height = max(MIN_OUTPUT_SIZE, round(width / aspect))
    if height > MAX_OUTPUT_HEIGHT:
        height = MAX_OUTPUT_HEIGHT
        width = max(MIN_OUTPUT_SIZE, round(height * aspect))

    return int(width), int(height)


def config_from_settings(
    settings: dict[str, Any],
    source_size: tuple[int, int] | None = None,
) -> pag.PixelArtConfig:
    default_width = 1024
    default_height = 576
    if source_size is not None:
        default_width, default_height = source_aspect_dimensions(source_size[0], source_size[1])

    width = as_int(settings, "targetWidth", default_width, MIN_OUTPUT_SIZE, MAX_OUTPUT_WIDTH)
    height = as_int(settings, "targetHeight", default_height, MIN_OUTPUT_SIZE, MAX_OUTPUT_HEIGHT)
    if source_size is not None and as_bool(settings, "aspectLock", True):
        driver = as_choice(settings, "aspectDriver", "height", ("width", "height"))
        width, height = source_aspect_dimensions(
            source_size[0],
            source_size[1],
            driver=driver,
            requested_width=width,
            requested_height=height,
        )

    colors = as_int(settings, "colors", 64, 2, 1024)
    protected_ranges = parse_ranges(str(settings.get("protectedHueRanges", "")).strip())
    return pag.PixelArtConfig(
        target_width=width,
        target_height=height,
        colors=colors,
        preview_scale=1,
        dither=as_choice(settings, "dither", "none", ("ordered", "floyd", "none")),
        dither_strength=as_float(settings, "ditherStrength", 14.0, 0.0, 128.0),
        dither_scope=as_choice(settings, "ditherScope", "adaptive", ("global", "adaptive")),
        dither_edge_threshold=as_float(settings, "ditherEdgeThreshold", 0.28, 0.0, 1.0),
        dither_luma_range=as_float(settings, "ditherLumaRange", 45.0, 0.0, 255.0),
        dither_error_threshold=as_float(settings, "ditherErrorThreshold", 3.0, 0.0, 255.0),
        saturation=as_float(settings, "saturation", 1.0, 0.0, 8.0),
        contrast=as_float(settings, "contrast", 1.0, 0.0, 8.0),
        sharpness=as_float(settings, "sharpness", 0.0, 0.0, 500.0),
        autocontrast_cutoff=as_float(settings, "autocontrastCutoff", 0.0, 0.0, 30.0),
        resample=as_choice(settings, "resample", "box", ("box", "bicubic", "lanczos")),
        grid_snap_enabled=as_bool(settings, "gridSnap", False),
        grid_snap_method=as_choice(settings, "gridSnapMethod", "dark-stroke", ("cell-mode", "center", "dark-stroke")),
        grid_snap_quantize_first=as_bool(settings, "gridQuantizeFirst", True),
        grid_snap_dark_threshold=as_float(settings, "gridDarkThreshold", 38.0, 0.0, 255.0),
        preserve_luma=as_bool(settings, "preserveLuma", False),
        preserve_saturation=as_bool(settings, "preserveSaturation", False),
        palette_source=None,
        bilateral_radius=as_int(settings, "bilateralRadius", 0, 0, 8),
        bilateral_sigma_color=as_float(settings, "bilateralSigmaColor", 18.0, 1.0, 128.0),
        bilateral_sigma_space=as_float(settings, "bilateralSigmaSpace", 1.4, 0.1, 16.0),
        edge_palette_weight=as_float(settings, "edgePaletteWeight", 0.45, 0.0, 12.0),
        edge_sharpen=as_float(settings, "edgeSharpen", 0.0, 0.0, 8.0),
        edge_threshold=as_float(settings, "edgeThreshold", 0.04, 0.0, 1.0),
        palette_strategy=as_choice(settings, "paletteStrategy", "projected-rare", PALETTE_STRATEGIES),
        color_distance=as_choice(settings, "colorDistance", "oklab", ("rgb", "oklab")),
        accent_palette_weight=as_float(settings, "accentPaletteWeight", 0.8, 0.0, 12.0),
        hue_rarity_weight=as_float(settings, "hueRarityWeight", 1.6, 0.0, 12.0),
        interesting_color_slots=as_int(settings, "interestingColorSlots", 0, 0, 1024),
        interesting_min_saturation=as_float(settings, "interestingMinSaturation", 0.07, 0.0, 1.0),
        interesting_min_value=as_float(settings, "interestingMinValue", 0.05, 0.0, 1.0),
        protected_hue_ranges=protected_ranges,
        protected_hue_weight=as_float(settings, "protectedHueWeight", 0.0, 0.0, 20.0),
        protected_hue_slots=as_int(settings, "protectedHueSlots", 0, 0, 1024),
        protected_hue_min_saturation=as_float(settings, "protectedHueMinSaturation", 0.08, 0.0, 1.0),
        hue_match_weight=as_float(settings, "hueMatchWeight", 0.35, 0.0, 12.0),
        flat_region_palette_colors=as_int(settings, "flatRegionPaletteColors", 0, 0, 1024),
        flat_region_channel_step=as_int(settings, "flatRegionChannelStep", 0, 0, 255),
        flat_region_max_saturation=as_float(settings, "flatRegionMaxSaturation", 0.35, 0.0, 1.0),
        flat_region_edge_threshold=as_float(settings, "flatRegionEdgeThreshold", 0.18, 0.0, 1.0),
        flat_region_luma_range=as_float(settings, "flatRegionLumaRange", 10.0, 0.0, 255.0),
        mixel_cleanup_passes=as_int(settings, "mixelCleanupPasses", 0, 0, 8),
        mixel_cleanup_min_neighbors=as_int(settings, "mixelCleanupMinNeighbors", 3, 1, 9),
        mixel_cleanup_distance=as_float(settings, "mixelCleanupDistance", 18.0, 0.0, 255.0),
        mixel_cleanup_max_saturation=as_float(settings, "mixelCleanupMaxSaturation", 0.45, 0.0, 1.0),
    )


def clone_render_result(result: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(result)
    if isinstance(result.get("stats"), dict):
        cloned["stats"] = dict(result["stats"])
    if isinstance(result.get("palette"), list):
        cloned["palette"] = list(result["palette"])
    return cloned


def output_cache_config(config: pag.PixelArtConfig) -> pag.PixelArtConfig:
    if config.dither == "ordered":
        return config
    return replace(
        config,
        dither_strength=0.0,
        dither_scope="global",
        dither_edge_threshold=0.0,
        dither_luma_range=0.0,
        dither_error_threshold=0.0,
    )


def cache_lookup(
    cache: dict[str, OrderedDict[tuple[Any, ...], Any]] | None,
    namespace: str,
    key: tuple[Any, ...],
    factory,
    limit: int = MAX_STAGE_CACHE_ENTRIES,
) -> tuple[Any, bool]:
    if cache is None:
        return factory(), False

    bucket = cache.setdefault(namespace, OrderedDict())
    if key in bucket:
        bucket.move_to_end(key)
        return bucket[key], True

    value = factory()
    bucket[key] = value
    while len(bucket) > limit:
        bucket.popitem(last=False)
    return value, False


def convert_in_memory(
    image: Image.Image,
    settings: dict[str, Any],
    cache: dict[str, OrderedDict[tuple[Any, ...], Any]] | None = None,
    version: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    include_edge_preview = as_bool(
        {"includeEdgePreview": settings.get("includeEdgePreview", False)},
        "includeEdgePreview",
        False,
    )

    stage_cache_hits = 0

    def cached(namespace: str, key: tuple[Any, ...], factory):
        nonlocal stage_cache_hits
        value, hit = cache_lookup(cache, namespace, key, factory)
        if hit:
            stage_cache_hits += 1
        return value

    grid_variants: list[dict[str, Any]] = []
    selected_grid_variant: dict[str, Any] | None = None
    settings_for_config = dict(settings)
    grid_snap_requested = as_bool(settings, "gridSnap", False)
    grid_auto_size = as_bool(settings, "gridAutoSize", True)
    if grid_snap_requested and grid_auto_size:
        grid_variants = cached(
            "stage",
            ("grid-detect", version, image.size[0], image.size[1], MAX_OUTPUT_WIDTH, MAX_OUTPUT_HEIGHT),
            lambda: pag.detect_mixel_grid_variants(
                image,
                max_output_width=MAX_OUTPUT_WIDTH,
                max_output_height=MAX_OUTPUT_HEIGHT,
                min_output_size=MIN_OUTPUT_SIZE,
                max_variants=9,
            ),
        )
        if grid_variants:
            variant_index = as_int(settings, "gridVariant", 0, 0, len(grid_variants) - 1)
            selected_grid_variant = grid_variants[variant_index]
            settings_for_config["targetWidth"] = int(selected_grid_variant["width"])
            settings_for_config["targetHeight"] = int(selected_grid_variant["height"])
            settings_for_config["aspectDriver"] = "height"

    config = config_from_settings(settings_for_config, source_size=image.size)
    if config.dither != "none" and config.colors > 256:
        raise ValueError("dither modes support at most 256 colors")

    base_key = (
        "base",
        version,
        config.target_width,
        config.target_height,
        config.resample,
        config.colors if config.grid_snap_quantize_first else 0,
        config.grid_snap_enabled,
        config.grid_snap_method,
        config.grid_snap_quantize_first,
        config.grid_snap_dark_threshold,
    )
    base = cached("stage", base_key, lambda: pag.prepare_base_image(image, config))
    edge_mode = as_choice(settings, "edgeMode", "sobel", ("sobel", "laplacian", "highpass", "contour", "none"))
    edge_key = ("edge", base_key, edge_mode, config.edge_threshold)
    edge_mask = cached("stage", edge_key, lambda: build_lab_edge_mask(base, edge_mode, config.edge_threshold))
    processed_key = (
        "processed",
        base_key,
        config.bilateral_radius,
        config.bilateral_sigma_color,
        config.bilateral_sigma_space,
    )
    processed = cached(
        "stage",
        processed_key,
        lambda: pag.bilateral_smooth(
            base,
            radius=config.bilateral_radius,
            sigma_color=config.bilateral_sigma_color,
            sigma_space=config.bilateral_sigma_space,
        ),
    )

    prepared_key = (
        "prepared",
        processed_key,
        edge_key,
        config.saturation,
        config.contrast,
        config.sharpness,
        config.autocontrast_cutoff,
        config.edge_sharpen,
    )

    def build_prepared() -> Image.Image:
        prepared_image = pag.grade_image(processed, config)
        if edge_mask is not None:
            prepared_image = pag.selective_edge_sharpen(prepared_image, edge_mask, config.edge_sharpen)
        return prepared_image

    prepared = cached("stage", prepared_key, build_prepared)

    def attach_previews(result: dict[str, Any]) -> dict[str, Any]:
        result["source"] = None
        result["edge"] = None
        if edge_mask is not None and include_edge_preview:
            result["edge"] = cached(
                "data-url",
                ("edge-data-url", edge_key),
                lambda: image_to_data_url(edge_mask.convert("RGB")),
            )
        return result

    palette_mode = as_choice(settings, "paletteInput", "prepared", ("prepared", "original", "graded"))
    palette_image: Image.Image | None = None
    palette_edge_mask: Image.Image | None = None
    if palette_mode == "original":
        palette_image = base
        palette_edge_mask = edge_mask
    elif palette_mode == "graded":
        palette_image = prepared
        palette_edge_mask = edge_mask

    render_key = (
        "render-core",
        version,
        edge_mode,
        palette_mode,
        grid_auto_size,
        (
            int(selected_grid_variant["width"]),
            int(selected_grid_variant["height"]),
        )
        if selected_grid_variant
        else None,
        output_cache_config(config),
    )

    if cache is not None:
        render_bucket = cache.setdefault("render", OrderedDict())
        cached_render = render_bucket.get(render_key)
        if cached_render is not None:
            render_bucket.move_to_end(render_key)
            result = attach_previews(clone_render_result(cached_render))
            stats = result.setdefault("stats", {})
            stats["elapsedMs"] = round((time.perf_counter() - started) * 1000)
            stats["cacheHit"] = True
            stats["stageCacheHits"] = stage_cache_hits
            return result

    pixel_art, palette = pag.quantize_to_palette(
        prepared,
        config,
        edge_mask=edge_mask,
        palette_image=palette_image,
        palette_edge_mask=palette_edge_mask,
    )

    source_luma = pag.luma_mean(base)
    source_saturation = pag.luma_weighted_saturation_mean(base)
    quantized_luma = pag.luma_mean(pixel_art)
    quantized_saturation = pag.luma_weighted_saturation_mean(pixel_art)

    if config.preserve_luma:
        pixel_art = pag.match_luma_mean(pixel_art, source_luma)
    if config.preserve_saturation:
        pixel_art = pag.match_luma_weighted_saturation(pixel_art, source_saturation)
    if config.preserve_luma:
        pixel_art = pag.match_luma_mean(pixel_art, source_luma)

    if edge_mask is not None:
        pixel_art = pag.snap_low_detail_regions(pixel_art, edge_mask, config)
    pixel_art = pag.cleanup_single_pixel_mixels(pixel_art, config)
    pixel_art, palette = pag.clamp_to_color_limit(pixel_art, config.colors, config)

    output_luma = pag.luma_mean(pixel_art)
    output_saturation = pag.luma_weighted_saturation_mean(pixel_art)
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    result = {
        "output": image_to_data_url(pixel_art),
        "source": None,
        "edge": None,
        "width": pixel_art.width,
        "height": pixel_art.height,
        "palette": palette,
        "stats": {
            "elapsedMs": elapsed_ms,
            "colorsRequested": config.colors,
            "colorsWritten": len(palette),
            "sourceLuma": round(source_luma, 3),
            "quantizedLumaBeforeMatch": round(quantized_luma, 3),
            "outputLuma": round(output_luma, 3),
            "sourceSaturation": round(source_saturation, 4),
            "quantizedSaturationBeforeMatch": round(quantized_saturation, 4),
            "outputSaturation": round(output_saturation, 4),
            "edgeMode": edge_mode,
            "paletteInput": palette_mode,
            "ditherScope": config.dither_scope if config.dither == "ordered" else None,
            "gridSnap": config.grid_snap_enabled,
            "gridSnapMethod": config.grid_snap_method if config.grid_snap_enabled else None,
            "gridQuantizeFirst": config.grid_snap_quantize_first if config.grid_snap_enabled else None,
            "gridAutoSize": grid_auto_size if config.grid_snap_enabled else False,
            "gridVariant": selected_grid_variant,
            "gridVariants": grid_variants[:9],
            "cacheHit": False,
            "stageCacheHits": stage_cache_hits,
        },
    }
    if cache is not None:
        render_bucket = cache.setdefault("render", OrderedDict())
        render_bucket[render_key] = clone_render_result(result)
        while len(render_bucket) > MAX_RENDER_CACHE_ENTRIES:
            render_bucket.popitem(last=False)
    result = attach_previews(result)
    result["stats"]["stageCacheHits"] = stage_cache_hits
    return result


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pixel Art Lab</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0e14;
      --panel: #151a24;
      --panel-strong: #1b2230;
      --panel-soft: #10151f;
      --field: #0d121b;
      --line: #30394c;
      --line-strong: #47536c;
      --text: #edf2ff;
      --muted: #a5b0c5;
      --faint: #727d92;
      --accent: #9fb8ff;
      --accent-strong: #c4d2ff;
      --ok: #9ee6ac;
      --busy: #f5d487;
      --warn: #f0b15b;
      --error: #ff9d9d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }
    .app {
      display: grid;
      grid-template-columns: minmax(380px, 430px) 1fr;
      height: 100vh;
    }
    aside {
      overflow: auto;
      padding: 16px;
      background: var(--panel);
      border-right: 1px solid var(--line);
      scrollbar-color: var(--line-strong) transparent;
    }
    main {
      display: grid;
      grid-template-rows: 56px 1fr;
      min-width: 0;
      min-height: 0;
    }
    .topbar {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
    }
    .topbar strong {
      color: var(--text);
      font-weight: 700;
    }
    .topbar .spacer {
      flex: 1;
      min-width: 16px;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      min-width: 118px;
      min-height: 28px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #0f1520;
      font-weight: 700;
    }
    .viewer {
      position: relative;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      background:
        linear-gradient(45deg, #0b0d13 25%, transparent 25%),
        linear-gradient(-45deg, #0b0d13 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #0b0d13 75%),
        linear-gradient(-45deg, transparent 75%, #0b0d13 75%);
      background-color: #11141c;
      background-size: 32px 32px;
      background-position: 0 0, 0 16px, 16px -16px, -16px 0;
      cursor: grab;
    }
    .viewer.dragging { cursor: grabbing; }
    #canvas {
      width: 100%;
      height: 100%;
      display: block;
      image-rendering: pixelated;
    }
    #tooltip {
      position: fixed;
      z-index: 20;
      width: 260px;
      height: 284px;
      padding: 4px;
      display: none;
      pointer-events: none;
      background: #090b10;
      border: 1px solid #6d7590;
    }
    #tooltip canvas {
      width: 250px;
      height: 250px;
      image-rendering: pixelated;
      display: block;
    }
    #tooltip .label {
      display: flex;
      justify-content: space-between;
      color: #dce4f6;
      font-size: 11px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      margin-top: 3px;
      padding: 0 2px;
    }
    .brand {
      padding: 2px 2px 14px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 14px;
    }
    .eyebrow {
      margin: 0 0 5px;
      color: var(--accent);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    h1 {
      font-size: 24px;
      margin: 0 0 6px;
      letter-spacing: 0;
    }
    .intro {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    fieldset {
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 0 0 14px;
      padding: 12px;
      background: var(--panel-soft);
    }
    legend {
      color: var(--accent);
      font-weight: 700;
      font-size: 12px;
      padding: 0 6px;
      text-transform: uppercase;
      letter-spacing: .05em;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 9px;
    }
    .label-title {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 8px;
    }
    .field-hint {
      color: var(--faint);
      font-size: 11px;
      line-height: 1.35;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 7px 0;
      color: var(--muted);
      font-size: 12px;
    }
    label.inline {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 7px 0;
    }
    input, select, button {
      width: 100%;
      border: 1px solid #40485d;
      border-radius: 5px;
      background: var(--field);
      color: var(--text);
      min-height: 34px;
      padding: 6px 9px;
      font: inherit;
      font-size: 13px;
    }
    input[type="file"] {
      padding: 7px;
      min-height: 40px;
    }
    input[type="file"]::file-selector-button {
      border: 0;
      border-radius: 5px;
      margin-right: 10px;
      padding: 6px 10px;
      background: #2a3752;
      color: var(--text);
      font-weight: 700;
      cursor: pointer;
    }
    input[type="checkbox"] {
      width: 16px;
      min-height: 16px;
      padding: 0;
    }
    input[type="range"] {
      padding: 0;
    }
    input:disabled,
    select:disabled {
      opacity: .52;
      cursor: not-allowed;
      border-color: #30384a;
      color: var(--faint);
    }
    label.disabled {
      opacity: .68;
    }
    button {
      background: #26314a;
      cursor: pointer;
      font-weight: 700;
    }
    button:hover { background: #303c59; }
    button.secondary {
      background: #202838;
      color: var(--accent-strong);
    }
    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .row3 {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
    }
    .section-note {
      margin: 2px 0 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .presets {
      display: grid;
      gap: 8px;
      margin-bottom: 8px;
    }
    .preset-actions {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
    }
    .small {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .metric {
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    .warn { color: var(--warn); }
    .palette {
      display: grid;
      grid-template-columns: repeat(16, 1fr);
      gap: 2px;
      margin-top: 10px;
    }
    .swatch {
      height: 14px;
      border: 1px solid #0008;
    }
    .status-ok { color: var(--ok); }
    .status-busy { color: var(--busy); }
    .status-error { color: var(--error); }
    .hidden { display: none; }
    @media (max-width: 980px) {
      body { overflow: auto; }
      .app {
        grid-template-columns: 1fr;
        grid-template-rows: auto minmax(520px, 70vh);
        height: auto;
        min-height: 100vh;
      }
      aside {
        max-height: none;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      main {
        min-height: 520px;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        <p class="eyebrow">Local operator tool</p>
        <h1>Pixel Art Lab</h1>
        <p class="intro">Convert AI or faux pixel art into a strict grid while preserving rare colors, outlines, and readable one-pixel details.</p>
      </div>
      <fieldset>
        <legend>Start</legend>
        <p class="section-note">Import one image. Every control updates the rendered output after a short debounce.</p>
        <label>
          <span class="label-title">Source image <span class="field-hint">PNG, JPG, or WebP</span></span>
          <input id="file" type="file" accept="image/*" title="Load the source image for this local session.">
        </label>
        <div id="inputInfo" class="small">No image loaded yet.</div>
        <div class="button-row">
          <button id="resetZoom" class="secondary" title="Return the preview to 100% zoom and recenter it.">Reset view</button>
          <button id="saveSettings" class="secondary" title="Export all current controls as a JSON preset file.">Save settings</button>
        </div>
      </fieldset>

      <fieldset>
        <legend>My Presets</legend>
        <p class="section-note">Browser-local presets for your own tested combinations. Built-in experiment buttons are intentionally not included.</p>
        <div class="presets">
          <label>
            <span class="label-title">Preset <span class="field-hint">stored in this browser</span></span>
            <select id="customPresetSelect" title="Choose a preset saved in this browser."></select>
          </label>
          <label>
            <span class="label-title">Name <span class="field-hint">required to save</span></span>
            <input id="presetName" type="text" placeholder="preset name" title="Name for saving or replacing a custom preset.">
          </label>
          <div class="preset-actions">
            <button id="loadPreset" title="Apply the selected preset to all controls.">Load</button>
            <button id="savePreset" title="Save current controls under the typed name.">Save</button>
            <button id="deletePreset" title="Delete the selected custom preset from this browser.">Delete</button>
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>Output & Dither</legend>
        <p class="section-note">Set the real pixel canvas. With the aspect lock on, editing one side updates the other after commit.</p>
        <input id="aspectDriver" data-setting type="hidden" value="height">
        <div class="row">
          <label>
            <span class="label-title">Width <span class="field-hint">output pixels</span></span>
            <input id="targetWidth" data-setting type="number" min="16" max="4096" value="1024" title="Logical output width. Press Enter or leave the field to render.">
          </label>
          <label>
            <span class="label-title">Height <span class="field-hint">max 1024</span></span>
            <input id="targetHeight" data-setting type="number" min="16" max="1024" value="576" title="Logical output height. Source-ratio default is capped at 1024.">
          </label>
        </div>
        <label class="check" title="Keep output proportions equal to the imported source image.">
          <input id="aspectLock" data-setting type="checkbox" checked> lock source aspect
        </label>
        <div id="aspectInfo" class="small">Output size will use the source image ratio after import.</div>
        <div class="row">
          <label>
            <span class="label-title">Colors <span class="field-hint">palette cap</span></span>
            <input id="colors" data-setting type="number" min="2" max="1024" value="64" title="Maximum colors in the output palette.">
          </label>
          <label>
            <span class="label-title">Dither <span class="field-hint">use sparingly</span></span>
            <select id="dither" data-setting title="Adds controlled texture for gradients after palette mapping.">
              <option value="none">none</option>
              <option value="ordered">ordered Bayer</option>
              <option value="floyd">Floyd-Steinberg</option>
            </select>
          </label>
        </div>
        <label>
          <span class="label-title">Dither strength <span class="field-hint">0-64</span></span>
          <input id="ditherStrength" data-setting type="range" min="0" max="64" step="1" value="14" title="Higher values make dithering more visible.">
        </label>
        <div class="row">
          <label>
            <span class="label-title">Dither scope <span class="field-hint">where it applies</span></span>
            <select id="ditherScope" data-setting title="Adaptive mode suppresses dithering on edges and detailed objects.">
              <option value="global">global</option>
              <option value="adaptive" selected>adaptive smooth areas</option>
            </select>
          </label>
          <label>
            <span class="label-title">Error min <span class="field-hint">skip tiny errors</span></span>
            <input id="ditherErrorThreshold" data-setting type="number" min="0" max="255" step="0.5" value="3" title="Minimum nearest-palette error before adaptive dithering is allowed.">
          </label>
        </div>
        <div class="row">
          <label>
            <span class="label-title">Edge max <span class="field-hint">protect details</span></span>
            <input id="ditherEdgeThreshold" data-setting type="number" min="0" max="1" step="0.01" value="0.28" title="Adaptive dithering is blocked above this edge strength.">
          </label>
          <label>
            <span class="label-title">Luma range <span class="field-hint">smoothness gate</span></span>
            <input id="ditherLumaRange" data-setting type="number" min="0" max="255" step="1" value="45" title="Adaptive dithering is allowed only in areas with local brightness variation under this value.">
          </label>
        </div>
        <div class="row">
          <label>
            <span class="label-title">Resample <span class="field-hint">resize filter</span></span>
            <select id="resample" data-setting title="Resize filter used before palette conversion when Grid Snap is off.">
              <option value="box">box</option>
              <option value="bicubic">bicubic</option>
              <option value="lanczos">lanczos</option>
            </select>
          </label>
          <label>
            <span class="label-title">Color distance <span class="field-hint">palette mapping</span></span>
            <select id="colorDistance" data-setting title="Distance metric for choosing the nearest palette color.">
              <option value="oklab">OKLab</option>
              <option value="rgb">weighted RGB</option>
            </select>
          </label>
        </div>
      </fieldset>

      <fieldset>
        <legend>Hidden Grid</legend>
        <p class="section-note">Use when the source is enlarged AI pixel art. The detector estimates the source's pseudo-pixel cells and transfers them without averaging.</p>
        <label class="check" title="Replace normal resizing with hidden-grid transfer for generated mixel art.">
          <input id="gridSnap" data-setting type="checkbox"> Grid Snap
        </label>
        <label class="check" title="Estimate likely logical output sizes from source edge rhythms. Manual size still works when this is off.">
          <input id="gridAutoSize" data-setting type="checkbox" checked> auto size from detected mixels
        </label>
        <label class="check" title="Reduce source colors before cell voting so rare intended colors are not averaged away.">
          <input id="gridQuantizeFirst" data-setting type="checkbox" checked> quantize before grid vote
        </label>
        <div class="row">
          <label>
            <span class="label-title">Cell reducer <span class="field-hint">pixel pick rule</span></span>
            <select id="gridSnapMethod" data-setting title="How each detected source cell becomes one output pixel.">
              <option value="dark-stroke" selected>dark-stroke bias</option>
              <option value="cell-mode">cell mode</option>
              <option value="center">nearest center sample</option>
            </select>
          </label>
          <label id="gridDarkThresholdWrap">
            <span class="label-title">Dark threshold <span class="field-hint">stroke bias</span></span>
            <input id="gridDarkThreshold" data-setting type="number" min="0" max="255" step="1" value="38" title="Minimum contrast for preserving a narrow dark stroke inside a detected cell.">
          </label>
        </div>
        <label>
          <span class="label-title">Auto variant <span class="field-hint">candidate grid</span></span>
          <input id="gridVariant" data-setting type="range" min="0" max="8" step="1" value="0" title="Choose between detector candidates after auto-size render.">
        </label>
        <div id="gridInfo" class="small">Auto grid size is optional; manual Width/Height still works when it is off.</div>
      </fieldset>

      <fieldset>
        <legend>Palette Builder</legend>
        <p class="section-note">Choose how source colors are allocated before the image is mapped into the final palette.</p>
        <label>
          <span class="label-title">Strategy <span class="field-hint">slot allocation</span></span>
          <select id="paletteStrategy" data-setting title="Palette extraction strategy. Projected modes choose source colors and remap the target into them.">
            <option value="median-cut">median-cut</option>
            <option value="interesting">interesting</option>
            <option value="hue-mass">hue-mass</option>
            <option value="spectrum-peaks">spectrum-peaks</option>
            <option value="shadow-spectrum">shadow-spectrum</option>
            <option value="projected-mass">projected-mass</option>
            <option value="projected-rare" selected>projected-rare</option>
            <option value="projected-edge">projected-edge</option>
            <option value="projected-islands">projected-islands</option>
            <option value="projected-anchors">projected-anchors</option>
            <option value="projected-frontier">projected-frontier</option>
            <option value="projected-graft">projected-graft</option>
          </select>
        </label>
        <label>
          <span class="label-title">Palette input <span class="field-hint">color donor</span></span>
          <select id="paletteInput" data-setting title="Which processed image supplies colors for palette extraction.">
            <option value="prepared" selected>prepared target</option>
            <option value="original">original resized</option>
            <option value="graded">graded target</option>
          </select>
        </label>
        <div class="row">
          <label>
            <span class="label-title">Accent weight <span class="field-hint">rare colors</span></span>
            <input id="accentPaletteWeight" data-setting type="number" min="0" max="12" step="0.1" value="0.8" title="Extra palette pressure for saturated or visually interesting colors.">
          </label>
          <label>
            <span class="label-title">Hue rarity <span class="field-hint">unusual hues</span></span>
            <input id="hueRarityWeight" data-setting type="number" min="0" max="12" step="0.1" value="1.6" title="Extra palette pressure for hues that occupy little image area.">
          </label>
        </div>
        <label>
          <span class="label-title">Hue match weight <span class="field-hint">mapping bias</span></span>
          <input id="hueMatchWeight" data-setting type="number" min="0" max="12" step="0.05" value="0.35" title="Bias nearest-color mapping toward colors with a closer hue.">
        </label>
        <div class="row">
          <label>
            <span class="label-title">Interesting slots <span class="field-hint">reserved colors</span></span>
            <input id="interestingColorSlots" data-setting type="number" min="0" max="1024" value="0" title="Palette slots reserved for colorful rare pixels before mass allocation.">
          </label>
          <label>
            <span class="label-title">Min saturation <span class="field-hint">slot gate</span></span>
            <input id="interestingMinSaturation" data-setting type="number" min="0" max="1" step="0.01" value="0.07" title="Minimum saturation for interesting-color reservation.">
          </label>
        </div>
        <label>
          <span class="label-title">Min value <span class="field-hint">slot gate</span></span>
          <input id="interestingMinValue" data-setting type="number" min="0" max="1" step="0.01" value="0.05" title="Minimum value/brightness for interesting-color reservation.">
        </label>
        <label>
          <span class="label-title">Protected hue ranges <span class="field-hint">example 250-330</span></span>
          <input id="protectedHueRanges" data-setting type="text" value="" placeholder="250-330,330-20" title="Comma-separated hue ranges to protect, including wraparound ranges.">
        </label>
        <div class="row3">
          <label>
            <span class="label-title">Hue weight <span class="field-hint">boost</span></span>
            <input id="protectedHueWeight" data-setting type="number" min="0" max="20" step="0.1" value="0" title="Weight boost for pixels inside protected hue ranges.">
          </label>
          <label>
            <span class="label-title">Hue slots <span class="field-hint">reserve</span></span>
            <input id="protectedHueSlots" data-setting type="number" min="0" max="1024" value="0" title="Palette slots reserved for protected hue ranges.">
          </label>
          <label>
            <span class="label-title">Hue min sat <span class="field-hint">gate</span></span>
            <input id="protectedHueMinSaturation" data-setting type="number" min="0" max="1" step="0.01" value="0.08" title="Minimum saturation for protected hue reservation.">
          </label>
        </div>
      </fieldset>

      <fieldset>
        <legend>Edges</legend>
        <p class="section-note">Edge masks protect contours and can push palette slots toward line-art pixels.</p>
        <div class="row">
          <label>
            <span class="label-title">Edge filter <span class="field-hint">mask source</span></span>
            <select id="edgeMode" data-setting title="Filter used to detect contours for palette weighting and detail protection.">
              <option value="sobel" selected>Sobel</option>
              <option value="laplacian">Laplacian</option>
              <option value="highpass">High-pass</option>
              <option value="contour">Pillow contour</option>
              <option value="none">none</option>
            </select>
          </label>
          <label>
            <span class="label-title">Threshold <span class="field-hint">mask cutoff</span></span>
            <input id="edgeThreshold" data-setting type="number" min="0" max="1" step="0.005" value="0.04" title="Lower values mark more pixels as edges.">
          </label>
        </div>
        <div class="row">
          <label>
            <span class="label-title">Palette edge weight <span class="field-hint">slot boost</span></span>
            <input id="edgePaletteWeight" data-setting type="number" min="0" max="12" step="0.05" value="0.45" title="Extra palette weight for edge pixels.">
          </label>
          <label>
            <span class="label-title">Edge sharpen <span class="field-hint">contour contrast</span></span>
            <input id="edgeSharpen" data-setting type="number" min="0" max="8" step="0.05" value="0" title="Selective sharpening on detected contours before palette mapping.">
          </label>
        </div>
        <label class="check" title="Return the edge mask in the render response for debugging.">
          <input id="includeEdgePreview" data-setting type="checkbox"> include edge mask in response
        </label>
      </fieldset>

      <fieldset>
        <legend>Color Prep</legend>
        <p class="section-note">Small source corrections before palette extraction. Keep smoothing off when one-pixel strokes matter.</p>
        <div class="row">
          <label>
            <span class="label-title">Saturation <span class="field-hint">color gain</span></span>
            <input id="saturation" data-setting type="number" min="0" max="8" step="0.05" value="1" title="Color multiplier before palette extraction.">
          </label>
          <label>
            <span class="label-title">Contrast <span class="field-hint">tone gain</span></span>
            <input id="contrast" data-setting type="number" min="0" max="8" step="0.05" value="1" title="Contrast multiplier before palette extraction.">
          </label>
        </div>
        <div class="row">
          <label>
            <span class="label-title">Sharpness <span class="field-hint">unsharp %</span></span>
            <input id="sharpness" data-setting type="number" min="0" max="500" step="5" value="0" title="Unsharp-mask amount before quantization.">
          </label>
          <label>
            <span class="label-title">Autocontrast <span class="field-hint">cutoff %</span></span>
            <input id="autocontrastCutoff" data-setting type="number" min="0" max="30" step="0.5" value="0" title="Trim extremes and stretch tonal range before conversion.">
          </label>
        </div>
        <div class="row3">
          <label>
            <span class="label-title">Bilateral radius <span class="field-hint">smooth</span></span>
            <input id="bilateralRadius" data-setting type="number" min="0" max="8" value="0" title="Edge-preserving smoothing radius. It does not cut palette directly, but can remove rare source colors before quantization.">
          </label>
          <label>
            <span class="label-title">Sigma color <span class="field-hint">range</span></span>
            <input id="bilateralSigmaColor" data-setting type="number" min="1" max="128" step="1" value="18" title="How far colors may differ and still be smoothed together.">
          </label>
          <label>
            <span class="label-title">Sigma space <span class="field-hint">spread</span></span>
            <input id="bilateralSigmaSpace" data-setting type="number" min="0.1" max="16" step="0.1" value="1.4" title="Spatial falloff for bilateral smoothing.">
          </label>
        </div>
        <label class="check" title="Match the output's mean brightness back toward the resized source.">
          <input id="preserveLuma" data-setting type="checkbox"> preserve luma
        </label>
        <label class="check" title="Match output saturation back toward the resized source to reduce washed-out accents.">
          <input id="preserveSaturation" data-setting type="checkbox"> preserve saturation
        </label>
      </fieldset>

      <fieldset>
        <legend>Cleanup</legend>
        <p class="section-note">Optional post-pass for smooth areas and isolated near-color pixels. Use after palette choice is settled.</p>
        <div class="row">
          <label>
            <span class="label-title">Flat palette colors <span class="field-hint">0 off</span></span>
            <input id="flatRegionPaletteColors" data-setting type="number" min="0" max="1024" value="0" title="Map low-detail non-edge areas to a small local palette.">
          </label>
          <label>
            <span class="label-title">Flat channel step <span class="field-hint">0 off</span></span>
            <input id="flatRegionChannelStep" data-setting type="number" min="0" max="255" value="0" title="Snap low-detail channels to fixed RGB steps.">
          </label>
        </div>
        <div class="row">
          <label>
            <span class="label-title">Flat max sat <span class="field-hint">protect accents</span></span>
            <input id="flatRegionMaxSaturation" data-setting type="number" min="0" max="1" step="0.01" value="0.35" title="Only low-saturation areas below this value can be flattened.">
          </label>
          <label>
            <span class="label-title">Flat edge max <span class="field-hint">protect lines</span></span>
            <input id="flatRegionEdgeThreshold" data-setting type="number" min="0" max="1" step="0.01" value="0.18" title="Flattening is blocked above this edge strength.">
          </label>
        </div>
        <label>
          <span class="label-title">Flat luma range <span class="field-hint">smoothness</span></span>
          <input id="flatRegionLumaRange" data-setting type="number" min="0" max="255" step="1" value="10" title="Maximum local brightness range eligible for flat-region cleanup.">
        </label>
        <div class="row3">
          <label>
            <span class="label-title">Mixel passes <span class="field-hint">0 off</span></span>
            <input id="mixelCleanupPasses" data-setting type="number" min="0" max="8" value="0" title="Number of isolated-pixel cleanup passes.">
          </label>
          <label>
            <span class="label-title">Neighbors <span class="field-hint">3x3 vote</span></span>
            <input id="mixelCleanupMinNeighbors" data-setting type="number" min="1" max="9" value="3" title="Minimum matching neighbors needed before replacing an isolated pixel.">
          </label>
          <label>
            <span class="label-title">Distance <span class="field-hint">near color</span></span>
            <input id="mixelCleanupDistance" data-setting type="number" min="0" max="255" step="1" value="18" title="Maximum color distance allowed for mixel replacement.">
          </label>
        </div>
        <label>
          <span class="label-title">Mixel max sat <span class="field-hint">protect accents</span></span>
          <input id="mixelCleanupMaxSaturation" data-setting type="number" min="0" max="1" step="0.01" value="0.45" title="Pixels above this saturation are protected from mixel cleanup.">
        </label>
      </fieldset>

      <fieldset>
        <legend>Export</legend>
        <p class="section-note">Save the current render. The swatches show the actual palette returned by the latest output.</p>
        <button id="savePng" title="Download the latest rendered output as a PNG.">Save current PNG</button>
        <div id="palette" class="palette"></div>
      </fieldset>
    </aside>

    <main>
      <div class="topbar">
        <span id="status" class="status-pill status-ok">waiting for image</span>
        <span id="zoomInfo" class="metric">zoom 100%</span>
        <span id="stats"></span>
        <span class="spacer"></span>
        <span class="small">Wheel zooms at cursor. Drag pans. Hold Z to compare original/output.</span>
      </div>
      <div id="viewer" class="viewer">
        <canvas id="canvas"></canvas>
      </div>
    </main>
  </div>
  <div id="tooltip">
    <canvas id="tipCanvas" width="250" height="250"></canvas>
    <div class="label"><span>original full-res</span><span>output</span></div>
  </div>

  <script>
    const MIN_OUTPUT_SIZE = 16;
    const MAX_OUTPUT_HEIGHT = 1024;
    const MAX_OUTPUT_WIDTH = 4096;
    const PRESET_STORAGE_KEY = 'pixel-art-lab-custom-presets-v1';

    const state = {
      imageLoaded: false,
      sourceWidth: 0,
      sourceHeight: 0,
      sourceAspect: 0,
      outputImg: null,
      originalImg: null,
      outputDataUrl: null,
      originalDataUrl: null,
      width: 0,
      height: 0,
      zoom: 1,
      offsetX: 0,
      offsetY: 0,
      dragging: false,
      dragStartX: 0,
      dragStartY: 0,
      dragOffsetX: 0,
      dragOffsetY: 0,
      hoverX: 0,
      hoverY: 0,
      clientX: 0,
      clientY: 0,
      hovering: false,
      zDown: false,
      renderTimer: 0,
      renderSeq: 0,
      activeController: null,
      syncingDimensions: false,
    };

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const viewer = document.getElementById('viewer');
    const tooltip = document.getElementById('tooltip');
    const tipCanvas = document.getElementById('tipCanvas');
    const tipCtx = tipCanvas.getContext('2d');
    const statusEl = document.getElementById('status');
    const zoomInfo = document.getElementById('zoomInfo');
    const statsEl = document.getElementById('stats');
    const paletteEl = document.getElementById('palette');
    const aspectInfo = document.getElementById('aspectInfo');
    const gridInfo = document.getElementById('gridInfo');
    const customPresetSelect = document.getElementById('customPresetSelect');
    const presetNameInput = document.getElementById('presetName');

    function setStatus(text, cls) {
      statusEl.className = `status-pill ${cls || ''}`;
      statusEl.textContent = text;
    }

    function settingEl(id) { return document.getElementById(id); }

    function isDimensionField(el) {
      return el.id === 'targetWidth' || el.id === 'targetHeight';
    }

    function clampValue(value, minValue, maxValue) {
      return Math.min(maxValue, Math.max(minValue, value));
    }

    function sourceAspectDimensions(driver, requestedWidth, requestedHeight) {
      if (!state.sourceAspect || !state.sourceWidth || !state.sourceHeight) {
        return {
          width: clampValue(Math.round(requestedWidth || 1024), MIN_OUTPUT_SIZE, MAX_OUTPUT_WIDTH),
          height: clampValue(Math.round(requestedHeight || 576), MIN_OUTPUT_SIZE, MAX_OUTPUT_HEIGHT),
        };
      }

      const defaultHeight = Math.min(state.sourceHeight, MAX_OUTPUT_HEIGHT);
      const defaultWidth = Math.round(defaultHeight * state.sourceAspect);
      let width = Math.round(requestedWidth || defaultWidth);
      let height = Math.round(requestedHeight || defaultHeight);

      if (driver === 'width') {
        width = clampValue(width, MIN_OUTPUT_SIZE, MAX_OUTPUT_WIDTH);
        height = Math.max(MIN_OUTPUT_SIZE, Math.round(width / state.sourceAspect));
      } else {
        height = clampValue(height, MIN_OUTPUT_SIZE, MAX_OUTPUT_HEIGHT);
        width = Math.max(MIN_OUTPUT_SIZE, Math.round(height * state.sourceAspect));
      }

      if (height > MAX_OUTPUT_HEIGHT) {
        height = MAX_OUTPUT_HEIGHT;
        width = Math.max(MIN_OUTPUT_SIZE, Math.round(height * state.sourceAspect));
      }
      if (width > MAX_OUTPUT_WIDTH) {
        width = MAX_OUTPUT_WIDTH;
        height = Math.max(MIN_OUTPUT_SIZE, Math.round(width / state.sourceAspect));
      }
      if (height > MAX_OUTPUT_HEIGHT) {
        height = MAX_OUTPUT_HEIGHT;
        width = Math.max(MIN_OUTPUT_SIZE, Math.round(height * state.sourceAspect));
      }

      return { width, height };
    }

    function setDimensionInputs(width, height) {
      state.syncingDimensions = true;
      settingEl('targetWidth').value = width;
      settingEl('targetHeight').value = height;
      state.syncingDimensions = false;
      if (state.sourceWidth && state.sourceHeight) {
        aspectInfo.textContent = `source ${state.sourceWidth}x${state.sourceHeight}, output ${width}x${height}`;
      }
    }

    function applySourceOutputSize(sourceWidth, sourceHeight, targetWidth, targetHeight) {
      state.sourceWidth = sourceWidth;
      state.sourceHeight = sourceHeight;
      state.sourceAspect = sourceHeight > 0 ? sourceWidth / sourceHeight : 0;
      settingEl('aspectLock').checked = true;
      settingEl('aspectDriver').value = 'height';
      const dims = sourceAspectDimensions(
        'height',
        targetWidth || 0,
        targetHeight || Math.min(sourceHeight, MAX_OUTPUT_HEIGHT)
      );
      setDimensionInputs(dims.width, dims.height);
    }

    function syncLockedDimensions(driver) {
      if (state.syncingDimensions || !settingEl('aspectLock').checked || !state.sourceAspect) return false;
      settingEl('aspectDriver').value = driver;
      const dims = sourceAspectDimensions(
        driver,
        Number(settingEl('targetWidth').value),
        Number(settingEl('targetHeight').value)
      );
      setDimensionInputs(dims.width, dims.height);
      return true;
    }

    function syncConditionalControls() {
      const enabled = settingEl('gridSnapMethod').value === 'dark-stroke';
      const input = settingEl('gridDarkThreshold');
      const wrapper = document.getElementById('gridDarkThresholdWrap');
      input.disabled = !enabled;
      input.title = enabled
        ? 'Minimum contrast for preserving a narrow dark stroke inside a detected cell.'
        : 'Used only by the dark-stroke bias cell reducer.';
      if (wrapper) wrapper.classList.toggle('disabled', !enabled);
    }

    function collectSettings() {
      const settings = {};
      document.querySelectorAll('[data-setting]').forEach((el) => {
        if (el.type === 'checkbox') settings[el.id] = el.checked;
        else if (el.type === 'number' || el.type === 'range') settings[el.id] = Number(el.value);
        else settings[el.id] = el.value;
      });
      return settings;
    }

    function readPresetStore() {
      try {
        const raw = localStorage.getItem(PRESET_STORAGE_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
      } catch (_err) {
        return {};
      }
    }

    function writePresetStore(store) {
      localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(store));
    }

    function presetNames(store = readPresetStore()) {
      return Object.keys(store).sort((a, b) => a.localeCompare(b));
    }

    function refreshPresetSelect(selectedName = '') {
      const store = readPresetStore();
      const names = presetNames(store);
      customPresetSelect.innerHTML = '';
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = names.length ? 'select preset' : 'no saved presets';
      customPresetSelect.appendChild(placeholder);
      names.forEach((name) => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        customPresetSelect.appendChild(option);
      });
      customPresetSelect.value = selectedName && store[selectedName] ? selectedName : '';
      if (customPresetSelect.value) presetNameInput.value = customPresetSelect.value;
    }

    function applySettings(settings) {
      if (!settings || typeof settings !== 'object') return;
      document.querySelectorAll('[data-setting]').forEach((el) => {
        if (!Object.prototype.hasOwnProperty.call(settings, el.id)) return;
        if (el.type === 'checkbox') el.checked = Boolean(settings[el.id]);
        else el.value = settings[el.id];
      });
      syncConditionalControls();
      scheduleRender(40);
    }

    function saveCurrentPreset() {
      const name = presetNameInput.value.trim();
      if (!name) {
        setStatus('enter a preset name', 'status-error');
        return;
      }
      const store = readPresetStore();
      store[name] = collectSettings();
      try {
        writePresetStore(store);
      } catch (err) {
        setStatus(err.message || 'could not save preset', 'status-error');
        return;
      }
      refreshPresetSelect(name);
      setStatus(`saved preset: ${name}`, 'status-ok');
    }

    function loadSelectedPreset() {
      const name = customPresetSelect.value;
      if (!name) {
        setStatus('select a preset first', 'status-error');
        return;
      }
      const store = readPresetStore();
      if (!store[name]) {
        refreshPresetSelect();
        setStatus('preset not found', 'status-error');
        return;
      }
      presetNameInput.value = name;
      applySettings(store[name]);
      setStatus(`loaded preset: ${name}`, 'status-ok');
    }

    function deleteSelectedPreset() {
      const name = customPresetSelect.value;
      if (!name) {
        setStatus('select a preset first', 'status-error');
        return;
      }
      const store = readPresetStore();
      delete store[name];
      writePresetStore(store);
      presetNameInput.value = '';
      refreshPresetSelect();
      setStatus(`deleted preset: ${name}`, 'status-ok');
    }

    function resizeCanvas() {
      const rect = viewer.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      clampPan();
      draw();
    }

    function viewportSize() {
      const rect = viewer.getBoundingClientRect();
      return { w: rect.width, h: rect.height };
    }

    function clampPan() {
      if (!state.outputImg) return;
      const { w, h } = viewportSize();
      const imgW = state.width * state.zoom;
      const imgH = state.height * state.zoom;
      if (imgW <= w) state.offsetX = (w - imgW) / 2;
      else state.offsetX = Math.min(0, Math.max(w - imgW, state.offsetX));
      if (imgH <= h) state.offsetY = (h - imgH) / 2;
      else state.offsetY = Math.min(0, Math.max(h - imgH, state.offsetY));
    }

    function drawPlaceholder() {
      const { w, h } = viewportSize();
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#11141c';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#9da7bb';
      ctx.font = '15px ui-monospace, monospace';
      ctx.fillText('load an image to start', 28, 38);
    }

    function draw() {
      const { w, h } = viewportSize();
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, w, h);
      if (!state.outputImg) {
        drawPlaceholder();
        return;
      }
      clampPan();
      ctx.drawImage(
        state.outputImg,
        state.offsetX,
        state.offsetY,
        state.width * state.zoom,
        state.height * state.zoom
      );
      zoomInfo.textContent = `zoom ${Math.round(state.zoom * 100)}%`;
      if (state.zDown && state.hovering) drawTooltip();
    }

    function viewerPoint(evt) {
      const rect = canvas.getBoundingClientRect();
      return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
    }

    function imagePointFromViewer(x, y) {
      return {
        x: (x - state.offsetX) / state.zoom,
        y: (y - state.offsetY) / state.zoom,
      };
    }

    canvas.addEventListener('wheel', (evt) => {
      if (!state.outputImg) return;
      evt.preventDefault();
      const point = viewerPoint(evt);
      const before = imagePointFromViewer(point.x, point.y);
      const factor = evt.deltaY < 0 ? 1.2 : 1 / 1.2;
      const nextZoom = Math.min(16, Math.max(1, state.zoom * factor));
      state.zoom = nextZoom;
      state.offsetX = point.x - before.x * state.zoom;
      state.offsetY = point.y - before.y * state.zoom;
      clampPan();
      draw();
    }, { passive: false });

    canvas.addEventListener('pointerdown', (evt) => {
      if (!state.outputImg) return;
      state.dragging = true;
      viewer.classList.add('dragging');
      state.dragStartX = evt.clientX;
      state.dragStartY = evt.clientY;
      state.dragOffsetX = state.offsetX;
      state.dragOffsetY = state.offsetY;
      canvas.setPointerCapture(evt.pointerId);
    });

    canvas.addEventListener('pointermove', (evt) => {
      const point = viewerPoint(evt);
      state.hoverX = point.x;
      state.hoverY = point.y;
      state.clientX = evt.clientX;
      state.clientY = evt.clientY;
      state.hovering = true;
      if (state.dragging) {
        state.offsetX = state.dragOffsetX + (evt.clientX - state.dragStartX);
        state.offsetY = state.dragOffsetY + (evt.clientY - state.dragStartY);
        clampPan();
        draw();
      } else if (state.zDown) {
        drawTooltip();
      }
    });

    canvas.addEventListener('pointerleave', () => {
      state.hovering = false;
      hideTooltip();
    });

    canvas.addEventListener('pointerup', (evt) => {
      state.dragging = false;
      viewer.classList.remove('dragging');
      try { canvas.releasePointerCapture(evt.pointerId); } catch (_err) {}
    });

    function hideTooltip() {
      tooltip.style.display = 'none';
    }

    function originalCropRect() {
      const sourceWidth = state.originalImg ? state.originalImg.naturalWidth : state.sourceWidth;
      const sourceHeight = state.originalImg ? state.originalImg.naturalHeight : state.sourceHeight;
      if (!sourceWidth || !sourceHeight || !state.width || !state.height) {
        return { x: 0, y: 0, w: sourceWidth || 1, h: sourceHeight || 1 };
      }

      const sourceAspect = sourceWidth / sourceHeight;
      const targetAspect = state.width / state.height;
      if (Math.abs(sourceAspect - targetAspect) <= 0.000001) {
        return { x: 0, y: 0, w: sourceWidth, h: sourceHeight };
      }

      if (sourceAspect > targetAspect) {
        const cropWidth = Math.round(sourceHeight * targetAspect);
        return { x: Math.floor((sourceWidth - cropWidth) / 2), y: 0, w: cropWidth, h: sourceHeight };
      }

      const cropHeight = Math.round(sourceWidth / targetAspect);
      return { x: 0, y: Math.floor((sourceHeight - cropHeight) / 2), w: sourceWidth, h: cropHeight };
    }

    function drawTooltip() {
      if (!state.outputImg || !state.originalImg) return hideTooltip();
      const p = imagePointFromViewer(state.hoverX, state.hoverY);
      if (p.x < 0 || p.y < 0 || p.x >= state.width || p.y >= state.height) return hideTooltip();

      const crop = Math.max(24, Math.min(160, Math.round(96 / state.zoom)));
      const sx = Math.max(0, Math.min(state.width - crop, Math.round(p.x - crop / 2)));
      const sy = Math.max(0, Math.min(state.height - crop, Math.round(p.y - crop / 2)));
      const originalRect = originalCropRect();
      const osx = originalRect.x + sx / state.width * originalRect.w;
      const osy = originalRect.y + sy / state.height * originalRect.h;
      const ocw = crop / state.width * originalRect.w;
      const och = crop / state.height * originalRect.h;
      tipCtx.clearRect(0, 0, 250, 250);
      tipCtx.imageSmoothingEnabled = true;
      tipCtx.drawImage(state.originalImg, osx, osy, ocw, och, 0, 0, 250, 250);
      tipCtx.save();
      tipCtx.beginPath();
      tipCtx.rect(125, 0, 125, 250);
      tipCtx.clip();
      tipCtx.imageSmoothingEnabled = false;
      tipCtx.drawImage(state.outputImg, sx, sy, crop, crop, 0, 0, 250, 250);
      tipCtx.restore();
      tipCtx.strokeStyle = '#eef1f8';
      tipCtx.lineWidth = 1;
      tipCtx.beginPath();
      tipCtx.moveTo(125.5, 0);
      tipCtx.lineTo(125.5, 250);
      tipCtx.stroke();

      const pad = 16;
      const tw = 260;
      const th = 284;
      let left = state.clientX + pad;
      let top = state.clientY + pad;
      if (left + tw > window.innerWidth) left = window.innerWidth - tw - pad;
      if (top + th > window.innerHeight) top = window.innerHeight - th - pad;
      tooltip.style.left = left + 'px';
      tooltip.style.top = top + 'px';
      tooltip.style.display = 'block';
    }

    document.addEventListener('keydown', (evt) => {
      if (evt.key.toLowerCase() === 'z') {
        state.zDown = true;
        if (state.hovering) drawTooltip();
      }
    });

    document.addEventListener('keyup', (evt) => {
      if (evt.key.toLowerCase() === 'z') {
        state.zDown = false;
        hideTooltip();
      }
    });

    async function postJson(url, payload, signal) {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }

    function loadImageUrl(url) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = url;
      });
    }

    async function renderNow() {
      if (!state.imageLoaded) return;
      state.renderSeq += 1;
      const seq = state.renderSeq;
      if (state.activeController) state.activeController.abort();
      const controller = new AbortController();
      state.activeController = controller;
      setStatus('rendering...', 'status-busy');
      try {
        const result = await postJson('/api/render', { settings: collectSettings() }, controller.signal);
        if (seq !== state.renderSeq) return;
        const outputImg = await loadImageUrl(result.output);
        state.outputImg = outputImg;
        state.outputDataUrl = result.output;
        state.width = result.width;
        state.height = result.height;
        state.zoom = Math.max(1, state.zoom);
        clampPan();
        draw();
        drawPalette(result.palette || []);
        const s = result.stats || {};
        const cacheText = s.cacheHit ? ', cached' : (s.stageCacheHits ? `, ${s.stageCacheHits} stage hits` : '');
        statsEl.textContent = `${state.width}x${state.height}, ${s.colorsWritten}/${s.colorsRequested} colors, ${s.elapsedMs} ms${cacheText}, luma ${s.outputLuma}, sat ${s.outputSaturation}`;
        const variants = Array.isArray(s.gridVariants) ? s.gridVariants : [];
        settingEl('gridVariant').max = Math.max(0, variants.length - 1);
        if (s.gridAutoSize && s.gridVariant) {
          setDimensionInputs(state.width, state.height);
        }
        if (s.gridSnap && s.gridVariant) {
          const v = s.gridVariant;
          const top = variants.slice(0, 5).map((item, index) => `${index}: ${item.width}x${item.height} @${item.cellSize}px`).join(' | ');
          gridInfo.textContent = `selected ${v.width}x${v.height}, source mixel ${v.cellSize}px, score ${v.score}. ${top}`;
        } else if (settingEl('gridSnap').checked) {
          gridInfo.textContent = 'Grid snap uses manual Width/Height. Enable auto size to choose detector variants.';
        } else {
          gridInfo.textContent = 'Auto grid size is optional; manual Width/Height still works when it is off.';
        }
        setStatus('live', 'status-ok');
      } catch (err) {
        if (err.name === 'AbortError') return;
        setStatus(err.message || String(err), 'status-error');
      }
    }

    function scheduleRender(delay = 220) {
      clearTimeout(state.renderTimer);
      state.renderTimer = setTimeout(renderNow, delay);
    }

    function drawPalette(palette) {
      paletteEl.innerHTML = '';
      palette.forEach((color) => {
        const div = document.createElement('div');
        div.className = 'swatch';
        div.title = color;
        div.style.background = color;
        paletteEl.appendChild(div);
      });
    }

    document.querySelectorAll('[data-setting]').forEach((el) => {
      el.addEventListener('input', () => {
        if (isDimensionField(el)) {
          clearTimeout(state.renderTimer);
          return;
        }
        if (el.id === 'gridSnapMethod') syncConditionalControls();
        if (el.id === 'aspectLock' && el.checked) syncLockedDimensions(settingEl('aspectDriver').value || 'height');
        scheduleRender();
      });
      el.addEventListener('change', () => {
        if (el.id === 'targetWidth') syncLockedDimensions('width');
        else if (el.id === 'targetHeight') syncLockedDimensions('height');
        else if (el.id === 'aspectLock' && el.checked) syncLockedDimensions(settingEl('aspectDriver').value || 'height');
        else if (el.id === 'gridSnapMethod') syncConditionalControls();
        scheduleRender(40);
      });
      el.addEventListener('keydown', (evt) => {
        if (!isDimensionField(el) || evt.key !== 'Enter') return;
        evt.preventDefault();
        if (el.id === 'targetWidth') syncLockedDimensions('width');
        else syncLockedDimensions('height');
        scheduleRender(40);
        el.blur();
      });
    });

    customPresetSelect.addEventListener('change', () => {
      presetNameInput.value = customPresetSelect.value;
    });
    document.getElementById('loadPreset').addEventListener('click', loadSelectedPreset);
    document.getElementById('savePreset').addEventListener('click', saveCurrentPreset);
    document.getElementById('deletePreset').addEventListener('click', deleteSelectedPreset);

    document.getElementById('resetZoom').addEventListener('click', () => {
      state.zoom = 1;
      clampPan();
      draw();
    });

    document.getElementById('savePng').addEventListener('click', () => {
      if (!state.outputDataUrl) return;
      const link = document.createElement('a');
      link.href = state.outputDataUrl;
      link.download = `pixel-art-${state.width}x${state.height}-${settingEl('colors').value}c.png`;
      link.click();
    });

    document.getElementById('saveSettings').addEventListener('click', () => {
      const data = 'data:application/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(collectSettings(), null, 2));
      const link = document.createElement('a');
      link.href = data;
      link.download = 'pixel-art-lab-settings.json';
      link.click();
    });

    document.getElementById('file').addEventListener('change', async (evt) => {
      const file = evt.target.files && evt.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async () => {
        setStatus('uploading...', 'status-busy');
        try {
          const originalDataUrl = reader.result;
          const [result, originalImg] = await Promise.all([
            postJson('/api/image', {
              name: file.name,
              data: originalDataUrl,
            }),
            loadImageUrl(originalDataUrl),
          ]);
          state.imageLoaded = true;
          state.originalImg = originalImg;
          state.originalDataUrl = originalDataUrl;
          applySourceOutputSize(result.width, result.height, result.targetWidth, result.targetHeight);
          document.getElementById('inputInfo').textContent =
            `${result.name}: ${result.width}x${result.height}`;
          setStatus('image loaded', 'status-ok');
          state.zoom = 1;
          state.offsetX = 0;
          state.offsetY = 0;
          scheduleRender(40);
        } catch (err) {
          setStatus(err.message || String(err), 'status-error');
        }
      };
      reader.readAsDataURL(file);
    });

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    syncConditionalControls();
    refreshPresetSelect();
  </script>
</body>
</html>
"""


class LabHandler(BaseHTTPRequestHandler):
    server_version = "PixelArtLab/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_UPLOAD_BYTES * 2:
            raise ValueError("request body is too large")
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def send_bytes(self, content: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def send_error_json(self, error: Exception, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self.send_json({"error": str(error)}, status=status)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            with STATE_LOCK:
                image = STATE.image
                payload = {
                    "loaded": image is not None,
                    "name": STATE.name,
                    "width": image.width if image else 0,
                    "height": image.height if image else 0,
                }
            self.send_json(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/image":
                payload = self.read_json()
                image = data_url_to_image(str(payload.get("data", "")))
                name = str(payload.get("name", "uploaded-image"))
                with RENDER_LOCK:
                    with STATE_LOCK:
                        STATE.image = image
                        STATE.name = name
                        STATE.uploaded_at = time.time()
                        STATE.version += 1
                        STATE.cache.clear()
                target_width, target_height = source_aspect_dimensions(image.width, image.height)
                self.send_json(
                    {
                        "ok": True,
                        "name": name,
                        "width": image.width,
                        "height": image.height,
                        "targetWidth": target_width,
                        "targetHeight": target_height,
                        "maxTargetHeight": MAX_OUTPUT_HEIGHT,
                    }
                )
                return

            if path == "/api/render":
                payload = self.read_json()
                settings = payload.get("settings", {})
                if not isinstance(settings, dict):
                    raise ValueError("settings must be an object")
                with STATE_LOCK:
                    if STATE.image is None:
                        raise ValueError("load an image first")
                    image = STATE.image
                    version = STATE.version
                    cache = STATE.cache
                with RENDER_LOCK:
                    result = convert_in_memory(image, settings, cache=cache, version=version)
                self.send_json(result)
                return

            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover - user-facing local tool path.
            self.send_error_json(exc)


def find_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"could not find a free port from {start} to {start + 49}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local pixel-art conversion GUI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host, default: 127.0.0.1.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Preferred port, default: {DEFAULT_PORT}.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    port = find_port(args.port)
    server = ThreadingHTTPServer((args.host, port), LabHandler)
    url = f"http://{args.host}:{port}/"
    print(f"Pixel Art Lab running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Pixel Art Lab.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
