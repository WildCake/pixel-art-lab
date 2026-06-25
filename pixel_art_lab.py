#!/usr/bin/env python3
"""Local browser lab for the pixel-art palette conversion pipeline."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import mimetypes
import re
import secrets
import socket
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

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
SESSION_COOKIE_NAME = "pixel_art_lab_session"
MAX_SESSION_COUNT = 128
SERVER_BROWSER_ROOT = SCRIPT_DIR.parent / "assets" / "generated"
SERVER_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
SERVER_THUMBNAIL_SIZE = (160, 120)
PIXEL_LAB_SAVE_SUFFIX = "_PIXEL_LAB"
PALETTE_STRATEGIES = (
    "median-cut",
    "kmeans",
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
    source_path: Path | None = None
    uploaded_at: float = 0.0
    touched_at: float = field(default_factory=time.time)
    version: int = 0
    cache: dict[str, OrderedDict[tuple[Any, ...], Any]] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)
    render_lock: threading.Lock = field(default_factory=threading.Lock)


STATE = LabState()
SESSIONS: OrderedDict[str, LabState] = OrderedDict()
SESSIONS_LOCK = threading.Lock()


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


def bilateral_mode_from_settings(settings: dict[str, Any]) -> str:
    legacy_mode = as_choice(settings, "bilateralMode", "edge-safe", ("standard", "edge-safe"))
    safe_edges = as_bool(settings, "bilateralSafeEdges", legacy_mode == "edge-safe")
    return "edge-safe" if safe_edges else "standard"


def data_url_to_image(data_url: str) -> Image.Image:
    if "," in data_url:
        _header, data_url = data_url.split(",", 1)
    raw = base64.b64decode(data_url, validate=False)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("image upload is too large")
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image.convert("RGBA") if pag.image_has_alpha(image) else image.convert("RGB")


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def data_url_to_png_bytes(data_url: str) -> bytes:
    if "," in data_url:
        _header, data_url = data_url.split(",", 1)
    raw = base64.b64decode(data_url, validate=False)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("image output is too large")
    image = Image.open(io.BytesIO(raw))
    image.load()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def server_root() -> Path:
    return SERVER_BROWSER_ROOT.resolve()


def path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def normalize_server_rel_path(value: str) -> str:
    normalized = str(value or "").replace("\\", "/").strip("/")
    if normalized in {"", "."}:
        return ""
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError("server path must stay inside assets/generated")
    return "/".join(parts)


def resolve_server_path(value: str) -> Path:
    root = server_root()
    rel_path = normalize_server_rel_path(value)
    path = (root / rel_path).resolve()
    if not path_is_relative_to(path, root):
        raise ValueError("server path must stay inside assets/generated")
    return path


def server_relative_path(path: Path) -> str:
    root = server_root()
    resolved = path.resolve()
    if not path_is_relative_to(resolved, root):
        raise ValueError("server path must stay inside assets/generated")
    if resolved == root:
        return ""
    return resolved.relative_to(root).as_posix()


def is_supported_server_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SERVER_IMAGE_EXTENSIONS


def sanitize_session_id(value: str | None) -> str:
    if value and re.fullmatch(r"[A-Za-z0-9_-]{16,96}", value):
        return value
    return secrets.token_urlsafe(24)


def get_session(session_id: str) -> LabState:
    now = time.time()
    with SESSIONS_LOCK:
        state = SESSIONS.get(session_id)
        if state is None:
            state = LabState()
            SESSIONS[session_id] = state
        else:
            SESSIONS.move_to_end(session_id)
        state.touched_at = now
        while len(SESSIONS) > MAX_SESSION_COUNT:
            SESSIONS.popitem(last=False)
        return state


def open_server_image(path: Path) -> Image.Image:
    if not is_supported_server_image(path):
        raise ValueError("server file must be a PNG, JPG, WebP, or GIF image")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("server file is too large")
    image = Image.open(path)
    image.load()
    return image.convert("RGBA") if pag.image_has_alpha(image) else image.convert("RGB")


def pixel_lab_save_path(source_path: Path) -> Path:
    root = server_root()
    server_relative_path(source_path)
    stem = source_path.stem
    if stem.lower().endswith(PIXEL_LAB_SAVE_SUFFIX.lower()):
        target_path = source_path.with_suffix(".png")
    else:
        target_path = source_path.with_name(f"{stem}{PIXEL_LAB_SAVE_SUFFIX}.png")
    resolved = target_path.resolve()
    if not path_is_relative_to(resolved, root):
        raise ValueError("save path must stay inside assets/generated")
    return resolved


def server_thumbnail_png(path: Path) -> bytes:
    if not is_supported_server_image(path):
        raise ValueError("server file must be a supported image")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("server file is too large")
    with Image.open(path) as image:
        image.load()
        thumbnail = image.convert("RGBA") if pag.image_has_alpha(image) else image.convert("RGB")
    thumbnail.thumbnail(SERVER_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def image_response_payload(image: Image.Image, name: str, source_path: Path | None = None) -> dict[str, Any]:
    target_width, target_height = source_aspect_dimensions(image.width, image.height)
    save_target_path = pixel_lab_save_path(source_path) if source_path is not None else None
    return {
        "ok": True,
        "name": name,
        "width": image.width,
        "height": image.height,
        "targetWidth": target_width,
        "targetHeight": target_height,
        "maxTargetHeight": MAX_OUTPUT_HEIGHT,
        "sourcePath": server_relative_path(source_path) if source_path is not None else None,
        "saveTargetPath": server_relative_path(save_target_path) if save_target_path is not None else None,
        "canSaveInPlace": save_target_path is not None,
    }


def list_server_files(value: str) -> dict[str, Any]:
    root = server_root()
    path = resolve_server_path(value)
    if not path.exists():
        raise ValueError("server folder does not exist")
    if not path.is_dir():
        raise ValueError("server path must be a folder")
    entries: list[dict[str, Any]] = []
    for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            entries.append(
                {
                    "type": "dir",
                    "name": child.name,
                    "path": server_relative_path(child),
                }
            )
            continue
        if is_supported_server_image(child):
            stat = child.stat()
            entries.append(
                {
                    "type": "file",
                    "name": child.name,
                    "path": server_relative_path(child),
                    "size": stat.st_size,
                    "mtime": int(stat.st_mtime),
                }
            )
    rel_path = server_relative_path(path)
    parent = ""
    if rel_path:
        parent = server_relative_path(path.parent)
    return {
        "root": "assets/generated",
        "path": rel_path,
        "parent": parent,
        "entries": entries,
    }


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
    grid_snap_enabled = as_bool(settings, "gridSnap", False)
    grid_snap_quantize_first = as_bool(settings, "gridQuantizeFirst", False)
    dither = as_choice(settings, "dither", "none", ("ordered", "floyd", "none"))
    if grid_snap_enabled and grid_snap_quantize_first:
        dither = "none"
    protected_ranges = parse_ranges(str(settings.get("protectedHueRanges", "")).strip())
    return pag.PixelArtConfig(
        target_width=width,
        target_height=height,
        colors=colors,
        preview_scale=1,
        dither=dither,
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
        grid_snap_enabled=grid_snap_enabled,
        grid_snap_method=as_choice(settings, "gridSnapMethod", "center", ("cell-mode", "center", "dark-stroke")),
        grid_snap_quantize_first=grid_snap_quantize_first,
        grid_snap_dark_threshold=as_float(settings, "gridDarkThreshold", 38.0, 0.0, 255.0),
        grid_snap_topology=as_choice(settings, "gridTopology", "elastic", ("uniform", "elastic")),
        grid_snap_axis_stabilization=as_choice(
            settings,
            "gridAxisStabilization",
            "conservative",
            ("off", "conservative", "aggressive"),
        ),
        preserve_luma=as_bool(settings, "preserveLuma", False),
        preserve_saturation=as_bool(settings, "preserveSaturation", False),
        preserve_alpha=as_bool(settings, "preserveAlpha", True),
        palette_source=None,
        bilateral_radius=as_int(settings, "bilateralRadius", 0, 0, 8),
        bilateral_mode=bilateral_mode_from_settings(settings),
        bilateral_sigma_color=as_float(settings, "bilateralSigmaColor", 18.0, 1.0, 128.0),
        bilateral_sigma_space=as_float(settings, "bilateralSigmaSpace", 1.4, 0.1, 16.0),
        edge_palette_weight=as_float(settings, "edgePaletteWeight", 0.45, 0.0, 12.0),
        edge_sharpen=as_float(settings, "edgeSharpen", 0.0, 0.0, 8.0),
        edge_threshold=as_float(settings, "edgeThreshold", 0.04, 0.0, 1.0),
        palette_strategy=as_choice(settings, "paletteStrategy", "projected-rare", PALETTE_STRATEGIES),
        color_distance=as_choice(settings, "colorDistance", "rgb", ("rgb", "oklab")),
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
    requested_dither = as_choice(settings, "dither", "none", ("ordered", "floyd", "none"))
    dither_disabled_reason = (
        "gridQuantizeFirst"
        if requested_dither != "none" and config.grid_snap_enabled and config.grid_snap_quantize_first
        else None
    )

    if config.dither != "none" and config.colors > 256:
        raise ValueError("dither modes support at most 256 colors")

    grid_vote_palette_key = (
        (
            config.colors,
            config.palette_strategy,
            config.color_distance,
            config.accent_palette_weight,
            config.hue_rarity_weight,
            config.interesting_color_slots,
            config.interesting_min_saturation,
            config.interesting_min_value,
            config.protected_hue_ranges,
            config.protected_hue_weight,
            config.protected_hue_slots,
            config.protected_hue_min_saturation,
            config.hue_match_weight,
        )
        if config.grid_snap_quantize_first
        else None
    )
    base_key = (
        "base",
        version,
        config.target_width,
        config.target_height,
        config.resample,
        grid_vote_palette_key,
        config.grid_snap_enabled,
        config.grid_snap_method,
        config.grid_snap_quantize_first,
        config.grid_snap_dark_threshold,
        config.grid_snap_topology,
        config.grid_snap_axis_stabilization,
    )
    base = cached("stage", base_key, lambda: pag.prepare_base_image(image, config))
    alpha_key = ("alpha", base_key, config.preserve_alpha)
    alpha_channel = cached("stage", alpha_key, lambda: pag.prepare_alpha_channel(image, config))
    requested_edge_mode = as_choice(settings, "edgeMode", "sobel", ("sobel", "laplacian", "highpass", "contour", "none"))
    edge_mode_disabled_reason = None
    edge_mode = requested_edge_mode
    if requested_edge_mode == "contour" and config.dither != "none":
        edge_mode = "sobel"
        edge_mode_disabled_reason = "ditherContour"
    edge_key = ("edge", base_key, edge_mode, config.edge_threshold)
    edge_mask = cached("stage", edge_key, lambda: build_lab_edge_mask(base, edge_mode, config.edge_threshold))
    processed_key = (
        "processed",
        base_key,
        edge_key if config.bilateral_mode == "edge-safe" else None,
        config.bilateral_radius,
        config.bilateral_mode,
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
            mode=config.bilateral_mode,
            edge_mask=edge_mask,
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
    alpha_preserved = alpha_channel is not None
    grid_axis_stabilization = (
        config.grid_snap_axis_stabilization
        if config.grid_snap_enabled and config.grid_snap_topology == "elastic"
        else "off"
    )
    grid_cut_path = (
        "elastic-cuts"
        if config.grid_snap_enabled and config.grid_snap_topology == "elastic"
        else ("uniform-origin" if config.grid_snap_enabled else None)
    )

    output_luma = pag.luma_mean(pixel_art)
    output_saturation = pag.luma_weighted_saturation_mean(pixel_art)
    pixel_art = pag.attach_alpha(pixel_art, alpha_channel)
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
            "edgeModeRequested": requested_edge_mode,
            "edgeModeDisabledReason": edge_mode_disabled_reason,
            "paletteInput": palette_mode,
            "bilateralRadius": config.bilateral_radius,
            "bilateralMode": config.bilateral_mode if config.bilateral_radius > 0 else None,
            "dither": config.dither,
            "ditherRequested": requested_dither,
            "ditherDisabledReason": dither_disabled_reason,
            "ditherScope": config.dither_scope if config.dither == "ordered" else None,
            "gridSnap": config.grid_snap_enabled,
            "gridSnapMethod": config.grid_snap_method if config.grid_snap_enabled else None,
            "gridQuantizeFirst": config.grid_snap_quantize_first if config.grid_snap_enabled else None,
            "gridAutoSize": grid_auto_size if config.grid_snap_enabled else False,
            "gridTopology": config.grid_snap_topology if config.grid_snap_enabled else None,
            "gridAxisStabilization": grid_axis_stabilization if config.grid_snap_enabled else None,
            "gridCutPath": grid_cut_path,
            "gridVariant": selected_grid_variant,
            "gridVariants": grid_variants[:9],
            "alphaPreserved": alpha_preserved,
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
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20viewBox%3D%220%200%2064%2064%22%3E%3Ctext%20x%3D%2232%22%20y%3D%2252%22%20text-anchor%3D%22middle%22%20font-size%3D%2256%22%3E%F0%9F%A7%AA%3C%2Ftext%3E%3C%2Fsvg%3E">
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
      --tooltip-layer: 2147483647;
      --tooltip-bg: #080d15;
      --tooltip-border: #dfe8ff;
      --tooltip-shell: #02050a;
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
    .topbar button {
      width: auto;
      min-width: 96px;
      min-height: 30px;
      padding: 0 12px;
      white-space: nowrap;
      touch-action: none;
    }
    .topbar button.active {
      background: #526386;
      border-color: #9da7bb;
      color: #f5f7fb;
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
    .viewer-stats-overlay {
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 2;
      max-width: calc(100% - 24px);
      color: #ffffff;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.35;
      white-space: normal;
      pointer-events: none;
      text-shadow: 0 1px 2px #000000, 0 0 5px #000000;
    }
    .viewer.viewer-empty .viewer-stats-overlay {
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      max-width: calc(100% - 48px);
      font-size: 36px;
      line-height: 1.1;
      font-weight: 800;
      text-align: center;
    }
    .viewer.viewer-empty #viewerDetectorLine {
      display: none;
    }
    #canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    #tooltipLayer {
      position: fixed;
      inset: 0;
      z-index: var(--tooltip-layer);
      pointer-events: none;
    }
    #tooltip {
      position: fixed;
      z-index: 2;
      width: 260px;
      height: 284px;
      padding: 4px;
      display: none;
      pointer-events: none;
      background: var(--tooltip-bg);
      border: 1px solid var(--tooltip-border);
      box-shadow: 0 0 0 2px var(--tooltip-shell);
    }
    #tooltip canvas {
      width: 250px;
      height: 250px;
      display: block;
      background: var(--tooltip-shell);
    }
    #tooltip .label {
      display: flex;
      justify-content: space-between;
      background: var(--tooltip-bg);
      color: #dce4f6;
      font-size: 11px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      margin-top: 3px;
      padding: 0 2px;
    }
    #uiTooltip {
      position: fixed;
      z-index: 3;
      display: none;
      max-width: 320px;
      padding: 9px 11px;
      pointer-events: none;
      background: var(--tooltip-bg);
      border: 1px solid var(--tooltip-border);
      border-radius: 6px;
      box-shadow: 0 0 0 2px var(--tooltip-shell);
      color: #eef3ff;
      font-size: 12px;
      line-height: 1.35;
      font-weight: 650;
    }
    #uiTooltip[data-visible="true"] {
      display: block;
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
    button:disabled {
      cursor: not-allowed;
      border-color: #30384a;
      background: #182033;
      color: var(--faint);
    }
    button:disabled:hover { background: #182033; }
    button.secondary {
      background: #202838;
      color: var(--accent-strong);
    }
    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .button-row.start-actions {
      grid-template-columns: 1fr 1fr 1fr;
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
    .section-note.notice {
      display: none;
      margin-top: -2px;
      color: var(--warn);
    }
    .section-note.notice[data-visible="true"] {
      display: block;
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
    .grid-summary {
      color: var(--text);
      font-size: 12px;
      line-height: 1.35;
      min-height: 18px;
    }
    .grid-variant-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .grid-variant-chip {
      width: auto;
      min-height: 26px;
      padding: 0 9px;
      border-radius: 6px;
      border: 1px solid #37445d;
      background: #111827;
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .grid-variant-chip.selected {
      background: #e8edf7;
      border-color: #f8fbff;
      color: #111827;
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
    .server-browser {
      position: fixed;
      inset: 24px;
      z-index: 30;
      display: none;
      grid-template-rows: auto 1fr;
      min-width: 0;
      min-height: 0;
      background: #0d121b;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      box-shadow: 0 16px 70px #000000cc;
    }
    .server-browser[data-visible="true"] {
      display: grid;
    }
    .server-browser-header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
    }
    .server-browser-title {
      min-width: 0;
    }
    .server-browser-title strong,
    .server-browser-title span {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .server-browser-title span {
      margin-top: 3px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    .server-browser-body {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(152px, 1fr));
      gap: 10px;
      align-content: start;
      min-height: 0;
      overflow: auto;
      padding: 8px;
    }
    .server-entry {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      width: 100%;
      min-height: 40px;
      margin: 0;
      text-align: left;
      border-color: #30394c;
      background: #121827;
    }
    .server-entry-folder {
      grid-column: 1 / -1;
    }
    .server-entry-file {
      grid-template-columns: 1fr;
      grid-template-rows: 108px minmax(34px, auto) auto;
      gap: 7px;
      align-items: stretch;
      min-height: 174px;
      padding: 8px;
    }
    .server-entry:hover {
      background: #1a2334;
    }
    .server-entry span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .server-entry-file span {
      display: -webkit-box;
      min-height: 34px;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      white-space: normal;
    }
    .server-entry small {
      color: var(--faint);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
      white-space: nowrap;
    }
    .server-thumbnail {
      width: 100%;
      height: 108px;
      object-fit: contain;
      border: 1px solid #252d3d;
      border-radius: 6px;
      background: #070a10;
    }
    .server-thumbnail-fallback {
      display: grid;
      place-items: center;
      width: 100%;
      height: 108px;
      border: 1px solid #252d3d;
      border-radius: 6px;
      color: var(--faint);
      background: #070a10;
      font-size: 12px;
    }
    .server-browser-empty {
      padding: 24px;
      color: var(--muted);
      text-align: center;
    }
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
          <input id="file" type="file" accept="image/*" data-tooltip="Load the source image for this local session.">
        </label>
        <div id="inputInfo" class="small">No image loaded yet.</div>
        <div class="button-row start-actions">
          <button id="resetZoom" class="secondary" data-tooltip="Return the preview to 100% zoom and recenter it.">Reset view</button>
          <button id="saveSettings" class="secondary" data-tooltip="Export all current controls as a JSON preset file.">Save settings</button>
          <button id="openFromServer" class="secondary" data-tooltip="Browse assets/generated on this server and open an image into this session.">Open from server</button>
        </div>
      </fieldset>

      <fieldset>
        <legend>My Presets</legend>
        <p class="section-note">Browser-local presets for your own tested combinations. Built-in experiment buttons are intentionally not included.</p>
        <div class="presets">
          <label>
            <span class="label-title">Preset <span class="field-hint">stored in this browser</span></span>
            <select id="customPresetSelect" data-tooltip="Choose a preset saved in this browser."></select>
          </label>
          <label>
            <span class="label-title">Name <span class="field-hint">required to save</span></span>
            <input id="presetName" type="text" placeholder="preset name" data-tooltip="Name for saving or replacing a custom preset.">
          </label>
          <div class="preset-actions">
            <button id="loadPreset" data-tooltip="Apply the selected preset to all controls.">Load</button>
            <button id="savePreset" data-tooltip="Save current controls under the typed name.">Save</button>
            <button id="deletePreset" data-tooltip="Delete the selected custom preset from this browser.">Delete</button>
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
            <input id="targetWidth" data-setting type="number" min="16" max="4096" value="1024" data-tooltip="Logical output width. Press Enter or leave the field to render.">
          </label>
          <label>
            <span class="label-title">Height <span class="field-hint">max 1024</span></span>
            <input id="targetHeight" data-setting type="number" min="16" max="1024" value="576" data-tooltip="Logical output height. Source-ratio default is capped at 1024.">
          </label>
        </div>
        <label class="check" data-tooltip="Keep output proportions equal to the imported source image.">
          <input id="aspectLock" data-setting type="checkbox" checked> lock source aspect
        </label>
        <div id="aspectInfo" class="small">Output size will use the source image ratio after import.</div>
        <div class="row">
          <label>
            <span class="label-title">Colors <span class="field-hint">palette cap</span></span>
            <input id="colors" data-setting type="number" min="2" max="1024" value="64" data-tooltip="Maximum colors in the output palette.">
          </label>
          <label>
            <span class="label-title">Dither <span class="field-hint">use sparingly</span></span>
            <select id="dither" data-setting data-tooltip="Adds controlled texture for gradients after palette mapping.">
              <option value="none">none</option>
              <option value="ordered">ordered Bayer</option>
              <option value="floyd">Floyd-Steinberg</option>
            </select>
          </label>
        </div>
        <label>
          <span class="label-title">Dither strength <span class="field-hint">0-64</span></span>
          <input id="ditherStrength" data-setting type="range" min="0" max="64" step="1" value="14" data-tooltip="Higher values make dithering more visible.">
        </label>
        <div class="row">
          <label>
            <span class="label-title">Dither scope <span class="field-hint">where it applies</span></span>
            <select id="ditherScope" data-setting data-tooltip="Adaptive mode suppresses dithering on edges and detailed objects.">
              <option value="global">global</option>
              <option value="adaptive" selected>adaptive smooth areas</option>
            </select>
          </label>
          <label>
            <span class="label-title">Error min <span class="field-hint">skip tiny errors</span></span>
            <input id="ditherErrorThreshold" data-setting type="number" min="0" max="255" step="0.5" value="3" data-tooltip="Minimum nearest-palette error before adaptive dithering is allowed.">
          </label>
        </div>
        <div class="row">
          <label>
            <span class="label-title">Edge max <span class="field-hint">protect details</span></span>
            <input id="ditherEdgeThreshold" data-setting type="number" min="0" max="1" step="0.01" value="0.28" data-tooltip="Adaptive dithering is blocked above this edge strength.">
          </label>
          <label>
            <span class="label-title">Luma range <span class="field-hint">smoothness gate</span></span>
            <input id="ditherLumaRange" data-setting type="number" min="0" max="255" step="1" value="45" data-tooltip="Adaptive dithering is allowed only in areas with local brightness variation under this value.">
          </label>
        </div>
        <div class="row">
          <label>
            <span class="label-title">Resample <span class="field-hint">resize filter</span></span>
            <select id="resample" data-setting data-tooltip="Resize filter used before palette conversion when Grid Snap is off.">
              <option value="box">box</option>
              <option value="bicubic">bicubic</option>
              <option value="lanczos">lanczos</option>
            </select>
          </label>
          <label>
            <span class="label-title">Color distance <span class="field-hint">palette mapping</span></span>
            <select id="colorDistance" data-setting data-tooltip="Distance metric for choosing the nearest palette color.">
              <option value="rgb" selected>weighted RGB</option>
              <option value="oklab">OKLab</option>
            </select>
          </label>
        </div>
      </fieldset>

      <fieldset>
        <legend>Hidden Grid</legend>
        <p class="section-note">Use when the source is enlarged AI pixel art. The detector estimates the source's pseudo-pixel cells and transfers them without averaging.</p>
        <label class="check" data-tooltip="Replace normal resizing with hidden-grid transfer for generated mixel art.">
          <input id="gridSnap" data-setting type="checkbox"> Grid Snap
        </label>
        <label class="check" data-tooltip="Estimate likely logical output sizes from source edge rhythms. Manual size still works when this is off.">
          <input id="gridAutoSize" data-setting type="checkbox" checked> auto size from detected mixels
        </label>
        <label class="check" data-tooltip="Reduce source colors with the active Palette Builder before cell voting so rare intended colors are not averaged away.">
          <input id="gridQuantizeFirst" data-setting type="checkbox"> quantize before grid vote
        </label>
        <div class="row">
          <label>
            <span class="label-title">Grid topology <span class="field-hint">cell cuts</span></span>
            <select id="gridTopology" data-setting data-tooltip="Uniform is the old phase-aligned grid. Elastic follows detected cell lines when AI mixels drift.">
              <option value="uniform">uniform legacy</option>
              <option value="elastic" selected>elastic lines</option>
            </select>
          </label>
          <label>
            <span class="label-title">Axis repair <span class="field-hint">line stability</span></span>
            <select id="gridAxisStabilization" data-setting data-tooltip="Repairs uneven detected cuts before grid transfer. Off keeps the raw detected lines.">
              <option value="off">off</option>
              <option value="conservative" selected>conservative</option>
              <option value="aggressive">aggressive</option>
            </select>
          </label>
        </div>
        <div class="row">
          <label>
            <span class="label-title">Cell reducer <span class="field-hint">pixel pick rule</span></span>
            <select id="gridSnapMethod" data-setting data-tooltip="How each detected source cell becomes one output pixel.">
              <option value="dark-stroke">dark-stroke bias</option>
              <option value="cell-mode">cell mode</option>
              <option value="center" selected>nearest center sample</option>
            </select>
          </label>
          <label id="gridDarkThresholdWrap">
            <span class="label-title">Dark threshold <span class="field-hint">stroke bias</span></span>
            <input id="gridDarkThreshold" data-setting type="number" min="0" max="255" step="1" value="38" data-tooltip="Minimum contrast for preserving a narrow dark stroke inside a detected cell.">
          </label>
        </div>
        <label>
          <span class="label-title">Auto variant <span class="field-hint">candidate grid</span></span>
          <input id="gridVariant" data-setting type="range" min="0" max="8" step="1" value="0" data-tooltip="Choose between detector candidates after auto-size render.">
        </label>
        <div id="gridInfo" class="grid-summary">Auto grid size is optional; manual Width/Height still works when it is off.</div>
        <div id="gridVariantList" class="grid-variant-list" aria-live="polite"></div>
      </fieldset>

      <fieldset>
        <legend>Palette Builder</legend>
        <p class="section-note">Choose how source colors are allocated before the image is mapped into the final palette.</p>
        <label>
          <span class="label-title">Strategy <span class="field-hint">slot allocation</span></span>
            <select id="paletteStrategy" data-setting data-tooltip="Palette extraction strategy. Projected modes choose source colors and remap the target into them.">
              <option value="median-cut">median-cut</option>
              <option value="kmeans">kmeans</option>
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
          <select id="paletteInput" data-setting data-tooltip="Which processed image supplies colors for palette extraction.">
            <option value="prepared" selected>prepared target</option>
            <option value="original">original resized</option>
            <option value="graded">graded target</option>
          </select>
        </label>
        <div class="row">
          <label>
            <span class="label-title">Accent weight <span class="field-hint">rare colors</span></span>
            <input id="accentPaletteWeight" data-setting type="number" min="0" max="12" step="0.1" value="0.8" data-tooltip="Extra palette pressure for saturated or visually interesting colors.">
          </label>
          <label>
            <span class="label-title">Hue rarity <span class="field-hint">unusual hues</span></span>
            <input id="hueRarityWeight" data-setting type="number" min="0" max="12" step="0.1" value="1.6" data-tooltip="Extra palette pressure for hues that occupy little image area.">
          </label>
        </div>
        <label>
          <span class="label-title">Hue match weight <span class="field-hint">mapping bias</span></span>
          <input id="hueMatchWeight" data-setting type="number" min="0" max="12" step="0.05" value="0.35" data-tooltip="Bias nearest-color mapping toward colors with a closer hue.">
        </label>
        <div class="row">
          <label>
            <span class="label-title">Interesting slots <span class="field-hint">reserved colors</span></span>
            <input id="interestingColorSlots" data-setting type="number" min="0" max="1024" value="0" data-tooltip="Palette slots reserved for colorful rare pixels before mass allocation.">
          </label>
          <label>
            <span class="label-title">Min saturation <span class="field-hint">slot gate</span></span>
            <input id="interestingMinSaturation" data-setting type="number" min="0" max="1" step="0.01" value="0.07" data-tooltip="Minimum saturation for interesting-color reservation.">
          </label>
        </div>
        <label>
          <span class="label-title">Min value <span class="field-hint">slot gate</span></span>
          <input id="interestingMinValue" data-setting type="number" min="0" max="1" step="0.01" value="0.05" data-tooltip="Minimum value/brightness for interesting-color reservation.">
        </label>
        <label>
          <span class="label-title">Protected hue ranges <span class="field-hint">example 250-330</span></span>
          <input id="protectedHueRanges" data-setting type="text" value="" placeholder="250-330,330-20" data-tooltip="Comma-separated hue ranges to protect, including wraparound ranges.">
        </label>
        <div class="row3">
          <label>
            <span class="label-title">Hue weight <span class="field-hint">boost</span></span>
            <input id="protectedHueWeight" data-setting type="number" min="0" max="20" step="0.1" value="0" data-tooltip="Weight boost for pixels inside protected hue ranges.">
          </label>
          <label>
            <span class="label-title">Hue slots <span class="field-hint">reserve</span></span>
            <input id="protectedHueSlots" data-setting type="number" min="0" max="1024" value="0" data-tooltip="Palette slots reserved for protected hue ranges.">
          </label>
          <label>
            <span class="label-title">Hue min sat <span class="field-hint">gate</span></span>
            <input id="protectedHueMinSaturation" data-setting type="number" min="0" max="1" step="0.01" value="0.08" data-tooltip="Minimum saturation for protected hue reservation.">
          </label>
        </div>
      </fieldset>

      <fieldset>
        <legend>Edges</legend>
        <p class="section-note">Edge masks protect contours and can push palette slots toward line-art pixels.</p>
        <div class="row">
          <label>
            <span class="label-title">Edge filter <span class="field-hint">mask source</span></span>
            <select id="edgeMode" data-setting data-tooltip="Filter used to detect contours for palette weighting and detail protection.">
              <option value="sobel" selected>Sobel</option>
              <option value="laplacian">Laplacian</option>
              <option value="highpass">High-pass</option>
              <option value="contour">Pillow contour</option>
              <option value="none">none</option>
            </select>
          </label>
          <label>
            <span class="label-title">Threshold <span class="field-hint">mask cutoff</span></span>
            <input id="edgeThreshold" data-setting type="number" min="0" max="1" step="0.005" value="0.04" data-tooltip="Lower values mark more pixels as edges.">
          </label>
        </div>
        <p id="edgeContourNotice" class="section-note notice">Pillow contour is unavailable while dithering is enabled. Use Sobel, Laplacian, or High-pass, or turn dithering off.</p>
        <div class="row">
          <label>
            <span class="label-title">Palette edge weight <span class="field-hint">slot boost</span></span>
            <input id="edgePaletteWeight" data-setting type="number" min="0" max="12" step="0.05" value="0.45" data-tooltip="Extra palette weight for edge pixels.">
          </label>
          <label>
            <span class="label-title">Edge sharpen <span class="field-hint">contour contrast</span></span>
            <input id="edgeSharpen" data-setting type="number" min="0" max="8" step="0.05" value="0" data-tooltip="Selective sharpening on detected contours before palette mapping.">
          </label>
        </div>
        <label class="check" data-tooltip="Return the edge mask in the render response for debugging.">
          <input id="includeEdgePreview" data-setting type="checkbox"> include edge mask in response
        </label>
      </fieldset>

      <fieldset>
        <legend>Color Prep</legend>
        <p class="section-note">Small source corrections before palette extraction. Keep smoothing off when one-pixel strokes matter.</p>
        <div class="row">
          <label>
            <span class="label-title">Saturation <span class="field-hint">color gain</span></span>
            <input id="saturation" data-setting type="number" min="0" max="8" step="0.05" value="1" data-tooltip="Color multiplier before palette extraction.">
          </label>
          <label>
            <span class="label-title">Contrast <span class="field-hint">tone gain</span></span>
            <input id="contrast" data-setting type="number" min="0" max="8" step="0.05" value="1" data-tooltip="Contrast multiplier before palette extraction.">
          </label>
        </div>
        <div class="row">
          <label>
            <span class="label-title">Sharpness <span class="field-hint">unsharp %</span></span>
            <input id="sharpness" data-setting type="number" min="0" max="500" step="5" value="0" data-tooltip="Unsharp-mask amount before quantization.">
          </label>
          <label>
            <span class="label-title">Autocontrast <span class="field-hint">cutoff %</span></span>
            <input id="autocontrastCutoff" data-setting type="number" min="0" max="30" step="0.001" value="0" data-tooltip="Trim extremes and stretch tonal range before conversion.">
          </label>
        </div>
        <div class="row">
          <label>
            <span class="label-title">Bilateral radius <span class="field-hint">smooth</span></span>
            <input id="bilateralRadius" data-setting type="number" min="0" max="8" value="0" data-tooltip="Edge-preserving smoothing radius. It does not cut palette directly, but can remove rare source colors before quantization.">
          </label>
          <label class="check" data-tooltip="When enabled, bilateral smoothing refuses to blur across contours, black strokes, and local detail jumps.">
            <input id="bilateralSafeEdges" data-setting type="checkbox" checked> Bilateral safe edges
          </label>
        </div>
        <div class="row">
          <label>
            <span class="label-title">Sigma color <span class="field-hint">range</span></span>
            <input id="bilateralSigmaColor" data-setting type="number" min="1" max="128" step="1" value="18" data-tooltip="How far colors may differ and still be smoothed together.">
          </label>
          <label>
            <span class="label-title">Sigma space <span class="field-hint">spread</span></span>
            <input id="bilateralSigmaSpace" data-setting type="number" min="0.1" max="16" step="0.1" value="1.4" data-tooltip="Spatial falloff for bilateral smoothing.">
          </label>
        </div>
        <label class="check" data-tooltip="Match the output's mean brightness back toward the resized source.">
          <input id="preserveLuma" data-setting type="checkbox"> preserve luma
        </label>
        <label class="check" data-tooltip="Match output saturation back toward the resized source to reduce washed-out accents.">
          <input id="preserveSaturation" data-setting type="checkbox"> preserve saturation
        </label>
        <label class="check" data-tooltip="Keep source transparency in the rendered PNG when the input has an alpha channel.">
          <input id="preserveAlpha" data-setting type="checkbox" checked> preserve alpha
        </label>
      </fieldset>

      <fieldset>
        <legend>Cleanup</legend>
        <p class="section-note">Optional post-pass for smooth areas and isolated near-color pixels. Use after palette choice is settled.</p>
        <div class="row">
          <label>
            <span class="label-title">Flat palette colors <span class="field-hint">0 off</span></span>
            <input id="flatRegionPaletteColors" data-setting type="number" min="0" max="1024" value="0" data-tooltip="Map low-detail non-edge areas to a small local palette.">
          </label>
          <label>
            <span class="label-title">Flat channel step <span class="field-hint">0 off</span></span>
            <input id="flatRegionChannelStep" data-setting type="number" min="0" max="255" value="0" data-tooltip="Snap low-detail channels to fixed RGB steps.">
          </label>
        </div>
        <div class="row">
          <label>
            <span class="label-title">Flat max sat <span class="field-hint">protect accents</span></span>
            <input id="flatRegionMaxSaturation" data-setting type="number" min="0" max="1" step="0.01" value="0.35" data-tooltip="Only low-saturation areas below this value can be flattened.">
          </label>
          <label>
            <span class="label-title">Flat edge max <span class="field-hint">protect lines</span></span>
            <input id="flatRegionEdgeThreshold" data-setting type="number" min="0" max="1" step="0.01" value="0.18" data-tooltip="Flattening is blocked above this edge strength.">
          </label>
        </div>
        <label>
          <span class="label-title">Flat luma range <span class="field-hint">smoothness</span></span>
          <input id="flatRegionLumaRange" data-setting type="number" min="0" max="255" step="1" value="10" data-tooltip="Maximum local brightness range eligible for flat-region cleanup.">
        </label>
        <div class="row3">
          <label>
            <span class="label-title">Mixel passes <span class="field-hint">0 off</span></span>
            <input id="mixelCleanupPasses" data-setting type="number" min="0" max="8" value="0" data-tooltip="Number of isolated-pixel cleanup passes.">
          </label>
          <label>
            <span class="label-title">Neighbors <span class="field-hint">3x3 vote</span></span>
            <input id="mixelCleanupMinNeighbors" data-setting type="number" min="1" max="9" value="3" data-tooltip="Minimum matching neighbors needed before replacing an isolated pixel.">
          </label>
          <label>
            <span class="label-title">Distance <span class="field-hint">near color</span></span>
            <input id="mixelCleanupDistance" data-setting type="number" min="0" max="255" step="1" value="18" data-tooltip="Maximum color distance allowed for mixel replacement.">
          </label>
        </div>
        <label>
          <span class="label-title">Mixel max sat <span class="field-hint">protect accents</span></span>
          <input id="mixelCleanupMaxSaturation" data-setting type="number" min="0" max="1" step="0.01" value="0.45" data-tooltip="Pixels above this saturation are protected from mixel cleanup.">
        </label>
      </fieldset>

      <fieldset>
        <legend>Export</legend>
        <p class="section-note">Swatches show the actual palette returned by the latest output.</p>
        <div id="palette" class="palette"></div>
      </fieldset>
    </aside>

    <main>
      <div class="topbar">
        <span id="status" class="status-pill status-ok">waiting for image</span>
        <span id="zoomInfo" class="metric">zoom 100%</span>
        <button id="holdBefore" class="secondary" data-tooltip="Hold to draw the imported original over the output with matching pan and zoom.">Hold Before</button>
        <button id="saveInPlace" data-tooltip="Save the current output over the server file that was opened in this session.">Save</button>
        <button id="savePng" data-tooltip="Download the latest rendered output as a PNG.">Save As</button>
        <span class="spacer"></span>
        <span class="small">Wheel zooms at cursor. Drag pans. Hold Before or Z to compare.</span>
      </div>
      <div id="viewer" class="viewer viewer-empty">
        <canvas id="canvas"></canvas>
        <div id="viewerStatsOverlay" class="viewer-stats-overlay" aria-live="polite">
          <div id="viewerDetectorLine"></div>
          <div id="viewerStatsLine">load an image</div>
        </div>
      </div>
    </main>
  </div>
  <div id="serverBrowser" class="server-browser" aria-hidden="true">
    <div class="server-browser-header">
      <div class="server-browser-title">
        <strong>Open from server</strong>
        <span id="serverBrowserPath">assets/generated</span>
      </div>
      <button id="closeServerBrowser" class="secondary">Close</button>
    </div>
    <div id="serverBrowserBody" class="server-browser-body"></div>
  </div>
  <div id="tooltipLayer">
    <div id="tooltip">
      <canvas id="tipCanvas" width="250" height="250"></canvas>
      <div class="label"><span>original full-res</span><span>output</span></div>
    </div>
    <div id="uiTooltip" role="tooltip" aria-hidden="true"></div>
  </div>

  <script>
    const MIN_OUTPUT_SIZE = 16;
    const MAX_OUTPUT_HEIGHT = 1024;
    const MAX_OUTPUT_WIDTH = 4096;
    const VIEW_FIT_MIN_MARGIN = 16;
    const VIEW_FIT_MAX_MARGIN = 48;
    const PRESET_STORAGE_KEY = 'pixel-art-lab-custom-presets-v1';
    const APP_BASE_PATH = window.location.pathname.endsWith('/')
      ? window.location.pathname
      : window.location.pathname.replace(/[^/]*$/, '');
    let defaultSettings = null;

    const state = {
      imageLoaded: false,
      sourceWidth: 0,
      sourceHeight: 0,
      sourceAspect: 0,
      sourceName: '',
      sourcePath: null,
      saveTargetPath: null,
      canSaveInPlace: false,
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
      beforeDown: false,
      renderTimer: 0,
      renderSeq: 0,
      activeController: null,
      pendingFitOnRender: false,
      syncingDimensions: false,
    };

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const viewer = document.getElementById('viewer');
    const tooltip = document.getElementById('tooltip');
    const uiTooltip = document.getElementById('uiTooltip');
    const tipCanvas = document.getElementById('tipCanvas');
    const tipCtx = tipCanvas.getContext('2d');
    const statusEl = document.getElementById('status');
    const zoomInfo = document.getElementById('zoomInfo');
    const holdBeforeButton = document.getElementById('holdBefore');
    const viewerDetectorLine = document.getElementById('viewerDetectorLine');
    const viewerStatsLine = document.getElementById('viewerStatsLine');
    const paletteEl = document.getElementById('palette');
    const aspectInfo = document.getElementById('aspectInfo');
    const gridInfo = document.getElementById('gridInfo');
    const gridVariantList = document.getElementById('gridVariantList');
    const customPresetSelect = document.getElementById('customPresetSelect');
    const presetNameInput = document.getElementById('presetName');
    const saveInPlaceButton = document.getElementById('saveInPlace');
    const serverBrowser = document.getElementById('serverBrowser');
    const serverBrowserBody = document.getElementById('serverBrowserBody');
    const serverBrowserPath = document.getElementById('serverBrowserPath');
    let activeUiTooltipTarget = null;

    function setStatus(text, cls) {
      statusEl.className = `status-pill ${cls || ''}`;
      statusEl.textContent = text;
    }

    function settingEl(id) { return document.getElementById(id); }

    function appPath(path) {
      const cleanPath = String(path || '').replace(/^\/+/, '');
      return `${APP_BASE_PATH}${cleanPath}`;
    }

    function updateSaveInPlaceAvailability() {
      const enabled = Boolean(state.outputDataUrl && state.canSaveInPlace && state.saveTargetPath);
      saveInPlaceButton.disabled = !enabled;
      saveInPlaceButton.dataset.tooltip = enabled
        ? `Save over assets/generated/${state.saveTargetPath}.`
        : 'Open an image from server and render it before using Save.';
    }

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

    function numberSetting(id) {
      const value = Number(settingEl(id).value);
      return Number.isFinite(value) ? value : 0;
    }

    function settingControls() {
      return Array.from(document.querySelectorAll('[data-setting]'));
    }

    function readSettingValue(el) {
      if (el.type === 'checkbox') return el.checked;
      if (el.type === 'number' || el.type === 'range') {
        const value = Number(el.value);
        return Number.isFinite(value) ? value : 0;
      }
      return el.value;
    }

    function coerceSettingValue(el, value) {
      if (el.type === 'checkbox') return Boolean(value);
      if (el.type === 'number' || el.type === 'range') {
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric : readSettingValue(el);
      }
      return value === undefined || value === null ? '' : String(value);
    }

    function writeSettingValue(el, value) {
      const coerced = coerceSettingValue(el, value);
      if (el.type === 'checkbox') el.checked = coerced;
      else el.value = coerced;
    }

    function captureDefaultSettings() {
      const defaults = {};
      settingControls().forEach((el) => {
        defaults[el.id] = readSettingValue(el);
      });
      return defaults;
    }

    function normalizeSettings(settings = {}) {
      const normalized = {};
      const source = settings && typeof settings === 'object' ? settings : {};
      settingControls().forEach((el) => {
        let value = defaultSettings ? defaultSettings[el.id] : readSettingValue(el);
        if (Object.prototype.hasOwnProperty.call(source, el.id)) {
          value = source[el.id];
        } else if (
          el.id === 'bilateralSafeEdges' &&
          Object.prototype.hasOwnProperty.call(source, 'bilateralMode')
        ) {
          value = source.bilateralMode === 'edge-safe';
        }
        normalized[el.id] = coerceSettingValue(el, value);
      });
      return normalized;
    }

    function setControlDisabled(id, disabled, disabledTitle) {
      const input = settingEl(id);
      if (!input) return;
      if (input.dataset.enabledTooltip === undefined) input.dataset.enabledTooltip = input.dataset.tooltip || '';
      input.disabled = disabled;
      if (disabled) input.dataset.tooltip = disabledTitle;
      else if (input.dataset.enabledTooltip) input.dataset.tooltip = input.dataset.enabledTooltip;
      else delete input.dataset.tooltip;
      const label = input.closest('label');
      if (label) {
        if (label.dataset.enabledTooltip === undefined) label.dataset.enabledTooltip = label.dataset.tooltip || '';
        if (disabled) label.dataset.tooltip = disabledTitle;
        else if (label.dataset.enabledTooltip) label.dataset.tooltip = label.dataset.enabledTooltip;
        else delete label.dataset.tooltip;
        label.classList.toggle('disabled', disabled);
      }
      if (activeUiTooltipTarget === input || activeUiTooltipTarget === label) hideUiTooltip();
    }

    function tooltipTargetFrom(node) {
      return node instanceof Element ? node.closest('[data-tooltip]') : null;
    }

    function showUiTooltip(target, clientX, clientY) {
      const text = target && target.dataset ? target.dataset.tooltip : '';
      if (!text) {
        hideUiTooltip();
        return;
      }
      activeUiTooltipTarget = target;
      uiTooltip.textContent = text;
      uiTooltip.setAttribute('data-visible', 'true');
      uiTooltip.setAttribute('aria-hidden', 'false');
      positionUiTooltip(clientX, clientY);
    }

    function positionUiTooltip(clientX, clientY) {
      if (!activeUiTooltipTarget) return;
      const pad = 12;
      const gap = 14;
      let left = clientX + gap;
      let top = clientY + gap;
      const rect = uiTooltip.getBoundingClientRect();
      if (left + rect.width + pad > window.innerWidth) left = clientX - rect.width - gap;
      if (top + rect.height + pad > window.innerHeight) top = clientY - rect.height - gap;
      left = Math.max(pad, Math.min(window.innerWidth - rect.width - pad, left));
      top = Math.max(pad, Math.min(window.innerHeight - rect.height - pad, top));
      uiTooltip.style.left = `${left}px`;
      uiTooltip.style.top = `${top}px`;
    }

    function hideUiTooltip() {
      activeUiTooltipTarget = null;
      uiTooltip.removeAttribute('data-visible');
      uiTooltip.setAttribute('aria-hidden', 'true');
    }

    document.addEventListener('pointerover', (evt) => {
      const target = tooltipTargetFrom(evt.target);
      if (target) showUiTooltip(target, evt.clientX, evt.clientY);
    });

    document.addEventListener('pointermove', (evt) => {
      if (activeUiTooltipTarget) positionUiTooltip(evt.clientX, evt.clientY);
    });

    document.addEventListener('pointerout', (evt) => {
      if (!activeUiTooltipTarget) return;
      const relatedTarget = evt.relatedTarget instanceof Element ? evt.relatedTarget : null;
      if (!relatedTarget || !activeUiTooltipTarget.contains(relatedTarget)) hideUiTooltip();
    });

    document.addEventListener('focusin', (evt) => {
      const target = tooltipTargetFrom(evt.target);
      if (!target) return;
      const rect = target.getBoundingClientRect();
      showUiTooltip(target, rect.left + rect.width / 2, rect.bottom);
    });

    document.addEventListener('focusout', hideUiTooltip);
    document.addEventListener('scroll', hideUiTooltip, true);

    function syncConditionalControls() {
      let dither = settingEl('dither').value;
      const edgeModeEl = settingEl('edgeMode');
      let edgeMode = edgeModeEl.value;
      const gridSnap = settingEl('gridSnap').checked;
      const gridAutoSize = settingEl('gridAutoSize').checked;
      const gridQuantizeFirst = settingEl('gridQuantizeFirst').checked;
      const gridTopology = settingEl('gridTopology').value;
      const gridMethod = settingEl('gridSnapMethod').value;
      const protectedHueActive = settingEl('protectedHueRanges').value.trim().length > 0;
      const bilateralActive = numberSetting('bilateralRadius') > 0;
      const ditherBlockedByGridVote = gridSnap && gridQuantizeFirst;
      const ditherEl = settingEl('dither');
      if (ditherBlockedByGridVote && dither !== 'none') {
        ditherEl.dataset.blockedValue = dither;
        ditherEl.value = 'none';
        dither = 'none';
      } else if (!ditherBlockedByGridVote && ditherEl.dataset.blockedValue) {
        if (dither === 'none') {
          ditherEl.value = ditherEl.dataset.blockedValue;
          dither = ditherEl.value;
        }
        delete ditherEl.dataset.blockedValue;
      }
      const contourOption = Array.from(edgeModeEl.options).find((option) => option.value === 'contour');
      const contourBlockedByDither = dither !== 'none';
      if (contourOption) {
        contourOption.disabled = contourBlockedByDither;
        contourOption.textContent = contourBlockedByDither ? 'Pillow contour (dither off only)' : 'Pillow contour';
      }
      if (contourBlockedByDither && edgeMode === 'contour') {
        edgeModeEl.dataset.blockedValue = 'contour';
        edgeModeEl.value = 'sobel';
        edgeMode = 'sobel';
      } else if (!contourBlockedByDither && edgeModeEl.dataset.blockedValue === 'contour') {
        if (edgeMode === 'sobel') {
          edgeModeEl.value = 'contour';
          edgeMode = 'contour';
        }
        delete edgeModeEl.dataset.blockedValue;
      }
      const edgeContourNotice = document.getElementById('edgeContourNotice');
      if (edgeContourNotice) {
        edgeContourNotice.dataset.visible = contourBlockedByDither ? 'true' : 'false';
      }
      const orderedDither = dither === 'ordered';
      const adaptiveDither = orderedDither && settingEl('ditherScope').value === 'adaptive';
      const edgeActive = edgeMode !== 'none';
      const flatCleanupActive =
        edgeActive &&
        (numberSetting('flatRegionPaletteColors') > 0 || numberSetting('flatRegionChannelStep') > 1);
      const mixelCleanupActive = numberSetting('mixelCleanupPasses') > 0;

      setControlDisabled(
        'dither',
        ditherBlockedByGridVote,
        'Unavailable while Grid Snap quantize before grid vote is enabled.'
      );
      setControlDisabled(
        'ditherStrength',
        ditherBlockedByGridVote || !orderedDither,
        ditherBlockedByGridVote ? 'Dither is unavailable while quantize before grid vote is enabled.' : 'Used only by ordered Bayer dithering.'
      );
      setControlDisabled(
        'ditherScope',
        ditherBlockedByGridVote || !orderedDither,
        ditherBlockedByGridVote ? 'Dither is unavailable while quantize before grid vote is enabled.' : 'Used only by ordered Bayer dithering.'
      );
      setControlDisabled(
        'ditherErrorThreshold',
        ditherBlockedByGridVote || !adaptiveDither,
        ditherBlockedByGridVote ? 'Dither is unavailable while quantize before grid vote is enabled.' : 'Used only by adaptive ordered dithering.'
      );
      setControlDisabled(
        'ditherLumaRange',
        ditherBlockedByGridVote || !adaptiveDither,
        ditherBlockedByGridVote ? 'Dither is unavailable while quantize before grid vote is enabled.' : 'Used only by adaptive ordered dithering.'
      );
      setControlDisabled(
        'ditherEdgeThreshold',
        ditherBlockedByGridVote || !(adaptiveDither && edgeActive),
        ditherBlockedByGridVote
          ? 'Dither is unavailable while quantize before grid vote is enabled.'
          : (edgeActive ? 'Used only by adaptive ordered dithering.' : 'Requires an edge filter other than none.')
      );

      setControlDisabled('resample', gridSnap, 'Grid Snap replaces normal resizing and does not use this resample filter.');
      setControlDisabled('gridAutoSize', !gridSnap, 'Used only when Grid Snap is enabled.');
      setControlDisabled('gridQuantizeFirst', !gridSnap, 'Used only when Grid Snap is enabled.');
      setControlDisabled('gridTopology', !gridSnap, 'Used only when Grid Snap is enabled.');
      setControlDisabled(
        'gridAxisStabilization',
        !(gridSnap && gridTopology === 'elastic'),
        gridSnap ? 'Used only by elastic grid lines.' : 'Used only when Grid Snap is enabled.'
      );
      setControlDisabled('gridSnapMethod', !gridSnap, 'Used only when Grid Snap is enabled.');
      setControlDisabled('gridVariant', !(gridSnap && gridAutoSize), 'Used only when Grid Snap auto size is enabled.');
      setControlDisabled(
        'gridDarkThreshold',
        !(gridSnap && gridMethod === 'dark-stroke'),
        gridSnap ? 'Used only by the dark-stroke bias cell reducer.' : 'Used only when Grid Snap is enabled.'
      );

      setControlDisabled('edgeThreshold', !edgeActive, 'Used only when an edge filter is selected.');
      setControlDisabled('edgePaletteWeight', !edgeActive, 'Requires an edge filter other than none.');
      setControlDisabled('edgeSharpen', !edgeActive, 'Requires an edge filter other than none.');
      setControlDisabled('includeEdgePreview', !edgeActive, 'Requires an edge filter other than none.');

      setControlDisabled('protectedHueWeight', !protectedHueActive, 'Add protected hue ranges first.');
      setControlDisabled('protectedHueSlots', !protectedHueActive, 'Add protected hue ranges first.');
      setControlDisabled('protectedHueMinSaturation', !protectedHueActive, 'Add protected hue ranges first.');
      setControlDisabled(
        'interestingColorSlots',
        settingEl('paletteStrategy').value !== 'interesting',
        'Used only by the interesting palette strategy.'
      );

      setControlDisabled('bilateralSafeEdges', !bilateralActive, 'Used only when Bilateral radius is above 0.');
      setControlDisabled('bilateralSigmaColor', !bilateralActive, 'Used only when Bilateral radius is above 0.');
      setControlDisabled('bilateralSigmaSpace', !bilateralActive, 'Used only when Bilateral radius is above 0.');

      setControlDisabled('flatRegionPaletteColors', !edgeActive, 'Flat cleanup requires an edge filter.');
      setControlDisabled('flatRegionChannelStep', !edgeActive, 'Flat cleanup requires an edge filter.');
      setControlDisabled('flatRegionMaxSaturation', !flatCleanupActive, 'Enable flat palette colors or flat channel step first.');
      setControlDisabled('flatRegionEdgeThreshold', !flatCleanupActive, 'Enable flat palette colors or flat channel step first.');
      setControlDisabled('flatRegionLumaRange', !flatCleanupActive, 'Enable flat palette colors or flat channel step first.');

      setControlDisabled('mixelCleanupMinNeighbors', !mixelCleanupActive, 'Used only when Mixel passes is above 0.');
      setControlDisabled('mixelCleanupDistance', !mixelCleanupActive, 'Used only when Mixel passes is above 0.');
      setControlDisabled('mixelCleanupMaxSaturation', !mixelCleanupActive, 'Used only when Mixel passes is above 0.');
    }

    function collectSettings() {
      const settings = {};
      settingControls().forEach((el) => {
        settings[el.id] = readSettingValue(el);
      });
      const ditherEl = settingEl('dither');
      if (ditherEl && ditherEl.dataset.blockedValue) {
        settings.dither = ditherEl.dataset.blockedValue;
      }
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
      const normalized = normalizeSettings(settings);
      const ditherEl = settingEl('dither');
      if (ditherEl) delete ditherEl.dataset.blockedValue;
      settingControls().forEach((el) => {
        writeSettingValue(el, normalized[el.id]);
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
      store[name] = normalizeSettings(collectSettings());
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

    function fitViewMargin() {
      const { w, h } = viewportSize();
      return clampValue(Math.round(Math.min(w, h) * 0.04), VIEW_FIT_MIN_MARGIN, VIEW_FIT_MAX_MARGIN);
    }

    function minZoomForCurrentImage() {
      if (!state.width || !state.height) return 1;
      const { w, h } = viewportSize();
      const margin = fitViewMargin();
      const availableW = Math.max(1, w - margin * 2);
      const availableH = Math.max(1, h - margin * 2);
      const fitZoom = Math.min(availableW / state.width, availableH / state.height);
      return clampValue(Math.min(1, fitZoom), 0.02, 1);
    }

    function clampZoom(value) {
      return clampValue(value, minZoomForCurrentImage(), 16);
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
    }

    function draw() {
      const { w, h } = viewportSize();
      ctx.imageSmoothingEnabled = state.zoom < 1;
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
      if (state.beforeDown) {
        drawOriginalOverlay();
      }
      zoomInfo.textContent = `zoom ${Math.round(state.zoom * 100)}%`;
      if (state.zDown && state.hovering) drawTooltip();
    }

    function drawOriginalOverlay() {
      if (!state.originalImg || !state.width || !state.height) return;
      const originalRect = originalCropRect();
      ctx.save();
      ctx.imageSmoothingEnabled = state.zoom < 1;
      ctx.drawImage(
        state.originalImg,
        originalRect.x,
        originalRect.y,
        originalRect.w,
        originalRect.h,
        state.offsetX,
        state.offsetY,
        state.width * state.zoom,
        state.height * state.zoom
      );
      ctx.restore();
      ctx.imageSmoothingEnabled = state.zoom < 1;
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
      const nextZoom = clampZoom(state.zoom * factor);
      state.zoom = nextZoom;
      state.offsetX = point.x - before.x * state.zoom;
      state.offsetY = point.y - before.y * state.zoom;
      clampPan();
      draw();
    }, { passive: false });

    canvas.addEventListener('pointerdown', (evt) => {
      if (!state.outputImg) return;
      hideUiTooltip();
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

    function setBeforeDown(active) {
      if (state.beforeDown === active) return;
      state.beforeDown = active;
      holdBeforeButton.classList.toggle('active', active);
      draw();
    }

    holdBeforeButton.addEventListener('pointerdown', (evt) => {
      evt.preventDefault();
      if (!state.outputImg || !state.originalImg) return;
      holdBeforeButton.setPointerCapture(evt.pointerId);
      setBeforeDown(true);
    });

    holdBeforeButton.addEventListener('pointerup', (evt) => {
      try { holdBeforeButton.releasePointerCapture(evt.pointerId); } catch (_err) {}
      setBeforeDown(false);
    });

    holdBeforeButton.addEventListener('pointercancel', () => {
      setBeforeDown(false);
    });

    holdBeforeButton.addEventListener('lostpointercapture', () => {
      setBeforeDown(false);
    });

    window.addEventListener('blur', () => {
      setBeforeDown(false);
      hideUiTooltip();
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

    async function getJson(url, signal) {
      const response = await fetch(url, { signal });
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

    function formatGridResolution(item) {
      return `${item.width}x${item.height}`;
    }

    function applyLoadedImage(result, originalImg, originalDataUrl) {
      state.imageLoaded = true;
      state.sourceName = result.name || '';
      state.sourcePath = result.sourcePath || null;
      state.saveTargetPath = result.saveTargetPath || null;
      state.canSaveInPlace = Boolean(result.canSaveInPlace && result.saveTargetPath);
      state.originalImg = originalImg;
      state.originalDataUrl = originalDataUrl;
      state.outputDataUrl = null;
      state.outputImg = null;
      updateSaveInPlaceAvailability();
      applySourceOutputSize(result.width, result.height, result.targetWidth, result.targetHeight);
      const label = state.sourcePath ? `assets/generated/${state.sourcePath}` : result.name;
      document.getElementById('inputInfo').textContent = `${label}: ${result.width}x${result.height}`;
      setStatus('image loaded', 'status-ok');
      state.zoom = 1;
      state.offsetX = 0;
      state.offsetY = 0;
      state.pendingFitOnRender = true;
      setBeforeDown(false);
      scheduleRender(40);
    }

    function showServerBrowser() {
      serverBrowser.dataset.visible = 'true';
      serverBrowser.setAttribute('aria-hidden', 'false');
      loadServerFolder('');
    }

    function hideServerBrowser() {
      serverBrowser.removeAttribute('data-visible');
      serverBrowser.setAttribute('aria-hidden', 'true');
    }

    function formatBytes(value) {
      const bytes = Number(value) || 0;
      if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
      if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
      return `${bytes} B`;
    }

    function serverThumbnailUrl(entry) {
      const version = Number(entry.mtime) || Date.now();
      return `${appPath('api/server/thumbnail')}?path=${encodeURIComponent(entry.path)}&v=${version}`;
    }

    function createServerFolderEntry(label, meta, onClick) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'server-entry server-entry-folder';
      const name = document.createElement('span');
      name.textContent = label;
      const metaNode = document.createElement('small');
      metaNode.textContent = meta;
      button.append(name, metaNode);
      button.addEventListener('click', onClick);
      return button;
    }

    function createServerFileEntry(entry) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'server-entry server-entry-file';
      const thumbnail = document.createElement('img');
      thumbnail.className = 'server-thumbnail';
      thumbnail.src = serverThumbnailUrl(entry);
      thumbnail.alt = entry.name;
      thumbnail.loading = 'lazy';
      thumbnail.decoding = 'async';
      thumbnail.addEventListener('error', () => {
        const fallback = document.createElement('div');
        fallback.className = 'server-thumbnail-fallback';
        fallback.textContent = 'preview unavailable';
        thumbnail.replaceWith(fallback);
      }, { once: true });
      const name = document.createElement('span');
      name.textContent = entry.name;
      const meta = document.createElement('small');
      meta.textContent = formatBytes(entry.size);
      button.append(thumbnail, name, meta);
      button.addEventListener('click', () => openServerFile(entry.path));
      return button;
    }

    async function loadServerFolder(path) {
      try {
        setStatus('loading files...', 'status-busy');
        const data = await getJson(`${appPath('api/server/files')}?path=${encodeURIComponent(path || '')}`);
        serverBrowserPath.textContent = data.path ? `assets/generated/${data.path}` : 'assets/generated';
        serverBrowserBody.innerHTML = '';
        if (data.path) {
          serverBrowserBody.appendChild(createServerFolderEntry('..', 'folder', () => loadServerFolder(data.parent || '')));
        }
        const entries = Array.isArray(data.entries) ? data.entries : [];
        if (!entries.length && !data.path) {
          const empty = document.createElement('div');
          empty.className = 'server-browser-empty';
          empty.textContent = 'No supported images found in assets/generated.';
          serverBrowserBody.appendChild(empty);
        }
        entries.forEach((entry) => {
          if (entry.type === 'dir') {
            serverBrowserBody.appendChild(createServerFolderEntry(entry.name, 'folder', () => loadServerFolder(entry.path)));
          } else {
            serverBrowserBody.appendChild(createServerFileEntry(entry));
          }
        });
        setStatus('files loaded', 'status-ok');
      } catch (err) {
        setStatus(err.message || String(err), 'status-error');
      }
    }

    async function openServerFile(path) {
      try {
        setStatus('opening server file...', 'status-busy');
        const imageUrl = `${appPath('api/server/file')}?path=${encodeURIComponent(path)}&v=${Date.now()}`;
        const [result, originalImg] = await Promise.all([
          postJson(appPath('api/server/open'), { path }),
          loadImageUrl(imageUrl),
        ]);
        applyLoadedImage(result, originalImg, imageUrl);
        hideServerBrowser();
      } catch (err) {
        setStatus(err.message || String(err), 'status-error');
      }
    }

    function selectedGridVariantIndex(variants, selected) {
      if (!selected) return 0;
      const index = variants.findIndex((item) =>
        Number(item.width) === Number(selected.width) &&
        Number(item.height) === Number(selected.height) &&
        String(item.sourceAxis || '') === String(selected.sourceAxis || '')
      );
      return index >= 0 ? index : 0;
    }

    function renderGridVariantList(variants, selectedIndex) {
      gridVariantList.innerHTML = '';
      if (!variants.length) return;
      variants.forEach((item, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `grid-variant-chip${index === selectedIndex ? ' selected' : ''}`;
        button.textContent = formatGridResolution(item);
        button.dataset.tooltip = index === selectedIndex ? 'Selected auto grid size.' : 'Use this auto grid size.';
        button.addEventListener('click', () => {
          const slider = settingEl('gridVariant');
          if (Number(slider.value) === index) return;
          slider.value = String(index);
          scheduleRender(0);
        });
        gridVariantList.appendChild(button);
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
        const result = await postJson(appPath('api/render'), { settings: collectSettings() }, controller.signal);
        if (seq !== state.renderSeq) return;
        const outputImg = await loadImageUrl(result.output);
        state.outputImg = outputImg;
        state.outputDataUrl = result.output;
        state.width = result.width;
        state.height = result.height;
        updateSaveInPlaceAvailability();
        if (state.pendingFitOnRender) {
          state.zoom = minZoomForCurrentImage();
          state.pendingFitOnRender = false;
        } else {
          state.zoom = clampZoom(state.zoom);
        }
        clampPan();
        viewer.classList.remove('viewer-empty');
        draw();
        drawPalette(result.palette || []);
        const s = result.stats || {};
        const variants = Array.isArray(s.gridVariants) ? s.gridVariants : [];
        settingEl('gridVariant').max = Math.max(0, variants.length - 1);
        let selectedVariantIndex = 0;
        if (s.gridAutoSize && s.gridVariant) {
          selectedVariantIndex = selectedGridVariantIndex(variants, s.gridVariant);
          settingEl('gridVariant').value = String(selectedVariantIndex);
          setDimensionInputs(state.width, state.height);
        }
        if (s.gridSnap && s.gridVariant) {
          viewerDetectorLine.textContent = 'Auto grid';
          viewerStatsLine.textContent = `${state.width}x${state.height}`;
          gridInfo.textContent = `Selected ${state.width}x${state.height}`;
          renderGridVariantList(variants, selectedVariantIndex);
        } else if (settingEl('gridSnap').checked) {
          viewerDetectorLine.textContent = 'Manual grid';
          viewerStatsLine.textContent = `${state.width}x${state.height}`;
          gridInfo.textContent = `Manual size ${state.width}x${state.height}`;
          renderGridVariantList([], 0);
        } else {
          viewerDetectorLine.textContent = 'Output';
          viewerStatsLine.textContent = `${state.width}x${state.height}`;
          gridInfo.textContent = `Output size ${state.width}x${state.height}`;
          renderGridVariantList([], 0);
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
        div.dataset.tooltip = color;
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
        syncConditionalControls();
        if (el.id === 'aspectLock' && el.checked) syncLockedDimensions(settingEl('aspectDriver').value || 'height');
        scheduleRender();
      });
      el.addEventListener('change', () => {
        if (el.id === 'targetWidth') syncLockedDimensions('width');
        else if (el.id === 'targetHeight') syncLockedDimensions('height');
        else if (el.id === 'aspectLock' && el.checked) syncLockedDimensions(settingEl('aspectDriver').value || 'height');
        syncConditionalControls();
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

    function saveCurrentPng() {
      if (!state.outputDataUrl) return;
      const link = document.createElement('a');
      link.href = state.outputDataUrl;
      link.download = `${pixelLabOutputBaseName(state.sourceName)}_PIXEL_LAB.png`;
      link.click();
    }

    async function saveCurrentInPlace() {
      if (!state.outputDataUrl || !state.canSaveInPlace || !state.saveTargetPath) {
        setStatus('open a server image first', 'status-error');
        return;
      }
      try {
        setStatus('saving...', 'status-busy');
        const result = await postJson(appPath('api/save'), { data: state.outputDataUrl });
        state.sourcePath = result.sourcePath || result.path || state.sourcePath;
        state.saveTargetPath = result.saveTargetPath || result.path || state.saveTargetPath;
        state.canSaveInPlace = Boolean(state.saveTargetPath);
        updateSaveInPlaceAvailability();
        setStatus(`saved ${result.width}x${result.height}`, 'status-ok');
      } catch (err) {
        setStatus(err.message || String(err), 'status-error');
      }
    }

    function pixelLabOutputBaseName(sourceName) {
      const rawName = String(sourceName || '').split(/[\\/]/).pop() || 'pixel-art';
      const withoutExtension = rawName.replace(/\.[^.]*$/, '') || rawName;
      const cleaned = withoutExtension
        .replace(/[<>:"/\\|?*\x00-\x1f]+/g, '_')
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/[. ]+$/g, '');
      return cleaned || 'pixel-art';
    }

    document.getElementById('saveInPlace').addEventListener('click', saveCurrentInPlace);
    document.getElementById('savePng').addEventListener('click', saveCurrentPng);
    document.getElementById('openFromServer').addEventListener('click', showServerBrowser);
    document.getElementById('closeServerBrowser').addEventListener('click', hideServerBrowser);

    document.getElementById('saveSettings').addEventListener('click', () => {
      const data = 'data:application/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(normalizeSettings(collectSettings()), null, 2));
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
            postJson(appPath('api/image'), {
              name: file.name,
              data: originalDataUrl,
            }),
            loadImageUrl(originalDataUrl),
          ]);
          applyLoadedImage(result, originalImg, originalDataUrl);
        } catch (err) {
          setStatus(err.message || String(err), 'status-error');
        }
      };
      reader.readAsDataURL(file);
    });

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    defaultSettings = captureDefaultSettings();
    syncConditionalControls();
    refreshPresetSelect();
    updateSaveInPlaceAvailability();
  </script>
</body>
</html>
"""


class LabHandler(BaseHTTPRequestHandler):
    server_version = "PixelArtLab/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))

    def current_session_id(self) -> str:
        if hasattr(self, "_pixel_lab_session_id"):
            return self._pixel_lab_session_id
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        raw_cookie = cookie.get(SESSION_COOKIE_NAME)
        session_id = sanitize_session_id(raw_cookie.value if raw_cookie else None)
        self._pixel_lab_session_id = session_id
        return session_id

    def current_state(self) -> LabState:
        return get_session(self.current_session_id())

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_UPLOAD_BYTES * 2:
            raise ValueError("request body is too large")
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def send_bytes(
        self,
        content: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Set-Cookie",
            f"{SESSION_COOKIE_NAME}={self.current_session_id()}; Path=/; HttpOnly; SameSite=Lax",
        )
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
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
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            state = self.current_state()
            with state.lock:
                image = state.image
                save_target_path = pixel_lab_save_path(state.source_path) if state.source_path else None
                payload = {
                    "loaded": image is not None,
                    "name": state.name,
                    "width": image.width if image else 0,
                    "height": image.height if image else 0,
                    "sourcePath": server_relative_path(state.source_path) if state.source_path else None,
                    "saveTargetPath": server_relative_path(save_target_path) if save_target_path else None,
                    "canSaveInPlace": save_target_path is not None,
                }
            self.send_json(payload)
            return
        if path == "/api/server/files":
            query = parse_qs(parsed.query)
            self.send_json(list_server_files(query.get("path", [""])[0]))
            return
        if path == "/api/server/file":
            query = parse_qs(parsed.query)
            server_path = resolve_server_path(query.get("path", [""])[0])
            if not is_supported_server_image(server_path):
                self.send_error_json(ValueError("server file must be a supported image"), HTTPStatus.BAD_REQUEST)
                return
            content = server_path.read_bytes()
            content_type = mimetypes.guess_type(server_path.name)[0] or "application/octet-stream"
            self.send_bytes(content, content_type)
            return
        if path == "/api/server/thumbnail":
            query = parse_qs(parsed.query)
            server_path = resolve_server_path(query.get("path", [""])[0])
            self.send_bytes(server_thumbnail_png(server_path), "image/png")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/image":
                payload = self.read_json()
                image = data_url_to_image(str(payload.get("data", "")))
                name = str(payload.get("name", "uploaded-image"))
                state = self.current_state()
                with state.render_lock:
                    with state.lock:
                        state.image = image
                        state.name = name
                        state.source_path = None
                        state.uploaded_at = time.time()
                        state.version += 1
                        state.cache.clear()
                self.send_json(image_response_payload(image, name))
                return

            if path == "/api/server/open":
                payload = self.read_json()
                server_path = resolve_server_path(str(payload.get("path", "")))
                image = open_server_image(server_path)
                state = self.current_state()
                with state.render_lock:
                    with state.lock:
                        state.image = image
                        state.name = server_path.name
                        state.source_path = server_path
                        state.uploaded_at = time.time()
                        state.version += 1
                        state.cache.clear()
                self.send_json(image_response_payload(image, server_path.name, server_path))
                return

            if path == "/api/render":
                payload = self.read_json()
                settings = payload.get("settings", {})
                if not isinstance(settings, dict):
                    raise ValueError("settings must be an object")
                state = self.current_state()
                with state.lock:
                    if state.image is None:
                        raise ValueError("load an image first")
                    image = state.image
                    version = state.version
                    cache = state.cache
                with state.render_lock:
                    result = convert_in_memory(image, settings, cache=cache, version=version)
                self.send_json(result)
                return

            if path == "/api/save":
                payload = self.read_json()
                png_bytes = data_url_to_png_bytes(str(payload.get("data", "")))
                state = self.current_state()
                with state.lock:
                    source_path = state.source_path
                if source_path is None:
                    raise ValueError("open an image from server before using Save")
                if not is_supported_server_image(source_path):
                    raise ValueError("source server file is not a supported image")
                save_target_path = pixel_lab_save_path(source_path)
                save_target_path.write_bytes(png_bytes)
                image = open_server_image(save_target_path)
                with state.render_lock:
                    with state.lock:
                        state.image = image
                        state.name = save_target_path.name
                        state.source_path = save_target_path
                        state.uploaded_at = time.time()
                        state.version += 1
                        state.cache.clear()
                self.send_json(
                    {
                        "ok": True,
                        "path": server_relative_path(save_target_path),
                        "sourcePath": server_relative_path(save_target_path),
                        "saveTargetPath": server_relative_path(pixel_lab_save_path(save_target_path)),
                        "bytes": len(png_bytes),
                        "width": image.width,
                        "height": image.height,
                    }
                )
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
