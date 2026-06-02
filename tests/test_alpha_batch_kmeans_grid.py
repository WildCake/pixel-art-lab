#!/usr/bin/env python3
"""Regression checks for alpha, k-means, auto-grid metadata, and batch CLI."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pixel_art_lab as lab


def decode_output(data_url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


def build_rgba_fixture() -> Image.Image:
    image = Image.new("RGBA", (96, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for y in range(8, 56, 8):
        draw.rectangle((8, y, 87, y + 4), fill=(180, 80 + y, 120, 255))
    draw.rectangle((22, 18, 74, 46), fill=(245, 214, 88, 132))
    draw.line((8, 8, 87, 55), fill=(8, 10, 18, 255), width=3)
    return image


def base_settings() -> dict:
    return {
        "targetWidth": 48,
        "targetHeight": 32,
        "aspectDriver": "height",
        "aspectLock": True,
        "colors": 8,
        "dither": "none",
        "resample": "box",
        "colorDistance": "rgb",
        "edgeMode": "sobel",
        "edgeThreshold": 0.04,
        "gridSnap": True,
        "gridAutoSize": True,
        "gridQuantizeFirst": False,
        "gridSnapMethod": "center",
        "paletteInput": "prepared",
        "preserveLuma": False,
        "preserveSaturation": False,
        "flatRegionPaletteColors": 0,
        "flatRegionChannelStep": 0,
        "mixelCleanupPasses": 0,
        "paletteStrategy": "kmeans",
    }


def main() -> None:
    source = build_rgba_fixture()
    settings = base_settings()
    assert 'id="gridTopology"' in lab.HTML
    assert 'id="gridAxisStabilization"' in lab.HTML
    assert 'id="preserveAlpha"' in lab.HTML
    assert 'id="viewerStatsOverlay"' in lab.HTML
    assert 'class="viewer viewer-empty"' in lab.HTML
    assert ">load an image<" in lab.HTML
    assert 'id="savePng"' in lab.HTML
    config = lab.config_from_settings(settings, source_size=source.size)
    assert config.grid_snap_topology == "elastic"
    assert config.grid_snap_axis_stabilization == "conservative"
    assert config.preserve_alpha is True
    assert config.palette_strategy == "kmeans"

    result = lab.convert_in_memory(source, settings, cache={}, version=11)
    output = decode_output(result["output"])
    assert output.mode == "RGBA"
    assert output.getchannel("A").getextrema()[0] < 255
    assert result["stats"]["alphaPreserved"] is True
    assert result["stats"]["gridTopology"] == "elastic"
    assert result["stats"]["gridAxisStabilization"] == "conservative"
    assert result["stats"]["gridVariant"]["confidence"] >= 0
    assert "axisRatio" in result["stats"]["gridVariant"]

    uniform_settings = {
        **settings,
        "gridAutoSize": False,
        "gridTopology": "uniform",
        "gridAxisStabilization": "aggressive",
    }
    uniform_result = lab.convert_in_memory(source, uniform_settings, cache={}, version=11)
    assert uniform_result["stats"]["gridTopology"] == "uniform"
    assert uniform_result["stats"]["gridAxisStabilization"] == "off"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        input_dir = temp / "input"
        output_dir = temp / "output"
        input_dir.mkdir()
        source.save(input_dir / "alpha.png")
        source.convert("RGB").save(input_dir / "solid.webp")
        command = [
            sys.executable,
            str(ROOT / "pixel_art_grid.py"),
            str(input_dir),
            "-o",
            str(output_dir),
            "--size",
            "48x32",
            "--colors",
            "8",
            "--palette-strategy",
            "kmeans",
            "--grid-snap",
            "--preview-scale",
            "2",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        report = json.loads(completed.stdout)
        assert report["files_found"] == 2
        assert report["files_written"] == 2
        assert report["errors"] == []
        assert (output_dir / "alpha.png").exists()
        assert (output_dir / "solid.png").exists()
        assert Image.open(output_dir / "alpha.png").mode == "RGBA"

    print("alpha/batch/kmeans/grid metadata passed")


if __name__ == "__main__":
    main()
