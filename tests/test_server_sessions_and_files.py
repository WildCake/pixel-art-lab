#!/usr/bin/env python3
"""Regression checks for server file browsing and per-user sessions."""

from __future__ import annotations

import base64
import io
import re
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pixel_art_lab as lab


def data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "assets" / "generated"
        nested = root / "ui" / "backgrounds"
        nested.mkdir(parents=True)
        image_path = nested / "server-open.png"
        Image.new("RGB", (8, 6), (12, 34, 56)).save(image_path)
        large_image_path = nested / "wide-server-open.png"
        Image.new("RGB", (640, 480), (78, 90, 123)).save(large_image_path)

        old_root = lab.SERVER_BROWSER_ROOT
        old_sessions = lab.SESSIONS.copy()
        try:
            lab.SERVER_BROWSER_ROOT = root
            lab.SESSIONS.clear()

            listed_root = lab.list_server_files("")
            assert listed_root["path"] == ""
            assert listed_root["entries"][0]["type"] == "dir"
            assert listed_root["entries"][0]["path"] == "ui"

            listed_nested = lab.list_server_files("ui/backgrounds")
            assert listed_nested["parent"] == "ui"
            assert listed_nested["entries"][0]["type"] == "file"
            assert listed_nested["entries"][0]["path"] == "ui/backgrounds/server-open.png"

            opened = lab.open_server_image(lab.resolve_server_path("ui/backgrounds/server-open.png"))
            assert opened.size == (8, 6)
            payload = lab.image_response_payload(opened, image_path.name, image_path)
            assert payload["sourcePath"] == "ui/backgrounds/server-open.png"
            assert payload["saveTargetPath"] == "ui/backgrounds/server-open_PIXEL_LAB.png"
            assert payload["canSaveInPlace"] is True
            assert lab.pixel_lab_save_path(image_path).name == "server-open_PIXEL_LAB.png"
            suffixed_path = nested / "server-open_PIXEL_LAB.png"
            Image.new("RGB", (8, 6), (12, 34, 56)).save(suffixed_path)
            assert lab.pixel_lab_save_path(suffixed_path).name == "server-open_PIXEL_LAB.png"

            png_bytes = lab.data_url_to_png_bytes(data_url(Image.new("RGBA", (3, 2), (1, 2, 3, 255))))
            assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

            thumbnail_bytes = lab.server_thumbnail_png(large_image_path)
            assert thumbnail_bytes.startswith(b"\x89PNG\r\n\x1a\n")
            thumbnail = Image.open(io.BytesIO(thumbnail_bytes))
            assert thumbnail.size == lab.SERVER_THUMBNAIL_SIZE

            try:
                lab.resolve_server_path("../outside.png")
            except ValueError as exc:
                assert "assets/generated" in str(exc)
            else:
                raise AssertionError("path traversal was not rejected")

            session_a = lab.get_session("aaaaaaaaaaaaaaaa")
            session_b = lab.get_session("bbbbbbbbbbbbbbbb")
            assert session_a is not session_b
            assert session_a.render_lock is not session_b.render_lock

            assert 'id="openFromServer"' in lab.HTML
            assert 'id="saveInPlace"' in lab.HTML
            assert 'id="loadProject"' in lab.HTML
            assert 'id="saveProject"' in lab.HTML
            assert 'id="projectFile"' in lab.HTML
            assert "diliada.pixel-art-lab.project" in lab.HTML
            assert ".pixelartlab" in lab.HTML
            assert "pixel-art-lab-presets" in lab.HTML
            assert "indexedDB.open(PRESET_DB_NAME" in lab.HTML
            assert "localStorage.removeItem(PRESET_STORAGE_KEY)" in lab.HTML
            assert "async function saveCurrentPreset()" in lab.HTML
            assert "pixel-art-lab-recovered-local-presets-seeded-v1" in lab.HTML
            assert "Backs_1" in lab.HTML
            assert "Backs_2" in lab.HTML
            assert "kek" in lab.HTML
            assert "api/server/thumbnail" in lab.HTML
            assert "server-entry-file" in lab.HTML
            assert 'id="tooltipLayer"' in lab.HTML
            assert "--tooltip-layer: 2147483647" in lab.HTML
            assert "var(--tooltip-bg)" in lab.HTML
            assert ">Save As<" in lab.HTML
            overlay_match = re.search(r"function drawOriginalOverlay\(\) \{(?P<body>.*?)function viewerPoint", lab.HTML, re.S)
            assert overlay_match is not None
            overlay_body = overlay_match.group("body")
            assert "ctx.imageSmoothingEnabled = state.zoom < 1;" in overlay_body
            assert "ctx.imageSmoothingEnabled = true;" not in overlay_body
        finally:
            lab.SERVER_BROWSER_ROOT = old_root
            lab.SESSIONS.clear()
            lab.SESSIONS.update(old_sessions)

    print("server sessions/files passed")


if __name__ == "__main__":
    main()
