#!/usr/bin/env python3
"""Regression checks for server file browsing and per-user sessions."""

from __future__ import annotations

import base64
import io
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
            assert payload["canSaveInPlace"] is True

            png_bytes = lab.data_url_to_png_bytes(data_url(Image.new("RGBA", (3, 2), (1, 2, 3, 255))))
            assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

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
            assert ">Save As<" in lab.HTML
        finally:
            lab.SERVER_BROWSER_ROOT = old_root
            lab.SESSIONS.clear()
            lab.SESSIONS.update(old_sessions)

    print("server sessions/files passed")


if __name__ == "__main__":
    main()
