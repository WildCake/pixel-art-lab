# AGENTS.md

## Purpose

Pixel Art Lab is a local-only Python web tool for converting AI-generated or faux pixel art into a strict pixel grid.

## Run Command

Use this from the repository root:

```sh
python3 pixel_art_lab.py --host 127.0.0.1 --port 8767 --no-browser
```

If the port is busy, the service chooses the next available port and prints the final URL.

## Development Rules

- Keep the app single-machine and local-only.
- Do not add remote runtime assets, analytics, telemetry, or hosted image uploads.
- Put new expensive image operations in NumPy or Numba first.
- Keep manual width/height editing deterministic: only commit dimensions on Enter, blur, or control change, not while the user is still typing.
- Preserve the `Z` comparison tooltip behavior: original and output must show the same normalized visual region.
- Update `README.md` when run commands, dependencies, or operator workflow changes.

## Verification

Run syntax checks before handing off:

```sh
python3 -c "from pathlib import Path; [compile(Path(p).read_text(), p, 'exec') for p in ['pixel_art_grid.py', 'pixel_art_lab.py']]"
```

When possible, start the GUI and verify `/api/status`:

```sh
python3 pixel_art_lab.py --port 8767 --no-browser
curl http://127.0.0.1:8767/api/status
```
