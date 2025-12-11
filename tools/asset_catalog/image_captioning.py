"""Simple thumbnail captioning helper used by the asset catalog tools."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover - no pillow installed
    Image = None


def caption_thumbnail(thumbnail_path: str, fallback: Optional[str] = None) -> str:
    """Generate a short caption for a thumbnail image.

    This is a lightweight, dependency-free stand-in for richer captioning
    services. If Pillow is available, it reports the image size to aid debugging;
    otherwise it returns the provided fallback or filename.
    """

    path = Path(thumbnail_path)
    if not path.exists():
        return fallback or f"missing thumbnail: {path.name}"

    if Image is None:
        return fallback or path.stem.replace("_", " ")

    try:
        with Image.open(path) as img:
            width, height = img.size
        return fallback or f"{path.stem.replace('_', ' ')} ({width}x{height})"
    except Exception:
        return fallback or path.stem.replace("_", " ")

