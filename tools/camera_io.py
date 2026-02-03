#!/usr/bin/env python3
"""Shared camera frame IO helpers for base64, npy, and raw/encoded bytes."""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

PNG_SIG = b"\x89PNG\r\n\x1a\n"
JPEG_SIG = b"\xff\xd8\xff"


def _is_png_bytes(data: bytes) -> bool:
    return data.startswith(PNG_SIG)


def _is_jpeg_bytes(data: bytes) -> bool:
    return data.startswith(JPEG_SIG)


def _decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    if not data:
        return None
    pil_spec = importlib.util.find_spec("PIL.Image")
    if pil_spec is not None:
        from PIL import Image
        import io
        try:
            with Image.open(io.BytesIO(data)) as img:
                return np.array(img)
        except Exception:
            return None
    imageio_spec = importlib.util.find_spec("imageio")
    if imageio_spec is not None:
        try:
            import imageio.v3 as iio
        except Exception:
            import imageio as iio
        try:
            return iio.imread(data)
        except Exception:
            return None
    return None


def _parse_raw_encoding(
    encoding: str,
    *,
    kind: str,
) -> Tuple[np.dtype, int, str]:
    """Return dtype, channels, and color order for raw encodings."""
    normalized = (encoding or "").lower()
    order = "rgb"
    if "bgr" in normalized:
        order = "bgr"
    if kind == "depth":
        if "32f" in normalized or "float32" in normalized:
            return (np.float32, 1, order)
        if "16u" in normalized or "uint16" in normalized or "16" in normalized:
            return (np.uint16, 1, order)
        if "16f" in normalized or "float16" in normalized:
            return (np.float16, 1, order)
        return (np.float32, 1, order)
    if "rgba" in normalized:
        return (np.uint8, 4, order)
    if "rgb" in normalized:
        return (np.uint8, 3, order)
    if "bgr" in normalized:
        return (np.uint8, 3, order)
    if "mono" in normalized or "8uc1" in normalized or "l8" in normalized:
        return (np.uint8, 1, order)
    return (np.uint8, 3, order)


def expected_byte_count(
    encoding: str,
    *,
    width: int,
    height: int,
    kind: str,
) -> Optional[int]:
    if width <= 0 or height <= 0:
        return None
    normalized = (encoding or "").lower()
    if not normalized:
        return None
    if "png" in normalized or "jpeg" in normalized or "jpg" in normalized:
        return None
    dtype, channels, _order = _parse_raw_encoding(encoding, kind=kind)
    if kind == "depth":
        channels = 1
    return int(width * height * channels * dtype().itemsize)


def decode_camera_bytes(
    raw: bytes,
    *,
    width: int,
    height: int,
    encoding: str,
    kind: str,
) -> Optional[np.ndarray]:
    if not raw:
        return None
    normalized = (encoding or "").lower()
    if "png" in normalized or "jpeg" in normalized or "jpg" in normalized or _is_png_bytes(raw) or _is_jpeg_bytes(raw):
        return _decode_image_bytes(raw)
    dtype, channels, order = _parse_raw_encoding(encoding, kind=kind)
    if width <= 0 or height <= 0:
        return None
    expected = int(width * height * (channels if kind == "rgb" else 1))
    if raw and len(raw) < expected * dtype().itemsize:
        return None
    arr = np.frombuffer(raw, dtype=dtype, count=expected)
    if kind == "rgb":
        arr = arr.reshape((height, width, channels)) if channels > 1 else arr.reshape((height, width))
        if order == "bgr" and arr.ndim == 3:
            arr = arr[..., ::-1]
        return arr
    arr = arr.reshape((height, width))
    return arr


def resolve_npy_path(
    val: str,
    ep_dir: Optional[Path] = None,
    frames_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve a .npy path reference across episode and frames directories."""
    if not val:
        return None
    raw_path = Path(val)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path
    if ep_dir is not None:
        candidate = ep_dir / val
        if candidate.exists():
            return candidate
    if frames_dir is not None:
        candidate = frames_dir / val
        if candidate.exists():
            return candidate
    if ep_dir is not None and raw_path.name == val:
        try:
            for subdir in ep_dir.glob("*_frames"):
                candidate = subdir / val
                if candidate.exists():
                    return candidate
        except Exception:
            return None
    return None


def load_camera_frame(
    cam_data: dict,
    key: str,
    *,
    ep_dir: Optional[Path] = None,
    frames_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    val = cam_data.get(key)
    if val is None:
        return None
    width = int(cam_data.get("width") or 0)
    height = int(cam_data.get("height") or 0)
    encoding = (
        cam_data.get(f"{key}_encoding")
        or cam_data.get("encoding")
        or cam_data.get("rgb_encoding" if key == "rgb" else "depth_encoding")
        or ""
    )
    if isinstance(val, str) and val.endswith(".npy"):
        npy_path = resolve_npy_path(val, ep_dir=ep_dir, frames_dir=frames_dir)
        if npy_path is None or not npy_path.exists():
            return None
        return np.load(npy_path)
    if isinstance(val, bytes):
        return decode_camera_bytes(val, width=width, height=height, encoding=encoding, kind=key)
    if isinstance(val, bytearray):
        return decode_camera_bytes(bytes(val), width=width, height=height, encoding=encoding, kind=key)
    if isinstance(val, str):
        try:
            raw = base64.b64decode(val)
        except Exception:
            return None
        return decode_camera_bytes(raw, width=width, height=height, encoding=encoding, kind=key)
    if isinstance(val, np.ndarray):
        return val
    if isinstance(val, list):
        try:
            arr = np.array(val)
        except Exception:
            return None
        return arr
    return None


def validate_camera_array(data: Any, context: str = "") -> Optional[np.ndarray]:
    """
    Validate that data is a numpy array suitable for image processing.

    Args:
        data: The data to validate (should be numpy array from load_camera_frame)
        context: Optional context string for logging

    Returns:
        The numpy array if valid, None otherwise
    """
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return data
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "Camera data is not ndarray (%s): type=%s",
        context or "unknown",
        type(data).__name__,
    )
    return None


def strip_camera_data(obs: Any) -> Any:
    """Remove bulky camera data from observation for parquet storage."""
    if not isinstance(obs, dict):
        return obs
    obs = dict(obs)
    cf = obs.get("camera_frames")
    if isinstance(cf, dict):
        stripped_cf = {}
        for cam_id, cam_data in cf.items():
            if isinstance(cam_data, dict):
                stripped_cf[cam_id] = {
                    k: v for k, v in cam_data.items()
                    if k not in ("rgb", "depth")
                }
            else:
                stripped_cf[cam_id] = cam_data
        obs["camera_frames"] = stripped_cf
    return obs


def coerce_rgb_frame(
    frame: Any,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    encoding: str = "",
    ep_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    if isinstance(frame, np.ndarray):
        return frame
    if isinstance(frame, str) and frame.endswith(".npy"):
        if ep_dir is None:
            return None
        npy_path = ep_dir / frame
        if npy_path.exists():
            return np.load(npy_path)
        return None
    if isinstance(frame, str):
        try:
            raw = base64.b64decode(frame)
        except Exception:
            return None
        return decode_camera_bytes(
            raw,
            width=width or 0,
            height=height or 0,
            encoding=encoding,
            kind="rgb",
        )
    if isinstance(frame, (bytes, bytearray)):
        return decode_camera_bytes(
            bytes(frame),
            width=width or 0,
            height=height or 0,
            encoding=encoding,
            kind="rgb",
        )
    if isinstance(frame, list):
        try:
            return np.array(frame)
        except Exception:
            return None
    return None
