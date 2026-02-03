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
        if npy_path is None:
            # Path could not be resolved - log warning and return None
            # This prevents returning a string path that would cause .astype() errors
            import logging
            logging.getLogger(__name__).warning(
                "NPY path '%s' cannot be resolved: ep_dir=%s, frames_dir=%s",
                val, ep_dir, frames_dir,
            )
            return None
        if not npy_path.exists():
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


def validate_rgb_frame_quality(
    rgb: np.ndarray,
    *,
    min_unique_colors: int = 100,
    min_std: float = 10.0,
    context: str = "",
) -> Tuple[bool, dict]:
    """
    Validate that RGB frame is a real rendered image, not a placeholder.

    Real renders have many unique colors and reasonable variance. Placeholders
    (color bars, test patterns) have very few unique values.

    Args:
        rgb: RGB image array (H, W, 3) or (H, W, 4)
        min_unique_colors: Minimum unique RGB triplets required (default: 100)
        min_std: Minimum standard deviation required (default: 10.0)
        context: Optional context string for diagnostics

    Returns:
        (is_valid, diagnostics_dict) where diagnostics contains:
        - unique_colors: int
        - std: float
        - is_black: bool
        - shape: tuple
        - reason: str (if invalid)
    """
    diagnostics = {
        "context": context,
        "shape": None,
        "unique_colors": 0,
        "std": 0.0,
        "is_black": True,
        "is_valid": False,
    }

    if rgb is None or not isinstance(rgb, np.ndarray):
        diagnostics["reason"] = "not_ndarray"
        return False, diagnostics

    diagnostics["shape"] = rgb.shape

    if rgb.size == 0:
        diagnostics["reason"] = "empty_array"
        return False, diagnostics

    # Check for black frame
    nonzero_count = np.count_nonzero(rgb)
    diagnostics["is_black"] = nonzero_count == 0

    if diagnostics["is_black"]:
        diagnostics["reason"] = "black_frame"
        return False, diagnostics

    # Check standard deviation
    diagnostics["std"] = float(np.std(rgb.astype(float)))

    # Count unique colors (RGB triplets)
    try:
        if rgb.ndim >= 3 and rgb.shape[-1] >= 3:
            # Reshape to (N, 3) for RGB channels
            flat = rgb.reshape(-1, rgb.shape[-1])[:, :3]
            unique_colors = np.unique(flat, axis=0)
            diagnostics["unique_colors"] = len(unique_colors)
        else:
            # Grayscale or unusual shape
            unique_vals = np.unique(rgb)
            diagnostics["unique_colors"] = len(unique_vals)
    except Exception:
        diagnostics["unique_colors"] = 0

    # Determine validity
    is_valid = (
        diagnostics["unique_colors"] >= min_unique_colors
        and diagnostics["std"] >= min_std
        and not diagnostics["is_black"]
    )
    diagnostics["is_valid"] = is_valid

    if not is_valid:
        if diagnostics["unique_colors"] < min_unique_colors:
            diagnostics["reason"] = f"too_few_colors ({diagnostics['unique_colors']} < {min_unique_colors})"
        elif diagnostics["std"] < min_std:
            diagnostics["reason"] = f"low_variance (std={diagnostics['std']:.2f} < {min_std})"
        else:
            diagnostics["reason"] = "unknown"

    return is_valid, diagnostics


def validate_frame_sequence_variety(
    frames: list,
    *,
    min_diff_threshold: float = 5.0,
) -> Tuple[bool, dict]:
    """
    Validate that frame sequence has variety (not static copies).

    Compares consecutive frames to ensure they're not identical or nearly identical,
    which would indicate capture failure or static placeholder data.

    Args:
        frames: List of numpy arrays (RGB frames)
        min_diff_threshold: Minimum mean pixel difference between frames (default: 5.0)

    Returns:
        (is_valid, diagnostics_dict) where diagnostics contains:
        - avg_frame_diff: float
        - max_frame_diff: float
        - num_frames: int
        - is_static: bool
    """
    diagnostics = {
        "num_frames": len(frames) if frames else 0,
        "avg_frame_diff": 0.0,
        "max_frame_diff": 0.0,
        "is_static": False,
        "is_valid": True,
    }

    if not frames or len(frames) < 2:
        diagnostics["reason"] = "insufficient_frames"
        return True, diagnostics  # Can't validate with < 2 frames

    diffs = []
    for i in range(1, len(frames)):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]

        if prev_frame is None or curr_frame is None:
            continue
        if not isinstance(prev_frame, np.ndarray) or not isinstance(curr_frame, np.ndarray):
            continue
        if prev_frame.shape != curr_frame.shape:
            continue

        try:
            diff = np.mean(np.abs(curr_frame.astype(float) - prev_frame.astype(float)))
            diffs.append(diff)
        except Exception:
            continue

    if not diffs:
        diagnostics["reason"] = "no_valid_comparisons"
        return True, diagnostics

    diagnostics["avg_frame_diff"] = float(np.mean(diffs))
    diagnostics["max_frame_diff"] = float(np.max(diffs))
    diagnostics["is_static"] = diagnostics["avg_frame_diff"] < min_diff_threshold

    is_valid = not diagnostics["is_static"]
    diagnostics["is_valid"] = is_valid

    if not is_valid:
        diagnostics["reason"] = f"static_sequence (avg_diff={diagnostics['avg_frame_diff']:.2f} < {min_diff_threshold})"

    return is_valid, diagnostics


def save_debug_thumbnail(
    rgb: np.ndarray,
    output_dir: Path,
    filename: str,
    *,
    max_size: int = 256,
) -> Optional[Path]:
    """
    Save a low-resolution thumbnail for human QA verification.

    Args:
        rgb: RGB image array
        output_dir: Directory to save thumbnail
        filename: Thumbnail filename (e.g., "frame_0001.png")
        max_size: Maximum dimension for thumbnail (default: 256)

    Returns:
        Path to saved thumbnail, or None on failure
    """
    if rgb is None or not isinstance(rgb, np.ndarray):
        return None

    try:
        from PIL import Image

        # Create thumbnails directory if needed
        thumb_dir = output_dir / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        # Convert to PIL Image
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        img = Image.fromarray(rgb)

        # Resize to thumbnail
        img.thumbnail((max_size, max_size))

        # Save
        thumb_path = thumb_dir / filename
        img.save(thumb_path)

        return thumb_path
    except Exception:
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
                stripped = {
                    k: v for k, v in cam_data.items()
                    if k not in ("rgb", "depth")
                }
                for key in ("rgb", "depth"):
                    val = cam_data.get(key)
                    if isinstance(val, str) and val.lower().endswith((".npy", ".npz", ".png", ".jpg", ".jpeg", ".exr")):
                        stripped[key] = val
                stripped_cf[cam_id] = stripped
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


def is_placeholder_rgb(data: Any) -> bool:
    """Detect placeholder RGB frames (few colors, mostly zeros)."""
    if data is None:
        return False
    if not isinstance(data, np.ndarray):
        try:
            data = np.asarray(data)
        except Exception:
            return False
    if data.size == 0:
        return False
    zero_ratio = float(np.count_nonzero(data == 0)) / float(data.size)
    if zero_ratio <= 0.70:
        return False
    try:
        if data.ndim >= 3 and data.shape[-1] >= 3:
            flat = data.reshape(-1, data.shape[-1])[:, :3]
            unique_colors = np.unique(flat, axis=0)
            return unique_colors.shape[0] <= 4
        unique_vals = np.unique(data)
        return unique_vals.size <= 4
    except Exception:
        return False


def is_placeholder_depth(data: Any) -> bool:
    """Detect placeholder depth frames (all inf or all zeros)."""
    if data is None:
        return False
    if not isinstance(data, np.ndarray):
        try:
            data = np.asarray(data)
        except Exception:
            return False
    if data.size == 0:
        return False
    if np.all(~np.isfinite(data)):
        return True
    if np.all(data == 0):
        return True
    return False
