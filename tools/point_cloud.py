#!/usr/bin/env python3
"""Point cloud utilities derived from depth + camera calibration.

This module is intentionally dependency-light (numpy only) so it can be used
from export pipelines without requiring Isaac Sim / Open3D.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def _as_extrinsic_matrix(extrinsic: Any) -> Optional[np.ndarray]:
    """Coerce a camera-to-world extrinsic into a (4, 4) float64 matrix.

    Accepts:
    - None
    - list/tuple length 16 (row-major)
    - numpy array shape (4, 4) or (16,)
    """
    if extrinsic is None:
        return None
    if isinstance(extrinsic, np.ndarray):
        arr = extrinsic
    else:
        try:
            arr = np.asarray(extrinsic, dtype=np.float64)
        except Exception:
            return None
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        try:
            return arr.reshape((4, 4))
        except Exception:
            return None
    return None


def depth_to_points(
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    *,
    rgb: Optional[np.ndarray] = None,
    extrinsic_cam_to_world: Optional[Any] = None,
    max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    min_depth_m: float = 1e-6,
    max_depth_m: float = 10.0,
) -> Dict[str, np.ndarray]:
    """Convert a depth image to a point cloud (optionally with colors).

    Args:
        depth_m: (H, W) depth in meters.
        intr: Camera intrinsics (fx, fy, cx, cy) in pixels.
        rgb: Optional (H, W, 3|4) uint8 image. If provided, colors are sampled
            at the same pixels as the point cloud.
        extrinsic_cam_to_world: Optional camera-to-world (4,4) transform.
        max_points: If set, randomly sample up to this many points.
        rng: Random generator used for subsampling.
        min_depth_m: Minimum valid depth.
        max_depth_m: Maximum valid depth.

    Returns:
        Dict containing:
        - points: (N, 3) float32 points in camera frame or world frame
        - colors: (N, 3) uint8 (only if rgb provided and compatible)
        - pixel_uv: (N, 2) int32 (u, v) pixel coordinates for each point
    """
    if depth_m is None or not isinstance(depth_m, np.ndarray):
        return {"points": np.zeros((0, 3), dtype=np.float32)}
    if depth_m.ndim != 2:
        # Best-effort: take first channel if depth is encoded as HxWxC.
        if depth_m.ndim == 3 and depth_m.shape[-1] >= 1:
            depth_m = depth_m[..., 0]
        else:
            return {"points": np.zeros((0, 3), dtype=np.float32)}

    depth = depth_m.astype(np.float64, copy=False)
    valid = np.isfinite(depth) & (depth > float(min_depth_m)) & (depth < float(max_depth_m))
    ys, xs = np.where(valid)
    if xs.size == 0:
        out: Dict[str, np.ndarray] = {"points": np.zeros((0, 3), dtype=np.float32)}
        if rgb is not None:
            out["colors"] = np.zeros((0, 3), dtype=np.uint8)
        out["pixel_uv"] = np.zeros((0, 2), dtype=np.int32)
        return out

    if max_points is not None and int(max_points) > 0 and xs.size > int(max_points):
        if rng is None:
            rng = np.random.default_rng()
        choice = rng.choice(xs.size, size=int(max_points), replace=False)
        xs = xs[choice]
        ys = ys[choice]

    z = depth[ys, xs]
    x = (xs.astype(np.float64) - float(intr.cx)) * z / float(intr.fx)
    y = (ys.astype(np.float64) - float(intr.cy)) * z / float(intr.fy)
    points = np.stack([x, y, z], axis=1)

    extr = _as_extrinsic_matrix(extrinsic_cam_to_world)
    if extr is not None:
        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        hom = np.concatenate([points, ones], axis=1)
        points = (extr @ hom.T).T[:, :3]

    out = {
        "points": points.astype(np.float32),
        "pixel_uv": np.stack([xs.astype(np.int32), ys.astype(np.int32)], axis=1),
    }

    if rgb is not None and isinstance(rgb, np.ndarray) and rgb.ndim == 3 and rgb.shape[0] == depth_m.shape[0] and rgb.shape[1] == depth_m.shape[1]:
        colors = rgb[ys, xs]
        if colors.shape[-1] >= 3:
            out["colors"] = colors[..., :3].astype(np.uint8, copy=False)

    return out


def pad_or_truncate_points(
    points: np.ndarray,
    target_n: int,
    *,
    pad_value: float = float("nan"),
) -> Tuple[np.ndarray, int]:
    """Pad/truncate an (N, 3) point array to (target_n, 3)."""
    if target_n <= 0:
        return np.zeros((0, 3), dtype=np.float32), 0
    if points is None or not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[-1] != 3:
        padded = np.full((target_n, 3), pad_value, dtype=np.float32)
        return padded, 0
    n = int(points.shape[0])
    if n >= target_n:
        return points[:target_n].astype(np.float32, copy=False), target_n
    padded = np.full((target_n, 3), pad_value, dtype=np.float32)
    padded[:n] = points.astype(np.float32, copy=False)
    return padded, n


def pad_or_truncate_colors(
    colors: Optional[np.ndarray],
    target_n: int,
    *,
    pad_value: int = 0,
) -> np.ndarray:
    """Pad/truncate an (N, 3) uint8 color array to (target_n, 3)."""
    if target_n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if colors is None or not isinstance(colors, np.ndarray) or colors.ndim != 2 or colors.shape[-1] < 3:
        return np.full((target_n, 3), int(pad_value), dtype=np.uint8)
    n = int(colors.shape[0])
    if n >= target_n:
        return colors[:target_n, :3].astype(np.uint8, copy=False)
    padded = np.full((target_n, 3), int(pad_value), dtype=np.uint8)
    padded[:n] = colors[:, :3].astype(np.uint8, copy=False)
    return padded

