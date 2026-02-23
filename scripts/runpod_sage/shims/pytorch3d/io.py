"""Minimal subset of :mod:`pytorch3d.io` used by SAGE Stage 4.

This shim implements ``save_obj`` so pose augmentation can run even when the
full PyTorch3D package is not installed in the runtime image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


def _to_numpy(x: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=dtype or np.float32)
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        arr = x.detach().cpu().numpy()
    elif hasattr(x, "cpu") and hasattr(x, "numpy"):
        arr = x.cpu().numpy()
    else:
        arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _iter_rows(arr: np.ndarray) -> Iterable[np.ndarray]:
    if arr.ndim == 1:
        for v in arr.reshape(1, -1):
            yield v
        return
    for v in arr:
        yield v


def save_obj(
    f: Any,
    verts: Any,
    faces: Any,
    decimal_places: Optional[int] = None,
    verts_uvs: Any = None,
    faces_uvs: Any = None,
    texture_map: Any = None,  # noqa: ARG001 - kept for signature compatibility.
    **_: Any,
) -> None:
    verts_np = _to_numpy(verts, dtype=np.float64)
    faces_np = _to_numpy(faces, dtype=np.int64)
    vts_np = _to_numpy(verts_uvs, dtype=np.float64) if verts_uvs is not None else np.zeros((0, 2), dtype=np.float64)
    fts_np = _to_numpy(faces_uvs, dtype=np.int64) if faces_uvs is not None else np.zeros((0, 3), dtype=np.int64)

    if verts_np.ndim != 2 or verts_np.shape[1] < 3:
        raise ValueError("save_obj expects verts with shape (N, >=3)")
    if faces_np.ndim != 2 or faces_np.shape[1] < 3:
        raise ValueError("save_obj expects faces with shape (F, >=3)")

    fmt = "{:.6f}" if decimal_places is None else "{:0." + str(int(decimal_places)) + "f}"

    def _open_target(target: Any):
        if hasattr(target, "write"):
            return target, False
        path = Path(str(target)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8")
        return handle, True

    handle, should_close = _open_target(f)
    try:
        for row in _iter_rows(verts_np[:, :3]):
            handle.write(f"v {fmt.format(float(row[0]))} {fmt.format(float(row[1]))} {fmt.format(float(row[2]))}\n")

        has_uv = vts_np.size > 0
        if has_uv:
            for row in _iter_rows(vts_np[:, :2]):
                handle.write(f"vt {fmt.format(float(row[0]))} {fmt.format(float(row[1]))}\n")

        faces_idx = faces_np[:, :3].astype(np.int64, copy=False) + 1
        if has_uv and fts_np.shape[0] == faces_idx.shape[0]:
            uv_idx = fts_np[:, :3].astype(np.int64, copy=False) + 1
            for face, uv_face in zip(faces_idx, uv_idx):
                handle.write(
                    f"f {int(face[0])}/{int(uv_face[0])} {int(face[1])}/{int(uv_face[1])} {int(face[2])}/{int(uv_face[2])}\n"
                )
        else:
            for face in faces_idx:
                handle.write(f"f {int(face[0])} {int(face[1])} {int(face[2])}\n")
    finally:
        if should_close:
            handle.close()
