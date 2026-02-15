from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pxr import Gf, Usd, UsdGeom


_DYNAMIC_ROLES = {"clutter", "manipulable_object", "interactive", "deformable_object"}


def _gf_matrix_to_np(matrix: Gf.Matrix4d) -> np.ndarray:
    return np.array([[float(matrix[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)


def _np_to_gf_matrix(matrix: np.ndarray) -> Gf.Matrix4d:
    m = np.array(matrix, dtype=np.float64).reshape(4, 4)
    return Gf.Matrix4d(*m.flatten().tolist())


def _iter_object_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    root = stage.GetPrimAtPath("/World/Objects")
    if not root or not root.IsValid():
        return []
    return [prim for prim in root.GetChildren() if prim and prim.IsValid()]


def _get_half_extents(prim: Usd.Prim) -> np.ndarray:
    attr = prim.GetAttribute("halfExtents")
    if attr and attr.HasValue():
        value = attr.Get()
        if value is not None:
            return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)
    return np.array([0.2, 0.2, 0.2], dtype=np.float64)


def _get_sim_role(prim: Usd.Prim) -> str:
    attr = prim.GetAttribute("sim_role")
    if attr and attr.HasValue():
        return str(attr.Get() or "")
    return ""


def _get_transform_matrix(prim: Usd.Prim) -> Optional[np.ndarray]:
    attr = prim.GetAttribute("xformOp:transform")
    if not attr or not attr.HasValue():
        return None
    value = attr.Get()
    if value is None:
        return None
    return _gf_matrix_to_np(value)


def _set_transform_matrix(prim: Usd.Prim, matrix: np.ndarray) -> None:
    attr = prim.GetAttribute("xformOp:transform")
    if attr:
        attr.Set(_np_to_gf_matrix(matrix))


def _aabb_from_state(center: np.ndarray, half_extents: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    return (
        float(center[0] - half_extents[0]),
        float(center[1] - half_extents[1]),
        float(center[2] - half_extents[2]),
        float(center[0] + half_extents[0]),
        float(center[1] + half_extents[1]),
        float(center[2] + half_extents[2]),
    )


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _pairwise_penetration(a: Tuple[float, float, float, float, float, float], b: Tuple[float, float, float, float, float, float]) -> Tuple[float, float, float]:
    ox = _overlap_1d(a[0], a[3], b[0], b[3])
    oy = _overlap_1d(a[1], a[4], b[1], b[4])
    oz = _overlap_1d(a[2], a[5], b[2], b[5])
    return ox, oy, oz


def resolve_penetrations(stage: Usd.Stage, *, max_iters: int = 8, margin: float = 0.002) -> Dict[str, Any]:
    prims = _iter_object_prims(stage)
    state: List[Dict[str, Any]] = []
    for prim in prims:
        matrix = _get_transform_matrix(prim)
        if matrix is None:
            continue
        state.append(
            {
                "prim": prim,
                "matrix": matrix,
                "center": matrix[:3, 3].copy(),
                "half_extents": _get_half_extents(prim),
            }
        )

    if len(state) < 2:
        return {"enabled": True, "iterations": 0, "resolved_contacts": 0, "max_penetration_mm": 0.0}

    resolved_contacts = 0
    max_penetration = 0.0
    iterations = 0

    for it in range(max_iters):
        moved = False
        iterations = it + 1
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                a = state[i]
                b = state[j]
                box_a = _aabb_from_state(a["center"], a["half_extents"])
                box_b = _aabb_from_state(b["center"], b["half_extents"])
                ox, oy, oz = _pairwise_penetration(box_a, box_b)
                if ox <= 0.0 or oy <= 0.0 or oz <= 0.0:
                    continue
                moved = True
                resolved_contacts += 1
                max_penetration = max(max_penetration, min(ox, oy, oz))

                if ox <= oz:
                    push = (ox + margin) * 0.5
                    sign = -1.0 if a["center"][0] <= b["center"][0] else 1.0
                    a["center"][0] += sign * push
                    b["center"][0] -= sign * push
                else:
                    push = (oz + margin) * 0.5
                    sign = -1.0 if a["center"][2] <= b["center"][2] else 1.0
                    a["center"][2] += sign * push
                    b["center"][2] -= sign * push
        if not moved:
            break

    for item in state:
        matrix = item["matrix"].copy()
        matrix[:3, 3] = item["center"]
        _set_transform_matrix(item["prim"], matrix)
        item["matrix"] = matrix

    return {
        "enabled": True,
        "iterations": iterations,
        "resolved_contacts": resolved_contacts,
        "max_penetration_mm": round(max_penetration * 1000.0, 3),
    }


def gravity_settle(stage: Usd.Stage, *, floor_y: float = 0.0, settle_delta: float = 0.01) -> Dict[str, Any]:
    prims = _iter_object_prims(stage)
    settled_count = 0
    adjusted_count = 0

    for prim in prims:
        role = _get_sim_role(prim).strip().lower()
        if role not in _DYNAMIC_ROLES:
            continue
        matrix = _get_transform_matrix(prim)
        if matrix is None:
            continue
        half_extents = _get_half_extents(prim)
        center_y = float(matrix[1, 3])
        min_y = center_y - float(half_extents[1])
        if min_y < floor_y:
            center_y += (floor_y - min_y)
            adjusted_count += 1
        else:
            center_y = max(center_y - settle_delta, floor_y + float(half_extents[1]))
            adjusted_count += 1
        matrix[1, 3] = center_y
        _set_transform_matrix(prim, matrix)
        settled_count += 1

    return {
        "enabled": True,
        "dynamic_objects": settled_count,
        "adjusted_objects": adjusted_count,
        "backend": "isaac_settle_proxy",
    }


def collect_settled_transforms(stage: Usd.Stage) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for prim in _iter_object_prims(stage):
        matrix = _get_transform_matrix(prim)
        if matrix is None:
            continue
        payload[str(prim.GetPath())] = {
            "translation": [round(float(matrix[0, 3]), 6), round(float(matrix[1, 3]), 6), round(float(matrix[2, 3]), 6)],
            "matrix_row_major": [round(float(v), 6) for v in matrix.reshape(-1).tolist()],
        }
    return payload


def run_postprocess(stage: Usd.Stage) -> Dict[str, Any]:
    penetration = resolve_penetrations(stage)
    settle = gravity_settle(stage)
    return {
        "penetration_resolve": penetration,
        "gravity_settle": settle,
        "settled_transforms": collect_settled_transforms(stage),
    }


def write_settled_transforms(path: Path, transforms: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")

