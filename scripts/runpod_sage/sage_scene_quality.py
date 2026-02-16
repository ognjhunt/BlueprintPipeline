#!/usr/bin/env python3
"""
Fast, deterministic scene quality gates + bounded repairs for SAGE layout dirs.

This is intentionally CPU-only and does not require Isaac Sim.

It operates on SAGE "room dict" JSON files:
  - room_*.json
  - pose augmentation variants referenced by pose_aug_*/meta.json

Quality metrics (AABB-based, XY-plane):
  - collision rate (objects involved in any collision)
  - mean/max penetration depth (minimum translation to separate in XY)
  - floor violations (z < 0)
  - floating heuristic (z > 0 and no supporting surface below)

Repairs (bounded, fast):
  1) AABB push-apart in XY (no vertical motion)
  2) Floor clamp (z >= 0)
  3) Surface snap for manipulables onto nearby surface tops
  4) Room-bounds clamp (keep inside room dimensions)

Exit code:
  0: all checked JSONs pass profile thresholds after repairs
  3: one or more JSONs still fail after repairs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _log(msg: str) -> None:
    print(f"[sage-quality {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", file=sys.stderr, flush=True)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _normalize_type(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _resolve_pose_aug_variants(meta_path: Path, layout_dir: Path) -> List[Path]:
    meta = _load_json(meta_path)
    candidates: List[Any] = []
    if isinstance(meta, list):
        candidates = meta
    elif isinstance(meta, dict):
        for key in ("layouts", "layout_dict_paths", "layout_paths", "variants", "feasible_layouts"):
            if key in meta and isinstance(meta[key], list):
                candidates = meta[key]
                break
    else:
        raise ValueError(f"Unsupported meta.json type: {type(meta).__name__}")

    paths: List[Path] = []
    for item in candidates:
        p: Optional[str] = None
        if isinstance(item, str):
            p = item
        elif isinstance(item, dict):
            for k in ("layout_dict_path", "layout_dict_save_path", "layout_path", "json_path", "path"):
                if k in item and isinstance(item[k], str):
                    p = item[k]
                    break
        if not p:
            continue

        path = Path(p)
        if not path.is_absolute():
            if (meta_path.parent / path).exists():
                path = meta_path.parent / path
            else:
                path = layout_dir / path
        if path.exists():
            paths.append(path)

    # Deduplicate, preserve order.
    seen = set()
    out: List[Path] = []
    for p in paths:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        out.append(p)
    return out


@dataclass(frozen=True)
class _Profile:
    name: str
    max_collision_rate_objects: float
    max_penetration_m: float
    max_floor_violations: int
    max_floating_violations_rate: float
    require_manipulable: bool = True
    require_surface: bool = True


_PROFILES: Dict[str, _Profile] = {
    "standard": _Profile(
        name="standard",
        max_collision_rate_objects=0.08,
        max_penetration_m=0.02,
        max_floor_violations=0,
        max_floating_violations_rate=0.05,
    ),
    "strict": _Profile(
        name="strict",
        max_collision_rate_objects=0.02,
        max_penetration_m=0.005,
        max_floor_violations=0,
        max_floating_violations_rate=0.0,
    ),
}


_SURFACE_TOKENS = ("table", "counter", "desk", "island", "bench")
_WALL_MOUNT_TOKENS = ("picture", "painting", "mirror", "clock", "poster", "frame", "wall")
_TABLETOP_OBJECT_TOKENS = (
    "plate",
    "bowl",
    "cup",
    "mug",
    "glass",
    "bottle",
    "can",
    "jar",
    "salt",
    "pepper",
    "shaker",
    "utensil",
    "fork",
    "spoon",
    "knife",
)
_MASS_PRESETS_KG: Dict[str, Dict[str, float]] = {
    "salt": {"target": 0.30, "min": 0.05, "max": 0.80},
    "pepper": {"target": 0.30, "min": 0.05, "max": 0.80},
    "salt_shaker": {"target": 0.30, "min": 0.05, "max": 0.80},
    "pepper_shaker": {"target": 0.30, "min": 0.05, "max": 0.80},
    "plate": {"target": 0.50, "min": 0.10, "max": 2.00},
    "bowl": {"target": 0.40, "min": 0.10, "max": 2.00},
    "mug": {"target": 0.30, "min": 0.05, "max": 1.50},
    "cup": {"target": 0.25, "min": 0.05, "max": 1.50},
    "bottle": {"target": 0.60, "min": 0.10, "max": 2.00},
}


def _object_type(obj: Dict[str, Any]) -> str:
    return _normalize_type(str(obj.get("type", "")))


def _match_mass_preset_key(obj_type: str) -> Optional[str]:
    if not obj_type:
        return None
    for key in _MASS_PRESETS_KG:
        if key in obj_type:
            return key
    return None


def _is_tabletop_candidate(obj: Dict[str, Any]) -> bool:
    t = _object_type(obj)
    if any(tok in t for tok in _TABLETOP_OBJECT_TOKENS):
        return True
    if _is_manipulable(obj):
        return True
    return False


def _dims(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    d = obj.get("dimensions", {}) or {}
    w = _safe_float(d.get("width"), default=float("nan"))
    l = _safe_float(d.get("length"), default=float("nan"))
    h = _safe_float(d.get("height"), default=float("nan"))
    if not (math.isfinite(w) and math.isfinite(l) and math.isfinite(h)):
        return None
    if w <= 0.0 or l <= 0.0 or h <= 0.0:
        return None
    return float(w), float(l), float(h)


def _pos(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    p = obj.get("position", {}) or {}
    x = _safe_float(p.get("x"), default=float("nan"))
    y = _safe_float(p.get("y"), default=float("nan"))
    z = _safe_float(p.get("z"), default=float("nan"))
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return float(x), float(y), float(z)


def _aabb(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float, float, float, float]]:
    p = _pos(obj)
    d = _dims(obj)
    if p is None or d is None:
        return None
    x, y, z = p
    w, l, h = d
    # SAGE convention in this repo: x/y are footprint center; z is bottom.
    return (x - w * 0.5, y - l * 0.5, z, x + w * 0.5, y + l * 0.5, z + h)


def _is_manipulable(obj: Dict[str, Any]) -> bool:
    d = _dims(obj)
    if d is None:
        return False
    w, l, _ = d
    if min(w, l) > 0.35 and not any(tok in _object_type(obj) for tok in _TABLETOP_OBJECT_TOKENS):
        return False
    return bool(obj.get("source_id"))


def _is_surface(obj: Dict[str, Any]) -> bool:
    t = _normalize_type(str(obj.get("type", "")))
    return any(tok in t for tok in _SURFACE_TOKENS)


def _is_wall_mounted(obj: Dict[str, Any]) -> bool:
    t = _normalize_type(str(obj.get("type", "")))
    return any(tok in t for tok in _WALL_MOUNT_TOKENS)


def _room_dims(room: Dict[str, Any]) -> Dict[str, float]:
    d = room.get("dimensions", {}) or {}
    width = _safe_float(d.get("width"), default=6.0)
    length = _safe_float(d.get("length"), default=6.0)
    height = _safe_float(d.get("height"), default=3.0)
    # Best-effort fallback if missing.
    if not (math.isfinite(width) and width > 0.5):
        width = 6.0
    if not (math.isfinite(length) and length > 0.5):
        length = 6.0
    if not (math.isfinite(height) and height > 0.5):
        height = 3.0
    return {"width": float(width), "length": float(length), "height": float(height)}


def _collision_metrics(objs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    aabbs: List[Optional[Tuple[float, float, float, float, float, float]]] = [_aabb(o) for o in objs]
    n = len(objs)
    sample_pairs: List[Dict[str, Any]] = []
    total_pairs = 0
    total_sum_pen = 0.0
    total_max_pen = 0.0
    objects_in_collision: set[int] = set()

    for i in range(n):
        ai = aabbs[i]
        if ai is None:
            continue
        for j in range(i + 1, n):
            aj = aabbs[j]
            if aj is None:
                continue
            # 3D AABB intersection (avoid flagging stacked support like "mug on table").
            ox = min(ai[3], aj[3]) - max(ai[0], aj[0])
            oy = min(ai[4], aj[4]) - max(ai[1], aj[1])
            oz = min(ai[5], aj[5]) - max(ai[2], aj[2])
            if ox <= 0.0 or oy <= 0.0 or oz <= 0.0:
                continue
            pen = float(min(ox, oy, oz))
            total_pairs += 1
            total_sum_pen += pen
            total_max_pen = max(total_max_pen, pen)
            objects_in_collision.add(i)
            objects_in_collision.add(j)
            if len(sample_pairs) < 50:
                sample_pairs.append(
                    {
                        "i": i,
                        "j": j,
                        "type_i": str(objs[i].get("type", "")),
                        "type_j": str(objs[j].get("type", "")),
                        "penetration_m": round(pen, 6),
                        "overlap_x_m": round(float(ox), 6),
                        "overlap_y_m": round(float(oy), 6),
                        "overlap_z_m": round(float(oz), 6),
                    }
                )

    collision_rate_objects = float(len(objects_in_collision)) / float(n) if n else 0.0
    mean_pen = float(total_sum_pen) / float(total_pairs) if total_pairs else 0.0
    return {
        "num_objects": int(n),
        "num_colliding_pairs": int(total_pairs),
        "num_objects_in_collision": int(len(objects_in_collision)),
        "collision_rate_objects": round(collision_rate_objects, 6),
        "mean_penetration_m": round(mean_pen, 6),
        "max_penetration_m": round(float(total_max_pen), 6),
        "sample_pairs": sample_pairs,
    }


def _floating_and_floor_metrics(room: Dict[str, Any]) -> Dict[str, Any]:
    objs = list(room.get("objects", []) or [])
    eps_floor = 1e-4
    floor_violations = 0
    floating_violations = 0
    tabletop_support_violations = 0

    surfaces: List[Tuple[Dict[str, Any], Tuple[float, float, float, float, float, float]]] = []
    for obj in objs:
        if not _is_surface(obj):
            continue
        a = _aabb(obj)
        if a is None:
            continue
        surfaces.append((obj, a))

    for obj in objs:
        p = _pos(obj)
        d = _dims(obj)
        if p is None or d is None:
            continue
        x, y, z = p
        if z < -eps_floor:
            floor_violations += 1

        if _is_wall_mounted(obj):
            continue
        requires_surface = bool(surfaces) and (not _is_surface(obj)) and _is_tabletop_candidate(obj)
        if z <= 0.05 and not requires_surface:
            continue

        # Supported if any surface top is near the object's bottom and overlaps in XY.
        supported = False
        a = _aabb(obj)
        if a is not None:
            for _surface_obj, s in surfaces:
                ox = min(a[3], s[3]) - max(a[0], s[0])
                oy = min(a[4], s[4]) - max(a[1], s[1])
                if ox <= 0.0 or oy <= 0.0:
                    continue
                top = float(s[5])
                support_tol = 0.08 if requires_surface else 0.05
                if abs(z - top) <= support_tol:
                    supported = True
                    break
        if not supported:
            floating_violations += 1
        if requires_surface and (not supported or z <= 0.08):
            tabletop_support_violations += 1

    return {
        "floor_violations": int(floor_violations),
        "floating_violations": int(floating_violations),
        "tabletop_support_violations": int(tabletop_support_violations),
        "num_surfaces": int(len(surfaces)),
    }


def _object_count_sanity(room: Dict[str, Any]) -> Dict[str, Any]:
    objs = list(room.get("objects", []) or [])
    num_manip = sum(1 for o in objs if _is_manipulable(o))
    num_surf = sum(1 for o in objs if _is_surface(o))
    return {"num_manipulable": int(num_manip), "num_surfaces": int(num_surf)}


def _mass_outlier_metrics(room: Dict[str, Any]) -> Dict[str, Any]:
    objs = list(room.get("objects", []) or [])
    outliers: List[Dict[str, Any]] = []
    known = 0
    for obj in objs:
        key = _match_mass_preset_key(_object_type(obj))
        if key is None:
            continue
        known += 1
        profile = _MASS_PRESETS_KG[key]
        mass_raw = (obj.get("physics", {}) or {}).get("mass")
        mass = _safe_float(mass_raw, default=float("nan"))
        if not math.isfinite(mass):
            outliers.append(
                {"id": str(obj.get("id", "")), "type": str(obj.get("type", "")), "reason": "missing_mass", "preset_key": key}
            )
            continue
        if mass < float(profile["min"]) or mass > float(profile["max"]):
            outliers.append(
                {
                    "id": str(obj.get("id", "")),
                    "type": str(obj.get("type", "")),
                    "reason": "mass_out_of_range",
                    "preset_key": key,
                    "mass": float(mass),
                    "min": float(profile["min"]),
                    "max": float(profile["max"]),
                }
            )
    return {
        "num_known_mass_objects": int(known),
        "mass_outlier_count": int(len(outliers)),
        "mass_outliers": outliers[:100],
    }


def evaluate_room(room: Dict[str, Any]) -> Dict[str, Any]:
    objs = list(room.get("objects", []) or [])
    metrics = {}
    metrics.update(_collision_metrics(objs))
    metrics.update(_floating_and_floor_metrics(room))
    metrics.update(_object_count_sanity(room))
    metrics.update(_mass_outlier_metrics(room))
    return metrics


def _profile_pass(metrics: Dict[str, Any], profile: _Profile) -> Tuple[bool, List[str]]:
    n = int(metrics.get("num_objects", 0) or 0)
    errors: List[str] = []
    if n <= 0:
        errors.append("room_has_zero_objects")
        return False, errors

    if profile.require_manipulable and int(metrics.get("num_manipulable", 0) or 0) < 1:
        errors.append("no_manipulable_object")
    if profile.require_surface and int(metrics.get("num_surfaces", 0) or 0) < 1:
        errors.append("no_support_surface")

    if float(metrics.get("collision_rate_objects", 0.0) or 0.0) > profile.max_collision_rate_objects:
        errors.append("collision_rate_objects_exceeds_threshold")
    if float(metrics.get("max_penetration_m", 0.0) or 0.0) > profile.max_penetration_m:
        errors.append("max_penetration_exceeds_threshold")
    if int(metrics.get("floor_violations", 0) or 0) > profile.max_floor_violations:
        errors.append("floor_violations_exceeds_threshold")
    if int(metrics.get("tabletop_support_violations", 0) or 0) > 0:
        errors.append("tabletop_support_violations_exceeds_threshold")
    if int(metrics.get("mass_outlier_count", 0) or 0) > 0:
        errors.append("mass_outliers_exceed_threshold")

    max_float = int(math.ceil(profile.max_floating_violations_rate * float(n)))
    if int(metrics.get("floating_violations", 0) or 0) > max_float:
        errors.append("floating_violations_exceeds_threshold")

    return (len(errors) == 0), errors


def _clamp_room_bounds(
    room: Dict[str, Any],
    objs: Sequence[Dict[str, Any]],
    *,
    margin: float = 0.02,
    correction_log: Optional[List[Dict[str, Any]]] = None,
) -> int:
    dims = _room_dims(room)
    room_w = float(dims["width"])
    room_l = float(dims["length"])
    changed = 0
    for obj in objs:
        p = obj.get("position", {}) or {}
        d = _dims(obj)
        if d is None:
            continue
        w, l, _ = d
        x = _safe_float(p.get("x"), default=0.0)
        y = _safe_float(p.get("y"), default=0.0)
        min_x = w * 0.5 + margin
        max_x = room_w - w * 0.5 - margin
        min_y = l * 0.5 + margin
        max_y = room_l - l * 0.5 - margin
        nx, ny = x, y
        if max_x >= min_x:
            nx = min(max(nx, min_x), max_x)
        if max_y >= min_y:
            ny = min(max(ny, min_y), max_y)
        if abs(nx - x) > 1e-6:
            p["x"] = float(nx)
            changed += 1
            _record_correction(
                correction_log,
                obj=obj,
                field="position.x",
                before=float(x),
                after=float(nx),
                reason="room_bounds_clamp",
            )
        if abs(ny - y) > 1e-6:
            p["y"] = float(ny)
            changed += 1
            _record_correction(
                correction_log,
                obj=obj,
                field="position.y",
                before=float(y),
                after=float(ny),
                reason="room_bounds_clamp",
            )
        obj["position"] = p
    return changed


def _floor_clamp(
    objs: Sequence[Dict[str, Any]],
    *,
    eps: float = 0.0,
    correction_log: Optional[List[Dict[str, Any]]] = None,
) -> int:
    changed = 0
    for obj in objs:
        p = obj.get("position", {}) or {}
        z = _safe_float(p.get("z"), default=0.0)
        nz = max(float(eps), float(z))
        if abs(nz - z) > 1e-9:
            p["z"] = nz
            obj["position"] = p
            changed += 1
            _record_correction(
                correction_log,
                obj=obj,
                field="position.z",
                before=float(z),
                after=float(nz),
                reason="floor_clamp",
            )
    return changed


def _record_correction(
    correction_log: Optional[List[Dict[str, Any]]],
    *,
    obj: Dict[str, Any],
    field: str,
    before: Any,
    after: Any,
    reason: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if correction_log is None:
        return
    payload: Dict[str, Any] = {
        "object_id": str(obj.get("id", "")),
        "object_type": str(obj.get("type", "")),
        "field": field,
        "before": before,
        "after": after,
        "reason": reason,
    }
    if extra:
        payload.update(extra)
    correction_log.append(payload)


def _normalize_mass_presets(
    objs: Sequence[Dict[str, Any]],
    *,
    correction_log: Optional[List[Dict[str, Any]]] = None,
) -> int:
    changed = 0
    for obj in objs:
        key = _match_mass_preset_key(_object_type(obj))
        if key is None:
            continue
        profile = _MASS_PRESETS_KG[key]
        physics = obj.get("physics", {}) or {}
        mass_before = _safe_float(physics.get("mass"), default=float("nan"))
        mass_after = float(profile["target"])
        if math.isfinite(mass_before) and float(profile["min"]) <= mass_before <= float(profile["max"]):
            continue
        physics["mass"] = mass_after
        obj["physics"] = physics
        changed += 1
        _record_correction(
            correction_log,
            obj=obj,
            field="physics.mass",
            before=(None if not math.isfinite(mass_before) else float(mass_before)),
            after=float(mass_after),
            reason="mass_preset_normalization",
            extra={"preset_key": key, "allowed_min": float(profile["min"]), "allowed_max": float(profile["max"])},
        )
    return changed


def _surface_snap(
    room: Dict[str, Any],
    objs: Sequence[Dict[str, Any]],
    *,
    z_eps: float = 0.005,
    correction_log: Optional[List[Dict[str, Any]]] = None,
) -> int:
    surfaces: List[Tuple[Dict[str, Any], Tuple[float, float, float, float, float, float]]] = []
    for o in objs:
        if not _is_surface(o):
            continue
        a = _aabb(o)
        if a is None:
            continue
        surfaces.append((o, a))

    changed = 0
    for o in objs:
        if _is_surface(o):
            continue
        if not _is_tabletop_candidate(o):
            continue
        a = _aabb(o)
        if a is None:
            continue
        p = o.get("position", {}) or {}
        z = _safe_float(p.get("z"), default=0.0)

        best_top = None
        best_dist = 1e9
        best_support_id = ""
        best_support_xy = 1e9
        best_support_xy_top = None
        obj_center_x = _safe_float(p.get("x"), default=0.0)
        obj_center_y = _safe_float(p.get("y"), default=0.0)
        for s_obj, s_aabb in surfaces:
            s_pos = s_obj.get("position", {}) or {}
            s_cx = _safe_float(s_pos.get("x"), default=0.0)
            s_cy = _safe_float(s_pos.get("y"), default=0.0)
            xy_dist = math.sqrt((obj_center_x - s_cx) ** 2 + (obj_center_y - s_cy) ** 2)
            if xy_dist < best_support_xy:
                best_support_xy = xy_dist
                best_support_xy_top = float(s_aabb[5])

            ox = min(a[3], s_aabb[3]) - max(a[0], s_aabb[0])
            oy = min(a[4], s_aabb[4]) - max(a[1], s_aabb[1])
            if ox <= 0.0 or oy <= 0.0:
                continue
            top = float(s_aabb[5])
            dist = abs(z - top)
            if dist < best_dist:
                best_dist = dist
                best_top = top
                best_support_id = str(s_obj.get("id", ""))

        if best_top is None and best_support_xy_top is not None and z <= 0.08:
            best_top = float(best_support_xy_top)
            best_dist = abs(z - best_top)
            best_support_id = "nearest_surface"
        if best_top is None:
            continue
        # Snap only when already near a plausible surface top.
        if best_dist <= 0.20 or z <= 0.08:
            nz = float(best_top + z_eps)
            if abs(nz - z) > 1e-6:
                p["z"] = nz
                o["position"] = p
                changed += 1
                _record_correction(
                    correction_log,
                    obj=o,
                    field="position.z",
                    before=float(z),
                    after=float(nz),
                    reason=("tabletop_floor_correction" if z <= 0.08 else "surface_snap"),
                    extra={"support_id": best_support_id},
                )
    return changed


def _push_apart_xy(
    room: Dict[str, Any],
    objs: Sequence[Dict[str, Any]],
    *,
    sep_eps: float = 0.002,
    max_step_m: float = 0.25,
    correction_log: Optional[List[Dict[str, Any]]] = None,
) -> int:
    # Accumulate XY displacements.
    n = len(objs)
    deltas = [[0.0, 0.0] for _ in range(n)]
    aabbs = [_aabb(o) for o in objs]

    def area(obj: Dict[str, Any]) -> float:
        d = _dims(obj)
        if d is None:
            return 0.0
        return float(d[0] * d[1])

    areas = [max(area(o), 1e-6) for o in objs]
    fixed = [bool(o.get("fixed", False)) for o in objs]

    for i in range(n):
        ai = aabbs[i]
        if ai is None:
            continue
        for j in range(i + 1, n):
            aj = aabbs[j]
            if aj is None:
                continue
            ox = min(ai[3], aj[3]) - max(ai[0], aj[0])
            oy = min(ai[4], aj[4]) - max(ai[1], aj[1])
            oz = min(ai[5], aj[5]) - max(ai[2], aj[2])
            if ox <= 0.0 or oy <= 0.0 or oz <= 0.0:
                continue
            if fixed[i] and fixed[j]:
                continue

            # If the collision is easiest to resolve vertically, don't push sideways.
            if oz < min(ox, oy):
                continue

            axis = 0 if ox < oy else 1  # 0=x,1=y
            pen = float(min(ox, oy))
            sep = float(pen + sep_eps)

            pi = _pos(objs[i]) or (0.0, 0.0, 0.0)
            pj = _pos(objs[j]) or (0.0, 0.0, 0.0)
            di = pi[axis]
            dj = pj[axis]
            sign = 1.0 if di >= dj else -1.0

            wi = 0.0 if fixed[i] else 1.0 / areas[i]
            wj = 0.0 if fixed[j] else 1.0 / areas[j]
            tot = wi + wj
            if tot <= 0.0:
                continue
            fi = wi / tot
            fj = wj / tot

            if axis == 0:
                deltas[i][0] += sign * sep * fi
                deltas[j][0] -= sign * sep * fj
            else:
                deltas[i][1] += sign * sep * fi
                deltas[j][1] -= sign * sep * fj

    changed = 0
    for idx, obj in enumerate(objs):
        p = obj.get("position", {}) or {}
        x = _safe_float(p.get("x"), default=0.0)
        y = _safe_float(p.get("y"), default=0.0)
        dx, dy = float(deltas[idx][0]), float(deltas[idx][1])
        dx = max(-max_step_m, min(max_step_m, dx))
        dy = max(-max_step_m, min(max_step_m, dy))
        nx, ny = x + dx, y + dy
        if abs(nx - x) > 1e-9:
            p["x"] = float(nx)
            changed += 1
            _record_correction(
                correction_log,
                obj=obj,
                field="position.x",
                before=float(x),
                after=float(nx),
                reason="collision_push_apart",
            )
        if abs(ny - y) > 1e-9:
            p["y"] = float(ny)
            changed += 1
            _record_correction(
                correction_log,
                obj=obj,
                field="position.y",
                before=float(y),
                after=float(ny),
                reason="collision_push_apart",
            )
        obj["position"] = p
    return changed


def repair_room_inplace(
    room: Dict[str, Any],
    *,
    profile: _Profile,
    max_iters: int,
    auto_fix: bool = True,
    max_corrected_ratio: float = 0.20,
) -> Dict[str, Any]:
    objs = list(room.get("objects", []) or [])
    before = evaluate_room(room)
    pass_before, errors_before = _profile_pass(before, profile)
    corrections: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "pass_before": bool(pass_before),
        "errors_before": errors_before,
        "before": before,
        "iters": [],
        "auto_fix": bool(auto_fix),
        "max_corrected_ratio": float(max_corrected_ratio),
    }
    if pass_before and bool(auto_fix):
        report["pass_after"] = True
        report["errors_after"] = []
        report["after"] = before
        report["iterations_used"] = 0
        report["changed"] = False
        report["corrections"] = []
        report["corrected_object_count"] = 0
        report["corrected_object_ratio"] = 0.0
        return report

    changed_any = False
    if not bool(auto_fix):
        after = before
        pass_after, errors_after = _profile_pass(after, profile)
        report["pass_after"] = bool(pass_after)
        report["errors_after"] = errors_after
        report["after"] = after
        report["iterations_used"] = 0
        report["changed"] = False
        report["corrections"] = []
        report["corrected_object_count"] = 0
        report["corrected_object_ratio"] = 0.0
        return report

    for it in range(int(max_iters)):
        it_changed = 0
        it_changed += _normalize_mass_presets(objs, correction_log=corrections)
        it_changed += _push_apart_xy(room, objs, correction_log=corrections)
        it_changed += _floor_clamp(objs, correction_log=corrections)
        it_changed += _surface_snap(room, objs, correction_log=corrections)
        it_changed += _clamp_room_bounds(room, objs, correction_log=corrections)
        changed_any = changed_any or (it_changed > 0)

        metrics = evaluate_room(room)
        ok, errs = _profile_pass(metrics, profile)
        report["iters"].append(
            {
                "iter": it + 1,
                "changed_fields": int(it_changed),
                "pass": bool(ok),
                "errors": errs,
                "metrics": metrics,
            }
        )
        if ok:
            break

    after = evaluate_room(room)
    pass_after, errors_after = _profile_pass(after, profile)
    corrected_object_ids = {
        str(item.get("object_id", "")) for item in corrections if str(item.get("object_id", ""))
    }
    corrected_ratio = float(len(corrected_object_ids) / max(1, len(objs)))
    if corrected_ratio > float(max_corrected_ratio):
        errors_after = list(errors_after) + ["corrected_object_ratio_exceeds_threshold"]
        pass_after = False
    report["pass_after"] = bool(pass_after)
    report["errors_after"] = errors_after
    report["after"] = after
    report["iterations_used"] = len(report["iters"])
    report["changed"] = bool(changed_any)
    report["corrections"] = corrections
    report["corrected_object_count"] = int(len(corrected_object_ids))
    report["corrected_object_ratio"] = round(corrected_ratio, 6)
    return report


def _iter_target_jsons(layout_dir: Path, pose_aug_name: str) -> List[Path]:
    targets: List[Path] = []
    # Base room json(s) first.
    for p in sorted(layout_dir.glob("room_*.json")):
        targets.append(p)
    # Pose-aug variants (if any).
    meta = layout_dir / pose_aug_name / "meta.json"
    if meta.exists():
        for v in _resolve_pose_aug_variants(meta, layout_dir):
            targets.append(v)
    # Deduplicate.
    seen = set()
    out: List[Path] = []
    for p in targets:
        sp = str(p.resolve())
        if sp in seen:
            continue
        seen.add(sp)
        out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="SAGE scene quality gates + repairs (CPU-only)")
    parser.add_argument("--layout_dir", required=True, help="Path to SAGE layout dir (results/layout_*)")
    parser.add_argument("--pose_aug_name", default=os.getenv("POSE_AUG_NAME", "pose_aug_0"))
    parser.add_argument("--profile", default=os.getenv("SAGE_QUALITY_PROFILE", "standard"), choices=sorted(_PROFILES.keys()))
    parser.add_argument("--max_iters", type=int, default=int(os.getenv("SAGE_QUALITY_MAX_ITERS", "6")))
    parser.add_argument("--auto-fix", dest="auto_fix", action="store_true", default=_safe_float(os.getenv("SAGE_AUTO_FIX_LAYOUT", "1"), 1.0) >= 0.5)
    parser.add_argument("--no-auto-fix", dest="auto_fix", action="store_false")
    parser.add_argument("--max-corrected-ratio", type=float, default=float(os.getenv("SAGE_LAYOUT_MAX_CORRECTED_RATIO", "0.20")))
    parser.add_argument("--layout-fix-report", default=os.getenv("SAGE_LAYOUT_FIX_REPORT", ""))
    parser.add_argument("--write", action="store_true", default=True)
    parser.add_argument("--no-write", dest="write", action="store_false")
    args = parser.parse_args()

    layout_dir = Path(args.layout_dir).expanduser().resolve()
    if not layout_dir.exists():
        raise FileNotFoundError(f"layout_dir not found: {layout_dir}")

    profile = _PROFILES[str(args.profile)]
    targets = _iter_target_jsons(layout_dir, str(args.pose_aug_name))
    if not targets:
        _log(f"No target JSONs found under {layout_dir}")
        return 3

    _log(
        f"layout_dir={layout_dir} profile={profile.name} targets={len(targets)} "
        f"max_iters={args.max_iters} auto_fix={args.auto_fix} "
        f"max_corrected_ratio={args.max_corrected_ratio}"
    )

    per_file: List[Dict[str, Any]] = []
    all_pass = True
    any_changed = False
    for path in targets:
        try:
            room = _load_json(path)
            if not isinstance(room, dict):
                raise ValueError("room json is not an object")
        except Exception as exc:
            per_file.append({"path": str(path), "error": f"failed_to_load: {exc}"})
            all_pass = False
            continue

        rep = repair_room_inplace(
            room,
            profile=profile,
            max_iters=int(args.max_iters),
            auto_fix=bool(args.auto_fix),
            max_corrected_ratio=float(args.max_corrected_ratio),
        )
        rep["path"] = str(path)
        per_file.append(rep)
        any_changed = any_changed or bool(rep.get("changed"))
        all_pass = all_pass and bool(rep.get("pass_after"))

        if args.write and bool(rep.get("changed")):
            _write_json(path, room)

    report = {
        "layout_dir": str(layout_dir),
        "profile": profile.name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_iters": int(args.max_iters),
        "write": bool(args.write),
        "all_pass": bool(all_pass),
        "any_changed": bool(any_changed),
        "files": per_file,
    }
    out_path = layout_dir / "quality" / "scene_quality_report.json"
    _write_json(out_path, report)
    _log(f"Wrote report: {out_path} all_pass={all_pass} any_changed={any_changed}")

    raw_layout_fix_report = str(args.layout_fix_report or "").strip()
    if not raw_layout_fix_report:
        layout_fix_path = layout_dir / "quality" / "layout_fix_report.json"
    else:
        layout_fix_path = Path(raw_layout_fix_report).expanduser()
        if not layout_fix_path.is_absolute():
            layout_fix_path = (layout_dir / layout_fix_path).resolve()
    fix_report = {
        "layout_dir": str(layout_dir),
        "profile": profile.name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auto_fix": bool(args.auto_fix),
        "max_corrected_ratio": float(args.max_corrected_ratio),
        "all_pass": bool(all_pass),
        "files": [
            {
                "path": entry.get("path"),
                "pass_after": bool(entry.get("pass_after")),
                "errors_after": entry.get("errors_after", []),
                "corrected_object_count": int(entry.get("corrected_object_count", 0)),
                "corrected_object_ratio": float(entry.get("corrected_object_ratio", 0.0)),
                "corrections": entry.get("corrections", []),
            }
            for entry in per_file
        ],
    }
    _write_json(layout_fix_path, fix_report)
    _log(f"Wrote layout fix report: {layout_fix_path}")

    return 0 if all_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
