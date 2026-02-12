from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dims(obj: Mapping[str, Any]) -> Dict[str, float]:
    dims = obj.get("dimensions_est") if isinstance(obj.get("dimensions_est"), Mapping) else {}
    return {
        "width": max(0.02, _safe_float(dims.get("width"), 0.25)),
        "height": max(0.02, _safe_float(dims.get("height"), 0.25)),
        "depth": max(0.02, _safe_float(dims.get("depth"), 0.25)),
    }


def _position(obj: Mapping[str, Any]) -> Dict[str, float]:
    transform = obj.get("transform") if isinstance(obj.get("transform"), Mapping) else {}
    pos = transform.get("position") if isinstance(transform.get("position"), Mapping) else {}
    return {
        "x": _safe_float(pos.get("x"), 0.0),
        "y": _safe_float(pos.get("y"), 0.0),
        "z": _safe_float(pos.get("z"), 0.0),
    }


def _set_position(obj: Dict[str, Any], *, x: float, y: float, z: float) -> None:
    transform = obj.setdefault("transform", {})
    if not isinstance(transform, dict):
        transform = {}
        obj["transform"] = transform
    position = transform.setdefault("position", {})
    if not isinstance(position, dict):
        position = {}
        transform["position"] = position
    position["x"] = round(float(x), 4)
    position["y"] = round(float(y), 4)
    position["z"] = round(float(z), 4)


def _is_manipulable(obj: Mapping[str, Any]) -> bool:
    return str(obj.get("sim_role") or "").strip().lower() in {"manipulable_object", "clutter"}


def _aabb(obj: Mapping[str, Any]) -> Tuple[float, float, float, float, float, float]:
    pos = _position(obj)
    dims = _dims(obj)
    x = pos["x"]
    y = pos["y"]
    z = pos["z"]
    half_w = dims["width"] * 0.5
    half_d = dims["depth"] * 0.5
    return (
        x - half_w,
        y,
        z - half_d,
        x + half_w,
        y + dims["height"],
        z + half_d,
    )


def _overlap_1d(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def _boxes_overlap(a: Tuple[float, float, float, float, float, float], b: Tuple[float, float, float, float, float, float], *, margin: float = 0.0) -> bool:
    ox = _overlap_1d(a[0] - margin, a[3] + margin, b[0] - margin, b[3] + margin)
    oy = _overlap_1d(a[1] - margin, a[4] + margin, b[1] - margin, b[4] + margin)
    oz = _overlap_1d(a[2] - margin, a[5] + margin, b[2] - margin, b[5] + margin)
    return ox > 0 and oy > 0 and oz > 0


def check_physics(
    *,
    candidate_obj: Mapping[str, Any],
    placed_objects: Sequence[Mapping[str, Any]],
    room_box: Optional[Mapping[str, Any]],
    margin: float = 0.005,
) -> bool:
    """Cheap placement check: room bounds + AABB non-overlap."""
    box = _aabb(candidate_obj)

    if room_box and isinstance(room_box, Mapping):
        room_min = room_box.get("min")
        room_max = room_box.get("max")
        if isinstance(room_min, list) and isinstance(room_max, list) and len(room_min) == 3 and len(room_max) == 3:
            if box[0] < float(room_min[0]) + margin or box[3] > float(room_max[0]) - margin:
                return False
            if box[2] < float(room_min[2]) + margin or box[5] > float(room_max[2]) - margin:
                return False
            if box[1] < float(room_min[1]) - margin or box[4] > float(room_max[1]) + margin:
                return False

    for other in placed_objects:
        if other.get("id") == candidate_obj.get("id"):
            continue
        if _boxes_overlap(box, _aabb(other), margin=margin):
            return False
    return True


def detect_support_surfaces(objects: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    surfaces: List[Dict[str, Any]] = []
    for obj in objects:
        if _is_manipulable(obj):
            continue
        oid = str(obj.get("id") or "")
        if not oid:
            continue
        dims = _dims(obj)
        if dims["width"] * dims["depth"] < 0.06:
            continue
        pos = _position(obj)
        top_y = pos["y"] + dims["height"]
        half_x = dims["width"] * 0.45
        half_z = dims["depth"] * 0.45
        surfaces.append(
            {
                "id": f"surface_{oid}_top",
                "owner_object_id": oid,
                "kind": "top_face",
                "center": {"x": round(pos["x"], 4), "y": round(top_y, 4), "z": round(pos["z"], 4)},
                "extents": {"x": round(half_x, 4), "z": round(half_z, 4)},
                "normal": [0.0, 1.0, 0.0],
            }
        )
    return surfaces


def sample_surface_se2(surface: Mapping[str, Any], *, rng: random.Random, margin: float = 0.02) -> Dict[str, float]:
    extents = surface.get("extents") if isinstance(surface.get("extents"), Mapping) else {}
    ex = max(0.01, _safe_float(extents.get("x"), 0.1) - margin)
    ez = max(0.01, _safe_float(extents.get("z"), 0.1) - margin)
    return {
        "x": round(rng.uniform(-ex, ex), 4),
        "z": round(rng.uniform(-ez, ez), 4),
        "yaw_rad": round(rng.uniform(-3.14159, 3.14159), 4),
    }


def lift_se2_to_se3(
    *,
    surface: Mapping[str, Any],
    local_se2: Mapping[str, Any],
    object_height: float,
    z_clearance: float = 0.002,
) -> Dict[str, float]:
    center = surface.get("center") if isinstance(surface.get("center"), Mapping) else {}
    return {
        "x": round(_safe_float(center.get("x"), 0.0) + _safe_float(local_se2.get("x"), 0.0), 4),
        "y": round(_safe_float(center.get("y"), 0.0) + z_clearance, 4),
        "z": round(_safe_float(center.get("z"), 0.0) + _safe_float(local_se2.get("z"), 0.0), 4),
        "object_height": round(float(object_height), 4),
    }


def snap_to_object(
    *,
    lifted_pose: Mapping[str, Any],
    surface: Mapping[str, Any],
    padding: float = 0.004,
) -> Dict[str, float]:
    center = surface.get("center") if isinstance(surface.get("center"), Mapping) else {}
    extents = surface.get("extents") if isinstance(surface.get("extents"), Mapping) else {}
    min_x = _safe_float(center.get("x"), 0.0) - _safe_float(extents.get("x"), 0.1) + padding
    max_x = _safe_float(center.get("x"), 0.0) + _safe_float(extents.get("x"), 0.1) - padding
    min_z = _safe_float(center.get("z"), 0.0) - _safe_float(extents.get("z"), 0.1) + padding
    max_z = _safe_float(center.get("z"), 0.0) + _safe_float(extents.get("z"), 0.1) - padding
    x = min(max(_safe_float(lifted_pose.get("x"), 0.0), min_x), max_x)
    z = min(max(_safe_float(lifted_pose.get("z"), 0.0), min_z), max_z)
    return {
        "x": round(x, 4),
        "y": round(_safe_float(lifted_pose.get("y"), 0.0), 4),
        "z": round(z, 4),
    }


def _sample_floor_position(
    *,
    rng: random.Random,
    index: int,
    room_box: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    if room_box and isinstance(room_box, Mapping):
        room_min = room_box.get("min")
        room_max = room_box.get("max")
        if isinstance(room_min, list) and isinstance(room_max, list) and len(room_min) == 3 and len(room_max) == 3:
            x_min = float(room_min[0]) + 0.5
            x_max = float(room_max[0]) - 0.5
            z_min = float(room_min[2]) + 0.5
            z_max = float(room_max[2]) - 0.5
            lanes = 5
            lane = (index - 1) % lanes
            row = (index - 1) // lanes
            lane_x = x_min + ((x_max - x_min) / max(1, lanes - 1)) * lane
            row_step = 1.15
            row_z = z_min + row * row_step
            if row_z > z_max:
                row_z = rng.uniform(z_min, z_max)
            return {
                "x": round(lane_x + rng.uniform(-0.08, 0.08), 4),
                "y": round(max(0.0, rng.uniform(0.0, 0.02)), 4),
                "z": round(row_z + rng.uniform(-0.08, 0.08), 4),
            }
    lane = (index - 1) % 4
    row = (index - 1) // 4
    return {
        "x": round(-2.4 + lane * 1.6 + rng.uniform(-0.06, 0.06), 4),
        "y": round(max(0.0, rng.uniform(0.0, 0.02)), 4),
        "z": round(-2.6 + row * 1.2 + rng.uniform(-0.06, 0.06), 4),
    }


def place_furniture_stage(
    *,
    objects: List[Dict[str, Any]],
    room_box: Optional[Mapping[str, Any]],
    rng: random.Random,
    max_attempts: int = 10,
) -> None:
    placed: List[Dict[str, Any]] = []
    furniture_idx = 0
    for obj in objects:
        if _is_manipulable(obj):
            continue
        furniture_idx += 1
        attempt = 0
        placed_ok = False
        while attempt < max_attempts and not placed_ok:
            attempt += 1
            pos = _sample_floor_position(rng=rng, index=furniture_idx + attempt - 1, room_box=room_box)
            _set_position(obj, **pos)
            if check_physics(candidate_obj=obj, placed_objects=placed, room_box=room_box):
                placed_ok = True
                break
        obj["placement_stage"] = "furniture"
        placed.append(obj)


def place_manipulands_stage(
    *,
    objects: List[Dict[str, Any]],
    room_box: Optional[Mapping[str, Any]],
    rng: random.Random,
    max_attempts: int = 8,
) -> List[Dict[str, Any]]:
    placed = [obj for obj in objects if not _is_manipulable(obj)]
    surfaces = detect_support_surfaces(placed)
    if not surfaces:
        surfaces = [
            {
                "id": "surface_floor_fallback",
                "owner_object_id": "room_floor",
                "kind": "floor",
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "extents": {"x": 2.2, "z": 2.2},
                "normal": [0.0, 1.0, 0.0],
            }
        ]

    manip_idx = 0
    for obj in objects:
        if not _is_manipulable(obj):
            continue
        manip_idx += 1
        dims = _dims(obj)
        selected = surfaces[(manip_idx - 1) % len(surfaces)]
        placed_ok = False
        for _ in range(max_attempts):
            local = sample_surface_se2(selected, rng=rng, margin=0.02)
            lifted = lift_se2_to_se3(surface=selected, local_se2=local, object_height=dims["height"], z_clearance=0.003)
            snapped = snap_to_object(lifted_pose=lifted, surface=selected)
            _set_position(obj, **snapped)
            if check_physics(candidate_obj=obj, placed_objects=placed, room_box=room_box):
                obj["placement_stage"] = "manipulands"
                obj["parent_support_id"] = selected["owner_object_id"]
                obj["surface_local_se2"] = local
                placed.append(obj)
                placed_ok = True
                break

        if not placed_ok:
            floor_pos = _sample_floor_position(rng=rng, index=manip_idx + 100, room_box=room_box)
            _set_position(obj, **floor_pos)
            obj["placement_stage"] = "manipulands"
            obj["parent_support_id"] = "room_floor"
            obj["surface_local_se2"] = {"x": 0.0, "z": 0.0, "yaw_rad": 0.0}
            placed.append(obj)

    return detect_support_surfaces([obj for obj in objects if not _is_manipulable(obj)])

