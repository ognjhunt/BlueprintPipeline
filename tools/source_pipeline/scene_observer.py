from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _position(obj: Mapping[str, Any]) -> Dict[str, float]:
    transform = obj.get("transform") if isinstance(obj.get("transform"), Mapping) else {}
    pos = transform.get("position") if isinstance(transform.get("position"), Mapping) else {}
    return {
        "x": _safe_float(pos.get("x"), 0.0),
        "y": _safe_float(pos.get("y"), 0.0),
        "z": _safe_float(pos.get("z"), 0.0),
    }


def _room_bounds(room_box: Mapping[str, Any] | None) -> tuple[float, float, float, float]:
    if isinstance(room_box, Mapping):
        room_min = room_box.get("min")
        room_max = room_box.get("max")
        if isinstance(room_min, list) and isinstance(room_max, list) and len(room_min) == 3 and len(room_max) == 3:
            return float(room_min[0]), float(room_max[0]), float(room_min[2]), float(room_max[2])
    return -3.0, 3.0, -3.0, 3.0


def render_ascii_topdown(
    *,
    objects: Sequence[Mapping[str, Any]],
    room_box: Mapping[str, Any] | None,
    width: int = 40,
    height: int = 20,
) -> str:
    x_min, x_max, z_min, z_max = _room_bounds(room_box)
    grid = [["." for _ in range(width)] for _ in range(height)]

    def mark(x: float, z: float, char: str) -> None:
        if x_max <= x_min or z_max <= z_min:
            return
        u = int((x - x_min) / (x_max - x_min) * (width - 1))
        v = int((z - z_min) / (z_max - z_min) * (height - 1))
        u = max(0, min(width - 1, u))
        v = max(0, min(height - 1, v))
        grid[height - 1 - v][u] = char

    for obj in objects:
        role = str(obj.get("sim_role") or "").lower()
        pos = _position(obj)
        if role in {"static", "articulated_furniture", "articulated_appliance"}:
            char = "F"
        elif role in {"manipulable_object", "clutter"}:
            char = "m"
        else:
            char = "o"
        mark(pos["x"], pos["z"], char)

    return "\n".join("".join(row) for row in grid)


def summarize_scene(
    *,
    objects: Sequence[Mapping[str, Any]],
    room_box: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    furniture = 0
    manipulands = 0
    articulated = 0
    for obj in objects:
        role = str(obj.get("sim_role") or "").strip().lower()
        if role in {"static", "articulated_furniture", "articulated_appliance"}:
            furniture += 1
        if role in {"manipulable_object", "clutter"}:
            manipulands += 1
        if role in {"articulated_furniture", "articulated_appliance"}:
            articulated += 1
    return {
        "object_count": len(objects),
        "furniture_count": furniture,
        "manipuland_count": manipulands,
        "articulated_count": articulated,
        "topdown_ascii": render_ascii_topdown(objects=objects, room_box=room_box),
    }

