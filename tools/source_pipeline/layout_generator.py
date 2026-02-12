from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Optional


_ROOM_DIMENSIONS: Dict[str, Dict[str, float]] = {
    "kitchen": {"width": 7.0, "depth": 6.0, "height": 3.1},
    "living_room": {"width": 8.0, "depth": 7.0, "height": 3.2},
    "bedroom": {"width": 6.0, "depth": 5.5, "height": 3.0},
    "office": {"width": 7.0, "depth": 6.0, "height": 3.0},
    "lab": {"width": 8.5, "depth": 7.0, "height": 3.4},
    "warehouse": {"width": 10.0, "depth": 9.0, "height": 4.2},
    "bathroom": {"width": 4.0, "depth": 3.5, "height": 2.8},
}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_layout_plan(
    *,
    room_type: str,
    rng: random.Random,
    constraints: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a single-room architectural shell plan used by Stage 1/2."""

    constraints = constraints or {}
    defaults = _ROOM_DIMENSIONS.get(room_type, {"width": 6.0, "depth": 6.0, "height": 3.0})

    width = max(4.0, _safe_float(constraints.get("room_width_m"), defaults["width"]) + rng.uniform(-0.25, 0.25))
    depth = max(4.0, _safe_float(constraints.get("room_depth_m"), defaults["depth"]) + rng.uniform(-0.25, 0.25))
    height = max(2.4, _safe_float(constraints.get("room_height_m"), defaults["height"]))
    wall_thickness_m = max(0.06, _safe_float(constraints.get("wall_thickness_m"), 0.12))

    room_min = [-round(width * 0.5, 4), 0.0, -round(depth * 0.5, 4)]
    room_max = [round(width * 0.5, 4), round(height, 4), round(depth * 0.5, 4)]

    door_width = max(0.75, min(1.4, _safe_float(constraints.get("door_width_m"), 0.9)))
    window_width = max(0.5, min(2.4, _safe_float(constraints.get("window_width_m"), 1.2)))
    window_sill = max(0.5, min(1.5, _safe_float(constraints.get("window_sill_m"), 1.0)))
    window_height = max(0.5, min(2.0, _safe_float(constraints.get("window_height_m"), 1.0)))

    openings = [
        {
            "id": "opening_door_main",
            "type": "door",
            "wall": "south",
            "center": [0.0, door_width * 0.0, room_min[2]],
            "width_m": round(door_width, 4),
            "height_m": 2.05,
        },
        {
            "id": "opening_window_main",
            "type": "window",
            "wall": "east",
            "center": [room_max[0], round(window_sill + window_height * 0.5, 4), 0.0],
            "width_m": round(window_width, 4),
            "height_m": round(window_height, 4),
            "sill_height_m": round(window_sill, 4),
        },
    ]

    return {
        "schema_version": "v1",
        "room_type": room_type,
        "room_box": {"min": room_min, "max": room_max},
        "wall_thickness_m": round(wall_thickness_m, 4),
        "openings": openings,
    }

