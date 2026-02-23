#!/usr/bin/env python3
"""
Repair navigation bottlenecks in a SAGE layout JSON before pose augmentation.

Goal:
  Reduce all-variant Stage 4b failures where pick/place are individually reachable
  but no collision-free mobile-base path exists between them.

Strategy:
  - Identify pick and place anchor objects (prefer IDs from policy_analysis).
  - Build a corridor around the segment between anchors.
  - Reposition movable seat-like blockers (chairs/stools/benches) that intersect
    the corridor to nearby free wall-side locations.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_SEAT_TOKENS = ("chair", "stool", "bench", "ottoman", "barstool", "bar_stool")
_PICK_PREF = ("mug", "cup", "glass", "bottle", "can", "jar", "book", "plate", "bowl")
_SURFACE_PREF = ("table", "counter", "desk", "island", "bench")


def _norm(s: Any) -> str:
    return str(s or "").strip().lower().replace(" ", "_")


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _get_objects(layout: Dict[str, Any]) -> List[Dict[str, Any]]:
    rooms = layout.get("rooms")
    if isinstance(rooms, list) and rooms and isinstance(rooms[0], dict):
        objs = rooms[0].get("objects")
        if isinstance(objs, list):
            return objs
    objs = layout.get("objects")
    if isinstance(objs, list):
        return objs
    room = layout.get("room")
    if isinstance(room, dict):
        objs = room.get("objects")
        if isinstance(objs, list):
            return objs
    return []


def _get_room_dims(layout: Dict[str, Any]) -> Tuple[float, float]:
    room = None
    rooms = layout.get("rooms")
    if isinstance(rooms, list) and rooms and isinstance(rooms[0], dict):
        room = rooms[0]
    if room is None:
        room = layout
    dims = room.get("dimensions", {}) if isinstance(room, dict) else {}
    w = _safe_float((dims or {}).get("width"), 6.0)
    l = _safe_float((dims or {}).get("length"), 6.0)
    if not math.isfinite(w) or w <= 1.0:
        w = 6.0
    if not math.isfinite(l) or l <= 1.0:
        l = 6.0
    return float(w), float(l)


def _dims(obj: Dict[str, Any]) -> Tuple[float, float, float]:
    d = obj.get("dimensions", {}) or {}
    w = _safe_float(d.get("width"), 0.6)
    l = _safe_float(d.get("length"), 0.6)
    h = _safe_float(d.get("height"), 0.9)
    return max(w, 0.1), max(l, 0.1), max(h, 0.1)


def _pos(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    p = obj.get("position", {}) or {}
    x = _safe_float(p.get("x"), float("nan"))
    y = _safe_float(p.get("y"), float("nan"))
    z = _safe_float(p.get("z"), float("nan"))
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return float(x), float(y), float(z)


def _find_by_id(objects: Sequence[Dict[str, Any]], oid: str) -> Optional[Dict[str, Any]]:
    for o in objects:
        if str(o.get("id")) == str(oid):
            return o
    return None


def _find_first_type(objects: Sequence[Dict[str, Any]], tokens: Iterable[str]) -> Optional[Dict[str, Any]]:
    token_list = tuple(_norm(t) for t in tokens)
    for o in objects:
        t = _norm(o.get("type"))
        if any(tok in t for tok in token_list):
            return o
    return None


def _extract_task_targets(task_desc: str) -> Tuple[Optional[str], Optional[str]]:
    raw = str(task_desc or "").lower()
    if not raw:
        return None, None
    task = " ".join(raw.replace("_", " ").split())
    pick_hint = None
    place_hint = None
    for tok in _PICK_PREF:
        if tok in task:
            pick_hint = tok
            break
    for tok in _SURFACE_PREF:
        if tok in task:
            place_hint = tok
            break
    return pick_hint, place_hint


def _infer_pick_place_ids(
    layout: Dict[str, Any],
    objects: Sequence[Dict[str, Any]],
    *,
    pick_object_id: str,
    place_surface_id: str,
    task_desc: str,
) -> Tuple[Optional[str], Optional[str], str]:
    pa = layout.get("policy_analysis", {})
    if not isinstance(pa, dict):
        pa = {}
    utd = pa.get("updated_task_decomposition")
    pick_id = None
    place_id = None
    pick_source = "heuristic"
    place_source = "heuristic"

    if pick_object_id and _find_by_id(objects, pick_object_id) is not None:
        pick_id = str(pick_object_id)
        pick_source = "cli_override"
    if place_surface_id and _find_by_id(objects, place_surface_id) is not None:
        place_id = str(place_surface_id)
        place_source = "cli_override"

    if isinstance(utd, list):
        for step in utd:
            if not isinstance(step, dict):
                continue
            action = _norm(step.get("action"))
            if not pick_id and "pick" in action and step.get("target_object_id"):
                maybe = str(step["target_object_id"])
                if _find_by_id(objects, maybe) is not None:
                    pick_id = maybe
                    pick_source = "task_decomposition"
            if not place_id and "place" in action and step.get("location_object_id"):
                maybe = str(step["location_object_id"])
                if _find_by_id(objects, maybe) is not None:
                    place_id = maybe
                    place_source = "task_decomposition"

    pick_hint, place_hint = _extract_task_targets(task_desc)
    if not pick_id and pick_hint:
        p = _find_first_type(objects, (pick_hint,))
        if p is not None and p.get("id") is not None:
            pick_id = str(p["id"])
            pick_source = "task_desc"
    if not place_id and place_hint:
        s = _find_first_type(objects, (place_hint,))
        if s is not None and s.get("id") is not None:
            place_id = str(s["id"])
            place_source = "task_desc"

    if pick_id is None:
        p = _find_first_type(objects, _PICK_PREF)
        if p is not None and p.get("id") is not None:
            pick_id = str(p["id"])
            pick_source = "heuristic"
    if place_id is None:
        s = _find_first_type(objects, _SURFACE_PREF)
        if s is not None and s.get("id") is not None:
            place_id = str(s["id"])
            place_source = "heuristic"
    source = pick_source if pick_source == place_source else f"mixed:{pick_source}+{place_source}"
    return pick_id, place_id, source


def _pt_seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    c1 = vx * wx + vy * wy
    if c1 <= 0.0:
        return math.hypot(px - ax, py - ay)
    c2 = vx * vx + vy * vy
    if c2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = min(1.0, max(0.0, c1 / c2))
    qx = ax + t * vx
    qy = ay + t * vy
    return math.hypot(px - qx, py - qy)


def _is_seat_like(obj: Dict[str, Any]) -> bool:
    t = _norm(obj.get("type"))
    return any(tok in t for tok in _SEAT_TOKENS)


def _clamp_position(x: float, y: float, obj: Dict[str, Any], room_w: float, room_l: float, pad: float) -> Tuple[float, float]:
    w, l, _ = _dims(obj)
    hx = w * 0.5 + pad
    hy = l * 0.5 + pad
    x = min(max(x, hx), room_w - hx)
    y = min(max(y, hy), room_l - hy)
    return float(x), float(y)


def _score_candidate(x: float, y: float, ax: float, ay: float, bx: float, by: float) -> float:
    return _pt_seg_dist(x, y, ax, ay, bx, by)


def _move_off_corridor(
    obj: Dict[str, Any],
    *,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    room_w: float,
    room_l: float,
    corridor_r: float,
    aggressive: bool,
) -> bool:
    p = _pos(obj)
    if p is None:
        return False
    x, y, z = p
    w, l, _ = _dims(obj)
    current_d = _pt_seg_dist(x, y, ax, ay, bx, by)
    thresh = corridor_r + max(w, l) * 0.55
    if current_d > thresh:
        return False

    # Direction perpendicular to path.
    vx = bx - ax
    vy = by - ay
    n = math.hypot(vx, vy)
    if n < 1e-6:
        nx, ny = 1.0, 0.0
    else:
        nx, ny = -vy / n, vx / n

    shift = corridor_r + max(w, l) + (0.45 if aggressive else 0.30)
    pad = 0.06
    candidates: List[Tuple[float, float]] = []

    # Move normal to the corridor.
    candidates.append(_clamp_position(x + nx * shift, y + ny * shift, obj, room_w, room_l, pad))
    candidates.append(_clamp_position(x - nx * shift, y - ny * shift, obj, room_w, room_l, pad))

    # Wall parking fallbacks.
    candidates.append(_clamp_position(0.18, y, obj, room_w, room_l, pad))
    candidates.append(_clamp_position(room_w - 0.18, y, obj, room_w, room_l, pad))
    candidates.append(_clamp_position(x, 0.18, obj, room_w, room_l, pad))
    candidates.append(_clamp_position(x, room_l - 0.18, obj, room_w, room_l, pad))

    best = (x, y)
    best_score = current_d
    for cx, cy in candidates:
        s = _score_candidate(cx, cy, ax, ay, bx, by)
        if s > best_score + 0.08:
            best = (cx, cy)
            best_score = s

    if best == (x, y):
        return False
    obj.setdefault("position", {})
    obj["position"]["x"] = round(best[0], 4)
    obj["position"]["y"] = round(best[1], 4)
    obj["position"]["z"] = z
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Repair navigation corridor bottlenecks in a layout JSON.")
    ap.add_argument("--layout_json", required=True, help="Path to layout_*/layout_*.json")
    ap.add_argument("--min_corridor_m", type=float, default=0.95, help="Target corridor width (meters)")
    ap.add_argument("--max_moves", type=int, default=8, help="Max blocker objects to move")
    ap.add_argument("--aggressive", action="store_true", help="Move more blockers and with larger shift")
    ap.add_argument("--dry_run", action="store_true", help="Analyze but do not write file")
    ap.add_argument("--pick-object-id", default="", help="Explicit pick object id override")
    ap.add_argument("--place-surface-id", default="", help="Explicit place surface id override")
    ap.add_argument("--task-desc", default="", help="Task text used for anchor resolution")
    args = ap.parse_args()

    path = Path(args.layout_json)
    if not path.exists():
        raise FileNotFoundError(f"Layout JSON not found: {path}")

    layout = json.loads(path.read_text(encoding="utf-8"))
    objects = _get_objects(layout)
    if not objects:
        print("[nav-repair] no objects found; skipping")
        return 0

    pick_id, place_id, anchor_source = _infer_pick_place_ids(
        layout,
        objects,
        pick_object_id=str(args.pick_object_id or ""),
        place_surface_id=str(args.place_surface_id or ""),
        task_desc=str(args.task_desc or ""),
    )
    if not pick_id or not place_id:
        print("[nav-repair] could not infer pick/place anchors; skipping")
        return 0

    pick = _find_by_id(objects, pick_id)
    place = _find_by_id(objects, place_id)
    if pick is None or place is None:
        print(f"[nav-repair] anchor ids not found (pick={pick_id}, place={place_id}); skipping")
        return 0

    p_pick = _pos(pick)
    p_place = _pos(place)
    if p_pick is None or p_place is None:
        print("[nav-repair] anchor positions unavailable; skipping")
        return 0

    room_w, room_l = _get_room_dims(layout)
    ax, ay = p_pick[0], p_pick[1]
    bx, by = p_place[0], p_place[1]
    corridor_r = max(0.30, float(args.min_corridor_m) * 0.5)

    moved = []
    max_moves = int(args.max_moves)
    if args.aggressive:
        max_moves = max(max_moves, 12)

    for obj in objects:
        if len(moved) >= max_moves:
            break
        oid = str(obj.get("id", ""))
        if not oid or oid in (pick_id, place_id):
            continue
        if not _is_seat_like(obj):
            continue
        if _move_off_corridor(
            obj,
            ax=ax,
            ay=ay,
            bx=bx,
            by=by,
            room_w=room_w,
            room_l=room_l,
            corridor_r=corridor_r,
            aggressive=bool(args.aggressive),
        ):
            moved.append(oid)

    print(
        "[nav-repair] anchors:",
        f"pick={pick_id}",
        f"place={place_id}",
        f"source={anchor_source}",
        f"corridor_m={float(args.min_corridor_m):.2f}",
        f"moved={len(moved)}",
    )
    if moved:
        print(f"[nav-repair] moved_ids={moved}")

    if args.dry_run:
        return 0

    if moved:
        backup = path.with_suffix(path.suffix + ".navbak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
