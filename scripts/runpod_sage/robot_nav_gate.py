#!/usr/bin/env python3
"""Hard mobile-base navigability gate for SAGE layout JSONs.

Checks whether there exists a collision-free 2D base path from a pick-support
region to a place-support region, using an occupancy grid with robot-footprint
inflation. Intended to fail fast before expensive downstream stages.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


_SMALL_PICK_TOKENS = ("mug", "cup", "glass", "bottle", "can", "jar", "book", "plate", "bowl")
_SURFACE_TOKENS = ("table", "counter", "desk", "island", "bench")


def _norm(s: Any) -> str:
    return str(s or "").strip().lower().replace(" ", "_")


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _get_objects(layout: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rooms = layout.get("rooms")
    if isinstance(rooms, list) and rooms and isinstance(rooms[0], Mapping):
        objects = rooms[0].get("objects")
        if isinstance(objects, list):
            return [dict(o) for o in objects if isinstance(o, Mapping)]

    objects = layout.get("objects")
    if isinstance(objects, list):
        return [dict(o) for o in objects if isinstance(o, Mapping)]

    room = layout.get("room")
    if isinstance(room, Mapping):
        objects = room.get("objects")
        if isinstance(objects, list):
            return [dict(o) for o in objects if isinstance(o, Mapping)]

    return []


def _room_dims(layout: Mapping[str, Any]) -> Tuple[float, float]:
    dims = None
    rooms = layout.get("rooms")
    if isinstance(rooms, list) and rooms and isinstance(rooms[0], Mapping):
        dims = rooms[0].get("dimensions")
    if not isinstance(dims, Mapping):
        dims = layout.get("dimensions")
    if not isinstance(dims, Mapping):
        dims = {}
    w = _safe_float(dims.get("width"), 6.0)
    l = _safe_float(dims.get("length"), 6.0)
    if not math.isfinite(w) or w <= 0.5:
        w = 6.0
    if not math.isfinite(l) or l <= 0.5:
        l = 6.0
    return float(w), float(l)


def _dims(obj: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    d = obj.get("dimensions")
    if not isinstance(d, Mapping):
        d = {}
    w = _safe_float(d.get("width"), float("nan"))
    l = _safe_float(d.get("length"), float("nan"))
    h = _safe_float(d.get("height"), float("nan"))
    if not (math.isfinite(w) and math.isfinite(l) and math.isfinite(h)):
        return None
    if w <= 0.0 or l <= 0.0 or h <= 0.0:
        return None
    return float(w), float(l), float(h)


def _pos(obj: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    p = obj.get("position")
    if not isinstance(p, Mapping):
        p = {}
    x = _safe_float(p.get("x"), float("nan"))
    y = _safe_float(p.get("y"), float("nan"))
    z = _safe_float(p.get("z"), float("nan"))
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return float(x), float(y), float(z)


def _find_by_id(objects: Sequence[Mapping[str, Any]], oid: str) -> Optional[Mapping[str, Any]]:
    for obj in objects:
        if str(obj.get("id")) == str(oid):
            return obj
    return None


def _find_first_by_tokens(objects: Sequence[Mapping[str, Any]], tokens: Iterable[str]) -> Optional[Mapping[str, Any]]:
    token_list = tuple(_norm(t) for t in tokens)
    for obj in objects:
        t = _norm(obj.get("type"))
        if any(tok in t for tok in token_list):
            return obj
    return None


def _infer_anchor_ids(layout: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]) -> Tuple[Optional[str], Optional[str], str]:
    pa = layout.get("policy_analysis")
    if not isinstance(pa, Mapping):
        pa = {}

    utd = pa.get("updated_task_decomposition")
    if not isinstance(utd, list):
        utd = layout.get("updated_task_decomposition")

    pick_id: Optional[str] = None
    place_surface_id: Optional[str] = None
    if isinstance(utd, list):
        for step in utd:
            if not isinstance(step, Mapping):
                continue
            action = _norm(step.get("action"))
            if "pick" in action and step.get("target_object_id"):
                pick_id = str(step.get("target_object_id"))
            if "place" in action and step.get("location_object_id"):
                place_surface_id = str(step.get("location_object_id"))

    if not pick_id or not place_surface_id:
        mros = pa.get("minimum_required_objects")
        if isinstance(mros, list):
            for mro in mros:
                if not isinstance(mro, Mapping):
                    continue
                obj_type = _norm(mro.get("object_type"))
                ids = mro.get("matched_object_ids")
                if not isinstance(ids, list) or not ids:
                    continue
                if not pick_id and any(tok in obj_type for tok in _SMALL_PICK_TOKENS):
                    pick_id = str(ids[0])
                if not place_surface_id and any(tok in obj_type for tok in _SURFACE_TOKENS):
                    place_surface_id = str(ids[0])

    if not pick_id:
        pick = _find_first_by_tokens(objects, _SMALL_PICK_TOKENS)
        if pick and pick.get("id") is not None:
            pick_id = str(pick.get("id"))
    if not place_surface_id:
        place = _find_first_by_tokens(objects, _SURFACE_TOKENS)
        if place and place.get("id") is not None:
            place_surface_id = str(place.get("id"))

    source = "heuristic"
    if isinstance(utd, list):
        source = "task_decomposition"
    elif isinstance(pa.get("minimum_required_objects"), list):
        source = "minimum_required_objects"
    return pick_id, place_surface_id, source


def _build_grid(room_w: float, room_l: float, res: float) -> Tuple[int, int, List[List[bool]]]:
    nx = max(1, int(math.ceil(room_w / res)))
    ny = max(1, int(math.ceil(room_l / res)))
    occupied = [[False for _ in range(ny)] for _ in range(nx)]
    return nx, ny, occupied


def _mark_rect(
    occupied: List[List[bool]],
    *,
    res: float,
    nx: int,
    ny: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    i0 = max(0, int(math.floor(xmin / res)))
    i1 = min(nx - 1, int(math.ceil(xmax / res)) - 1)
    j0 = max(0, int(math.floor(ymin / res)))
    j1 = min(ny - 1, int(math.ceil(ymax / res)) - 1)
    if i1 < i0 or j1 < j0:
        return
    for i in range(i0, i1 + 1):
        row = occupied[i]
        for j in range(j0, j1 + 1):
            row[j] = True


def _is_small_pick_like(obj: Mapping[str, Any], obstacle_min_size: float) -> bool:
    t = _norm(obj.get("type"))
    if any(tok in t for tok in _SMALL_PICK_TOKENS):
        return True
    d = _dims(obj)
    if d is None:
        return False
    w, l, _ = d
    return max(w, l) < obstacle_min_size


def _cell_center(i: int, j: int, res: float) -> Tuple[float, float]:
    return (i + 0.5) * res, (j + 0.5) * res


def _sample_anchor_cells(
    occupied: List[List[bool]],
    *,
    nx: int,
    ny: int,
    res: float,
    anchor_xy: Tuple[float, float],
    rmin: float,
    rmax: float,
) -> Set[Tuple[int, int]]:
    ax, ay = anchor_xy
    cells: Set[Tuple[int, int]] = set()
    i0 = max(0, int(math.floor((ax - rmax) / res)))
    i1 = min(nx - 1, int(math.ceil((ax + rmax) / res)))
    j0 = max(0, int(math.floor((ay - rmax) / res)))
    j1 = min(ny - 1, int(math.ceil((ay + rmax) / res)))

    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            if occupied[i][j]:
                continue
            cx, cy = _cell_center(i, j, res)
            d = math.hypot(cx - ax, cy - ay)
            if rmin <= d <= rmax:
                cells.add((i, j))
    return cells


def _path_exists(
    occupied: List[List[bool]],
    *,
    nx: int,
    ny: int,
    starts: Set[Tuple[int, int]],
    goals: Set[Tuple[int, int]],
) -> bool:
    if not starts or not goals:
        return False
    goals_set = set(goals)
    q: deque[Tuple[int, int]] = deque()
    seen: Set[Tuple[int, int]] = set()
    for s in starts:
        if s in goals_set:
            return True
        q.append(s)
        seen.add(s)

    while q:
        i, j = q.popleft()
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                continue
            if occupied[ni][nj]:
                continue
            node = (ni, nj)
            if node in seen:
                continue
            if node in goals_set:
                return True
            seen.add(node)
            q.append(node)
    return False


def _robot_radius_m(explicit: float) -> float:
    if explicit > 0:
        return explicit
    base_l = _safe_float(os.getenv("OMRON_BASE_LENGTH_M"), 0.707)
    base_w = _safe_float(os.getenv("OMRON_BASE_WIDTH_M"), 0.530)
    safety = _safe_float(os.getenv("OMRON_OCCUPANCY_SAFETY_MARGIN_M"), 0.04)
    half_diag = math.hypot(base_l * 0.5, base_w * 0.5)
    return max(0.2, half_diag + max(0.0, safety))


def run_gate(
    *,
    layout: Mapping[str, Any],
    grid_res_m: float,
    pick_radius_min_m: float,
    pick_radius_max_m: float,
    robot_radius_m: float,
    obstacle_min_size_m: float,
) -> Dict[str, Any]:
    objects = _get_objects(layout)
    room_w, room_l = _room_dims(layout)
    pick_id, place_id, anchor_source = _infer_anchor_ids(layout, objects)

    report: Dict[str, Any] = {
        "status": "fail",
        "anchor_source": anchor_source,
        "pick_object_id": pick_id,
        "place_surface_id": place_id,
        "grid_res_m": grid_res_m,
        "room_dims_m": {"width": room_w, "length": room_l},
        "robot_radius_m": robot_radius_m,
        "pick_radius_min_m": pick_radius_min_m,
        "pick_radius_max_m": pick_radius_max_m,
    }

    if not pick_id or not place_id:
        report["reason"] = "anchors_missing"
        return report

    pick_obj = _find_by_id(objects, pick_id)
    place_obj = _find_by_id(objects, place_id)
    if pick_obj is None or place_obj is None:
        report["reason"] = "anchor_ids_not_found"
        return report

    pick_pos = _pos(pick_obj)
    place_pos = _pos(place_obj)
    if pick_pos is None or place_pos is None:
        report["reason"] = "anchor_positions_missing"
        return report

    nx, ny, occupied = _build_grid(room_w, room_l, grid_res_m)
    blocked_cells = 0
    obstacle_count = 0

    for obj in objects:
        oid = str(obj.get("id") or "")
        if oid and oid == pick_id and _is_small_pick_like(obj, obstacle_min_size_m):
            continue
        if _is_small_pick_like(obj, obstacle_min_size_m):
            continue
        p = _pos(obj)
        d = _dims(obj)
        if p is None or d is None:
            continue
        x, y, _z = p
        w, l, _h = d
        obstacle_count += 1
        _mark_rect(
            occupied,
            res=grid_res_m,
            nx=nx,
            ny=ny,
            xmin=x - (w * 0.5 + robot_radius_m),
            xmax=x + (w * 0.5 + robot_radius_m),
            ymin=y - (l * 0.5 + robot_radius_m),
            ymax=y + (l * 0.5 + robot_radius_m),
        )

    for i in range(nx):
        row = occupied[i]
        blocked_cells += sum(1 for v in row if v)

    pick_cells = _sample_anchor_cells(
        occupied,
        nx=nx,
        ny=ny,
        res=grid_res_m,
        anchor_xy=(pick_pos[0], pick_pos[1]),
        rmin=pick_radius_min_m,
        rmax=pick_radius_max_m,
    )
    place_cells = _sample_anchor_cells(
        occupied,
        nx=nx,
        ny=ny,
        res=grid_res_m,
        anchor_xy=(place_pos[0], place_pos[1]),
        rmin=pick_radius_min_m,
        rmax=pick_radius_max_m,
    )

    path_ok = _path_exists(occupied, nx=nx, ny=ny, starts=pick_cells, goals=place_cells)
    report.update(
        {
            "obstacle_object_count": obstacle_count,
            "grid_cells_total": nx * ny,
            "grid_cells_blocked": blocked_cells,
            "grid_blocked_ratio": round((blocked_cells / float(max(1, nx * ny))), 6),
            "pick_candidate_cells": len(pick_cells),
            "place_candidate_cells": len(place_cells),
            "path_exists": bool(path_ok),
            "pick_anchor_xy": {"x": pick_pos[0], "y": pick_pos[1]},
            "place_anchor_xy": {"x": place_pos[0], "y": place_pos[1]},
        }
    )

    if len(pick_cells) == 0 or len(place_cells) == 0:
        report["reason"] = "no_anchor_candidates"
        return report
    if not path_ok:
        report["reason"] = "no_path_between_anchor_regions"
        return report

    report["status"] = "pass"
    report["reason"] = "ok"
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Hard robot-nav gate for layout acceptance.")
    ap.add_argument("--layout_json", required=True, help="Path to layout_*/layout_*.json")
    ap.add_argument("--report_path", default="", help="Optional JSON report output path")
    ap.add_argument("--grid_res_m", type=float, default=0.05)
    ap.add_argument("--pick_radius_min_m", type=float, default=0.55)
    ap.add_argument("--pick_radius_max_m", type=float, default=1.40)
    ap.add_argument("--robot_radius_m", type=float, default=0.0, help="Override inflated robot radius (m); 0=auto")
    ap.add_argument("--obstacle_min_size_m", type=float, default=0.25)
    args = ap.parse_args()

    layout_path = Path(args.layout_json)
    if not layout_path.is_file():
        raise FileNotFoundError(f"Layout JSON not found: {layout_path}")

    layout = _load_json(layout_path)
    report = run_gate(
        layout=layout if isinstance(layout, Mapping) else {},
        grid_res_m=max(0.02, float(args.grid_res_m)),
        pick_radius_min_m=max(0.2, float(args.pick_radius_min_m)),
        pick_radius_max_m=max(float(args.pick_radius_min_m) + 0.1, float(args.pick_radius_max_m)),
        robot_radius_m=_robot_radius_m(float(args.robot_radius_m)),
        obstacle_min_size_m=max(0.05, float(args.obstacle_min_size_m)),
    )

    if args.report_path:
        _write_json(Path(args.report_path), report)

    print(
        f"[nav-gate] status={report.get('status')} "
        f"reason={report.get('reason')} "
        f"pick_cells={report.get('pick_candidate_cells', 0)} "
        f"place_cells={report.get('place_candidate_cells', 0)} "
        f"path_exists={report.get('path_exists', False)}"
    )
    if report.get("status") == "pass":
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
