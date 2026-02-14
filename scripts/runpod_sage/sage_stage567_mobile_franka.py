#!/usr/bin/env python3
"""
Strict SAGE-owned Stage 5–7 runner for Mobile Franka (RidgebackFranka).

Stage 4 (pose augmentation) is assumed to have produced:
  results/<layout_id>/<pose_aug_name>/meta.json

This script enforces that:
- Stage 5 uses SAGE's M2T2 utilities (nvdiffrast-based render path) with real weights.
- Stage 6 uses cuRobo motion planning (no linear/placeholder fallbacks in strict mode).
- Stage 7 runs Isaac Sim headless capture and writes robomimic-compatible HDF5.

BlueprintPipeline is used only for pre-collection "SimReady Lite" physics presets
and post-collection scoring (handled elsewhere).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SAGE_SERVER_DIR = os.environ.get("SAGE_SERVER_DIR", "/workspace/SAGE/server")
sys.path.insert(0, SAGE_SERVER_DIR)


M2T2_WEIGHTS_DEFAULT = "/workspace/SAGE/M2T2/m2t2.pth"
ISAACSIM_PY_DEFAULT = "/workspace/isaacsim_env/bin/python3"


def _log(msg: str) -> None:
    print(f"[sage-stage567 {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _room_json_for_layout_dir(layout_dir: Path) -> Path:
    room_jsons = sorted(layout_dir.glob("room_*.json"))
    if not room_jsons:
        raise FileNotFoundError(f"No room_*.json found in {layout_dir}")
    return room_jsons[0]


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
            # Try relative to meta dir first, then layout dir.
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
        if str(p) in seen:
            continue
        seen.add(str(p))
        out.append(p)
    return out


def _normalize_type(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")


def _select_pick_and_place(
    room: Dict[str, Any],
    *,
    task_desc: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    objects = list(room.get("objects", []))
    if not objects:
        raise ValueError("Room has 0 objects")

    task = (task_desc or "").lower()

    def dims(obj: Dict[str, Any]) -> Tuple[float, float, float]:
        d = obj.get("dimensions", {}) or {}
        return float(d.get("width", 1.0)), float(d.get("length", 1.0)), float(d.get("height", 1.0))

    # Heuristic: manipulable = small footprint.
    manipulable: List[Dict[str, Any]] = []
    for obj in objects:
        w, l, h = dims(obj)
        if min(w, l) <= 0.20 and obj.get("source_id"):
            manipulable.append(obj)

    if not manipulable:
        # Fallback: pick the smallest object with a mesh.
        manipulable = [o for o in objects if o.get("source_id")]
        manipulable.sort(key=lambda o: dims(o)[0] * dims(o)[1] * dims(o)[2])

    surface_tokens = ("table", "counter", "desk", "island", "bench")
    surfaces = []
    for obj in objects:
        t = _normalize_type(obj.get("type", ""))
        if any(tok in t for tok in surface_tokens):
            surfaces.append(obj)
    if not surfaces:
        # Fallback: choose largest footprint object (often a surface).
        surfaces = sorted(objects, key=lambda o: dims(o)[0] * dims(o)[1], reverse=True)

    # Try to match task keywords.
    pick = None
    for obj in manipulable:
        t = _normalize_type(obj.get("type", ""))
        if t and t in task:
            pick = obj
            break
        if "mug" in task and "mug" in t:
            pick = obj
            break
        if "cup" in task and "cup" in t:
            pick = obj
            break
    if pick is None:
        pick = manipulable[0]

    place = None
    for obj in surfaces:
        t = _normalize_type(obj.get("type", ""))
        if any(tok in task for tok in surface_tokens) and any(tok in t for tok in surface_tokens):
            place = obj
            break
    if place is None:
        place = surfaces[0]

    return pick, place


def _build_occupancy_grid(
    room: Dict[str, Any],
    *,
    resolution_m: float = 0.05,
    margin_m: float = 0.35,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    dims = room.get("dimensions", {}) or {}
    width = float(dims.get("width", 6.0))
    length = float(dims.get("length", 6.0))
    nx = max(1, int(math.ceil(width / resolution_m)))
    ny = max(1, int(math.ceil(length / resolution_m)))
    grid = np.zeros((ny, nx), dtype=np.uint8)  # y,x

    def mark_rect(cx: float, cy: float, w: float, l: float) -> None:
        x0 = int(max(0, math.floor((cx - w / 2 - margin_m) / resolution_m)))
        x1 = int(min(nx - 1, math.ceil((cx + w / 2 + margin_m) / resolution_m)))
        y0 = int(max(0, math.floor((cy - l / 2 - margin_m) / resolution_m)))
        y1 = int(min(ny - 1, math.ceil((cy + l / 2 + margin_m) / resolution_m)))
        grid[y0 : y1 + 1, x0 : x1 + 1] = 1

    for obj in room.get("objects", []) or []:
        pos = obj.get("position", {}) or {}
        d = obj.get("dimensions", {}) or {}
        cx = float(pos.get("x", 0.0))
        cy = float(pos.get("y", 0.0))
        w = float(d.get("width", 0.0))
        l = float(d.get("length", 0.0))
        if w > 0.0 and l > 0.0:
            mark_rect(cx, cy, w, l)

    return grid, {"resolution_m": resolution_m, "width_m": width, "length_m": length}


def _grid_free(grid: np.ndarray, grid_meta: Dict[str, Any], x: float, y: float) -> bool:
    res = float(grid_meta["resolution_m"])
    nx = grid.shape[1]
    ny = grid.shape[0]
    gx = int(round(x / res))
    gy = int(round(y / res))
    if gx < 0 or gx >= nx or gy < 0 or gy >= ny:
        return False
    return bool(grid[gy, gx] == 0)


def _sample_base_pose_near(
    *,
    target_xy: Tuple[float, float],
    room_dims: Dict[str, Any],
    grid: np.ndarray,
    grid_meta: Dict[str, Any],
    approach_dist: float = 0.65,
) -> Tuple[float, float, float]:
    tx, ty = target_xy
    width = float(room_dims.get("width", 6.0))
    length = float(room_dims.get("length", 6.0))
    margin = 0.45

    best = None
    best_clear = -1.0

    for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
        bx = tx + approach_dist * math.cos(angle)
        by = ty + approach_dist * math.sin(angle)
        if not (margin < bx < width - margin and margin < by < length - margin):
            continue
        if not _grid_free(grid, grid_meta, bx, by):
            continue
        # clearance score: count free neighbors in radius 2 cells
        res = float(grid_meta["resolution_m"])
        gx = int(round(bx / res))
        gy = int(round(by / res))
        patch = grid[max(0, gy - 2) : min(grid.shape[0], gy + 3), max(0, gx - 2) : min(grid.shape[1], gx + 3)]
        clear = float((patch == 0).mean())
        if clear > best_clear:
            best_clear = clear
            yaw = float(angle + math.pi)  # face target
            best = (bx, by, yaw)

    if best is None:
        # Last resort: center-ish.
        bx = float(np.clip(tx - approach_dist, margin, width - margin))
        by = float(np.clip(ty, margin, length - margin))
        yaw = 0.0
        return bx, by, yaw
    return best


def _astar_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    # 8-connected A*.
    import heapq

    def h(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    open_heap: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            # reconstruct
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in neighbors:
            nx = current[0] + dx
            ny = current[1] + dy
            if ny < 0 or ny >= grid.shape[0] or nx < 0 or nx >= grid.shape[1]:
                continue
            if grid[ny, nx] != 0:
                continue
            tentative = g_score[current] + (1.4142 if dx != 0 and dy != 0 else 1.0)
            nxt = (nx, ny)
            if tentative < g_score.get(nxt, 1e9):
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + h(nxt, goal)
                heapq.heappush(open_heap, (f, nxt))
    return None


def _plan_base_traj(
    *,
    grid: np.ndarray,
    grid_meta: Dict[str, Any],
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    fixed_yaw: float,
) -> np.ndarray:
    res = float(grid_meta["resolution_m"])
    sx = int(round(start_xy[0] / res))
    sy = int(round(start_xy[1] / res))
    gx = int(round(goal_xy[0] / res))
    gy = int(round(goal_xy[1] / res))
    path = _astar_path(grid, (sx, sy), (gx, gy))
    if not path:
        raise RuntimeError("Base A* planning failed (no path)")
    pts = []
    for x, y in path:
        pts.append([float(x * res), float(y * res), float(fixed_yaw)])
    return np.asarray(pts, dtype=np.float32)


def _quat_from_rot(R: np.ndarray) -> List[float]:
    # Robust rotation-matrix -> quaternion (wxyz).
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(m))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        # Find major diagonal element
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def _plan_arm_segments_with_curobo(
    *,
    grasp_pose_base: np.ndarray,
    place_pose_base: np.ndarray,
    motion_gen,
    strict: bool,
) -> Dict[str, np.ndarray]:
    import torch
    from curobo.types.math import Pose

    # Franka home joints (7)
    home = np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

    def plan_to_pose(start_q: np.ndarray, target_T: np.ndarray) -> np.ndarray:
        pos = target_T[:3, 3].astype(np.float32)
        quat = _quat_from_rot(target_T[:3, :3])
        target = Pose(position=pos.tolist(), quaternion=quat)
        start_state = torch.tensor(start_q, dtype=torch.float32)
        result = motion_gen.plan_single(start_state, target)
        if not bool(getattr(result, "success", False)):
            raise RuntimeError("cuRobo plan_single failed")
        traj = result.get_interpolated_plan()
        return traj.cpu().numpy()

    # Pregrasp/backoff along grasp approach (-Z of grasp frame).
    approach = -grasp_pose_base[:3, 2]
    approach = approach / max(float(np.linalg.norm(approach)), 1e-6)
    pregrasp = grasp_pose_base.copy()
    pregrasp[:3, 3] = grasp_pose_base[:3, 3] + 0.10 * approach
    lift = grasp_pose_base.copy()
    lift[:3, 3] = grasp_pose_base[:3, 3] + np.asarray([0.0, 0.0, 0.10], dtype=np.float32)

    preplace = place_pose_base.copy()
    preplace[:3, 3] = place_pose_base[:3, 3] + np.asarray([0.0, 0.0, 0.10], dtype=np.float32)
    retreat = preplace.copy()

    q0 = home
    q_pregrasp = plan_to_pose(q0, pregrasp)
    q_grasp = plan_to_pose(q_pregrasp[-1], grasp_pose_base)
    q_lift = plan_to_pose(q_grasp[-1], lift)
    q_preplace = plan_to_pose(q_lift[-1], preplace)
    q_place = plan_to_pose(q_preplace[-1], place_pose_base)
    q_retreat = plan_to_pose(q_place[-1], retreat)

    return {
        "home": home,
        "approach_pick": q_pregrasp,
        "grasp": q_grasp,
        "lift": q_lift,
        "approach_place": q_preplace,
        "place": q_place,
        "retreat_place": q_retreat,
    }


def _init_curobo_motion_gen():
    try:
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"cuRobo not available: {exc}") from exc

    config = MotionGenConfig.load_from_robot_config("franka.yml", interpolation_dt=0.02)
    mg = MotionGen(config)
    mg.warmup()
    return mg


def _call_generate_m2t2_data(
    generate_fn,
    *,
    layout_id: str,
    layout_dir: Path,
    layout_dict_path: Path,
    num_views: int,
    strict: bool,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    sig = inspect.signature(generate_fn)
    params = sig.parameters

    kwargs: Dict[str, Any] = {}
    if "layout_id" in params:
        kwargs["layout_id"] = layout_id
    if "layout_dir" in params:
        kwargs["layout_dir"] = str(layout_dir)
    if "results_dir" in params:
        kwargs["results_dir"] = str(layout_dir.parent)
    if "layout_dict_path" in params:
        kwargs["layout_dict_path"] = str(layout_dict_path)
    if "layout_dict_save_path" in params:
        kwargs["layout_dict_save_path"] = str(layout_dict_path)
    if "room_dict_save_path" in params:
        kwargs["room_dict_save_path"] = str(layout_dict_path)
    if "num_views" in params:
        kwargs["num_views"] = int(num_views)
    if "num_viewpoints" in params:
        kwargs["num_viewpoints"] = int(num_views)

    missing_required = [
        name
        for name, p in params.items()
        if p.default is inspect._empty and name not in kwargs
    ]
    if missing_required:
        raise RuntimeError(
            "generate_m2t2_data signature mismatch. Missing required params: "
            f"{missing_required}. Full signature: {sig}"
        )

    out = generate_fn(**kwargs)

    # Normalize output into list[(meta_data, vis_data)]
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    if isinstance(out, tuple) and len(out) == 2:
        meta, vis = out
        if isinstance(meta, list) and isinstance(vis, list):
            for m, v in zip(meta, vis):
                pairs.append((m, v))
        elif isinstance(meta, dict) and isinstance(vis, dict):
            pairs.append((meta, vis))
    elif isinstance(out, dict):
        # Common forms: {"meta_data": ..., "vis_data": ...} or plural.
        if "meta_data" in out and "vis_data" in out and isinstance(out["meta_data"], dict) and isinstance(out["vis_data"], dict):
            pairs.append((out["meta_data"], out["vis_data"]))
        elif "meta_datas" in out and "vis_datas" in out and isinstance(out["meta_datas"], list) and isinstance(out["vis_datas"], list):
            for m, v in zip(out["meta_datas"], out["vis_datas"]):
                pairs.append((m, v))
    elif isinstance(out, list):
        for item in out:
            if isinstance(item, dict) and "meta_data" in item and "vis_data" in item:
                pairs.append((item["meta_data"], item["vis_data"]))

    if not pairs:
        raise RuntimeError(
            "generate_m2t2_data returned an unsupported structure. "
            f"type={type(out).__name__}"
        )
    return pairs


def _infer_grasps_for_variant(
    *,
    layout_id: str,
    layout_dir: Path,
    variant_layout_json: Path,
    num_views: int,
    strict: bool,
    model,
    cfg,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from m2t2_utils.data import generate_m2t2_data
        from m2t2_utils.infer import infer_m2t2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Cannot import SAGE M2T2 utilities: {exc}") from exc

    pairs = _call_generate_m2t2_data(
        generate_m2t2_data,
        layout_id=layout_id,
        layout_dir=layout_dir,
        layout_dict_path=variant_layout_json,
        num_views=num_views,
        strict=strict,
    )

    all_grasps: List[np.ndarray] = []
    all_contacts: List[np.ndarray] = []
    for meta_data, vis_data in pairs:
        grasps_out = infer_m2t2(meta_data, vis_data, model, cfg, return_contacts=True)
        if isinstance(grasps_out, tuple) and len(grasps_out) == 2:
            grasps, contacts = grasps_out
        else:
            grasps, contacts = grasps_out, np.zeros((0, 3), dtype=np.float32)
        grasps = np.asarray(grasps)
        contacts = np.asarray(contacts)
        if grasps.size == 0:
            continue
        all_grasps.append(grasps)
        all_contacts.append(contacts if contacts.size else np.zeros((grasps.shape[0], 3), dtype=np.float32))

    if not all_grasps:
        return np.zeros((0, 4, 4), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    grasps = np.concatenate(all_grasps, axis=0)
    contacts = np.concatenate(all_contacts, axis=0) if all_contacts else np.zeros((grasps.shape[0], 3), dtype=np.float32)
    return grasps.astype(np.float32), contacts.astype(np.float32)


def _world_from_base(base_pose: Sequence[float]) -> np.ndarray:
    x, y, yaw = float(base_pose[0]), float(base_pose[1]), float(base_pose[2])
    c = math.cos(yaw)
    s = math.sin(yaw)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    T[:3, 3] = np.asarray([x, y, 0.0], dtype=np.float32)
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = R.T
    inv[:3, 3] = -(R.T @ t)
    return inv


def _default_place_pose_base(place_surface: Dict[str, Any], base_pose: Sequence[float]) -> np.ndarray:
    pos = place_surface.get("position", {}) or {}
    dims = place_surface.get("dimensions", {}) or {}
    world_place = np.eye(4, dtype=np.float32)
    world_place[:3, 3] = np.asarray(
        [
            float(pos.get("x", 0.0)),
            float(pos.get("y", 0.0)),
            float(pos.get("z", 0.0)) + float(dims.get("height", 0.0)) + 0.05,
        ],
        dtype=np.float32,
    )
    # Top-down orientation
    world_place[:3, :3] = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    base_T_world = _invert_T(_world_from_base(base_pose))
    return base_T_world @ world_place


@dataclass
class DemoPlan:
    demo_idx: int
    variant_layout_json: str
    pick_object: Dict[str, Any]
    place_surface: Dict[str, Any]
    base_pick: List[float]
    base_place: List[float]
    trajectory_base: List[List[float]]
    trajectory_arm: List[List[float]]
    trajectory_gripper: List[List[float]]
    step_labels: List[str]
    chosen_grasp_T_world: List[List[float]]


def main() -> None:
    parser = argparse.ArgumentParser(description="SAGE stages 5–7 (mobile_franka) — strict runner")
    parser.add_argument("--layout_id", required=True)
    parser.add_argument("--results_dir", default="/workspace/SAGE/server/results")
    parser.add_argument("--pose_aug_name", default="pose_aug_0")
    parser.add_argument("--num_demos", type=int, required=True)
    parser.add_argument("--num_views_m2t2", type=int, default=1)
    parser.add_argument("--grasp_top_k", type=int, default=8)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--task_desc", default=os.getenv("TASK_DESC", ""))

    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--enable_cameras", dest="enable_cameras", action="store_true", default=True)
    parser.add_argument("--disable_cameras", dest="enable_cameras", action="store_false")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")

    parser.add_argument("--m2t2_weights", default=M2T2_WEIGHTS_DEFAULT)
    parser.add_argument("--isaacsim_py", default=ISAACSIM_PY_DEFAULT)
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0") or "0"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    layout_dir = Path(args.results_dir) / args.layout_id
    if not layout_dir.exists():
        raise FileNotFoundError(f"Layout dir not found: {layout_dir}")

    if not args.output_dir:
        args.output_dir = str(layout_dir / "demos")

    meta_path = layout_dir / args.pose_aug_name / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing pose augmentation meta: {meta_path}")

    variants = _resolve_pose_aug_variants(meta_path, layout_dir)
    if args.strict and not variants:
        raise RuntimeError(f"Pose augmentation meta contains 0 resolvable layouts: {meta_path}")

    weights_path = Path(args.m2t2_weights)
    if args.strict and not weights_path.exists():
        raise FileNotFoundError(f"M2T2 weights missing: {weights_path}")

    _log(f"layout_id={args.layout_id}")
    _log(f"pose_aug_variants={len(variants)}")
    _log(f"num_demos={args.num_demos} strict={args.strict} cameras={args.enable_cameras} headless={args.headless}")

    # Load M2T2 once.
    try:
        from m2t2_utils.infer import load_m2t2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Cannot import m2t2_utils.infer.load_m2t2: {exc}") from exc

    _log("Loading M2T2 model...")
    t0 = time.time()
    model, cfg = load_m2t2()
    _log(f"M2T2 loaded in {time.time() - t0:.1f}s")

    _log("Initializing cuRobo MotionGen...")
    t1 = time.time()
    motion_gen = _init_curobo_motion_gen()
    _log(f"cuRobo MotionGen ready in {time.time() - t1:.1f}s")

    plans: List[DemoPlan] = []
    all_grasps_for_report: List[np.ndarray] = []
    all_contacts_for_report: List[np.ndarray] = []

    for demo_idx in range(args.num_demos):
        # Deterministic variant selection (rotate) for reproducibility.
        variant_json = variants[demo_idx % len(variants)]
        room = _load_json(variant_json)
        room_dims = room.get("dimensions", {}) or {}

        pick_obj, place_surface = _select_pick_and_place(room, task_desc=args.task_desc)
        grid, grid_meta = _build_occupancy_grid(room)

        pick_xy = (float(pick_obj["position"]["x"]), float(pick_obj["position"]["y"]))
        place_xy = (float(place_surface["position"]["x"]), float(place_surface["position"]["y"]))

        base_pick = _sample_base_pose_near(
            target_xy=pick_xy, room_dims=room_dims, grid=grid, grid_meta=grid_meta
        )
        base_place = _sample_base_pose_near(
            target_xy=place_xy, room_dims=room_dims, grid=grid, grid_meta=grid_meta
        )

        _log(f"demo={demo_idx}: variant={variant_json.name} pick={pick_obj.get('type')} place={place_surface.get('type')}")

        grasps_world, contacts = _infer_grasps_for_variant(
            layout_id=args.layout_id,
            layout_dir=layout_dir,
            variant_layout_json=variant_json,
            num_views=args.num_views_m2t2,
            strict=args.strict,
            model=model,
            cfg=cfg,
        )
        if args.strict and grasps_world.shape[0] == 0:
            raise RuntimeError(f"Stage 5 produced 0 grasps for demo={demo_idx} (variant={variant_json})")

        # Choose grasp using top-K candidates and require first successful cuRobo plan.
        top_k = max(1, int(args.grasp_top_k))
        grasp_candidates = min(grasps_world.shape[0], top_k)
        if args.strict and grasp_candidates <= 0:
            raise RuntimeError(f"Stage 5 produced 0 grasps after filtering for demo={demo_idx} (variant={variant_json})")

        # Convert base pose for frame conversion once.
        base_T_world = _invert_T(_world_from_base(base_pick))
        place_pose_base = _default_place_pose_base(place_surface, base_place)

        chosen_grasp_world: Optional[np.ndarray] = None
        chosen_segments: Optional[Dict[str, np.ndarray]] = None
        chosen_idx = -1
        plan_error = None
        for cand_idx in range(grasp_candidates):
            grasp_world = grasps_world[cand_idx]
            try:
                grasp_pose_base = base_T_world @ grasp_world
                segments = _plan_arm_segments_with_curobo(
                    grasp_pose_base=grasp_pose_base,
                    place_pose_base=place_pose_base,
                    motion_gen=motion_gen,
                    strict=args.strict,
                )
                chosen_grasp_world = grasp_world
                chosen_segments = segments
                chosen_idx = cand_idx
                break
            except Exception as e:  # pragma: no cover
                plan_error = str(e)
                _log(f"demo={demo_idx}: grasp candidate {cand_idx + 1}/{grasp_candidates} failed: {e}")
                continue

        if chosen_segments is None or chosen_grasp_world is None:
            raise RuntimeError(
                f"Stage 6 failed to plan arm trajectory for demo={demo_idx}. "
                f"No feasible grasps in top-K ({grasp_candidates}). last_error={plan_error}"
            )

        all_grasps_for_report.append(chosen_grasp_world[None, ...])
        all_contacts_for_report.append((contacts[chosen_idx : chosen_idx + 1] if contacts.shape[0] else np.zeros((1, 3), dtype=np.float32)))

        # Base path (A*), yaw fixed to face forward.
        base_path = _plan_base_traj(
            grid=grid,
            grid_meta=grid_meta,
            start_xy=(base_pick[0], base_pick[1]),
            goal_xy=(base_place[0], base_place[1]),
            fixed_yaw=float(base_pick[2]),
        )

        # Build a single timeline.
        step_labels: List[str] = []
        base_traj: List[List[float]] = []
        arm_traj: List[List[float]] = []
        grip_traj: List[List[float]] = []

        def append_arm_segment(label: str, q_traj: np.ndarray, base_pose: Tuple[float, float, float], grip: float) -> None:
            for q in q_traj:
                step_labels.append(label)
                base_traj.append([float(base_pose[0]), float(base_pose[1]), float(base_pose[2])])
                arm_traj.append([float(v) for v in q.tolist()])
                grip_traj.append([float(grip)])

        assert chosen_segments is not None
        home = chosen_segments["home"]
        append_arm_segment("approach_pick", chosen_segments["approach_pick"], base_pick, grip=0.04)
        append_arm_segment("grasp", chosen_segments["grasp"], base_pick, grip=0.0)
        append_arm_segment("lift", chosen_segments["lift"], base_pick, grip=0.0)

        # Navigate: keep arm at last lift joint config.
        q_hold = chosen_segments["lift"][-1]
        for pose in base_path:
            step_labels.append("navigate")
            base_traj.append([float(pose[0]), float(pose[1]), float(pose[2])])
            arm_traj.append([float(v) for v in q_hold.tolist()])
            grip_traj.append([0.0])

        # After navigate, snap base pose to base_place.
        append_arm_segment("approach_place", chosen_segments["approach_place"], base_place, grip=0.0)
        append_arm_segment("place", chosen_segments["place"], base_place, grip=0.04)
        append_arm_segment("retreat_place", chosen_segments["retreat_place"], base_place, grip=0.04)

        plans.append(
            DemoPlan(
                demo_idx=demo_idx,
                variant_layout_json=str(variant_json),
                pick_object={
                    "id": pick_obj.get("id", ""),
                    "type": pick_obj.get("type", ""),
                    "source_id": pick_obj.get("source_id", ""),
                },
                place_surface={
                    "id": place_surface.get("id", ""),
                    "type": place_surface.get("type", ""),
                    "source_id": place_surface.get("source_id", ""),
                },
                base_pick=[float(base_pick[0]), float(base_pick[1]), float(base_pick[2])],
                base_place=[float(base_place[0]), float(base_place[1]), float(base_place[2])],
                trajectory_base=base_traj,
                trajectory_arm=arm_traj,
                trajectory_gripper=grip_traj,
                step_labels=step_labels,
                chosen_grasp_T_world=chosen_grasp_world.tolist(),
            )
        )

    # Save grasps (only the selected ones per demo to keep artifact small).
    grasps_out_dir = layout_dir / "grasps"
    grasps_out_dir.mkdir(parents=True, exist_ok=True)
    selected_grasps = np.concatenate(all_grasps_for_report, axis=0) if all_grasps_for_report else np.zeros((0, 4, 4), dtype=np.float32)
    selected_contacts = np.concatenate(all_contacts_for_report, axis=0) if all_contacts_for_report else np.zeros((0, 3), dtype=np.float32)
    grasp_payload = {
        "num_grasps": int(selected_grasps.shape[0]),
        "grasps": selected_grasps.tolist(),
        "contacts": selected_contacts.tolist(),
        "source": "m2t2",
        "model_path": str(weights_path),
        "num_views_m2t2": int(args.num_views_m2t2),
        "per_demo": [
            {
                "demo_idx": p.demo_idx,
                "variant_layout_json": p.variant_layout_json,
                "pick_object": p.pick_object,
                "place_surface": p.place_surface,
            }
            for p in plans
        ],
    }
    _write_json(grasps_out_dir / "grasp_transforms.json", grasp_payload)

    plan_bundle = {
        "layout_id": args.layout_id,
        "results_dir": str(Path(args.results_dir)),
        "pose_aug_name": args.pose_aug_name,
        "headless": bool(args.headless),
        "enable_cameras": bool(args.enable_cameras),
        "strict": bool(args.strict),
        "task_desc": args.task_desc,
        "demos": [p.__dict__ for p in plans],
    }
    plan_dir = layout_dir / "plans"
    plan_path = plan_dir / "plan_bundle.json"
    _write_json(plan_path, plan_bundle)
    _log(f"Wrote plan bundle: {plan_path}")

    collector = Path(__file__).with_name("isaacsim_collect_mobile_franka.py")
    if not collector.exists():
        raise FileNotFoundError(f"Missing collector script: {collector}")

    cmd = [
        args.isaacsim_py,
        str(collector),
        "--plan_bundle",
        str(plan_path),
        "--output_dir",
        str(args.output_dir),
    ]
    if args.headless:
        cmd.append("--headless")
    if args.enable_cameras:
        cmd.append("--enable_cameras")
    if args.strict:
        cmd.append("--strict")

    _log("Launching Stage 7 Isaac Sim collector...")
    _log(" ".join(cmd))
    subprocess.check_call(cmd)

    # Final checks (strict).
    demos_h5 = Path(args.output_dir) / "dataset.hdf5"
    if args.strict and not demos_h5.exists():
        raise RuntimeError(f"Stage 7 did not produce {demos_h5}")

    _log("Stage 5–7 complete.")


if __name__ == "__main__":
    main()
