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
import hashlib
import inspect
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
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


def _sanitize_pythonpath_for_isaacsim(env: Dict[str, str]) -> None:
    raw = env.get("PYTHONPATH", "")
    if not raw:
        return
    blocked = os.path.realpath(SAGE_SERVER_DIR)
    cleaned: List[str] = []
    for part in raw.split(os.pathsep):
        if not part:
            continue
        real = os.path.realpath(part)
        if real == blocked:
            continue
        cleaned.append(part)
    if cleaned:
        env["PYTHONPATH"] = os.pathsep.join(cleaned)
    else:
        env.pop("PYTHONPATH", None)


def _configure_vulkan_icd_env(env: Dict[str, str]) -> None:
    if not env.get("VK_ICD_FILENAMES") and not env.get("VK_DRIVER_FILES"):
        for candidate in ("/usr/share/vulkan/icd.d/nvidia_icd.json", "/etc/vulkan/icd.d/nvidia_icd.json"):
            if Path(candidate).exists():
                env["VK_ICD_FILENAMES"] = candidate
                env["VK_DRIVER_FILES"] = candidate
                _log(f"Using NVIDIA Vulkan ICD: {candidate}")
                break
    if env.get("VK_ICD_FILENAMES") and not env.get("VK_DRIVER_FILES"):
        env["VK_DRIVER_FILES"] = env["VK_ICD_FILENAMES"]
    if env.get("VK_DRIVER_FILES") and not env.get("VK_ICD_FILENAMES"):
        env["VK_ICD_FILENAMES"] = env["VK_DRIVER_FILES"]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_commit_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _build_stage7_command_and_env(
    *,
    args: argparse.Namespace,
    collector: Path,
    plan_path: Path,
    output_dir: Path,
    run_id: str = "",
) -> Tuple[List[str], Dict[str, str]]:
    cmd = [
        args.isaacsim_py,
        "-P",
        str(collector),
        "--plan_bundle",
        str(plan_path),
        "--output_dir",
        str(output_dir),
    ]
    cmd.append("--headless" if args.headless else "--no-headless")
    cmd.append("--enable_cameras" if args.enable_cameras else "--disable_cameras")
    cmd.append("--strict" if args.strict else "--no-strict")

    env = os.environ.copy()
    env["OMNI_KIT_ACCEPT_EULA"] = "yes"
    env.setdefault("ISAAC_ASSETS_ROOT", "/workspace/isaacsim_assets/Assets/Isaac/5.1")
    env.setdefault("SAGE_ALLOW_REMOTE_ISAAC_ASSETS", "0")
    env.setdefault("SAGE_SENSOR_FAILURE_POLICY", "fail" if bool(args.strict) else "warn")
    env.setdefault("SAGE_RENDER_WARMUP_FRAMES", "100")
    env.setdefault("SAGE_SENSOR_MIN_RGB_STD", "0.01")
    env.setdefault("SAGE_SENSOR_MIN_DEPTH_STD", "0.0001")
    env.setdefault("SAGE_MIN_VALID_DEPTH_PX", "1024")
    env.setdefault("SAGE_SENSOR_CHECK_FRAME", "10")
    env.setdefault("SAGE_ENFORCE_BUNDLE_STRICT", "1")
    if run_id:
        env.setdefault("SAGE_RUN_ID", run_id)
    env["PYTHONSAFEPATH"] = "1"
    _sanitize_pythonpath_for_isaacsim(env)
    _configure_vulkan_icd_env(env)
    return cmd, env


def _run_isaacsim_import_preflight(*, isaacsim_py: str, env: Dict[str, str], cwd: Path) -> None:
    preflight_code = """
import importlib
import inspect
import os
import pathlib
import sys

blocked = pathlib.Path(os.environ.get("SAGE_SERVER_DIR", "/workspace/SAGE/server")).resolve()

def _is_shadowed(path_str: str) -> bool:
    if not path_str:
        return False
    try:
        path = pathlib.Path(path_str).resolve()
    except Exception:
        return False
    return path == blocked or blocked in path.parents

issues = []
isaacsim = importlib.import_module("isaacsim")
isaacsim_file = str(getattr(isaacsim, "__file__", "") or "")
if isaacsim_file and _is_shadowed(isaacsim_file):
    issues.append(f"isaacsim module is shadowed by SAGE server path: {isaacsim_file}")

from isaacsim.simulation_app import SimulationApp
simapp_file = str(inspect.getfile(SimulationApp))
if _is_shadowed(simapp_file):
    issues.append(f"SimulationApp import is shadowed by SAGE server path: {simapp_file}")

print(f"isaacsim.__file__={isaacsim_file or '<namespace package>'}")
print(f"SimulationApp.file={simapp_file}")

if issues:
    for issue in issues:
        print(issue, file=sys.stderr)
    sys.exit(86)
"""
    cmd = [isaacsim_py, "-P", "-c", preflight_code]
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)
    if proc.returncode == 0:
        if proc.stdout.strip():
            _log(f"Stage 7 import preflight:\n{proc.stdout.strip()}")
        return

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    details = "\n".join(x for x in (out, err) if x)
    raise RuntimeError(
        "Stage 7 preflight failed before launch: Isaac Sim import/shadowing check failed. "
        f"python={isaacsim_py}. "
        "Remediation: run scripts/runpod_sage/apply_sage_patches.sh to refresh SAGE/server symlinks, "
        "launch via scripts/runpod_sage/run_full_pipeline.sh, and ensure ISAACSIM_PY_STAGE7 points to "
        "a valid Isaac Sim Python.\n"
        f"{details}"
    )


def _run_stage7_collector_with_tee(*, cmd: Sequence[str], env: Dict[str, str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"# cmd: {' '.join(cmd)}\n")
        log_f.write(f"# cwd: {cwd}\n")
        log_f.flush()
        proc = subprocess.Popen(
            list(cmd),
            env=env,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, list(cmd))


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

    # Fallback for cases where pose augmentation produced variant files but left
    # meta.json empty after planner filtering.
    if not out:
        all_variants = sorted(meta_path.parent.glob("variant_*.json"))
        if all_variants:
            preferred = meta_path.parent / "variant_000.json"
            out = [preferred] if preferred.exists() else [all_variants[0]]
            _log(
                f"WARNING: pose meta resolved to 0 layouts; using fallback variant "
                f"{out[0].name} from {meta_path.parent}"
            )
    return out


def _normalize_type(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")


def _extract_place_surface_token(task: str, surface_tokens: Sequence[str]) -> Optional[str]:
    # Normalize to space-separated tokens so word boundaries work in regex.
    task_n = re.sub(r"[^a-z0-9]+", " ", str(task or "").lower()).strip()
    if not task_n:
        return None
    # Prefer explicit place intent ("place/put/set/move ... on/to <surface>").
    for tok in surface_tokens:
        pat = rf"\b(?:place|put|set|move|drop)\b.*?\b(?:on|onto|to|into)\b.*?\b{re.escape(tok)}\b"
        if re.search(pat, task_n):
            return tok
    # Fallback: any place-intent mention of a surface token.
    for tok in surface_tokens:
        pat = rf"\b(?:place|put|set|move|drop)\b[^.]*\b{re.escape(tok)}\b"
        if re.search(pat, task_n):
            return tok
    return None


def _type_tokens(raw_type: str) -> List[str]:
    toks = [t for t in re.split(r"[^a-z0-9]+", _normalize_type(raw_type)) if t]
    stop = {
        "object",
        "item",
        "small",
        "large",
        "kitchen",
        "dining",
        "table",
        "counter",
        "desk",
        "island",
        "bench",
        "surface",
    }
    return [t for t in toks if len(t) >= 3 and t not in stop]


def _is_object_on_surface(obj: Dict[str, Any], surface: Dict[str, Any]) -> bool:
    """Heuristic support check in XY + top-height proximity."""
    op = obj.get("position", {}) or {}
    od = obj.get("dimensions", {}) or {}
    sp = surface.get("position", {}) or {}
    sd = surface.get("dimensions", {}) or {}

    ox = float(op.get("x", 0.0))
    oy = float(op.get("y", 0.0))
    oz = float(op.get("z", 0.0))  # object bottom in SAGE convention
    ow = max(0.0, float(od.get("width", 0.0)))
    ol = max(0.0, float(od.get("length", 0.0)))
    oh = max(0.0, float(od.get("height", 0.0)))

    sx = float(sp.get("x", 0.0))
    sy = float(sp.get("y", 0.0))
    sz = float(sp.get("z", 0.0))
    sw = max(0.0, float(sd.get("width", 0.0)))
    sl = max(0.0, float(sd.get("length", 0.0)))
    sh = max(0.0, float(sd.get("height", 0.0)))

    xy_tol = max(0.10, min(0.30, 0.15 + 0.25 * max(ow, ol)))
    on_x = abs(ox - sx) <= (0.5 * sw + xy_tol)
    on_y = abs(oy - sy) <= (0.5 * sl + xy_tol)
    if not (on_x and on_y):
        return False

    surface_top = sz + sh
    z_tol = max(0.08, min(0.25, 0.08 + 0.5 * oh))
    return abs(oz - surface_top) <= z_tol


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
        if min(w, l) <= 0.35 and obj.get("source_id"):
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
        ttoks = _type_tokens(t)
        if any(tok in task for tok in ttoks):
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
    explicit_place_tok = _extract_place_surface_token(task, surface_tokens)
    if explicit_place_tok is not None:
        for obj in surfaces:
            t = _normalize_type(obj.get("type", ""))
            if explicit_place_tok in t:
                place = obj
                break
    # Fallback: prefer any surface whose type keyword appears in the task.
    if place is None:
        for obj in surfaces:
            t = _normalize_type(obj.get("type", ""))
            for tok in surface_tokens:
                if tok in t and tok in task:
                    place = obj
                    break
            if place is not None:
                break
    if place is None:
        # Fallback: first surface
        place = surfaces[0]

    return pick, place


def _build_occupancy_grid(
    room: Dict[str, Any],
    *,
    resolution_m: float = 0.05,
    margin_m: float = 0.15,
    min_footprint_for_obstacle: float = 0.25,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build 2D occupancy grid.  Only objects with min(w,l) > min_footprint_for_obstacle
    are treated as obstacles.  Small / manipulable objects (cups, salt, etc.) are excluded."""
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
        if w > 0.0 and l > 0.0 and min(w, l) > min_footprint_for_obstacle:
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
    approach_dist: float = 0.40,
) -> Tuple[float, float, float]:
    tx, ty = target_xy
    width = float(room_dims.get("width", 6.0))
    length = float(room_dims.get("length", 6.0))
    margin = 0.45

    best = None
    best_clear = -1.0

    for angle in np.linspace(0.0, 2.0 * math.pi, 36, endpoint=False):
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
    from curobo.types.robot import JointState

    # Franka home joints (7)
    home = np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

    def plan_to_pose(start_q: np.ndarray, target_T: np.ndarray) -> np.ndarray:
        pos = torch.tensor(target_T[:3, 3].astype(np.float32)).unsqueeze(0).cuda()
        quat_list = _quat_from_rot(target_T[:3, :3])
        quat = torch.tensor(quat_list, dtype=torch.float32).unsqueeze(0).cuda()
        target = Pose(position=pos, quaternion=quat)
        q_pos = torch.tensor(start_q, dtype=torch.float32).unsqueeze(0).cuda()
        q_vel = torch.zeros_like(q_pos)
        q_acc = torch.zeros_like(q_pos)
        start_state = JointState(position=q_pos, velocity=q_vel, acceleration=q_acc)
        result = motion_gen.plan_single(start_state, target)
        success = getattr(result, "success", None)
        if success is not None and hasattr(success, "item"):
            success = success.item()
        if not success:
            raise RuntimeError(f"cuRobo plan_single failed (success={success})")
        traj = result.get_interpolated_plan()
        if traj is None:
            raise RuntimeError("cuRobo get_interpolated_plan returned None")
        traj_pos = getattr(traj, "position", None)
        if traj_pos is None:
            # Some cuRobo versions return the JointState directly
            traj_pos = getattr(traj, "joint_state", traj)
            if hasattr(traj_pos, "position"):
                traj_pos = traj_pos.position
        if hasattr(traj_pos, "cpu"):
            arr = traj_pos.cpu().numpy()
        else:
            arr = np.asarray(traj_pos)
        if arr.ndim == 3:
            arr = arr.squeeze(0)  # (1, T, 7) -> (T, 7)
        return arr

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
        import torch
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
        from curobo.types.base import TensorDeviceType
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"cuRobo not available: {exc}") from exc

    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    config = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        interpolation_dt=0.02,
        tensor_args=tensor_args,
        num_ik_seeds=32,
    )
    mg = MotionGen(config)
    mg.warmup()
    return mg


def _build_mesh_dict_list(variant_layout_json: Path, room_id: str):
    """Convert a layout JSON into the mesh_dict_list that M2T2 expects."""
    from tex_utils import dict_to_floor_plan, export_single_room_layout_to_mesh_dict_list

    layout_data = _load_json(variant_layout_json)
    layout = dict_to_floor_plan(layout_data)
    # Use the first room if room_id not found (single-room layouts).
    rid = room_id
    if not any(r.id == rid for r in layout.rooms):
        rid = layout.rooms[0].id
    mesh_info_dict = export_single_room_layout_to_mesh_dict_list(layout, rid)
    return mesh_info_dict


def _infer_grasps_for_variant(
    *,
    layout_id: str,
    layout_dir: Path,
    variant_layout_json: Path,
    pick_obj_id: str,
    base_pos: Tuple[float, float, float],
    room_id: str,
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

    mesh_dict_list = _build_mesh_dict_list(variant_layout_json, room_id)

    # generate_m2t2_data places camera at base_pos + [0,0,0.8].
    # For objects on elevated surfaces (counters, tables), the camera may be
    # below the object.  Raise base_pos.z so the camera is above the target.
    target_mesh = mesh_dict_list.get(pick_obj_id, {}).get("mesh")
    adjusted_base = list(base_pos)
    if target_mesh is not None:
        obj_top_z = float(target_mesh.bounds[1, 2])  # max Z of target
        cam_z_needed = obj_top_z + 0.3  # camera should be above the object
        cam_offset = 0.8  # offset added inside generate_m2t2_data
        if adjusted_base[2] + cam_offset < cam_z_needed:
            adjusted_base[2] = cam_z_needed - cam_offset
            _log(f"Raised M2T2 camera base_z to {adjusted_base[2]:.3f} (obj_top={obj_top_z:.3f})")

    meta_data, vis_data = generate_m2t2_data(mesh_dict_list, pick_obj_id, adjusted_base)
    pairs = [(meta_data, vis_data)]

    all_grasps: List[np.ndarray] = []
    all_contacts: List[np.ndarray] = []
    for meta_data, vis_data in pairs:
        try:
            grasps_out = infer_m2t2(meta_data, vis_data, model, cfg, return_contacts=True)
        except (ZeroDivisionError, RuntimeError) as exc:
            _log(f"WARNING: M2T2 inference failed for {pick_obj_id}: {exc}")
            continue
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


RIDGEBACK_ARM_BASE_HEIGHT = 0.40  # Ridgeback mobile base top + mount ≈ 0.40m

def _world_from_base(base_pose: Sequence[float]) -> np.ndarray:
    """Transform from Franka arm base frame to world frame.

    The arm base is on a Ridgeback mobile base at height RIDGEBACK_ARM_BASE_HEIGHT.
    base_pose = (x, y, yaw) of the mobile base on the floor.
    """
    x, y, yaw = float(base_pose[0]), float(base_pose[1]), float(base_pose[2])
    c = math.cos(yaw)
    s = math.sin(yaw)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    T[:3, 3] = np.asarray([x, y, RIDGEBACK_ARM_BASE_HEIGHT], dtype=np.float32)
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
    parser.add_argument("--strict", dest="strict", action="store_true", default=False)
    parser.add_argument("--no-strict", dest="strict", action="store_false")

    parser.add_argument("--m2t2_weights", default=M2T2_WEIGHTS_DEFAULT)
    parser.add_argument("--isaacsim_py", default=ISAACSIM_PY_DEFAULT)
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0") or "0"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_id = str(os.environ.get("SAGE_RUN_ID", "")).strip()
    if not run_id:
        run_id = f"{args.layout_id}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"

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
    _log(f"run_id={run_id}")
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
      try:
        # Deterministic variant selection (rotate) for reproducibility.
        variant_json = variants[demo_idx % len(variants)]
        _variant_data = _load_json(variant_json)
        # Handle both room-level and layout-level JSONs.
        if "rooms" in _variant_data and isinstance(_variant_data["rooms"], list) and _variant_data["rooms"]:
            room = _variant_data["rooms"][0]
        else:
            room = _variant_data
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

        # If placing back on the same support (e.g. "pick salt and place on table"),
        # keep a single base pose and skip mobile-base navigation entirely.
        same_support = _is_object_on_surface(pick_obj, place_surface)
        base_dist = math.sqrt((base_pick[0] - base_place[0]) ** 2 + (base_pick[1] - base_place[1]) ** 2)
        skip_nav = same_support or (base_dist < 0.5)
        if skip_nav:
            reason = "same_support" if same_support else f"bases_within_{base_dist:.2f}m"
            _log(f"demo={demo_idx}: skipping base navigation ({reason})")
            base_place = base_pick

        _log(f"demo={demo_idx}: variant={variant_json.name} pick={pick_obj.get('type')} place={place_surface.get('type')}")

        grasps_world, contacts = _infer_grasps_for_variant(
            layout_id=args.layout_id,
            layout_dir=layout_dir,
            variant_layout_json=variant_json,
            pick_obj_id=pick_obj["id"],
            base_pos=(base_pick[0], base_pick[1], 0.0),
            room_id=room.get("id", "room_0"),
            num_views=args.num_views_m2t2,
            strict=args.strict,
            model=model,
            cfg=cfg,
        )
        if grasps_world.shape[0] == 0:
            _log(f"demo={demo_idx}: Stage 5 produced 0 grasps — skipping")
            if args.strict:
                raise RuntimeError(f"Stage 5 produced 0 grasps for demo={demo_idx} (variant={variant_json})")
            continue

        # Choose grasp using top-K candidates and require first successful cuRobo plan.
        top_k = max(1, int(args.grasp_top_k))
        grasp_candidates = min(grasps_world.shape[0], top_k)
        if args.strict and grasp_candidates <= 0:
            raise RuntimeError(f"Stage 5 produced 0 grasps after filtering for demo={demo_idx} (variant={variant_json})")

        # Convert base pose for frame conversion once.
        base_T_world = _invert_T(_world_from_base(base_pick))
        place_pose_base = _default_place_pose_base(place_surface, base_place)

        # Debug: show grasp in base frame
        if grasps_world.shape[0] > 0:
            g0 = base_T_world @ grasps_world[0]
            _log(f"demo={demo_idx}: base_pick=({base_pick[0]:.2f},{base_pick[1]:.2f},{base_pick[2]:.2f}) "
                 f"grasp[0]_base=({g0[0,3]:.3f},{g0[1,3]:.3f},{g0[2,3]:.3f}) "
                 f"dist={np.linalg.norm(g0[:3,3]):.3f}m")

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
                import traceback
                plan_error = str(e)
                tb = traceback.format_exc()
                _log(f"demo={demo_idx}: grasp candidate {cand_idx + 1}/{grasp_candidates} failed: {e}\n{tb}")
                continue

        if chosen_segments is None or chosen_grasp_world is None:
            msg = (
                f"Stage 6 failed to plan arm trajectory for demo={demo_idx}. "
                f"No feasible grasps in top-K ({grasp_candidates}). last_error={plan_error}"
            )
            if args.strict:
                raise RuntimeError(msg)
            _log(f"WARNING: {msg} — skipping demo")
            continue

        all_grasps_for_report.append(chosen_grasp_world[None, ...])
        all_contacts_for_report.append((contacts[chosen_idx : chosen_idx + 1] if contacts.shape[0] else np.zeros((1, 3), dtype=np.float32)))

        # Base path (A*), yaw fixed to face forward.
        if skip_nav:
            _log(f"demo={demo_idx}: skipping base navigation (same base pose)")
            base_path = np.array([[base_pick[0], base_pick[1], base_pick[2]]], dtype=np.float32)
        else:
            try:
                base_path = _plan_base_traj(
                    grid=grid,
                    grid_meta=grid_meta,
                    start_xy=(base_pick[0], base_pick[1]),
                    goal_xy=(base_place[0], base_place[1]),
                    fixed_yaw=float(base_pick[2]),
                )
            except RuntimeError as exc:
                # Retry once with a relaxed occupancy model to reduce false negatives
                # in narrow kitchens where chairs/props make the conservative grid disconnected.
                _log(f"demo={demo_idx}: base A* failed on primary grid ({exc}); retrying with relaxed grid")
                grid_relaxed, meta_relaxed = _build_occupancy_grid(
                    room,
                    resolution_m=0.05,
                    margin_m=0.08,
                    min_footprint_for_obstacle=0.35,
                )
                base_path = _plan_base_traj(
                    grid=grid_relaxed,
                    grid_meta=meta_relaxed,
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
      except Exception as exc:
        if args.strict:
            raise
        _log(f"ERROR: demo={demo_idx} failed: {exc} — skipping")
        continue

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
        "run_id": run_id,
        "layout_id": args.layout_id,
        "results_dir": str(Path(args.results_dir)),
        "pose_aug_name": args.pose_aug_name,
        "headless": bool(args.headless),
        "enable_cameras": bool(args.enable_cameras),
        "strict": bool(args.strict),
        "task_desc": args.task_desc,
        "demos": [p.__dict__ for p in plans],
        "provenance": {
            "run_id": run_id,
            "stage567_script": str(Path(__file__).resolve()),
            "stage567_script_sha256": _file_sha256(Path(__file__).resolve()),
            "bp_git_commit": _git_commit_sha(Path(__file__).resolve().parents[2]),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage7_log_path": str(layout_dir / "stage7_output" / "stage7.log"),
            "collector_script": str(Path(__file__).with_name("isaacsim_collect_mobile_franka.py").resolve()),
            "policy_env": {
                "ISAAC_ASSETS_ROOT": os.environ.get("ISAAC_ASSETS_ROOT", ""),
                "REQUIRE_LOCAL_ROBOT_ASSET": os.environ.get("REQUIRE_LOCAL_ROBOT_ASSET", ""),
                "SAGE_ALLOW_REMOTE_ISAAC_ASSETS": os.environ.get("SAGE_ALLOW_REMOTE_ISAAC_ASSETS", "0"),
                "SAGE_SENSOR_FAILURE_POLICY": os.environ.get("SAGE_SENSOR_FAILURE_POLICY", "auto"),
                "SAGE_RENDER_WARMUP_FRAMES": os.environ.get("SAGE_RENDER_WARMUP_FRAMES", ""),
                "SAGE_SENSOR_MIN_RGB_STD": os.environ.get("SAGE_SENSOR_MIN_RGB_STD", ""),
                "SAGE_SENSOR_MIN_DEPTH_STD": os.environ.get("SAGE_SENSOR_MIN_DEPTH_STD", ""),
                "SAGE_MIN_VALID_DEPTH_PX": os.environ.get("SAGE_MIN_VALID_DEPTH_PX", ""),
                "SAGE_SENSOR_CHECK_FRAME": os.environ.get("SAGE_SENSOR_CHECK_FRAME", ""),
                "SAGE_ENFORCE_BUNDLE_STRICT": os.environ.get("SAGE_ENFORCE_BUNDLE_STRICT", ""),
            },
        },
    }
    plan_dir = layout_dir / "plans"
    plan_path = plan_dir / "plan_bundle.json"
    _write_json(plan_path, plan_bundle)
    _log(f"Wrote plan bundle: {plan_path}")

    if not plans:
        _log("WARNING: No successful demo plans. Skipping Stage 7 (Isaac Sim collection).")
        _log("Stage 5–6 complete with 0 demos.")
        return

    collector = Path(__file__).with_name("isaacsim_collect_mobile_franka.py")
    if not collector.exists():
        raise FileNotFoundError(f"Missing collector script: {collector}")
    stage7_log_path = layout_dir / "stage7_output" / "stage7.log"
    cmd, env = _build_stage7_command_and_env(
        args=args,
        collector=collector,
        plan_path=plan_path,
        output_dir=Path(args.output_dir),
        run_id=run_id,
    )
    _run_isaacsim_import_preflight(
        isaacsim_py=args.isaacsim_py,
        env=env,
        cwd=Path(__file__).resolve().parents[2],
    )
    _log("Launching Stage 7 Isaac Sim collector...")
    _log(" ".join(cmd))
    _log(f"Stage 7 log path: {stage7_log_path}")
    _run_stage7_collector_with_tee(
        cmd=cmd,
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        log_path=stage7_log_path,
    )

    # Final checks (strict).
    demos_h5 = Path(args.output_dir) / "dataset.hdf5"
    if args.strict and not demos_h5.exists():
        raise RuntimeError(f"Stage 7 did not produce {demos_h5}")

    _log("Stage 5–7 complete.")


if __name__ == "__main__":
    main()
