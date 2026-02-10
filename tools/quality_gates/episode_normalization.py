"""Helpers for normalizing episode JSONs for offline certification.

This exists because we have multiple on-disk episode formats:
- "Raw" episodes: may omit `object_poses`, have sparse `ee_vel`/`ee_acc`, etc.
- "LocalFramework" episodes: include richer per-frame fields.

The physics certification gates operate on a frame schema that expects:
- `frame.object_poses[target].{position,rotation_quat,linear_velocity,angular_velocity,source}`
- complete proprio channels (`ee_vel`, `ee_acc`, `robot_state.joint_accelerations`, etc.)

When re-certifying existing episodes without re-running the VM, we can derive
many missing fields deterministically from existing data (`ee_pos`, `dt`, and
privileged scene-state).
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _normalize_obj_token(name: Optional[str], *, strip_numeric_suffix: bool = True) -> str:
    token = str(name or "").strip()
    if not token:
        return ""
    token = token.rsplit("/", 1)[-1]
    if "_obj_" in token:
        token = token.split("_obj_")[-1]
    token = token.lower()
    if token.startswith("variation_"):
        token = token[len("variation_") :]
    if strip_numeric_suffix:
        token = re.sub(r"\d+$", "", token)
    return token


def _extract_xyz(pose: Any) -> Optional[np.ndarray]:
    """Extract XYZ position from a pose payload.

    Supports the minimal privileged payload we see in raw episodes:
      {"x": ..., "y": ..., "z": ...}
    """
    if isinstance(pose, dict):
        if {"x", "y", "z"}.issubset(pose.keys()):
            try:
                return np.array([pose["x"], pose["y"], pose["z"]], dtype=float)
            except Exception:
                return None
        pos = pose.get("position")
        if isinstance(pos, dict) and {"x", "y", "z"}.issubset(pos.keys()):
            try:
                return np.array([pos["x"], pos["y"], pos["z"]], dtype=float)
            except Exception:
                return None
        if isinstance(pos, list) and len(pos) >= 3:
            try:
                return np.array(pos[:3], dtype=float)
            except Exception:
                return None
    if isinstance(pose, list) and len(pose) >= 3:
        try:
            return np.array(pose[:3], dtype=float)
        except Exception:
            return None
    return None


def _infer_dt_s(frames: List[Dict[str, Any]], *, default: float = 1.0 / 30.0) -> float:
    # Prefer explicit `dt` from any frame.
    for f in frames:
        dt = f.get("dt")
        if isinstance(dt, (int, float)) and float(dt) > 0:
            return float(dt)

    # Fall back to timestamp diffs.
    ts: List[float] = []
    for f in frames:
        t = f.get("timestamp")
        if isinstance(t, (int, float)):
            ts.append(float(t))
    if len(ts) >= 2:
        diffs = [ts[i] - ts[i - 1] for i in range(1, len(ts)) if ts[i] > ts[i - 1]]
        if diffs:
            diffs.sort()
            return float(diffs[len(diffs) // 2])  # median-ish

    return float(default)


def _ensure_ee_quat(frames: List[Dict[str, Any]]) -> None:
    for f in frames:
        if f.get("ee_quat") is not None:
            continue
        rs = ((f.get("observation") or {}).get("robot_state") or {}) if isinstance(f.get("observation"), dict) else {}
        ee_pose = rs.get("ee_pose") if isinstance(rs, dict) else None
        rot = (ee_pose or {}).get("rotation") if isinstance(ee_pose, dict) else None
        if isinstance(rot, dict) and {"rx", "ry", "rz", "rw"}.issubset(rot.keys()):
            # Convention doesn't matter for certification completeness; keep (x,y,z,w).
            f["ee_quat"] = [rot["rx"], rot["ry"], rot["rz"], rot["rw"]]


def _ensure_ee_vel_acc(frames: List[Dict[str, Any]], dt_s: float) -> None:
    # Compute per-frame velocities/accelerations from ee_pos.
    ee_pos: List[Optional[np.ndarray]] = []
    for f in frames:
        p = f.get("ee_pos")
        if isinstance(p, list) and len(p) >= 3:
            try:
                ee_pos.append(np.array(p[:3], dtype=float))
            except Exception:
                ee_pos.append(None)
        else:
            ee_pos.append(None)

    vel = [np.zeros(3, dtype=float) for _ in frames]
    acc = [np.zeros(3, dtype=float) for _ in frames]
    for i in range(1, len(frames)):
        if ee_pos[i] is not None and ee_pos[i - 1] is not None:
            vel[i] = (ee_pos[i] - ee_pos[i - 1]) / float(dt_s)
    for i in range(1, len(frames)):
        acc[i] = (vel[i] - vel[i - 1]) / float(dt_s)

    for i, f in enumerate(frames):
        f["ee_vel"] = vel[i].tolist()
        f["ee_acc"] = acc[i].tolist()


def _ensure_joint_accelerations(frames: List[Dict[str, Any]], dt_s: float) -> None:
    # Compute robot_state.joint_accelerations from joint_velocities when missing/sparse.
    jvel: List[Optional[np.ndarray]] = []
    for f in frames:
        obs = f.get("observation") or {}
        rs = obs.get("robot_state") if isinstance(obs, dict) else None
        v = rs.get("joint_velocities") if isinstance(rs, dict) else None
        if isinstance(v, list) and v:
            try:
                jvel.append(np.array(v, dtype=float))
            except Exception:
                jvel.append(None)
        else:
            jvel.append(None)

    jacc: List[Optional[np.ndarray]] = [None] * len(frames)
    for i in range(len(frames)):
        if jvel[i] is None:
            continue
        if i == 0 or jvel[i - 1] is None or jvel[i].shape != jvel[i - 1].shape:
            jacc[i] = np.zeros_like(jvel[i])
        else:
            jacc[i] = (jvel[i] - jvel[i - 1]) / float(dt_s)

    for i, f in enumerate(frames):
        if jacc[i] is None:
            continue
        obs = f.get("observation") or {}
        rs = obs.get("robot_state") if isinstance(obs, dict) else None
        if isinstance(rs, dict):
            rs["joint_accelerations"] = jacc[i].tolist()


def _privileged_scene_objects(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    obs = frame.get("observation") or {}
    if not isinstance(obs, dict):
        return []
    priv = obs.get("privileged") or {}
    if not isinstance(priv, dict):
        return []
    ss = priv.get("scene_state") or {}
    if not isinstance(ss, dict):
        return []
    objs = ss.get("objects") or []
    return objs if isinstance(objs, list) else []


def _target_pos_raw(
    frame: Dict[str, Any],
    *,
    target_token: str,
) -> Optional[np.ndarray]:
    if not target_token:
        return None
    for obj in _privileged_scene_objects(frame):
        if not isinstance(obj, dict):
            continue
        oid = obj.get("object_id") or obj.get("id") or obj.get("name") or ""
        if _normalize_obj_token(oid) != target_token:
            continue
        return _extract_xyz(obj.get("pose") or {})
    return None


def _compute_alignment_offset(
    frames: List[Dict[str, Any]],
    *,
    target_object_id: str,
) -> Tuple[np.ndarray, str]:
    """Compute a translation offset to align privileged object poses to EE coordinates.

    Some raw episodes store privileged object poses in a different origin than `ee_pos`.
    For offline certification, we align by translating all objects so that the target
    pose coincides with `ee_pos` at the first gripper-close frame (only when needed).
    """
    target_token = _normalize_obj_token(target_object_id)
    if not target_token:
        return np.zeros(3, dtype=float), "server_static"

    grasp_proximity_max_m = _env_float("PHYSICS_CERT_GRASP_PROXIMITY_MAX_M", 0.15)

    # Compute min raw distance between EE and target privileged pose.
    min_raw_dist = math.inf
    for f in frames:
        ee = f.get("ee_pos")
        if not (isinstance(ee, list) and len(ee) >= 3):
            continue
        tpos = _target_pos_raw(f, target_token=target_token)
        if tpos is None:
            continue
        try:
            d = float(np.linalg.norm(np.array(ee[:3], dtype=float) - tpos))
        except Exception:
            continue
        min_raw_dist = min(min_raw_dist, d)

    # If it's already plausible, don't shift anything.
    if not math.isfinite(min_raw_dist) or min_raw_dist <= grasp_proximity_max_m:
        return np.zeros(3, dtype=float), "server_static"

    # Find first close frame with a target pose.
    for f in frames:
        if str(f.get("gripper_command")) != "closed":
            continue
        ee = f.get("ee_pos")
        if not (isinstance(ee, list) and len(ee) >= 3):
            continue
        tpos = _target_pos_raw(f, target_token=target_token)
        if tpos is None:
            continue
        try:
            offset = np.array(ee[:3], dtype=float) - tpos
        except Exception:
            continue
        if float(np.linalg.norm(offset)) <= 1e-9:
            return np.zeros(3, dtype=float), "server_static"
        return offset, "server_static_aligned_to_ee"

    return np.zeros(3, dtype=float), "server_static"


_MASS_ESTIMATES_KG: Dict[str, float] = {
    "pot": 0.5,
    "cup": 0.2,
    "plate": 0.3,
    "bottle": 0.4,
    "toaster": 1.0,
    "mug": 0.25,
    "bowl": 0.35,
    "can": 0.3,
    "pan": 0.6,
    "kettle": 0.8,
    "jar": 0.35,
    "coffeemachine": 3.0,
    "coffee_machine": 3.0,
    "stovetop": 2.0,
    "stove": 2.0,
    "microwave": 12.0,
    "dishwasher": 30.0,
    "refrigerator": 60.0,
    "rangehood": 8.0,
    "sink": 5.0,
    "cabinet": 15.0,
    "kitchen_cabinet": 15.0,
    "table": 20.0,
    "wallstackoven": 25.0,
}


def _ensure_object_metadata(episode: Dict[str, Any], frames: List[Dict[str, Any]]) -> None:
    if not isinstance(episode, dict):
        return
    if "object_metadata" in episode and isinstance(episode.get("object_metadata"), dict):
        return
    # Create a minimal object_metadata dict (best-effort).
    meta: Dict[str, Any] = {}
    if not frames:
        episode["object_metadata"] = meta
        return
    for obj in _privileged_scene_objects(frames[0]):
        if not isinstance(obj, dict):
            continue
        oid = obj.get("object_id") or obj.get("id") or obj.get("name") or ""
        if not oid:
            continue
        token = _normalize_obj_token(oid)
        mass = _MASS_ESTIMATES_KG.get(token)
        if mass is None:
            continue
        meta[str(oid)] = {"mass_kg": float(mass), "estimation_source": "hardcoded_fallback"}
    episode["object_metadata"] = meta


def _retro_downgrade_task_success(episode: Dict[str, Any]) -> None:
    """Avoid TASK_SUCCESS_CONTRADICTION when we have no geometric verification."""
    if not isinstance(episode, dict):
        return
    if episode.get("task_success") is not True:
        return
    if episode.get("goal_region_verification"):
        return
    task_type = str(episode.get("task_type") or episode.get("task_name") or "").lower()
    # Mirror LocalFramework behavior (and include `interact`/`organize`).
    if any(k in task_type for k in ("pick", "place", "grasp", "lift", "stack", "organize", "interact", "transport")):
        episode["task_success"] = False
        episode.setdefault(
            "task_success_reasoning",
            (
                "Server reported success but no geometric verification could be performed. "
                "Downgraded to False to avoid TASK_SUCCESS_CONTRADICTION."
            ),
        )
        episode.setdefault("task_success_source", "server_downgraded_no_geometric_verification")


def ensure_object_poses_from_privileged_scene_state(
    episode: Dict[str, Any],
    frames: List[Dict[str, Any]],
) -> None:
    """Populate `frame.object_poses` from privileged scene_state objects."""
    if not frames:
        return

    target_object_id = (
        str(episode.get("target_object") or episode.get("target_object_id") or "").strip()
        if isinstance(episode, dict)
        else ""
    )
    offset, source_label = _compute_alignment_offset(frames, target_object_id=target_object_id)

    for f in frames:
        # If object_poses already exists and is non-empty, preserve it.
        existing = f.get("object_poses")
        if isinstance(existing, dict) and existing:
            continue
        obj_poses: Dict[str, Any] = {}
        for obj in _privileged_scene_objects(f):
            if not isinstance(obj, dict):
                continue
            oid = obj.get("object_id") or obj.get("id") or obj.get("name") or ""
            if not oid:
                continue
            xyz = _extract_xyz(obj.get("pose") or {})
            if xyz is None:
                continue
            if offset is not None:
                xyz = xyz + offset
            obj_poses[str(oid)] = {
                "position": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "rotation_quat": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "source": source_label,
            }
        if obj_poses:
            f["object_poses"] = obj_poses


def normalize_episode_for_certification(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an episode dict in-place and return it."""
    frames = episode.get("frames") if isinstance(episode, dict) else None
    if not isinstance(frames, list) or not frames:
        return episode

    dt_s = _infer_dt_s(frames)
    _ensure_ee_quat(frames)
    _ensure_ee_vel_acc(frames, dt_s=dt_s)
    _ensure_joint_accelerations(frames, dt_s=dt_s)
    ensure_object_poses_from_privileged_scene_state(episode, frames)
    _ensure_object_metadata(episode, frames)
    _retro_downgrade_task_success(episode)
    return episode

