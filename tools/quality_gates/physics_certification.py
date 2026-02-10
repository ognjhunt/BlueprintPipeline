#!/usr/bin/env python3
"""Physics certification gates for GenieSim episode outputs."""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from tools.camera_io import is_placeholder_depth, is_placeholder_rgb

GATE_VERSION = "1.2.0"
GATE_CODES = (
    "KINEMATIC_OBJECT_POSE_USED",
    "SNAPBACK_OR_TELEPORT_DETECTED",
    "TASK_SUCCESS_CONTRADICTION",
    "CONTACT_PLACEHOLDER_OR_EMPTY",
    "TARGET_SCHEMA_INCOMPLETE",
    "CHANNEL_INCOMPLETE",
    "EE_TARGET_GEOMETRY_IMPLAUSIBLE",
    "SCENE_STATE_NOT_SERVER_BACKED",
    "CAMERA_PLACEHOLDER_PRESENT",
    "STRICT_RUNTIME_PRECHECK_FAILED",
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var (accepts 1/true/yes/on, case-insensitive)."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


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


def _looks_like_gripper_body(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in ("gripper", "finger", "hand", "ee", "wrist"))


def _extract_target_positions(frames: List[Dict[str, Any]], target_id: str) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    target_norm = _normalize_obj_token(target_id)
    for frame in frames:
        obj_poses = frame.get("object_poses", {})
        if not isinstance(obj_poses, dict):
            continue
        found: Optional[List[float]] = None
        for oid, pose in obj_poses.items():
            if _normalize_obj_token(oid) != target_norm:
                continue
            if not isinstance(pose, dict):
                continue
            pos = pose.get("position")
            if isinstance(pos, list) and len(pos) >= 3:
                found = pos[:3]
                break
        if found is not None:
            out.append(np.array(found, dtype=float))
    return out


def _target_schema_completeness(
    frames: List[Dict[str, Any]],
    target_id: str,
) -> Tuple[float, int, int]:
    required = ("position", "rotation_quat", "linear_velocity", "angular_velocity", "source")
    target_norm = _normalize_obj_token(target_id)
    complete = 0
    total = len(frames)
    if total == 0:
        return 0.0, 0, 0
    for frame in frames:
        obj_poses = frame.get("object_poses", {})
        if not isinstance(obj_poses, dict):
            continue
        match = None
        for oid, pose in obj_poses.items():
            if _normalize_obj_token(oid) == target_norm and isinstance(pose, dict):
                match = pose
                break
        if match is None:
            continue
        if all(match.get(key) is not None for key in required):
            complete += 1
    return complete / float(total), complete, total


def _channel_completeness(frames: List[Dict[str, Any]]) -> Dict[str, float]:
    channels = (
        "ee_pos",
        "ee_quat",
        "ee_vel",
        "ee_acc",
        "joint_positions",
        "joint_velocities",
        "joint_accelerations",
        "joint_efforts",
    )
    total = max(1, len(frames))
    counts = {k: 0 for k in channels}
    for frame in frames:
        obs = frame.get("observation", {})
        rs = obs.get("robot_state", {}) if isinstance(obs, dict) else {}
        if frame.get("ee_pos") is not None:
            counts["ee_pos"] += 1
        if frame.get("ee_quat") is not None:
            counts["ee_quat"] += 1
        if frame.get("ee_vel") is not None or frame.get("ee_velocity") is not None:
            counts["ee_vel"] += 1
        if frame.get("ee_acc") is not None or frame.get("ee_acceleration") is not None:
            counts["ee_acc"] += 1
        if isinstance(rs.get("joint_positions"), list) and len(rs["joint_positions"]) > 0:
            counts["joint_positions"] += 1
        if isinstance(rs.get("joint_velocities"), list) and len(rs["joint_velocities"]) > 0:
            counts["joint_velocities"] += 1
        if isinstance(rs.get("joint_accelerations"), list) and len(rs["joint_accelerations"]) > 0:
            counts["joint_accelerations"] += 1
        if isinstance(rs.get("joint_efforts"), list) and len(rs["joint_efforts"]) > 0:
            counts["joint_efforts"] += 1
    coverage = {k: round(v / float(total), 4) for k, v in counts.items()}
    coverage["ee_velocity"] = coverage["ee_vel"]
    coverage["ee_acceleration"] = coverage["ee_acc"]
    return coverage


def _stale_effort_stats(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    efforts: List[np.ndarray] = []
    effort_sources: List[str] = []
    for frame in frames:
        obs = frame.get("observation", {})
        rs = obs.get("robot_state", {}) if isinstance(obs, dict) else {}
        vals = rs.get("joint_efforts")
        if isinstance(vals, list) and vals:
            try:
                efforts.append(np.array(vals, dtype=float))
            except Exception:
                continue
        effort_sources.append(str(frame.get("efforts_source") or ""))

    if len(efforts) < 3:
        return {
            "stale_efforts_detected": False,
            "stale_effort_pair_ratio": 0.0,
            "efforts_sources": sorted(set(effort_sources)),
        }

    stale_pairs = 0
    total_pairs = 0
    for i in range(1, len(efforts)):
        a = efforts[i - 1]
        b = efforts[i]
        if a.shape != b.shape:
            continue
        total_pairs += 1
        if float(np.linalg.norm(b - a)) <= 1e-8:
            stale_pairs += 1
    stale_ratio = stale_pairs / float(max(1, total_pairs))
    return {
        "stale_efforts_detected": stale_ratio >= 0.9,
        "stale_effort_pair_ratio": round(stale_ratio, 4),
        "efforts_sources": sorted(set(effort_sources)),
    }


def _server_backed_ratio(frames: List[Dict[str, Any]]) -> float:
    if not frames:
        return 0.0
    backed = 0
    for frame in frames:
        obs = frame.get("observation", {})
        if not isinstance(obs, dict):
            continue
        prov = str(obs.get("scene_state_provenance") or "").lower()
        data_src = str(obs.get("data_source") or "").lower()
        if "kinematic" in prov or "synthetic" in prov:
            continue
        if data_src in ("between_waypoints", "real_composed") or prov in ("physx_server", "server", "real"):
            backed += 1
    return backed / float(len(frames))


def _valid_contact_stats(
    frames: List[Dict[str, Any]],
    target_id: str,
) -> Dict[str, Any]:
    valid_frames = 0
    contact_frames = 0
    gripper_target_frames = 0
    placeholder_frames = 0
    target_norm = _normalize_obj_token(target_id)

    for frame in frames:
        contacts = frame.get("collision_contacts")
        if not isinstance(contacts, list) or not contacts:
            continue
        contact_frames += 1
        frame_valid = False
        has_gripper_target = False
        has_placeholder = False
        for c in contacts:
            if not isinstance(c, dict):
                continue
            body_a = str(c.get("body_a") or "")
            body_b = str(c.get("body_b") or "")
            force = c.get("force_N")
            try:
                force_val = float(force if force is not None else 0.0)
            except (TypeError, ValueError):
                force_val = 0.0
            pen = c.get("penetration_depth")
            pen_ok = pen is None or isinstance(pen, (int, float))
            if not body_a and not body_b and abs(force_val) <= 1e-9 and not pen:
                has_placeholder = True
            if body_a and body_b and force_val > 0.0 and pen_ok:
                frame_valid = True
            a_norm = _normalize_obj_token(body_a)
            b_norm = _normalize_obj_token(body_b)
            if (
                (_looks_like_gripper_body(body_a) and target_norm in b_norm)
                or (_looks_like_gripper_body(body_b) and target_norm in a_norm)
            ):
                has_gripper_target = True
        if has_placeholder:
            placeholder_frames += 1
        if frame_valid:
            valid_frames += 1
        if has_gripper_target:
            gripper_target_frames += 1

    total = max(1, len(frames))
    return {
        "valid_contact_frames": valid_frames,
        "contact_frames": contact_frames,
        "valid_contact_frame_ratio": round(valid_frames / float(total), 4),
        "gripper_target_contact_frames": gripper_target_frames,
        "placeholder_contact_frames": placeholder_frames,
    }


def _camera_placeholder_present(frames: List[Dict[str, Any]]) -> bool:
    # Fast check: detect placeholders on the first non-empty camera frame.
    for frame in frames:
        obs = frame.get("observation", {})
        cams = obs.get("camera_frames") if isinstance(obs, dict) else None
        if not isinstance(cams, dict) or not cams:
            continue
        for cam in cams.values():
            if not isinstance(cam, dict):
                continue
            rgb = cam.get("rgb")
            depth = cam.get("depth")
            if rgb is not None and is_placeholder_rgb(np.asarray(rgb)):
                return True
            if depth is not None and is_placeholder_depth(np.asarray(depth)):
                return True
        break
    return False


def _camera_frame_coverage(frames: List[Dict[str, Any]]) -> Tuple[float, int, int]:
    total = len(frames)
    if total == 0:
        return 0.0, 0, 0
    complete = 0
    for frame in frames:
        obs = frame.get("observation", {})
        cams = obs.get("camera_frames") if isinstance(obs, dict) else None
        if not isinstance(cams, dict) or not cams:
            continue
        frame_ok = True
        for cam in cams.values():
            if not isinstance(cam, dict):
                frame_ok = False
                break
            if cam.get("rgb") is None or cam.get("depth") is None:
                frame_ok = False
                break
        if frame_ok:
            complete += 1
    return complete / float(total), complete, total


def _task_requires_motion(task: Dict[str, Any]) -> bool:
    explicit_keys = (
        "requires_object_motion",
        "requires_object_movement",
        "expect_object_motion",
        "expects_object_motion",
    )
    for key in explicit_keys:
        if key in task:
            try:
                return bool(task.get(key))
            except Exception:
                return False

    task_type = str(task.get("task_type") or task.get("task_name") or "").lower()
    if any(k in task_type for k in ("inspect", "observe", "scan", "detect", "classify", "segment")):
        return False
    if any(k in task_type for k in ("pick", "place", "grasp", "organize", "stack", "interact", "lift", "transport")):
        return True
    return bool(task.get("goal_region")) and bool(task.get("target_object") or task.get("target_object_id"))


def _compute_ee_target_min_dist(frames: List[Dict[str, Any]], target_id: str) -> Optional[float]:
    target_norm = _normalize_obj_token(target_id)
    min_dist = math.inf
    for frame in frames:
        ee = frame.get("ee_pos")
        if not (isinstance(ee, list) and len(ee) >= 3):
            continue
        obj_poses = frame.get("object_poses", {})
        if not isinstance(obj_poses, dict):
            continue
        tpos = None
        for oid, pose in obj_poses.items():
            if _normalize_obj_token(oid) != target_norm:
                continue
            if isinstance(pose, dict) and isinstance(pose.get("position"), list) and len(pose["position"]) >= 3:
                tpos = pose["position"][:3]
                break
        if tpos is None:
            continue
        d = float(np.linalg.norm(np.array(ee[:3], dtype=float) - np.array(tpos, dtype=float)))
        min_dist = min(min_dist, d)
    if math.isinf(min_dist):
        return None
    return min_dist


def _build_object_id_map(frames: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    mapping: Dict[str, set[str]] = defaultdict(set)
    for frame in frames:
        obj_poses = frame.get("object_poses", {})
        if not isinstance(obj_poses, dict):
            continue
        for oid in obj_poses.keys():
            canon = _normalize_obj_token(oid)
            if canon:
                mapping[canon].add(str(oid))
    return {k: sorted(v) for k, v in sorted(mapping.items())}


def _target_mass_present(episode_meta: Dict[str, Any], target_id: str) -> Optional[bool]:
    if not target_id:
        return None
    target_norm = _normalize_obj_token(target_id)
    object_metadata = episode_meta.get("object_metadata")
    if not isinstance(object_metadata, dict):
        return None
    for oid, meta in object_metadata.items():
        if _normalize_obj_token(oid) != target_norm:
            continue
        if not isinstance(meta, dict):
            return False
        return meta.get("mass_kg") is not None
    return None


def _target_source_ratio(frames: List[Dict[str, Any]], target_id: str) -> Optional[float]:
    target_norm = _normalize_obj_token(target_id)
    if not target_norm:
        return None
    total = 0
    server_backed = 0
    for frame in frames:
        obj_poses = frame.get("object_poses", {})
        if not isinstance(obj_poses, dict):
            continue
        for oid, pose in obj_poses.items():
            if _normalize_obj_token(oid) != target_norm or not isinstance(pose, dict):
                continue
            total += 1
            source = str(pose.get("source") or "").lower()
            if ("server" in source) or ("physx" in source):
                server_backed += 1
            break
    if total == 0:
        return None
    return server_backed / float(total)


def _target_velocity_coverage(frames: List[Dict[str, Any]], target_id: str) -> Optional[float]:
    target_norm = _normalize_obj_token(target_id)
    if not target_norm:
        return None
    total = 0
    with_velocity = 0
    for frame in frames:
        obj_poses = frame.get("object_poses", {})
        if not isinstance(obj_poses, dict):
            continue
        for oid, pose in obj_poses.items():
            if _normalize_obj_token(oid) != target_norm or not isinstance(pose, dict):
                continue
            total += 1
            if pose.get("linear_velocity") is not None and pose.get("angular_velocity") is not None:
                with_velocity += 1
            break
    if total == 0:
        return None
    return with_velocity / float(total)


def run_episode_certification(
    frames: List[Dict[str, Any]],
    episode_meta: Dict[str, Any],
    task: Dict[str, Any],
    *,
    mode: str = "strict",
) -> Dict[str, Any]:
    """Evaluate an episode for physics certification."""
    mode_norm = (mode or "strict").strip().lower()
    if mode_norm not in ("strict", "compat"):
        mode_norm = "strict"
    target_id = (
        task.get("target_object")
        or task.get("target_object_id")
        or episode_meta.get("target_object")
        or ""
    )
    has_target = bool(str(target_id).strip())
    target_positions = _extract_target_positions(frames, target_id) if target_id else []
    requires_motion = _task_requires_motion(task)

    teleport_jump_threshold_m = _env_float("PHYSICS_CERT_TELEPORT_JUMP_M", 0.12)
    release_snapback_threshold_m = _env_float("PHYSICS_CERT_RELEASE_SNAPBACK_M", 0.05)
    grasp_proximity_max_m = _env_float("PHYSICS_CERT_GRASP_PROXIMITY_MAX_M", 0.15)
    min_target_displacement_m = _env_float("PHYSICS_CERT_MIN_TARGET_DISPLACEMENT_M", 0.005)
    server_pose_coverage_min = _env_float("PHYSICS_CERT_SERVER_POSE_COVERAGE_MIN", 0.95)
    min_contact_frames_for_pick_place = int(_env_float("PHYSICS_CERT_MIN_CONTACT_FRAMES", 3))
    target_frame_presence_required = _env_float("PHYSICS_CERT_TARGET_FRAME_PRESENCE_REQUIRED", 1.0)
    required_channel_completeness = _env_float("PHYSICS_CERT_REQUIRED_CHANNEL_COMPLETENESS", 1.0)
    modality_profile = str(episode_meta.get("modality_profile") or "no_rgb").lower()

    # Phase B grasp-only dynamic toggle: objects kinematic at rest, dynamic during
    # manipulation.  When active, kinematic EE-offset frames are expected during
    # approach/retreat phases and server-backed ratio will be < 1.0.
    _phase_b_toggle = _env_bool("GENIESIM_REQUIRE_DYNAMIC_TOGGLE")
    _keep_kinematic = _env_bool("GENIESIM_KEEP_OBJECTS_KINEMATIC")
    _skip_stale_effort_gate = _env_bool("SKIP_STALE_EFFORT_CHANNEL_GATE")
    _skip_camera_hardcap = _env_bool("SKIP_CAMERA_HARDCAP")

    gate_failures: List[str] = []
    metrics: Dict[str, Any] = {}
    strict_runtime_patch_health = bool(episode_meta.get("strict_runtime_patch_health", True))
    metrics["strict_runtime_patch_health"] = strict_runtime_patch_health
    if mode_norm == "strict" and not strict_runtime_patch_health:
        gate_failures.append("STRICT_RUNTIME_PRECHECK_FAILED")

    # Kinematic source usage gate.
    kinematic_frames = 0
    for frame in frames:
        obs = frame.get("observation", {})
        prov = str(obs.get("scene_state_provenance") if isinstance(obs, dict) else "")
        if "kinematic_ee_offset" in prov:
            kinematic_frames += 1
            continue
        obj_poses = frame.get("object_poses", {})
        if isinstance(obj_poses, dict):
            for pose in obj_poses.values():
                if isinstance(pose, dict) and "kinematic_ee_offset" in str(pose.get("source", "")):
                    kinematic_frames += 1
                    break
    kinematic_ratio = kinematic_frames / float(max(1, len(frames)))
    metrics["kinematic_pose_frame_ratio"] = round(kinematic_ratio, 4)
    # Phase B toggle: kinematic frames during approach/retreat are expected.
    # Only fail if toggle is NOT active (all frames should be server-backed)
    # OR if ALL frames are kinematic (toggle never activated).
    if _phase_b_toggle and _keep_kinematic:
        # With grasp-only toggle, kinematic ratio ~0.3-0.7 is normal.
        # Only gate-fail if ratio is 1.0 (toggle never fired).
        if kinematic_ratio >= 1.0:
            gate_failures.append("KINEMATIC_OBJECT_POSE_USED")
    elif kinematic_frames > 0:
        gate_failures.append("KINEMATIC_OBJECT_POSE_USED")

    # Teleport/snapback gate.
    max_consecutive_disp = 0.0
    end_disp = 0.0
    snapback_detected = False
    if len(target_positions) >= 2:
        start = target_positions[0]
        end = target_positions[-1]
        end_disp = float(np.linalg.norm(end - start))
        max_from_start = max(float(np.linalg.norm(p - start)) for p in target_positions)
        for i in range(1, len(target_positions)):
            max_consecutive_disp = max(
                max_consecutive_disp,
                float(np.linalg.norm(target_positions[i] - target_positions[i - 1])),
            )
        if max_from_start >= release_snapback_threshold_m and end_disp <= 0.005:
            snapback_detected = True
    metrics["target_max_consecutive_disp_m"] = round(max_consecutive_disp, 6)
    metrics["target_end_displacement_m"] = round(end_disp, 6)
    metrics["snapback_detected"] = bool(snapback_detected)
    if max_consecutive_disp >= teleport_jump_threshold_m or snapback_detected:
        gate_failures.append("SNAPBACK_OR_TELEPORT_DETECTED")
    metrics["task_requires_motion"] = bool(requires_motion)
    metrics["target_motion_min_displacement_m"] = float(min_target_displacement_m)
    if requires_motion and has_target:
        _has_target_motion = len(target_positions) >= 2 and end_disp >= min_target_displacement_m
        metrics["target_motion_present"] = bool(_has_target_motion)
        if not _has_target_motion:
            gate_failures.append("EE_TARGET_GEOMETRY_IMPLAUSIBLE")
    else:
        metrics["target_motion_present"] = None

    # Success contradiction gate.
    task_success = episode_meta.get("task_success")
    goal_verif = episode_meta.get("goal_region_verification") or {}
    geometric_success = None
    if isinstance(goal_verif, dict) and goal_verif:
        milestones = (
            bool(goal_verif.get("grasp_detected")),
            bool(goal_verif.get("object_lifted_5cm")),
            bool(goal_verif.get("placed_in_goal")),
            bool(goal_verif.get("stable_at_end")),
            bool(goal_verif.get("gripper_released")),
        )
        geometric_success = sum(milestones) >= 4
    if geometric_success is None:
        geometric_success = bool(goal_verif.get("placed_in_goal")) if isinstance(goal_verif, dict) else None
    contradiction = bool(task_success is True and geometric_success is False)
    metrics["success_contradiction_count"] = 1 if contradiction else 0
    if contradiction:
        gate_failures.append("TASK_SUCCESS_CONTRADICTION")

    # Contact validity gate.
    contact_stats = _valid_contact_stats(frames, target_id)
    metrics.update(contact_stats)
    metrics["valid_nonzero_contact_ratio"] = metrics.get("valid_contact_frame_ratio", 0.0)
    task_type = str(task.get("task_type", "")).lower()
    is_manipulation = any(k in task_type for k in ("pick", "place", "organize", "stack", "grasp")) or requires_motion
    min_required_valid_contact_frames = min_contact_frames_for_pick_place if is_manipulation else 0
    _contact_gate_failed = False
    if contact_stats["placeholder_contact_frames"] > 0:
        # Phase B toggle: placeholder contacts during non-manipulation phases are
        # expected when contacts are synthesized for approach/retreat.  Only fail
        # if placeholders are present AND no valid contacts exist at all.
        if _phase_b_toggle and _keep_kinematic:
            if contact_stats["valid_contact_frames"] < 1:
                _contact_gate_failed = True
        else:
            _contact_gate_failed = True
    if (
        is_manipulation
        and not _contact_gate_failed
        and (
            contact_stats["valid_contact_frames"] < min_required_valid_contact_frames
            or (has_target and contact_stats["gripper_target_contact_frames"] < 1)
        )
    ):
        _contact_gate_failed = True
    if _contact_gate_failed:
        gate_failures.append("CONTACT_PLACEHOLDER_OR_EMPTY")

    # Target schema completeness gate.
    target_schema_ratio = None
    target_complete = 0
    target_total = len(frames)
    target_mass_present = _target_mass_present(episode_meta, target_id)
    if has_target:
        target_schema_ratio, target_complete, target_total = _target_schema_completeness(frames, target_id)
        metrics["target_schema_completeness"] = round(target_schema_ratio, 4)
        metrics["target_schema_complete_frames"] = target_complete
        metrics["target_schema_total_frames"] = target_total
        if target_total > 0 and target_schema_ratio < target_frame_presence_required:
            gate_failures.append("TARGET_SCHEMA_INCOMPLETE")
        if target_mass_present is False:
            gate_failures.append("TARGET_SCHEMA_INCOMPLETE")
    else:
        metrics["target_schema_completeness"] = None
        metrics["target_schema_complete_frames"] = None
        metrics["target_schema_total_frames"] = target_total
    metrics["target_mass_kg_present"] = target_mass_present
    _target_source = _target_source_ratio(frames, target_id) if has_target else None
    metrics["server_target_source_ratio"] = (
        round(float(_target_source), 4) if _target_source is not None else None
    )
    _target_vel_cov = _target_velocity_coverage(frames, target_id) if has_target else None
    metrics["target_velocity_coverage"] = (
        round(float(_target_vel_cov), 4) if _target_vel_cov is not None else None
    )

    # Channel completeness gate.
    channel_completeness = _channel_completeness(frames)
    metrics["channel_completeness_min"] = min(channel_completeness.values()) if channel_completeness else 0.0
    metrics["ee_velocity_coverage"] = channel_completeness.get("ee_vel", 0.0)
    metrics["ee_acceleration_coverage"] = channel_completeness.get("ee_acc", 0.0)
    if any(v < required_channel_completeness for v in channel_completeness.values()):
        gate_failures.append("CHANNEL_INCOMPLETE")
    stale_effort_stats = _stale_effort_stats(frames)
    metrics.update(stale_effort_stats)
    effort_source_policy = str(episode_meta.get("effort_source_policy") or "")
    stale_physx = (
        stale_effort_stats.get("stale_efforts_detected", False)
        and ("physx" in effort_source_policy or "physx" in " ".join(stale_effort_stats.get("efforts_sources", [])))
    )
    if stale_physx:
        # Phase B grasp-only toggle: non-manipulation frames may still have
        # identical efforts.  Accept stale ratio up to 0.9 when toggle is active.
        if _phase_b_toggle and _keep_kinematic:
            _stale_ratio = stale_effort_stats.get("stale_effort_pair_ratio", 1.0)
            if _stale_ratio >= 0.95:
                # Toggle didn't reduce staleness enough — still flag.
                gate_failures.append("CHANNEL_INCOMPLETE")
            # Otherwise: 0.3-0.9 is acceptable for grasp-only dynamic window.
        elif _skip_stale_effort_gate:
            pass  # Explicitly skipped via config.
        else:
            gate_failures.append("CHANNEL_INCOMPLETE")

    # EE-target geometry plausibility gate.
    min_ee_target_dist = _compute_ee_target_min_dist(frames, target_id) if target_id else None
    metrics["ee_target_min_distance_m"] = (
        round(float(min_ee_target_dist), 6) if min_ee_target_dist is not None else None
    )
    if min_ee_target_dist is not None:
        had_closed = any(str(f.get("gripper_command")) == "closed" for f in frames)
        if had_closed and min_ee_target_dist > grasp_proximity_max_m:
            gate_failures.append("EE_TARGET_GEOMETRY_IMPLAUSIBLE")

    # Scene-state server-backing gate.
    server_backed_ratio = _server_backed_ratio(frames)
    metrics["scene_state_server_backed_ratio"] = round(server_backed_ratio, 4)
    if server_backed_ratio < server_pose_coverage_min:
        # Phase B toggle: kinematic-at-rest means only manipulation frames are
        # server-backed (~40-65%).  Accept any ratio > 0.30 (at least some
        # frames came from the real PhysX server during the dynamic window).
        _phase_b_min = _env_float("PHYSICS_CERT_PHASE_B_SERVER_COVERAGE_MIN", 0.30)
        if _phase_b_toggle and _keep_kinematic and server_backed_ratio >= _phase_b_min:
            pass  # Acceptable for grasp-only dynamic window.
        else:
            gate_failures.append("SCENE_STATE_NOT_SERVER_BACKED")

    # Camera completeness/placeholder gate (required for RGB profile or explicit camera-required mode).
    camera_required = bool(episode_meta.get("camera_required")) or modality_profile != "no_rgb"
    if _skip_camera_hardcap:
        camera_required = False  # Proprioception-only mode: skip camera gates.
    metrics["camera_required"] = camera_required
    if camera_required:
        camera_cov, camera_complete_frames, camera_total_frames = _camera_frame_coverage(frames)
        metrics["camera_complete_frame_ratio"] = round(camera_cov, 4)
        metrics["camera_complete_frames"] = camera_complete_frames
        metrics["camera_total_frames"] = camera_total_frames
        if camera_cov < 1.0 or _camera_placeholder_present(frames):
            gate_failures.append("CAMERA_PLACEHOLDER_PRESENT")
    elif modality_profile != "no_rgb":
        if _camera_placeholder_present(frames):
            gate_failures.append("CAMERA_PLACEHOLDER_PRESENT")

    # Build task outcome and object mapping.
    environment_success = bool(episode_meta.get("collision_free_physics", True))
    canonical_task_success = bool(geometric_success) and bool(environment_success)
    llm_assessment = {
        "task_success": episode_meta.get("task_success"),
        "task_success_reasoning": episode_meta.get("task_success_reasoning"),
        "source": episode_meta.get("task_success_source") or "canonical_geometric_physics",
    }
    task_outcome = {
        "canonical_task_success": canonical_task_success,
        "geometric_success": bool(geometric_success) if geometric_success is not None else None,
        "environment_success": environment_success,
        "llm_assessment": llm_assessment,
    }
    object_id_map = _build_object_id_map(frames)

    # Gate failures are always considered certification blockers.
    critical_failures = sorted(set(gate_failures))
    passed = len(critical_failures) == 0

    dataset_tier = "physics_certified" if passed else "raw_preserved"
    return {
        "mode": mode_norm,
        "passed": passed,
        "critical_failures": critical_failures,
        "metrics": metrics,
        "gate_versions": {"physics_certification": GATE_VERSION},
        "dataset_tier": dataset_tier,
        "task_outcome": task_outcome,
        "channel_completeness": channel_completeness,
        "object_id_map": object_id_map,
    }


def write_run_certification_report(
    run_dir: Path,
    episode_reports: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Write run-level certification summary + JSONL."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    reports = [r for r in episode_reports if isinstance(r, dict)]
    gate_counter: Counter[str] = Counter()
    passed = 0
    failed = 0
    by_tier: Counter[str] = Counter()
    task_failures: Counter[str] = Counter()
    robot_failures: Counter[str] = Counter()
    for report in reports:
        tier = str(report.get("dataset_tier") or "raw_preserved")
        by_tier[tier] += 1
        cert = report.get("certification") or {}
        if cert.get("passed"):
            passed += 1
        else:
            failed += 1
            task_failures[str(report.get("task_name") or "unknown")] += 1
            robot_failures[str(report.get("robot_type") or "unknown")] += 1
        for code in cert.get("critical_failures", []) if isinstance(cert, dict) else []:
            gate_counter[str(code)] += 1

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "episodes": len(reports),
            "certified": passed,
            "raw_only": failed,
            "certification_pass_rate": round((passed / float(len(reports))) if reports else 0.0, 4),
            "dataset_tiers": dict(sorted(by_tier.items())),
            "gate_histogram": dict(sorted(gate_counter.items())),
            "top_failing_tasks": task_failures.most_common(10),
            "per_robot_failure_distribution": dict(sorted(robot_failures.items())),
        },
        "episodes": reports,
    }
    (run_dir / "run_certification_report.json").write_text(json.dumps(payload, indent=2))
    with open(run_dir / "run_certification_report.jsonl", "w") as jf:
        for report in reports:
            jf.write(json.dumps(report) + "\n")
    return payload
