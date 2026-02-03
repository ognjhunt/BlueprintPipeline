#!/usr/bin/env python3
"""Offline dataset audit for GenieSim outputs (no Isaac Sim required)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_SCAN_DIRS = ("downloaded_episodes", "local_runs")


@dataclass
class AuditConfig:
    data_tier: str = "full"
    missing_threshold: float = 0.1
    require_camera: bool = True
    require_depth: bool = True
    require_segmentation: bool = True
    require_calibration: bool = True
    require_contacts: bool = True
    require_ee_wrench: bool = True
    require_efforts: bool = True
    require_privileged: bool = True


def _resolve_requirements(data_tier: str) -> AuditConfig:
    tier = (data_tier or "core").lower()
    require_camera = True
    require_depth = tier in ("plus", "full")
    require_segmentation = tier in ("plus", "full")
    require_calibration = tier == "full"
    require_contacts = tier == "full"
    require_ee_wrench = tier == "full"
    require_efforts = tier == "full"
    require_privileged = tier == "full"
    return AuditConfig(
        data_tier=tier,
        require_camera=require_camera,
        require_depth=require_depth,
        require_segmentation=require_segmentation,
        require_calibration=require_calibration,
        require_contacts=require_contacts,
        require_ee_wrench=require_ee_wrench,
        require_efforts=require_efforts,
        require_privileged=require_privileged,
    )


def _iter_episode_files(paths: Iterable[Path]) -> Iterable[Path]:
    skip_names = {
        "dataset_info.json",
        "info.json",
        "episode_index.json",
        "_task_complete.json",
        "_completed_tasks.json",
        "scene_manifest.json",
        "task_config.json",
    }
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path.name in skip_names:
                continue
            if "ep" not in path.name:
                continue
            yield path


def _load_episode(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "frames" not in data or not isinstance(data.get("frames"), list):
        return None
    return data


def _has_media(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bytes, bytearray)):
        return len(value) > 0
    if isinstance(value, str):
        if not value:
            return False
        return True
    if isinstance(value, list):
        return len(value) > 0
    return True


def _collect_camera_ids(frames: List[Dict[str, Any]]) -> List[str]:
    camera_ids: List[str] = []
    for frame in frames:
        cam_frames = (frame.get("observation") or {}).get("camera_frames")
        if isinstance(cam_frames, dict):
            for cam_id in cam_frames.keys():
                if cam_id not in camera_ids:
                    camera_ids.append(cam_id)
    return camera_ids


def _extract_frame_timestamp(frame: Dict[str, Any]) -> Optional[float]:
    if frame.get("timestamp") is not None:
        return float(frame.get("timestamp"))
    obs = frame.get("observation") or {}
    if obs.get("timestamp") is not None:
        return float(obs.get("timestamp"))
    return None


def _contains_nan_inf(values: List[Any]) -> bool:
    for val in values:
        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                return True
        elif isinstance(val, list):
            if _contains_nan_inf(val):
                return True
    return False


def _extract_numeric_fields(obs: Dict[str, Any], frame: Dict[str, Any]) -> List[Any]:
    robot_state = obs.get("robot_state") or {}
    fields = [
        robot_state.get("joint_positions"),
        robot_state.get("joint_velocities"),
        robot_state.get("joint_efforts"),
        robot_state.get("joint_accelerations"),
        robot_state.get("ee_pos"),
        robot_state.get("ee_quat"),
        frame.get("action"),
    ]
    return [field for field in fields if field is not None]


def _audit_episode(path: Path, data: Dict[str, Any], config: AuditConfig) -> Dict[str, Any]:
    frames = data.get("frames", [])
    total_frames = len(frames)
    camera_ids = _collect_camera_ids(frames)

    missing_rgb_frames = 0
    missing_depth_frames = 0
    missing_seg_frames = 0
    missing_contact_frames = 0
    missing_effort_frames = 0
    missing_wrench_frames = 0
    missing_privileged_frames = 0
    joint_mismatch_frames = 0
    nan_inf_frames = 0

    seg_keys = (
        "semantic",
        "semantic_mask",
        "semantic_data",
        "instance",
        "instance_mask",
        "instance_segmentation",
        "semantic_segmentation",
    )

    timestamps: List[float] = []
    dts: List[float] = []
    last_ts: Optional[float] = None

    for frame in frames:
        obs = frame.get("observation") or {}
        cam_frames = obs.get("camera_frames") or {}
        if config.require_privileged:
            privileged = obs.get("privileged")
            if not isinstance(privileged, dict) or not privileged:
                missing_privileged_frames += 1

        frame_missing_rgb = False
        frame_missing_depth = False
        frame_missing_seg = False

        if config.require_camera and camera_ids:
            for cam_id in camera_ids:
                cam = cam_frames.get(cam_id) if isinstance(cam_frames, dict) else None
                if not isinstance(cam, dict):
                    frame_missing_rgb = True
                    if config.require_depth:
                        frame_missing_depth = True
                    if config.require_segmentation:
                        frame_missing_seg = True
                    continue
                if not _has_media(cam.get("rgb") or cam.get("rgb_ref")):
                    frame_missing_rgb = True
                if config.require_depth and not _has_media(cam.get("depth") or cam.get("depth_ref")):
                    frame_missing_depth = True
                if config.require_segmentation:
                    if not any(_has_media(cam.get(key)) for key in seg_keys):
                        frame_missing_seg = True
            if frame_missing_rgb:
                missing_rgb_frames += 1
            if frame_missing_depth:
                missing_depth_frames += 1
            if frame_missing_seg:
                missing_seg_frames += 1

        if config.require_contacts:
            privileged = obs.get("privileged") or {}
            contact_forces = privileged.get("contact_forces") if isinstance(privileged, dict) else None
            if not isinstance(contact_forces, dict) or contact_forces.get("available") is False:
                missing_contact_frames += 1

        if config.require_efforts:
            efforts = (obs.get("robot_state") or {}).get("joint_efforts", [])
            if not isinstance(efforts, list) or not any(abs(v) > 1e-6 for v in efforts):
                missing_effort_frames += 1

        if config.require_ee_wrench:
            wrench = (obs.get("robot_state") or {}).get("ee_wrench")
            if not wrench:
                wrench = frame.get("ee_wrench")
            if not wrench:
                missing_wrench_frames += 1

        robot_state = obs.get("robot_state") or {}
        jp = robot_state.get("joint_positions", [])
        jv = robot_state.get("joint_velocities", [])
        je = robot_state.get("joint_efforts", [])
        ja = robot_state.get("joint_accelerations", [])
        if isinstance(jp, list) and jp:
            for seq in (jv, je, ja):
                if isinstance(seq, list) and len(seq) not in (0, len(jp)):
                    joint_mismatch_frames += 1
                    break

        numeric_fields = _extract_numeric_fields(obs, frame)
        if numeric_fields and _contains_nan_inf(numeric_fields):
            nan_inf_frames += 1

        ts = _extract_frame_timestamp(frame)
        if ts is not None:
            timestamps.append(ts)
            if last_ts is not None:
                dts.append(ts - last_ts)
            last_ts = ts

    calibration_ok = True
    if config.require_calibration and camera_ids:
        calib = data.get("camera_calibration")
        if isinstance(calib, dict) and calib:
            for cam_id in camera_ids:
                cam_calib = calib.get(cam_id, {})
                if not cam_calib:
                    calibration_ok = False
                    break
                if cam_calib.get("fx") is None or cam_calib.get("fy") is None:
                    calibration_ok = False
                    break
                if cam_calib.get("ppx") is None or cam_calib.get("ppy") is None:
                    calibration_ok = False
                    break
                if cam_calib.get("extrinsic_matrix") is None and cam_calib.get("extrinsic") is None:
                    calibration_ok = False
                    break
        else:
            calibration_ok = False

    non_monotonic = any(dt < 0 for dt in dts)
    dt_outlier_ratio = 0.0
    if dts:
        median_dt = sorted(dts)[len(dts) // 2]
        if median_dt > 0:
            outliers = sum(1 for dt in dts if abs(dt - median_dt) > 0.5 * median_dt)
            dt_outlier_ratio = outliers / len(dts)

    def _ratio(count: int) -> float:
        return count / max(1, total_frames)

    failures: List[str] = []
    if config.require_camera and _ratio(missing_rgb_frames) >= config.missing_threshold:
        failures.append("missing_rgb")
    if config.require_depth and _ratio(missing_depth_frames) >= config.missing_threshold:
        failures.append("missing_depth")
    if config.require_segmentation and _ratio(missing_seg_frames) >= config.missing_threshold:
        failures.append("missing_segmentation")
    if config.require_contacts and _ratio(missing_contact_frames) >= config.missing_threshold:
        failures.append("missing_contacts")
    if config.require_efforts and _ratio(missing_effort_frames) >= config.missing_threshold:
        failures.append("missing_efforts")
    if config.require_ee_wrench and _ratio(missing_wrench_frames) >= config.missing_threshold:
        failures.append("missing_ee_wrench")
    if config.require_privileged and _ratio(missing_privileged_frames) >= config.missing_threshold:
        failures.append("missing_privileged")
    if config.require_calibration and not calibration_ok:
        failures.append("missing_calibration")
    if joint_mismatch_frames > 0:
        failures.append("joint_length_mismatch")
    if nan_inf_frames > 0:
        failures.append("nan_inf_values")
    if non_monotonic:
        failures.append("timestamp_non_monotonic")
    if dt_outlier_ratio >= config.missing_threshold:
        failures.append("dt_inconsistent")

    return {
        "path": str(path),
        "episode_id": data.get("episode_id", path.stem),
        "frame_count": total_frames,
        "camera_ids": camera_ids,
        "metrics": {
            "missing_rgb_frames": missing_rgb_frames,
            "missing_depth_frames": missing_depth_frames,
            "missing_segmentation_frames": missing_seg_frames,
            "missing_contact_frames": missing_contact_frames,
            "missing_effort_frames": missing_effort_frames,
            "missing_wrench_frames": missing_wrench_frames,
            "missing_privileged_frames": missing_privileged_frames,
            "joint_mismatch_frames": joint_mismatch_frames,
            "nan_inf_frames": nan_inf_frames,
            "timestamp_non_monotonic": non_monotonic,
            "dt_outlier_ratio": round(dt_outlier_ratio, 4),
            "calibration_ok": calibration_ok,
        },
        "ratios": {
            "missing_rgb_ratio": round(_ratio(missing_rgb_frames), 4),
            "missing_depth_ratio": round(_ratio(missing_depth_frames), 4),
            "missing_segmentation_ratio": round(_ratio(missing_seg_frames), 4),
            "missing_contact_ratio": round(_ratio(missing_contact_frames), 4),
            "missing_effort_ratio": round(_ratio(missing_effort_frames), 4),
            "missing_wrench_ratio": round(_ratio(missing_wrench_frames), 4),
            "missing_privileged_ratio": round(_ratio(missing_privileged_frames), 4),
        },
        "failures": failures,
        "passed": len(failures) == 0,
    }


def run_audit(paths: Iterable[Path], config: AuditConfig) -> Dict[str, Any]:
    episodes: List[Dict[str, Any]] = []
    for path in _iter_episode_files(paths):
        data = _load_episode(path)
        if not data:
            continue
        episodes.append(_audit_episode(path, data, config))

    failed = [ep for ep in episodes if not ep.get("passed", False)]
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": asdict(config),
        "summary": {
            "episodes_scanned": len(episodes),
            "failed_episodes": len(failed),
            "passed_episodes": len(episodes) - len(failed),
        },
        "episodes": episodes,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline dataset audit (no Isaac Sim required).")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=list(DEFAULT_SCAN_DIRS),
        help="Directories to scan for episode JSONs.",
    )
    parser.add_argument(
        "--data-tier",
        default="full",
        choices=["core", "plus", "full"],
        help="Data tier requirements to enforce.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.1,
        help="Max allowed missing ratio before failing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write JSON report.",
    )
    args = parser.parse_args()

    config = _resolve_requirements(args.data_tier)
    config.missing_threshold = args.missing_threshold
    paths = [Path(p) for p in args.paths]

    report = run_audit(paths, config)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    else:
        print(json.dumps(report, indent=2))

    return 0 if report["summary"]["failed_episodes"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
