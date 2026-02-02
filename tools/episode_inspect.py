#!/usr/bin/env python3
"""Inspect a GenieSim episode JSON for camera storage, quality, and confidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _classify_value(val: Any) -> str:
    if val is None:
        return "none"
    if isinstance(val, str):
        if val.endswith(".npy"):
            return "npy"
        return "base64"
    return type(val).__name__


def inspect_episode(path: Path) -> Dict[str, Any]:
    size_bytes = path.stat().st_size
    with path.open() as f:
        data = json.load(f)
    frames = data.get("frames", [])
    cam_summary: Dict[str, Dict[str, int]] = {}
    for frame in frames:
        obs = frame.get("observation", {}) or {}
        cam_frames = obs.get("camera_frames", {}) or {}
        if not isinstance(cam_frames, dict):
            continue
        for cam_id, cam_data in cam_frames.items():
            if not isinstance(cam_data, dict):
                continue
            cam_entry = cam_summary.setdefault(cam_id, {})
            for key in ("rgb", "depth"):
                v = cam_data.get(key)
                k = f"{key}_{_classify_value(v)}"
                cam_entry[k] = cam_entry.get(k, 0) + 1
    return {
        "episode_id": data.get("episode_id"),
        "frame_count": len(frames),
        "quality_score": data.get("quality_score"),
        "quality_score_breakdown_present": data.get("quality_score_breakdown") is not None,
        "channel_confidence": data.get("channel_confidence", {}),
        "file_size_bytes": size_bytes,
        "file_size_mb": round(size_bytes / (1024 * 1024), 3),
        "camera_summary": cam_summary,
        "frame_validation_errors": len((data.get("frame_validation") or {}).get("errors", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a GenieSim episode JSON.")
    parser.add_argument("episode_json", type=Path, help="Path to episode JSON")
    args = parser.parse_args()
    summary = inspect_episode(args.episode_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
