from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.geniesim_adapter.local_framework import GenieSimConfig, GenieSimLocalFramework


@pytest.mark.unit
def test_geniesim_lerobot_export_v3_writes_depth_point_cloud_and_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording_dir = tmp_path / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = recording_dir / "task_0_ep0000_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    def _write_rgb(cam_id: str, frame_idx: int, rgb: np.ndarray) -> str:
        rel = f"task_0_ep0000_frames/{cam_id}_rgb_{frame_idx:06d}.npy"
        np.save(recording_dir / rel, rgb)
        return rel

    def _write_depth(cam_id: str, frame_idx: int, depth: np.ndarray) -> str:
        rel = f"task_0_ep0000_frames/{cam_id}_depth_{frame_idx:06d}.npy"
        np.save(recording_dir / rel, depth)
        return rel

    rgb0 = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb1 = np.full((8, 8, 3), 255, dtype=np.uint8)
    depth0 = np.full((8, 8), 1.0, dtype=np.float32)
    depth1 = np.full((8, 8), 1.1, dtype=np.float32)

    extr = np.eye(4, dtype=np.float64).tolist()

    episode = {
        "episode_id": "task_0_ep0000",
        "task_name": "task_0",
        "task_description": "Pick and place.",
        "task_id": "pick_place::mug",
        "required_camera_ids": ["left"],
        "dataset_tier": "physics_certified",
        "certification": {"passed": True, "critical_failures": []},
        "quality_score": 0.99,
        "frames": [
            {
                "timestamp": 0.0,
                "observation": {
                    "robot_state": {"joint_positions": [0.0] * 7},
                    "camera_frames": {
                        "left": {
                            "rgb": _write_rgb("left", 0, rgb0),
                            "depth": _write_depth("left", 0, depth0),
                            "width": 8,
                            "height": 8,
                            "fx": 100.0,
                            "fy": 100.0,
                            "ppx": 4.0,
                            "ppy": 4.0,
                            "extrinsic": extr,
                            "depth_encoding": "npy_float32_m",
                        }
                    },
                },
                "action": [0.0] * 8,
                "reward": 0.0,
            },
            {
                "timestamp": 0.1,
                "observation": {
                    "robot_state": {"joint_positions": [0.1] * 7},
                    "camera_frames": {
                        "left": {
                            "rgb": _write_rgb("left", 1, rgb1),
                            "depth": _write_depth("left", 1, depth1),
                            "width": 8,
                            "height": 8,
                            "fx": 100.0,
                            "fy": 100.0,
                            "ppx": 4.0,
                            "ppy": 4.0,
                            "extrinsic": extr,
                            "depth_encoding": "npy_float32_m",
                        }
                    },
                },
                "action": [0.1] * 8,
                "reward": 1.0,
                "done": True,
            },
        ],
    }
    (recording_dir / "task_0_ep0000.json").write_text(json.dumps(episode))

    cfg = GenieSimConfig(
        recording_dir=recording_dir,
        log_dir=tmp_path / "logs",
    )
    framework = GenieSimLocalFramework(cfg, verbose=False)

    monkeypatch.setenv("LEROBOT_EXPORT_INCLUDE_DEPTH", "1")
    monkeypatch.setenv("LEROBOT_EXPORT_INCLUDE_POINT_CLOUD", "1")
    monkeypatch.setenv("LEROBOT_POINT_CLOUD_MAX_POINTS", "16")
    monkeypatch.setenv("LEROBOT_POINT_CLOUD_MAX_DEPTH_M", "10.0")
    monkeypatch.setenv("LEROBOT_POINT_CLOUD_FRAME", "world")

    out_dir = tmp_path / "lerobot"
    export = framework.export_to_lerobot(
        recording_dir=recording_dir,
        output_dir=out_dir,
        export_format="lerobot_v3",
        min_quality_score=0.0,
    )
    assert export["success"] is True
    assert export["exported"] == 1

    # v3 chunk/file naming for episode_index=0
    calib_path = out_dir / "ground_truth" / "chunk-000" / "camera_calibration" / "file-0000.json"
    depth_path = out_dir / "ground_truth" / "chunk-000" / "depth" / "left" / "file-0000.npz"
    pc_path = out_dir / "ground_truth" / "chunk-000" / "point_cloud" / "left" / "file-0000.npz"

    assert calib_path.is_file()
    assert depth_path.is_file()
    assert pc_path.is_file()

    depth_npz = np.load(depth_path)
    assert depth_npz["depth"].shape == (2, 8, 8)
    assert depth_npz["valid_mask"].shape == (2,)
    assert bool(depth_npz["valid_mask"][0]) is True

    pc_npz = np.load(pc_path)
    assert pc_npz["points"].shape == (2, 16, 3)
    assert pc_npz["colors"].shape == (2, 16, 3)
    assert pc_npz["valid_counts"].shape == (2,)

    calib = json.loads(calib_path.read_text())
    assert "camera_calibration" in calib
    assert "left" in calib["camera_calibration"]
    assert calib["camera_calibration"]["left"]["fx"] == 100.0

    # Ensure the episodes index references the ground-truth artifacts.
    meta_line = (out_dir / "meta" / "episodes.jsonl").read_text().splitlines()[0]
    meta = json.loads(meta_line)
    gt = meta.get("ground_truth") or {}
    assert gt.get("camera_calibration_path")
    assert "left" in (gt.get("depth_paths") or {})
    assert "left" in (gt.get("point_cloud_paths") or {})

