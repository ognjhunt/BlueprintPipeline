from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.geniesim_adapter import local_framework as lf


def test_scene_state_rotation_and_velocity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PIPELINE_ENV", "development")
    monkeypatch.setenv("ALLOW_GENIESIM_MOCK", "1")
    monkeypatch.setattr(lf, "IK_PLANNING_AVAILABLE", False)

    recording_dir = tmp_path / "recordings"
    recording_dir.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    config = lf.GenieSimConfig(
        geniesim_root=tmp_path / "missing_geniesim",
        recording_dir=recording_dir,
        log_dir=log_dir,
        robot_type="franka",
    )
    framework = lf.GenieSimLocalFramework(config=config, verbose=False)
    framework._units_per_meter = 1.0
    framework._client._joint_names = [f"joint_{i}" for i in range(7)]

    trajectory = [
        {"joint_positions": [0.0] * 7, "timestamp": 0.0, "gripper_aperture": 1.0, "phase": "approach"},
        {"joint_positions": [0.1] * 7, "timestamp": 0.1, "gripper_aperture": 1.0, "phase": "approach"},
    ]

    observations = [
        {
            "timestamp": 0.0,
            "data_source": "real_composed",
            "scene_state": {
                "objects": [
                    {
                        "object_id": "obj1",
                        "object_type": "box",
                        "pose": {
                            "position": [0.0, 0.0, 0.0],
                            "rotation": {"rw": 1.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
                        },
                    }
                ]
            },
            "robot_state": {
                "joint_positions": [0.0] * 7,
                "joint_velocities": [0.0] * 7,
                "joint_efforts": [0.0] * 7,
            },
        },
        {
            "timestamp": 0.1,
            "data_source": "real_composed",
            "scene_state": {
                "objects": [
                    {
                        "object_id": "obj1",
                        "object_type": "box",
                        "pose": {
                            "position": [0.1, 0.0, 0.0],
                            "rotation": {"rw": 1.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
                        },
                    }
                ]
            },
            "robot_state": {
                "joint_positions": [0.1] * 7,
                "joint_velocities": [0.0] * 7,
                "joint_efforts": [0.0] * 7,
            },
        },
    ]

    frames, _stats = framework._build_frames_from_trajectory(
        trajectory,
        observations,
        task={
            "task_name": "test",
            "target_object": "obj1",
            "robot_config": {"base_position": [0.0, 0.0, 0.0]},
        },
        episode_id="ep_test",
        output_dir=tmp_path,
    )

    moving_idx = None
    for i, frame in enumerate(frames):
        pos = frame.get("object_poses", {}).get("obj1", {}).get("position")
        if pos and pos[0] > 0.0:
            moving_idx = i
            break

    assert moving_idx is not None
    obj_state = frames[moving_idx]["object_poses"]["obj1"]
    assert obj_state["rotation_quat"] == [1.0, 0.0, 0.0, 0.0]
    if moving_idx > 0:
        prev_pos = frames[moving_idx - 1]["object_poses"]["obj1"]["position"]
        dt = frames[moving_idx]["timestamp"] - frames[moving_idx - 1]["timestamp"]
        if dt > 0:
            expected_v = (obj_state["position"][0] - prev_pos[0]) / dt
            assert obj_state["linear_velocity"][0] == pytest.approx(expected_v, rel=1e-3)
