import json
from pathlib import Path

import pytest

from tools.geniesim_adapter.local_framework import GenieSimConfig, GenieSimLocalFramework


def test_export_to_lerobot_requires_camera_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording_dir = tmp_path / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)

    episode = {
        "episode_id": "task_0_ep0000",
        "task_name": "task_0",
        "dataset_tier": "physics_certified",
        "certification": {"passed": True, "critical_failures": []},
        "quality_score": 0.9,
        "frames": [
            {
                "timestamp": 0.0,
                "observation": {
                    "camera_frames": {
                        "wrist": {
                            "rgb": None,
                            "depth": None,
                            "width": 64,
                            "height": 64,
                        }
                    }
                },
                "action": [0.0],
            }
        ],
    }

    episode_path = recording_dir / "task_0_ep0000.json"
    episode_path.write_text(json.dumps(episode))

    cfg = GenieSimConfig(
        recording_dir=recording_dir,
        log_dir=tmp_path / "logs",
    )
    framework = GenieSimLocalFramework(cfg, verbose=False)

    monkeypatch.setenv("REQUIRE_CAMERA_DATA", "true")
    with pytest.raises(RuntimeError):
        framework.export_to_lerobot(
            recording_dir=recording_dir,
            output_dir=tmp_path / "lerobot",
        )


def test_export_to_lerobot_skips_non_certified_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording_dir = tmp_path / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)

    episode = {
        "episode_id": "task_0_ep0001",
        "task_name": "task_0",
        "dataset_tier": "raw_preserved",
        "certification": {"passed": False, "critical_failures": ["CHANNEL_INCOMPLETE"]},
        "quality_score": 0.95,
        "frames": [
            {
                "timestamp": 0.0,
                "observation": {"robot_state": {"joint_positions": [0.0] * 7}},
                "action": [0.0],
            }
        ],
    }
    (recording_dir / "task_0_ep0001.json").write_text(json.dumps(episode))

    cfg = GenieSimConfig(
        recording_dir=recording_dir,
        log_dir=tmp_path / "logs",
    )
    framework = GenieSimLocalFramework(cfg, verbose=False)

    monkeypatch.setenv("LEROBOT_EXPORT_CERTIFIED_ONLY", "true")
    export = framework.export_to_lerobot(
        recording_dir=recording_dir,
        output_dir=tmp_path / "lerobot",
    )
    dataset_info = json.loads((tmp_path / "lerobot" / "dataset_info.json").read_text())

    assert export["exported"] == 0
    assert export["skipped"] == 1
    assert export["exported_raw_only"] == 1
    assert export["rejected_by_gate_code"]["CHANNEL_INCOMPLETE"] == 1
    assert dataset_info["eligible_certified"] == 0
    assert dataset_info["exported_raw_only"] == 1
