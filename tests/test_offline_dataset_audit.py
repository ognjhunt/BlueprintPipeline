import json
from pathlib import Path

from tools.quality_gates import offline_dataset_audit as audit


def test_offline_audit_detects_missing_modalities(tmp_path: Path) -> None:
    episode_dir = tmp_path / "downloaded_episodes" / "task_000_task_0" / "raw_episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    episode = {
        "episode_id": "task_0_ep0000",
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
                    },
                    "privileged": {},
                    "robot_state": {
                        "joint_positions": [0.0] * 7,
                        "joint_efforts": [0.0] * 7,
                    },
                },
                "action": [0.0],
            }
        ],
    }

    episode_path = episode_dir / "task_0_ep0000.json"
    episode_path.write_text(json.dumps(episode))

    config = audit._resolve_requirements("full")
    report = audit.run_audit([tmp_path], config, workers=1)

    assert report["summary"]["failed_episodes"] == 1
    assert "gate_histogram" in report["summary"]
    assert "baseline_metrics" in report["summary"]
