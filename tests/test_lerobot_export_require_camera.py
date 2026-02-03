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
