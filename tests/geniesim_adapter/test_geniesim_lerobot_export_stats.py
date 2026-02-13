from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.geniesim_adapter.local_framework import GenieSimConfig, GenieSimLocalFramework


@pytest.mark.unit
def test_geniesim_lerobot_export_v3_writes_vector_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording_dir = tmp_path / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)

    episode = {
        "episode_id": "task_0_ep0000",
        "task_name": "task_0",
        "task_description": "Pick and place.",
        "dataset_tier": "physics_certified",
        "certification": {"passed": True, "critical_failures": []},
        "quality_score": 0.95,
        "frames": [
            {
                "timestamp": 0.0,
                "observation": {"robot_state": {"joint_positions": [0.0] * 7}},
                "action": [0.0] * 8,
                "reward": 0.0,
            },
            {
                "timestamp": 0.1,
                "observation": {"robot_state": {"joint_positions": [0.1] * 7}},
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

    monkeypatch.setenv("LEROBOT_EXPORT_CERTIFIED_ONLY", "true")
    out_dir = tmp_path / "lerobot"
    export = framework.export_to_lerobot(
        recording_dir=recording_dir,
        output_dir=out_dir,
        export_format="lerobot_v3",
        min_quality_score=0.0,
    )
    assert export["success"] is True
    assert export["exported"] == 1

    stats = json.loads((out_dir / "meta" / "stats.json").read_text())
    assert "observation.state" in stats
    assert "action" in stats
    assert len(stats["observation.state"]["mean"]) == 7
    assert len(stats["action"]["mean"]) == 8

