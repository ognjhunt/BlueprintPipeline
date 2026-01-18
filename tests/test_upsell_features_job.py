from __future__ import annotations

import json
import sys

from tests.contract_utils import validate_json_schema


def test_pipeline_integration_dry_run_outputs_config(
    load_job_module,
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    module = load_job_module("upsell_features", "pipeline_integration.py")
    scene_dir = tmp_path / "scene_123"
    scene_dir.mkdir()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline_integration.py",
            "--scene-dir",
            str(scene_dir),
            "--tier",
            "pro",
            "--robot-type",
            "ur5e",
            "--dry-run",
        ],
    )

    module.main()

    output = capsys.readouterr().out
    json_start = output.find("{")
    assert json_start != -1
    config = json.loads(output[json_start:])

    schema = {
        "type": "object",
        "required": [
            "scene_id",
            "bundle_tier",
            "robot_type",
            "episode_generation",
            "sensor_capture",
            "data_export",
        ],
        "properties": {
            "scene_id": {"type": "string"},
            "bundle_tier": {"type": "string"},
            "robot_type": {"type": "string"},
            "episode_generation": {
                "type": "object",
                "required": [
                    "max_variations",
                    "episodes_per_variation",
                    "total_episodes",
                ],
                "properties": {
                    "max_variations": {"type": "integer"},
                    "episodes_per_variation": {"type": "integer"},
                    "total_episodes": {"type": "integer"},
                },
            },
            "sensor_capture": {"type": "object"},
            "data_export": {"type": "object"},
        },
    }
    validate_json_schema(config, schema)

    assert config["scene_id"] == "scene_123"
    assert config["bundle_tier"] == "pro"
    assert config["robot_type"] == "ur5e"
