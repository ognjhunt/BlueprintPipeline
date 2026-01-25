import json

from tools.geniesim_idempotency import (
    build_geniesim_idempotency_inputs,
    build_quality_thresholds,
)


def test_geniesim_idempotency_inputs_match_workflow_shape() -> None:
    quality_thresholds = build_quality_thresholds(
        min_quality_score=0.85,
        filter_low_quality=True,
    )
    local_inputs = build_geniesim_idempotency_inputs(
        scene_id="scene-1",
        task_config={"tasks": [{"name": "pick"}]},
        robot_types=["franka", "ur10"],
        episodes_per_task=10,
        quality_thresholds=quality_thresholds,
    )
    cloud_inputs = {
        "scene_id": "scene-1",
        "task_config": {"tasks": [{"name": "pick"}]},
        "robot_types": ["franka", "ur10"],
        "episodes_per_task": 10,
        "quality_thresholds": {
            "min_quality_score": 0.85,
            "filter_low_quality": True,
        },
    }

    local_serialized = json.dumps(local_inputs, sort_keys=True, separators=(",", ":"))
    cloud_serialized = json.dumps(cloud_inputs, sort_keys=True, separators=(",", ":"))

    assert local_serialized == cloud_serialized
