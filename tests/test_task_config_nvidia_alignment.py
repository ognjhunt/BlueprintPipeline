from __future__ import annotations

from tools.geniesim_adapter.task_config import GenieSimTaskConfig, RobotConfig, SuggestedTask


def test_suggested_task_to_dict_emits_curriculum_and_cameras() -> None:
    task = SuggestedTask(
        task_type="pick_place",
        target_object="mug_0",
        required_camera_ids=["head", "left_wrist"],
    )

    payload = task.to_dict(scene_id="scene_123")

    assert payload["task_id"]
    assert payload["task_name"]
    assert payload["task_complexity"] in {"atomic", "composite"}
    assert payload["curriculum_split"] in {"pretrain", "target"}
    assert set(payload["required_camera_ids"]) == {"left", "right", "wrist"}


def test_task_config_to_dict_contains_required_taxonomy_fields() -> None:
    config = GenieSimTaskConfig(
        scene_id="scene_123",
        environment_type="kitchen",
        suggested_tasks=[
            SuggestedTask(
                task_type="organize",
                target_object="bowl_0",
                task_name="organize_bowl",
            )
        ],
        robot_config=RobotConfig(robot_type="franka"),
    )

    payload = config.to_dict()
    assert "suggested_tasks" in payload
    assert len(payload["suggested_tasks"]) == 1
    task = payload["suggested_tasks"][0]
    assert task["task_id"]
    assert task["task_name"] == "organize_bowl"
    assert task["task_complexity"] == "composite"
    assert task["curriculum_split"] == "target"
    assert set(task["required_camera_ids"]) == {"left", "right", "wrist"}
