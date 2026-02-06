import json
from pathlib import Path

from tools.quality_gates.physics_certification import (
    run_episode_certification,
    write_run_certification_report,
)


def _frame(
    *,
    scene_provenance: str = "physx_server",
    source: str = "physx_server",
    position=None,
    closed: bool = False,
    effort_value: float = 0.1,
):
    if position is None:
        position = [0.1, 0.0, 0.2]
    return {
        "ee_pos": [0.1, 0.0, 0.2],
        "ee_quat": [1.0, 0.0, 0.0, 0.0],
        "ee_vel": [0.0, 0.0, 0.0],
        "ee_acc": [0.0, 0.0, 0.0],
        "gripper_command": "closed" if closed else "open",
        "object_poses": {
            "lightwheel_kitchen_obj_Toaster003": {
                "position": list(position),
                "rotation_quat": [1.0, 0.0, 0.0, 0.0],
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "source": source,
            }
        },
        "collision_contacts": [
            {
                "body_a": "franka_gripper",
                "body_b": "lightwheel_kitchen_obj_Toaster003",
                "force_N": 1.2,
                "penetration_depth": 0.001,
                "position": [0.0, 0.0, 0.0],
                "normal": [0.0, 0.0, 1.0],
            }
        ],
        "observation": {
            "data_source": "real_composed",
            "scene_state_provenance": scene_provenance,
            "robot_state": {
                "joint_positions": [0.0] * 7,
                "joint_velocities": [0.0] * 7,
                "joint_accelerations": [0.0] * 7,
                "joint_efforts": [effort_value] * 7,
            },
        },
    }


def _episode_meta():
    return {
        "target_object": "lightwheel_kitchen_obj_Toaster003",
        "task_success": False,
        "task_success_reasoning": "Placed correctly.",
        "goal_region_verification": {
            "grasp_detected": True,
            "object_lifted_5cm": True,
            "placed_in_goal": True,
            "stable_at_end": True,
            "gripper_released": True,
        },
        "collision_free_physics": True,
        "modality_profile": "no_rgb",
        "effort_source_policy": "physx",
        "object_metadata": {
            "lightwheel_kitchen_obj_Toaster003": {"mass_kg": 0.8}
        },
    }


def test_physics_certification_flags_kinematic_object_pose():
    frames = [
        _frame(scene_provenance="kinematic_ee_offset_blocked", source="kinematic_ee_offset_blocked"),
        _frame(scene_provenance="kinematic_ee_offset_blocked", source="kinematic_ee_offset_blocked"),
    ]
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["dataset_tier"] == "raw_preserved"
    assert report["passed"] is False
    assert "KINEMATIC_OBJECT_POSE_USED" in report["critical_failures"]


def test_physics_certification_passes_clean_server_backed_episode():
    frames = [
        _frame(position=[0.1, 0.0, 0.2], effort_value=0.1),
        _frame(position=[0.11, 0.0, 0.2], effort_value=0.2),
        _frame(position=[0.12, 0.0, 0.2], effort_value=0.3),
    ]
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["passed"] is True
    assert report["dataset_tier"] == "physics_certified"
    assert report["critical_failures"] == []
    assert report["task_outcome"]["canonical_task_success"] is True


def test_physics_certification_detects_success_contradiction():
    frames = [
        _frame(position=[0.1, 0.0, 0.2]),
        _frame(position=[0.11, 0.0, 0.2]),
    ]
    meta = _episode_meta()
    meta["task_success"] = True
    meta["goal_region_verification"] = {
        "grasp_detected": True,
        "object_lifted_5cm": False,
        "placed_in_goal": False,
        "stable_at_end": False,
        "gripper_released": True,
    }
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, meta, task, mode="strict")

    assert report["passed"] is False
    assert "TASK_SUCCESS_CONTRADICTION" in report["critical_failures"]


def test_write_run_certification_report_outputs_json_and_jsonl(tmp_path: Path):
    reports = [
        {
            "episode_id": "ep1",
            "task_name": "task_a",
            "robot_type": "franka",
            "dataset_tier": "physics_certified",
            "certification": {"passed": True, "critical_failures": []},
        },
        {
            "episode_id": "ep2",
            "task_name": "task_b",
            "robot_type": "franka",
            "dataset_tier": "raw_preserved",
            "certification": {"passed": False, "critical_failures": ["CHANNEL_INCOMPLETE"]},
        },
    ]
    payload = write_run_certification_report(tmp_path, reports)

    assert (tmp_path / "run_certification_report.json").exists()
    assert (tmp_path / "run_certification_report.jsonl").exists()
    assert payload["summary"]["episodes"] == 2
    assert payload["summary"]["gate_histogram"]["CHANNEL_INCOMPLETE"] == 1

    jsonl_lines = (tmp_path / "run_certification_report.jsonl").read_text().strip().splitlines()
    assert len(jsonl_lines) == 2
    assert json.loads(jsonl_lines[0])["episode_id"] == "ep1"
