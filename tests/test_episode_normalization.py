from tools.quality_gates.episode_normalization import (
    normalize_episode_for_certification,
    _retro_downgrade_task_success,
)
from tools.quality_gates.physics_certification import run_episode_certification


def test_normalize_episode_for_certification_fills_sparse_fields_but_does_not_fake_physics() -> None:
    # Raw episodes can be missing `object_poses` and have sparse joint/EE channels.
    # Normalization should fill what can be derived deterministically, but strict
    # physics certification must still fail when the episode lacks real contacts
    # and measurable target motion.
    episode = {
        "episode_id": "task_0_ep0000",
        "task_name": "task_0",
        "task_type": "pick_place",
        "target_object": "lightwheel_kitchen_obj_Toaster003",
        "task_success": True,
        "goal_region_verification": None,
        "frames": [
            {
                "dt": 0.1,
                "timestamp": 0.0,
                "ee_pos": [1.0, 1.0, 1.0],
                "gripper_command": "open",
                "observation": {
                    "data_source": "between_waypoints",
                    "robot_state": {
                        "ee_pose": {"rotation": {"rx": 0.0, "ry": 0.0, "rz": 0.0, "rw": 1.0}},
                        "joint_positions": [0.0] * 7,
                        "joint_velocities": [0.0] * 7,
                        "joint_efforts": [0.1] * 7,
                    },
                    "privileged": {
                        "scene_state": {
                            "objects": [
                                {"object_id": "/World/Toaster003", "pose": {"x": 0.0, "y": 0.0, "z": 0.0}},
                            ]
                        }
                    },
                },
            },
            {
                "dt": 0.1,
                "timestamp": 0.1,
                "ee_pos": [2.0, 2.0, 2.0],
                "gripper_command": "closed",
                "observation": {
                    "data_source": "between_waypoints",
                    "robot_state": {
                        "ee_pose": {"rotation": {"rx": 0.0, "ry": 0.0, "rz": 0.0, "rw": 1.0}},
                        "joint_positions": [0.0] * 7,
                        "joint_velocities": [0.0] * 7,
                        "joint_efforts": [0.1] * 7,
                    },
                    "privileged": {
                        "scene_state": {
                            "objects": [
                                {"object_id": "/World/Toaster003", "pose": {"x": 0.0, "y": 0.0, "z": 0.0}},
                            ]
                        }
                    },
                },
            },
            {
                "dt": 0.1,
                "timestamp": 0.2,
                "ee_pos": [3.0, 3.0, 3.0],
                "gripper_command": "open",
                "observation": {
                    "data_source": "between_waypoints",
                    "robot_state": {
                        "ee_pose": {"rotation": {"rx": 0.0, "ry": 0.0, "rz": 0.0, "rw": 1.0}},
                        "joint_positions": [0.0] * 7,
                        "joint_velocities": [0.0] * 7,
                        "joint_efforts": [0.1] * 7,
                    },
                    "privileged": {
                        "scene_state": {
                            "objects": [
                                {"object_id": "/World/Toaster003", "pose": {"x": 0.0, "y": 0.0, "z": 0.0}},
                            ]
                        }
                    },
                },
            },
        ],
    }

    normalize_episode_for_certification(episode)

    # Sanity checks: we filled missing channels and derived object_poses.
    for frame in episode["frames"]:
        assert frame.get("ee_quat") is not None
        assert frame.get("ee_vel") is not None
        assert frame.get("ee_acc") is not None
        rs = frame["observation"]["robot_state"]
        assert isinstance(rs.get("joint_accelerations"), list) and len(rs["joint_accelerations"]) > 0
        assert isinstance(frame.get("object_poses"), dict) and frame["object_poses"]

    task = {
        "task_type": episode["task_type"],
        "target_object": episode["target_object"],
        "requires_object_motion": True,
    }
    report = run_episode_certification(
        frames=episode["frames"],
        episode_meta=episode,
        task=task,
        mode="strict",
    )

    assert report["passed"] is False
    assert "CONTACT_PLACEHOLDER_OR_EMPTY" in report["critical_failures"]
    assert "EE_TARGET_GEOMETRY_IMPLAUSIBLE" in report["critical_failures"]


def test_retro_downgrade_overrides_success_when_zero_displacement() -> None:
    """When goal_region_verification shows zero displacement for a motion task,
    _retro_downgrade_task_success should override task_success to False."""
    episode = {
        "task_success": True,
        "task_type": "pick_place",
        "goal_region_verification": {
            "grasp_detected": True,
            "object_lifted_5cm": False,
            "placed_in_goal": False,
            "stable_at_end": True,
            "gripper_released": True,
            "displacement_m": 0.0,
        },
    }
    _retro_downgrade_task_success(episode)
    assert episode["task_success"] is False
    assert episode["task_success_source"] == "physics_override_zero_displacement"
    assert "task_success_physics_override" in episode
    assert episode["task_success_physics_override"]["override_reason"] == "zero_object_displacement"


def test_retro_downgrade_keeps_success_when_displacement_above_threshold() -> None:
    """When displacement is above 1cm, success should not be overridden."""
    episode = {
        "task_success": True,
        "task_type": "pick_place",
        "goal_region_verification": {
            "grasp_detected": True,
            "object_lifted_5cm": True,
            "placed_in_goal": True,
            "stable_at_end": True,
            "gripper_released": True,
            "displacement_m": 0.15,
        },
    }
    _retro_downgrade_task_success(episode)
    assert episode["task_success"] is True
