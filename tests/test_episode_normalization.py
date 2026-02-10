from tools.quality_gates.episode_normalization import normalize_episode_for_certification
from tools.quality_gates.physics_certification import run_episode_certification


def test_normalize_episode_for_certification_allows_offline_recert_phase_b(monkeypatch):
    # Match the Phase B / kinematic assumptions used by scripts/recertify_episodes.py
    monkeypatch.setenv("GENIESIM_REQUIRE_DYNAMIC_TOGGLE", "1")
    monkeypatch.setenv("GENIESIM_KEEP_OBJECTS_KINEMATIC", "1")
    monkeypatch.setenv("SKIP_STALE_EFFORT_CHANNEL_GATE", "true")
    monkeypatch.setenv("SKIP_CAMERA_HARDCAP", "true")

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

    assert report["passed"] is True
    assert report["critical_failures"] == []
    # The align-to-EE fallback should make the EE-target min distance plausible.
    assert report["metrics"]["ee_target_min_distance_m"] == 0.0
