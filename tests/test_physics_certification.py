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
    efforts_source: str = "physx",
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
        "efforts_source": efforts_source,
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
    assert report["metrics"]["server_target_source_ratio"] == 1.0
    assert report["metrics"]["ee_velocity_coverage"] == 1.0
    assert report["metrics"]["ee_acceleration_coverage"] == 1.0
    assert report["metrics"]["target_velocity_coverage"] == 1.0
    assert report["metrics"]["strict_runtime_patch_health"] is True


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


def test_physics_certification_tracks_llm_physics_override():
    """When local_framework already overrode task_success due to zero displacement,
    certification should record the LLM-physics contradiction in metrics."""
    frames = [
        _frame(position=[0.1, 0.0, 0.2]),
        _frame(position=[0.1, 0.0, 0.2]),
    ]
    meta = _episode_meta()
    meta["task_success"] = False  # Already overridden by physics
    meta["task_success_physics_override"] = {
        "original_success": True,
        "override_reason": "zero_object_displacement",
        "displacement_m": 0.0,
        "threshold_m": 0.01,
        "llm_assessment": True,
        "previous_source": "canonical_geometric_physics",
    }
    meta["goal_region_verification"] = {
        "grasp_detected": True,
        "object_lifted_5cm": False,
        "placed_in_goal": False,
        "stable_at_end": True,
        "gripper_released": True,
        "displacement_m": 0.0,
    }
    task = {"task_type": "pick_place", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, meta, task, mode="strict")
    metrics = report.get("metrics", {})
    assert metrics.get("llm_physics_contradiction") is True
    assert metrics.get("llm_physics_override_reason") == "zero_object_displacement"
    # task_success is False, so TASK_SUCCESS_CONTRADICTION should NOT fire
    assert "TASK_SUCCESS_CONTRADICTION" not in report.get("critical_failures", [])


def test_physics_certification_fails_when_strict_runtime_precheck_failed():
    frames = [
        _frame(position=[0.1, 0.0, 0.2]),
        _frame(position=[0.11, 0.0, 0.2]),
    ]
    meta = _episode_meta()
    meta["strict_runtime_patch_health"] = False
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}

    report = run_episode_certification(frames, meta, task, mode="strict")

    assert report["passed"] is False
    assert "STRICT_RUNTIME_PRECHECK_FAILED" in report["critical_failures"]
    assert report["dataset_tier"] == "raw_preserved"


def test_physics_certification_accepts_ee_velocity_alias_fields():
    frames = [
        _frame(position=[0.1, 0.0, 0.2]),
        _frame(position=[0.11, 0.0, 0.2]),
    ]
    for frame in frames:
        frame.pop("ee_vel", None)
        frame.pop("ee_acc", None)
        frame["ee_velocity"] = [0.0, 0.0, 0.0]
        frame["ee_acceleration"] = [0.0, 0.0, 0.0]

    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["metrics"]["ee_velocity_coverage"] == 1.0
    assert report["metrics"]["ee_acceleration_coverage"] == 1.0


def test_physics_certification_resolves_variation_target_alias():
    frames = [
        _frame(position=[0.1, 0.0, 0.2], effort_value=0.1),
        _frame(position=[0.11, 0.0, 0.2], effort_value=0.2),
        _frame(position=[0.12, 0.0, 0.2], effort_value=0.3),
    ]
    meta = _episode_meta()
    meta["target_object"] = "variation_toaster"
    task = {"task_type": "inspect", "target_object": "variation_toaster"}

    report = run_episode_certification(frames, meta, task, mode="strict")

    assert report["passed"] is True
    assert report["metrics"]["target_schema_completeness"] == 1.0
    assert report["metrics"]["server_target_source_ratio"] == 1.0


def test_physics_certification_fails_static_target_motion_task():
    frames = [
        _frame(position=[0.1, 0.0, 0.2], closed=True),
        _frame(position=[0.1, 0.0, 0.2], closed=True),
        _frame(position=[0.1, 0.0, 0.2], closed=False),
    ]
    meta = _episode_meta()
    meta["task_success"] = True
    meta["goal_region_verification"] = {
        "grasp_detected": True,
        "object_lifted_5cm": True,
        "placed_in_goal": True,
        "stable_at_end": True,
        "gripper_released": True,
    }
    task = {"task_type": "pick_place", "target_object": "lightwheel_kitchen_obj_Toaster003"}

    report = run_episode_certification(frames, meta, task, mode="strict")

    assert report["passed"] is False
    assert "EE_TARGET_GEOMETRY_IMPLAUSIBLE" in report["critical_failures"]
    assert report["dataset_tier"] == "raw_preserved"


def test_phase_b_env_does_not_skip_motion_gate_when_kinematic(monkeypatch):
    # Strict mode remains fail-closed even when kinematic env flags are present.
    monkeypatch.setenv("GENIESIM_REQUIRE_DYNAMIC_TOGGLE", "1")
    monkeypatch.setenv("GENIESIM_KEEP_OBJECTS_KINEMATIC", "1")

    frames = [
        _frame(position=[0.1, 0.0, 0.2], closed=True),
        _frame(position=[0.1, 0.0, 0.2], closed=True),
        _frame(position=[0.1, 0.0, 0.2], closed=False),
    ]
    task = {"task_type": "pick_place", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert "EE_TARGET_GEOMETRY_IMPLAUSIBLE" in report["critical_failures"]


def test_motion_gate_fires_when_not_kinematic(monkeypatch):
    # Without kinematic mode, zero displacement on a pick_place task must fail.
    monkeypatch.delenv("GENIESIM_KEEP_OBJECTS_KINEMATIC", raising=False)
    monkeypatch.delenv("GENIESIM_REQUIRE_DYNAMIC_TOGGLE", raising=False)

    frames = [
        _frame(position=[0.1, 0.0, 0.2], closed=True),
        _frame(position=[0.1, 0.0, 0.2], closed=True),
        _frame(position=[0.1, 0.0, 0.2], closed=False),
    ]
    task = {"task_type": "pick_place", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["passed"] is False
    assert "EE_TARGET_GEOMETRY_IMPLAUSIBLE" in report["critical_failures"]


def test_phase_b_env_does_not_skip_contact_gate(monkeypatch):
    # Phase B grasp-toggle env vars must not accept zero contacts in strict mode.
    monkeypatch.setenv("GENIESIM_REQUIRE_DYNAMIC_TOGGLE", "1")
    monkeypatch.setenv("GENIESIM_KEEP_OBJECTS_KINEMATIC", "1")

    frames = [
        _frame(position=[0.1, 0.0, 0.2], closed=True),
        _frame(position=[0.11, 0.0, 0.2], closed=True),
        _frame(position=[0.12, 0.0, 0.2], closed=False),
    ]
    for frame in frames:
        frame.pop("collision_contacts", None)

    task = {"task_type": "pick_place", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["passed"] is False
    assert "CONTACT_PLACEHOLDER_OR_EMPTY" in report["critical_failures"]


def test_phase_b_env_does_not_skip_kinematic_pose_gate(monkeypatch):
    # Phase B grasp-toggle env vars must not relax kinematic pose gating in strict mode.
    monkeypatch.setenv("GENIESIM_REQUIRE_DYNAMIC_TOGGLE", "1")
    monkeypatch.setenv("GENIESIM_KEEP_OBJECTS_KINEMATIC", "1")

    frames = [
        _frame(source="kinematic_ee_offset_blocked", effort_value=0.1),
        _frame(source="kinematic_ee_offset_blocked", effort_value=0.2),
    ]
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["passed"] is False
    assert "KINEMATIC_OBJECT_POSE_USED" in report["critical_failures"]


def test_phase_b_env_does_not_skip_scene_state_server_backing_gate(monkeypatch):
    # Phase B grasp-toggle env vars must not relax server-backed ratio gating in strict mode.
    monkeypatch.setenv("GENIESIM_REQUIRE_DYNAMIC_TOGGLE", "1")
    monkeypatch.setenv("GENIESIM_KEEP_OBJECTS_KINEMATIC", "1")

    frames = [
        _frame(scene_provenance="synthetic_fallback", effort_value=0.1),
        _frame(scene_provenance="synthetic_fallback", effort_value=0.2),
    ]
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["passed"] is False
    assert "SCENE_STATE_NOT_SERVER_BACKED" in report["critical_failures"]


def test_physics_certification_requires_camera_when_flagged():
    frames = [
        _frame(position=[0.1, 0.0, 0.2]),
        _frame(position=[0.11, 0.0, 0.2]),
    ]
    meta = _episode_meta()
    meta["camera_required"] = True
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}

    report = run_episode_certification(frames, meta, task, mode="strict")

    assert report["passed"] is False
    assert "CAMERA_PLACEHOLDER_PRESENT" in report["critical_failures"]


def test_physics_certification_flags_contact_placeholder_for_manipulation() -> None:
    frames = [
        _frame(position=[0.1, 0.0, 0.2], closed=True),
        _frame(position=[0.11, 0.0, 0.2], closed=True),
    ]
    for frame in frames:
        frame["collision_contacts"] = [
            {
                "body_a": "",
                "body_b": "",
                "force_N": 0.0,
                "penetration_depth": 0.0,
            }
        ]
    task = {"task_type": "pick_place", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["passed"] is False
    assert "CONTACT_PLACEHOLDER_OR_EMPTY" in report["critical_failures"]


def test_physics_certification_flags_target_schema_incomplete_when_mass_missing() -> None:
    frames = [
        _frame(position=[0.1, 0.0, 0.2], effort_value=0.1),
        _frame(position=[0.11, 0.0, 0.2], effort_value=0.2),
    ]
    for frame in frames:
        pose = frame["object_poses"]["lightwheel_kitchen_obj_Toaster003"]
        pose.pop("linear_velocity", None)

    meta = _episode_meta()
    meta["object_metadata"] = {"lightwheel_kitchen_obj_Toaster003": {}}
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, meta, task, mode="strict")

    assert report["passed"] is False
    assert "TARGET_SCHEMA_INCOMPLETE" in report["critical_failures"]


def test_physics_certification_flags_channel_incomplete_on_missing_ee_velocity() -> None:
    frames = [
        _frame(position=[0.1, 0.0, 0.2], effort_value=0.1),
        _frame(position=[0.11, 0.0, 0.2], effort_value=0.1),
        _frame(position=[0.12, 0.0, 0.2], effort_value=0.1),
    ]
    for frame in frames:
        frame.pop("ee_vel", None)
        frame.pop("ee_acc", None)

    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["passed"] is False
    assert "CHANNEL_INCOMPLETE" in report["critical_failures"]


def test_physics_certification_flags_scene_state_not_server_backed() -> None:
    frames = [
        _frame(scene_provenance="synthetic_fallback", source="synthetic_fallback"),
        _frame(scene_provenance="synthetic_fallback", source="synthetic_fallback"),
    ]
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["passed"] is False
    assert "SCENE_STATE_NOT_SERVER_BACKED" in report["critical_failures"]


def test_physics_certification_flags_snapback_or_teleport() -> None:
    frames = [
        _frame(position=[0.1, 0.0, 0.2]),
        _frame(position=[0.35, 0.0, 0.2]),
        _frame(position=[0.1, 0.0, 0.2]),
    ]
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["passed"] is False
    assert "SNAPBACK_OR_TELEPORT_DETECTED" in report["critical_failures"]


def test_physics_certification_warns_on_any_majority_synthesized_channel() -> None:
    frames = [
        _frame(position=[0.1, 0.0, 0.2], effort_value=0.1),
        _frame(position=[0.11, 0.0, 0.2], effort_value=0.2),
        _frame(position=[0.12, 0.0, 0.2], effort_value=0.3),
    ]
    target = "lightwheel_kitchen_obj_Toaster003"
    for idx, frame in enumerate(frames):
        frame["efforts_source"] = "physx"
        frame["collision_contacts"][0]["provenance"] = "physx_contact_report"
        frame["object_poses"][target]["source"] = "physx_server"
        if idx < 2:
            frame["observation"]["robot_state"]["joint_velocities_source"] = "finite_difference"

    task = {"task_type": "inspect", "target_object": target}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["passed"] is True
    assert report["warnings"] == ["joint_velocities 67% synthesized"]
    assert report["synthesis_provenance"]["joint_velocities"]["synthesized_fraction"] == 0.6667
    assert report["metrics"]["synthesis_provenance"] == report["synthesis_provenance"]


def test_physics_certification_fails_when_efforts_not_real_physx() -> None:
    frames = [
        _frame(position=[0.1, 0.0, 0.2], effort_value=0.1),
        _frame(position=[0.11, 0.0, 0.2], effort_value=0.2),
        _frame(position=[0.12, 0.0, 0.2], effort_value=0.3),
    ]
    for frame in frames:
        frame["efforts_source"] = "estimated_inverse_dynamics"
        frame["collision_contacts"][0]["provenance"] = "physx_contact_report"
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["passed"] is False
    assert "CHANNEL_INCOMPLETE" in report["critical_failures"]


def test_physics_certification_requires_camera_checks_when_payload_present() -> None:
    frames = [_frame(position=[0.1, 0.0, 0.2]), _frame(position=[0.11, 0.0, 0.2])]
    for frame in frames:
        frame["observation"]["camera_frames"] = {
            "wrist": {
                "width": 2,
                "height": 2,
                "rgb": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # placeholder-like
                "depth": [[float("inf"), float("inf")], [float("inf"), float("inf")]],
            }
        }
    meta = _episode_meta()
    meta["modality_profile"] = "no_rgb"
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}

    report = run_episode_certification(frames, meta, task, mode="strict")
    assert report["passed"] is False
    assert "CAMERA_PLACEHOLDER_PRESENT" in report["critical_failures"]
    assert report["metrics"]["camera_required"] is True


def test_physics_certification_flags_unresolved_camera_paths() -> None:
    frames = [_frame(position=[0.1, 0.0, 0.2]), _frame(position=[0.11, 0.0, 0.2])]
    for frame in frames:
        frame["observation"]["camera_frames"] = {
            "wrist": {
                "width": 640,
                "height": 480,
                "rgb": "missing_rgb.npy",
                "depth": "missing_depth.npy",
            }
        }
    meta = _episode_meta()
    meta["modality_profile"] = "no_rgb"
    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}

    report = run_episode_certification(frames, meta, task, mode="strict")
    assert report["passed"] is False
    assert "CAMERA_PLACEHOLDER_PRESENT" in report["critical_failures"]


def test_physics_certification_joint_velocity_provenance_infers_real_when_server_backed() -> None:
    frames = [
        _frame(position=[0.1, 0.0, 0.2], effort_value=0.1),
        _frame(position=[0.11, 0.0, 0.2], effort_value=0.2),
    ]
    for frame in frames:
        frame["efforts_source"] = "physx"
        frame["collision_contacts"][0]["provenance"] = "physx_contact_report"

    task = {"task_type": "inspect", "target_object": "lightwheel_kitchen_obj_Toaster003"}
    report = run_episode_certification(frames, _episode_meta(), task, mode="strict")

    assert report["passed"] is True
    assert report["synthesis_provenance"]["joint_velocities"]["real_fraction"] == 1.0
    assert report["synthesis_provenance"]["joint_velocities"]["synthesized_fraction"] == 0.0
    assert "joint_velocities 100% synthesized" not in report["warnings"]


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
