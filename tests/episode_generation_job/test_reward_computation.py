from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.unit
def test_reward_computation_outputs_in_range(load_job_module) -> None:
    reward_module = load_job_module("episode_generation", "reward_computation.py")
    trajectory_solver = load_job_module("episode_generation", "trajectory_solver.py")
    motion_planner = load_job_module("episode_generation", "motion_planner.py")

    robot_config = trajectory_solver.ROBOT_CONFIGS["franka"]
    states = [
        trajectory_solver.JointState(
            frame_idx=0,
            timestamp=0.0,
            joint_positions=robot_config.default_joint_positions.copy(),
            gripper_position=0.04,
            ee_position=np.array([0.0, 0.0, 0.5]),
            phase=motion_planner.MotionPhase.APPROACH,
        ),
        trajectory_solver.JointState(
            frame_idx=1,
            timestamp=0.5,
            joint_positions=robot_config.default_joint_positions.copy() + 0.05,
            gripper_position=0.02,
            ee_position=np.array([0.2, 0.0, 0.6]),
            phase=motion_planner.MotionPhase.PLACE,
        ),
        trajectory_solver.JointState(
            frame_idx=2,
            timestamp=1.0,
            joint_positions=robot_config.default_joint_positions.copy() + 0.1,
            gripper_position=0.03,
            ee_position=np.array([0.25, 0.05, 0.6]),
            phase=motion_planner.MotionPhase.RELEASE,
        ),
    ]

    trajectory = trajectory_solver.JointTrajectory(
        trajectory_id="traj_reward",
        robot_type="franka",
        robot_config=robot_config,
        states=states,
        fps=30.0,
        total_duration=1.0,
    )

    motion_plan = SimpleNamespace(
        target_position=np.array([0.2, 0.0, 0.6]),
        place_position=np.array([0.25, 0.05, 0.6]),
        total_duration=1.0,
    )

    computer = reward_module.RewardComputer(verbose=False)
    total_reward, components = computer.compute_episode_reward(
        trajectory=trajectory,
        motion_plan=motion_plan,
        validation_result=None,
        scene_objects=None,
    )

    assert 0.0 <= total_reward <= 1.0
    assert 0.0 <= components.total() <= 1.0
    assert components.to_dict()["placement_success"] in {0.0, 1.0, 0.5}


@pytest.mark.unit
def test_reward_config_from_environment(monkeypatch, load_job_module) -> None:
    reward_module = load_job_module("episode_generation", "reward_computation.py")

    monkeypatch.setenv("REWARD_PLACEMENT_THRESHOLD", "0.08")
    monkeypatch.setenv("REWARD_SCALE", "1.2")

    config = reward_module.RewardConfig.from_environment()
    assert config.placement_accuracy_threshold == 0.08
    assert config.reward_scale == 1.2
