import numpy as np
import pytest


@pytest.mark.unit
def test_within_joint_limits_detects_out_of_bounds(load_job_module):
    motion_planner = load_job_module("episode_generation", "motion_planner.py")
    planner = motion_planner.AIMotionPlanner(
        robot_type="franka",
        use_llm=False,
        use_curobo=False,
        verbose=False,
    )
    joints = planner.robot_config["default_joint_positions"].copy()

    assert planner._within_joint_limits(joints) is True

    joints[0] = planner.robot_config["joint_limits_upper"][0] + 0.01
    assert planner._within_joint_limits(joints) is False


@pytest.mark.unit
def test_validate_joint_trajectory_limits_reports_violation(load_job_module):
    motion_planner = load_job_module("episode_generation", "motion_planner.py")
    planner = motion_planner.AIMotionPlanner(
        robot_type="franka",
        use_llm=False,
        use_curobo=False,
        verbose=False,
    )
    base = planner.robot_config["default_joint_positions"].copy()
    trajectory = np.vstack([base, base])
    trajectory[1, 1] = planner.robot_config["joint_limits_lower"][1] - 0.05

    errors = planner._validate_joint_trajectory_limits(trajectory)

    assert "Joint trajectory violates configured joint limits" in errors
