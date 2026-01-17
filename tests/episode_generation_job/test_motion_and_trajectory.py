import numpy as np
import pytest


@pytest.mark.unit
def test_motion_plan_recalculates_timing(load_job_module) -> None:
    motion_planner = load_job_module("episode_generation", "motion_planner.py")

    waypoints = [
        motion_planner.Waypoint(
            position=np.array([0.0, 0.0, 0.5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            duration_to_next=0.4,
        ),
        motion_planner.Waypoint(
            position=np.array([0.1, 0.1, 0.6]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            duration_to_next=0.6,
        ),
    ]

    plan = motion_planner.MotionPlan(
        plan_id="plan_01",
        task_name="move",
        task_description="Move to target",
        waypoints=waypoints,
    )

    assert plan.total_duration == pytest.approx(0.4)
    assert plan.waypoints[0].timestamp == pytest.approx(0.0)
    assert plan.waypoints[1].timestamp == pytest.approx(0.4)


@pytest.mark.unit
def test_trajectory_solver_generates_states(monkeypatch, load_job_module) -> None:
    motion_planner = load_job_module("episode_generation", "motion_planner.py")
    trajectory_solver = load_job_module("episode_generation", "trajectory_solver.py")

    waypoints = [
        motion_planner.Waypoint(
            position=np.array([0.0, 0.0, 0.5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            duration_to_next=0.5,
            gripper_aperture=0.5,
            phase=motion_planner.MotionPhase.APPROACH,
        ),
        motion_planner.Waypoint(
            position=np.array([0.1, 0.0, 0.6]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            duration_to_next=0.5,
            gripper_aperture=0.2,
            phase=motion_planner.MotionPhase.PLACE,
        ),
    ]

    plan = motion_planner.MotionPlan(
        plan_id="plan_traj",
        task_name="move",
        task_description="Move",
        waypoints=waypoints,
    )

    robot_config = trajectory_solver.ROBOT_CONFIGS["franka"]
    joint_targets = [
        robot_config.default_joint_positions.copy(),
        robot_config.default_joint_positions.copy() + 0.1,
    ]

    def fake_solve_waypoint_ik(self, _waypoints):
        return joint_targets

    monkeypatch.setattr(trajectory_solver.TrajectorySolver, "_solve_waypoint_ik", fake_solve_waypoint_ik)

    solver = trajectory_solver.TrajectorySolver(robot_type="franka", fps=10.0, verbose=False)
    trajectory = solver.solve(plan)

    assert trajectory.num_frames >= 2
    assert trajectory.states[0].frame_idx == 0
    assert trajectory.states[-1].frame_idx == trajectory.num_frames - 1
    assert trajectory.total_duration == pytest.approx(plan.total_duration)
