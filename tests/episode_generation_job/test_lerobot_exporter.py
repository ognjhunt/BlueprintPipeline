import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.unit
def test_lerobot_exporter_writes_metadata_and_data(load_job_module, tmp_path: Path) -> None:
    trajectory_solver = load_job_module("episode_generation", "trajectory_solver.py")
    lerobot_exporter = load_job_module("episode_generation", "lerobot_exporter.py")

    robot_config = trajectory_solver.ROBOT_CONFIGS["franka"]
    states = [
        trajectory_solver.JointState(
            frame_idx=0,
            timestamp=0.0,
            joint_positions=robot_config.default_joint_positions.copy(),
            joint_velocities=np.zeros(robot_config.num_joints),
            gripper_position=0.0,
            ee_position=np.array([0.4, 0.0, 0.7]),
            phase=trajectory_solver.MotionPhase.APPROACH,
        ),
        trajectory_solver.JointState(
            frame_idx=1,
            timestamp=1.0 / 30.0,
            joint_positions=robot_config.default_joint_positions.copy() + 0.01,
            joint_velocities=np.zeros(robot_config.num_joints),
            gripper_position=0.02,
            ee_position=np.array([0.45, 0.05, 0.72]),
            phase=trajectory_solver.MotionPhase.PLACE,
        ),
    ]

    trajectory = trajectory_solver.JointTrajectory(
        trajectory_id="traj_test",
        robot_type="franka",
        robot_config=robot_config,
        states=states,
        source_plan_id="plan_001",
        fps=30.0,
        total_duration=1.0 / 30.0,
    )

    config = lerobot_exporter.LeRobotDatasetConfig(
        dataset_name="unit_test_dataset",
        robot_type="franka",
        output_dir=tmp_path / "lerobot_dataset",
    )
    exporter = lerobot_exporter.LeRobotExporter(config, verbose=False)
    exporter.add_episode(trajectory, "Pick and place test")

    output_dir = exporter.finalize()

    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data"
    assert meta_dir.exists()
    assert data_dir.exists()
    assert (meta_dir / "info.json").exists()
    assert (meta_dir / "tasks.jsonl").exists()
    assert (meta_dir / "episodes.jsonl").exists()

    info = json.loads((meta_dir / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_tasks"] == 1
    assert info["data_pack"]["tier"] == "core"

    chunk_dir = data_dir / "chunk-000"
    assert chunk_dir.exists()

    if lerobot_exporter.HAVE_PYARROW:
        assert (chunk_dir / "episode_000000.parquet").exists()
    else:
        assert (chunk_dir / "episode_000000.json").exists()


@pytest.mark.unit
def test_lerobot_exporter_requires_complete_episodes(load_job_module, tmp_path: Path) -> None:
    trajectory_solver = load_job_module("episode_generation", "trajectory_solver.py")
    lerobot_exporter = load_job_module("episode_generation", "lerobot_exporter.py")

    robot_config = trajectory_solver.ROBOT_CONFIGS["franka"]
    states = [
        trajectory_solver.JointState(
            frame_idx=0,
            timestamp=0.0,
            joint_positions=robot_config.default_joint_positions.copy(),
            joint_velocities=np.zeros(robot_config.num_joints),
            gripper_position=0.0,
            ee_position=np.array([0.4, 0.0, 0.7]),
            phase=trajectory_solver.MotionPhase.APPROACH,
        ),
        trajectory_solver.JointState(
            frame_idx=1,
            timestamp=1.0 / 30.0,
            joint_positions=robot_config.default_joint_positions.copy() + 0.01,
            joint_velocities=np.zeros(robot_config.num_joints),
            gripper_position=0.02,
            ee_position=np.array([0.45, 0.05, 0.72]),
            phase=trajectory_solver.MotionPhase.PLACE,
        ),
    ]

    trajectory = trajectory_solver.JointTrajectory(
        trajectory_id="traj_test",
        robot_type="franka",
        robot_config=robot_config,
        states=states,
        source_plan_id="plan_001",
        fps=30.0,
        total_duration=1.0 / 30.0,
    )

    config = lerobot_exporter.LeRobotDatasetConfig(
        dataset_name="unit_test_dataset",
        robot_type="franka",
        output_dir=tmp_path / "lerobot_dataset",
        require_complete_episodes=True,
    )
    exporter = lerobot_exporter.LeRobotExporter(config, verbose=False)
    exporter.add_episode(trajectory, "Pick and place test")
    exporter.add_episode(trajectory, "")

    with pytest.raises(ValueError, match="Incomplete episodes detected"):
        exporter.finalize()
