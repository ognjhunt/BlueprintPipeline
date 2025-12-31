#!/usr/bin/env python3
"""
LeRobot Format Exporter for Episode Generation.

Exports generated robot episodes to LeRobot v2.0 format for direct training.
Compatible with the LeRobot training pipeline and HuggingFace datasets.

LeRobot Dataset Structure (v2.0):
    dataset/
    ├── meta/
    │   ├── info.json           # Dataset metadata
    │   ├── stats.json          # Statistics per feature
    │   ├── tasks.jsonl         # Task descriptions
    │   └── episodes.jsonl      # Episode metadata
    ├── data/
    │   ├── chunk-000/
    │   │   ├── episode_000000.parquet
    │   │   ├── episode_000001.parquet
    │   │   └── ...
    │   └── ...
    └── videos/                  # Optional video data
        ├── chunk-000/
        │   ├── observation.images.camera/
        │   │   ├── episode_000000.mp4
        │   │   └── ...
        │   └── ...
        └── ...

See: https://github.com/huggingface/lerobot
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trajectory_solver import JointTrajectory, JointState, RobotConfig, ROBOT_CONFIGS

# Try to import pyarrow for Parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False
    pa = None
    pq = None


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class LeRobotEpisode:
    """A single episode in LeRobot format."""

    episode_index: int
    task_index: int
    task_description: str

    # Trajectory data
    trajectory: JointTrajectory

    # Scene context
    scene_id: str = ""
    variation_index: int = 0

    # Optional image data paths
    camera_video_path: Optional[Path] = None

    # Metadata
    success: bool = True
    total_reward: float = 1.0


@dataclass
class LeRobotDatasetConfig:
    """Configuration for LeRobot dataset export."""

    # Dataset identification
    dataset_name: str
    robot_type: str = "franka"

    # Data configuration
    fps: float = 30.0
    chunk_size: int = 1000  # Episodes per chunk

    # Feature configuration
    state_dim: int = 7  # Joint positions
    action_dim: int = 8  # Joints + gripper

    # Image configuration (optional)
    include_images: bool = False
    image_resolution: Tuple[int, int] = (640, 480)

    # Output paths
    output_dir: Path = Path("./lerobot_dataset")


# =============================================================================
# LeRobot Exporter
# =============================================================================


class LeRobotExporter:
    """
    Exports robot episodes to LeRobot v2.0 format.

    LeRobot format is the standard for robot learning datasets, compatible with:
    - HuggingFace Datasets
    - LeRobot training pipeline
    - Various policy architectures (ACT, Diffusion Policy, etc.)

    Usage:
        exporter = LeRobotExporter(config)
        exporter.add_episode(trajectory, task_description)
        exporter.add_episode(trajectory2, task_description2)
        exporter.finalize()
    """

    def __init__(self, config: LeRobotDatasetConfig, verbose: bool = True):
        """
        Initialize exporter.

        Args:
            config: Dataset configuration
            verbose: Print progress
        """
        self.config = config
        self.verbose = verbose
        self.robot_config = ROBOT_CONFIGS.get(config.robot_type, ROBOT_CONFIGS["franka"])

        # Episode storage
        self.episodes: List[LeRobotEpisode] = []
        self.tasks: List[Dict[str, str]] = []
        self.task_to_index: Dict[str, int] = {}

        # Statistics tracking
        self.stats = {
            "observation.state": {"min": [], "max": [], "mean": [], "std": []},
            "action": {"min": [], "max": [], "mean": [], "std": []},
        }

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[LEROBOT-EXPORTER] [{level}] {msg}")

    def add_episode(
        self,
        trajectory: JointTrajectory,
        task_description: str,
        scene_id: str = "",
        variation_index: int = 0,
        success: bool = True,
    ) -> int:
        """
        Add an episode to the dataset.

        Args:
            trajectory: Joint trajectory for the episode
            task_description: Natural language task description
            scene_id: Scene identifier
            variation_index: Variation index within scene
            success: Whether episode was successful

        Returns:
            Episode index
        """
        # Get or create task index
        if task_description not in self.task_to_index:
            task_index = len(self.tasks)
            self.tasks.append({"task_index": task_index, "task": task_description})
            self.task_to_index[task_description] = task_index
        else:
            task_index = self.task_to_index[task_description]

        episode_index = len(self.episodes)

        episode = LeRobotEpisode(
            episode_index=episode_index,
            task_index=task_index,
            task_description=task_description,
            trajectory=trajectory,
            scene_id=scene_id,
            variation_index=variation_index,
            success=success,
        )

        self.episodes.append(episode)
        self.log(f"Added episode {episode_index}: {task_description[:50]}...")

        return episode_index

    def finalize(self) -> Path:
        """
        Write the complete dataset to disk.

        Returns:
            Path to the dataset directory
        """
        self.log("=" * 60)
        self.log("Finalizing LeRobot Dataset Export")
        self.log("=" * 60)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create directories
        meta_dir = output_dir / "meta"
        data_dir = output_dir / "data"
        meta_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)

        # Step 1: Write episode data
        self.log("Writing episode data...")
        self._write_episodes(data_dir)

        # Step 2: Calculate and write statistics
        self.log("Calculating statistics...")
        self._calculate_stats()
        self._write_stats(meta_dir)

        # Step 3: Write metadata
        self.log("Writing metadata...")
        self._write_info(meta_dir)
        self._write_tasks(meta_dir)
        self._write_episodes_meta(meta_dir)

        self.log("=" * 60)
        self.log(f"Dataset exported: {output_dir}")
        self.log(f"  Episodes: {len(self.episodes)}")
        self.log(f"  Tasks: {len(self.tasks)}")
        self.log("=" * 60)

        return output_dir

    def _write_episodes(self, data_dir: Path) -> None:
        """Write episode data to Parquet files."""

        chunk_idx = 0
        chunk_dir = data_dir / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(exist_ok=True)

        for episode in self.episodes:
            # Create episode data
            episode_data = self._trajectory_to_arrow_table(episode)

            # Write to Parquet
            episode_path = chunk_dir / f"episode_{episode.episode_index:06d}.parquet"

            if HAVE_PYARROW:
                pq.write_table(episode_data, episode_path)
            else:
                # Fallback: write as JSON
                self._write_episode_json(episode, episode_path.with_suffix(".json"))

            # Handle chunk rotation
            if (episode.episode_index + 1) % self.config.chunk_size == 0:
                chunk_idx += 1
                chunk_dir = data_dir / f"chunk-{chunk_idx:03d}"
                chunk_dir.mkdir(exist_ok=True)

    def _trajectory_to_arrow_table(self, episode: LeRobotEpisode) -> Union[Any, Dict]:
        """Convert trajectory to PyArrow table."""

        trajectory = episode.trajectory
        states = trajectory.states

        # Extract data arrays
        timestamps = [s.timestamp for s in states]
        frame_indices = list(range(len(states)))
        episode_indices = [episode.episode_index] * len(states)

        # State: joint positions
        state_data = [s.joint_positions.tolist() for s in states]

        # Action: joint positions + gripper (next frame's state as action)
        # Standard convention: action[t] leads to state[t+1]
        action_data = []
        for i, s in enumerate(states):
            if i < len(states) - 1:
                next_state = states[i + 1]
                action = list(next_state.joint_positions) + [next_state.gripper_position]
            else:
                # Last frame: repeat current
                action = list(s.joint_positions) + [s.gripper_position]
            action_data.append(action)

        # Index within episode
        index_in_episode = list(range(len(states)))

        if HAVE_PYARROW:
            # Create Arrow arrays
            table = pa.table({
                "timestamp": pa.array(timestamps, type=pa.float64()),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "episode_index": pa.array(episode_indices, type=pa.int64()),
                "index": pa.array(index_in_episode, type=pa.int64()),
                "task_index": pa.array([episode.task_index] * len(states), type=pa.int64()),
                "observation.state": pa.array(state_data, type=pa.list_(pa.float32())),
                "action": pa.array(action_data, type=pa.list_(pa.float32())),
                # Optional: gripper as separate field
                "observation.gripper_position": pa.array(
                    [s.gripper_position for s in states], type=pa.float32()
                ),
                # EE position (if available)
                "observation.ee_position": pa.array(
                    [s.ee_position.tolist() if s.ee_position is not None else [0, 0, 0] for s in states],
                    type=pa.list_(pa.float32()),
                ),
            })
            return table
        else:
            # Return as dict for JSON fallback
            return {
                "timestamps": timestamps,
                "frame_indices": frame_indices,
                "episode_index": episode.episode_index,
                "task_index": episode.task_index,
                "states": state_data,
                "actions": action_data,
                "gripper_positions": [s.gripper_position for s in states],
            }

    def _write_episode_json(self, episode: LeRobotEpisode, path: Path) -> None:
        """Fallback: write episode as JSON."""
        data = self._trajectory_to_arrow_table(episode)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _calculate_stats(self) -> None:
        """Calculate statistics for normalization."""

        all_states = []
        all_actions = []

        for episode in self.episodes:
            for i, state in enumerate(episode.trajectory.states):
                all_states.append(state.joint_positions)

                # Action
                if i < len(episode.trajectory.states) - 1:
                    next_state = episode.trajectory.states[i + 1]
                    action = np.append(next_state.joint_positions, next_state.gripper_position)
                else:
                    action = np.append(state.joint_positions, state.gripper_position)
                all_actions.append(action)

        all_states = np.array(all_states)
        all_actions = np.array(all_actions)

        self.stats["observation.state"] = {
            "min": all_states.min(axis=0).tolist(),
            "max": all_states.max(axis=0).tolist(),
            "mean": all_states.mean(axis=0).tolist(),
            "std": all_states.std(axis=0).tolist(),
        }

        self.stats["action"] = {
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
        }

    def _write_stats(self, meta_dir: Path) -> None:
        """Write statistics JSON."""
        stats_path = meta_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

    def _write_info(self, meta_dir: Path) -> None:
        """Write dataset info JSON."""
        info = {
            "codebase_version": "v2.0",
            "robot_type": self.config.robot_type,
            "fps": self.config.fps,
            "total_episodes": len(self.episodes),
            "total_frames": sum(ep.trajectory.num_frames for ep in self.episodes),
            "total_tasks": len(self.tasks),
            "total_chunks": (len(self.episodes) - 1) // self.config.chunk_size + 1,
            "chunks_size": self.config.chunk_size,
            "data_path": "data",
            "videos_path": "videos" if self.config.include_images else None,
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [self.robot_config.num_joints],
                    "names": self.robot_config.joint_names,
                },
                "observation.gripper_position": {
                    "dtype": "float32",
                    "shape": [1],
                },
                "observation.ee_position": {
                    "dtype": "float32",
                    "shape": [3],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [self.robot_config.num_joints + 1],  # + gripper
                    "names": self.robot_config.joint_names + ["gripper"],
                },
            },
            "splits": {
                "train": f"0:{len(self.episodes)}",
            },
            "created_at": datetime.utcnow().isoformat() + "Z",
            "generator": "BlueprintPipeline/episode-generation-job",
        }

        info_path = meta_dir / "info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def _write_tasks(self, meta_dir: Path) -> None:
        """Write tasks JSONL."""
        tasks_path = meta_dir / "tasks.jsonl"
        with open(tasks_path, "w") as f:
            for task in self.tasks:
                f.write(json.dumps(task) + "\n")

    def _write_episodes_meta(self, meta_dir: Path) -> None:
        """Write episodes metadata JSONL."""
        episodes_path = meta_dir / "episodes.jsonl"
        with open(episodes_path, "w") as f:
            for episode in self.episodes:
                meta = {
                    "episode_index": episode.episode_index,
                    "task_index": episode.task_index,
                    "task": episode.task_description,
                    "length": episode.trajectory.num_frames,
                    "scene_id": episode.scene_id,
                    "variation_index": episode.variation_index,
                    "success": episode.success,
                }
                f.write(json.dumps(meta) + "\n")


# =============================================================================
# Convenience Functions
# =============================================================================


def export_trajectories_to_lerobot(
    trajectories: List[Tuple[JointTrajectory, str]],
    dataset_name: str,
    output_dir: Path,
    robot_type: str = "franka",
    fps: float = 30.0,
) -> Path:
    """
    Convenience function to export trajectories to LeRobot format.

    Args:
        trajectories: List of (trajectory, task_description) tuples
        dataset_name: Name for the dataset
        output_dir: Output directory
        robot_type: Robot type
        fps: Frames per second

    Returns:
        Path to exported dataset
    """
    config = LeRobotDatasetConfig(
        dataset_name=dataset_name,
        robot_type=robot_type,
        fps=fps,
        output_dir=output_dir,
    )

    exporter = LeRobotExporter(config, verbose=True)

    for trajectory, task_desc in trajectories:
        exporter.add_episode(trajectory, task_desc)

    return exporter.finalize()


if __name__ == "__main__":
    from motion_planner import AIMotionPlanner
    from trajectory_solver import TrajectorySolver

    print("Testing LeRobot Export Pipeline")
    print("=" * 60)

    # Generate some test episodes
    planner = AIMotionPlanner(robot_type="franka", use_llm=False, verbose=False)
    solver = TrajectorySolver(robot_type="franka", fps=30.0, verbose=False)

    # Create exporter
    config = LeRobotDatasetConfig(
        dataset_name="test_pick_place",
        robot_type="franka",
        fps=30.0,
        output_dir=Path("/tmp/lerobot_test"),
    )
    exporter = LeRobotExporter(config, verbose=True)

    # Generate and add episodes
    tasks = [
        ("pick_cup", "Pick up the coffee cup from the counter", [0.5, 0.1, 0.85], [0.3, -0.2, 0.9]),
        ("pick_bowl", "Pick up the bowl and place on shelf", [0.4, 0.0, 0.82], [0.2, 0.3, 0.95]),
        ("pick_plate", "Pick up the plate from table", [0.6, -0.1, 0.80], [0.4, 0.2, 0.85]),
    ]

    for task_name, desc, target_pos, place_pos in tasks:
        plan = planner.plan_motion(
            task_name=task_name,
            task_description=desc,
            target_object={"id": task_name, "position": target_pos, "dimensions": [0.08, 0.08, 0.1]},
            place_position=place_pos,
        )
        trajectory = solver.solve(plan)
        exporter.add_episode(trajectory, desc)

    # Export
    dataset_path = exporter.finalize()

    # Verify output
    print("\nVerifying output...")
    for path in sorted(dataset_path.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(dataset_path)}")
