#!/usr/bin/env python3
"""
LeRobot Format Exporter for Episode Generation.

Exports generated robot episodes to LeRobot v2.0 format for direct training.
Compatible with the LeRobot training pipeline and HuggingFace datasets.

ENHANCED (2025): Now supports full visual observations and ground-truth labels:
- RGB images/videos per camera
- Depth maps
- Segmentation masks (semantic + instance)
- 2D/3D bounding boxes
- Object poses
- Contact information
- Privileged state

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
    ├── videos/                  # Video data (RGB per camera)
    │   ├── chunk-000/
    │   │   ├── observation.images.wrist/
    │   │   │   ├── episode_000000.mp4
    │   │   │   └── ...
    │   │   ├── observation.images.overhead/
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── ground_truth/            # Ground-truth labels (Plus/Full packs)
        ├── chunk-000/
        │   ├── depth/
        │   ├── segmentation/
        │   ├── bboxes/
        │   ├── object_poses/
        │   └── contacts/
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

# Import reward computation
try:
    from reward_computation import RewardComputer, RewardConfig, compute_episode_reward
    HAVE_REWARD_COMPUTATION = True
except ImportError:
    HAVE_REWARD_COMPUTATION = False
    RewardComputer = None

# Try to import pyarrow for Parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False
    pa = None
    pq = None

# Try to import video writing libraries
try:
    import imageio
    HAVE_IMAGEIO = True
except ImportError:
    HAVE_IMAGEIO = False
    imageio = None

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False
    Image = None

# Sensor data capture (optional import)
try:
    from sensor_data_capture import (
        EpisodeSensorData,
        FrameSensorData,
        SensorDataConfig,
        DataPackTier,
    )
    from data_pack_config import DataPackConfig, get_data_pack_config
    HAVE_SENSOR_CAPTURE = True
except ImportError:
    HAVE_SENSOR_CAPTURE = False
    EpisodeSensorData = None
    DataPackConfig = None


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

    # Sensor data (enhanced - for visual observations)
    sensor_data: Optional[Any] = None  # EpisodeSensorData when available

    # Metadata
    success: bool = True
    total_reward: float = 0.0  # Computed by RewardComputer
    quality_score: float = 1.0
    is_mock: bool = False  # P0-5 FIX: Flag for mock sensor data

    # Reward breakdown (for interpretability)
    reward_components: Dict[str, float] = field(default_factory=dict)

    # Source motion plan (for reward computation)
    motion_plan: Optional[Any] = None
    validation_result: Optional[Any] = None

    # Ground-truth metadata
    object_metadata: Dict[str, Any] = field(default_factory=dict)


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

    # Image configuration (enhanced)
    include_images: bool = False
    image_resolution: Tuple[int, int] = (640, 480)
    camera_types: List[str] = field(default_factory=lambda: ["wrist"])
    video_codec: str = "h264"

    # Data pack configuration (Core/Plus/Full)
    data_pack_tier: str = "core"  # "core", "plus", "full"

    # Ground-truth configuration (Plus/Full packs)
    include_depth: bool = False
    include_segmentation: bool = False
    include_bboxes: bool = False
    include_object_poses: bool = False
    include_contacts: bool = False
    include_privileged_state: bool = False

    # Output paths
    output_dir: Path = Path("./lerobot_dataset")

    @classmethod
    def from_data_pack(
        cls,
        dataset_name: str,
        data_pack_tier: str,
        robot_type: str = "franka",
        num_cameras: int = 1,
        resolution: Tuple[int, int] = (640, 480),
        fps: float = 30.0,
        output_dir: Path = Path("./lerobot_dataset"),
    ) -> "LeRobotDatasetConfig":
        """Create config from data pack tier."""
        camera_types = ["wrist"]
        if num_cameras >= 2:
            camera_types.append("overhead")
        if num_cameras >= 3:
            camera_types.append("side")
        if num_cameras >= 4:
            camera_types.append("front")

        config = cls(
            dataset_name=dataset_name,
            robot_type=robot_type,
            fps=fps,
            image_resolution=resolution,
            camera_types=camera_types[:num_cameras],
            data_pack_tier=data_pack_tier,
            output_dir=output_dir,
        )

        # Configure based on tier
        tier = data_pack_tier.lower()

        if tier in ["core", "plus", "full"]:
            config.include_images = True

        if tier in ["plus", "full"]:
            config.include_depth = True
            config.include_segmentation = True
            config.include_bboxes = True

        if tier == "full":
            config.include_object_poses = True
            config.include_contacts = True
            config.include_privileged_state = True

        return config


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

    def _get_data_source_info(self) -> Dict[str, Any]:
        """
        Determine the data source and quality information.

        This checks:
        - Whether Isaac Sim was used for physics
        - Whether sensor data is from simulation or mock
        - Whether validation was physics-based or heuristic

        Returns:
            Dict with physics_validated, sensor_source, validation_type, etc.
        """
        import os

        warnings = []

        # Check for Isaac Sim availability
        try:
            from isaac_sim_integration import is_isaac_sim_available, is_physx_available
            isaac_sim_available = is_isaac_sim_available()
            physx_available = is_physx_available()
        except ImportError:
            isaac_sim_available = False
            physx_available = False

        # Determine sensor source from episodes
        sensor_source = "unknown"
        if self.episodes:
            sample_episode = self.episodes[0]
            if sample_episode.sensor_data is not None:
                # Check if sensor data has source annotation
                if hasattr(sample_episode.sensor_data, 'frames') and sample_episode.sensor_data.frames:
                    frame = sample_episode.sensor_data.frames[0]
                    if hasattr(frame, 'privileged_state') and frame.privileged_state:
                        source = frame.privileged_state.get('data_source', 'unknown')
                        if source == 'simulation':
                            sensor_source = "isaac_sim_replicator"
                        elif source == 'input_fallback':
                            sensor_source = "mock_random"
                            warnings.append("Sensor data from mock capture - random noise images")
                        else:
                            sensor_source = source
                    else:
                        # Check if it's a mock sensor class
                        sensor_class = sample_episode.sensor_data.__class__.__name__
                        if 'Mock' in sensor_class:
                            sensor_source = "mock_random"
                            warnings.append("Sensor data from mock capture - random noise images")
                        else:
                            sensor_source = "simulation"
            else:
                sensor_source = "none"
                if self.config.include_images:
                    warnings.append("No sensor data captured despite images being requested")

        # Determine validation type from episodes
        validation_type = "unknown"
        physics_validated = False
        if self.episodes:
            sample_episode = self.episodes[0]
            if sample_episode.validation_result is not None:
                val_result = sample_episode.validation_result
                if hasattr(val_result, 'validation_type'):
                    validation_type = val_result.validation_type
                elif hasattr(val_result, 'physics_checked') and val_result.physics_checked:
                    validation_type = "physx"
                else:
                    validation_type = "heuristic"
                    warnings.append("Validation was heuristic-based, not physics-verified")

                physics_validated = validation_type == "physx"
            else:
                validation_type = "none"
                warnings.append("No validation performed on episodes")

        # Check environment indicators
        is_production = (
            os.getenv("KUBERNETES_SERVICE_HOST") is not None or
            os.getenv("K_SERVICE") is not None or
            os.path.exists("/.dockerenv")
        )

        if not isaac_sim_available and is_production:
            warnings.append("Running in production without Isaac Sim")

        if not physx_available:
            warnings.append("PhysX not available - physics not validated")

        return {
            "physics_validated": physics_validated,
            "sensor_source": sensor_source,
            "validation_type": validation_type,
            "isaac_sim_used": isaac_sim_available,
            "physx_available": physx_available,
            "warnings": warnings,
            "production_mode": is_production,
        }

    def add_episode(
        self,
        trajectory: JointTrajectory,
        task_description: str,
        scene_id: str = "",
        variation_index: int = 0,
        success: bool = True,
        quality_score: float = 1.0,
        sensor_data: Optional[Any] = None,
        object_metadata: Optional[Dict[str, Any]] = None,
        motion_plan: Optional[Any] = None,
        validation_result: Optional[Any] = None,
    ) -> int:
        """
        Add an episode to the dataset.

        Args:
            trajectory: Joint trajectory for the episode
            task_description: Natural language task description
            scene_id: Scene identifier
            variation_index: Variation index within scene
            success: Whether episode was successful
            quality_score: Quality score from validation (0.0-1.0)
            sensor_data: EpisodeSensorData with visual observations (optional)
            object_metadata: Object metadata for ground-truth (optional)
            motion_plan: Original motion plan for reward computation (optional)
            validation_result: Validation result with physics data (optional)

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

        # Compute reward using RewardComputer if available
        total_reward = 0.0
        reward_components = {}

        if HAVE_REWARD_COMPUTATION and motion_plan is not None:
            try:
                reward_computer = RewardComputer(verbose=False)
                total_reward, components = reward_computer.compute_episode_reward(
                    trajectory=trajectory,
                    motion_plan=motion_plan,
                    validation_result=validation_result,
                )
                reward_components = components.to_dict()
            except Exception as e:
                self.log(f"Reward computation failed: {e}", "WARNING")
                # Fallback to heuristic
                total_reward = 0.7 if success else 0.0
        else:
            # Fallback: use success + quality_score
            total_reward = quality_score * (1.0 if success else 0.3)
            reward_components = {"fallback": True, "success": float(success), "quality": quality_score}

        # P0-5 FIX: Extract is_mock flag from sensor_data
        is_mock = False
        if sensor_data is not None:
            # Check if sensor_data has frames with is_mock flag
            if hasattr(sensor_data, 'frames') and len(sensor_data.frames) > 0:
                first_frame = sensor_data.frames[0]
                if hasattr(first_frame, 'is_mock'):
                    is_mock = first_frame.is_mock
                # Also check state dict if available
                elif hasattr(first_frame, 'state') and isinstance(first_frame.state, dict):
                    is_mock = first_frame.state.get('is_mock', False)

        episode = LeRobotEpisode(
            episode_index=episode_index,
            task_index=task_index,
            task_description=task_description,
            trajectory=trajectory,
            scene_id=scene_id,
            variation_index=variation_index,
            success=success,
            total_reward=total_reward,
            quality_score=quality_score,
            is_mock=is_mock,
            reward_components=reward_components,
            sensor_data=sensor_data,
            motion_plan=motion_plan,
            validation_result=validation_result,
            object_metadata=object_metadata or {},
        )

        self.episodes.append(episode)

        # Log with visual observation info if present
        visual_info = ""
        if sensor_data is not None:
            num_frames = sensor_data.num_frames if hasattr(sensor_data, 'num_frames') else 0
            cameras = sensor_data.camera_ids if hasattr(sensor_data, 'camera_ids') else []
            visual_info = f" [visual: {num_frames} frames, {len(cameras)} cameras]"

        reward_info = f" [reward: {total_reward:.2f}]"

        self.log(f"Added episode {episode_index}: {task_description[:50]}...{visual_info}{reward_info}")

        return episode_index

    def finalize(self) -> Path:
        """
        Write the complete dataset to disk.

        Returns:
            Path to the dataset directory
        """
        self.log("=" * 60)
        self.log("Finalizing LeRobot Dataset Export")
        self.log(f"Data Pack: {self.config.data_pack_tier}")
        self.log("=" * 60)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create directories
        meta_dir = output_dir / "meta"
        data_dir = output_dir / "data"
        meta_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)

        # Step 1: Write episode data (joint-space trajectories)
        self.log("Writing episode data...")
        self._write_episodes(data_dir)

        # Step 2: Write visual observations (RGB videos per camera)
        if self.config.include_images:
            self.log("Writing visual observations...")
            self._write_visual_observations(output_dir)

        # Step 3: Write ground-truth labels (Plus/Full packs)
        if any([
            self.config.include_depth,
            self.config.include_segmentation,
            self.config.include_bboxes,
            self.config.include_object_poses,
            self.config.include_contacts,
        ]):
            self.log("Writing ground-truth labels...")
            self._write_ground_truth(output_dir)

        # Step 4: Calculate and write statistics
        self.log("Calculating statistics...")
        self._calculate_stats()
        self._write_stats(meta_dir)

        # Step 5: Write metadata
        self.log("Writing metadata...")
        self._write_info(meta_dir)
        self._write_tasks(meta_dir)
        self._write_episodes_meta(meta_dir)

        # Summary
        self.log("=" * 60)
        self.log(f"Dataset exported: {output_dir}")
        self.log(f"  Data Pack: {self.config.data_pack_tier}")
        self.log(f"  Episodes: {len(self.episodes)}")
        self.log(f"  Tasks: {len(self.tasks)}")

        # Count episodes with visual data
        visual_episodes = sum(1 for e in self.episodes if e.sensor_data is not None)
        if visual_episodes > 0:
            self.log(f"  Episodes with visual obs: {visual_episodes}")
            self.log(f"  Cameras: {self.config.camera_types}")

        # Report ground-truth streams
        gt_streams = []
        if self.config.include_depth:
            gt_streams.append("depth")
        if self.config.include_segmentation:
            gt_streams.append("segmentation")
        if self.config.include_bboxes:
            gt_streams.append("bboxes")
        if self.config.include_object_poses:
            gt_streams.append("object_poses")
        if self.config.include_contacts:
            gt_streams.append("contacts")
        if self.config.include_privileged_state:
            gt_streams.append("privileged_state")

        if gt_streams:
            self.log(f"  Ground-truth streams: {', '.join(gt_streams)}")

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

    def _validate_lerobot_schema(
        self,
        timestamps: List[float],
        state_data: List[List[float]],
        action_data: List[List[float]],
        episode_index: int,
    ) -> None:
        """
        Validate LeRobot v2.0 schema compliance.

        Validates:
        - observation.state dimensions match robot DOF
        - action dimensions match robot DOF + gripper
        - timestamps are monotonically increasing
        - P1-2 FIX: Action continuity (no large jumps)
        - P1-2 FIX: Joint velocity/acceleration limits
        - P1-2 FIX: Gripper position range (0.0-1.0)

        Raises:
            ValueError: If validation fails
        """
        # Validate state dimensions
        expected_state_dim = self.robot_config.num_dof
        for i, state in enumerate(state_data):
            if len(state) != expected_state_dim:
                raise ValueError(
                    f"Episode {episode_index}, frame {i}: observation.state dimension mismatch. "
                    f"Expected {expected_state_dim}, got {len(state)}"
                )

        # Validate action dimensions (DOF + gripper)
        expected_action_dim = self.robot_config.num_dof + 1
        for i, action in enumerate(action_data):
            if len(action) != expected_action_dim:
                raise ValueError(
                    f"Episode {episode_index}, frame {i}: action dimension mismatch. "
                    f"Expected {expected_action_dim} (DOF + gripper), got {len(action)}"
                )

        # Validate timestamps are monotonically increasing
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i - 1]:
                raise ValueError(
                    f"Episode {episode_index}, frame {i}: timestamps not monotonically increasing. "
                    f"Frame {i-1}: {timestamps[i-1]:.6f}, Frame {i}: {timestamps[i]:.6f}"
                )

        # Validate timestamps are non-negative
        if any(t < 0 for t in timestamps):
            raise ValueError(
                f"Episode {episode_index}: negative timestamps detected"
            )

        # P1-2 FIX: Validate gripper position range (0.0-1.0)
        for i, action in enumerate(action_data):
            gripper_pos = action[-1]  # Last element is gripper
            if not (0.0 <= gripper_pos <= 1.0):
                raise ValueError(
                    f"Episode {episode_index}, frame {i}: gripper position {gripper_pos:.3f} out of range [0.0, 1.0]"
                )

        # P1-2 FIX: Validate action continuity (detect large jumps between frames)
        # Max joint displacement per frame (radians or meters depending on joint type)
        # This is a heuristic - actual limits depend on robot and control frequency
        max_joint_delta = 0.5  # radians per timestep (conservative for 30Hz control)
        max_gripper_delta = 0.3  # gripper can move faster (0-1 normalized)

        for i in range(1, len(action_data)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0:
                continue  # Already validated monotonic timestamps

            prev_action = action_data[i - 1]
            curr_action = action_data[i]

            # Check joint continuity
            for j in range(expected_action_dim - 1):  # Exclude gripper
                delta = abs(curr_action[j] - prev_action[j])
                # Scale by time delta for variable-rate trajectories
                velocity = delta / dt if dt > 0 else float('inf')

                if delta > max_joint_delta and velocity > max_joint_delta * 30:  # Assume ~30Hz base rate
                    raise ValueError(
                        f"Episode {episode_index}, frame {i}: Large joint discontinuity detected. "
                        f"Joint {j}: delta={delta:.3f} rad, velocity={velocity:.3f} rad/s. "
                        f"This may indicate a trajectory solver error or collision."
                    )

            # Check gripper continuity
            gripper_delta = abs(curr_action[-1] - prev_action[-1])
            if gripper_delta > max_gripper_delta:
                # Gripper jumps are acceptable (open/close), but log warning for extreme values
                if gripper_delta > 0.8:  # Very large jump (nearly full range)
                    self.log(
                        f"Episode {episode_index}, frame {i}: Large gripper jump {gripper_delta:.3f}. "
                        f"This is acceptable but verify grasp timing.",
                        "WARNING"
                    )

        # P1-2 FIX: Validate joint velocity limits (heuristic based on robot type)
        # Standard industrial robot velocity limits (conservative)
        max_joint_velocity = 2.0  # rad/s (conservative for 7-DOF arms)

        for i in range(1, len(state_data)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0:
                continue

            prev_state = state_data[i - 1]
            curr_state = state_data[i]

            for j in range(expected_state_dim):
                velocity = abs(curr_state[j] - prev_state[j]) / dt
                if velocity > max_joint_velocity:
                    raise ValueError(
                        f"Episode {episode_index}, frame {i}: Joint velocity limit exceeded. "
                        f"Joint {j}: velocity={velocity:.3f} rad/s, limit={max_joint_velocity:.3f} rad/s. "
                        f"This trajectory may not be executable on real hardware."
                    )

        # P1-2 FIX: Validate joint acceleration limits (heuristic)
        max_joint_acceleration = 5.0  # rad/s² (conservative)

        for i in range(2, len(state_data)):
            dt1 = timestamps[i - 1] - timestamps[i - 2]
            dt2 = timestamps[i] - timestamps[i - 1]
            if dt1 <= 0 or dt2 <= 0:
                continue

            prev_state = state_data[i - 2]
            curr_state = state_data[i - 1]
            next_state = state_data[i]

            for j in range(expected_state_dim):
                vel1 = (curr_state[j] - prev_state[j]) / dt1
                vel2 = (next_state[j] - curr_state[j]) / dt2
                acceleration = abs(vel2 - vel1) / ((dt1 + dt2) / 2)

                if acceleration > max_joint_acceleration:
                    raise ValueError(
                        f"Episode {episode_index}, frame {i}: Joint acceleration limit exceeded. "
                        f"Joint {j}: acceleration={acceleration:.3f} rad/s², limit={max_joint_acceleration:.3f} rad/s². "
                        f"This trajectory may not be executable on real hardware."
                    )

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

        # Validate LeRobot v2.0 schema compliance
        self._validate_lerobot_schema(
            timestamps=timestamps,
            state_data=state_data,
            action_data=action_data,
            episode_index=episode.episode_index,
        )

        # P1-1 FIX: Validate ee_position is not None (cannot use [0,0,0] as it's a valid position)
        for i, s in enumerate(states):
            if s.ee_position is None:
                raise ValueError(
                    f"Episode {episode.episode_index}: ee_position is None at frame {i}. "
                    f"Cannot use [0,0,0] as fallback since it's a valid position. "
                    f"Ensure trajectory solver computes ee_position for all frames."
                )

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
                # EE position (now guaranteed to be non-None)
                "observation.ee_position": pa.array(
                    [s.ee_position.tolist() for s in states],
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
        # Base features
        features = {
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
        }

        # Add visual observation features
        if self.config.include_images:
            for camera_type in self.config.camera_types:
                features[f"observation.images.{camera_type}"] = {
                    "dtype": "video",
                    "shape": [self.config.image_resolution[1], self.config.image_resolution[0], 3],
                    "video_info": {
                        "fps": self.config.fps,
                        "codec": self.config.video_codec,
                    },
                }

        # Add depth features (Plus/Full packs)
        if self.config.include_depth:
            for camera_type in self.config.camera_types:
                features[f"observation.depth.{camera_type}"] = {
                    "dtype": "float32",
                    "shape": [self.config.image_resolution[1], self.config.image_resolution[0]],
                    "unit": "meters",
                }

        # Add segmentation features (Plus/Full packs)
        if self.config.include_segmentation:
            for camera_type in self.config.camera_types:
                features[f"observation.segmentation.{camera_type}"] = {
                    "dtype": "uint8",
                    "shape": [self.config.image_resolution[1], self.config.image_resolution[0]],
                    "includes": ["semantic", "instance"],
                }

        # Add ground-truth features (Plus/Full packs)
        if self.config.include_bboxes:
            features["ground_truth.bboxes_2d"] = {
                "dtype": "json",
                "format": "coco",
            }
            features["ground_truth.bboxes_3d"] = {
                "dtype": "json",
                "format": "camera_space",
            }

        # Add object pose features (Full pack)
        if self.config.include_object_poses:
            features["ground_truth.object_poses"] = {
                "dtype": "json",
                "format": "quaternion_position",
                "coordinate_system": "world",
            }

        # Add contact features (Full pack)
        if self.config.include_contacts:
            features["ground_truth.contacts"] = {
                "dtype": "json",
                "format": "contact_list",
            }

        # Add privileged state (Full pack)
        if self.config.include_privileged_state:
            features["ground_truth.privileged_state"] = {
                "dtype": "json",
                "format": "physics_state",
            }

        # Check data source for validation
        # This determines if the data was generated with real physics or mock
        data_source_info = self._get_data_source_info()

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
            "ground_truth_path": "ground_truth" if any([
                self.config.include_depth,
                self.config.include_segmentation,
                self.config.include_bboxes,
                self.config.include_object_poses,
                self.config.include_contacts,
            ]) else None,
            "features": features,
            "data_pack": {
                "tier": self.config.data_pack_tier,
                "cameras": self.config.camera_types,
                "resolution": list(self.config.image_resolution),
                "includes_visual_obs": self.config.include_images,
                "includes_depth": self.config.include_depth,
                "includes_segmentation": self.config.include_segmentation,
                "includes_bboxes": self.config.include_bboxes,
                "includes_object_poses": self.config.include_object_poses,
                "includes_contacts": self.config.include_contacts,
                "includes_privileged_state": self.config.include_privileged_state,
            },
            # Data quality and source information
            "data_quality": {
                "physics_validated": data_source_info["physics_validated"],
                "sensor_source": data_source_info["sensor_source"],
                "validation_type": data_source_info["validation_type"],
                "isaac_sim_used": data_source_info["isaac_sim_used"],
                "warnings": data_source_info["warnings"],
            },
            "splits": {
                "train": f"0:{len(self.episodes)}",
            },
            "created_at": datetime.utcnow().isoformat() + "Z",
            "generator": "BlueprintPipeline/episode-generation-job",
            "generator_version": "2.0.0",
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
                    "quality_score": episode.quality_score,
                    "is_mock": episode.is_mock,  # P0-5 FIX: Explicit flag for mock data
                }
                # Add sensor data info if available
                if episode.sensor_data is not None:
                    meta["has_visual_obs"] = True
                    meta["cameras"] = list(episode.sensor_data.camera_ids) if hasattr(episode.sensor_data, 'camera_ids') else []
                f.write(json.dumps(meta) + "\n")

    def _write_visual_observations(self, output_dir: Path) -> None:
        """Write visual observations (images, videos) for all episodes."""
        if not self.config.include_images:
            return

        videos_dir = output_dir / "videos"

        chunk_idx = 0

        for episode in self.episodes:
            # Determine chunk
            if episode.episode_index > 0 and episode.episode_index % self.config.chunk_size == 0:
                chunk_idx += 1

            chunk_dir = videos_dir / f"chunk-{chunk_idx:03d}"

            if episode.sensor_data is not None:
                self._write_episode_videos(episode, chunk_dir)
            elif episode.camera_video_path is not None:
                # Legacy: copy existing video if available
                self._copy_video(episode.camera_video_path, chunk_dir, episode.episode_index)

    def _write_episode_videos(self, episode: LeRobotEpisode, chunk_dir: Path) -> None:
        """Write video files for an episode's visual observations."""
        sensor_data = episode.sensor_data
        if sensor_data is None or not hasattr(sensor_data, 'frames'):
            return

        # Write video for each camera
        for camera_id in sensor_data.camera_ids if hasattr(sensor_data, 'camera_ids') else []:
            video_dir = chunk_dir / f"observation.images.{camera_id}"
            video_dir.mkdir(parents=True, exist_ok=True)

            video_path = video_dir / f"episode_{episode.episode_index:06d}.mp4"

            # Collect RGB frames for this camera
            frames = []
            for frame in sensor_data.frames:
                if hasattr(frame, 'rgb_images') and camera_id in frame.rgb_images:
                    frames.append(frame.rgb_images[camera_id])

            if frames:
                self._write_video(frames, video_path, self.config.fps)

    def _write_video(self, frames: List[np.ndarray], video_path: Path, fps: float) -> None:
        """Write frames to a video file."""
        if not frames:
            return

        if HAVE_IMAGEIO:
            try:
                writer = imageio.get_writer(
                    str(video_path),
                    fps=fps,
                    codec="libx264",
                    quality=8,
                )
                for frame in frames:
                    # Ensure RGB format (H, W, 3)
                    if frame.ndim == 3 and frame.shape[-1] == 3:
                        writer.append_data(frame)
                writer.close()
                self.log(f"  Wrote video: {video_path}")
            except Exception as e:
                self.log(f"  Video write failed: {e}", "WARNING")
                # Fallback to individual frames
                self._write_frames_as_images(frames, video_path.parent, video_path.stem)
        else:
            # Fallback to individual frames
            self._write_frames_as_images(frames, video_path.parent, video_path.stem)

    def _write_frames_as_images(self, frames: List[np.ndarray], output_dir: Path, prefix: str) -> None:
        """Write frames as individual images (fallback)."""
        frames_dir = output_dir / f"{prefix}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_path = frames_dir / f"frame_{i:06d}.png"
            if HAVE_PIL:
                Image.fromarray(frame).save(frame_path)
            else:
                np.save(frame_path.with_suffix(".npy"), frame)

    def _copy_video(self, src_path: Path, chunk_dir: Path, episode_index: int) -> None:
        """Copy an existing video file to the output directory."""
        import shutil

        video_dir = chunk_dir / "observation.images.camera"
        video_dir.mkdir(parents=True, exist_ok=True)
        dst_path = video_dir / f"episode_{episode_index:06d}.mp4"
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            self.log(f"  Failed to copy video: {e}", "WARNING")

    def _write_ground_truth(self, output_dir: Path) -> None:
        """Write ground-truth labels for all episodes (Plus/Full packs)."""
        if not any([
            self.config.include_depth,
            self.config.include_segmentation,
            self.config.include_bboxes,
            self.config.include_object_poses,
            self.config.include_contacts,
        ]):
            return

        gt_dir = output_dir / "ground_truth"
        chunk_idx = 0

        for episode in self.episodes:
            if episode.episode_index > 0 and episode.episode_index % self.config.chunk_size == 0:
                chunk_idx += 1

            chunk_dir = gt_dir / f"chunk-{chunk_idx:03d}"

            if episode.sensor_data is not None:
                self._write_episode_ground_truth(episode, chunk_dir)

    def _write_episode_ground_truth(self, episode: LeRobotEpisode, chunk_dir: Path) -> None:
        """Write ground-truth data for a single episode."""
        sensor_data = episode.sensor_data
        if sensor_data is None or not hasattr(sensor_data, 'frames'):
            return

        episode_idx = episode.episode_index

        # Depth maps
        if self.config.include_depth:
            self._write_depth_data(sensor_data, chunk_dir, episode_idx)

        # Segmentation masks
        if self.config.include_segmentation:
            self._write_segmentation_data(sensor_data, chunk_dir, episode_idx)

        # Bounding boxes
        if self.config.include_bboxes:
            self._write_bbox_data(sensor_data, chunk_dir, episode_idx)

        # Object poses
        if self.config.include_object_poses:
            self._write_object_pose_data(sensor_data, chunk_dir, episode_idx)

        # Contacts
        if self.config.include_contacts:
            self._write_contact_data(sensor_data, chunk_dir, episode_idx)

        # Privileged state
        if self.config.include_privileged_state:
            self._write_privileged_state_data(sensor_data, chunk_dir, episode_idx)

    def _write_depth_data(self, sensor_data: Any, chunk_dir: Path, episode_idx: int) -> None:
        """Write depth maps for an episode."""
        for camera_id in sensor_data.camera_ids if hasattr(sensor_data, 'camera_ids') else []:
            depth_dir = chunk_dir / "depth" / camera_id
            depth_dir.mkdir(parents=True, exist_ok=True)

            depth_frames = []
            for frame in sensor_data.frames:
                if hasattr(frame, 'depth_maps') and camera_id in frame.depth_maps:
                    depth_frames.append(frame.depth_maps[camera_id])

            if depth_frames:
                depth_path = depth_dir / f"episode_{episode_idx:06d}.npz"
                np.savez_compressed(
                    depth_path,
                    depth=np.stack(depth_frames),
                    fps=self.config.fps,
                    unit="meters",
                )

    def _write_segmentation_data(self, sensor_data: Any, chunk_dir: Path, episode_idx: int) -> None:
        """Write segmentation masks for an episode."""
        for camera_id in sensor_data.camera_ids if hasattr(sensor_data, 'camera_ids') else []:
            seg_dir = chunk_dir / "segmentation" / camera_id
            seg_dir.mkdir(parents=True, exist_ok=True)

            semantic_frames = []
            instance_frames = []

            for frame in sensor_data.frames:
                if hasattr(frame, 'semantic_masks') and camera_id in frame.semantic_masks:
                    semantic_frames.append(frame.semantic_masks[camera_id])
                if hasattr(frame, 'instance_masks') and camera_id in frame.instance_masks:
                    instance_frames.append(frame.instance_masks[camera_id])

            if semantic_frames or instance_frames:
                seg_path = seg_dir / f"episode_{episode_idx:06d}.npz"
                data = {"fps": self.config.fps}
                if semantic_frames:
                    data["semantic"] = np.stack(semantic_frames)
                if instance_frames:
                    data["instance"] = np.stack(instance_frames)
                if hasattr(sensor_data, 'semantic_labels'):
                    data["label_mapping"] = json.dumps(sensor_data.semantic_labels)
                np.savez_compressed(seg_path, **data)

    def _write_bbox_data(self, sensor_data: Any, chunk_dir: Path, episode_idx: int) -> None:
        """Write bounding box annotations for an episode."""
        bbox_dir = chunk_dir / "bboxes"
        bbox_dir.mkdir(parents=True, exist_ok=True)

        bbox_data = {"episode_index": episode_idx, "frames": []}

        for frame in sensor_data.frames:
            frame_bboxes = {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "bboxes_2d": {},
                "bboxes_3d": {},
            }

            if hasattr(frame, 'bboxes_2d'):
                frame_bboxes["bboxes_2d"] = frame.bboxes_2d
            if hasattr(frame, 'bboxes_3d'):
                frame_bboxes["bboxes_3d"] = frame.bboxes_3d

            bbox_data["frames"].append(frame_bboxes)

        bbox_path = bbox_dir / f"episode_{episode_idx:06d}.json"
        with open(bbox_path, "w") as f:
            json.dump(bbox_data, f, indent=2, default=self._json_serializer)

    def _write_object_pose_data(self, sensor_data: Any, chunk_dir: Path, episode_idx: int) -> None:
        """Write object pose data for an episode."""
        pose_dir = chunk_dir / "object_poses"
        pose_dir.mkdir(parents=True, exist_ok=True)

        pose_data = {
            "episode_index": episode_idx,
            "object_metadata": sensor_data.object_metadata if hasattr(sensor_data, 'object_metadata') else {},
            "frames": [],
        }

        for frame in sensor_data.frames:
            frame_poses = {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "poses": frame.object_poses if hasattr(frame, 'object_poses') else {},
            }
            pose_data["frames"].append(frame_poses)

        pose_path = pose_dir / f"episode_{episode_idx:06d}.json"
        with open(pose_path, "w") as f:
            json.dump(pose_data, f, indent=2, default=self._json_serializer)

    def _write_contact_data(self, sensor_data: Any, chunk_dir: Path, episode_idx: int) -> None:
        """Write contact information for an episode."""
        contact_dir = chunk_dir / "contacts"
        contact_dir.mkdir(parents=True, exist_ok=True)

        contact_data = {"episode_index": episode_idx, "frames": []}

        for frame in sensor_data.frames:
            frame_contacts = {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "contacts": frame.contacts if hasattr(frame, 'contacts') else [],
            }
            contact_data["frames"].append(frame_contacts)

        contact_path = contact_dir / f"episode_{episode_idx:06d}.json"
        with open(contact_path, "w") as f:
            json.dump(contact_data, f, indent=2, default=self._json_serializer)

    def _write_privileged_state_data(self, sensor_data: Any, chunk_dir: Path, episode_idx: int) -> None:
        """Write privileged physics state for an episode."""
        priv_dir = chunk_dir / "privileged_state"
        priv_dir.mkdir(parents=True, exist_ok=True)

        priv_data = {"episode_index": episode_idx, "frames": []}

        for frame in sensor_data.frames:
            frame_state = {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "state": frame.privileged_state if hasattr(frame, 'privileged_state') else None,
            }
            priv_data["frames"].append(frame_state)

        priv_path = priv_dir / f"episode_{episode_idx:06d}.json"
        with open(priv_path, "w") as f:
            json.dump(priv_data, f, indent=2, default=self._json_serializer)

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.uint8, np.uint16)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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
