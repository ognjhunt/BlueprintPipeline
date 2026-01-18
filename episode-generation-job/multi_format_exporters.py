#!/usr/bin/env python3
"""
Multi-Format Dataset Exporters.

Provides export functionality for multiple formats beyond LeRobot:
- RLDS (TensorFlow Datasets) - For academic labs using TF/JAX
- HDF5 - For robomimic and other academic pipelines
- ROS bag - For legacy ROS-based systems

These formats are requested by labs per the research AI conversation:
- RLDS/TFDS-style: Many academic labs use this
- HDF5: Very common in sim + some real pipelines
- ROS bag: Common in robotics stacks

Reference datasets using these formats:
- Open X-Embodiment: RLDS
- robomimic: HDF5
- DROID: HDF5 + ROS bag components
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tools.utils.atomic_write import write_json_atomic

logger = logging.getLogger(__name__)

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# RLDS Exporter (TensorFlow Datasets format)
# =============================================================================


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class RLDSEpisode:
    """
    Episode data in RLDS format.

    RLDS (Reinforcement Learning Datasets) is the format used by
    Open X-Embodiment and many academic robotics datasets.

    Structure follows RLDS spec:
    - observation: Dict of observation features
    - action: Action taken
    - reward: Reward signal
    - is_first: Boolean, true for first step
    - is_last: Boolean, true for last step
    - is_terminal: Boolean, true if episode terminated
    - discount: Discount factor
    """

    episode_id: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RLDSExporter:
    """
    Export episodes to RLDS (TensorFlow Datasets) format.

    Output structure:
    dataset/
    ├── train/
    │   ├── episode_000000.tfrecord
    │   └── ...
    ├── val/
    │   └── ...
    ├── test/
    │   └── ...
    ├── dataset_info.json
    └── features.json

    Compatible with:
    - TensorFlow Datasets
    - JAX-based training pipelines
    - Open X-Embodiment format
    """

    def __init__(
        self,
        output_dir: Path,
        dataset_name: str = "blueprint_robotics",
        version: str = "1.0.0",
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.version = version
        self.verbose = verbose

        # Try to import TensorFlow (optional dependency)
        self._tf_available = False
        try:
            import tensorflow as tf
            self._tf = tf
            self._tf_available = True
        except ImportError:
            if self.verbose:
                logger.warning("[RLDS] TensorFlow not available, using fallback export")

    def log(self, msg: str) -> None:
        if self.verbose:
            logger.info("[RLDS-EXPORT] %s", msg)

    def export_episodes(
        self,
        episodes: List[Dict[str, Any]],
        splits: Dict[str, List[str]],
        robot_type: str = "franka",
        camera_names: List[str] = None,
    ) -> Path:
        """
        Export episodes to RLDS format.

        Args:
            episodes: List of episode data dictionaries
            splits: Dict mapping split name to list of episode IDs
            robot_type: Robot type for metadata
            camera_names: List of camera names

        Returns:
            Path to output directory
        """
        self.log(f"Exporting {len(episodes)} episodes to RLDS format")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create split directories
        for split_name in splits:
            (self.output_dir / split_name).mkdir(exist_ok=True)

        # Build feature spec
        features = self._build_feature_spec(episodes, camera_names or ["wrist"])

        # Export episodes by split
        episode_map = {ep["episode_id"]: ep for ep in episodes}

        for split_name, episode_ids in splits.items():
            split_episodes = [episode_map[eid] for eid in episode_ids if eid in episode_map]
            self._export_split(split_name, split_episodes, features)

        # Write dataset info
        self._write_dataset_info(features, splits, robot_type)
        self._write_checksums_manifest()

        self.log(f"RLDS export complete: {self.output_dir}")
        return self.output_dir

    def _build_feature_spec(
        self,
        episodes: List[Dict[str, Any]],
        camera_names: List[str],
    ) -> Dict[str, Any]:
        """Build RLDS feature specification."""
        # Get sample episode for shapes
        sample = episodes[0] if episodes else {}
        sample_frame = sample.get("frames", [{}])[0] if sample.get("frames") else {}

        num_joints = len(sample_frame.get("joint_positions", [0] * 7))
        resolution = sample.get("resolution", [640, 480])

        features = {
            "observation": {
                "state": {
                    "dtype": "float32",
                    "shape": [num_joints],
                    "description": "Joint positions (radians)",
                },
                "gripper_position": {
                    "dtype": "float32",
                    "shape": [1],
                    "description": "Gripper position (meters)",
                },
            },
            "action": {
                "dtype": "float32",
                "shape": [num_joints + 1],
                "description": "Joint commands + gripper",
            },
            "reward": {
                "dtype": "float32",
                "shape": [],
                "description": "Reward signal",
            },
            "is_first": {"dtype": "bool", "shape": []},
            "is_last": {"dtype": "bool", "shape": []},
            "is_terminal": {"dtype": "bool", "shape": []},
            "discount": {"dtype": "float32", "shape": []},
        }

        # Add camera features
        for cam_name in camera_names:
            features["observation"][f"image_{cam_name}"] = {
                "dtype": "uint8",
                "shape": [resolution[1], resolution[0], 3],
                "description": f"RGB image from {cam_name} camera",
            }

        # Add optional dynamics features
        features["observation"]["joint_velocities"] = {
            "dtype": "float32",
            "shape": [num_joints],
            "description": "Joint velocities (rad/s)",
        }
        features["observation"]["joint_torques"] = {
            "dtype": "float32",
            "shape": [num_joints],
            "description": "Joint torques (Nm)",
        }
        features["observation"]["ee_position"] = {
            "dtype": "float32",
            "shape": [3],
            "description": "End-effector position (meters)",
        }

        # Add language instruction
        features["language_instruction"] = {
            "dtype": "string",
            "shape": [],
            "description": "Natural language task description",
        }

        return features

    def _export_split(
        self,
        split_name: str,
        episodes: List[Dict[str, Any]],
        features: Dict[str, Any],
    ) -> None:
        """Export episodes for a single split."""
        split_dir = self.output_dir / split_name

        if self._tf_available:
            self._export_tfrecord(split_dir, episodes, features)
        else:
            self._export_json_fallback(split_dir, episodes, features)

    def _export_tfrecord(
        self,
        split_dir: Path,
        episodes: List[Dict[str, Any]],
        features: Dict[str, Any],
    ) -> None:
        """Export to TFRecord format."""
        tf = self._tf

        for i, episode in enumerate(episodes):
            output_path = split_dir / f"episode_{i:06d}.tfrecord"

            with tf.io.TFRecordWriter(str(output_path)) as writer:
                frames = episode.get("frames", [])

                for j, frame in enumerate(frames):
                    example = self._frame_to_tf_example(
                        frame,
                        is_first=(j == 0),
                        is_last=(j == len(frames) - 1),
                        language_instruction=episode.get("task", ""),
                    )
                    writer.write(example.SerializeToString())

            self.log(f"  Wrote {output_path.name}")

    def _frame_to_tf_example(
        self,
        frame: Dict[str, Any],
        is_first: bool,
        is_last: bool,
        language_instruction: str,
    ) -> Any:
        """Convert frame to TensorFlow Example."""
        tf = self._tf

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
            if isinstance(value, (list, np.ndarray)):
                return tf.train.Feature(float_list=tf.train.FloatList(value=value))
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        def _int_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        feature_dict = {
            "observation/state": _float_feature(frame.get("joint_positions", [])),
            "observation/gripper_position": _float_feature([frame.get("gripper_position", 0.0)]),
            "action": _float_feature(frame.get("action", [])),
            "reward": _float_feature(frame.get("reward", 0.0)),
            "is_first": _int_feature(1 if is_first else 0),
            "is_last": _int_feature(1 if is_last else 0),
            "is_terminal": _int_feature(1 if is_last else 0),
            "discount": _float_feature(1.0),
            "language_instruction": _bytes_feature(language_instruction.encode("utf-8")),
        }

        # Add optional fields
        if "joint_velocities" in frame:
            feature_dict["observation/joint_velocities"] = _float_feature(
                frame["joint_velocities"]
            )
        if "joint_torques" in frame:
            feature_dict["observation/joint_torques"] = _float_feature(
                frame["joint_torques"]
            )
        if "ee_position" in frame:
            feature_dict["observation/ee_position"] = _float_feature(
                frame["ee_position"]
            )

        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def _export_json_fallback(
        self,
        split_dir: Path,
        episodes: List[Dict[str, Any]],
        features: Dict[str, Any],
    ) -> None:
        """Fallback export to JSON when TensorFlow not available."""
        for i, episode in enumerate(episodes):
            output_path = split_dir / f"episode_{i:06d}.json"

            rlds_episode = {
                "episode_id": episode.get("episode_id", f"episode_{i}"),
                "language_instruction": episode.get("task", ""),
                "steps": [],
            }

            frames = episode.get("frames", [])
            for j, frame in enumerate(frames):
                step = {
                    "observation": {
                        "state": frame.get("joint_positions", []),
                        "gripper_position": frame.get("gripper_position", 0.0),
                    },
                    "action": frame.get("action", []),
                    "reward": frame.get("reward", 0.0),
                    "is_first": j == 0,
                    "is_last": j == len(frames) - 1,
                    "is_terminal": j == len(frames) - 1,
                    "discount": 1.0,
                }

                # Add optional fields
                for key in ["joint_velocities", "joint_torques", "ee_position"]:
                    if key in frame:
                        step["observation"][key] = frame[key]

                rlds_episode["steps"].append(step)

            write_json_atomic(output_path, rlds_episode, indent=2, default=str)

            self.log(f"  Wrote {output_path.name} (JSON fallback)")

    def _write_dataset_info(
        self,
        features: Dict[str, Any],
        splits: Dict[str, List[str]],
        robot_type: str,
    ) -> None:
        """Write dataset info files."""
        # Dataset info
        dataset_info = {
            "name": self.dataset_name,
            "version": self.version,
            "description": f"Blueprint robotics dataset for {robot_type}",
            "homepage": "https://blueprint.ai",
            "citation": "",
            "splits": {name: {"num_episodes": len(ids)} for name, ids in splits.items()},
            "total_episodes": sum(len(ids) for ids in splits.values()),
            "robot_type": robot_type,
            "created_at": datetime.now().isoformat(),
            "format": "rlds",
            "compatible_with": ["tensorflow_datasets", "jax", "open_x_embodiment"],
            "checksums": {
                "algorithm": "sha256",
                "manifest_path": "checksums.json",
                "format": "relative path -> {sha256, size_bytes}",
            },
        }

        write_json_atomic(self.output_dir / "dataset_info.json", dataset_info, indent=2)

        # Features spec
        write_json_atomic(self.output_dir / "features.json", features, indent=2)

        self.log("Wrote dataset_info.json and features.json")

    def _write_checksums_manifest(self) -> None:
        """Write checksum manifest for RLDS outputs."""
        files = [
            path
            for path in self.output_dir.rglob("*")
            if path.is_file() and path.name != "checksums.json"
        ]
        checksums: Dict[str, Dict[str, Union[str, int]]] = {}
        for path in sorted(files, key=lambda p: p.as_posix()):
            rel_path = path.relative_to(self.output_dir).as_posix()
            checksums[rel_path] = {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "root": ".",
            "algorithm": "sha256",
            "files": checksums,
        }
        write_json_atomic(self.output_dir / "checksums.json", payload, indent=2)


# =============================================================================
# HDF5 Exporter (robomimic format)
# =============================================================================


class HDF5Exporter:
    """
    Export episodes to HDF5 format.

    Compatible with:
    - robomimic
    - Many academic robotics pipelines
    - DROID dataset format

    Output structure:
    dataset.hdf5
    ├── data/
    │   ├── demo_0/
    │   │   ├── obs/
    │   │   │   ├── joint_positions
    │   │   │   ├── gripper_position
    │   │   │   ├── image_wrist
    │   │   │   └── ...
    │   │   ├── actions
    │   │   ├── rewards
    │   │   └── dones
    │   └── demo_1/
    │       └── ...
    └── mask/
        ├── train
        ├── valid
        └── test
    """

    def __init__(
        self,
        output_path: Path,
        verbose: bool = True,
    ):
        self.output_path = Path(output_path)
        self.verbose = verbose

        # Try to import h5py
        self._h5py_available = False
        try:
            import h5py
            self._h5py = h5py
            self._h5py_available = True
        except ImportError:
            if self.verbose:
                logger.warning("[HDF5] h5py not available")

    def log(self, msg: str) -> None:
        if self.verbose:
            logger.info("[HDF5-EXPORT] %s", msg)

    def export_episodes(
        self,
        episodes: List[Dict[str, Any]],
        splits: Dict[str, List[str]],
        robot_type: str = "franka",
        compression: str = "gzip",
    ) -> Path:
        """
        Export episodes to HDF5 format.

        Args:
            episodes: List of episode data dictionaries
            splits: Dict mapping split name to list of episode IDs
            robot_type: Robot type for metadata
            compression: HDF5 compression type

        Returns:
            Path to output file
        """
        if not self._h5py_available:
            raise RuntimeError("h5py not available. Install with: pip install h5py")

        self.log(f"Exporting {len(episodes)} episodes to HDF5 format")

        # Create parent directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        h5py = self._h5py

        with h5py.File(self.output_path, "w") as f:
            # Create data group
            data_grp = f.create_group("data")

            # Episode ID to index mapping
            episode_map = {}

            for i, episode in enumerate(episodes):
                episode_id = episode.get("episode_id", f"demo_{i}")
                episode_map[episode_id] = i

                demo_grp = data_grp.create_group(f"demo_{i}")
                self._write_episode(demo_grp, episode, compression)

                if self.verbose and (i + 1) % 100 == 0:
                    self.log(f"  Wrote {i + 1}/{len(episodes)} episodes")

            # Create mask groups for splits
            mask_grp = f.create_group("mask")
            for split_name, episode_ids in splits.items():
                indices = [episode_map[eid] for eid in episode_ids if eid in episode_map]
                mask_grp.create_dataset(split_name, data=np.array(indices))
                self.log(f"  {split_name}: {len(indices)} episodes")

            # Write metadata
            f.attrs["robot_type"] = robot_type
            f.attrs["total_episodes"] = len(episodes)
            f.attrs["created_at"] = datetime.now().isoformat()
            f.attrs["format"] = "hdf5_robomimic"
            f.attrs["version"] = "1.0.0"

        self.log(f"HDF5 export complete: {self.output_path}")
        return self.output_path

    def _write_episode(
        self,
        grp: Any,
        episode: Dict[str, Any],
        compression: str,
    ) -> None:
        """Write a single episode to HDF5 group."""
        frames = episode.get("frames", [])
        if not frames:
            return

        # Collect arrays for each field
        obs_data = {
            "joint_positions": [],
            "joint_velocities": [],
            "joint_torques": [],
            "joint_efforts": [],
            "gripper_position": [],
            "gripper_force": [],
            "ee_position": [],
            "ee_orientation": [],
            "ee_velocity": [],
        }

        actions = []
        rewards = []
        dones = []

        for i, frame in enumerate(frames):
            # Observations
            obs_data["joint_positions"].append(frame.get("joint_positions", []))
            obs_data["joint_velocities"].append(frame.get("joint_velocities", []))
            obs_data["joint_torques"].append(frame.get("joint_torques", []))
            obs_data["joint_efforts"].append(frame.get("joint_efforts", []))
            obs_data["gripper_position"].append(frame.get("gripper_position", 0.0))
            obs_data["gripper_force"].append(frame.get("gripper_force", 0.0))
            obs_data["ee_position"].append(frame.get("ee_position", [0, 0, 0]))
            obs_data["ee_orientation"].append(frame.get("ee_orientation", [1, 0, 0, 0]))
            obs_data["ee_velocity"].append(frame.get("ee_velocity", [0, 0, 0, 0, 0, 0]))

            # Actions
            actions.append(frame.get("action", []))

            # Rewards and dones
            rewards.append(frame.get("reward", 0.0))
            dones.append(1 if i == len(frames) - 1 else 0)

        # Create observation group
        obs_grp = grp.create_group("obs")

        for key, values in obs_data.items():
            if values and values[0] is not None:
                try:
                    arr = np.array(values, dtype=np.float32)
                    if arr.size > 0:
                        obs_grp.create_dataset(key, data=arr, compression=compression)
                except (ValueError, TypeError):
                    pass  # Skip if can't convert to array

        # Actions, rewards, dones
        if actions:
            grp.create_dataset("actions", data=np.array(actions, dtype=np.float32), compression=compression)
        if rewards:
            grp.create_dataset("rewards", data=np.array(rewards, dtype=np.float32), compression=compression)
        if dones:
            grp.create_dataset("dones", data=np.array(dones, dtype=np.int32), compression=compression)

        # Episode metadata
        grp.attrs["num_samples"] = len(frames)
        grp.attrs["task"] = episode.get("task", "")
        grp.attrs["success"] = episode.get("success", True)
        grp.attrs["quality_score"] = episode.get("quality_score", 1.0)

        # Skill segments (for skill-based learning)
        if "skill_segments" in episode:
            segments = episode["skill_segments"]
            segments_grp = grp.create_group("skill_segments")
            for j, seg in enumerate(segments):
                seg_grp = segments_grp.create_group(f"segment_{j}")
                seg_grp.attrs["skill_type"] = seg.get("skill_type", "unknown")
                seg_grp.attrs["start_frame"] = seg.get("start_frame", 0)
                seg_grp.attrs["end_frame"] = seg.get("end_frame", 0)


# =============================================================================
# ROS Bag Exporter (Legacy ROS systems)
# =============================================================================


class ROSBagExporter:
    """
    Export episodes to ROS bag format.

    Compatible with:
    - ROS/ROS2 based systems
    - Many industrial robotics pipelines
    - Visualization in RViz

    Note: This is a simplified ROS bag format for compatibility.
    Full ROS bag support requires rosbag/rosbag2 packages.
    """

    def __init__(
        self,
        output_dir: Path,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Check for rosbag availability
        self._rosbag_available = False
        try:
            import rosbag
            self._rosbag = rosbag
            self._rosbag_available = True
        except ImportError:
            if self.verbose:
                logger.warning("[ROSBAG] rosbag not available, using JSON fallback")

    def log(self, msg: str) -> None:
        if self.verbose:
            logger.info("[ROSBAG-EXPORT] %s", msg)

    def export_episodes(
        self,
        episodes: List[Dict[str, Any]],
        robot_type: str = "franka",
    ) -> Path:
        """
        Export episodes to ROS bag format.

        Args:
            episodes: List of episode data dictionaries
            robot_type: Robot type for message types

        Returns:
            Path to output directory
        """
        self.log(f"Exporting {len(episodes)} episodes to ROS bag format")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for i, episode in enumerate(episodes):
            episode_id = episode.get("episode_id", f"episode_{i:06d}")

            if self._rosbag_available:
                self._export_rosbag(episode, episode_id, robot_type)
            else:
                self._export_json_fallback(episode, episode_id, robot_type)

            if self.verbose and (i + 1) % 100 == 0:
                self.log(f"  Exported {i + 1}/{len(episodes)} episodes")

        # Write metadata
        metadata = {
            "format": "rosbag",
            "robot_type": robot_type,
            "total_episodes": len(episodes),
            "created_at": datetime.now().isoformat(),
            "topics": self._get_topic_list(robot_type),
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.log(f"ROS bag export complete: {self.output_dir}")
        return self.output_dir

    def _get_topic_list(self, robot_type: str) -> List[str]:
        """Get list of ROS topics for robot type."""
        return [
            f"/{robot_type}/joint_states",
            f"/{robot_type}/gripper_state",
            f"/{robot_type}/ee_pose",
            f"/{robot_type}/ee_wrench",
            "/camera/wrist/image_raw",
            "/camera/wrist/camera_info",
            "/camera/overhead/image_raw",
            "/camera/overhead/camera_info",
            "/task_description",
        ]

    def _export_rosbag(
        self,
        episode: Dict[str, Any],
        episode_id: str,
        robot_type: str,
    ) -> None:
        """Export to actual ROS bag format."""
        rosbag = self._rosbag

        output_path = self.output_dir / f"{episode_id}.bag"

        with rosbag.Bag(str(output_path), "w") as bag:
            frames = episode.get("frames", [])

            for frame in frames:
                timestamp = frame.get("timestamp", 0.0)

                # Joint states
                joint_msg = self._create_joint_state_msg(frame, robot_type)
                if joint_msg:
                    bag.write(f"/{robot_type}/joint_states", joint_msg, t=timestamp)

                # EE pose
                if "ee_position" in frame:
                    pose_msg = self._create_pose_msg(frame)
                    bag.write(f"/{robot_type}/ee_pose", pose_msg, t=timestamp)

        self.log(f"  Wrote {output_path.name}")

    def _export_json_fallback(
        self,
        episode: Dict[str, Any],
        episode_id: str,
        robot_type: str,
    ) -> None:
        """Fallback export to JSON when rosbag not available."""
        output_path = self.output_dir / f"{episode_id}_rosbag.json"

        rosbag_data = {
            "episode_id": episode_id,
            "robot_type": robot_type,
            "topics": {},
        }

        frames = episode.get("frames", [])

        # Joint states topic
        joint_states = []
        for frame in frames:
            joint_states.append({
                "timestamp": frame.get("timestamp", 0.0),
                "name": frame.get("joint_names", []),
                "position": frame.get("joint_positions", []),
                "velocity": frame.get("joint_velocities", []),
                "effort": frame.get("joint_torques", []),
            })
        rosbag_data["topics"][f"/{robot_type}/joint_states"] = joint_states

        # EE pose topic
        ee_poses = []
        for frame in frames:
            if "ee_position" in frame:
                ee_poses.append({
                    "timestamp": frame.get("timestamp", 0.0),
                    "position": frame.get("ee_position", [0, 0, 0]),
                    "orientation": frame.get("ee_orientation", [1, 0, 0, 0]),
                })
        rosbag_data["topics"][f"/{robot_type}/ee_pose"] = ee_poses

        # Gripper state topic
        gripper_states = []
        for frame in frames:
            gripper_states.append({
                "timestamp": frame.get("timestamp", 0.0),
                "position": frame.get("gripper_position", 0.0),
                "force": frame.get("gripper_force", 0.0),
            })
        rosbag_data["topics"][f"/{robot_type}/gripper_state"] = gripper_states

        # EE wrench topic
        ee_wrenches = []
        for frame in frames:
            if "ee_wrench" in frame and frame["ee_wrench"]:
                ee_wrenches.append({
                    "timestamp": frame.get("timestamp", 0.0),
                    "force": frame["ee_wrench"].get("force", [0, 0, 0]),
                    "torque": frame["ee_wrench"].get("torque", [0, 0, 0]),
                })
        rosbag_data["topics"][f"/{robot_type}/ee_wrench"] = ee_wrenches

        # Task description
        rosbag_data["topics"]["/task_description"] = [{
            "timestamp": 0.0,
            "data": episode.get("task", ""),
        }]

        with open(output_path, "w") as f:
            json.dump(rosbag_data, f, indent=2, default=str)

        self.log(f"  Wrote {output_path.name} (JSON fallback)")

    def _create_joint_state_msg(self, frame: Dict[str, Any], robot_type: str) -> Any:
        """Create ROS JointState message."""
        # This would create actual ROS message if rosbag is available
        # For now, return a dict representation
        return {
            "name": frame.get("joint_names", []),
            "position": frame.get("joint_positions", []),
            "velocity": frame.get("joint_velocities", []),
            "effort": frame.get("joint_torques", []),
        }

    def _create_pose_msg(self, frame: Dict[str, Any]) -> Any:
        """Create ROS Pose message."""
        return {
            "position": frame.get("ee_position", [0, 0, 0]),
            "orientation": frame.get("ee_orientation", [1, 0, 0, 0]),
        }


# =============================================================================
# Point Cloud Generator
# =============================================================================


class PointCloudGenerator:
    """
    Generate point clouds from depth images and camera calibration.

    Useful for:
    - 3D-aware world models
    - Point cloud policies (PointNet, etc.)
    - Geometric reasoning
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            logger.info("[POINTCLOUD] %s", msg)

    def depth_to_pointcloud(
        self,
        depth_image: np.ndarray,
        intrinsic_matrix: np.ndarray,
        rgb_image: Optional[np.ndarray] = None,
        extrinsic_matrix: Optional[np.ndarray] = None,
        max_points: int = 10000,
        depth_scale: float = 1.0,
        max_depth: float = 10.0,
    ) -> Dict[str, np.ndarray]:
        """
        Convert depth image to point cloud.

        Args:
            depth_image: Depth image (H, W) in meters
            intrinsic_matrix: 3x3 camera intrinsic matrix
            rgb_image: Optional RGB image (H, W, 3) for colors
            extrinsic_matrix: Optional 4x4 camera-to-world transform
            max_points: Maximum number of points to return
            depth_scale: Scale factor for depth values
            max_depth: Maximum depth to consider

        Returns:
            Dict with 'points' (N, 3), 'colors' (N, 3), 'normals' (N, 3)
        """
        h, w = depth_image.shape

        # Create pixel coordinate grid
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)

        # Get valid depth mask
        depth = depth_image * depth_scale
        valid_mask = (depth > 0) & (depth < max_depth)

        # Flatten and filter
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth[valid_mask]

        # Unproject to 3D camera coordinates
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid

        points_camera = np.stack([x, y, z], axis=1)

        # Transform to world coordinates if extrinsic provided
        if extrinsic_matrix is not None:
            points_homogeneous = np.hstack([
                points_camera,
                np.ones((len(points_camera), 1))
            ])
            points_world = (extrinsic_matrix @ points_homogeneous.T).T[:, :3]
            points = points_world
        else:
            points = points_camera

        # Get colors if RGB image provided
        colors = None
        if rgb_image is not None:
            colors = rgb_image[valid_mask] / 255.0  # Normalize to 0-1

        # Subsample if too many points
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]

        # Estimate normals (simple finite difference approach)
        normals = self._estimate_normals(depth_image, intrinsic_matrix, valid_mask, max_points)

        result = {"points": points.astype(np.float32)}
        if colors is not None:
            result["colors"] = colors.astype(np.float32)
        if normals is not None:
            result["normals"] = normals.astype(np.float32)

        return result

    def _estimate_normals(
        self,
        depth_image: np.ndarray,
        intrinsic_matrix: np.ndarray,
        valid_mask: np.ndarray,
        max_points: int,
    ) -> Optional[np.ndarray]:
        """Estimate surface normals from depth image."""
        try:
            # Compute gradients
            gy, gx = np.gradient(depth_image)

            # Create normal vectors
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]

            nx = -gx * fx
            ny = -gy * fy
            nz = np.ones_like(depth_image)

            # Normalize
            norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
            nx /= norm
            ny /= norm
            nz /= norm

            # Stack and filter
            normals = np.stack([nx, ny, nz], axis=2)
            normals_valid = normals[valid_mask]

            # Subsample if needed
            if len(normals_valid) > max_points:
                indices = np.random.choice(len(normals_valid), max_points, replace=False)
                normals_valid = normals_valid[indices]

            return normals_valid

        except Exception:
            return None


# =============================================================================
# Unified Multi-Format Exporter
# =============================================================================


class MultiFormatExporter:
    """
    Unified exporter that can output to multiple formats.

    Supports:
    - LeRobot (primary)
    - RLDS (TensorFlow Datasets)
    - HDF5 (robomimic)
    - ROS bag (legacy systems)
    """

    def __init__(
        self,
        output_dir: Path,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            logger.info("[MULTI-FORMAT] %s", msg)

    def export(
        self,
        episodes: List[Dict[str, Any]],
        splits: Dict[str, List[str]],
        formats: List[str],
        robot_type: str = "franka",
        dataset_name: str = "blueprint_robotics",
    ) -> Dict[str, Path]:
        """
        Export episodes to multiple formats.

        Args:
            episodes: List of episode data dictionaries
            splits: Dict mapping split name to list of episode IDs
            formats: List of format names ("lerobot", "rlds", "hdf5", "rosbag")
            robot_type: Robot type for metadata
            dataset_name: Name for the dataset

        Returns:
            Dict mapping format name to output path
        """
        self.log(f"Exporting {len(episodes)} episodes to formats: {formats}")

        outputs = {}

        for fmt in formats:
            fmt_lower = fmt.lower()

            if fmt_lower == "rlds":
                exporter = RLDSExporter(
                    output_dir=self.output_dir / "rlds",
                    dataset_name=dataset_name,
                    verbose=self.verbose,
                )
                outputs["rlds"] = exporter.export_episodes(
                    episodes, splits, robot_type
                )

            elif fmt_lower == "hdf5":
                exporter = HDF5Exporter(
                    output_path=self.output_dir / "hdf5" / f"{dataset_name}.hdf5",
                    verbose=self.verbose,
                )
                outputs["hdf5"] = exporter.export_episodes(
                    episodes, splits, robot_type
                )

            elif fmt_lower == "rosbag":
                exporter = ROSBagExporter(
                    output_dir=self.output_dir / "rosbag",
                    verbose=self.verbose,
                )
                outputs["rosbag"] = exporter.export_episodes(episodes, robot_type)

            else:
                self.log(f"Unknown format: {fmt}, skipping")

        self.log(f"Multi-format export complete: {list(outputs.keys())}")
        return outputs


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse
    from tools.logging_config import init_logging

    init_logging()
    parser = argparse.ArgumentParser(description="Multi-format episode exporter")
    parser.add_argument("--format", choices=["rlds", "hdf5", "rosbag", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("./exported_data"))
    parser.add_argument("--demo", action="store_true", help="Run demo export")

    args = parser.parse_args()

    if args.demo:
        logger.info("Running demo export...")

        # Create sample episodes
        sample_episodes = []
        for i in range(5):
            episode = {
                "episode_id": f"demo_{i:04d}",
                "task": f"Pick up object {i}",
                "success": True,
                "quality_score": 0.9,
                "frames": [],
            }

            for j in range(30):
                frame = {
                    "timestamp": j / 30.0,
                    "joint_positions": np.random.randn(7).tolist(),
                    "joint_velocities": np.random.randn(7).tolist(),
                    "joint_torques": np.random.randn(7).tolist(),
                    "gripper_position": 0.04 if j < 15 else 0.0,
                    "ee_position": [0.5 + j * 0.01, 0.0, 0.5],
                    "ee_orientation": [1, 0, 0, 0],
                    "action": np.random.randn(8).tolist(),
                    "reward": 1.0 if j == 29 else 0.0,
                }
                episode["frames"].append(frame)

            sample_episodes.append(episode)

        # Create splits
        splits = {
            "train": [f"demo_{i:04d}" for i in range(3)],
            "val": [f"demo_{i:04d}" for i in range(3, 4)],
            "test": [f"demo_{i:04d}" for i in range(4, 5)],
        }

        # Export
        formats = ["rlds", "hdf5", "rosbag"] if args.format == "all" else [args.format]

        exporter = MultiFormatExporter(
            output_dir=args.output_dir,
            verbose=True,
        )

        outputs = exporter.export(
            episodes=sample_episodes,
            splits=splits,
            formats=formats,
            robot_type="franka",
            dataset_name="demo_dataset",
        )

        logger.info("Exported to: %s", outputs)
