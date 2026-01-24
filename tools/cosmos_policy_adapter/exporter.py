"""Cosmos Policy dataset exporter.

Converts BlueprintPipeline episode data to the format expected by
NVIDIA's Cosmos Policy training loop. The output is a self-contained
dataset directory that can be directly loaded by the Cosmos Policy
data loader.

Key format requirements (from the Cosmos Policy codebase):
- Actions normalized to [-1, +1]
- Multi-view images as ordered MP4 videos per camera
- Proprioception as float32 vectors (joint positions + gripper)
- Language task descriptions (T5-encoded at training time)
- Episode-level Parquet files with per-frame observation/action pairs

Reference: https://github.com/nvlabs/cosmos-policy
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.cosmos_policy_adapter.config import (
    CosmosPolicyConfig,
    COSMOS_POLICY_DEFAULTS,
)
from tools.cosmos_policy_adapter.normalizer import ActionNormalizer
from tools.utils.atomic_write import write_json_atomic

logger = logging.getLogger(__name__)


class CosmosPolicyExporter:
    """Export episodes to Cosmos Policy training format.

    Converts episode data (observations, actions, proprioception, task
    descriptions) into the format expected by the Cosmos Policy fine-tuning
    pipeline. Handles action normalization, video linking/copying, and
    metadata generation.

    Output structure:
        cosmos_policy/
        ├── meta/
        │   ├── info.json
        │   ├── tasks.jsonl
        │   ├── episodes.jsonl
        │   └── normalization_stats.json
        ├── data/
        │   └── chunk-000/
        │       ├── episode_000000.parquet
        │       └── ...
        ├── videos/
        │   └── chunk-000/
        │       ├── observation.images.{camera}/
        │       │   ├── episode_000000.mp4
        │       │   └── ...
        │       └── ...
        └── config/
            └── training_config.yaml
    """

    def __init__(
        self,
        output_dir: Path,
        config: Optional[CosmosPolicyConfig] = None,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.config = config or COSMOS_POLICY_DEFAULTS
        self.verbose = verbose
        self.normalizer = ActionNormalizer(
            target_range=self.config.action_range,
        )

        # Tracking
        self._episodes: List[Dict[str, Any]] = []
        self._tasks: Dict[str, int] = {}  # task_description -> task_index
        self._all_actions: List[np.ndarray] = []
        self._all_states: List[np.ndarray] = []
        self._total_frames: int = 0

    def log(self, msg: str) -> None:
        if self.verbose:
            logger.info("[COSMOS-POLICY-EXPORT] %s", msg)

    def export_episodes(
        self,
        episodes: List[Dict[str, Any]],
        robot_type: str = "franka",
        scene_id: str = "unknown",
        source_videos_dir: Optional[Path] = None,
    ) -> Path:
        """Export episodes to Cosmos Policy format.

        Args:
            episodes: List of episode data dictionaries. Each episode should have:
                - episode_id: str
                - task: str (language task description)
                - frames: List[Dict] with keys:
                    - joint_positions: List[float]
                    - joint_velocities: List[float] (optional)
                    - gripper_position: float
                    - ee_position: List[float] (optional)
                    - ee_orientation: List[float] (optional)
                    - action: List[float]
                    - timestamp: float
                - success: bool (optional)
                - quality_score: float (optional)
            robot_type: Robot type identifier
            scene_id: Scene identifier
            source_videos_dir: Optional path to source video files for linking

        Returns:
            Path to output directory
        """
        if not self.config.enabled:
            self.log("Cosmos Policy export disabled (ENABLE_COSMOS_POLICY_EXPORT=false)")
            return self.output_dir

        self.log(f"Exporting {len(episodes)} episodes for Cosmos Policy fine-tuning")

        # Create directory structure
        meta_dir = self.output_dir / "meta"
        data_dir = self.output_dir / "data" / "chunk-000"
        videos_dir = self.output_dir / "videos" / "chunk-000"
        config_dir = self.output_dir / "config"

        for d in [meta_dir, data_dir, config_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Create video directories per camera
        for camera_id in self.config.camera_ids:
            (videos_dir / f"observation.images.{camera_id}").mkdir(
                parents=True, exist_ok=True
            )

        # Phase 1: Collect all actions/states for normalization
        self.log("Phase 1: Computing normalization statistics...")
        self._collect_normalization_data(episodes)
        self._fit_normalizer()

        # Phase 2: Export each episode
        self.log("Phase 2: Exporting episodes...")
        for i, episode in enumerate(episodes):
            self._export_episode(
                episode=episode,
                episode_index=i,
                data_dir=data_dir,
                videos_dir=videos_dir,
                source_videos_dir=source_videos_dir,
            )

            if self.verbose and (i + 1) % 50 == 0:
                self.log(f"  Exported {i + 1}/{len(episodes)} episodes")

        # Phase 3: Write metadata
        self.log("Phase 3: Writing metadata...")
        self._write_info(meta_dir, robot_type, scene_id)
        self._write_tasks(meta_dir)
        self._write_episodes_index(meta_dir)
        self._write_normalization_stats(meta_dir)
        self._write_training_config(config_dir, robot_type)
        self._write_checksums(self.output_dir)

        self.log(
            f"Cosmos Policy export complete: {len(self._episodes)} episodes, "
            f"{self._total_frames} frames -> {self.output_dir}"
        )

        return self.output_dir

    def _collect_normalization_data(self, episodes: List[Dict[str, Any]]) -> None:
        """Collect all actions and states for normalization fitting."""
        self._all_actions = []
        self._all_states = []

        for episode in episodes:
            frames = episode.get("frames", [])
            if not frames:
                continue

            ep_actions = []
            ep_states = []

            for frame in frames:
                action = frame.get("action", [])
                if action:
                    ep_actions.append(action)

                # Proprioception: joint_positions + gripper
                joint_pos = frame.get("joint_positions", [])
                gripper = frame.get("gripper_position", 0.0)
                if joint_pos:
                    state = list(joint_pos) + [gripper]
                    ep_states.append(state)

            if ep_actions:
                self._all_actions.append(np.array(ep_actions, dtype=np.float32))
            if ep_states:
                self._all_states.append(np.array(ep_states, dtype=np.float32))

    def _fit_normalizer(self) -> None:
        """Fit the normalizer on collected data."""
        if self.config.normalize_actions and self._all_actions:
            self.normalizer.fit(
                actions=self._all_actions,
                states=self._all_states if self._all_states else None,
            )
            self.log(
                f"  Fitted normalizer: action_dim={self.normalizer.action_stats.dim}, "
                f"state_dim={self.normalizer.state_stats.dim if self.normalizer.state_stats else 'N/A'}"
            )

    def _export_episode(
        self,
        episode: Dict[str, Any],
        episode_index: int,
        data_dir: Path,
        videos_dir: Path,
        source_videos_dir: Optional[Path] = None,
    ) -> None:
        """Export a single episode to Parquet + video files."""
        frames = episode.get("frames", [])
        if not frames:
            return

        task_description = episode.get("task", "manipulation task")
        task_index = self._get_or_create_task(task_description)

        # Build per-frame data arrays
        timestamps = []
        actions_raw = []
        actions_normalized = []
        proprio_raw = []
        proprio_normalized = []
        ee_positions = []
        ee_orientations = []

        for frame in frames:
            timestamps.append(frame.get("timestamp", 0.0))

            # Actions
            action = frame.get("action", [0.0] * self.config.action_dim)
            # Pad or truncate to action_dim
            action = self._pad_or_truncate(action, self.config.action_dim)
            actions_raw.append(action)

            # Proprioception: joint_positions + gripper_position
            joint_pos = frame.get("joint_positions", [0.0] * 7)
            gripper = frame.get("gripper_position", 0.0)
            proprio = list(joint_pos) + [gripper]
            proprio = self._pad_or_truncate(proprio, self.config.state_dim + 1)
            proprio_raw.append(proprio)

            # End-effector pose
            ee_pos = frame.get("ee_position", [0.0, 0.0, 0.0])
            ee_orient = frame.get("ee_orientation", [1.0, 0.0, 0.0, 0.0])
            ee_positions.append(ee_pos)
            ee_orientations.append(ee_orient)

        # Normalize
        actions_array = np.array(actions_raw, dtype=np.float32)
        proprio_array = np.array(proprio_raw, dtype=np.float32)

        if self.config.normalize_actions and self.normalizer.action_stats:
            actions_normalized = self.normalizer.normalize_actions(actions_array)
        else:
            actions_normalized = actions_array

        if self.config.normalize_actions and self.normalizer.state_stats:
            proprio_normalized = self.normalizer.normalize_states(proprio_array)
        else:
            proprio_normalized = proprio_array

        # Build Parquet-compatible data structure
        # Using dict-of-lists format for Parquet serialization
        episode_data = {
            "timestamp": timestamps,
            "episode_index": [episode_index] * len(frames),
            "frame_index": list(range(len(frames))),
            "task_index": [task_index] * len(frames),
        }

        # Actions (normalized, per dimension)
        for dim_i in range(actions_normalized.shape[1] if len(actions_normalized.shape) > 1 else self.config.action_dim):
            episode_data[f"action.dim_{dim_i}"] = (
                actions_normalized[:, dim_i].tolist()
                if len(actions_normalized.shape) > 1
                else [0.0] * len(frames)
            )

        # Proprioception (normalized, per dimension)
        for dim_i in range(proprio_normalized.shape[1] if len(proprio_normalized.shape) > 1 else self.config.state_dim + 1):
            episode_data[f"observation.state.dim_{dim_i}"] = (
                proprio_normalized[:, dim_i].tolist()
                if len(proprio_normalized.shape) > 1
                else [0.0] * len(frames)
            )

        # EE pose (unnormalized — used for auxiliary supervision)
        if self.config.include_ee_pose:
            ee_pos_array = np.array(ee_positions, dtype=np.float32)
            for dim_i in range(3):
                episode_data[f"observation.ee_position.dim_{dim_i}"] = (
                    ee_pos_array[:, dim_i].tolist()
                )
            ee_orient_array = np.array(ee_orientations, dtype=np.float32)
            for dim_i in range(4):
                episode_data[f"observation.ee_orientation.dim_{dim_i}"] = (
                    ee_orient_array[:, dim_i].tolist()
                )

        # Write Parquet file
        output_parquet = data_dir / f"episode_{episode_index:06d}.parquet"
        self._write_parquet(episode_data, output_parquet)

        # Link/copy video files
        self._link_videos(episode, episode_index, videos_dir, source_videos_dir)

        # Track episode metadata
        self._episodes.append({
            "episode_index": episode_index,
            "task_index": task_index,
            "task": task_description,
            "num_frames": len(frames),
            "duration_seconds": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0,
            "success": episode.get("success", True),
            "quality_score": episode.get("quality_score", 1.0),
            "data_path": str(output_parquet.relative_to(self.output_dir)),
        })

        self._total_frames += len(frames)

    def _get_or_create_task(self, description: str) -> int:
        """Get or create a task index for a description."""
        if description not in self._tasks:
            self._tasks[description] = len(self._tasks)
        return self._tasks[description]

    def _pad_or_truncate(self, values: List[float], target_dim: int) -> List[float]:
        """Pad with zeros or truncate to target dimension."""
        if len(values) >= target_dim:
            return values[:target_dim]
        return values + [0.0] * (target_dim - len(values))

    def _write_parquet(self, data: Dict[str, List], output_path: Path) -> None:
        """Write episode data to Parquet format."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table(data)
            pq.write_table(table, output_path, compression="zstd")

        except ImportError:
            # Fallback: write as JSON if PyArrow not available
            fallback_path = output_path.with_suffix(".json")
            with open(fallback_path, "w") as f:
                json.dump(data, f)
            logger.warning(
                "[COSMOS-POLICY-EXPORT] PyArrow not available, wrote JSON fallback: %s",
                fallback_path,
            )

    def _link_videos(
        self,
        episode: Dict[str, Any],
        episode_index: int,
        videos_dir: Path,
        source_videos_dir: Optional[Path],
    ) -> None:
        """Link or copy video files for this episode.

        Cosmos Policy expects one MP4 per camera per episode, organized as:
        videos/chunk-000/observation.images.{camera}/episode_{index:06d}.mp4
        """
        if source_videos_dir is None:
            return

        episode_id = episode.get("episode_id", f"episode_{episode_index:06d}")

        for camera_id in self.config.camera_ids:
            target_path = (
                videos_dir
                / f"observation.images.{camera_id}"
                / f"episode_{episode_index:06d}.mp4"
            )

            if target_path.exists():
                continue

            # Try common source video path patterns
            source_candidates = [
                source_videos_dir / f"chunk-000" / f"observation.images.{camera_id}" / f"episode_{episode_index:06d}.mp4",
                source_videos_dir / f"observation.images.{camera_id}" / f"episode_{episode_index:06d}.mp4",
                source_videos_dir / camera_id / f"{episode_id}.mp4",
                source_videos_dir / f"{camera_id}_{episode_id}.mp4",
            ]

            for source_path in source_candidates:
                if source_path.exists():
                    # Use symlink if on same filesystem, copy otherwise
                    try:
                        target_path.symlink_to(source_path.resolve())
                    except (OSError, NotImplementedError):
                        shutil.copy2(source_path, target_path)
                    break

    def _write_info(self, meta_dir: Path, robot_type: str, scene_id: str) -> None:
        """Write dataset info.json."""
        info = {
            "format": "cosmos_policy",
            "format_version": "1.0.0",
            "cosmos_policy_version": "2601.16163",  # arxiv paper ID
            "robot_type": robot_type,
            "scene_id": scene_id,
            "total_episodes": len(self._episodes),
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "fps": 30.0,
            "action_dim": self.config.action_dim,
            "state_dim": self.config.state_dim + 1,  # joints + gripper
            "action_chunk_size": self.config.action_chunk_size,
            "cameras": self.config.camera_ids,
            "image_size": list(self.config.image_size),
            "normalize_actions": self.config.normalize_actions,
            "action_range": list(self.config.action_range),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "pipeline": "BlueprintPipeline",
            "compatible_with": [
                "nvidia/Cosmos-Policy-Predict2-2B",
                "cosmos-policy>=1.0",
            ],
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [self.config.state_dim + 1],
                    "description": "Normalized joint positions + gripper",
                },
                "action": {
                    "dtype": "float32",
                    "shape": [self.config.action_dim],
                    "description": "Normalized action (joint targets + gripper cmd)",
                },
                "observation.ee_position": {
                    "dtype": "float32",
                    "shape": [3],
                    "description": "End-effector position (meters)",
                },
                "observation.ee_orientation": {
                    "dtype": "float32",
                    "shape": [4],
                    "description": "End-effector orientation (quaternion)",
                },
            },
        }

        # Add camera features
        for camera_id in self.config.camera_ids:
            info["features"][f"observation.images.{camera_id}"] = {
                "dtype": "video",
                "shape": list(self.config.image_size) + [3],
                "description": f"RGB from {camera_id} camera",
                "video_info": {
                    "fps": 30.0,
                    "codec": "h264",
                },
            }

        write_json_atomic(meta_dir / "info.json", info, indent=2)

    def _write_tasks(self, meta_dir: Path) -> None:
        """Write tasks.jsonl with language descriptions."""
        tasks_path = meta_dir / "tasks.jsonl"
        lines = []
        for description, task_index in sorted(self._tasks.items(), key=lambda x: x[1]):
            lines.append(json.dumps({
                "task_index": task_index,
                "task": description,
            }))

        with open(tasks_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _write_episodes_index(self, meta_dir: Path) -> None:
        """Write episodes.jsonl index."""
        episodes_path = meta_dir / "episodes.jsonl"
        lines = []
        for ep_info in self._episodes:
            lines.append(json.dumps(ep_info))

        with open(episodes_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _write_normalization_stats(self, meta_dir: Path) -> None:
        """Write normalization statistics for use at inference time."""
        stats_path = meta_dir / "normalization_stats.json"
        self.normalizer.save(stats_path)

    def _write_training_config(self, config_dir: Path, robot_type: str) -> None:
        """Write turnkey training configuration YAML."""
        training = self.config.training_config

        config_content = f"""# Cosmos Policy Fine-Tuning Configuration
# Generated by BlueprintPipeline
# Robot: {robot_type}
# Episodes: {len(self._episodes)}
# Total frames: {self._total_frames}
#
# Usage:
#   uv run --extra cu128 python train.py --config config/training_config.yaml
#
# Reference: https://github.com/nvlabs/cosmos-policy

model:
  name: "{training.model_name}"
  base_model: "{training.base_model}"
  pretrained_weights: "nvidia/Cosmos-Policy-Predict2-2B"

training:
  learning_rate: {training.learning_rate}
  batch_size: {training.batch_size}
  num_epochs: {training.num_epochs}
  warmup_steps: {training.warmup_steps}
  gradient_accumulation_steps: {training.gradient_accumulation_steps}
  weight_decay: {training.weight_decay}
  max_grad_norm: {training.max_grad_norm}
  precision: "{training.training_precision}"
  seed: 42

diffusion:
  num_steps_train: {training.num_diffusion_steps_train}
  num_steps_inference: {training.num_diffusion_steps_inference}
  noise_schedule: "{training.noise_schedule}"
  sigma_min: {training.sigma_min}

action_chunking:
  chunk_size: {training.action_chunk_size}
  proprio_chunk_size: {training.proprio_chunk_size}

auxiliary_objectives:
  world_model:
    enabled: {str(training.enable_world_model).lower()}
    batch_ratio: {training.world_model_batch_ratio}
  value_function:
    enabled: {str(training.enable_value_function).lower()}
    batch_ratio: {training.value_function_batch_ratio}

planning:
  enabled: {str(training.enable_planning).lower()}
  num_candidates: {training.planning_num_candidates}
  depth: {training.planning_depth}

data:
  dataset_path: ".."
  image_size: [{training.image_size[0]}, {training.image_size[1]}]
  cameras: {json.dumps(self.config.camera_ids)}
  action_dim: {self.config.action_dim}
  state_dim: {self.config.state_dim + 1}
  normalization_stats: "../meta/normalization_stats.json"
  max_episode_length: {training.max_episode_length}

hardware:
  min_gpus: {training.min_gpus}
  min_gpu_memory_gb: {training.min_gpu_memory_gb}
  recommended: "{training.recommended_gpu}"

robot:
  type: "{robot_type}"
  action_space: "joint_position"
  control_frequency_hz: 30.0
"""

        config_path = config_dir / "training_config.yaml"
        with open(config_path, "w") as f:
            f.write(config_content)

    def _write_checksums(self, output_dir: Path) -> None:
        """Write SHA-256 checksums for all output files."""
        checksums: Dict[str, Dict[str, Any]] = {}
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.name != "checksums.json":
                rel = path.relative_to(output_dir).as_posix()
                hasher = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        hasher.update(chunk)
                checksums[rel] = {
                    "sha256": hasher.hexdigest(),
                    "size_bytes": path.stat().st_size,
                }

        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "algorithm": "sha256",
            "files": checksums,
        }
        write_json_atomic(output_dir / "checksums.json", payload, indent=2)
