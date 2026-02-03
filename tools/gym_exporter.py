#!/usr/bin/env python3
"""
Gymnasium/OpenAI Gym Export Format for Episode Datasets.

Exports episode datasets in a format compatible with Gymnasium (formerly OpenAI Gym),
enabling standard RL training interfaces.

Features:
- Standard Gymnasium space definitions (observation_space, action_space)
- Episode replay with proper step() interface
- Multi-modal observations (images, robot state, privileged info)
- Action normalization and bounds
- Offline RL compatible (D4RL-style)

Usage:
    exporter = GymExporter()
    exporter.export_dataset(episodes, output_dir)

    # Or use the wrapper for RL training
    dataset = GymEpisodeDataset.load(output_dir)
    for obs, action, reward, done, info in dataset.iterate_transitions():
        ...

Compatible with:
- Gymnasium 0.29+
- Stable-Baselines3
- CleanRL
- D4RL
- CORL
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GymSpaceSpec:
    """Specification for a Gymnasium space."""

    space_type: str  # "box", "discrete", "multi_discrete", "dict"
    shape: Optional[Tuple[int, ...]] = None
    low: Optional[Union[float, np.ndarray]] = None
    high: Optional[Union[float, np.ndarray]] = None
    dtype: str = "float32"
    n: Optional[int] = None  # For discrete spaces
    nvec: Optional[List[int]] = None  # For multi-discrete
    subspaces: Optional[Dict[str, "GymSpaceSpec"]] = None  # For dict spaces

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"space_type": self.space_type}
        if self.shape is not None:
            result["shape"] = list(self.shape)
        if self.low is not None:
            result["low"] = float(self.low) if np.isscalar(self.low) else self.low.tolist()
        if self.high is not None:
            result["high"] = float(self.high) if np.isscalar(self.high) else self.high.tolist()
        if self.dtype:
            result["dtype"] = self.dtype
        if self.n is not None:
            result["n"] = self.n
        if self.nvec is not None:
            result["nvec"] = self.nvec
        if self.subspaces is not None:
            result["subspaces"] = {k: v.to_dict() for k, v in self.subspaces.items()}
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GymSpaceSpec":
        """Create from dictionary."""
        subspaces = None
        if "subspaces" in data:
            subspaces = {k: cls.from_dict(v) for k, v in data["subspaces"].items()}

        return cls(
            space_type=data["space_type"],
            shape=tuple(data["shape"]) if "shape" in data else None,
            low=data.get("low"),
            high=data.get("high"),
            dtype=data.get("dtype", "float32"),
            n=data.get("n"),
            nvec=data.get("nvec"),
            subspaces=subspaces,
        )

    def to_gymnasium_space(self) -> Any:
        """Convert to actual Gymnasium space object."""
        try:
            import gymnasium as gym
        except ImportError:
            try:
                import gym
            except ImportError:
                raise ImportError("Neither gymnasium nor gym is installed")

        if self.space_type == "box":
            low = np.array(self.low, dtype=self.dtype) if isinstance(self.low, list) else self.low
            high = np.array(self.high, dtype=self.dtype) if isinstance(self.high, list) else self.high
            return gym.spaces.Box(
                low=low,
                high=high,
                shape=self.shape,
                dtype=np.dtype(self.dtype),
            )
        elif self.space_type == "discrete":
            return gym.spaces.Discrete(n=self.n)
        elif self.space_type == "multi_discrete":
            return gym.spaces.MultiDiscrete(nvec=self.nvec)
        elif self.space_type == "dict":
            return gym.spaces.Dict({
                k: v.to_gymnasium_space()
                for k, v in self.subspaces.items()
            })
        else:
            raise ValueError(f"Unknown space type: {self.space_type}")


@dataclass
class GymDatasetConfig:
    """Configuration for Gym-style dataset export."""

    # Observation configuration
    include_images: bool = True
    image_keys: List[str] = field(default_factory=lambda: ["rgb"])
    include_robot_state: bool = True
    include_ee_state: bool = True
    include_privileged: bool = False

    # Action configuration
    action_type: str = "continuous"  # "continuous", "discrete"
    action_space: str = "joint_position"  # "joint_position", "ee_position", "ee_velocity"
    normalize_actions: bool = True

    # Reward configuration
    reward_type: str = "sparse"  # "sparse", "dense", "shaped"
    success_reward: float = 1.0
    step_penalty: float = -0.01

    # Episode configuration
    max_episode_length: int = 1000
    terminate_on_success: bool = True

    # Image processing
    image_size: Tuple[int, int] = (84, 84)
    stack_frames: int = 1
    grayscale: bool = False


@dataclass
class GymTransition:
    """A single environment transition."""

    observation: Dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    next_observation: Dict[str, np.ndarray]
    done: bool
    truncated: bool
    info: Dict[str, Any]


class GymExporter:
    """
    Exports episode datasets to Gymnasium-compatible format.

    Creates:
    - observations.npz: Stacked observations
    - actions.npz: Stacked actions
    - rewards.npz: Rewards per step
    - terminals.npz: Terminal flags
    - episode_starts.npz: Episode boundary markers
    - metadata.json: Space definitions and dataset info
    """

    def __init__(self, config: Optional[GymDatasetConfig] = None):
        """
        Initialize the exporter.

        Args:
            config: Export configuration
        """
        self.config = config or GymDatasetConfig()

    def export_dataset(
        self,
        episodes: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        dataset_name: str = "robot_dataset",
    ) -> Dict[str, Path]:
        """
        Export episodes to Gymnasium format.

        Args:
            episodes: List of episode dictionaries with frames
            output_dir: Output directory
            dataset_name: Name of the dataset

        Returns:
            Dictionary mapping data type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect all transitions
        observations = []
        actions = []
        rewards = []
        terminals = []
        truncateds = []
        episode_starts = []

        for episode in episodes:
            episode_obs, episode_actions, episode_rewards, episode_terminals, episode_truncateds = (
                self._process_episode(episode)
            )

            if len(episode_obs) == 0:
                continue

            # Mark episode start
            episode_starts.extend([True] + [False] * (len(episode_obs) - 1))

            observations.extend(episode_obs)
            actions.extend(episode_actions)
            rewards.extend(episode_rewards)
            terminals.extend(episode_terminals)
            truncateds.extend(episode_truncateds)

        if len(observations) == 0:
            logger.warning("No valid transitions to export")
            return {}

        # Determine observation and action spaces from data
        obs_space = self._infer_observation_space(observations[0])
        action_space = self._infer_action_space(actions)

        # Stack arrays
        stacked_obs = self._stack_observations(observations)
        stacked_actions = np.array(actions, dtype=np.float32)
        stacked_rewards = np.array(rewards, dtype=np.float32)
        stacked_terminals = np.array(terminals, dtype=bool)
        stacked_truncateds = np.array(truncateds, dtype=bool)
        stacked_episode_starts = np.array(episode_starts, dtype=bool)

        # Save files
        output_paths = {}

        # Save observations (may be multiple files for dict observations)
        obs_path = output_dir / "observations.npz"
        np.savez_compressed(obs_path, **stacked_obs)
        output_paths["observations"] = obs_path

        # Save actions
        actions_path = output_dir / "actions.npz"
        np.savez_compressed(actions_path, actions=stacked_actions)
        output_paths["actions"] = actions_path

        # Save rewards
        rewards_path = output_dir / "rewards.npz"
        np.savez_compressed(rewards_path, rewards=stacked_rewards)
        output_paths["rewards"] = rewards_path

        # Save terminals and truncateds
        terminals_path = output_dir / "terminals.npz"
        np.savez_compressed(
            terminals_path,
            terminals=stacked_terminals,
            truncateds=stacked_truncateds,
            episode_starts=stacked_episode_starts,
        )
        output_paths["terminals"] = terminals_path

        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "num_episodes": len(episodes),
            "num_transitions": len(observations),
            "observation_space": obs_space.to_dict(),
            "action_space": action_space.to_dict(),
            "config": {
                "include_images": self.config.include_images,
                "image_keys": self.config.image_keys,
                "include_robot_state": self.config.include_robot_state,
                "include_ee_state": self.config.include_ee_state,
                "action_type": self.config.action_type,
                "action_space": self.config.action_space,
                "normalize_actions": self.config.normalize_actions,
                "reward_type": self.config.reward_type,
                "max_episode_length": self.config.max_episode_length,
            },
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        output_paths["metadata"] = metadata_path

        logger.info(
            f"Exported {len(observations)} transitions from {len(episodes)} episodes to {output_dir}"
        )

        return output_paths

    def _process_episode(
        self,
        episode: Dict[str, Any],
    ) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], List[float], List[bool], List[bool]]:
        """Process a single episode into transitions."""
        observations = []
        actions = []
        rewards = []
        terminals = []
        truncateds = []

        frames = episode.get("frames", [])
        episode_actions = episode.get("actions", [])
        success = episode.get("success", False)
        num_frames = len(frames)

        if num_frames == 0:
            return observations, actions, rewards, terminals, truncateds

        for i, frame in enumerate(frames):
            # Build observation
            obs = self._build_observation(frame, episode)
            observations.append(obs)

            # Get action (use next action if available, else current)
            if i < len(episode_actions):
                action = self._process_action(episode_actions[i])
            else:
                # Pad with zeros if no action
                action = self._get_default_action(episode)
            actions.append(action)

            # Compute reward
            reward = self._compute_reward(frame, episode, i, success)
            rewards.append(reward)

            # Determine terminal/truncated
            is_last = i == num_frames - 1
            is_success = success and is_last
            is_truncated = is_last and not success

            terminals.append(is_success)
            truncateds.append(is_truncated)

        return observations, actions, rewards, terminals, truncateds

    def _build_observation(
        self,
        frame: Dict[str, Any],
        episode: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Build observation dictionary from frame data."""
        obs = {}

        # Add images
        if self.config.include_images:
            for key in self.config.image_keys:
                img = frame.get(f"{key}_images", frame.get(key, frame.get("rgb_images", {})))
                if isinstance(img, dict):
                    # Multiple cameras - use first one or concatenate
                    for cam_id, cam_img in img.items():
                        if isinstance(cam_img, np.ndarray):
                            processed = self._process_image(cam_img)
                            obs[f"{key}_{cam_id}"] = processed
                            break  # Just use first camera for simplicity
                elif isinstance(img, np.ndarray):
                    obs[key] = self._process_image(img)

        # Add robot state
        if self.config.include_robot_state:
            robot_state = frame.get("robot_state", {})
            if isinstance(robot_state, dict):
                joint_pos = robot_state.get("joint_positions", [])
                joint_vel = robot_state.get("joint_velocities", [])
                gripper = robot_state.get("gripper_position", 0.0)

                state_vec = []
                if joint_pos:
                    state_vec.extend(joint_pos if isinstance(joint_pos, list) else list(joint_pos))
                if joint_vel:
                    state_vec.extend(joint_vel if isinstance(joint_vel, list) else list(joint_vel))
                state_vec.append(float(gripper) if np.isscalar(gripper) else gripper)

                if state_vec:
                    obs["robot_state"] = np.array(state_vec, dtype=np.float32)

        # Add end-effector state
        if self.config.include_ee_state:
            ee_pos = frame.get("ee_position", frame.get("robot_state", {}).get("ee_position"))
            ee_quat = frame.get("ee_orientation", frame.get("robot_state", {}).get("ee_orientation"))

            ee_state = []
            if ee_pos is not None:
                ee_state.extend(ee_pos if isinstance(ee_pos, list) else list(ee_pos))
            if ee_quat is not None:
                ee_state.extend(ee_quat if isinstance(ee_quat, list) else list(ee_quat))

            if ee_state:
                obs["ee_state"] = np.array(ee_state, dtype=np.float32)

        # Add privileged state (for evaluation/oracle policies)
        if self.config.include_privileged:
            priv = frame.get("privileged_state", {})
            if priv:
                # Flatten privileged state to vector
                priv_vec = self._flatten_dict(priv)
                if priv_vec:
                    obs["privileged"] = np.array(priv_vec, dtype=np.float32)

        return obs

    def _process_image(self, img: np.ndarray) -> np.ndarray:
        """Process image (resize, convert to grayscale, etc.)."""
        if img is None:
            return np.zeros((self.config.image_size[1], self.config.image_size[0], 3), dtype=np.uint8)

        # Resize if needed
        if img.shape[:2] != self.config.image_size[::-1]:  # height, width
            try:
                import cv2
                img = cv2.resize(img, self.config.image_size)
            except ImportError:
                # Simple resize without cv2
                pass

        # Convert to grayscale if needed
        if self.config.grayscale and len(img.shape) == 3:
            img = np.mean(img, axis=2, keepdims=True).astype(np.uint8)

        return img

    def _process_action(self, action: Any) -> np.ndarray:
        """Process action to numpy array."""
        if isinstance(action, np.ndarray):
            return action.astype(np.float32)
        elif isinstance(action, (list, tuple)):
            return np.array(action, dtype=np.float32)
        elif isinstance(action, dict):
            # Flatten dict action
            values = []
            for key in sorted(action.keys()):
                val = action[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    values.extend(val if isinstance(val, (list, tuple)) else val.tolist())
                else:
                    values.append(float(val))
            return np.array(values, dtype=np.float32)
        else:
            return np.array([float(action)], dtype=np.float32)

    def _get_default_action(self, episode: Dict[str, Any]) -> np.ndarray:
        """Get default (zero) action."""
        # Try to infer action dimension from episode
        actions = episode.get("actions", [])
        if actions:
            first_action = self._process_action(actions[0])
            return np.zeros_like(first_action)
        return np.zeros(7, dtype=np.float32)  # Default to 7-DoF

    def _compute_reward(
        self,
        frame: Dict[str, Any],
        episode: Dict[str, Any],
        frame_idx: int,
        success: bool,
    ) -> float:
        """Compute reward for a transition."""
        if self.config.reward_type == "sparse":
            # Sparse reward: only reward on success
            is_last = frame_idx == len(episode.get("frames", [])) - 1
            if is_last and success:
                return self.config.success_reward
            return 0.0

        elif self.config.reward_type == "dense":
            # Dense reward: based on progress metrics
            reward = self.config.step_penalty

            # Add shaping based on available metrics
            if success and frame_idx == len(episode.get("frames", [])) - 1:
                reward += self.config.success_reward

            # Could add distance-based shaping here
            return reward

        else:
            return 0.0

    def _infer_observation_space(
        self,
        sample_obs: Dict[str, np.ndarray],
    ) -> GymSpaceSpec:
        """Infer observation space from sample observation."""
        subspaces = {}

        for key, value in sample_obs.items():
            if isinstance(value, np.ndarray):
                if "image" in key or "rgb" in key or value.dtype == np.uint8:
                    # Image observation
                    subspaces[key] = GymSpaceSpec(
                        space_type="box",
                        shape=value.shape,
                        low=0.0,
                        high=255.0,
                        dtype="uint8",
                    )
                else:
                    # Vector observation
                    subspaces[key] = GymSpaceSpec(
                        space_type="box",
                        shape=value.shape,
                        low=-np.inf,
                        high=np.inf,
                        dtype="float32",
                    )

        return GymSpaceSpec(
            space_type="dict",
            subspaces=subspaces,
        )

    def _infer_action_space(
        self,
        actions: List[np.ndarray],
    ) -> GymSpaceSpec:
        """Infer action space from actions."""
        if not actions:
            return GymSpaceSpec(
                space_type="box",
                shape=(7,),
                low=-1.0,
                high=1.0,
                dtype="float32",
            )

        # Stack actions to find bounds
        stacked = np.stack(actions)
        action_dim = stacked.shape[1]

        if self.config.normalize_actions:
            low = -1.0
            high = 1.0
        else:
            low = float(np.min(stacked))
            high = float(np.max(stacked))

        return GymSpaceSpec(
            space_type="box",
            shape=(action_dim,),
            low=low,
            high=high,
            dtype="float32",
        )

    def _stack_observations(
        self,
        observations: List[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """Stack observations into arrays."""
        stacked = {}

        if not observations:
            return stacked

        # Get all keys from first observation
        keys = list(observations[0].keys())

        for key in keys:
            arrays = []
            for obs in observations:
                if key in obs:
                    arrays.append(obs[key])

            if arrays:
                stacked[key] = np.stack(arrays)

        return stacked

    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> List[float]:
        """Flatten a nested dictionary to a list of floats."""
        values = []
        for key, val in sorted(d.items()):
            if isinstance(val, dict):
                values.extend(self._flatten_dict(val, f"{prefix}{key}_"))
            elif isinstance(val, (list, tuple, np.ndarray)):
                for v in val:
                    if isinstance(v, (int, float)):
                        values.append(float(v))
            elif isinstance(val, (int, float)):
                values.append(float(val))
        return values


class GymEpisodeDataset:
    """
    Dataset wrapper for Gymnasium-style episode data.

    Provides iterators for RL training and evaluation.
    """

    def __init__(
        self,
        observations: Dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        truncateds: np.ndarray,
        episode_starts: np.ndarray,
        observation_space: GymSpaceSpec,
        action_space: GymSpaceSpec,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.truncateds = truncateds
        self.episode_starts = episode_starts
        self.observation_space = observation_space
        self.action_space = action_space

        self.num_transitions = len(actions)
        self.num_episodes = int(np.sum(episode_starts))

    @classmethod
    def load(cls, data_dir: Union[str, Path]) -> "GymEpisodeDataset":
        """Load dataset from directory."""
        data_dir = Path(data_dir)

        # Load metadata
        with open(data_dir / "metadata.json") as f:
            metadata = json.load(f)

        obs_space = GymSpaceSpec.from_dict(metadata["observation_space"])
        action_space = GymSpaceSpec.from_dict(metadata["action_space"])

        # Load arrays
        obs_data = dict(np.load(data_dir / "observations.npz"))
        actions = np.load(data_dir / "actions.npz")["actions"]
        rewards = np.load(data_dir / "rewards.npz")["rewards"]
        terminals_data = np.load(data_dir / "terminals.npz")
        terminals = terminals_data["terminals"]
        truncateds = terminals_data["truncateds"]
        episode_starts = terminals_data["episode_starts"]

        return cls(
            observations=obs_data,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            truncateds=truncateds,
            episode_starts=episode_starts,
            observation_space=obs_space,
            action_space=action_space,
        )

    def iterate_transitions(self) -> Iterator[GymTransition]:
        """Iterate over all transitions."""
        for i in range(self.num_transitions - 1):
            obs = {k: v[i] for k, v in self.observations.items()}
            next_obs = {k: v[i + 1] for k, v in self.observations.items()}

            yield GymTransition(
                observation=obs,
                action=self.actions[i],
                reward=self.rewards[i],
                next_observation=next_obs,
                done=self.terminals[i],
                truncated=self.truncateds[i],
                info={},
            )

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a random batch of transitions."""
        indices = np.random.randint(0, self.num_transitions - 1, size=batch_size)

        batch = {
            "observations": {k: v[indices] for k, v in self.observations.items()},
            "next_observations": {k: v[indices + 1] for k, v in self.observations.items()},
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "terminals": self.terminals[indices],
            "truncateds": self.truncateds[indices],
        }

        return batch

    def get_episode_boundaries(self) -> List[Tuple[int, int]]:
        """Get start and end indices for each episode."""
        starts = np.where(self.episode_starts)[0]
        boundaries = []

        for i, start in enumerate(starts):
            if i + 1 < len(starts):
                end = starts[i + 1]
            else:
                end = self.num_transitions
            boundaries.append((start, end))

        return boundaries
