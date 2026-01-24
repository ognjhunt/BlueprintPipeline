"""Configuration for Cosmos Policy export and training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CosmosPolicyTrainingConfig:
    """Training hyperparameters for Cosmos Policy fine-tuning.

    These match the defaults from the official Cosmos Policy codebase
    (https://github.com/nvlabs/cosmos-policy) for the Predict2-2B model.
    """

    # Model
    model_name: str = "nvidia/Cosmos-Policy-Predict2-2B"
    base_model: str = "Cosmos-Predict2-2B"

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Diffusion
    num_diffusion_steps_train: int = 1000
    num_diffusion_steps_inference: int = 10
    noise_schedule: str = "log_normal_uniform_hybrid"
    sigma_min: float = 4.0  # Higher than default for action accuracy

    # Action chunking
    action_chunk_size: int = 16  # Predict 16 future actions per step
    proprio_chunk_size: int = 1  # Current proprioception only

    # Auxiliary objectives (world model + value function)
    enable_world_model: bool = True
    enable_value_function: bool = True
    world_model_batch_ratio: float = 0.25  # 25% of batch for world model
    value_function_batch_ratio: float = 0.25  # 25% for value function

    # Planning (best-of-N at inference)
    enable_planning: bool = False  # Requires rollout data
    planning_num_candidates: int = 8
    planning_depth: int = 1

    # Hardware
    min_gpus: int = 8
    min_gpu_memory_gb: int = 80
    recommended_gpu: str = "H100-80GB"
    training_precision: str = "bf16"

    # Data
    image_size: Tuple[int, int] = (256, 256)
    max_episode_length: int = 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": {
                "name": self.model_name,
                "base_model": self.base_model,
            },
            "training": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "warmup_steps": self.warmup_steps,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "weight_decay": self.weight_decay,
                "max_grad_norm": self.max_grad_norm,
                "precision": self.training_precision,
            },
            "diffusion": {
                "num_steps_train": self.num_diffusion_steps_train,
                "num_steps_inference": self.num_diffusion_steps_inference,
                "noise_schedule": self.noise_schedule,
                "sigma_min": self.sigma_min,
            },
            "action_chunking": {
                "action_chunk_size": self.action_chunk_size,
                "proprio_chunk_size": self.proprio_chunk_size,
            },
            "auxiliary": {
                "world_model": self.enable_world_model,
                "value_function": self.enable_value_function,
                "world_model_batch_ratio": self.world_model_batch_ratio,
                "value_function_batch_ratio": self.value_function_batch_ratio,
            },
            "planning": {
                "enabled": self.enable_planning,
                "num_candidates": self.planning_num_candidates,
                "depth": self.planning_depth,
            },
            "hardware": {
                "min_gpus": self.min_gpus,
                "min_gpu_memory_gb": self.min_gpu_memory_gb,
                "recommended_gpu": self.recommended_gpu,
            },
            "data": {
                "image_size": list(self.image_size),
                "max_episode_length": self.max_episode_length,
            },
        }


@dataclass
class CosmosPolicyConfig:
    """Configuration for the Cosmos Policy export adapter.

    Environment Variables:
        ENABLE_COSMOS_POLICY_EXPORT: Enable/disable Cosmos Policy export (default: true)
        COSMOS_POLICY_ACTION_CHUNK_SIZE: Action chunk size (default: 16)
        COSMOS_POLICY_IMAGE_SIZE: Image resize target (default: 256)
        COSMOS_POLICY_CAMERAS: Comma-separated camera list (default: from pipeline)
        COSMOS_POLICY_ENABLE_PLANNING: Include planning data (default: false)
        COSMOS_POLICY_NORMALIZE_ACTIONS: Normalize actions to [-1,1] (default: true)
    """

    # Export control
    enabled: bool = True

    # Camera configuration
    camera_ids: List[str] = field(default_factory=lambda: ["wrist", "overhead"])
    image_size: Tuple[int, int] = (256, 256)

    # Action space
    action_dim: int = 8  # 7 joints + gripper
    state_dim: int = 7  # 7 joint positions
    normalize_actions: bool = True
    action_range: Tuple[float, float] = (-1.0, 1.0)

    # Chunking
    action_chunk_size: int = 16

    # Features to include
    include_proprioception: bool = True
    include_ee_pose: bool = True
    include_gripper_state: bool = True

    # Training config
    training_config: CosmosPolicyTrainingConfig = field(
        default_factory=CosmosPolicyTrainingConfig
    )

    # Firebase upload
    firebase_upload_prefix: str = "datasets/cosmos_policy"

    def __post_init__(self) -> None:
        """Apply environment variable overrides."""
        if "ENABLE_COSMOS_POLICY_EXPORT" in os.environ:
            from tools.config.env import parse_bool_env
            self.enabled = parse_bool_env(
                os.environ["ENABLE_COSMOS_POLICY_EXPORT"], default=True
            )

        if "COSMOS_POLICY_ACTION_CHUNK_SIZE" in os.environ:
            self.action_chunk_size = int(os.environ["COSMOS_POLICY_ACTION_CHUNK_SIZE"])
            self.training_config.action_chunk_size = self.action_chunk_size

        if "COSMOS_POLICY_IMAGE_SIZE" in os.environ:
            size = int(os.environ["COSMOS_POLICY_IMAGE_SIZE"])
            self.image_size = (size, size)
            self.training_config.image_size = self.image_size

        if "COSMOS_POLICY_CAMERAS" in os.environ:
            self.camera_ids = [
                c.strip() for c in os.environ["COSMOS_POLICY_CAMERAS"].split(",")
            ]

        if "COSMOS_POLICY_ENABLE_PLANNING" in os.environ:
            from tools.config.env import parse_bool_env
            self.training_config.enable_planning = parse_bool_env(
                os.environ["COSMOS_POLICY_ENABLE_PLANNING"], default=False
            )

        if "COSMOS_POLICY_NORMALIZE_ACTIONS" in os.environ:
            from tools.config.env import parse_bool_env
            self.normalize_actions = parse_bool_env(
                os.environ["COSMOS_POLICY_NORMALIZE_ACTIONS"], default=True
            )


# Default configuration matching Cosmos Policy paper recommendations
COSMOS_POLICY_DEFAULTS = CosmosPolicyConfig()
