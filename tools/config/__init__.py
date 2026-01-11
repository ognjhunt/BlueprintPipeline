"""Pipeline Configuration Module.

Provides centralized configuration management for the BlueprintPipeline.
Labs can customize thresholds, parameters, and behavior by modifying
the configuration files or providing overrides via environment variables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Configuration file paths
CONFIG_DIR = Path(__file__).parent
PIPELINE_CONFIG_PATH = CONFIG_DIR / "pipeline_config.json"
QUALITY_CONFIG_PATH = CONFIG_DIR.parent / "quality_gates" / "quality_config.json"


@dataclass
class PhysicsThresholds:
    """Physics validation thresholds."""
    mass_min_kg: float = 0.01
    mass_max_kg: float = 500.0
    friction_min: float = 0.0
    friction_max: float = 2.0
    inertia_min: float = 0.0


@dataclass
class EpisodeThresholds:
    """Episode quality thresholds."""
    collision_free_rate_min: float = 0.80
    quality_score_min: float = 0.85
    quality_pass_rate_min: float = 0.50
    min_episodes_required: int = 1


@dataclass
class SimulationThresholds:
    """Simulation stability thresholds."""
    min_stable_steps: int = 10
    max_penetration_depth_m: float = 0.01
    physics_stability_timeout_s: float = 30.0


@dataclass
class HumanApprovalConfig:
    """Human approval workflow configuration."""
    enabled: bool = True
    timeout_hours: float = 24.0
    auto_approve_on_timeout: bool = False
    approval_methods: List[str] = field(default_factory=lambda: ["dashboard", "email", "api"])
    default_approvers: List[str] = field(default_factory=list)
    escalation_after_hours: float = 12.0
    escalation_contacts: List[str] = field(default_factory=list)
    notification_channels: List[str] = field(default_factory=lambda: ["email", "console"])


@dataclass
class GateOverrideConfig:
    """Gate override configuration."""
    allow_manual_override: bool = True
    override_requires_reason: bool = True
    override_log_retention_days: int = 90
    allowed_overriders: List[str] = field(default_factory=list)


@dataclass
class QualityConfig:
    """Complete quality gate configuration."""
    physics: PhysicsThresholds = field(default_factory=PhysicsThresholds)
    episodes: EpisodeThresholds = field(default_factory=EpisodeThresholds)
    simulation: SimulationThresholds = field(default_factory=SimulationThresholds)
    human_approval: HumanApprovalConfig = field(default_factory=HumanApprovalConfig)
    gate_overrides: GateOverrideConfig = field(default_factory=GateOverrideConfig)


@dataclass
class VideoConfig:
    """Video capture configuration."""
    width: int = 640
    height: int = 480
    fps: int = 30
    codec: str = "h264"
    capture_depth: bool = True
    capture_segmentation: bool = True
    num_cameras: int = 3


@dataclass
class PhysicsConfig:
    """Physics simulation configuration."""
    timestep_hz: int = 120
    solver_iterations: int = 4
    solver_type: str = "TGS"
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    enable_gpu_dynamics: bool = True


@dataclass
class DomainRandomizationConfig:
    """Domain randomization configuration."""
    enabled: bool = True
    intensity: str = "medium"
    lighting_variation: float = 0.3
    texture_variation: float = 0.15
    object_position_noise_m: float = 0.03
    object_rotation_noise_deg: float = 15.0
    physics_noise: float = 0.1


@dataclass
class RewardConfig:
    """Reward shaping configuration."""
    sparse_reward: bool = False
    reward_scale: float = 1.0
    success_reward: float = 10.0
    failure_penalty: float = -1.0
    time_penalty_per_step: float = -0.01
    grasp_reward: float = 2.0


@dataclass
class ResourceConfig:
    """Resource allocation configuration."""
    gpu_memory_fraction: float = 0.8
    num_cpu_workers: int = 4
    num_gpu_workers: int = 1
    memory_limit_gb: int = 32


@dataclass
class EpisodeGenerationConfig:
    """Episode generation configuration."""
    episodes_per_task: int = 10
    num_variations: int = 5
    max_parallel_episodes: int = 8
    episode_timeout_seconds: int = 300
    episodes_by_tier: Dict[str, int] = field(default_factory=lambda: {
        "standard": 500,
        "pro": 1500,
        "enterprise": 2500,
        "foundation": 5000
    })


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    episode_generation: EpisodeGenerationConfig = field(default_factory=EpisodeGenerationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    domain_randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)


class ConfigLoader:
    """Configuration loader with environment variable override support.

    Loads configuration from JSON files and supports overrides via:
    1. Environment variables (prefixed with BP_)
    2. Custom config file paths
    3. Programmatic overrides

    Example:
        # Load default config
        config = ConfigLoader.load_quality_config()

        # Override via environment
        # BP_QUALITY_EPISODES_QUALITY_SCORE_MIN=0.9

        # Override programmatically
        config = ConfigLoader.load_quality_config(
            overrides={"episodes": {"quality_score_min": 0.9}}
        )
    """

    _quality_config_cache: Optional[Dict] = None
    _pipeline_config_cache: Optional[Dict] = None

    @classmethod
    def _load_json(cls, path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    @classmethod
    def _apply_env_overrides(
        cls,
        config: Dict[str, Any],
        prefix: str = "BP_"
    ) -> Dict[str, Any]:
        """Apply environment variable overrides.

        Environment variables are parsed as:
        BP_SECTION_KEY=value -> config["section"]["key"] = value

        Example:
            BP_QUALITY_EPISODES_QUALITY_SCORE_MIN=0.9
            -> config["thresholds"]["episodes"]["quality_score_min"] = 0.9
        """
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Parse key path
            parts = key[len(prefix):].lower().split("_")
            if len(parts) < 2:
                continue

            # Navigate to nested dict
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
                if not isinstance(current, dict):
                    break

            if isinstance(current, dict):
                # Convert value type
                final_key = parts[-1]
                try:
                    # Try numeric conversion
                    if "." in value:
                        current[final_key] = float(value)
                    elif value.lower() in ("true", "false"):
                        current[final_key] = value.lower() == "true"
                    elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                        current[final_key] = int(value)
                    else:
                        current[final_key] = value
                except (ValueError, AttributeError):
                    current[final_key] = value

        return config

    @classmethod
    def _deep_merge(cls, base: Dict, override: Dict) -> Dict:
        """Deep merge override into base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def load_quality_config(
        cls,
        config_path: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> QualityConfig:
        """Load quality gate configuration.

        Args:
            config_path: Optional custom config file path
            overrides: Optional dict of overrides to apply
            use_cache: Whether to use cached config

        Returns:
            QualityConfig dataclass with all thresholds
        """
        # Check cache
        if use_cache and cls._quality_config_cache is not None and not overrides:
            config = cls._quality_config_cache
        else:
            path = config_path or QUALITY_CONFIG_PATH
            config = cls._load_json(path)
            config = cls._apply_env_overrides(config, "BP_QUALITY_")

            if overrides:
                config = cls._deep_merge(config, overrides)

            if use_cache and not overrides:
                cls._quality_config_cache = config

        # Parse into dataclass
        thresholds = config.get("thresholds", {})

        return QualityConfig(
            physics=PhysicsThresholds(
                mass_min_kg=thresholds.get("physics", {}).get("mass_min_kg", 0.01),
                mass_max_kg=thresholds.get("physics", {}).get("mass_max_kg", 500.0),
                friction_min=thresholds.get("physics", {}).get("friction_min", 0.0),
                friction_max=thresholds.get("physics", {}).get("friction_max", 2.0),
            ),
            episodes=EpisodeThresholds(
                collision_free_rate_min=thresholds.get("episodes", {}).get("collision_free_rate_min", 0.80),
                quality_score_min=thresholds.get("episodes", {}).get("quality_score_min", 0.85),
                quality_pass_rate_min=thresholds.get("episodes", {}).get("quality_pass_rate_min", 0.50),
                min_episodes_required=thresholds.get("episodes", {}).get("min_episodes_required", 1),
            ),
            simulation=SimulationThresholds(
                min_stable_steps=thresholds.get("simulation", {}).get("min_stable_steps", 10),
                max_penetration_depth_m=thresholds.get("simulation", {}).get("max_penetration_depth_m", 0.01),
                physics_stability_timeout_s=thresholds.get("simulation", {}).get("physics_stability_timeout_s", 30.0),
            ),
            human_approval=HumanApprovalConfig(
                enabled=config.get("human_approval", {}).get("enabled", True),
                timeout_hours=config.get("human_approval", {}).get("timeout_hours", 24.0),
                auto_approve_on_timeout=config.get("human_approval", {}).get("auto_approve_on_timeout", False),
                approval_methods=config.get("human_approval", {}).get("approval_methods", ["dashboard", "email", "api"]),
                notification_channels=config.get("human_approval", {}).get("notification_channels", ["email", "console"]),
            ),
            gate_overrides=GateOverrideConfig(
                allow_manual_override=config.get("gate_overrides", {}).get("allow_manual_override", True),
                override_requires_reason=config.get("gate_overrides", {}).get("override_requires_reason", True),
            ),
        )

    @classmethod
    def load_pipeline_config(
        cls,
        config_path: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> PipelineConfig:
        """Load pipeline configuration.

        Args:
            config_path: Optional custom config file path
            overrides: Optional dict of overrides to apply
            use_cache: Whether to use cached config

        Returns:
            PipelineConfig dataclass with all settings
        """
        if use_cache and cls._pipeline_config_cache is not None and not overrides:
            config = cls._pipeline_config_cache
        else:
            path = config_path or PIPELINE_CONFIG_PATH
            config = cls._load_json(path)
            config = cls._apply_env_overrides(config, "BP_PIPELINE_")

            if overrides:
                config = cls._deep_merge(config, overrides)

            if use_cache and not overrides:
                cls._pipeline_config_cache = config

        # Parse into dataclass
        ep_gen = config.get("episode_generation", {})
        video = config.get("video", {})
        physics = config.get("physics", {})
        dr = config.get("domain_randomization", {})
        reward = config.get("reward_shaping", {})
        resources = config.get("resource_allocation", {})

        return PipelineConfig(
            episode_generation=EpisodeGenerationConfig(
                episodes_per_task=ep_gen.get("episodes_per_task", 10),
                num_variations=ep_gen.get("num_variations", 5),
                max_parallel_episodes=ep_gen.get("max_parallel_episodes", 8),
                episode_timeout_seconds=ep_gen.get("episode_timeout_seconds", 300),
                episodes_by_tier=ep_gen.get("episodes_per_scene", {
                    "standard": 500,
                    "pro": 1500,
                    "enterprise": 2500,
                    "foundation": 5000
                }),
            ),
            video=VideoConfig(
                width=video.get("resolution", {}).get("width", 640),
                height=video.get("resolution", {}).get("height", 480),
                fps=video.get("fps", 30),
                codec=video.get("codec", "h264"),
                capture_depth=video.get("capture_depth", True),
                capture_segmentation=video.get("capture_segmentation", True),
                num_cameras=video.get("num_cameras", 3),
            ),
            physics=PhysicsConfig(
                timestep_hz=physics.get("timestep_hz", 120),
                solver_iterations=physics.get("solver_iterations", 4),
                solver_type=physics.get("solver_type", "TGS"),
                gravity=physics.get("gravity", [0.0, 0.0, -9.81]),
                enable_gpu_dynamics=physics.get("enable_gpu_dynamics", True),
            ),
            domain_randomization=DomainRandomizationConfig(
                enabled=dr.get("enabled", True),
                intensity=dr.get("intensity", "medium"),
            ),
            reward=RewardConfig(
                sparse_reward=reward.get("sparse_reward", False),
                reward_scale=reward.get("reward_scale", 1.0),
                success_reward=reward.get("success_reward", 10.0),
                failure_penalty=reward.get("failure_penalty", -1.0),
            ),
            resources=ResourceConfig(
                gpu_memory_fraction=resources.get("gpu_memory_fraction", 0.8),
                num_cpu_workers=resources.get("num_cpu_workers", 4),
                num_gpu_workers=resources.get("num_gpu_workers", 1),
                memory_limit_gb=resources.get("memory_limit_gb", 32),
            ),
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clear configuration cache."""
        cls._quality_config_cache = None
        cls._pipeline_config_cache = None

    @classmethod
    def get_physics_profile(cls, profile_name: str) -> Dict[str, Any]:
        """Get a physics profile by name.

        Args:
            profile_name: One of: manipulation_contact_rich, standard, navigation,
                         vision_only, deformable

        Returns:
            Physics profile configuration dict
        """
        config = cls._load_json(PIPELINE_CONFIG_PATH)
        profiles = config.get("physics_profiles", {})

        if profile_name not in profiles:
            raise ValueError(
                f"Unknown physics profile: {profile_name}. "
                f"Available profiles: {list(profiles.keys())}"
            )

        return profiles[profile_name]


# Convenience functions
def load_quality_config(**kwargs) -> QualityConfig:
    """Load quality gate configuration."""
    return ConfigLoader.load_quality_config(**kwargs)


def load_pipeline_config(**kwargs) -> PipelineConfig:
    """Load pipeline configuration."""
    return ConfigLoader.load_pipeline_config(**kwargs)


def get_physics_profile(profile_name: str) -> Dict[str, Any]:
    """Get a physics profile by name."""
    return ConfigLoader.get_physics_profile(profile_name)
