"""Pipeline Configuration Module.

Provides centralized configuration management for the BlueprintPipeline.
Labs can customize thresholds, parameters, and behavior by modifying
the configuration files or providing overrides via environment variables.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .constants import (
    DEFAULT_AVERAGE_QUALITY_SCORE_MIN,
    DEFAULT_COLLISION_FREE_RATE_MIN,
    DEFAULT_INVENTORY_MIN_OBJECTS,
    DEFAULT_INTERACTIVE_FALLBACK_RATE_MAX,
    DEFAULT_INTERACTIVE_OK_RATE_MIN,
    DEFAULT_INTERACTIVE_STABILITY_RATE_MIN,
    DEFAULT_PHYSICS_VALIDATION_RATE_MIN,
    DEFAULT_QUALITY_SCORE_MIN,
    DEFAULT_SCALE_CONFIDENCE_MIN,
    DEFAULT_SCALE_FACTOR_MAX,
    DEFAULT_SCALE_FACTOR_MIN,
    DEFAULT_SENSOR_CAPTURE_RATE_MIN,
    DEFAULT_VARIATION_3D_SUCCESS_RATE_MIN,
    DEFAULT_VARIATION_MIN_OBJECTS,
)
from .production_mode import ensure_config_audit_for_production, resolve_production_mode

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency in constrained envs
    yaml = None  # type: ignore[assignment]

try:
    from pydantic import BaseModel, ConfigDict, ValidationError, validator
    HAVE_PYDANTIC = True
except ImportError:
    HAVE_PYDANTIC = False
    ValidationError = Exception  # type: ignore

logger = logging.getLogger(__name__)

_ENABLE_CONFIG_AUDIT = ensure_config_audit_for_production()


# Configuration file paths
CONFIG_DIR = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIG_PATH = CONFIG_DIR / "pipeline_config.json"
QUALITY_CONFIG_PATH = CONFIG_DIR.parent / "quality_gates" / "quality_config.json"
ADAPTIVE_TIMEOUTS_PATH = REPO_ROOT / "policy_configs" / "adaptive_timeouts.yaml"


def _normalize_robot_name(robot_name: str) -> str:
    return str(robot_name).strip().lower().replace("-", "_")


def _load_geniesim_robot_allowlist(catalog_path: Path) -> List[str]:
    with open(catalog_path, "r") as handle:
        payload = json.load(handle)
    robots = payload.get("robots", [])
    allowlist: List[str] = []
    for entry in robots:
        if isinstance(entry, str):
            allowlist.append(_normalize_robot_name(entry))
        elif isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str) and name.strip():
                allowlist.append(_normalize_robot_name(name))
    return sorted(set(allowlist))


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
    collision_free_rate_min: float = DEFAULT_COLLISION_FREE_RATE_MIN
    quality_score_min: float = DEFAULT_QUALITY_SCORE_MIN
    quality_pass_rate_min: float = 0.70
    min_episodes_required: int = 3
    tier_thresholds: Dict[str, Dict[str, Union[float, int]]] = field(default_factory=dict)


@dataclass
class DataQualityThresholds:
    """Data quality SLI thresholds."""
    min_average_quality_score: float = DEFAULT_AVERAGE_QUALITY_SCORE_MIN
    min_sensor_capture_rate: float = DEFAULT_SENSOR_CAPTURE_RATE_MIN
    min_physics_validation_rate: float = DEFAULT_PHYSICS_VALIDATION_RATE_MIN
    allowed_sensor_sources: List[str] = field(
        default_factory=lambda: ["isaac_sim_replicator", "simulation"]
    )
    allowed_physics_backends: List[str] = field(
        default_factory=lambda: ["isaac_sim", "isaac_lab"]
    )


@dataclass
class SimulationThresholds:
    """Simulation stability thresholds."""
    min_stable_steps: int = 20
    max_penetration_depth_m: float = 0.005
    physics_stability_timeout_s: float = 30.0


@dataclass
class UsdThresholds:
    """USD validation thresholds."""
    max_usd_size_bytes: int = 500_000_000
    max_broken_references: int = 0
    require_physics_scene: bool = True
    require_header_validation: bool = True


@dataclass
class ReplicatorThresholds:
    """Replicator bundle validation thresholds."""
    required_sensor_fields: Dict[str, List[str]] = field(default_factory=lambda: {
        "camera_list": ["cameras", "camera_list"],
        "resolution": ["resolution"],
        "modalities": ["modalities", "annotations"],
        "stream_ids": ["stream_ids", "streams"],
    })


@dataclass
class EpisodeMetadataThresholds:
    """Episode metadata validation thresholds."""
    required_fields: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "dataset_name": {"paths": ["dataset_name", "name"], "type": "string"},
        "scene_id": {"paths": ["scene_id", "scene.scene_id"], "type": "string"},
        "robot_type": {"paths": ["robot_type", "robot.type"], "type": "string"},
        "camera_specs": {
            "paths": ["camera_specs", "data_pack.cameras", "cameras"],
            "type": "array_or_object",
        },
        "fps": {"paths": ["fps"], "type": "number"},
        "action_space": {"paths": ["action_space", "action_space_info"], "type": "array_or_object"},
        "episode_stats": {"paths": ["episode_stats", "stats"], "type": "object"},
    })


@dataclass
class DwmThresholds:
    """DWM bundle validation thresholds."""
    required_files: List[str] = field(default_factory=lambda: [
        "manifest.json",
        "static_scene_video.mp4",
        "camera_trajectory.json",
        "metadata/scene_info.json",
        "metadata/prompt.txt",
    ])


@dataclass
class InventoryEnrichmentThresholds:
    """Inventory enrichment validation thresholds."""
    min_object_count: int = DEFAULT_INVENTORY_MIN_OBJECTS
    require_enrichment_metadata: bool = True
    allowed_statuses: List[str] = field(default_factory=lambda: ["success", "stub"])


@dataclass
class VariationAssetsThresholds:
    """Variation asset generation validation thresholds."""
    min_object_count: int = DEFAULT_VARIATION_MIN_OBJECTS
    min_3d_success_rate: float = DEFAULT_VARIATION_3D_SUCCESS_RATE_MIN


@dataclass
class ScaleThresholds:
    """Scale calibration validation thresholds."""
    min_scale_factor: float = DEFAULT_SCALE_FACTOR_MIN
    max_scale_factor: float = DEFAULT_SCALE_FACTOR_MAX
    min_confidence: float = DEFAULT_SCALE_CONFIDENCE_MIN


@dataclass
class InteractiveThresholds:
    """Interactive asset processing validation thresholds."""
    min_ok_rate: float = DEFAULT_INTERACTIVE_OK_RATE_MIN
    min_physics_stability_rate: float = DEFAULT_INTERACTIVE_STABILITY_RATE_MIN
    max_fallback_rate: float = DEFAULT_INTERACTIVE_FALLBACK_RATE_MAX


@dataclass
class GenieSimKinematicReachabilityThresholds:
    """Genie Sim kinematic reachability validation thresholds."""
    enabled: bool = True
    min_reachability_rate: float = 1.0
    max_unreachable_targets: int = 0
    check_place_targets: bool = True


@dataclass
class AdaptiveTimeoutConfig:
    """Adaptive timeout defaults from policy configuration."""
    default_timeout_seconds: float
    bundle_tier: Dict[str, float] = field(default_factory=dict)
    scene_complexity: Dict[str, float] = field(default_factory=dict)
    selected_timeout_seconds: float = 0.0
    selected_bundle_tier: Optional[str] = None
    selected_scene_complexity: Optional[str] = None


@dataclass
class GenieSimDataCollectionThresholds:
    """Genie Sim data collection quality thresholds."""
    min_collision_free_rate: float = 0.9
    min_task_success_rate: float = 0.8


@dataclass
class HumanApprovalConfig:
    """Human approval workflow configuration."""
    enabled: bool = True
    timeout_hours: float = 24.0
    # Auto-approve is only honored outside production and requires explicit non-prod allowance.
    auto_approve_on_timeout: bool = False
    allow_auto_approve_on_timeout_non_production: bool = False
    approval_methods: List[str] = field(default_factory=lambda: ["dashboard", "email", "api"])
    default_approvers: List[str] = field(default_factory=list)
    escalation_after_hours: float = 12.0
    escalation_contacts: List[str] = field(default_factory=list)
    notification_channels: List[str] = field(default_factory=list)


@dataclass
class NotificationsConfig:
    """Notification defaults for quality gate alerts."""
    email_recipients: List[str] = field(default_factory=list)
    sms_recipients: List[str] = field(default_factory=list)
    slack_webhook_url: Optional[str] = None
    notify_on_pass: bool = False
    notify_on_warning: bool = True
    notify_on_error: bool = True
    notify_on_approval_required: bool = True


@dataclass
class ApprovalStoreConfig:
    """Approval request storage configuration."""
    backend: str = "firestore"
    filesystem_path: str = "/var/lib/blueprintpipeline/approvals"
    firestore_collection: str = "quality_gate_approvals"
    migrate_from_filesystem: bool = False


@dataclass
class GateOverrideConfig:
    """Gate override configuration."""
    allow_manual_override: bool = True
    allow_override_in_production: bool = False
    override_requires_reason: bool = True
    override_log_retention_days: int = 90
    allowed_overriders: List[str] = field(default_factory=list)
    override_reason_schema: "OverrideReasonSchema" = field(default_factory=lambda: OverrideReasonSchema())


@dataclass
class OverrideReasonSchema:
    """Schema definition for structured override reasons."""
    required_fields: List[str] = field(default_factory=lambda: ["category", "ticket", "justification"])
    categories: List[str] = field(
        default_factory=lambda: ["data_gap", "tooling_failure", "known_issue", "customer_exception", "other"]
    )
    ticket_pattern: str = r"^(https?://|[A-Za-z][A-Za-z0-9._-]*-\d+)$"
    justification_min_length: int = 50


@dataclass
class QualityConfig:
    """Complete quality gate configuration."""
    physics: PhysicsThresholds = field(default_factory=PhysicsThresholds)
    episodes: EpisodeThresholds = field(default_factory=EpisodeThresholds)
    data_quality: DataQualityThresholds = field(default_factory=DataQualityThresholds)
    simulation: SimulationThresholds = field(default_factory=SimulationThresholds)
    usd: UsdThresholds = field(default_factory=UsdThresholds)
    replicator: ReplicatorThresholds = field(default_factory=ReplicatorThresholds)
    episode_metadata: EpisodeMetadataThresholds = field(default_factory=EpisodeMetadataThresholds)
    dwm: DwmThresholds = field(default_factory=DwmThresholds)
    inventory_enrichment: InventoryEnrichmentThresholds = field(default_factory=InventoryEnrichmentThresholds)
    variation_assets: VariationAssetsThresholds = field(default_factory=VariationAssetsThresholds)
    scale: ScaleThresholds = field(default_factory=ScaleThresholds)
    interactive: InteractiveThresholds = field(default_factory=InteractiveThresholds)
    geniesim_kinematic_reachability: GenieSimKinematicReachabilityThresholds = field(
        default_factory=GenieSimKinematicReachabilityThresholds
    )
    geniesim_data_collection: GenieSimDataCollectionThresholds = field(
        default_factory=GenieSimDataCollectionThresholds
    )
    human_approval: HumanApprovalConfig = field(default_factory=HumanApprovalConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    approval_store: ApprovalStoreConfig = field(default_factory=ApprovalStoreConfig)
    gate_overrides: GateOverrideConfig = field(default_factory=GateOverrideConfig)


VIDEO_PROFILE_RESOLUTIONS = {
    "low": {"width": 640, "height": 480},
    "medium": {"width": 960, "height": 540},
    "high": {"width": 1280, "height": 720},
}

DEFAULT_VIDEO_PROFILE = "high"


@dataclass
class VideoConfig:
    """Video capture configuration with profile-aware defaults."""
    profile: str = DEFAULT_VIDEO_PROFILE
    width: int = VIDEO_PROFILE_RESOLUTIONS[DEFAULT_VIDEO_PROFILE]["width"]
    height: int = VIDEO_PROFILE_RESOLUTIONS[DEFAULT_VIDEO_PROFILE]["height"]
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
class SceneGraphRelationInferenceConfig:
    """Relation inference thresholds for scene graph generation."""
    vertical_proximity_threshold: float = 0.05
    horizontal_proximity_threshold: float = 0.15
    alignment_angle_threshold: float = 5.0
    enable_physics_validation: bool = False
    physics_contact_depth_threshold: float = 0.001
    physics_containment_ratio_threshold: float = 0.8
    heuristic_confidence_scale: float = 0.6


@dataclass
class SceneGraphStreamingConfig:
    """Streaming configuration for large scene manifests."""
    batch_size: int = 100


@dataclass
class SceneGraphValidationConfig:
    """Validation configuration for scene graph conversion."""
    allow_unvalidated_input: bool = True
    require_physics_validation: bool = True


@dataclass
class SceneGraphConfig:
    """Scene graph conversion configuration."""
    relation_inference: SceneGraphRelationInferenceConfig = field(default_factory=SceneGraphRelationInferenceConfig)
    streaming: SceneGraphStreamingConfig = field(default_factory=SceneGraphStreamingConfig)
    validation: SceneGraphValidationConfig = field(default_factory=SceneGraphValidationConfig)


@dataclass
class ABTestingConfig:
    """A/B testing configuration."""
    enabled: bool = False
    split_ratios: Dict[str, float] = field(default_factory=lambda: {"A": 0.5, "B": 0.5})
    assignment_store_path: str = "./sim2real_experiments/ab_assignments.json"


@dataclass
class HealthChecksConfig:
    """Health check configuration defaults."""
    probe_timeout_s: float = 2.0


@dataclass
class ModelConfig:
    """LLM model configuration."""
    model_name: str
    default_model: str = "gemini-3-flash-preview"
    alternatives: List[str] = field(default_factory=list)
    timeout_seconds: int = 30


@dataclass
class ModelsConfig:
    """Complete models configuration."""
    placement_engine: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_name="placement_engine",
        default_model="gemini-3-flash-preview",
        alternatives=["gemini-3-flash", "gpt-4", "claude-3-opus"],
        timeout_seconds=30
    ))
    physics_validator: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_name="physics_validator",
        default_model="gemini-3-flash-preview",
        alternatives=["gemini-3-flash", "gpt-4", "claude-3-opus"],
        timeout_seconds=30
    ))
    intelligent_region_detector: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_name="intelligent_region_detector",
        default_model="gemini-3-flash-preview",
        alternatives=["gemini-3-flash", "gpt-4", "claude-3-opus"],
        timeout_seconds=30
    ))

    def get_model(self, component_name: str) -> Optional[ModelConfig]:
        """Get model config for a component."""
        return getattr(self, component_name, None)


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
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)
    ab_testing: ABTestingConfig = field(default_factory=ABTestingConfig)
    health_checks: HealthChecksConfig = field(default_factory=HealthChecksConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)


# =============================================================================
# Pydantic Validation Models
# =============================================================================

@dataclass
class ConfigSource:
    """Tracks the source of a configuration value."""
    key: str  # Config key path (e.g., "episode_generation.episodes_per_task")
    value: Any  # The actual value
    source: str  # "default", "json_file", "environment", "override", "cache"
    source_file: Optional[str] = None  # File path if loaded from JSON
    env_var: Optional[str] = None  # Environment variable name if from env
    timestamp: Optional[str] = None  # When this value was loaded


@dataclass
class ConfigAuditTrail:
    """Audit trail for configuration values."""
    config_type: str  # "pipeline" or "quality"
    timestamp: str
    sources: List[ConfigSource] = field(default_factory=list)

    def add_source(
        self,
        key: str,
        value: Any,
        source: str,
        source_file: Optional[str] = None,
        env_var: Optional[str] = None,
    ) -> None:
        """Add a configuration source to the trail."""
        from datetime import datetime
        self.sources.append(
            ConfigSource(
                key=key,
                value=value,
                source=source,
                source_file=source_file,
                env_var=env_var,
                timestamp=datetime.now().isoformat(),
            )
        )

    def get_sources_for_key(self, key: str) -> List[ConfigSource]:
        """Get all sources for a specific config key."""
        return [s for s in self.sources if s.key.startswith(key)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/reporting."""
        return {
            "config_type": self.config_type,
            "timestamp": self.timestamp,
            "sources": [
                {
                    "key": s.key,
                    "value": str(s.value),
                    "source": s.source,
                    "source_file": s.source_file,
                    "env_var": s.env_var,
                    "timestamp": s.timestamp,
                }
                for s in self.sources
            ],
        }


if HAVE_PYDANTIC:
    class ValidationReport(BaseModel):
        """Report from config validation."""
        is_valid: bool
        errors: List[str] = field(default_factory=list)
        warnings: List[str] = field(default_factory=list)

        model_config = ConfigDict(arbitrary_types_allowed=True)


# =============================================================================
# ConfigLoader
# =============================================================================


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
    _quality_audit_trail: Optional[ConfigAuditTrail] = None
    _pipeline_audit_trail: Optional[ConfigAuditTrail] = None
    _enable_audit_trail: bool = _ENABLE_CONFIG_AUDIT

    @classmethod
    def _load_json(cls, path: Path, validate: bool = True, config_type: str = "pipeline") -> Dict[str, Any]:
        """
        Load JSON configuration file with optional validation.

        Args:
            path: Path to JSON config file
            validate: Whether to validate the config
            config_type: Type of config for validation ("pipeline" or "quality")

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If validation fails
            json.JSONDecodeError: If JSON is invalid
        """
        if not path.exists():
            return {}

        try:
            with open(path) as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}")

        # Validate if requested
        if validate:
            errors = cls.validate_config(config, config_type)
            if errors:
                error_msg = "\n".join(f"  {key}: {msg}" for key, msg in errors.items())
                raise ValueError(f"Configuration validation failed in {path}:\n{error_msg}")

        return config

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
    def _resolve_quality_default_constants(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Replace placeholder strings with shared default constants."""
        replacements = {
            "DEFAULT_COLLISION_FREE_RATE_MIN": DEFAULT_COLLISION_FREE_RATE_MIN,
            "DEFAULT_QUALITY_SCORE_MIN": DEFAULT_QUALITY_SCORE_MIN,
            "DEFAULT_AVERAGE_QUALITY_SCORE_MIN": DEFAULT_AVERAGE_QUALITY_SCORE_MIN,
            "DEFAULT_SENSOR_CAPTURE_RATE_MIN": DEFAULT_SENSOR_CAPTURE_RATE_MIN,
            "DEFAULT_PHYSICS_VALIDATION_RATE_MIN": DEFAULT_PHYSICS_VALIDATION_RATE_MIN,
            "DEFAULT_INVENTORY_MIN_OBJECTS": DEFAULT_INVENTORY_MIN_OBJECTS,
            "DEFAULT_VARIATION_MIN_OBJECTS": DEFAULT_VARIATION_MIN_OBJECTS,
            "DEFAULT_VARIATION_3D_SUCCESS_RATE_MIN": DEFAULT_VARIATION_3D_SUCCESS_RATE_MIN,
            "DEFAULT_SCALE_FACTOR_MIN": DEFAULT_SCALE_FACTOR_MIN,
            "DEFAULT_SCALE_FACTOR_MAX": DEFAULT_SCALE_FACTOR_MAX,
            "DEFAULT_SCALE_CONFIDENCE_MIN": DEFAULT_SCALE_CONFIDENCE_MIN,
            "DEFAULT_INTERACTIVE_OK_RATE_MIN": DEFAULT_INTERACTIVE_OK_RATE_MIN,
            "DEFAULT_INTERACTIVE_STABILITY_RATE_MIN": DEFAULT_INTERACTIVE_STABILITY_RATE_MIN,
            "DEFAULT_INTERACTIVE_FALLBACK_RATE_MAX": DEFAULT_INTERACTIVE_FALLBACK_RATE_MAX,
        }

        def _resolve(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: _resolve(val) for key, val in value.items()}
            if isinstance(value, list):
                return [_resolve(item) for item in value]
            if isinstance(value, str) and value in replacements:
                return replacements[value]
            return value

        return _resolve(config)

    @classmethod
    def _warn_on_quality_threshold_conflicts(cls, config: Dict[str, Any]) -> None:
        thresholds = config.get("thresholds", {})
        if not isinstance(thresholds, dict):
            return
        geniesim_ik = thresholds.get("geniesim_kinematic_reachability", {})
        if not isinstance(geniesim_ik, dict):
            return
        if geniesim_ik.get("enabled", True) is False:
            return
        min_rate = geniesim_ik.get("min_reachability_rate")
        max_unreachable = geniesim_ik.get("max_unreachable_targets")
        if (
            isinstance(min_rate, (int, float))
            and isinstance(max_unreachable, int)
            and max_unreachable == 0
            and min_rate < 1.0
        ):
            logger.warning(
                "Genie Sim reachability thresholds are contradictory: min_reachability_rate=%.3f "
                "while max_unreachable_targets=0. Raise min_reachability_rate to 1.0 or allow "
                "some unreachable targets.",
                min_rate,
            )

    @classmethod
    def load_quality_config(
        cls,
        config_path: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        validate: bool = True,
    ) -> QualityConfig:
        """Load quality gate configuration.

        Args:
            config_path: Optional custom config file path
            overrides: Optional dict of overrides to apply
            use_cache: Whether to use cached config
            validate: Whether to validate the configuration

        Returns:
            QualityConfig dataclass with all thresholds

        Raises:
            ValueError: If validation is enabled and config is invalid
        """
        # Initialize audit trail if enabled
        if cls._enable_audit_trail and not overrides:
            cls._quality_audit_trail = cls._init_audit_trail("quality")

        # Check cache
        if use_cache and cls._quality_config_cache is not None and not overrides:
            config = cls._quality_config_cache
            if cls._quality_audit_trail:
                cls._quality_audit_trail.add_source(
                    "config",
                    "cached_config",
                    "cache",
                    source_file=None,
                )
        else:
            path = config_path or QUALITY_CONFIG_PATH
            config = cls._load_json(path, validate=False, config_type="quality")
            config = cls._resolve_quality_default_constants(config)
            if validate:
                errors = cls.validate_config(config, "quality")
                if errors:
                    error_msg = "\n".join(f"  {key}: {msg}" for key, msg in errors.items())
                    raise ValueError(f"Configuration validation failed in {path}:\n{error_msg}")

            if cls._quality_audit_trail:
                cls._quality_audit_trail.add_source(
                    "config_file",
                    str(path),
                    "json_file",
                    source_file=str(path),
                )

            config = cls._apply_env_overrides(config, "BP_QUALITY_")

            if cls._quality_audit_trail:
                for key, value in os.environ.items():
                    if key.startswith("BP_QUALITY_"):
                        cls._quality_audit_trail.add_source(
                            key,
                            value,
                            "environment",
                            env_var=key,
                        )

            if overrides:
                config = cls._deep_merge(config, overrides)
                if cls._quality_audit_trail:
                    cls._quality_audit_trail.add_source(
                        "overrides",
                        str(overrides),
                        "override",
                    )

            if "approval" in config and isinstance(config["approval"], dict):
                store_override = config["approval"].get("store")
                if isinstance(store_override, dict):
                    approval_store = config.get("approval_store", {})
                    if not isinstance(approval_store, dict):
                        approval_store = {}
                    config["approval_store"] = cls._deep_merge(approval_store, store_override)

            if use_cache and not overrides:
                cls._quality_config_cache = config

        cls._warn_on_quality_threshold_conflicts(config)

        # Parse into dataclass
        thresholds = config.get("thresholds", {})
        data_quality = thresholds.get("data_quality", {})
        usd_thresholds = thresholds.get("usd", {})
        replicator_thresholds = thresholds.get("replicator", {})
        episode_metadata_thresholds = thresholds.get("episode_metadata", {})
        dwm_thresholds = thresholds.get("dwm", {})
        inventory_thresholds = thresholds.get("inventory_enrichment", {})
        variation_thresholds = thresholds.get("variation_assets", {})
        scale_thresholds = thresholds.get("scale", {})
        interactive_thresholds = thresholds.get("interactive", {})
        geniesim_ik_thresholds = thresholds.get("geniesim_kinematic_reachability", {})
        geniesim_data_thresholds = thresholds.get("geniesim_data_collection", {})

        return QualityConfig(
            physics=PhysicsThresholds(
                mass_min_kg=thresholds.get("physics", {}).get("mass_min_kg", 0.01),
                mass_max_kg=thresholds.get("physics", {}).get("mass_max_kg", 500.0),
                friction_min=thresholds.get("physics", {}).get("friction_min", 0.0),
                friction_max=thresholds.get("physics", {}).get("friction_max", 2.0),
            ),
            episodes=EpisodeThresholds(
                collision_free_rate_min=thresholds.get("episodes", {}).get(
                    "collision_free_rate_min",
                    DEFAULT_COLLISION_FREE_RATE_MIN,
                ),
                quality_score_min=thresholds.get("episodes", {}).get(
                    "quality_score_min",
                    DEFAULT_QUALITY_SCORE_MIN,
                ),
                quality_pass_rate_min=thresholds.get("episodes", {}).get("quality_pass_rate_min", 0.70),
                min_episodes_required=thresholds.get("episodes", {}).get("min_episodes_required", 3),
                tier_thresholds=thresholds.get("episodes", {}).get("tier_thresholds", {}),
            ),
            data_quality=DataQualityThresholds(
                min_average_quality_score=data_quality.get(
                    "min_average_quality_score",
                    DEFAULT_AVERAGE_QUALITY_SCORE_MIN,
                ),
                min_sensor_capture_rate=data_quality.get(
                    "min_sensor_capture_rate",
                    DEFAULT_SENSOR_CAPTURE_RATE_MIN,
                ),
                min_physics_validation_rate=data_quality.get(
                    "min_physics_validation_rate",
                    DEFAULT_PHYSICS_VALIDATION_RATE_MIN,
                ),
                allowed_sensor_sources=data_quality.get(
                    "allowed_sensor_sources",
                    ["isaac_sim_replicator", "simulation"],
                ),
                allowed_physics_backends=data_quality.get(
                    "allowed_physics_backends",
                    ["isaac_sim", "isaac_lab"],
                ),
            ),
            simulation=SimulationThresholds(
                min_stable_steps=thresholds.get("simulation", {}).get("min_stable_steps", 20),
                max_penetration_depth_m=thresholds.get("simulation", {}).get("max_penetration_depth_m", 0.005),
                physics_stability_timeout_s=thresholds.get("simulation", {}).get("physics_stability_timeout_s", 30.0),
            ),
            usd=UsdThresholds(
                max_usd_size_bytes=usd_thresholds.get("max_usd_size_bytes", 500_000_000),
                max_broken_references=usd_thresholds.get("max_broken_references", 0),
                require_physics_scene=usd_thresholds.get("require_physics_scene", True),
                require_header_validation=usd_thresholds.get("require_header_validation", True),
            ),
            replicator=ReplicatorThresholds(
                required_sensor_fields=replicator_thresholds.get(
                    "required_sensor_fields",
                    {
                        "camera_list": ["cameras", "camera_list"],
                        "resolution": ["resolution"],
                        "modalities": ["modalities", "annotations"],
                        "stream_ids": ["stream_ids", "streams"],
                    },
                ),
            ),
            episode_metadata=EpisodeMetadataThresholds(
                required_fields=episode_metadata_thresholds.get(
                    "required_fields",
                    {
                        "dataset_name": {"paths": ["dataset_name", "name"], "type": "string"},
                        "scene_id": {"paths": ["scene_id", "scene.scene_id"], "type": "string"},
                        "robot_type": {"paths": ["robot_type", "robot.type"], "type": "string"},
                        "camera_specs": {
                            "paths": ["camera_specs", "data_pack.cameras", "cameras"],
                            "type": "array_or_object",
                        },
                        "fps": {"paths": ["fps"], "type": "number"},
                        "action_space": {"paths": ["action_space", "action_space_info"], "type": "array_or_object"},
                        "episode_stats": {"paths": ["episode_stats", "stats"], "type": "object"},
                    },
                ),
            ),
            dwm=DwmThresholds(
                required_files=dwm_thresholds.get(
                    "required_files",
                    [
                        "manifest.json",
                        "static_scene_video.mp4",
                        "camera_trajectory.json",
                        "metadata/scene_info.json",
                        "metadata/prompt.txt",
                    ],
                ),
            ),
            inventory_enrichment=InventoryEnrichmentThresholds(
                min_object_count=inventory_thresholds.get(
                    "min_object_count",
                    DEFAULT_INVENTORY_MIN_OBJECTS,
                ),
                require_enrichment_metadata=inventory_thresholds.get(
                    "require_enrichment_metadata",
                    True,
                ),
                allowed_statuses=inventory_thresholds.get(
                    "allowed_statuses",
                    ["success", "stub"],
                ),
            ),
            variation_assets=VariationAssetsThresholds(
                min_object_count=variation_thresholds.get(
                    "min_object_count",
                    DEFAULT_VARIATION_MIN_OBJECTS,
                ),
                min_3d_success_rate=variation_thresholds.get(
                    "min_3d_success_rate",
                    DEFAULT_VARIATION_3D_SUCCESS_RATE_MIN,
                ),
            ),
            scale=ScaleThresholds(
                min_scale_factor=scale_thresholds.get(
                    "min_scale_factor",
                    DEFAULT_SCALE_FACTOR_MIN,
                ),
                max_scale_factor=scale_thresholds.get(
                    "max_scale_factor",
                    DEFAULT_SCALE_FACTOR_MAX,
                ),
                min_confidence=scale_thresholds.get(
                    "min_confidence",
                    DEFAULT_SCALE_CONFIDENCE_MIN,
                ),
            ),
            interactive=InteractiveThresholds(
                min_ok_rate=interactive_thresholds.get(
                    "min_ok_rate",
                    DEFAULT_INTERACTIVE_OK_RATE_MIN,
                ),
                min_physics_stability_rate=interactive_thresholds.get(
                    "min_physics_stability_rate",
                    DEFAULT_INTERACTIVE_STABILITY_RATE_MIN,
                ),
                max_fallback_rate=interactive_thresholds.get(
                    "max_fallback_rate",
                    DEFAULT_INTERACTIVE_FALLBACK_RATE_MAX,
                ),
            ),
            geniesim_kinematic_reachability=GenieSimKinematicReachabilityThresholds(
                enabled=geniesim_ik_thresholds.get("enabled", True),
                min_reachability_rate=geniesim_ik_thresholds.get("min_reachability_rate", 1.0),
                max_unreachable_targets=geniesim_ik_thresholds.get("max_unreachable_targets", 0),
                check_place_targets=geniesim_ik_thresholds.get("check_place_targets", True),
            ),
            geniesim_data_collection=GenieSimDataCollectionThresholds(
                min_collision_free_rate=geniesim_data_thresholds.get(
                    "min_collision_free_rate",
                    0.9,
                ),
                min_task_success_rate=geniesim_data_thresholds.get(
                    "min_task_success_rate",
                    0.8,
                ),
            ),
            human_approval=HumanApprovalConfig(
                enabled=config.get("human_approval", {}).get("enabled", True),
                timeout_hours=config.get("human_approval", {}).get("timeout_hours", 24.0),
                auto_approve_on_timeout=config.get("human_approval", {}).get("auto_approve_on_timeout", False),
                allow_auto_approve_on_timeout_non_production=config.get("human_approval", {}).get(
                    "allow_auto_approve_on_timeout_non_production",
                    False,
                ),
                approval_methods=config.get("human_approval", {}).get("approval_methods", ["dashboard", "email", "api"]),
                notification_channels=config.get("human_approval", {}).get("notification_channels", []),
            ),
            notifications=NotificationsConfig(
                email_recipients=config.get("notifications", {}).get("email_recipients", []),
                sms_recipients=config.get("notifications", {}).get("sms_recipients", []),
                slack_webhook_url=config.get("notifications", {}).get("slack_webhook_url"),
                notify_on_pass=config.get("notifications", {}).get("notify_on_pass", False),
                notify_on_warning=config.get("notifications", {}).get("notify_on_warning", True),
                notify_on_error=config.get("notifications", {}).get("notify_on_error", True),
                notify_on_approval_required=config.get("notifications", {}).get(
                    "notify_on_approval_required",
                    True,
                ),
            ),
            approval_store=ApprovalStoreConfig(
                backend=config.get("approval_store", {}).get("backend", "filesystem"),
                filesystem_path=config.get("approval_store", {}).get(
                    "filesystem_path",
                    ApprovalStoreConfig().filesystem_path,
                ),
                firestore_collection=config.get("approval_store", {}).get(
                    "firestore_collection",
                    "quality_gate_approvals",
                ),
                migrate_from_filesystem=config.get("approval_store", {}).get(
                    "migrate_from_filesystem",
                    False,
                ),
            ),
            gate_overrides=GateOverrideConfig(
                allow_manual_override=config.get("gate_overrides", {}).get("allow_manual_override", True),
                allow_override_in_production=config.get(
                    "gate_overrides",
                    {},
                ).get("allow_override_in_production", False),
                override_requires_reason=config.get("gate_overrides", {}).get("override_requires_reason", True),
                override_log_retention_days=config.get("gate_overrides", {}).get("override_log_retention_days", 90),
                allowed_overriders=config.get("gate_overrides", {}).get("allowed_overriders", []),
                override_reason_schema=OverrideReasonSchema(
                    required_fields=config.get("gate_overrides", {}).get("override_reason_schema", {}).get(
                        "required_fields",
                        ["category", "ticket", "justification"],
                    ),
                    categories=config.get("gate_overrides", {}).get("override_reason_schema", {}).get(
                        "categories",
                        ["data_gap", "tooling_failure", "known_issue", "customer_exception", "other"],
                    ),
                    ticket_pattern=config.get("gate_overrides", {}).get("override_reason_schema", {}).get(
                        "ticket_pattern",
                        r"^(https?://|[A-Za-z][A-Za-z0-9._-]*-\d+)$",
                    ),
                    justification_min_length=config.get("gate_overrides", {}).get("override_reason_schema", {}).get(
                        "justification_min_length",
                        50,
                    ),
                ),
            ),
        )

    @classmethod
    def load_pipeline_config(
        cls,
        config_path: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        validate: bool = True,
    ) -> PipelineConfig:
        """Load pipeline configuration.

        Args:
            config_path: Optional custom config file path
            overrides: Optional dict of overrides to apply
            use_cache: Whether to use cached config
            validate: Whether to validate the configuration

        Returns:
            PipelineConfig dataclass with all settings

        Raises:
            ValueError: If validation is enabled and config is invalid
        """
        # Initialize audit trail if enabled
        if cls._enable_audit_trail and not overrides:
            cls._pipeline_audit_trail = cls._init_audit_trail("pipeline")

        if use_cache and cls._pipeline_config_cache is not None and not overrides:
            config = cls._pipeline_config_cache
            if cls._pipeline_audit_trail:
                cls._pipeline_audit_trail.add_source(
                    "config",
                    "cached_config",
                    "cache",
                    source_file=None,
                )
        else:
            path = config_path or PIPELINE_CONFIG_PATH
            config = cls._load_json(path, validate=validate, config_type="pipeline")

            if cls._pipeline_audit_trail:
                cls._pipeline_audit_trail.add_source(
                    "config_file",
                    str(path),
                    "json_file",
                    source_file=str(path),
                )

            config = cls._apply_env_overrides(config, "BP_PIPELINE_")

            if cls._pipeline_audit_trail:
                for key, value in os.environ.items():
                    if key.startswith("BP_PIPELINE_"):
                        cls._pipeline_audit_trail.add_source(
                            key,
                            value,
                            "environment",
                            env_var=key,
                        )

            if overrides:
                config = cls._deep_merge(config, overrides)
                if cls._pipeline_audit_trail:
                    cls._pipeline_audit_trail.add_source(
                        "overrides",
                        str(overrides),
                        "override",
                    )

            if use_cache and not overrides:
                cls._pipeline_config_cache = config

        # Parse into dataclass
        ep_gen = config.get("episode_generation", {})
        video = config.get("video", {})
        physics = config.get("physics", {})
        dr = config.get("domain_randomization", {})
        reward = config.get("reward_shaping", {})
        resources = config.get("resource_allocation", {})
        scene_graph = config.get("scene_graph", {})
        ab_testing = config.get("ab_testing", {})
        health_checks = config.get("health_checks", {})
        models_cfg = config.get("models", {})

        # Parse models configuration
        models = ModelsConfig(
            placement_engine=ModelConfig(
                model_name="placement_engine",
                default_model=models_cfg.get("placement_engine", {}).get("default_model", "gemini-3-flash-preview"),
                alternatives=models_cfg.get("placement_engine", {}).get("alternatives", []),
                timeout_seconds=models_cfg.get("placement_engine", {}).get("timeout_seconds", 30),
            ),
            physics_validator=ModelConfig(
                model_name="physics_validator",
                default_model=models_cfg.get("physics_validator", {}).get("default_model", "gemini-3-flash-preview"),
                alternatives=models_cfg.get("physics_validator", {}).get("alternatives", []),
                timeout_seconds=models_cfg.get("physics_validator", {}).get("timeout_seconds", 30),
            ),
            intelligent_region_detector=ModelConfig(
                model_name="intelligent_region_detector",
                default_model=models_cfg.get("intelligent_region_detector", {}).get("default_model", "gemini-3-flash-preview"),
                alternatives=models_cfg.get("intelligent_region_detector", {}).get("alternatives", []),
                timeout_seconds=models_cfg.get("intelligent_region_detector", {}).get("timeout_seconds", 30),
            ),
        )

        video_profile_value = os.getenv("BP_VIDEO_PROFILE", video.get("profile", DEFAULT_VIDEO_PROFILE))
        video_profile = (
            video_profile_value.lower()
            if isinstance(video_profile_value, str)
            else DEFAULT_VIDEO_PROFILE
        )
        profile_resolution = VIDEO_PROFILE_RESOLUTIONS.get(video_profile)
        if profile_resolution is None:
            logger.warning(
                "Unknown video profile '%s'; falling back to '%s'.",
                video_profile,
                DEFAULT_VIDEO_PROFILE,
            )
            video_profile = DEFAULT_VIDEO_PROFILE
            profile_resolution = VIDEO_PROFILE_RESOLUTIONS[video_profile]

        resolution = video.get("resolution", {})
        width = (
            resolution.get("width")
            if isinstance(resolution, dict) and resolution.get("width") is not None
            else profile_resolution["width"]
        )
        height = (
            resolution.get("height")
            if isinstance(resolution, dict) and resolution.get("height") is not None
            else profile_resolution["height"]
        )

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
                profile=video_profile,
                width=width,
                height=height,
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
            scene_graph=SceneGraphConfig(
                relation_inference=SceneGraphRelationInferenceConfig(
                    vertical_proximity_threshold=scene_graph.get("relation_inference", {}).get(
                        "vertical_proximity_threshold",
                        0.05,
                    ),
                    horizontal_proximity_threshold=scene_graph.get("relation_inference", {}).get(
                        "horizontal_proximity_threshold",
                        0.15,
                    ),
                    alignment_angle_threshold=scene_graph.get("relation_inference", {}).get(
                        "alignment_angle_threshold",
                        5.0,
                    ),
                    enable_physics_validation=scene_graph.get("relation_inference", {}).get(
                        "enable_physics_validation",
                        False,
                    ),
                    physics_contact_depth_threshold=scene_graph.get("relation_inference", {}).get(
                        "physics_contact_depth_threshold",
                        0.001,
                    ),
                    physics_containment_ratio_threshold=scene_graph.get("relation_inference", {}).get(
                        "physics_containment_ratio_threshold",
                        0.8,
                    ),
                    heuristic_confidence_scale=scene_graph.get("relation_inference", {}).get(
                        "heuristic_confidence_scale",
                        0.6,
                    ),
                ),
                streaming=SceneGraphStreamingConfig(
                    batch_size=scene_graph.get("streaming", {}).get("batch_size", 100),
                ),
                validation=SceneGraphValidationConfig(
                    allow_unvalidated_input=scene_graph.get("validation", {}).get(
                        "allow_unvalidated_input",
                        True,
                    ),
                    require_physics_validation=scene_graph.get("validation", {}).get(
                        "require_physics_validation",
                        True,
                    ),
                ),
            ),
            ab_testing=ABTestingConfig(
                enabled=ab_testing.get("enabled", False),
                split_ratios=ab_testing.get("split_ratios", {"A": 0.5, "B": 0.5}),
                assignment_store_path=ab_testing.get(
                    "assignment_store_path",
                    "./sim2real_experiments/ab_assignments.json",
                ),
            ),
            health_checks=HealthChecksConfig(
                probe_timeout_s=health_checks.get("probe_timeout_s", 2.0),
            ),
            models=models,
        )

    @classmethod
    def _init_audit_trail(cls, config_type: str) -> ConfigAuditTrail:
        """Initialize a new audit trail."""
        from datetime import datetime
        return ConfigAuditTrail(
            config_type=config_type,
            timestamp=datetime.now().isoformat(),
        )

    @classmethod
    def get_audit_trail(cls, config_type: str = "pipeline") -> Optional[ConfigAuditTrail]:
        """Get the audit trail for a configuration.

        Args:
            config_type: "pipeline" or "quality"

        Returns:
            ConfigAuditTrail if audit trail is enabled, None otherwise
        """
        if not cls._enable_audit_trail:
            return None

        if config_type == "pipeline":
            return cls._pipeline_audit_trail
        elif config_type == "quality":
            return cls._quality_audit_trail
        return None

    @classmethod
    def dump_audit_trail(cls, config_type: str = "pipeline") -> Optional[str]:
        """Dump audit trail as JSON for logging.

        Args:
            config_type: "pipeline" or "quality"

        Returns:
            JSON string of audit trail, or None
        """
        trail = cls.get_audit_trail(config_type)
        if trail:
            return json.dumps(trail.to_dict(), indent=2)
        return None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear configuration cache and audit trails."""
        cls._quality_config_cache = None
        cls._pipeline_config_cache = None
        cls._quality_audit_trail = None
        cls._pipeline_audit_trail = None

    @classmethod
    def validate_config(cls, config: Dict[str, Any], config_type: str = "pipeline") -> Optional[Dict[str, str]]:
        """
        Validate configuration against schema.

        Args:
            config: Configuration dictionary to validate
            config_type: Type of config ("pipeline" or "quality")

        Returns:
            Dictionary of validation errors, or None if valid

        Example:
            errors = ConfigLoader.validate_config(config, "pipeline")
            if errors:
                for key, msg in errors.items():
                    print(f"Validation error in {key}: {msg}")
        """
        validation_errors = {}

        if config_type == "pipeline":
            validation_errors.update(cls._validate_pipeline_config(config))
        elif config_type == "quality":
            validation_errors.update(cls._validate_quality_config(config))
        else:
            validation_errors["config_type"] = f"Unknown config type: {config_type}"

        return validation_errors if validation_errors else None

    @classmethod
    def _validate_pipeline_config(cls, config: Dict[str, Any]) -> Dict[str, str]:
        """Validate pipeline configuration."""
        errors = {}

        # Validate episode_generation section
        if "episode_generation" in config:
            ep_gen = config["episode_generation"]
            if isinstance(ep_gen, dict):
                if "episodes_per_task" in ep_gen and not isinstance(ep_gen["episodes_per_task"], int):
                    errors["episode_generation.episodes_per_task"] = "Must be an integer"
                if "episode_timeout_seconds" in ep_gen and ep_gen["episode_timeout_seconds"] < 0:
                    errors["episode_generation.episode_timeout_seconds"] = "Must be non-negative"
                if "episodes_per_scene" in ep_gen:
                    eps = ep_gen["episodes_per_scene"]
                    if isinstance(eps, dict):
                        for tier, count in eps.items():
                            if not isinstance(count, int) or count < 0:
                                errors[f"episode_generation.episodes_per_scene.{tier}"] = "Must be non-negative integer"

        # Validate video section
        if "video" in config:
            video = config["video"]
            if isinstance(video, dict):
                if "resolution" in video and isinstance(video["resolution"], dict):
                    res = video["resolution"]
                    if res.get("width", 0) <= 0 or res.get("height", 0) <= 0:
                        errors["video.resolution"] = "Width and height must be positive"
                if "profile" in video:
                    profile = video["profile"]
                    if (
                        not isinstance(profile, str)
                        or profile.lower() not in VIDEO_PROFILE_RESOLUTIONS
                    ):
                        errors["video.profile"] = "Profile must be one of: low, medium, high"
                if "fps" in video and video["fps"] <= 0:
                    errors["video.fps"] = "FPS must be positive"

        # Validate physics section
        if "physics" in config:
            physics = config["physics"]
            if isinstance(physics, dict):
                if "timestep_hz" in physics and physics["timestep_hz"] <= 0:
                    errors["physics.timestep_hz"] = "Timestep must be positive"
                if "gravity" in physics:
                    gravity = physics["gravity"]
                    if not isinstance(gravity, list) or len(gravity) != 3:
                        errors["physics.gravity"] = "Must be a list of 3 floats [x, y, z]"

        # Validate robot_config section
        if "robot_config" in config:
            robot_config = config["robot_config"]
            if isinstance(robot_config, dict):
                catalog_path_value = robot_config.get("geniesim_robot_catalog_path")
                allowlist: Optional[List[str]] = None
                if catalog_path_value:
                    catalog_path = Path(str(catalog_path_value))
                    if not catalog_path.is_absolute():
                        catalog_path = (REPO_ROOT / catalog_path).resolve()
                    try:
                        allowlist = _load_geniesim_robot_allowlist(catalog_path)
                    except (OSError, json.JSONDecodeError) as exc:
                        errors["robot_config.geniesim_robot_catalog_path"] = (
                            f"Failed to load Genie Sim robot catalog: {exc}"
                        )
                supported_robots = robot_config.get("supported_robots")
                if allowlist and isinstance(supported_robots, list):
                    normalized_allowlist = set(allowlist)
                    unsupported = [
                        robot
                        for robot in supported_robots
                        if _normalize_robot_name(str(robot)) not in normalized_allowlist
                    ]
                    if unsupported:
                        errors["robot_config.supported_robots"] = (
                            "Unsupported Genie Sim robots configured: "
                            f"{sorted(set(unsupported))}"
                        )
                default_robot = robot_config.get("default_robot")
                if allowlist and isinstance(default_robot, str):
                    if _normalize_robot_name(default_robot) not in set(allowlist):
                        errors["robot_config.default_robot"] = (
                            f"Unsupported Genie Sim robot: {default_robot}"
                        )

        # Validate resource_allocation section
        if "resource_allocation" in config:
            resources = config["resource_allocation"]
            if isinstance(resources, dict):
                if "gpu_memory_fraction" in resources:
                    frac = resources["gpu_memory_fraction"]
                    if not (0.0 <= frac <= 1.0):
                        errors["resource_allocation.gpu_memory_fraction"] = "Must be between 0.0 and 1.0"
                if "memory_limit_gb" in resources and resources["memory_limit_gb"] <= 0:
                    errors["resource_allocation.memory_limit_gb"] = "Must be positive"

        if "scene_graph" in config:
            scene_graph = config["scene_graph"]
            if isinstance(scene_graph, dict):
                relation_inference = scene_graph.get("relation_inference", {})
                if isinstance(relation_inference, dict):
                    for key in [
                        "vertical_proximity_threshold",
                        "horizontal_proximity_threshold",
                        "alignment_angle_threshold",
                    ]:
                        if key in relation_inference and relation_inference[key] < 0:
                            errors[f"scene_graph.relation_inference.{key}"] = "Must be non-negative"
                streaming = scene_graph.get("streaming", {})
                if isinstance(streaming, dict):
                    if "batch_size" in streaming and streaming["batch_size"] <= 0:
                        errors["scene_graph.streaming.batch_size"] = "Must be positive"
                validation = scene_graph.get("validation", {})
                if isinstance(validation, dict):
                    if "allow_unvalidated_input" in validation and not isinstance(
                        validation["allow_unvalidated_input"],
                        bool,
                    ):
                        errors["scene_graph.validation.allow_unvalidated_input"] = "Must be a boolean"
                    if "require_physics_validation" in validation and not isinstance(
                        validation["require_physics_validation"],
                        bool,
                    ):
                        errors["scene_graph.validation.require_physics_validation"] = "Must be a boolean"

        if "health_checks" in config:
            health_checks = config["health_checks"]
            if isinstance(health_checks, dict):
                if "probe_timeout_s" in health_checks and health_checks["probe_timeout_s"] <= 0:
                    errors["health_checks.probe_timeout_s"] = "Must be positive"

        if "ab_testing" in config:
            ab_testing = config["ab_testing"]
            if isinstance(ab_testing, dict):
                split_ratios = ab_testing.get("split_ratios")
                if split_ratios is not None:
                    if not isinstance(split_ratios, dict) or not split_ratios:
                        errors["ab_testing.split_ratios"] = "Must be a non-empty mapping"
                    else:
                        total = 0.0
                        for variant, weight in split_ratios.items():
                            if not isinstance(weight, (int, float)):
                                errors[f"ab_testing.split_ratios.{variant}"] = "Must be a number"
                                continue
                            if weight < 0:
                                errors[f"ab_testing.split_ratios.{variant}"] = "Must be non-negative"
                                continue
                            total += float(weight)
                        if total <= 0:
                            errors["ab_testing.split_ratios"] = "Total split ratios must be positive"

        return errors

    @classmethod
    def _validate_quality_config(cls, config: Dict[str, Any]) -> Dict[str, str]:
        """Validate quality gate configuration."""
        errors = {}

        # Validate thresholds section
        if "thresholds" in config:
            thresholds = config["thresholds"]
            if isinstance(thresholds, dict):
                # Physics thresholds
                if "physics" in thresholds:
                    physics = thresholds["physics"]
                    if isinstance(physics, dict):
                        if "mass_min_kg" in physics and physics["mass_min_kg"] < 0:
                            errors["thresholds.physics.mass_min_kg"] = "Must be non-negative"
                        if "mass_max_kg" in physics and physics["mass_max_kg"] < 0:
                            errors["thresholds.physics.mass_max_kg"] = "Must be non-negative"

                # Episode thresholds
                if "episodes" in thresholds:
                    episodes = thresholds["episodes"]
                    if isinstance(episodes, dict):
                        for key in ["collision_free_rate_min", "quality_score_min", "quality_pass_rate_min"]:
                            if key in episodes:
                                val = episodes[key]
                                if not (0.0 <= val <= 1.0):
                                    errors[f"thresholds.episodes.{key}"] = "Must be between 0.0 and 1.0"
                        if "min_episodes_required" in episodes and episodes["min_episodes_required"] < 0:
                            errors["thresholds.episodes.min_episodes_required"] = "Must be non-negative"
                        tier_thresholds = episodes.get("tier_thresholds", {})
                        if isinstance(tier_thresholds, dict):
                            for tier_name, tier_values in tier_thresholds.items():
                                if not isinstance(tier_values, dict):
                                    errors[f"thresholds.episodes.tier_thresholds.{tier_name}"] = "Must be a mapping"
                                    continue
                                for key in ["collision_free_rate_min", "quality_score_min", "quality_pass_rate_min"]:
                                    if key in tier_values:
                                        val = tier_values[key]
                                        if not (0.0 <= val <= 1.0):
                                            errors[
                                                f"thresholds.episodes.tier_thresholds.{tier_name}.{key}"
                                            ] = "Must be between 0.0 and 1.0"
                                if (
                                    "min_episodes_required" in tier_values
                                    and tier_values["min_episodes_required"] < 0
                                ):
                                    errors[
                                        f"thresholds.episodes.tier_thresholds.{tier_name}.min_episodes_required"
                                    ] = "Must be non-negative"

                        if resolve_production_mode():
                            production_floor = {
                                "collision_free_rate_min": 0.90,
                                "quality_pass_rate_min": 0.75,
                                "quality_score_min": 0.92,
                                "min_episodes_required": 5,
                            }
                            for key, floor_value in production_floor.items():
                                if key in episodes and episodes[key] < floor_value:
                                    errors[f"thresholds.episodes.{key}"] = (
                                        f"Must be >= {floor_value} in production"
                                    )

                # Data quality thresholds
                if "data_quality" in thresholds:
                    data_quality = thresholds["data_quality"]
                    if isinstance(data_quality, dict):
                        for key in [
                            "min_average_quality_score",
                            "min_sensor_capture_rate",
                            "min_physics_validation_rate",
                        ]:
                            if key in data_quality:
                                val = data_quality[key]
                                if not (0.0 <= val <= 1.0):
                                    errors[f"thresholds.data_quality.{key}"] = "Must be between 0.0 and 1.0"

                # Genie Sim kinematic reachability thresholds
                if "geniesim_kinematic_reachability" in thresholds:
                    geniesim_ik = thresholds["geniesim_kinematic_reachability"]
                    if isinstance(geniesim_ik, dict):
                        if "min_reachability_rate" in geniesim_ik:
                            val = geniesim_ik["min_reachability_rate"]
                            if not (0.0 <= val <= 1.0):
                                errors[
                                    "thresholds.geniesim_kinematic_reachability.min_reachability_rate"
                                ] = "Must be between 0.0 and 1.0"
                        if "max_unreachable_targets" in geniesim_ik:
                            val = geniesim_ik["max_unreachable_targets"]
                            if not isinstance(val, int) or val < 0:
                                errors[
                                    "thresholds.geniesim_kinematic_reachability.max_unreachable_targets"
                                ] = "Must be a non-negative integer"

                if "inventory_enrichment" in thresholds:
                    inventory = thresholds["inventory_enrichment"]
                    if isinstance(inventory, dict):
                        if "min_object_count" in inventory and inventory["min_object_count"] < 0:
                            errors["thresholds.inventory_enrichment.min_object_count"] = "Must be non-negative"
                        if "require_enrichment_metadata" in inventory and not isinstance(
                            inventory["require_enrichment_metadata"],
                            bool,
                        ):
                            errors["thresholds.inventory_enrichment.require_enrichment_metadata"] = "Must be boolean"

                if "variation_assets" in thresholds:
                    variation = thresholds["variation_assets"]
                    if isinstance(variation, dict):
                        if "min_object_count" in variation and variation["min_object_count"] < 0:
                            errors["thresholds.variation_assets.min_object_count"] = "Must be non-negative"
                        if "min_3d_success_rate" in variation:
                            val = variation["min_3d_success_rate"]
                            if not (0.0 <= val <= 1.0):
                                errors["thresholds.variation_assets.min_3d_success_rate"] = "Must be between 0.0 and 1.0"

                if "scale" in thresholds:
                    scale = thresholds["scale"]
                    if isinstance(scale, dict):
                        if "min_scale_factor" in scale and scale["min_scale_factor"] <= 0:
                            errors["thresholds.scale.min_scale_factor"] = "Must be positive"
                        if "max_scale_factor" in scale and scale["max_scale_factor"] <= 0:
                            errors["thresholds.scale.max_scale_factor"] = "Must be positive"
                        if (
                            "min_scale_factor" in scale
                            and "max_scale_factor" in scale
                            and scale["max_scale_factor"] < scale["min_scale_factor"]
                        ):
                            errors["thresholds.scale"] = "max_scale_factor must be >= min_scale_factor"
                        if "min_confidence" in scale:
                            val = scale["min_confidence"]
                            if not (0.0 <= val <= 1.0):
                                errors["thresholds.scale.min_confidence"] = "Must be between 0.0 and 1.0"

                if "interactive" in thresholds:
                    interactive = thresholds["interactive"]
                    if isinstance(interactive, dict):
                        for key in ["min_ok_rate", "min_physics_stability_rate", "max_fallback_rate"]:
                            if key in interactive:
                                val = interactive[key]
                                if not (0.0 <= val <= 1.0):
                                    errors[f"thresholds.interactive.{key}"] = "Must be between 0.0 and 1.0"

        if "gate_overrides" in config:
            gate_overrides = config["gate_overrides"]
            if isinstance(gate_overrides, dict):
                schema = gate_overrides.get("override_reason_schema", {})
                if schema:
                    required_fields = schema.get("required_fields")
                    if required_fields is not None and not isinstance(required_fields, list):
                        errors["gate_overrides.override_reason_schema.required_fields"] = "Must be a list"
                    categories = schema.get("categories")
                    if categories is not None and not isinstance(categories, list):
                        errors["gate_overrides.override_reason_schema.categories"] = "Must be a list"
                    justification_min_length = schema.get("justification_min_length")
                    if justification_min_length is not None and justification_min_length <= 0:
                        errors["gate_overrides.override_reason_schema.justification_min_length"] = (
                            "Must be a positive integer"
                        )

        if "approval_store" in config:
            approval_store = config["approval_store"]
            if isinstance(approval_store, dict):
                backend = approval_store.get("backend")
                if backend and backend not in {"filesystem", "firestore"}:
                    errors["approval_store.backend"] = "Must be 'filesystem' or 'firestore'"
                firestore_collection = approval_store.get("firestore_collection")
                if firestore_collection is not None and not str(firestore_collection).strip():
                    errors["approval_store.firestore_collection"] = "Must be a non-empty string"

        return errors

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


def load_adaptive_timeout_config(
    *,
    bundle_tier: Optional[str] = None,
    scene_complexity: Optional[str] = None,
    bundle_tier_env_var: str = "ADAPTIVE_TIMEOUT_BUNDLE_TIER",
    scene_complexity_env_var: str = "ADAPTIVE_TIMEOUT_SCENE_COMPLEXITY",
) -> AdaptiveTimeoutConfig:
    """Load adaptive timeout defaults with optional tier/complexity selection."""
    if yaml is None:
        raise ImportError("PyYAML is required to load adaptive timeout configuration.")
    if bundle_tier is None:
        bundle_tier = os.getenv(bundle_tier_env_var)
    if scene_complexity is None:
        scene_complexity = os.getenv(scene_complexity_env_var)

    if not ADAPTIVE_TIMEOUTS_PATH.is_file():
        raise FileNotFoundError(f"Adaptive timeout config not found: {ADAPTIVE_TIMEOUTS_PATH}")

    try:
        payload = yaml.safe_load(ADAPTIVE_TIMEOUTS_PATH.read_text()) or {}
    except yaml.YAMLError as exc:  # type: ignore[union-attr]
        raise ValueError(
            f"Failed to parse adaptive timeout config at {ADAPTIVE_TIMEOUTS_PATH}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"Adaptive timeout config at {ADAPTIVE_TIMEOUTS_PATH} must be a YAML mapping."
        )

    def _parse_timeout(value: Any, context: str) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError(f"{context} must be a number of seconds.")
        timeout_value = float(value)
        if timeout_value <= 0:
            raise ValueError(f"{context} must be > 0 seconds.")
        return timeout_value

    default_timeout_raw = payload.get("default_timeout_seconds", 1800)
    default_timeout = _parse_timeout(default_timeout_raw, "default_timeout_seconds")

    bundle_tier_map: Dict[str, float] = {}
    for key, value in (payload.get("bundle_tier") or {}).items():
        bundle_tier_map[str(key)] = _parse_timeout(
            value,
            f"bundle_tier.{key}",
        )

    scene_complexity_map: Dict[str, float] = {}
    for key, value in (payload.get("scene_complexity") or {}).items():
        scene_complexity_map[str(key)] = _parse_timeout(
            value,
            f"scene_complexity.{key}",
        )

    selected_timeout = default_timeout
    selected_bundle_tier = None
    selected_scene_complexity = None
    selected_candidates: List[float] = []
    if bundle_tier:
        candidate = bundle_tier_map.get(bundle_tier)
        if candidate is None:
            logger.warning(
                "Unknown adaptive bundle tier '%s' (expected one of %s).",
                bundle_tier,
                sorted(bundle_tier_map.keys()),
            )
        else:
            selected_bundle_tier = bundle_tier
            selected_candidates.append(candidate)
    if scene_complexity:
        candidate = scene_complexity_map.get(scene_complexity)
        if candidate is None:
            logger.warning(
                "Unknown adaptive scene complexity '%s' (expected one of %s).",
                scene_complexity,
                sorted(scene_complexity_map.keys()),
            )
        else:
            selected_scene_complexity = scene_complexity
            selected_candidates.append(candidate)

    if selected_candidates:
        selected_timeout = max(selected_candidates)

    return AdaptiveTimeoutConfig(
        default_timeout_seconds=default_timeout,
        bundle_tier=bundle_tier_map,
        scene_complexity=scene_complexity_map,
        selected_timeout_seconds=selected_timeout,
        selected_bundle_tier=selected_bundle_tier,
        selected_scene_complexity=selected_scene_complexity,
    )


def load_pipeline_step_timeouts() -> Dict[str, Optional[float]]:
    """Load pipeline step timeouts from environment configuration.

    Expects PIPELINE_STEP_TIMEOUTS_JSON to be a JSON object mapping pipeline
    step names to timeout seconds. Use null or 0 to explicitly disable a timeout.
    """
    raw = os.getenv("PIPELINE_STEP_TIMEOUTS_JSON")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "PIPELINE_STEP_TIMEOUTS_JSON must be valid JSON mapping step names to timeout seconds."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            "PIPELINE_STEP_TIMEOUTS_JSON must be a JSON object mapping step names to timeout seconds."
        )
    timeouts: Dict[str, Optional[float]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                "PIPELINE_STEP_TIMEOUTS_JSON keys must be non-empty strings."
            )
        if value is None:
            timeouts[key.strip()] = None
            continue
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"PIPELINE_STEP_TIMEOUTS_JSON timeout for '{key}' must be a number or null."
            )
        timeout_value = float(value)
        if timeout_value < 0:
            raise ValueError(
                f"PIPELINE_STEP_TIMEOUTS_JSON timeout for '{key}' must be >= 0 seconds."
            )
        if timeout_value == 0:
            timeouts[key.strip()] = None
            continue
        timeouts[key.strip()] = timeout_value
    return timeouts


def get_physics_profile(profile_name: str) -> Dict[str, Any]:
    """Get a physics profile by name."""
    return ConfigLoader.get_physics_profile(profile_name)
