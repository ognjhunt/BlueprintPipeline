"""
Pydantic schemas for configuration validation.

Provides type-safe validation for all configuration files in BlueprintPipeline.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tools.config.env import parse_bool_env
from tools.config.production_mode import resolve_production_mode

# ============================================================================
# Scene Manifest Schemas
# ============================================================================

class CoordinateFrame(str, Enum):
    """Coordinate frame options."""
    Y_UP = "y_up"
    Z_UP = "z_up"


class SimRole(str, Enum):
    """Object simulation roles."""
    STATIC = "static"
    INTERACTIVE = "interactive"
    MANIPULABLE_OBJECT = "manipulable_object"
    ARTICULATED_FURNITURE = "articulated_furniture"
    ARTICULATED_APPLIANCE = "articulated_appliance"
    CLUTTER = "clutter"
    BACKGROUND = "background"
    SCENE_SHELL = "scene_shell"
    UNKNOWN = "unknown"


class Vector3(BaseModel):
    """3D vector."""
    x: float
    y: float
    z: float

    @field_validator('x', 'y', 'z')
    @classmethod
    def validate_finite(cls, v: float) -> float:
        if not (-1e10 < v < 1e10):
            raise ValueError(f"Value {v} is not finite or is too large")
        return v


class Position(Vector3):
    """3D position."""
    pass


class Scale(Vector3):
    """3D scale."""

    @field_validator('x', 'y', 'z')
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Scale must be positive, got {v}")
        return v


class RotationEuler(BaseModel):
    """Euler rotation in degrees."""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


class RotationQuaternion(BaseModel):
    """Quaternion rotation."""
    w: float
    x: float
    y: float
    z: float

    @model_validator(mode='after')
    def validate_normalized(self) -> 'RotationQuaternion':
        """Ensure quaternion is normalized."""
        magnitude = (self.w**2 + self.x**2 + self.y**2 + self.z**2) ** 0.5
        if not (0.9 < magnitude < 1.1):
            raise ValueError(
                f"Quaternion should be normalized (magnitude ~1.0), got {magnitude}"
            )
        return self


class Transform(BaseModel):
    """Object transform (position, rotation, scale)."""
    position: Position
    rotation_euler: Optional[RotationEuler] = None
    rotation_quaternion: Optional[RotationQuaternion] = None
    scale: Scale

    @model_validator(mode='after')
    def validate_rotation(self) -> 'Transform':
        """Ensure at least one rotation format is provided."""
        if self.rotation_euler is None and self.rotation_quaternion is None:
            # Default to identity rotation
            self.rotation_euler = RotationEuler()
        return self


class Dimensions(BaseModel):
    """Object dimensions."""
    width: float = Field(..., gt=0, le=100.0)
    depth: float = Field(..., gt=0, le=100.0)
    height: float = Field(..., gt=0, le=100.0)


class AssetCandidate(BaseModel):
    """Asset candidate with score."""
    asset_path: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    dimensions: Optional[Dimensions] = None


class Asset(BaseModel):
    """Asset reference."""
    path: str = Field(..., min_length=1)
    asset_id: Optional[str] = None
    source: Optional[str] = None
    pack_name: Optional[str] = None
    relative_path: Optional[str] = None
    format: Optional[str] = None
    variants: Optional[Dict[str, str]] = None
    candidates: Optional[List[AssetCandidate]] = None
    simready_metadata: Optional[Dict[str, Any]] = None


class AffordanceSource(str, Enum):
    """Source of affordance annotation."""
    DETECTED = "detected"
    MANUAL = "manual"
    LLM = "llm"
    HEURISTIC = "heuristic"
    FALLBACK = "fallback"


class JointType(str, Enum):
    """Joint type for articulated objects."""
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


class AffordanceParams(BaseModel):
    """Parameters for an affordance."""
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: Optional[AffordanceSource] = None
    joint_type: Optional[JointType] = None
    joint_name: Optional[str] = None
    open_angle: Optional[float] = None
    open_distance: Optional[float] = None
    close_angle: Optional[float] = None
    close_distance: Optional[float] = None
    rotation_axis: Optional[str] = None
    rotation_range: Optional[tuple[float, float]] = None
    press_depth: Optional[float] = None
    toggle: Optional[bool] = None
    button_ids: Optional[List[str]] = None
    grasp_width_range: Optional[tuple[float, float]] = None
    approach_direction: Optional[str] = None
    preferred_grasp_points: Optional[List[Any]] = None
    insertion_axis: Optional[str] = None
    insertion_depth: Optional[float] = None
    receptacle_tolerance: Optional[float] = None
    stack_axis: Optional[str] = None
    max_stack_height: Optional[int] = None
    pour_axis: Optional[str] = None
    capacity_liters: Optional[float] = None
    notes: Optional[str] = None


class Semantics(BaseModel):
    """Semantic information for an object."""
    class_name: Optional[str] = Field(None, alias="class")
    instance_id: Optional[str] = None
    affordances: Optional[List[str]] = None
    affordance_params: Optional[Dict[str, AffordanceParams]] = None
    tags: Optional[List[str]] = None
    custom_labels: Optional[Dict[str, str]] = None

    model_config = ConfigDict(populate_by_name=True)


class ArticulationLimits(BaseModel):
    """Articulation joint limits."""
    lower: float
    upper: float

    @model_validator(mode='after')
    def validate_limits(self) -> 'ArticulationLimits':
        if self.lower >= self.upper:
            raise ValueError(f"Lower limit {self.lower} must be less than upper {self.upper}")
        return self


class Articulation(BaseModel):
    """Articulation information."""
    type: Optional[str] = None
    axis: Optional[str] = None
    limits: Optional[ArticulationLimits] = None
    damping: Optional[float] = Field(None, ge=0.0)
    physx_endpoint: Optional[str] = None


class Relationship(BaseModel):
    """Spatial or semantic relationship between objects."""
    type: str
    subject_id: Optional[str] = None
    object_id: Optional[str] = None


class SceneObject(BaseModel):
    """Object in the scene."""
    id: str = Field(..., pattern=r'^[a-zA-Z0-9_\-]+$')
    name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = Field(None, max_length=1024)
    sim_role: SimRole
    must_be_separate_asset: Optional[bool] = None
    placement_region: Optional[str] = None
    transform: Transform
    dimensions_est: Optional[Dimensions] = None
    asset: Asset
    semantics: Optional[Semantics] = None
    physics: Optional[Dict[str, Any]] = None
    physics_hints: Optional[Dict[str, Any]] = None
    placement: Optional[Dict[str, Any]] = None
    asset_generation: Optional[Dict[str, Any]] = None
    articulation: Optional[Articulation] = None
    relationships: Optional[List[Relationship]] = None
    variation_candidate: Optional[bool] = None
    source: Optional[Dict[str, Any]] = None

    @field_validator('description')
    @classmethod
    def sanitize_description(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Remove control characters
        return ''.join(c for c in v if c.isprintable())


class RoomBounds(BaseModel):
    """Room bounding box."""
    width: float = Field(..., gt=0)
    depth: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


class Room(BaseModel):
    """Room information."""
    bounds: Optional[RoomBounds] = None
    origin: Optional[tuple[float, float, float]] = None


class Gravity(BaseModel):
    """Gravity vector."""
    x: float = 0.0
    y: float = -9.81
    z: float = 0.0


class PhysicsDefaults(BaseModel):
    """Default physics settings."""
    gravity: Optional[Gravity] = None
    solver: Optional[str] = None
    time_steps_per_second: Optional[float] = Field(None, gt=0, le=1000)


class Scene(BaseModel):
    """Scene-level properties."""
    coordinate_frame: CoordinateFrame
    meters_per_unit: float = Field(..., gt=0)
    environment_type: Optional[str] = None
    room: Optional[Room] = None
    physics_defaults: Optional[PhysicsDefaults] = None


class AssetPack(BaseModel):
    """Asset pack reference."""
    name: str
    version_hash: Optional[str] = None
    local_path_template: Optional[str] = None
    gcs_path_template: Optional[str] = None


class Assets(BaseModel):
    """Asset configuration."""
    asset_root: Optional[str] = None
    packs: Optional[List[AssetPack]] = None


class Metadata(BaseModel):
    """Scene metadata."""
    scene_path: Optional[str] = None
    source_pipeline: Optional[str] = None
    notes: Optional[str] = None


class SceneManifest(BaseModel):
    """Complete scene manifest."""
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$')
    scene_id: str = Field(..., pattern=r'^[a-zA-Z0-9_\-]+$')
    scene: Scene
    assets: Optional[Assets] = None
    objects: List[SceneObject]
    background: Optional[Dict[str, Any]] = None
    metadata: Optional[Metadata] = None

    @field_validator('objects')
    @classmethod
    def validate_unique_ids(cls, v: List[SceneObject]) -> List[SceneObject]:
        """Ensure all object IDs are unique."""
        ids = [obj.id for obj in v]
        if len(ids) != len(set(ids)):
            duplicates = [id_ for id_ in ids if ids.count(id_) > 1]
            raise ValueError(f"Duplicate object IDs found: {set(duplicates)}")
        return v


# ============================================================================
# Environment Configuration Schemas
# ============================================================================

class EnvironmentConfig(BaseModel):
    """Environment configuration for pipeline jobs."""
    bucket: str = Field(..., min_length=1)
    scene_id: str = Field(..., pattern=r'^[a-zA-Z0-9_\-]+$')
    assets_prefix: str = Field(default="assets")
    geniesim_prefix: str = Field(default="geniesim")

    # Feature flags
    enable_premium_analytics: bool = Field(default=True)
    enable_multi_robot: bool = Field(default=True)
    enable_cuRobo: bool = Field(default=True)
    enable_cp_gen: bool = Field(default=True)

    # External services
    particulate_endpoint: Optional[str] = None

    # Processing options
    max_tasks: int = Field(default=10, ge=1, le=100)
    episodes_per_variation: int = Field(default=5, ge=1, le=50)
    num_variations: int = Field(default=3, ge=1, le=20)

    # Embedding settings
    generate_embeddings: bool = Field(default=False)
    require_embeddings: bool = Field(default=False)
    openai_api_key: Optional[str] = None
    qwen_api_key: Optional[str] = None
    dashscope_api_key: Optional[str] = None

    @field_validator('bucket')
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validate GCS bucket name format."""
        if not re.match(r'^[a-z0-9][a-z0-9\-_]{1,61}[a-z0-9]$', v):
            raise ValueError(
                f"Invalid bucket name: {v}. Must be lowercase alphanumeric with hyphens/underscores"
            )
        return v

    @model_validator(mode="after")
    def validate_embedding_credentials(self) -> "EnvironmentConfig":
        if self.generate_embeddings and self.require_embeddings:
            if not (self.openai_api_key or self.qwen_api_key or self.dashscope_api_key):
                raise ValueError(
                    "Embedding provider credentials are required when "
                    "GENERATE_EMBEDDINGS=true and REQUIRE_EMBEDDINGS=true. "
                    "Set OPENAI_API_KEY or QWEN_API_KEY/DASHSCOPE_API_KEY."
                )
        return self


# ============================================================================
# Robot Configuration Schemas
# ============================================================================

class RobotType(str, Enum):
    """Supported robot types."""
    FRANKA = "franka"
    UR10 = "ur10"
    FETCH = "fetch"
    SPOT = "spot"
    CUSTOM = "custom"


class RobotConfig(BaseModel):
    """Robot configuration."""
    robot_type: RobotType
    urdf_path: Optional[str] = None
    end_effector: Optional[str] = None
    base_frame: Optional[str] = None
    tcp_frame: Optional[str] = None

    # Motion planning
    max_velocity: float = Field(default=1.0, gt=0)
    max_acceleration: float = Field(default=2.0, gt=0)
    joint_limits: Optional[Dict[str, tuple[float, float]]] = None

    # cuRobo config
    enable_cuRobo: bool = Field(default=True)
    collision_buffer_m: float = Field(default=0.01, ge=0)


# ============================================================================
# Task Configuration Schemas
# ============================================================================

class TaskType(str, Enum):
    """Task types for episode generation."""
    PICK_AND_PLACE = "pick_and_place"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INSPECTION = "inspection"
    CUSTOM = "custom"


class TaskConfig(BaseModel):
    """Task configuration for episode generation."""
    task_type: TaskType
    num_tasks: int = Field(default=10, ge=1, le=100)
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    filter_commercial: bool = Field(default=True)

    # Task-specific parameters
    max_objects_per_task: int = Field(default=5, ge=1, le=20)
    require_articulation: bool = Field(default=False)
    require_grasping: bool = Field(default=True)


# ============================================================================
# Quality Certificate Schema
# ============================================================================

class QualityStatus(str, Enum):
    """Quality validation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class QualityCheck(BaseModel):
    """Individual quality check result."""
    check_name: str
    status: QualityStatus
    score: float = Field(..., ge=0.0, le=1.0)
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class QualityCertificate(BaseModel):
    """Quality certificate for generated data."""
    version: str = Field(default="1.0.0")
    scene_id: str
    timestamp: str
    overall_status: QualityStatus
    overall_score: float = Field(..., ge=0.0, le=1.0)
    checks: List[QualityCheck]
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('checks')
    @classmethod
    def validate_at_least_one_check(cls, v: List[QualityCheck]) -> List[QualityCheck]:
        if not v:
            raise ValueError("Quality certificate must have at least one check")
        return v


# ============================================================================
# Utility Functions
# ============================================================================

def load_and_validate_manifest(manifest_path: Union[str, Path]) -> SceneManifest:
    """
    Load and validate a scene manifest file.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        Validated SceneManifest

    Raises:
        ValidationError: If manifest is invalid
        FileNotFoundError: If file doesn't exist

    Example:
        manifest = load_and_validate_manifest("scene_manifest.json")
    """
    import json
    from pathlib import Path

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(path) as f:
        data = json.load(f)

    return SceneManifest(**data)


def load_and_validate_env_config() -> EnvironmentConfig:
    """
    Load and validate environment configuration from environment variables.

    Returns:
        Validated EnvironmentConfig

    Raises:
        ValidationError: If configuration is invalid

    Example:
        config = load_and_validate_env_config()
        print(f"Bucket: {config.bucket}")
    """
    import os

    production_mode = resolve_production_mode()
    return EnvironmentConfig(
        bucket=os.getenv("BUCKET", ""),
        scene_id=os.getenv("SCENE_ID", ""),
        assets_prefix=os.getenv("ASSETS_PREFIX", "assets"),
        geniesim_prefix=os.getenv("GENIESIM_PREFIX", "geniesim"),
        enable_premium_analytics=parse_bool_env(os.getenv("ENABLE_PREMIUM_ANALYTICS"), default=True),
        enable_multi_robot=parse_bool_env(os.getenv("ENABLE_MULTI_ROBOT"), default=True),
        enable_cuRobo=parse_bool_env(os.getenv("ENABLE_CUROBO"), default=True),
        enable_cp_gen=parse_bool_env(os.getenv("ENABLE_CP_GEN"), default=True),
        particulate_endpoint=os.getenv("PARTICULATE_ENDPOINT"),
        max_tasks=int(os.getenv("MAX_TASKS", "10")),
        episodes_per_variation=int(os.getenv("EPISODES_PER_VARIATION", "5")),
        num_variations=int(os.getenv("NUM_VARIATIONS", "3")),
        generate_embeddings=parse_bool_env(
            os.getenv("GENERATE_EMBEDDINGS"),
            default=production_mode,
        ),
        require_embeddings=parse_bool_env(
            os.getenv("REQUIRE_EMBEDDINGS"),
            default=production_mode,
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        qwen_api_key=os.getenv("QWEN_API_KEY"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
