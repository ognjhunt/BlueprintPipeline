#!/usr/bin/env python3
"""
Data Pack Configuration for Episode Datasets.

Defines configurable data pack tiers that determine what data streams
are included in generated episode datasets. This enables Blueprint to
sell different levels of data richness at different price points.

Data Pack Tiers:
- Core: RGB (1-N cams) + robot state + actions + episode metadata + success/QC
- Plus: Core + depth + segmentation + 2D/3D bboxes (best value for most labs)
- Full: Plus + object poses + contacts + normals + privileged state

Market Positioning:
- Core pack competes with existing trajectory-only datasets
- Plus pack differentiates with "low effort in sim" ground truth
- Full pack targets world model / perception + policy training

See: Episode Pricing Strategy and Competitive Analysis documentation.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DataPackTier(Enum):
    """Data pack tiers for episode datasets."""

    CORE = "core"
    PLUS = "plus"
    FULL = "full"


class OutputFormat(Enum):
    """Output format options."""

    LEROBOT = "lerobot"  # LeRobot v2.0 (default)
    RLDS = "rlds"  # TensorFlow Datasets format (TFDS)
    HDF5 = "hdf5"  # HDF5 format (robomimic/other academic labs)
    HF_DATASETS = "hf_datasets"  # HuggingFace Datasets
    ROSBAG = "rosbag"  # ROS bag format (legacy systems)
    RAW = "raw"  # Raw files (images, JSON, etc.)
    COSMOS_POLICY = "cosmos_policy"  # NVIDIA Cosmos Policy (video diffusion policy)
    GYMNASIUM = "gymnasium"  # Gymnasium/OpenAI Gym format (RL training)


@dataclass
class DatasetSplitConfig:
    """
    Configuration for train/val/test splits.

    This is REQUIRED for proper benchmarking - labs need reproducible
    splits to compare models fairly.

    DROID and BridgeData provide explicit splits; we should too.

    Environment Variables:
        BP_SPLIT_SEED: Override split seed (default: 42)
        BP_SPLIT_STRATEGY: Override split strategy (default: "random")
        BP_SPLIT_TRAIN_RATIO: Override train ratio (default: 0.8)
        BP_SPLIT_VAL_RATIO: Override validation ratio (default: 0.1)
        BP_SPLIT_TEST_RATIO: Override test ratio (default: 0.1)

    Examples:
        # Use default split seed (42) for reproducibility
        config = DatasetSplitConfig()

        # Override split seed via environment variable
        os.environ["BP_SPLIT_SEED"] = "1337"
        config = DatasetSplitConfig()
        # config.split_seed == 1337

        # Override programmatically
        config = DatasetSplitConfig(split_seed=999)
    """

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Seed for reproducible splits
    # Can be overridden via BP_SPLIT_SEED environment variable
    split_seed: int = 42

    # Split strategy
    # - "random": Random split by episode
    # - "scene": Split by scene (all episodes from a scene in same split)
    # - "task": Split by task type (train on some tasks, test on others)
    split_strategy: str = "random"

    # Optional explicit splits (overrides ratios if provided)
    # Maps episode_id to split name ("train", "val", "test")
    explicit_splits: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        """Apply environment variable overrides after initialization."""
        # Allow environment variables to override values if they were using defaults
        if "BP_SPLIT_SEED" in os.environ:
            self.split_seed = int(os.environ["BP_SPLIT_SEED"])
        if "BP_SPLIT_STRATEGY" in os.environ:
            self.split_strategy = os.environ["BP_SPLIT_STRATEGY"]
        if "BP_SPLIT_TRAIN_RATIO" in os.environ:
            self.train_ratio = float(os.environ["BP_SPLIT_TRAIN_RATIO"])
        if "BP_SPLIT_VAL_RATIO" in os.environ:
            self.val_ratio = float(os.environ["BP_SPLIT_VAL_RATIO"])
        if "BP_SPLIT_TEST_RATIO" in os.environ:
            self.test_ratio = float(os.environ["BP_SPLIT_TEST_RATIO"])

        # Run validation
        self.validate()

    def validate(self) -> None:
        """Validate split configuration."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if self.split_strategy not in ["random", "scene", "task"]:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")
        if self.split_seed < 0:
            raise ValueError(f"Split seed must be non-negative, got {self.split_seed}")


@dataclass
class FailureRolloutConfig:
    """
    Configuration for failure rollouts / hard negatives.

    Labs increasingly want failure data for:
    - Contrastive learning
    - Learning what NOT to do
    - Safety-aware policies
    - Robust error recovery

    Reference: Post-training data in the research AI conversation.
    """

    # Include failure episodes in dataset
    include_failures: bool = False

    # Ratio of failures to include (0.0 = none, 1.0 = all failures)
    failure_ratio: float = 0.1

    # Label failure reasons (collision, grasp_failure, etc.)
    include_failure_labels: bool = True

    # Include "near-miss" episodes (low quality but not failed)
    include_near_misses: bool = False
    near_miss_threshold: float = 0.5  # Quality score threshold

    # Separate failure episodes into own split
    separate_failure_split: bool = False


@dataclass
class GoalImageConfig:
    """
    Configuration for goal images.

    Goal-conditioned policies (like BridgeData V2) need "what success looks like".
    This captures the final state of successful episodes as goal images.
    """

    # Enable goal image capture
    enabled: bool = False

    # Which camera(s) to use for goal images
    goal_cameras: List[str] = field(default_factory=lambda: ["wrist", "overhead"])

    # Capture goal image at episode end or from separate "goal state"
    capture_mode: str = "episode_end"  # "episode_end", "explicit_goal", "both"

    # Store as separate files or inline in episode data
    storage_mode: str = "separate"  # "separate", "inline"

    # Resolution for goal images (can differ from observation resolution)
    goal_resolution: Optional[Tuple[int, int]] = None  # None = same as observation


@dataclass
class SkillSegmentConfig:
    """
    Configuration for skill segment annotations.

    Segments each episode into discrete skills (approach, grasp, lift, etc.).
    Useful for:
    - Skill-based learning
    - Hierarchical policies
    - Learning from sub-trajectories
    """

    # Export skill segment annotations
    enabled: bool = True

    # Include segment boundaries in observation data
    include_in_observations: bool = True

    # Export as separate annotation file
    export_separate_file: bool = True

    # Segment types to track
    segment_types: List[str] = field(default_factory=lambda: [
        "home", "approach", "pre_grasp", "grasp", "lift",
        "transport", "pre_place", "place", "release", "retract"
    ])


@dataclass
class IMUConfig:
    """
    Configuration for IMU (Inertial Measurement Unit) data.

    Important for mobile manipulation robots (Fetch, TIAGo, Spot).
    """

    # Enable IMU capture
    enabled: bool = False

    # IMU location on robot
    imu_location: str = "base"  # "base", "torso", "head"

    # Capture rate (Hz)
    capture_rate: float = 100.0

    # Data to capture
    capture_linear_acceleration: bool = True
    capture_angular_velocity: bool = True
    capture_orientation: bool = True  # Quaternion from IMU fusion

    # Add noise simulation (for sim-to-real)
    add_noise: bool = True
    accel_noise_std: float = 0.01  # m/s^2
    gyro_noise_std: float = 0.001  # rad/s


@dataclass
class PointCloudConfig:
    """
    Configuration for point cloud data.

    Useful for:
    - 3D-aware world models
    - Point cloud policies (PointNet, etc.)
    - Geometric reasoning
    """

    # Enable point cloud capture
    enabled: bool = False

    # Source for point clouds
    source: str = "depth"  # "depth" (derive from depth + intrinsics), "lidar", "both"

    # Point cloud parameters
    max_points: int = 10000
    subsample_method: str = "random"  # "random", "voxel", "fps"

    # Coordinate frame
    coordinate_frame: str = "world"  # "world", "camera", "robot_base"

    # Include RGB colors per point
    include_colors: bool = True

    # Include normals per point
    include_normals: bool = False

    # Output format
    output_format: str = "npz"  # "npz", "ply", "pcd"


@dataclass
class TactileIntegrationConfig:
    """
    Configuration for tactile sensor integration.

    Research shows tactile + visual policies achieve 81%+ success
    vs ~50% for vision-only in contact-rich tasks.
    """

    # Enable tactile sensor simulation
    enabled: bool = False

    # Sensor type (from upsell-features-job/tactile_sensor_sim.py)
    sensor_type: str = "gelslim"  # "gelslim", "gelsight", "digit", "magnetic"

    # Dual gripper (both fingers)
    dual_gripper: bool = True

    # Include tactile images (RGB-like visualization)
    include_tactile_images: bool = True

    # Include force maps
    include_force_maps: bool = True

    # Include marker displacements (for marker-based sensors)
    include_marker_displacements: bool = True

    # Storage format
    storage_format: str = "npz"  # "npz", "hdf5"


@dataclass
class JointDynamicsConfig:
    """
    Configuration for joint dynamics data (torques, efforts).

    Important for contact-rich manipulation and sim-to-real.
    """

    # Enable joint dynamics capture
    enabled: bool = True

    # Capture joint torques (from physics simulation)
    capture_torques: bool = True

    # Capture commanded efforts
    capture_efforts: bool = True

    # Capture joint velocities (often already captured, but explicit)
    capture_velocities: bool = True

    # Capture joint accelerations (derived)
    capture_accelerations: bool = False

    # Include external forces/torques on end-effector
    capture_ee_wrench: bool = True


@dataclass
class ExtendedSensorConfig:
    """
    Configuration for extended sensor data (NEW).

    These sensors are critical for:
    - Video diffusion models (optical flow)
    - Humanoid robots (balance, GRF, foot contacts)
    - Contact-rich manipulation (EE wrench)
    - Articulated object manipulation (drawer/door states)
    """

    # End-effector wrench (6D force/torque) - critical for contact manipulation
    include_ee_wrench: bool = False

    # Optical flow / motion vectors - for video diffusion models (Cosmos Policy)
    include_optical_flow: bool = False

    # Depth confidence/uncertainty maps - for sim-to-real quality
    include_depth_confidence: bool = False

    # Balance & stability metrics (CoM, ZMP, CoP) - for humanoid robots
    include_balance_state: bool = False

    # Ground reaction forces / foot contacts - for humanoid robots
    include_foot_contacts: bool = False

    # Articulated object states (drawers, doors, cabinets)
    include_articulated_objects: bool = False

    # Language annotations (task instructions, paraphrases)
    include_language_annotations: bool = False


@dataclass
class HumanoidSensorConfig:
    """
    Configuration for humanoid-specific sensors.

    For robots like G1, GR1, H1, NEO, Digit, Phoenix, Figure 01.
    """

    # Enable humanoid-specific sensors
    enabled: bool = False

    # Balance state (CoM, ZMP, CoP, stability margin)
    capture_balance_state: bool = True

    # Ground reaction forces (GRF) per foot
    capture_grf: bool = True

    # Foot contact states (stance/swing, contact position)
    capture_foot_contacts: bool = True

    # Torso IMU (orientation, angular velocity, acceleration)
    capture_torso_imu: bool = True

    # Pelvis state (for floating base control)
    capture_pelvis_state: bool = True


@dataclass
class CameraCalibrationConfig:
    """
    Configuration for camera calibration data export.

    DROID explicitly calls this out as a key differentiator.
    """

    # Export camera calibration data
    enabled: bool = True

    # Export intrinsic matrix (3x3)
    export_intrinsics: bool = True

    # Export distortion coefficients
    export_distortion: bool = True

    # Export extrinsic matrix (4x4 camera-to-world)
    export_extrinsics: bool = True

    # Export per-frame extrinsics (for moving cameras like wrist)
    export_per_frame_extrinsics: bool = True

    # Export camera-to-robot-base transform
    export_camera_to_robot: bool = True

    # Output format
    output_format: str = "json"  # "json", "yaml", "npz"


@dataclass
class CameraSpec:
    """Specification for a single camera view."""

    camera_id: str
    camera_type: str  # "wrist", "overhead", "side", "front", etc.
    description: str
    default_resolution: Tuple[int, int] = (640, 480)
    default_focal_length: float = 24.0  # mm
    typical_prim_path: str = ""

    # What this camera typically captures
    primary_use: str = "observation"  # "observation", "aux", "eval_only"


# Standard camera configurations for robotics
STANDARD_CAMERAS = {
    "wrist": CameraSpec(
        camera_id="wrist",
        camera_type="wrist",
        description="End-effector mounted camera (eye-in-hand)",
        default_resolution=(640, 480),
        typical_prim_path="/World/Robot/wrist_camera",
        primary_use="observation",
    ),
    "overhead": CameraSpec(
        camera_id="overhead",
        camera_type="overhead",
        description="Top-down view of workspace",
        default_resolution=(640, 480),
        typical_prim_path="/World/Cameras/overhead_camera",
        primary_use="observation",
    ),
    "side": CameraSpec(
        camera_id="side",
        camera_type="side",
        description="Side view of workspace",
        default_resolution=(640, 480),
        typical_prim_path="/World/Cameras/side_camera",
        primary_use="aux",
    ),
    "front": CameraSpec(
        camera_id="front",
        camera_type="front",
        description="Front-facing view",
        default_resolution=(640, 480),
        typical_prim_path="/World/Cameras/front_camera",
        primary_use="aux",
    ),
}


@dataclass
class DataStreamConfig:
    """Configuration for a single data stream."""

    stream_id: str
    stream_type: str  # "rgb", "depth", "segmentation", "bbox", "pose", etc.
    enabled: bool = True
    format: str = "default"  # Format-specific options
    compression: Optional[str] = None  # "h264", "png", "npz", etc.
    per_camera: bool = False  # True if this stream is per-camera
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataPackConfig:
    """
    Complete configuration for a data pack tier.

    This class defines all data streams, formats, and options for
    a specific data pack tier.

    Now includes ALL the data points labs want (per research AI conversation):
    - Camera calibration (DROID-style)
    - Train/val/test splits
    - Joint torques/efforts
    - Goal images
    - Failure rollouts
    - Skill segments
    - IMU (mobile robots)
    - Point clouds
    - Tactile integration
    """

    tier: DataPackTier
    name: str
    description: str

    # Data streams included
    streams: List[DataStreamConfig] = field(default_factory=list)

    # Camera configuration
    num_cameras: int = 1
    camera_types: List[str] = field(default_factory=lambda: ["wrist"])
    resolution: Tuple[int, int] = (640, 480)

    # Output format (now supports RLDS, HDF5, ROS bag)
    output_format: OutputFormat = OutputFormat.LEROBOT
    fps: float = 30.0

    # Quality settings
    min_quality_score: float = 0.7
    include_failed_episodes: bool = False

    # Feature flags
    include_task_descriptions: bool = True
    include_language_annotations: bool = False
    include_skill_segments: bool = False

    # Metadata
    version: str = "2.0.0"  # Bumped for new features
    pricing_tier: str = "standard"

    # =========================================================================
    # NEW: P0 - Critical for benchmarks (DROID-level quality)
    # =========================================================================

    # Train/val/test splits - REQUIRED for proper benchmarking
    split_config: DatasetSplitConfig = field(default_factory=DatasetSplitConfig)

    # Camera calibration - DROID explicitly calls this out
    camera_calibration_config: CameraCalibrationConfig = field(
        default_factory=CameraCalibrationConfig
    )

    # =========================================================================
    # NEW: P1 - High value for post-training and contact-rich tasks
    # =========================================================================

    # Joint dynamics (torques, efforts) - important for contact-rich
    joint_dynamics_config: JointDynamicsConfig = field(
        default_factory=JointDynamicsConfig
    )

    # Goal images for goal-conditioned policies
    goal_image_config: GoalImageConfig = field(default_factory=GoalImageConfig)

    # Failure rollouts for contrastive learning / hard negatives
    failure_config: FailureRolloutConfig = field(default_factory=FailureRolloutConfig)

    # Tactile sensor integration
    tactile_config: TactileIntegrationConfig = field(
        default_factory=TactileIntegrationConfig
    )

    # =========================================================================
    # NEW: P2 - Nice to have for specific use cases
    # =========================================================================

    # Skill segment export (approach, grasp, lift, etc.)
    skill_segment_config: SkillSegmentConfig = field(
        default_factory=SkillSegmentConfig
    )

    # IMU data for mobile robots
    imu_config: IMUConfig = field(default_factory=IMUConfig)

    # Point cloud generation
    point_cloud_config: PointCloudConfig = field(default_factory=PointCloudConfig)

    # =========================================================================
    # NEW: P0 - Extended Sensors (Video Diffusion, Humanoids, Contact-Rich)
    # =========================================================================

    # Extended sensor configuration (optical flow, depth confidence, etc.)
    extended_sensor_config: ExtendedSensorConfig = field(
        default_factory=ExtendedSensorConfig
    )

    # Humanoid-specific sensors (balance, GRF, foot contacts)
    humanoid_sensor_config: HumanoidSensorConfig = field(
        default_factory=HumanoidSensorConfig
    )

    # =========================================================================
    # Export format helpers
    # =========================================================================

    # Additional output formats to generate (alongside primary)
    additional_formats: List[OutputFormat] = field(default_factory=list)

    def get_stream(self, stream_id: str) -> Optional[DataStreamConfig]:
        """Get a stream configuration by ID."""
        for stream in self.streams:
            if stream.stream_id == stream_id:
                return stream
        return None

    def has_stream(self, stream_type: str) -> bool:
        """Check if a stream type is enabled."""
        return any(
            s.stream_type == stream_type and s.enabled for s in self.streams
        )

    @property
    def has_rgb(self) -> bool:
        return self.has_stream("rgb")

    @property
    def has_depth(self) -> bool:
        return self.has_stream("depth")

    @property
    def has_segmentation(self) -> bool:
        return self.has_stream("semantic_segmentation") or self.has_stream(
            "instance_segmentation"
        )

    @property
    def has_bboxes(self) -> bool:
        return self.has_stream("bbox_2d") or self.has_stream("bbox_3d")

    @property
    def has_object_poses(self) -> bool:
        return self.has_stream("object_pose")

    @property
    def has_contacts(self) -> bool:
        return self.has_stream("contact_info")

    @property
    def has_privileged_state(self) -> bool:
        return self.has_stream("privileged_state")


# =============================================================================
# Predefined Data Pack Configurations
# =============================================================================


def create_core_pack(
    num_cameras: int = 1,
    resolution: Tuple[int, int] = (640, 480),
    fps: float = 30.0,
) -> DataPackConfig:
    """
    Create Core data pack configuration.

    Core Pack includes:
    - RGB images (1-N cameras)
    - Robot state (joint positions, velocities, gripper)
    - Actions (joint commands)
    - Episode metadata (task, success, duration)
    - Quality metrics (sim-verified)
    - Camera calibration (intrinsics + extrinsics) [NEW - P0]
    - Train/val/test splits [NEW - P0]
    - Skill segment annotations [NEW - P2]

    Target: Labs training visuomotor policies (ACT, Diffusion Policy)
    """
    streams = [
        DataStreamConfig(
            stream_id="rgb",
            stream_type="rgb",
            enabled=True,
            format="mp4",
            compression="h264",
            per_camera=True,
            metadata={"colorspace": "sRGB", "antialiasing": True},
        ),
        DataStreamConfig(
            stream_id="robot_state",
            stream_type="robot_state",
            enabled=True,
            format="parquet",
            metadata={
                "includes": [
                    "joint_positions",
                    "joint_velocities",  # NEW
                    "gripper_position",
                    "ee_position",
                ]
            },
        ),
        DataStreamConfig(
            stream_id="actions",
            stream_type="actions",
            enabled=True,
            format="parquet",
            metadata={"action_space": "joint_position+gripper"},
        ),
        DataStreamConfig(
            stream_id="episode_metadata",
            stream_type="metadata",
            enabled=True,
            format="json",
            metadata={"includes": ["task", "success", "duration", "quality_score"]},
        ),
        # NEW: Camera calibration stream
        DataStreamConfig(
            stream_id="camera_calibration",
            stream_type="calibration",
            enabled=True,
            format="json",
            per_camera=True,
            metadata={
                "includes": ["intrinsic_matrix", "distortion_coeffs", "extrinsic_matrix"]
            },
        ),
        # NEW: Skill segments stream
        DataStreamConfig(
            stream_id="skill_segments",
            stream_type="annotation",
            enabled=True,
            format="json",
            metadata={"type": "skill_segment"},
        ),
    ]

    camera_types = ["wrist"]
    if num_cameras >= 2:
        camera_types.append("overhead")
    if num_cameras >= 3:
        camera_types.append("side")
    if num_cameras >= 4:
        camera_types.append("front")

    return DataPackConfig(
        tier=DataPackTier.CORE,
        name="Core Pack",
        description="RGB observations + robot state + actions + metadata + calibration + splits",
        streams=streams,
        num_cameras=num_cameras,
        camera_types=camera_types[:num_cameras],
        resolution=resolution,
        fps=fps,
        include_task_descriptions=True,
        pricing_tier="standard",
        # NEW: Enable P0 features by default
        split_config=DatasetSplitConfig(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            split_seed=42,
        ),
        camera_calibration_config=CameraCalibrationConfig(enabled=True),
        skill_segment_config=SkillSegmentConfig(enabled=True),
        # Joint dynamics basic (velocities only for core)
        joint_dynamics_config=JointDynamicsConfig(
            enabled=True,
            capture_velocities=True,
            capture_torques=False,  # Plus/Full only
            capture_efforts=False,
            capture_ee_wrench=False,
        ),
    )


def create_plus_pack(
    num_cameras: int = 2,
    resolution: Tuple[int, int] = (640, 480),
    fps: float = 30.0,
) -> DataPackConfig:
    """
    Create Plus data pack configuration.

    Plus Pack includes everything in Core, plus:
    - Depth maps (per camera)
    - Semantic segmentation masks
    - Instance segmentation masks
    - 2D bounding boxes (COCO format)
    - 3D bounding boxes (camera-space)
    - Joint torques/efforts [NEW - P1]
    - Goal images [NEW - P1]
    - Failure rollouts (optional) [NEW - P1]
    - Point clouds (derived from depth) [NEW - P2]

    Target: Labs training perception + policy, object-centric models
    Best value for most robotics research labs.
    """
    streams = [
        # Core streams
        DataStreamConfig(
            stream_id="rgb",
            stream_type="rgb",
            enabled=True,
            format="mp4",
            compression="h264",
            per_camera=True,
            metadata={"colorspace": "sRGB", "antialiasing": True},
        ),
        DataStreamConfig(
            stream_id="robot_state",
            stream_type="robot_state",
            enabled=True,
            format="parquet",
            metadata={
                "includes": [
                    "joint_positions",
                    "joint_velocities",
                    "joint_torques",  # NEW
                    "joint_efforts",  # NEW
                    "gripper_position",
                    "ee_position",
                    "ee_wrench",  # NEW
                ]
            },
        ),
        DataStreamConfig(
            stream_id="actions",
            stream_type="actions",
            enabled=True,
            format="parquet",
        ),
        DataStreamConfig(
            stream_id="episode_metadata",
            stream_type="metadata",
            enabled=True,
            format="json",
        ),
        # Camera calibration (P0)
        DataStreamConfig(
            stream_id="camera_calibration",
            stream_type="calibration",
            enabled=True,
            format="json",
            per_camera=True,
            metadata={
                "includes": [
                    "intrinsic_matrix",
                    "distortion_coeffs",
                    "extrinsic_matrix",
                    "camera_to_robot_base",
                ]
            },
        ),
        # Skill segments
        DataStreamConfig(
            stream_id="skill_segments",
            stream_type="annotation",
            enabled=True,
            format="json",
            metadata={"type": "skill_segment"},
        ),
        # Plus streams
        DataStreamConfig(
            stream_id="depth",
            stream_type="depth",
            enabled=True,
            format="npz",
            compression="zlib",
            per_camera=True,
            metadata={
                "near_clip": 0.1,
                "far_clip": 100.0,
                "output_linear": True,
                "unit": "meters",
            },
        ),
        DataStreamConfig(
            stream_id="semantic_segmentation",
            stream_type="semantic_segmentation",
            enabled=True,
            format="npz",
            per_camera=True,
            metadata={"colorize": False, "include_label_mapping": True},
        ),
        DataStreamConfig(
            stream_id="instance_segmentation",
            stream_type="instance_segmentation",
            enabled=True,
            format="npz",
            per_camera=True,
            metadata={"include_id_mapping": True},
        ),
        DataStreamConfig(
            stream_id="bbox_2d",
            stream_type="bbox_2d",
            enabled=True,
            format="json",
            per_camera=True,
            metadata={
                "format": "coco",
                "include_occluded": True,
                "visibility_threshold": 0.1,
            },
        ),
        DataStreamConfig(
            stream_id="bbox_3d",
            stream_type="bbox_3d",
            enabled=True,
            format="json",
            per_camera=True,
            metadata={
                "coordinate_system": "camera",
                "include_orientation": True,
            },
        ),
        # NEW: Goal images
        DataStreamConfig(
            stream_id="goal_images",
            stream_type="goal_image",
            enabled=True,
            format="png",
            per_camera=True,
            metadata={"capture_mode": "episode_end"},
        ),
        # NEW: Point clouds
        DataStreamConfig(
            stream_id="point_clouds",
            stream_type="point_cloud",
            enabled=True,
            format="npz",
            per_camera=True,
            metadata={
                "source": "depth",
                "max_points": 10000,
                "include_colors": True,
            },
        ),
    ]

    camera_types = ["wrist", "overhead"][:num_cameras]
    if num_cameras >= 3:
        camera_types.append("side")
    if num_cameras >= 4:
        camera_types.append("front")

    return DataPackConfig(
        tier=DataPackTier.PLUS,
        name="Plus Pack",
        description="Core + depth + segmentation + bboxes + torques + goals + point clouds",
        streams=streams,
        num_cameras=num_cameras,
        camera_types=camera_types[:num_cameras],
        resolution=resolution,
        fps=fps,
        include_task_descriptions=True,
        include_skill_segments=True,
        pricing_tier="premium",
        # P0: Splits and calibration
        split_config=DatasetSplitConfig(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            split_seed=42,
        ),
        camera_calibration_config=CameraCalibrationConfig(
            enabled=True,
            export_per_frame_extrinsics=True,  # Important for wrist camera
            export_camera_to_robot=True,
        ),
        # P1: Joint dynamics with torques
        joint_dynamics_config=JointDynamicsConfig(
            enabled=True,
            capture_velocities=True,
            capture_torques=True,
            capture_efforts=True,
            capture_ee_wrench=True,
        ),
        # P1: Goal images
        goal_image_config=GoalImageConfig(
            enabled=True,
            goal_cameras=["wrist", "overhead"],
            capture_mode="episode_end",
        ),
        # P1: Failure rollouts (opt-in)
        failure_config=FailureRolloutConfig(
            include_failures=False,  # Opt-in for Plus
            failure_ratio=0.1,
            include_failure_labels=True,
        ),
        # P2: Skill segments
        skill_segment_config=SkillSegmentConfig(
            enabled=True,
            include_in_observations=True,
            export_separate_file=True,
        ),
        # P2: Point clouds
        point_cloud_config=PointCloudConfig(
            enabled=True,
            source="depth",
            max_points=10000,
            include_colors=True,
        ),
    )


def create_full_pack(
    num_cameras: int = 3,
    resolution: Tuple[int, int] = (640, 480),
    fps: float = 30.0,
    enable_mobile_robot_sensors: bool = False,
) -> DataPackConfig:
    """
    Create Full data pack configuration.

    Full Pack includes everything in Plus, plus:
    - Object poses (world-space quaternion + position)
    - Contact information (forces, contact points)
    - Surface normals (per camera)
    - Privileged state (full physics state for evaluation)
    - Language annotations (task descriptions, skill labels)
    - Tactile sensor simulation [NEW - P1]
    - Failure rollouts with labels [NEW - P1]
    - IMU data (for mobile robots) [NEW - P2]
    - Full point clouds with normals [NEW - P2]
    - Multiple export formats (RLDS, HDF5) [NEW - P1]

    Target: Labs training world models, simulation-to-real transfer,
    full-stack robotics research.
    """
    streams = [
        # Core streams
        DataStreamConfig(
            stream_id="rgb",
            stream_type="rgb",
            enabled=True,
            format="mp4",
            compression="h264",
            per_camera=True,
        ),
        DataStreamConfig(
            stream_id="robot_state",
            stream_type="robot_state",
            enabled=True,
            format="parquet",
            metadata={
                "includes": [
                    "joint_positions",
                    "joint_velocities",
                    "joint_torques",
                    "joint_efforts",
                    "joint_accelerations",  # NEW - Full only
                    "gripper_position",
                    "gripper_force",  # NEW
                    "ee_position",
                    "ee_orientation",
                    "ee_velocity",  # NEW
                    "ee_wrench",
                ]
            },
        ),
        DataStreamConfig(
            stream_id="actions",
            stream_type="actions",
            enabled=True,
            format="parquet",
        ),
        DataStreamConfig(
            stream_id="episode_metadata",
            stream_type="metadata",
            enabled=True,
            format="json",
        ),
        # Camera calibration (P0) - Full calibration
        DataStreamConfig(
            stream_id="camera_calibration",
            stream_type="calibration",
            enabled=True,
            format="json",
            per_camera=True,
            metadata={
                "includes": [
                    "intrinsic_matrix",
                    "distortion_coeffs",
                    "extrinsic_matrix",
                    "camera_to_robot_base",
                    "per_frame_extrinsics",  # For wrist camera
                ]
            },
        ),
        # Skill segments
        DataStreamConfig(
            stream_id="skill_segments",
            stream_type="annotation",
            enabled=True,
            format="json",
            metadata={"type": "skill_segment", "include_boundaries": True},
        ),
        # Plus streams
        DataStreamConfig(
            stream_id="depth",
            stream_type="depth",
            enabled=True,
            format="npz",
            compression="zlib",
            per_camera=True,
        ),
        DataStreamConfig(
            stream_id="semantic_segmentation",
            stream_type="semantic_segmentation",
            enabled=True,
            format="npz",
            per_camera=True,
        ),
        DataStreamConfig(
            stream_id="instance_segmentation",
            stream_type="instance_segmentation",
            enabled=True,
            format="npz",
            per_camera=True,
        ),
        DataStreamConfig(
            stream_id="bbox_2d",
            stream_type="bbox_2d",
            enabled=True,
            format="json",
            per_camera=True,
        ),
        DataStreamConfig(
            stream_id="bbox_3d",
            stream_type="bbox_3d",
            enabled=True,
            format="json",
            per_camera=True,
        ),
        # Goal images
        DataStreamConfig(
            stream_id="goal_images",
            stream_type="goal_image",
            enabled=True,
            format="png",
            per_camera=True,
            metadata={"capture_mode": "episode_end"},
        ),
        # Point clouds with normals
        DataStreamConfig(
            stream_id="point_clouds",
            stream_type="point_cloud",
            enabled=True,
            format="npz",
            per_camera=True,
            metadata={
                "source": "depth",
                "max_points": 20000,  # More points for Full
                "include_colors": True,
                "include_normals": True,  # Full only
            },
        ),
        # Full streams
        DataStreamConfig(
            stream_id="normals",
            stream_type="normals",
            enabled=True,
            format="exr",
            per_camera=True,
            metadata={"coordinate_system": "camera"},
        ),
        DataStreamConfig(
            stream_id="object_pose",
            stream_type="object_pose",
            enabled=True,
            format="json",
            metadata={
                "coordinate_system": "world",
                "format": "quaternion",
                "include_velocity": True,
                "include_acceleration": True,  # Full only
            },
        ),
        DataStreamConfig(
            stream_id="contact_info",
            stream_type="contact_info",
            enabled=True,
            format="json",
            metadata={
                "include_force_magnitude": True,
                "include_contact_normal": True,
                "include_contact_point": True,
                "include_contact_impulse": True,  # Full only
            },
        ),
        DataStreamConfig(
            stream_id="privileged_state",
            stream_type="privileged_state",
            enabled=True,
            format="json",
            metadata={
                "includes": [
                    "object_velocities",
                    "gripper_force",
                    "contact_flags",
                    "grasp_status",
                    "object_accelerations",  # Full only
                    "physics_timestep",  # Full only
                ]
            },
        ),
        # NEW: Tactile sensors (Full only)
        DataStreamConfig(
            stream_id="tactile",
            stream_type="tactile",
            enabled=True,
            format="npz",
            metadata={
                "sensor_type": "gelslim",
                "dual_gripper": True,
                "include_images": True,
                "include_force_maps": True,
                "include_marker_displacements": True,
            },
        ),
        # NEW: Failure labels (when failure rollouts enabled)
        DataStreamConfig(
            stream_id="failure_labels",
            stream_type="annotation",
            enabled=True,
            format="json",
            metadata={
                "type": "failure_label",
                "include_reason": True,
                "include_frame": True,
            },
        ),
        # NEW: Optical flow / motion vectors (for video diffusion models)
        DataStreamConfig(
            stream_id="optical_flow",
            stream_type="optical_flow",
            enabled=True,
            format="npz",
            per_camera=True,
            metadata={
                "format": "flow_xy",  # (H, W, 2) float32
                "include_confidence": True,
            },
        ),
        # NEW: Depth confidence maps
        DataStreamConfig(
            stream_id="depth_confidence",
            stream_type="depth_confidence",
            enabled=True,
            format="npz",
            per_camera=True,
            metadata={
                "range": [0.0, 1.0],
                "computation": "gradient_based",
            },
        ),
        # NEW: Balance state (for humanoid robots)
        DataStreamConfig(
            stream_id="balance_state",
            stream_type="balance_state",
            enabled=True,
            format="parquet",
            metadata={
                "includes": ["com_position", "com_velocity", "zmp_position", "cop_position", "stability_margin"],
            },
        ),
        # NEW: Foot contacts / ground reaction forces (for humanoid robots)
        DataStreamConfig(
            stream_id="foot_contacts",
            stream_type="foot_contacts",
            enabled=True,
            format="parquet",
            metadata={
                "includes": ["contact_state", "contact_force", "contact_position", "contact_normal"],
                "per_foot": True,
            },
        ),
        # NEW: Articulated object states (drawers, doors, cabinets)
        DataStreamConfig(
            stream_id="articulated_objects",
            stream_type="articulated_object_state",
            enabled=True,
            format="parquet",
            metadata={
                "includes": ["joint_position", "joint_velocity"],
                "per_object": True,
            },
        ),
        # NEW: Language annotations (task instructions, paraphrases)
        DataStreamConfig(
            stream_id="language_annotations",
            stream_type="annotation",
            enabled=True,
            format="json",
            metadata={
                "type": "language",
                "includes": ["task_instruction", "subtask_instructions", "paraphrase_variations"],
            },
        ),
    ]

    # Add IMU stream for mobile robots
    if enable_mobile_robot_sensors:
        streams.append(
            DataStreamConfig(
                stream_id="imu",
                stream_type="imu",
                enabled=True,
                format="parquet",
                metadata={
                    "location": "base",
                    "rate_hz": 100.0,
                    "include_linear_acceleration": True,
                    "include_angular_velocity": True,
                    "include_orientation": True,
                },
            )
        )

    camera_types = ["wrist", "overhead", "side"][:num_cameras]
    if num_cameras >= 4:
        camera_types.append("front")

    return DataPackConfig(
        tier=DataPackTier.FULL,
        name="Full Pack",
        description="Complete data: Plus + poses + contacts + tactile + failures + IMU + multi-format export",
        streams=streams,
        num_cameras=num_cameras,
        camera_types=camera_types[:num_cameras],
        resolution=resolution,
        fps=fps,
        include_task_descriptions=True,
        include_language_annotations=True,
        include_skill_segments=True,
        pricing_tier="enterprise",
        # P0: Full splits and calibration
        split_config=DatasetSplitConfig(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            split_seed=42,
        ),
        camera_calibration_config=CameraCalibrationConfig(
            enabled=True,
            export_intrinsics=True,
            export_distortion=True,
            export_extrinsics=True,
            export_per_frame_extrinsics=True,
            export_camera_to_robot=True,
        ),
        # P1: Full joint dynamics
        joint_dynamics_config=JointDynamicsConfig(
            enabled=True,
            capture_velocities=True,
            capture_torques=True,
            capture_efforts=True,
            capture_accelerations=True,  # Full only
            capture_ee_wrench=True,
        ),
        # P1: Goal images from all cameras
        goal_image_config=GoalImageConfig(
            enabled=True,
            goal_cameras=["wrist", "overhead", "side"],
            capture_mode="episode_end",
        ),
        # P1: Failure rollouts ENABLED by default for Full
        failure_config=FailureRolloutConfig(
            include_failures=True,  # Enabled for Full pack
            failure_ratio=0.1,
            include_failure_labels=True,
            include_near_misses=True,
            near_miss_threshold=0.5,
        ),
        # P1: Tactile integration
        tactile_config=TactileIntegrationConfig(
            enabled=True,
            sensor_type="gelslim",
            dual_gripper=True,
            include_tactile_images=True,
            include_force_maps=True,
            include_marker_displacements=True,
        ),
        # P2: Full skill segments
        skill_segment_config=SkillSegmentConfig(
            enabled=True,
            include_in_observations=True,
            export_separate_file=True,
        ),
        # P2: IMU for mobile robots
        imu_config=IMUConfig(
            enabled=enable_mobile_robot_sensors,
            imu_location="base",
            capture_rate=100.0,
            capture_linear_acceleration=True,
            capture_angular_velocity=True,
            capture_orientation=True,
            add_noise=True,
        ),
        # P2: Full point clouds with normals
        point_cloud_config=PointCloudConfig(
            enabled=True,
            source="depth",
            max_points=20000,
            include_colors=True,
            include_normals=True,
        ),
        # P0: Extended sensors (video diffusion, contact-rich manipulation)
        extended_sensor_config=ExtendedSensorConfig(
            include_ee_wrench=True,
            include_optical_flow=True,
            include_depth_confidence=True,
            include_balance_state=True,  # For humanoids
            include_foot_contacts=True,  # For humanoids
            include_articulated_objects=True,
            include_language_annotations=True,
        ),
        # P0: Humanoid-specific sensors (for G1, H1, GR1, etc.)
        humanoid_sensor_config=HumanoidSensorConfig(
            enabled=enable_mobile_robot_sensors,  # Enable when mobile robot sensors requested
            capture_balance_state=True,
            capture_grf=True,
            capture_foot_contacts=True,
            capture_torso_imu=True,
            capture_pelvis_state=True,
        ),
        # P1: Additional export formats (including Gymnasium for RL training)
        additional_formats=[OutputFormat.RLDS, OutputFormat.HDF5, OutputFormat.GYMNASIUM],
    )


def get_data_pack_config(
    tier: DataPackTier,
    num_cameras: int = 1,
    resolution: Tuple[int, int] = (640, 480),
    fps: float = 30.0,
) -> DataPackConfig:
    """
    Get data pack configuration for a tier.

    Args:
        tier: Data pack tier
        num_cameras: Number of cameras
        resolution: Image resolution
        fps: Frames per second

    Returns:
        DataPackConfig for the specified tier
    """
    creators = {
        DataPackTier.CORE: create_core_pack,
        DataPackTier.PLUS: create_plus_pack,
        DataPackTier.FULL: create_full_pack,
    }

    creator = creators.get(tier, create_core_pack)
    return creator(num_cameras=num_cameras, resolution=resolution, fps=fps)


# =============================================================================
# Configuration Utilities
# =============================================================================


def data_pack_from_string(tier_str: str) -> DataPackTier:
    """Convert string to DataPackTier."""
    tier_map = {
        "core": DataPackTier.CORE,
        "plus": DataPackTier.PLUS,
        "full": DataPackTier.FULL,
    }
    return tier_map.get(tier_str.lower(), DataPackTier.CORE)


def get_tier_comparison() -> Dict[str, Dict[str, Any]]:
    """
    Get comparison of all data pack tiers.

    Returns:
        Dictionary with tier comparison for documentation/UI.
    """
    return {
        "core": {
            "name": "Core Pack",
            "includes": [
                "RGB images (1-N cameras)",
                "Robot state (joints, gripper, EE pose)",
                "Actions (joint commands)",
                "Episode metadata",
                "Quality metrics (sim-verified)",
            ],
            "use_cases": [
                "Visuomotor policy training (ACT, Diffusion Policy)",
                "Behavior cloning",
                "Basic imitation learning",
            ],
            "target_customers": [
                "Academic labs (policy learning focus)",
                "Startups doing behavior cloning",
            ],
        },
        "plus": {
            "name": "Plus Pack",
            "includes": [
                "Everything in Core",
                "Depth maps (per camera)",
                "Semantic segmentation",
                "Instance segmentation",
                "2D bounding boxes (COCO format)",
                "3D bounding boxes",
            ],
            "use_cases": [
                "Object-centric policy learning",
                "Perception + policy training",
                "Scene understanding models",
                "Affordance learning",
            ],
            "target_customers": [
                "Labs doing perception + manipulation",
                "Companies needing ground-truth labels",
            ],
            "best_value": True,
        },
        "full": {
            "name": "Full Pack",
            "includes": [
                "Everything in Plus",
                "Object poses (world-space)",
                "Contact information",
                "Surface normals",
                "Privileged state (physics)",
                "Language annotations",
            ],
            "use_cases": [
                "World model training",
                "Sim-to-real transfer research",
                "Contact-rich manipulation",
                "Full-stack robotics research",
            ],
            "target_customers": [
                "Research labs (world models, foundation models)",
                "Companies doing sim-to-real",
            ],
        },
    }


def get_leorbot_feature_config(pack_config: DataPackConfig) -> Dict[str, Any]:
    """
    Get LeRobot-compatible feature configuration from pack config.

    Returns:
        Dictionary suitable for LeRobot info.json features field.
    """
    features = {}

    # Robot state features (always present)
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [7],  # Configurable based on robot
        "names": None,  # Set by exporter
    }

    features["observation.gripper_position"] = {
        "dtype": "float32",
        "shape": [1],
    }

    features["observation.ee_position"] = {
        "dtype": "float32",
        "shape": [3],
    }

    features["action"] = {
        "dtype": "float32",
        "shape": [8],  # 7 joints + gripper
        "names": None,
    }

    # Camera-based features
    for i, camera_type in enumerate(pack_config.camera_types):
        cam_key = f"observation.images.{camera_type}"

        if pack_config.has_rgb:
            features[cam_key] = {
                "dtype": "video",
                "shape": list(pack_config.resolution) + [3],
                "video_info": {
                    "fps": pack_config.fps,
                    "codec": "h264",
                },
            }

        if pack_config.has_depth:
            features[f"observation.depth.{camera_type}"] = {
                "dtype": "float32",
                "shape": list(pack_config.resolution),
            }

        if pack_config.has_segmentation:
            features[f"observation.segmentation.{camera_type}"] = {
                "dtype": "uint8",
                "shape": list(pack_config.resolution),
            }

    # Ground-truth features (Plus and Full packs)
    if pack_config.has_bboxes:
        features["ground_truth.bboxes_2d"] = {
            "dtype": "json",
            "format": "coco",
        }
        features["ground_truth.bboxes_3d"] = {
            "dtype": "json",
            "format": "camera_space",
        }

    # Object pose features (Full pack)
    if pack_config.has_object_poses:
        features["ground_truth.object_poses"] = {
            "dtype": "json",
            "format": "quaternion_position",
        }

    # Contact features (Full pack)
    if pack_config.has_contacts:
        features["ground_truth.contacts"] = {
            "dtype": "json",
            "format": "contact_list",
        }

    # Privileged state (Full pack)
    if pack_config.has_privileged_state:
        features["ground_truth.privileged_state"] = {
            "dtype": "json",
            "format": "physics_state",
        }

    return features


# =============================================================================
# Main / Documentation
# =============================================================================


if __name__ == "__main__":
    from tools.logging_config import init_logging

    init_logging()
    logger.info("Data Pack Configuration System")
    logger.info("%s", "=" * 60)

    # Show tier comparison
    comparison = get_tier_comparison()

    for tier_key, info in comparison.items():
        logger.info("%s", "=" * 40)
        logger.info("%s", info["name"].upper())
        logger.info("%s", "=" * 40)

        logger.info("Includes:")
        for item in info["includes"]:
            logger.info("  - %s", item)

        logger.info("Use Cases:")
        for case in info["use_cases"]:
            logger.info("  - %s", case)

        logger.info("Target Customers:")
        for customer in info["target_customers"]:
            logger.info("  - %s", customer)

        if info.get("best_value"):
            logger.info("  *** BEST VALUE FOR MOST LABS ***")

    # Show example configurations
    logger.info("%s", "=" * 60)
    logger.info("Example Configurations")
    logger.info("%s", "=" * 60)

    for tier in [DataPackTier.CORE, DataPackTier.PLUS, DataPackTier.FULL]:
        config = get_data_pack_config(tier, num_cameras=2)
        logger.info("%s:", config.name)
        logger.info("  Streams: %s", len(config.streams))
        logger.info("  Cameras: %s", config.camera_types)
        logger.info("  Has RGB: %s", config.has_rgb)
        logger.info("  Has Depth: %s", config.has_depth)
        logger.info("  Has Segmentation: %s", config.has_segmentation)
        logger.info("  Has BBoxes: %s", config.has_bboxes)
        logger.info("  Has Object Poses: %s", config.has_object_poses)
        logger.info("  Has Contacts: %s", config.has_contacts)
