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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DataPackTier(Enum):
    """Data pack tiers for episode datasets."""

    CORE = "core"
    PLUS = "plus"
    FULL = "full"


class OutputFormat(Enum):
    """Output format options."""

    LEROBOT = "lerobot"  # LeRobot v2.0 (default)
    RLDS = "rlds"  # TensorFlow Datasets format
    HF_DATASETS = "hf_datasets"  # HuggingFace Datasets
    RAW = "raw"  # Raw files (images, JSON, etc.)


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

    # Output format
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
    version: str = "1.0.0"
    pricing_tier: str = "standard"

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
    - Robot state (joint positions, gripper)
    - Actions (joint commands)
    - Episode metadata (task, success, duration)
    - Quality metrics (sim-verified)

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
        description="RGB observations + robot state + actions + metadata",
        streams=streams,
        num_cameras=num_cameras,
        camera_types=camera_types[:num_cameras],
        resolution=resolution,
        fps=fps,
        include_task_descriptions=True,
        pricing_tier="standard",
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
    ]

    camera_types = ["wrist", "overhead"][:num_cameras]
    if num_cameras >= 3:
        camera_types.append("side")
    if num_cameras >= 4:
        camera_types.append("front")

    return DataPackConfig(
        tier=DataPackTier.PLUS,
        name="Plus Pack",
        description="Core + depth + segmentation + 2D/3D bounding boxes",
        streams=streams,
        num_cameras=num_cameras,
        camera_types=camera_types[:num_cameras],
        resolution=resolution,
        fps=fps,
        include_task_descriptions=True,
        include_skill_segments=True,
        pricing_tier="premium",
    )


def create_full_pack(
    num_cameras: int = 3,
    resolution: Tuple[int, int] = (640, 480),
    fps: float = 30.0,
) -> DataPackConfig:
    """
    Create Full data pack configuration.

    Full Pack includes everything in Plus, plus:
    - Object poses (world-space quaternion + position)
    - Contact information (forces, contact points)
    - Surface normals (per camera)
    - Privileged state (full physics state for evaluation)
    - Language annotations (task descriptions, skill labels)

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
                ]
            },
        ),
    ]

    camera_types = ["wrist", "overhead", "side"][:num_cameras]
    if num_cameras >= 4:
        camera_types.append("front")

    return DataPackConfig(
        tier=DataPackTier.FULL,
        name="Full Pack",
        description="Plus + object poses + contacts + normals + privileged state",
        streams=streams,
        num_cameras=num_cameras,
        camera_types=camera_types[:num_cameras],
        resolution=resolution,
        fps=fps,
        include_task_descriptions=True,
        include_language_annotations=True,
        include_skill_segments=True,
        pricing_tier="enterprise",
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
    print("Data Pack Configuration System")
    print("=" * 60)

    # Show tier comparison
    comparison = get_tier_comparison()

    for tier_key, info in comparison.items():
        print(f"\n{'=' * 40}")
        print(f"{info['name'].upper()}")
        print("=" * 40)

        print("\nIncludes:")
        for item in info["includes"]:
            print(f"  - {item}")

        print("\nUse Cases:")
        for case in info["use_cases"]:
            print(f"  - {case}")

        print("\nTarget Customers:")
        for customer in info["target_customers"]:
            print(f"  - {customer}")

        if info.get("best_value"):
            print("\n  *** BEST VALUE FOR MOST LABS ***")

    # Show example configurations
    print("\n" + "=" * 60)
    print("Example Configurations")
    print("=" * 60)

    for tier in [DataPackTier.CORE, DataPackTier.PLUS, DataPackTier.FULL]:
        config = get_data_pack_config(tier, num_cameras=2)
        print(f"\n{config.name}:")
        print(f"  Streams: {len(config.streams)}")
        print(f"  Cameras: {config.camera_types}")
        print(f"  Has RGB: {config.has_rgb}")
        print(f"  Has Depth: {config.has_depth}")
        print(f"  Has Segmentation: {config.has_segmentation}")
        print(f"  Has BBoxes: {config.has_bboxes}")
        print(f"  Has Object Poses: {config.has_object_poses}")
        print(f"  Has Contacts: {config.has_contacts}")
