"""
Data models for DWM (Dexterous World Model) preparation.

Based on the DWM paper (arXiv:2512.17907):
- Input: Static scene video + Hand mesh video + Text prompt
- Output: Egocentric interaction video

This module defines the data structures for preparing DWM conditioning inputs.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np


class TrajectoryType(str, Enum):
    """Types of egocentric camera trajectories."""

    # Smooth approach toward an object
    APPROACH = "approach"

    # Orbit around a point of interest
    ORBIT = "orbit"

    # Linear walk-through of the scene
    WALKTHROUGH = "walkthrough"

    # Reach and manipulate (hand-centric)
    REACH_MANIPULATE = "reach_manipulate"

    # Random exploration
    RANDOM_EXPLORE = "random_explore"

    # Custom spline path
    CUSTOM = "custom"


class HandActionType(str, Enum):
    """Types of hand manipulation actions (for trajectory generation)."""

    # Reaching toward an object
    REACH = "reach"

    # Grasping/gripping
    GRASP = "grasp"

    # Pulling (e.g., drawer, door)
    PULL = "pull"

    # Pushing
    PUSH = "push"

    # Rotating (e.g., knob, handle)
    ROTATE = "rotate"

    # Lifting
    LIFT = "lift"

    # Placing
    PLACE = "place"

    # Sliding
    SLIDE = "slide"


@dataclass
class CameraPose:
    """Single camera pose in a trajectory."""

    # Frame index (0-indexed)
    frame_idx: int

    # 4x4 camera-to-world transformation matrix
    # Column-major, OpenGL convention (camera looks down -Z)
    transform: np.ndarray

    # Timestamp in seconds (for video timing)
    timestamp: float = 0.0

    # Optional: intrinsics (if varying per frame)
    focal_length: Optional[float] = None

    @property
    def position(self) -> np.ndarray:
        """Camera position (world coordinates)."""
        return self.transform[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        """Camera rotation matrix (3x3)."""
        return self.transform[:3, :3]

    @property
    def forward(self) -> np.ndarray:
        """Camera forward direction (world coordinates)."""
        return -self.rotation[:, 2]

    @property
    def up(self) -> np.ndarray:
        """Camera up direction (world coordinates)."""
        return self.rotation[:, 1]


@dataclass
class CameraTrajectory:
    """A sequence of camera poses forming an egocentric trajectory."""

    # Unique identifier
    trajectory_id: str

    # Type of trajectory
    trajectory_type: TrajectoryType

    # Sequence of camera poses
    poses: list[CameraPose] = field(default_factory=list)

    # Target FPS for video rendering
    fps: float = 24.0

    # Camera intrinsics (shared across trajectory unless per-pose override)
    focal_length: float = 24.0
    sensor_width: float = 36.0  # mm
    sensor_height: float = 24.0  # mm

    # Resolution for rendering
    resolution: tuple[int, int] = (720, 480)  # DWM default

    # Optional metadata
    target_object_id: Optional[str] = None
    action_type: Optional[HandActionType] = None
    description: str = ""

    @property
    def num_frames(self) -> int:
        """Number of frames in trajectory."""
        return len(self.poses)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.num_frames / self.fps if self.fps > 0 else 0.0


@dataclass
class HandPose:
    """Single hand pose (MANO-compatible format)."""

    # Frame index
    frame_idx: int

    # Which hand (left/right)
    hand_side: str = "right"

    # Global hand position (wrist position in world coordinates)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Global hand rotation (3x3 rotation matrix or quaternion)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    # MANO pose parameters (45 for hand articulation)
    # 15 joints x 3 axis-angle = 45 parameters
    pose_params: Optional[np.ndarray] = None

    # MANO shape parameters (10 for hand shape)
    shape_params: Optional[np.ndarray] = None

    # Joint positions (21 joints x 3 = 63 values)
    # Alternative to pose_params for direct joint control
    joint_positions: Optional[np.ndarray] = None

    # Timestamp
    timestamp: float = 0.0

    # Contact state (which fingertips are in contact)
    contact_fingertips: list[bool] = field(default_factory=lambda: [False] * 5)


@dataclass
class HandTrajectory:
    """A sequence of hand poses for manipulation."""

    # Unique identifier
    trajectory_id: str

    # Type of manipulation action
    action_type: HandActionType

    # Sequence of hand poses
    poses: list[HandPose] = field(default_factory=list)

    # Associated camera trajectory ID
    camera_trajectory_id: Optional[str] = None

    # Target object being manipulated
    target_object_id: Optional[str] = None

    # Description (used as text prompt for DWM)
    description: str = ""

    # FPS (should match camera trajectory)
    fps: float = 24.0

    @property
    def num_frames(self) -> int:
        """Number of frames."""
        return len(self.poses)


@dataclass
class DWMConditioningBundle:
    """
    Complete conditioning bundle for DWM inference.

    Based on DWM paper requirements:
    1. Static scene video - rendered from static 3D scene along camera trajectory
    2. Hand mesh video - rendered hand meshes along same trajectory
    3. Text prompt - semantic description of intended action
    """

    # Bundle identifier
    bundle_id: str

    # Source scene ID
    scene_id: str

    # Camera trajectory used for rendering
    camera_trajectory: CameraTrajectory

    # Hand trajectory for manipulation
    hand_trajectory: Optional[HandTrajectory] = None

    # Path to rendered static scene video
    static_scene_video_path: Optional[Path] = None

    # Path to rendered hand mesh video
    hand_mesh_video_path: Optional[Path] = None

    # Text prompt for DWM
    text_prompt: str = ""

    # Paths to individual frames (if video not yet encoded)
    static_scene_frames_dir: Optional[Path] = None
    hand_mesh_frames_dir: Optional[Path] = None

    # Resolution
    resolution: tuple[int, int] = (720, 480)

    # Number of frames (DWM generates 49 frames)
    num_frames: int = 49

    # FPS
    fps: float = 24.0

    # Metadata for downstream processing
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if bundle has all required components."""
        has_scene_video = (
            self.static_scene_video_path is not None or
            self.static_scene_frames_dir is not None
        )
        has_hand_video = (
            self.hand_mesh_video_path is not None or
            self.hand_mesh_frames_dir is not None
        )
        has_prompt = bool(self.text_prompt)

        return has_scene_video and has_hand_video and has_prompt


@dataclass
class DWMSceneConfig:
    """Configuration for DWM bundle generation from a scene."""

    # Scene paths
    scene_manifest_path: Path
    scene_usd_path: Path

    # Output configuration
    output_dir: Path

    # Video parameters (DWM defaults)
    resolution: tuple[int, int] = (720, 480)
    num_frames: int = 49
    fps: float = 24.0

    # Trajectory generation parameters
    num_trajectories: int = 5
    trajectory_types: list[TrajectoryType] = field(
        default_factory=lambda: [
            TrajectoryType.APPROACH,
            TrajectoryType.REACH_MANIPULATE,
        ]
    )

    # Hand motion parameters
    generate_hand_motions: bool = True
    hand_action_types: list[HandActionType] = field(
        default_factory=lambda: [
            HandActionType.REACH,
            HandActionType.GRASP,
            HandActionType.PULL,
            HandActionType.PUSH,
        ]
    )

    # Rendering backend
    renderer: str = "isaac_sim"  # or "pyrender", "blender"

    # Hand model
    hand_model: str = "mano"  # or "shadow_hand", "allegro"

    # Quality settings
    render_quality: str = "medium"  # "low", "medium", "high"
    antialiasing: bool = True

    # Optional: specific objects to target
    target_object_ids: Optional[list[str]] = None


@dataclass
class DWMPipelineOutput:
    """Output from the DWM preparation pipeline step."""

    # Scene ID
    scene_id: str

    # Generated bundles
    bundles: list[DWMConditioningBundle] = field(default_factory=list)

    # Output directory
    output_dir: Path = None

    # Summary statistics
    num_trajectories: int = 0
    num_bundles: int = 0
    total_frames: int = 0

    # Per-bundle manifest for downstream consumption
    manifest_path: Optional[Path] = None

    # Errors encountered
    errors: list[str] = field(default_factory=list)

    # Timing
    generation_time_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """Check if pipeline completed successfully."""
        return len(self.bundles) > 0 and len(self.errors) == 0
