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
class RobotAction:
    """Per-frame robot control target derived from hand motion."""

    # Frame index aligned with hand/camera trajectories
    frame_idx: int

    # 4x4 pose for the wrist/end-effector in the robot base frame
    wrist_pose: np.ndarray

    # Joint positions for the target robot (ordered per joint_names)
    joint_positions: list[float] = field(default_factory=list)

    # Optional joint names (helpful for downstream controllers)
    joint_names: Optional[list[str]] = None

    # Gripper aperture/command (meters for parallel gripper)
    gripper_aperture: float = 0.0

    # Frames of reference
    base_frame: str = "world"
    end_effector_frame: str = "tool0"

    def to_json(self) -> dict:
        """Serialize robot action for JSON export."""
        return {
            "frame_idx": self.frame_idx,
            "wrist_pose": self.wrist_pose.tolist(),
            "joint_positions": self.joint_positions,
            "joint_names": self.joint_names,
            "gripper_aperture": self.gripper_aperture,
            "base_frame": self.base_frame,
            "end_effector_frame": self.end_effector_frame,
        }


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

    # Optional retargeted robot actions aligned to the hand poses
    robot_actions: list[RobotAction] = field(default_factory=list)

    # Grounding metadata for the manipulation target
    target_position: Optional[np.ndarray] = None
    handle_position: Optional[np.ndarray] = None
    approach_direction: Optional[np.ndarray] = None
    motion_axis: Optional[np.ndarray] = None
    joint_limits: Optional[dict[str, float]] = None
    affordance_template: Optional[str] = None
    grounding: dict[str, Any] = field(default_factory=dict)

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

    # Path to generated interaction video (DWM output)
    interaction_video_path: Optional[Path] = None

    # Directory containing interaction frames (DWM output)
    interaction_frames_dir: Optional[Path] = None

    # Text prompt for DWM
    text_prompt: str = ""

    # Paths to individual frames (if video not yet encoded)
    static_scene_frames_dir: Optional[Path] = None
    hand_mesh_frames_dir: Optional[Path] = None
    static_scene_depth_dir: Optional[Path] = None
    static_scene_seg_dir: Optional[Path] = None
    physics_log_path: Optional[Path] = None
    scene_state_path: Optional[Path] = None

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


# =============================================================================
# Episode-Based Models (Enhanced DWM Generation)
# =============================================================================


class EpisodePhase(str, Enum):
    """Phases within a manipulation episode."""
    APPROACH = "approach"
    REACH = "reach"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    POSITION = "position"
    PLACE = "place"
    RELEASE = "release"
    RETRACT = "retract"
    ARTICULATE = "articulate"  # For opening/closing
    IDLE = "idle"


@dataclass
class EpisodeClipConfig:
    """Configuration for a single clip within an episode."""

    # Clip identifier
    clip_id: str
    clip_index: int

    # Frame range within episode
    start_frame: int
    end_frame: int

    # Primary action in this clip
    primary_phase: EpisodePhase
    primary_action: HandActionType
    target_object_id: Optional[str] = None

    # Text prompt for DWM
    text_prompt: str = ""

    # Trajectory hints
    camera_trajectory_type: TrajectoryType = TrajectoryType.REACH_MANIPULATE
    approach_speed: float = 1.0  # Relative speed (1.0 = normal)

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class ManipulationEpisodeConfig:
    """Configuration for a complete manipulation episode."""

    # Episode identifier
    episode_id: str
    task_id: str
    task_name: str
    description: str

    # Scene context
    scene_id: str
    environment_type: str = "generic"

    # Clips in this episode (each is 49 frames)
    clips: list[EpisodeClipConfig] = field(default_factory=list)

    # Objects involved
    source_objects: list[str] = field(default_factory=list)
    target_objects: list[str] = field(default_factory=list)
    manipulated_object: Optional[str] = None

    # Timing
    total_frames: int = 0
    total_duration_seconds: float = 0.0

    # Metadata
    difficulty: str = "medium"
    priority: int = 1
    requires_articulation: bool = False

    @property
    def clip_count(self) -> int:
        return len(self.clips)

    def get_all_text_prompts(self) -> list[str]:
        """Get text prompts for all clips in order."""
        return [clip.text_prompt for clip in self.clips]


@dataclass
class DWMEpisodeBundle:
    """
    Complete DWM bundle for an episode (multiple clips).

    This is the enhanced output format that organizes clips by task/episode.
    """

    # Episode metadata
    episode_id: str
    task_id: str
    task_name: str
    description: str

    # Scene context
    scene_id: str
    environment_type: str = "generic"

    # Clip bundles (each is a complete DWMConditioningBundle)
    clip_bundles: list[DWMConditioningBundle] = field(default_factory=list)

    # Episode-level paths
    output_dir: Optional[Path] = None
    episode_manifest_path: Optional[Path] = None

    # Timing
    total_frames: int = 0
    total_duration_seconds: float = 0.0
    fps: float = 24.0

    # Objects
    source_objects: list[str] = field(default_factory=list)
    target_objects: list[str] = field(default_factory=list)

    # Generation metadata
    generated_at: Optional[str] = None
    generation_time_seconds: float = 0.0

    @property
    def clip_count(self) -> int:
        return len(self.clip_bundles)

    def is_complete(self) -> bool:
        """Check if all clips in episode are complete."""
        return all(clip.is_complete() for clip in self.clip_bundles)

    def get_all_prompts(self) -> list[str]:
        """Get all text prompts in order."""
        return [clip.text_prompt for clip in self.clip_bundles]


@dataclass
class DWMEpisodePipelineConfig:
    """Configuration for episode-based DWM generation."""

    # Scene paths
    scene_manifest_path: Path
    scene_usd_path: Optional[Path] = None

    # Output configuration
    output_dir: Path = Path("./dwm_output")

    # Video parameters (DWM defaults)
    resolution: tuple[int, int] = (720, 480)
    frames_per_clip: int = 49
    fps: float = 24.0

    # Episode generation parameters
    max_episodes: int = 10
    max_clips_per_episode: int = 10
    prioritize_by: str = "dwm_relevance"  # "dwm_relevance", "difficulty", "variety"

    # Scene analysis options
    use_llm_analysis: bool = True
    use_grounded_search: bool = True

    # Trajectory generation
    generate_camera_trajectories: bool = True
    generate_hand_trajectories: bool = True

    # Rendering options
    render_videos: bool = False  # Set True if renderer available
    renderer: str = "isaac_sim"  # "isaac_sim", "pyrender", "blender"

    # Hand model
    hand_model: str = "mano"  # "mano", "geometric"

    # Quality settings
    render_quality: str = "medium"  # "low", "medium", "high"

    # Optional filters
    target_policies: Optional[list[str]] = None  # Only generate for these policies
    target_object_ids: Optional[list[str]] = None  # Only target these objects


@dataclass
class DWMEpisodePipelineOutput:
    """Output from the episode-based DWM pipeline."""

    # Scene ID
    scene_id: str
    environment_type: str

    # Generated episode bundles
    episodes: list[DWMEpisodeBundle] = field(default_factory=list)

    # Statistics
    total_episodes: int = 0
    total_clips: int = 0
    total_frames: int = 0
    total_duration_seconds: float = 0.0

    # Output paths
    output_dir: Optional[Path] = None
    master_manifest_path: Optional[Path] = None

    # Scene analysis info
    scene_summary: str = ""
    recommended_policies: list[str] = field(default_factory=list)
    key_objects: list[str] = field(default_factory=list)

    # Errors
    errors: list[str] = field(default_factory=list)

    # Timing
    generation_time_seconds: float = 0.0
    analysis_time_seconds: float = 0.0
    planning_time_seconds: float = 0.0

    @property
    def success(self) -> bool:
        """Check if pipeline completed successfully."""
        return len(self.episodes) > 0 and len(self.errors) == 0

    def get_episode_by_task(self, task_id: str) -> Optional[DWMEpisodeBundle]:
        """Get episode bundle by task ID."""
        for episode in self.episodes:
            if episode.task_id == task_id:
                return episode
        return None
