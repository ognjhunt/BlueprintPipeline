"""
Data models for Dream2Flow (arXiv:2512.24766) preparation.

Dream2Flow Pipeline:
1. Input: RGB-D observation + language instruction
2. Video Generation: Generate "dreamed" video of task execution
3. Flow Extraction: Extract 3D object flow from video (masks, depth, point tracking)
4. Robot Control: Track object flow using trajectory optimization or RL

Key Concept - 3D Object Flow:
Imagine painting tiny dots on an object. The 3D object flow is the trajectory
of each dot over time. This provides an object-centric goal representation
that bridges the embodiment gap between video generation and robot control.

Reference: https://arxiv.org/abs/2512.24766
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np


class TaskType(str, Enum):
    """Types of manipulation tasks (from Dream2Flow paper examples)."""

    # Pushing tasks
    PUSH = "push"
    PUSH_T = "push_t"
    SWEEP = "sweep"

    # Grasping/manipulation tasks
    GRASP = "grasp"
    LIFT = "lift"
    PLACE = "place"
    PICK_AND_PLACE = "pick_and_place"

    # Container interaction
    PUT_IN = "put_in"
    COVER = "cover"

    # Articulation tasks
    OPEN = "open"
    CLOSE = "close"
    PULL = "pull"

    # Furniture interaction
    PULL_CHAIR = "pull_chair"

    # Other
    RECYCLE = "recycle"
    CUSTOM = "custom"


class FlowExtractionMethod(str, Enum):
    """Methods for extracting flow from generated video."""

    # Track points across frames using CoTracker, TAPIR, etc.
    POINT_TRACKING = "point_tracking"

    # Dense optical flow (RAFT, FlowFormer, etc.)
    OPTICAL_FLOW = "optical_flow"

    # Combine point tracking with depth estimation
    POINT_TRACKING_WITH_DEPTH = "point_tracking_with_depth"

    # Use segmentation + depth for object flow
    SEGMENTATION_DEPTH = "segmentation_depth"


class RobotTrackingMethod(str, Enum):
    """Methods for tracking object flow with robot."""

    # Trajectory optimization (MPC, iLQR, etc.)
    TRAJECTORY_OPTIMIZATION = "trajectory_optimization"

    # Reinforcement learning with flow as reward
    REINFORCEMENT_LEARNING = "reinforcement_learning"

    # Behavior cloning from flow demonstrations
    BEHAVIOR_CLONING = "behavior_cloning"

    # Hybrid approaches
    HYBRID = "hybrid"


class RobotEmbodiment(str, Enum):
    """Supported robot embodiments (from Dream2Flow paper)."""

    FRANKA_PANDA = "franka_panda"
    BOSTON_DYNAMICS_SPOT = "spot"
    FOURIER_GR1 = "gr1"
    UR5E = "ur5e"
    UR10 = "ur10"
    FETCH = "fetch"
    CUSTOM = "custom"


@dataclass
class RGBDObservation:
    """Single RGB-D observation of a scene."""

    # Frame index (0-indexed)
    frame_idx: int = 0

    # RGB image as numpy array (H, W, 3)
    rgb: Optional[np.ndarray] = None

    # Depth map as numpy array (H, W) in meters
    depth: Optional[np.ndarray] = None

    # Path to RGB image file
    rgb_path: Optional[Path] = None

    # Path to depth image file
    depth_path: Optional[Path] = None

    # Camera intrinsics (3x3 matrix)
    camera_intrinsics: Optional[np.ndarray] = None

    # Camera extrinsics (4x4 matrix, camera-to-world)
    camera_extrinsics: Optional[np.ndarray] = None

    # Timestamp in seconds
    timestamp: float = 0.0

    @property
    def resolution(self) -> tuple[int, int]:
        """Return (width, height) of the observation."""
        if self.rgb is not None:
            return (self.rgb.shape[1], self.rgb.shape[0])
        return (0, 0)


@dataclass
class TaskInstruction:
    """Natural language task instruction for Dream2Flow."""

    # The instruction text (e.g., "Put the bread in the bowl")
    text: str

    # Optional: parsed action verb
    action_verb: Optional[str] = None

    # Optional: target object mentioned in instruction
    target_object: Optional[str] = None

    # Optional: destination/goal mentioned in instruction
    destination: Optional[str] = None

    # Task type classification
    task_type: TaskType = TaskType.CUSTOM

    # Optional: detailed task parameters
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectMask:
    """Segmentation mask for an object across video frames."""

    # Object identifier
    object_id: str

    # Object category/class
    category: str = "object"

    # Per-frame masks as list of (frame_idx, mask_array) tuples
    # mask_array is binary (H, W) numpy array
    frame_masks: list[tuple[int, np.ndarray]] = field(default_factory=list)

    # Confidence scores per frame
    confidence_scores: list[float] = field(default_factory=list)

    # Whether this is the manipulated object
    is_manipulated: bool = False

    @property
    def num_frames(self) -> int:
        return len(self.frame_masks)


@dataclass
class TrackedPoint:
    """A single point tracked across video frames."""

    # Point identifier
    point_id: int

    # 2D positions over time (N, 2) - (x, y) in pixel coordinates
    positions_2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    # 3D positions over time (N, 3) - (x, y, z) in world coordinates
    # Computed by lifting 2D tracks with depth
    positions_3d: Optional[np.ndarray] = None

    # Visibility/occlusion flags per frame
    visibility: Optional[np.ndarray] = None

    # Confidence scores per frame
    confidence: Optional[np.ndarray] = None

    # Which object this point belongs to
    object_id: Optional[str] = None

    @property
    def num_frames(self) -> int:
        return self.positions_2d.shape[0] if self.positions_2d is not None else 0

    @property
    def has_3d(self) -> bool:
        return self.positions_3d is not None and self.positions_3d.shape[0] > 0


@dataclass
class ObjectFlow3D:
    """
    3D Object Flow - the core output of Dream2Flow.

    This represents the motion of an object as tracked 3D point trajectories.
    It's object-centric (not robot-centric), making it embodiment-agnostic.
    """

    # Flow identifier
    flow_id: str

    # Object being tracked
    object_id: str
    object_category: str = "object"

    # Tracked points on the object
    tracked_points: list[TrackedPoint] = field(default_factory=list)

    # Number of frames in flow
    num_frames: int = 0

    # FPS of the original video
    fps: float = 24.0

    # Start time offset
    start_time: float = 0.0

    # Flow extraction method used
    extraction_method: FlowExtractionMethod = FlowExtractionMethod.POINT_TRACKING_WITH_DEPTH

    # Quality metrics
    mean_confidence: float = 0.0
    coverage_ratio: float = 0.0  # Fraction of frames with valid tracking

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.num_frames / self.fps if self.fps > 0 else 0.0

    def get_point_cloud_at_frame(self, frame_idx: int) -> np.ndarray:
        """Get 3D point cloud of tracked points at a specific frame."""
        points = []
        for tracked_point in self.tracked_points:
            if tracked_point.has_3d and frame_idx < tracked_point.positions_3d.shape[0]:
                points.append(tracked_point.positions_3d[frame_idx])
        return np.array(points) if points else np.zeros((0, 3))

    def get_center_trajectory(self) -> np.ndarray:
        """Get trajectory of the object center (mean of tracked points)."""
        if not self.tracked_points or self.num_frames == 0:
            return np.zeros((0, 3))

        centers = []
        for frame_idx in range(self.num_frames):
            point_cloud = self.get_point_cloud_at_frame(frame_idx)
            if len(point_cloud) > 0:
                centers.append(np.mean(point_cloud, axis=0))
            elif centers:
                centers.append(centers[-1])  # Repeat last valid center
            else:
                centers.append(np.zeros(3))
        return np.array(centers)


@dataclass
class GeneratedVideo:
    """Video generated by video diffusion model (the 'dream')."""

    # Video identifier
    video_id: str

    # Path to video file
    video_path: Optional[Path] = None

    # Path to frames directory
    frames_dir: Optional[Path] = None

    # Resolution (width, height)
    resolution: tuple[int, int] = (720, 480)

    # Number of frames
    num_frames: int = 49

    # FPS
    fps: float = 24.0

    # The instruction that generated this video
    instruction: Optional[TaskInstruction] = None

    # Initial RGB-D observation
    initial_observation: Optional[RGBDObservation] = None

    # Video generation model used
    model_name: str = "unknown"

    # Generation quality metrics
    # (Video generation artifacts are a major failure mode per the paper)
    quality_score: float = 0.0
    has_morphing_artifacts: bool = False
    has_hallucinations: bool = False

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0


@dataclass
class FlowExtractionResult:
    """Result of extracting 3D object flow from a generated video."""

    # Generated video this was extracted from
    video_id: str

    # Extracted object flows
    object_flows: list[ObjectFlow3D] = field(default_factory=list)

    # Object masks used
    object_masks: list[ObjectMask] = field(default_factory=list)

    # Per-frame depth maps
    depth_maps: list[Optional[np.ndarray]] = field(default_factory=list)

    # Extraction method used
    method: FlowExtractionMethod = FlowExtractionMethod.POINT_TRACKING_WITH_DEPTH

    # Success flag
    success: bool = False

    # Error message if extraction failed
    error: Optional[str] = None

    # Extraction quality metrics
    # (Flow extraction failures are the second major failure mode per the paper)
    extraction_confidence: float = 0.0

    # Paths to intermediate outputs
    masks_dir: Optional[Path] = None
    depth_dir: Optional[Path] = None
    tracks_dir: Optional[Path] = None

    @property
    def num_objects(self) -> int:
        return len(self.object_flows)

    def get_primary_flow(self) -> Optional[ObjectFlow3D]:
        """Get the primary (manipulated) object's flow."""
        for flow in self.object_flows:
            for mask in self.object_masks:
                if mask.object_id == flow.object_id and mask.is_manipulated:
                    return flow
        # Fallback to first flow
        return self.object_flows[0] if self.object_flows else None


@dataclass
class RobotTrackingTarget:
    """Target for robot tracking based on 3D object flow."""

    # Target identifier
    target_id: str

    # The 3D object flow to track
    object_flow: ObjectFlow3D

    # Tracking method to use
    tracking_method: RobotTrackingMethod = RobotTrackingMethod.TRAJECTORY_OPTIMIZATION

    # Robot embodiment
    robot: RobotEmbodiment = RobotEmbodiment.FRANKA_PANDA

    # Whether to use flow as reward (for RL)
    use_as_reward: bool = True

    # Tracking parameters
    lookahead_frames: int = 5
    position_weight: float = 1.0
    velocity_weight: float = 0.1

    # Goal tolerance
    position_tolerance: float = 0.01  # meters

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotTrajectory:
    """Robot trajectory for executing the task."""

    # Trajectory identifier
    trajectory_id: str

    # Robot embodiment
    robot: RobotEmbodiment = RobotEmbodiment.FRANKA_PANDA

    # Per-frame end-effector poses (N, 4, 4) - homogeneous transforms
    ee_poses: Optional[np.ndarray] = None

    # Per-frame joint positions (N, num_joints)
    joint_positions: Optional[np.ndarray] = None

    # Per-frame gripper states (N,) - 0=closed, 1=open
    gripper_states: Optional[np.ndarray] = None

    # Joint names
    joint_names: list[str] = field(default_factory=list)

    # FPS
    fps: float = 24.0

    # Tracking error metrics
    mean_tracking_error: float = 0.0
    max_tracking_error: float = 0.0

    # Success flag
    success: bool = False

    @property
    def num_frames(self) -> int:
        if self.joint_positions is not None:
            return self.joint_positions.shape[0]
        if self.ee_poses is not None:
            return self.ee_poses.shape[0]
        return 0

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0


@dataclass
class Dream2FlowBundle:
    """
    Complete Dream2Flow bundle containing all pipeline outputs.

    This is the main output of the Dream2Flow preparation job,
    analogous to DWMConditioningBundle.
    """

    # Bundle identifier
    bundle_id: str

    # Scene identifier
    scene_id: str

    # Task instruction
    instruction: TaskInstruction

    # Initial RGB-D observation
    initial_observation: RGBDObservation

    # Generated video (the "dream")
    generated_video: Optional[GeneratedVideo] = None

    # Extracted 3D object flows
    flow_extraction: Optional[FlowExtractionResult] = None

    # Robot tracking target
    tracking_target: Optional[RobotTrackingTarget] = None

    # Generated robot trajectory (if tracking performed)
    robot_trajectory: Optional[RobotTrajectory] = None

    # Resolution
    resolution: tuple[int, int] = (720, 480)

    # Number of frames
    num_frames: int = 49

    # FPS
    fps: float = 24.0

    # Output paths
    bundle_dir: Optional[Path] = None
    manifest_path: Optional[Path] = None

    # Quality/success metrics
    video_generation_success: bool = False
    flow_extraction_success: bool = False
    robot_execution_success: bool = False

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        """Check if bundle has all required components."""
        return (
            self.initial_observation is not None and
            self.instruction is not None and
            self.generated_video is not None and
            self.flow_extraction is not None and
            self.flow_extraction.success
        )

    @property
    def success(self) -> bool:
        """Overall success based on pipeline stages."""
        return self.video_generation_success and self.flow_extraction_success


@dataclass
class Dream2FlowJobConfig:
    """Configuration for Dream2Flow preparation job."""

    # Input paths
    manifest_path: Path
    scene_usd_path: Optional[Path] = None

    # Output directory
    output_dir: Path = Path("./dream2flow_output")

    # Task generation
    num_tasks: int = 5
    task_types: list[TaskType] = None
    target_object_ids: Optional[list[str]] = None

    # Video parameters
    resolution: tuple[int, int] = (720, 480)
    num_frames: int = 49
    fps: float = 24.0

    # Video generation
    video_model: str = "placeholder"  # Will be updated when model is released
    video_api_endpoint: Optional[str] = None

    # Flow extraction
    flow_method: FlowExtractionMethod = FlowExtractionMethod.POINT_TRACKING_WITH_DEPTH
    num_tracking_points: int = 100

    # Robot tracking
    tracking_method: RobotTrackingMethod = RobotTrackingMethod.TRAJECTORY_OPTIMIZATION
    robot_embodiment: RobotEmbodiment = RobotEmbodiment.FRANKA_PANDA

    # Output options
    save_intermediate: bool = True
    save_videos: bool = True

    # Processing
    verbose: bool = True

    def __post_init__(self):
        if self.task_types is None:
            self.task_types = [
                TaskType.PUSH,
                TaskType.GRASP,
                TaskType.PICK_AND_PLACE,
                TaskType.OPEN,
            ]


@dataclass
class Dream2FlowPipelineOutput:
    """Output from the Dream2Flow preparation pipeline."""

    # Scene ID
    scene_id: str

    # Generated bundles
    bundles: list[Dream2FlowBundle] = field(default_factory=list)

    # Output directory
    output_dir: Optional[Path] = None

    # Summary statistics
    num_tasks: int = 0
    num_successful_videos: int = 0
    num_successful_flows: int = 0
    num_successful_trajectories: int = 0

    # Per-bundle manifest
    manifest_path: Optional[Path] = None

    # Errors encountered
    errors: list[str] = field(default_factory=list)

    # Timing
    generation_time_seconds: float = 0.0

    # Failure mode breakdown (as reported in Dream2Flow paper)
    video_generation_failures: int = 0
    flow_extraction_failures: int = 0
    robot_execution_failures: int = 0

    @property
    def success(self) -> bool:
        """Check if pipeline completed successfully."""
        return len(self.bundles) > 0 and len(self.errors) == 0

    def get_success_rate(self) -> dict[str, float]:
        """Get success rate for each pipeline stage."""
        total = len(self.bundles) if self.bundles else 1
        return {
            "video_generation": self.num_successful_videos / total,
            "flow_extraction": self.num_successful_flows / total,
            "robot_execution": self.num_successful_trajectories / total,
            "overall": sum(1 for b in self.bundles if b.success) / total,
        }
