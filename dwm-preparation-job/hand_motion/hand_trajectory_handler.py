"""
Hand trajectory generation for DWM conditioning.

Generates hand motion trajectories (MANO-format) that represent
manipulation actions like grasping, pulling, pushing, etc.

Based on DWM paper:
- Hand manipulation trajectory H_1:F
- Encoded as hand meshes (geometry + articulation)
- Must be aligned with camera trajectory C_1:F

Hand Model Formats Supported:
- MANO: Most common for research (45 pose + 10 shape params)
- Shadow Hand: For robotic hand simulation
- Simple skeleton: For lightweight/placeholder use
"""

import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    CameraTrajectory,
    HandActionType,
    HandPose,
    HandTrajectory,
)


@dataclass
class HandConfig:
    """Configuration for hand trajectory generation."""

    # Hand side
    hand_side: str = "right"

    # Hand dimensions (for positioning)
    wrist_to_fingertip: float = 0.18  # meters
    palm_width: float = 0.08  # meters

    # Movement parameters
    approach_speed: float = 0.3  # m/s
    manipulation_speed: float = 0.15  # m/s

    # Grasp aperture
    open_aperture: float = 0.12  # meters
    closed_aperture: float = 0.02  # meters

    # Default MANO shape parameters (average hand)
    default_shape_params: Optional[np.ndarray] = None


def ease_in_out_cubic(t: float) -> float:
    """Smooth ease-in-out interpolation."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def interpolate_pose_params(
    start_params: np.ndarray,
    end_params: np.ndarray,
    t: float,
) -> np.ndarray:
    """Interpolate MANO pose parameters."""
    t_smooth = ease_in_out_cubic(t)
    return start_params + (end_params - start_params) * t_smooth


def generate_reach_trajectory(
    start_position: np.ndarray,
    target_position: np.ndarray,
    num_frames: int = 49,
    fps: float = 24.0,
    hand_side: str = "right",
    trajectory_id: Optional[str] = None,
) -> HandTrajectory:
    """
    Generate a reaching trajectory toward a target.

    Simulates hand moving from a resting position toward an object.

    Args:
        start_position: Starting wrist position
        target_position: Target position to reach
        num_frames: Number of frames
        fps: Frames per second
        hand_side: Which hand
        trajectory_id: Optional trajectory ID

    Returns:
        HandTrajectory for reaching motion
    """
    if trajectory_id is None:
        trajectory_id = f"reach_{uuid.uuid4().hex[:8]}"

    start = np.asarray(start_position, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)

    poses = []

    # MANO rest pose (approximately open hand)
    rest_pose = np.zeros(45, dtype=np.float64)

    # MANO pre-grasp pose (fingers slightly curled, ready to grasp)
    pre_grasp_pose = np.zeros(45, dtype=np.float64)
    # Curl fingers slightly (simplified - would need proper MANO params)
    pre_grasp_pose[3:15] = 0.3  # First joints of fingers

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)

        # Position interpolation with slight arc
        arc_height = 0.05 * math.sin(t * math.pi)
        position = start + (target - start) * ease_in_out_cubic(t)
        position[1] += arc_height  # Add arc in Y

        # Rotation: point toward target
        direction = target - start
        direction_norm = np.linalg.norm(direction[:2])  # XZ plane
        if direction_norm > 1e-8:
            angle = math.atan2(direction[0], direction[2])
        else:
            angle = 0

        rotation = np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ], dtype=np.float64)

        # Pose: transition from rest to pre-grasp
        pose_params = interpolate_pose_params(rest_pose, pre_grasp_pose, t)

        poses.append(HandPose(
            frame_idx=i,
            hand_side=hand_side,
            position=position,
            rotation=rotation,
            pose_params=pose_params,
            timestamp=i / fps,
        ))

    return HandTrajectory(
        trajectory_id=trajectory_id,
        action_type=HandActionType.REACH,
        poses=poses,
        fps=fps,
        description="Reaching toward object",
    )


def generate_grasp_trajectory(
    object_position: np.ndarray,
    approach_direction: np.ndarray = None,
    grasp_width: float = 0.06,
    num_frames: int = 49,
    fps: float = 24.0,
    hand_side: str = "right",
    trajectory_id: Optional[str] = None,
) -> HandTrajectory:
    """
    Generate a grasping trajectory.

    Includes reach phase + grasp closing phase.

    Args:
        object_position: Position of object to grasp
        approach_direction: Direction to approach from (default: front)
        grasp_width: Width of grasp (object diameter)
        num_frames: Number of frames
        fps: Frames per second
        hand_side: Which hand
        trajectory_id: Optional trajectory ID

    Returns:
        HandTrajectory for grasping motion
    """
    if trajectory_id is None:
        trajectory_id = f"grasp_{uuid.uuid4().hex[:8]}"

    if approach_direction is None:
        approach_direction = np.array([0, 0, -1], dtype=np.float64)
    approach_direction = approach_direction / np.linalg.norm(approach_direction)

    target = np.asarray(object_position, dtype=np.float64)

    # Starting position: offset from target
    start_offset = 0.25
    start_position = target - approach_direction * start_offset

    poses = []

    # Phase boundaries
    reach_end = 0.5  # First half: reaching
    # Second half: grasping

    # MANO poses
    rest_pose = np.zeros(45, dtype=np.float64)
    open_grasp_pose = np.zeros(45, dtype=np.float64)
    open_grasp_pose[3:15] = 0.2  # Slight curl

    closed_grasp_pose = np.zeros(45, dtype=np.float64)
    closed_grasp_pose[3:15] = 0.8  # Strong curl for grasp

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)

        if t < reach_end:
            # Reach phase
            phase_t = t / reach_end
            position = start_position + (target - start_position) * ease_in_out_cubic(phase_t)
            pose_params = interpolate_pose_params(rest_pose, open_grasp_pose, phase_t)
        else:
            # Grasp phase
            phase_t = (t - reach_end) / (1 - reach_end)
            position = target.copy()  # Stay at object
            pose_params = interpolate_pose_params(open_grasp_pose, closed_grasp_pose, phase_t)

        # Rotation: palm facing object
        angle = math.atan2(approach_direction[0], approach_direction[2])
        rotation = np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ], dtype=np.float64)

        # Contact state: fingers in contact during grasp phase
        contact = [t > reach_end + 0.1] * 5 if t > reach_end else [False] * 5

        poses.append(HandPose(
            frame_idx=i,
            hand_side=hand_side,
            position=position,
            rotation=rotation,
            pose_params=pose_params,
            timestamp=i / fps,
            contact_fingertips=contact,
        ))

    return HandTrajectory(
        trajectory_id=trajectory_id,
        action_type=HandActionType.GRASP,
        poses=poses,
        fps=fps,
        description=f"Grasping object at {object_position}",
    )


def generate_pull_trajectory(
    handle_position: np.ndarray,
    pull_direction: np.ndarray = None,
    pull_distance: float = 0.2,
    num_frames: int = 49,
    fps: float = 24.0,
    hand_side: str = "right",
    trajectory_id: Optional[str] = None,
) -> HandTrajectory:
    """
    Generate a pulling trajectory (e.g., opening a drawer).

    Includes grasp + pull back motion.

    Args:
        handle_position: Position of handle
        pull_direction: Direction to pull (default: toward camera, -Z)
        pull_distance: How far to pull
        num_frames: Number of frames
        fps: Frames per second
        hand_side: Which hand
        trajectory_id: Optional trajectory ID

    Returns:
        HandTrajectory for pulling motion
    """
    if trajectory_id is None:
        trajectory_id = f"pull_{uuid.uuid4().hex[:8]}"

    if pull_direction is None:
        pull_direction = np.array([0, 0, -1], dtype=np.float64)
    pull_direction = pull_direction / np.linalg.norm(pull_direction)

    handle = np.asarray(handle_position, dtype=np.float64)

    poses = []

    # Phase boundaries
    approach_end = 0.3
    grasp_end = 0.4
    # Pull phase: 0.4 to 1.0

    # Starting position
    start_position = handle - pull_direction * 0.2  # Approach from pull direction

    # MANO poses
    rest_pose = np.zeros(45, dtype=np.float64)
    grasp_pose = np.zeros(45, dtype=np.float64)
    grasp_pose[3:15] = 0.7  # Tight grip for pulling

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)

        if t < approach_end:
            # Approach
            phase_t = t / approach_end
            position = start_position + (handle - start_position) * ease_in_out_cubic(phase_t)
            pose_params = interpolate_pose_params(rest_pose, grasp_pose, phase_t * 0.5)

        elif t < grasp_end:
            # Grasp (quick)
            phase_t = (t - approach_end) / (grasp_end - approach_end)
            position = handle.copy()
            pose_params = interpolate_pose_params(rest_pose * 0.5 + grasp_pose * 0.5, grasp_pose, phase_t)

        else:
            # Pull
            phase_t = (t - grasp_end) / (1 - grasp_end)
            pull_offset = pull_direction * pull_distance * ease_in_out_cubic(phase_t)
            position = handle + pull_offset
            pose_params = grasp_pose.copy()

        # Rotation: aligned with pull direction
        angle = math.atan2(pull_direction[0], pull_direction[2])
        rotation = np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ], dtype=np.float64)

        contact = [t > approach_end] * 5

        poses.append(HandPose(
            frame_idx=i,
            hand_side=hand_side,
            position=position,
            rotation=rotation,
            pose_params=pose_params,
            timestamp=i / fps,
            contact_fingertips=contact,
        ))

    return HandTrajectory(
        trajectory_id=trajectory_id,
        action_type=HandActionType.PULL,
        poses=poses,
        fps=fps,
        description=f"Pulling from {handle_position} by {pull_distance}m",
    )


def generate_push_trajectory(
    contact_position: np.ndarray,
    push_direction: np.ndarray = None,
    push_distance: float = 0.15,
    num_frames: int = 49,
    fps: float = 24.0,
    hand_side: str = "right",
    trajectory_id: Optional[str] = None,
) -> HandTrajectory:
    """
    Generate a pushing trajectory.

    Args:
        contact_position: Position where push starts
        push_direction: Direction to push (default: +Z)
        push_distance: How far to push
        num_frames: Number of frames
        fps: Frames per second
        hand_side: Which hand
        trajectory_id: Optional trajectory ID

    Returns:
        HandTrajectory for pushing motion
    """
    if trajectory_id is None:
        trajectory_id = f"push_{uuid.uuid4().hex[:8]}"

    if push_direction is None:
        push_direction = np.array([0, 0, 1], dtype=np.float64)
    push_direction = push_direction / np.linalg.norm(push_direction)

    contact = np.asarray(contact_position, dtype=np.float64)

    poses = []

    # Phases
    approach_end = 0.35
    # Push phase: 0.35 to 1.0

    # Starting position (hand approaches from opposite of push direction)
    start_position = contact - push_direction * 0.2

    # Open palm pose for pushing
    push_pose = np.zeros(45, dtype=np.float64)
    push_pose[0:3] = [-0.3, 0, 0]  # Slightly extend wrist

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)

        if t < approach_end:
            # Approach
            phase_t = t / approach_end
            position = start_position + (contact - start_position) * ease_in_out_cubic(phase_t)
            pose_params = push_pose * phase_t

        else:
            # Push
            phase_t = (t - approach_end) / (1 - approach_end)
            push_offset = push_direction * push_distance * ease_in_out_cubic(phase_t)
            position = contact + push_offset
            pose_params = push_pose.copy()

        # Rotation: palm facing push direction
        angle = math.atan2(push_direction[0], push_direction[2])
        rotation = np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ], dtype=np.float64)

        contact_state = [t > approach_end - 0.05] * 5

        poses.append(HandPose(
            frame_idx=i,
            hand_side=hand_side,
            position=position,
            rotation=rotation,
            pose_params=pose_params,
            timestamp=i / fps,
            contact_fingertips=contact_state,
        ))

    return HandTrajectory(
        trajectory_id=trajectory_id,
        action_type=HandActionType.PUSH,
        poses=poses,
        fps=fps,
        description=f"Pushing at {contact_position} by {push_distance}m",
    )


class HandTrajectoryGenerator:
    """
    Main class for generating hand trajectories aligned with camera trajectories.

    Usage:
        generator = HandTrajectoryGenerator()
        hand_traj = generator.generate_for_camera_trajectory(
            camera_trajectory,
            target_object_position,
            action_type=HandActionType.GRASP,
        )
    """

    def __init__(self, config: Optional[HandConfig] = None):
        """
        Initialize hand trajectory generator.

        Args:
            config: Hand configuration
        """
        self.config = config or HandConfig()

    def generate_for_camera_trajectory(
        self,
        camera_trajectory: CameraTrajectory,
        target_position: np.ndarray,
        action_type: HandActionType = HandActionType.GRASP,
        hand_side: str = "right",
    ) -> HandTrajectory:
        """
        Generate hand trajectory aligned with camera trajectory.

        The hand motion is generated relative to the camera view,
        making it appear as if reaching from the camera viewpoint.

        Args:
            camera_trajectory: Camera trajectory to align with
            target_position: World position of manipulation target
            action_type: Type of manipulation action
            hand_side: Which hand to use

        Returns:
            HandTrajectory aligned with camera motion
        """
        target = np.asarray(target_position, dtype=np.float64)

        # Determine trajectory based on action type
        if action_type == HandActionType.REACH:
            # Start position relative to camera
            first_pose = camera_trajectory.poses[0]
            start_pos = first_pose.position + first_pose.forward * 0.3 + np.array([0.15, -0.2, 0])

            return generate_reach_trajectory(
                start_position=start_pos,
                target_position=target,
                num_frames=camera_trajectory.num_frames,
                fps=camera_trajectory.fps,
                hand_side=hand_side,
            )

        elif action_type == HandActionType.GRASP:
            # Approach from camera direction
            last_pose = camera_trajectory.poses[-1]
            approach_dir = target - last_pose.position
            approach_dir = approach_dir / np.linalg.norm(approach_dir)

            return generate_grasp_trajectory(
                object_position=target,
                approach_direction=approach_dir,
                num_frames=camera_trajectory.num_frames,
                fps=camera_trajectory.fps,
                hand_side=hand_side,
            )

        elif action_type == HandActionType.PULL:
            # Pull toward camera
            last_pose = camera_trajectory.poses[-1]
            pull_dir = last_pose.position - target
            pull_dir = pull_dir / np.linalg.norm(pull_dir)

            return generate_pull_trajectory(
                handle_position=target,
                pull_direction=pull_dir,
                num_frames=camera_trajectory.num_frames,
                fps=camera_trajectory.fps,
                hand_side=hand_side,
            )

        elif action_type == HandActionType.PUSH:
            # Push away from camera
            last_pose = camera_trajectory.poses[-1]
            push_dir = target - last_pose.position
            push_dir[1] = 0  # Keep horizontal
            push_dir = push_dir / np.linalg.norm(push_dir)

            return generate_push_trajectory(
                contact_position=target,
                push_direction=push_dir,
                num_frames=camera_trajectory.num_frames,
                fps=camera_trajectory.fps,
                hand_side=hand_side,
            )

        else:
            # Default to grasp
            return generate_grasp_trajectory(
                object_position=target,
                num_frames=camera_trajectory.num_frames,
                fps=camera_trajectory.fps,
                hand_side=hand_side,
            )

    def export_trajectory(
        self,
        trajectory: HandTrajectory,
        output_path: Path,
    ) -> Path:
        """
        Export hand trajectory to JSON format.

        Args:
            trajectory: Hand trajectory to export
            output_path: Output file path

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "trajectory_id": trajectory.trajectory_id,
            "action_type": trajectory.action_type.value,
            "num_frames": trajectory.num_frames,
            "fps": trajectory.fps,
            "description": trajectory.description,
            "target_object_id": trajectory.target_object_id,
            "camera_trajectory_id": trajectory.camera_trajectory_id,
            "poses": [],
        }

        for pose in trajectory.poses:
            pose_data = {
                "frame_idx": pose.frame_idx,
                "timestamp": pose.timestamp,
                "hand_side": pose.hand_side,
                "position": pose.position.tolist(),
                "rotation": pose.rotation.tolist(),
                "contact_fingertips": pose.contact_fingertips,
            }

            if pose.pose_params is not None:
                pose_data["pose_params"] = pose.pose_params.tolist()
            if pose.shape_params is not None:
                pose_data["shape_params"] = pose.shape_params.tolist()
            if pose.joint_positions is not None:
                pose_data["joint_positions"] = pose.joint_positions.tolist()

            data["poses"].append(pose_data)

        output_path.write_text(json.dumps(data, indent=2))
        return output_path
