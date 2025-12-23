"""
Egocentric trajectory generation for DWM conditioning.

Generates camera trajectories that simulate egocentric (first-person)
viewpoints moving through a scene, typically toward objects for manipulation.

Based on DWM paper requirements:
- Camera trajectory C_1:F for F frames
- Aligned with hand manipulation trajectory H_1:F
- Used to render static scene video as conditioning input
"""

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    CameraPose,
    CameraTrajectory,
    HandActionType,
    TrajectoryType,
)


@dataclass
class SceneContext:
    """Context about the scene for trajectory generation."""

    # Room dimensions
    room_width: float = 5.0
    room_depth: float = 5.0
    room_height: float = 3.0

    # Objects in scene (id -> {position, bounds, category, sim_role})
    objects: dict[str, dict] = None

    # Manipulable objects (subset of objects)
    manipulable_objects: list[str] = None

    # Articulated objects (drawers, doors, etc.)
    articulated_objects: list[str] = None

    # Floor level
    floor_y: float = 0.0

    # Default eye height (standing person)
    eye_height: float = 1.6

    def __post_init__(self):
        if self.objects is None:
            self.objects = {}
        if self.manipulable_objects is None:
            self.manipulable_objects = []
        if self.articulated_objects is None:
            self.articulated_objects = []


def look_at_matrix(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray = None
) -> np.ndarray:
    """
    Compute 4x4 camera-to-world transformation matrix.

    Args:
        eye: Camera position
        target: Look-at target position
        up: Up vector (default: Y-up)

    Returns:
        4x4 transformation matrix
    """
    if up is None:
        up = np.array([0, 1, 0], dtype=np.float64)

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    # Forward direction (camera looks down -Z in OpenGL convention)
    forward = target - eye
    forward_len = np.linalg.norm(forward)
    if forward_len < 1e-8:
        forward = np.array([0, 0, -1], dtype=np.float64)
    else:
        forward = forward / forward_len

    # Right direction
    right = np.cross(forward, up)
    right_len = np.linalg.norm(right)
    if right_len < 1e-8:
        # Forward is parallel to up, choose arbitrary right
        right = np.array([1, 0, 0], dtype=np.float64)
    else:
        right = right / right_len

    # Recompute up to ensure orthogonality
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Build 4x4 matrix (OpenGL: -Z forward)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -forward  # OpenGL convention
    mat[:3, 3] = eye

    return mat


def cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t: float
) -> np.ndarray:
    """Evaluate cubic Bezier curve at parameter t."""
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt

    return mt3 * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3 * p3


def ease_in_out(t: float) -> float:
    """Smooth ease-in-out interpolation."""
    return t * t * (3 - 2 * t)


def generate_approach_trajectory(
    target_position: np.ndarray,
    start_distance: float = 2.0,
    end_distance: float = 0.5,
    approach_angle: float = 0.0,
    elevation_angle: float = 15.0,
    num_frames: int = 49,
    fps: float = 24.0,
    eye_height: float = 1.6,
    trajectory_id: Optional[str] = None,
) -> CameraTrajectory:
    """
    Generate a trajectory approaching an object.

    Simulates a person walking toward and reaching for an object.

    Args:
        target_position: Position of target object
        start_distance: Starting distance from target
        end_distance: Final distance from target
        approach_angle: Horizontal angle of approach (degrees, 0 = front)
        elevation_angle: Vertical angle offset from horizontal (degrees)
        num_frames: Number of frames to generate
        fps: Frames per second
        eye_height: Camera height (person's eye level)
        trajectory_id: Optional trajectory ID

    Returns:
        CameraTrajectory with approach motion
    """
    if trajectory_id is None:
        trajectory_id = f"approach_{uuid.uuid4().hex[:8]}"

    target = np.asarray(target_position, dtype=np.float64)

    # Convert angles to radians
    approach_rad = math.radians(approach_angle)
    elevation_rad = math.radians(elevation_angle)

    poses = []

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        t_smooth = ease_in_out(t)

        # Interpolate distance
        distance = start_distance + (end_distance - start_distance) * t_smooth

        # Add slight arc to approach (more natural)
        arc_offset = math.sin(t * math.pi) * 0.2

        # Compute camera position
        angle = approach_rad + arc_offset
        x = target[0] + distance * math.sin(angle) * math.cos(elevation_rad)
        z = target[2] + distance * math.cos(angle) * math.cos(elevation_rad)

        # Eye height with slight bob (natural walking motion)
        bob = math.sin(t * math.pi * 4) * 0.02 * (1 - t)  # Decreases as we approach
        y = eye_height + bob

        eye = np.array([x, y, z])

        # Look at target with slight downward gaze at end
        look_y = target[1] + (1 - t) * 0.2  # Gradually look down
        look_target = np.array([target[0], look_y, target[2]])

        transform = look_at_matrix(eye, look_target)
        timestamp = i / fps

        poses.append(CameraPose(
            frame_idx=i,
            transform=transform,
            timestamp=timestamp,
        ))

    return CameraTrajectory(
        trajectory_id=trajectory_id,
        trajectory_type=TrajectoryType.APPROACH,
        poses=poses,
        fps=fps,
        description=f"Approach toward target at {target_position}",
    )


def generate_orbit_trajectory(
    center_position: np.ndarray,
    radius: float = 1.5,
    start_angle: float = 0.0,
    end_angle: float = 90.0,
    height: float = 1.6,
    look_at_height: float = 1.0,
    num_frames: int = 49,
    fps: float = 24.0,
    trajectory_id: Optional[str] = None,
) -> CameraTrajectory:
    """
    Generate an orbit trajectory around a point.

    Useful for inspecting objects from multiple angles.

    Args:
        center_position: Center point to orbit around
        radius: Orbit radius
        start_angle: Starting angle (degrees)
        end_angle: Ending angle (degrees)
        height: Camera height
        look_at_height: Height of look-at point
        num_frames: Number of frames
        fps: Frames per second
        trajectory_id: Optional trajectory ID

    Returns:
        CameraTrajectory orbiting the center
    """
    if trajectory_id is None:
        trajectory_id = f"orbit_{uuid.uuid4().hex[:8]}"

    center = np.asarray(center_position, dtype=np.float64)

    poses = []

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        t_smooth = ease_in_out(t)

        angle = math.radians(start_angle + (end_angle - start_angle) * t_smooth)

        x = center[0] + radius * math.sin(angle)
        z = center[2] + radius * math.cos(angle)
        y = height

        eye = np.array([x, y, z])
        look_target = np.array([center[0], look_at_height, center[2]])

        transform = look_at_matrix(eye, look_target)
        timestamp = i / fps

        poses.append(CameraPose(
            frame_idx=i,
            transform=transform,
            timestamp=timestamp,
        ))

    return CameraTrajectory(
        trajectory_id=trajectory_id,
        trajectory_type=TrajectoryType.ORBIT,
        poses=poses,
        fps=fps,
        description=f"Orbit around {center_position}, radius={radius}",
    )


def generate_walkthrough_trajectory(
    start_position: np.ndarray,
    end_position: np.ndarray,
    look_ahead_distance: float = 2.0,
    eye_height: float = 1.6,
    num_frames: int = 49,
    fps: float = 24.0,
    trajectory_id: Optional[str] = None,
) -> CameraTrajectory:
    """
    Generate a linear walkthrough trajectory.

    Simulates walking through a room in a straight line.

    Args:
        start_position: Starting XZ position
        end_position: Ending XZ position
        look_ahead_distance: How far ahead to look
        eye_height: Camera height
        num_frames: Number of frames
        fps: Frames per second
        trajectory_id: Optional trajectory ID

    Returns:
        CameraTrajectory for walkthrough
    """
    if trajectory_id is None:
        trajectory_id = f"walkthrough_{uuid.uuid4().hex[:8]}"

    start = np.asarray(start_position, dtype=np.float64)
    end = np.asarray(end_position, dtype=np.float64)

    # Compute direction
    direction = end - start
    direction_len = np.linalg.norm(direction)
    if direction_len > 1e-8:
        direction = direction / direction_len

    poses = []

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        t_smooth = ease_in_out(t)

        # Position along path
        pos_xz = start + (end - start) * t_smooth

        # Add walking bob
        bob = math.sin(t * math.pi * 6) * 0.015
        sway = math.sin(t * math.pi * 3) * 0.01

        eye = np.array([
            pos_xz[0] + sway,
            eye_height + bob,
            pos_xz[1] if len(pos_xz) == 2 else pos_xz[2],
        ])

        # Look ahead
        look_pos = pos_xz + direction[:2] * look_ahead_distance if len(direction) >= 2 else pos_xz
        look_target = np.array([
            look_pos[0],
            eye_height - 0.1,  # Slightly downward gaze
            look_pos[1] if len(look_pos) == 2 else look_pos[2],
        ])

        transform = look_at_matrix(eye, look_target)
        timestamp = i / fps

        poses.append(CameraPose(
            frame_idx=i,
            transform=transform,
            timestamp=timestamp,
        ))

    return CameraTrajectory(
        trajectory_id=trajectory_id,
        trajectory_type=TrajectoryType.WALKTHROUGH,
        poses=poses,
        fps=fps,
        description=f"Walkthrough from {start_position} to {end_position}",
    )


def generate_reach_manipulate_trajectory(
    object_position: np.ndarray,
    action_type: HandActionType = HandActionType.GRASP,
    approach_distance: float = 0.8,
    manipulation_distance: float = 0.4,
    eye_height: float = 1.6,
    num_frames: int = 49,
    fps: float = 24.0,
    trajectory_id: Optional[str] = None,
) -> CameraTrajectory:
    """
    Generate trajectory for reaching and manipulating an object.

    This is the primary trajectory type for DWM - simulates a person
    reaching toward and manipulating an object while looking at it.

    The trajectory has phases:
    1. Approach phase (frames 0-20): Move toward object
    2. Reach phase (frames 20-35): Steady position, focus on object
    3. Manipulation phase (frames 35-49): Slight movement during action

    Args:
        object_position: Position of target object
        action_type: Type of manipulation (affects trajectory subtleties)
        approach_distance: Starting distance
        manipulation_distance: Final distance during manipulation
        eye_height: Camera height
        num_frames: Number of frames (DWM default: 49)
        fps: Frames per second
        trajectory_id: Optional trajectory ID

    Returns:
        CameraTrajectory for reach-and-manipulate motion
    """
    if trajectory_id is None:
        trajectory_id = f"reach_{action_type.value}_{uuid.uuid4().hex[:8]}"

    target = np.asarray(object_position, dtype=np.float64)

    # Phase boundaries (normalized 0-1)
    approach_end = 0.4
    reach_end = 0.7

    # Slight random offset for variety
    angle_offset = np.random.uniform(-15, 15)
    angle_rad = math.radians(angle_offset)

    poses = []

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)

        if t < approach_end:
            # Approach phase
            phase_t = t / approach_end
            phase_t = ease_in_out(phase_t)
            distance = approach_distance - (approach_distance - manipulation_distance - 0.1) * phase_t
            head_bob = math.sin(phase_t * math.pi * 2) * 0.01

        elif t < reach_end:
            # Reach phase - steady, focused
            phase_t = (t - approach_end) / (reach_end - approach_end)
            distance = manipulation_distance + 0.1 - 0.1 * phase_t
            head_bob = 0.0

        else:
            # Manipulation phase - slight movement
            phase_t = (t - reach_end) / (1 - reach_end)
            distance = manipulation_distance

            # Movement depends on action type
            if action_type in [HandActionType.PULL]:
                # Pull back slightly
                distance += 0.05 * phase_t
                head_bob = -0.02 * phase_t
            elif action_type in [HandActionType.PUSH]:
                # Lean forward slightly
                distance -= 0.03 * phase_t
                head_bob = 0.01 * phase_t
            elif action_type in [HandActionType.LIFT]:
                # Follow object up
                head_bob = 0.03 * phase_t
            else:
                head_bob = 0.0

        # Compute position
        x = target[0] + distance * math.sin(angle_rad)
        z = target[2] + distance * math.cos(angle_rad)
        y = eye_height + head_bob

        eye = np.array([x, y, z])

        # Look at object (with slight adjustment based on phase)
        if t < approach_end:
            look_y = target[1] + 0.1
        else:
            look_y = target[1]  # Direct focus during manipulation

        look_target = np.array([target[0], look_y, target[2]])

        transform = look_at_matrix(eye, look_target)
        timestamp = i / fps

        poses.append(CameraPose(
            frame_idx=i,
            transform=transform,
            timestamp=timestamp,
        ))

    return CameraTrajectory(
        trajectory_id=trajectory_id,
        trajectory_type=TrajectoryType.REACH_MANIPULATE,
        poses=poses,
        fps=fps,
        action_type=action_type,
        description=f"{action_type.value} action toward object at {object_position}",
    )


class EgocentricTrajectoryGenerator:
    """
    Main class for generating egocentric trajectories for DWM.

    Usage:
        generator = EgocentricTrajectoryGenerator(scene_context)
        trajectories = generator.generate_for_scene(
            num_trajectories=5,
            trajectory_types=[TrajectoryType.APPROACH, TrajectoryType.REACH_MANIPULATE],
        )
    """

    def __init__(
        self,
        scene_context: SceneContext,
        default_fps: float = 24.0,
        default_num_frames: int = 49,
    ):
        """
        Initialize trajectory generator.

        Args:
            scene_context: Context about the scene (objects, dimensions)
            default_fps: Default frames per second
            default_num_frames: Default number of frames (DWM uses 49)
        """
        self.scene = scene_context
        self.fps = default_fps
        self.num_frames = default_num_frames

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "EgocentricTrajectoryGenerator":
        """
        Create generator from scene manifest.

        Args:
            manifest_path: Path to scene_manifest.json

        Returns:
            Configured EgocentricTrajectoryGenerator
        """
        manifest = json.loads(manifest_path.read_text())

        # Extract room dimensions
        room = manifest.get("scene", {}).get("room", {})
        dims = room.get("dimensions", {})

        # Extract objects
        objects = {}
        manipulable = []
        articulated = []

        for obj in manifest.get("objects", []):
            obj_id = obj.get("id", "unknown")
            pos = obj.get("transform", {}).get("position", {})
            position = [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]

            objects[obj_id] = {
                "position": position,
                "bounds": obj.get("bounds", {}),
                "category": obj.get("category", ""),
                "sim_role": obj.get("sim_role", "static"),
            }

            sim_role = obj.get("sim_role", "static")
            if sim_role in ["manipulable_object", "clutter"]:
                manipulable.append(obj_id)
            elif sim_role in ["articulated_furniture", "articulated_appliance"]:
                articulated.append(obj_id)

        context = SceneContext(
            room_width=dims.get("width", 5.0),
            room_depth=dims.get("depth", 5.0),
            room_height=dims.get("height", 3.0),
            objects=objects,
            manipulable_objects=manipulable,
            articulated_objects=articulated,
        )

        return cls(context)

    def generate_for_object(
        self,
        object_id: str,
        trajectory_types: Optional[list[TrajectoryType]] = None,
        action_types: Optional[list[HandActionType]] = None,
    ) -> list[CameraTrajectory]:
        """
        Generate trajectories targeting a specific object.

        Args:
            object_id: ID of target object
            trajectory_types: Types of trajectories to generate
            action_types: Types of hand actions (for REACH_MANIPULATE)

        Returns:
            List of generated trajectories
        """
        if trajectory_types is None:
            trajectory_types = [TrajectoryType.APPROACH, TrajectoryType.REACH_MANIPULATE]

        if action_types is None:
            action_types = [HandActionType.GRASP]

        obj = self.scene.objects.get(object_id)
        if obj is None:
            raise ValueError(f"Object {object_id} not found in scene")

        position = np.array(obj["position"])
        trajectories = []

        for traj_type in trajectory_types:
            if traj_type == TrajectoryType.APPROACH:
                for angle in [0, 45, -45]:
                    traj = generate_approach_trajectory(
                        target_position=position,
                        approach_angle=angle,
                        num_frames=self.num_frames,
                        fps=self.fps,
                        eye_height=self.scene.eye_height,
                    )
                    traj.target_object_id = object_id
                    trajectories.append(traj)

            elif traj_type == TrajectoryType.REACH_MANIPULATE:
                for action in action_types:
                    traj = generate_reach_manipulate_trajectory(
                        object_position=position,
                        action_type=action,
                        num_frames=self.num_frames,
                        fps=self.fps,
                        eye_height=self.scene.eye_height,
                    )
                    traj.target_object_id = object_id
                    trajectories.append(traj)

            elif traj_type == TrajectoryType.ORBIT:
                traj = generate_orbit_trajectory(
                    center_position=position,
                    num_frames=self.num_frames,
                    fps=self.fps,
                )
                traj.target_object_id = object_id
                trajectories.append(traj)

        return trajectories

    def generate_for_scene(
        self,
        num_trajectories: int = 5,
        trajectory_types: Optional[list[TrajectoryType]] = None,
        action_types: Optional[list[HandActionType]] = None,
        target_object_ids: Optional[list[str]] = None,
    ) -> list[CameraTrajectory]:
        """
        Generate trajectories for the entire scene.

        Args:
            num_trajectories: Approximate number of trajectories to generate
            trajectory_types: Types of trajectories
            action_types: Types of hand actions
            target_object_ids: Specific objects to target (None = auto-select)

        Returns:
            List of generated trajectories
        """
        if trajectory_types is None:
            trajectory_types = [
                TrajectoryType.APPROACH,
                TrajectoryType.REACH_MANIPULATE,
                TrajectoryType.WALKTHROUGH,
            ]

        if action_types is None:
            action_types = [
                HandActionType.GRASP,
                HandActionType.PULL,
                HandActionType.PUSH,
            ]

        trajectories = []

        # Select target objects
        if target_object_ids is None:
            # Prioritize manipulable and articulated objects
            targets = (
                self.scene.manipulable_objects +
                self.scene.articulated_objects
            )
            if not targets:
                targets = list(self.scene.objects.keys())[:3]
        else:
            targets = target_object_ids

        # Generate trajectories for each target
        trajectories_per_object = max(1, num_trajectories // max(1, len(targets)))

        for obj_id in targets:
            obj_trajs = self.generate_for_object(
                obj_id,
                trajectory_types=trajectory_types,
                action_types=action_types,
            )
            trajectories.extend(obj_trajs[:trajectories_per_object])

        # Add some walkthrough trajectories
        if TrajectoryType.WALKTHROUGH in trajectory_types:
            # Generate walks across the room
            room_w = self.scene.room_width
            room_d = self.scene.room_depth

            walks = [
                (np.array([-room_w * 0.3, -room_d * 0.3]),
                 np.array([room_w * 0.3, room_d * 0.3])),
                (np.array([room_w * 0.3, -room_d * 0.3]),
                 np.array([-room_w * 0.3, room_d * 0.3])),
            ]

            for start, end in walks[:2]:
                traj = generate_walkthrough_trajectory(
                    start_position=start,
                    end_position=end,
                    num_frames=self.num_frames,
                    fps=self.fps,
                    eye_height=self.scene.eye_height,
                )
                trajectories.append(traj)

        return trajectories[:num_trajectories]

    def export_trajectory(
        self,
        trajectory: CameraTrajectory,
        output_path: Path,
    ) -> Path:
        """
        Export trajectory to JSON format.

        Args:
            trajectory: Trajectory to export
            output_path: Output file path

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "trajectory_id": trajectory.trajectory_id,
            "trajectory_type": trajectory.trajectory_type.value,
            "num_frames": trajectory.num_frames,
            "fps": trajectory.fps,
            "duration_seconds": trajectory.duration,
            "resolution": list(trajectory.resolution),
            "focal_length": trajectory.focal_length,
            "target_object_id": trajectory.target_object_id,
            "action_type": trajectory.action_type.value if trajectory.action_type else None,
            "description": trajectory.description,
            "poses": [],
        }

        for pose in trajectory.poses:
            data["poses"].append({
                "frame_idx": pose.frame_idx,
                "timestamp": pose.timestamp,
                "transform": pose.transform.tolist(),
                "position": pose.position.tolist(),
            })

        output_path.write_text(json.dumps(data, indent=2))
        return output_path
