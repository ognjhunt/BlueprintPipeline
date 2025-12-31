#!/usr/bin/env python3
"""
Constraint-Preserving Data Augmentation (CP-Gen Style).

This module implements the core insight from CP-Gen (CoRL 2025):
- Preserve contact manipulation constraints when generating variations
- Transform skill segments using keypoint-trajectory constraints
- Replan free-space motions for collision-free paths

Reference:
- CP-Gen: https://cp-gen.github.io/
- arXiv: https://arxiv.org/abs/2508.03944

Key Concepts:
1. Skill Segment Preservation: Contact-rich manipulation segments are
   preserved/transformed carefully using keypoint constraints.

2. Free-Space Replanning: Motion segments in free space are replanned
   using motion planning for each new object configuration.

3. Keypoint-Trajectory Constraints: Keypoints on robot/grasped object
   must track reference trajectories relative to task objects.

This is the "secret sauce" for generating thousands of valid episodes
from a single seed episode - preserving what matters (contact physics)
while adapting what can change (free-space motion).
"""

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task_specifier import (
    TaskSpecification,
    SkillSegment,
    SegmentType,
    Keypoint,
    KeypointConstraint,
    ConstraintType,
)
from motion_planner import MotionPlan, Waypoint, MotionPhase


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ObjectTransform:
    """Transform representing a change in object pose/geometry."""

    object_id: str

    # Position change
    position_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Rotation change (as quaternion: w, x, y, z)
    rotation_offset: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))

    # Scale change (for geometry variations)
    scale_factor: np.ndarray = field(default_factory=lambda: np.ones(3))

    def apply_to_position(self, pos: np.ndarray) -> np.ndarray:
        """Apply transform to a position."""
        # Apply rotation
        rot = Rotation.from_quat([
            self.rotation_offset[1],  # x
            self.rotation_offset[2],  # y
            self.rotation_offset[3],  # z
            self.rotation_offset[0],  # w
        ])
        rotated = rot.apply(pos * self.scale_factor)

        # Apply translation
        return rotated + self.position_offset

    def apply_to_orientation(self, quat: np.ndarray) -> np.ndarray:
        """Apply rotation to orientation."""
        # Convert to scipy format
        q1 = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        q2 = Rotation.from_quat([
            self.rotation_offset[1],
            self.rotation_offset[2],
            self.rotation_offset[3],
            self.rotation_offset[0],
        ])

        # Compose rotations
        composed = q2 * q1
        result = composed.as_quat()  # x, y, z, w

        return np.array([result[3], result[0], result[1], result[2]])  # w, x, y, z


@dataclass
class SeedEpisode:
    """
    A seed episode that can be augmented to create variations.

    Contains the original trajectory decomposed into segments
    with preserved skill constraints.
    """

    episode_id: str
    task_spec: TaskSpecification
    motion_plan: MotionPlan

    # Decomposed segments with waypoints
    segment_waypoints: Dict[str, List[Waypoint]] = field(default_factory=dict)

    # Reference object poses at seed configuration
    reference_object_poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_spec": self.task_spec.to_dict(),
            "motion_plan": self.motion_plan.to_dict(),
            "segment_waypoints": {
                seg_id: [w.to_dict() for w in waypoints]
                for seg_id, waypoints in self.segment_waypoints.items()
            },
            "reference_object_poses": {
                obj_id: {"position": pos.tolist(), "orientation": ori.tolist()}
                for obj_id, (pos, ori) in self.reference_object_poses.items()
            },
        }


@dataclass
class AugmentedEpisode:
    """
    An augmented episode generated from a seed.

    Contains transformed segments with new waypoints
    that preserve skill constraints.
    """

    episode_id: str
    seed_episode_id: str
    variation_index: int

    # New motion plan
    motion_plan: MotionPlan

    # Object transforms applied
    object_transforms: Dict[str, ObjectTransform] = field(default_factory=dict)

    # Augmentation quality metrics
    constraint_satisfaction: float = 1.0  # 0-1, how well constraints are preserved
    collision_free: bool = True
    planning_success: bool = True

    # Generation metadata
    augmentation_method: str = "cpgen"
    generation_time_seconds: float = 0.0


# =============================================================================
# Constraint Solver
# =============================================================================


class ConstraintSolver:
    """
    Solves for waypoint positions that satisfy keypoint constraints.

    Given:
    - A set of keypoint-trajectory constraints
    - Object transforms for the new configuration

    Finds waypoint positions that:
    - Place keypoints at correct positions relative to objects
    - Maintain orientation constraints
    - Satisfy contact/clearance requirements
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[CONSTRAINT-SOLVER] {msg}")

    def solve_waypoint(
        self,
        original_waypoint: Waypoint,
        constraints: List[KeypointConstraint],
        keypoints: List[Keypoint],
        object_transforms: Dict[str, ObjectTransform],
        original_object_poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Waypoint:
        """
        Solve for new waypoint position given transformed objects.

        Args:
            original_waypoint: Original waypoint in seed episode
            constraints: Active constraints for this waypoint
            keypoints: Keypoint definitions
            object_transforms: Transforms for each object
            original_object_poses: Original object poses

        Returns:
            New waypoint with transformed position
        """
        if not constraints:
            # No constraints - keep original
            return self._copy_waypoint(original_waypoint)

        # Start from original position
        new_position = original_waypoint.position.copy()
        new_orientation = original_waypoint.orientation.copy()

        # Apply each constraint
        for constraint in sorted(constraints, key=lambda c: -c.priority):
            if constraint.reference_object_id:
                transform = object_transforms.get(constraint.reference_object_id)
                if transform is None:
                    continue

                if constraint.constraint_type == ConstraintType.RELATIVE_POSITION:
                    # Keypoint should be at offset from object
                    if constraint.reference_offset is not None:
                        # Get original object position
                        orig_pos, orig_ori = original_object_poses.get(
                            constraint.reference_object_id,
                            (np.zeros(3), np.array([1, 0, 0, 0]))
                        )

                        # Transform the offset to new configuration
                        new_offset = transform.apply_to_position(constraint.reference_offset)
                        new_obj_pos = transform.apply_to_position(orig_pos)

                        # Find keypoint and compute EE position
                        keypoint = self._find_keypoint(constraint.keypoint_id, keypoints)
                        if keypoint:
                            # EE position = target keypoint position - keypoint offset
                            target_keypoint_pos = new_obj_pos + new_offset
                            new_position = target_keypoint_pos - keypoint.local_position

                elif constraint.constraint_type == ConstraintType.TRAJECTORY:
                    # Keypoint should follow trajectory relative to object
                    if constraint.trajectory_waypoints:
                        # Get time-appropriate waypoint in trajectory
                        t = original_waypoint.timestamp
                        t_norm = (t - constraint.start_time) / max(
                            0.001, constraint.end_time - constraint.start_time
                        )
                        t_norm = np.clip(t_norm, 0, 1)

                        # Interpolate trajectory
                        traj_offset = self._interpolate_trajectory(
                            constraint.trajectory_waypoints, t_norm
                        )

                        # Transform trajectory point
                        orig_pos, _ = original_object_poses.get(
                            constraint.reference_object_id,
                            (np.zeros(3), np.array([1, 0, 0, 0]))
                        )
                        new_traj_offset = transform.apply_to_position(traj_offset)
                        new_obj_pos = transform.apply_to_position(orig_pos)

                        keypoint = self._find_keypoint(constraint.keypoint_id, keypoints)
                        if keypoint:
                            target_keypoint_pos = new_obj_pos + new_traj_offset
                            new_position = target_keypoint_pos - keypoint.local_position

                elif constraint.constraint_type == ConstraintType.POSITION:
                    # Absolute position (already transformed in constraint)
                    if constraint.reference_offset is not None:
                        keypoint = self._find_keypoint(constraint.keypoint_id, keypoints)
                        if keypoint:
                            new_position = transform.apply_to_position(
                                constraint.reference_offset
                            ) - keypoint.local_position

        # Create new waypoint
        return Waypoint(
            position=new_position,
            orientation=new_orientation,
            gripper_aperture=original_waypoint.gripper_aperture,
            timestamp=original_waypoint.timestamp,
            duration_to_next=original_waypoint.duration_to_next,
            phase=original_waypoint.phase,
            max_velocity=original_waypoint.max_velocity,
            max_acceleration=original_waypoint.max_acceleration,
        )

    def _copy_waypoint(self, wp: Waypoint) -> Waypoint:
        """Create a copy of a waypoint."""
        return Waypoint(
            position=wp.position.copy(),
            orientation=wp.orientation.copy(),
            gripper_aperture=wp.gripper_aperture,
            timestamp=wp.timestamp,
            duration_to_next=wp.duration_to_next,
            phase=wp.phase,
            max_velocity=wp.max_velocity,
            max_acceleration=wp.max_acceleration,
        )

    def _find_keypoint(self, keypoint_id: str, keypoints: List[Keypoint]) -> Optional[Keypoint]:
        """Find keypoint by ID."""
        for kp in keypoints:
            if kp.keypoint_id == keypoint_id:
                return kp
        return None

    def _interpolate_trajectory(
        self,
        waypoints: List[np.ndarray],
        t: float,
    ) -> np.ndarray:
        """Interpolate along trajectory waypoints."""
        if not waypoints:
            return np.zeros(3)
        if len(waypoints) == 1:
            return waypoints[0].copy()

        # Find segment
        num_segments = len(waypoints) - 1
        segment_idx = int(t * num_segments)
        segment_idx = min(segment_idx, num_segments - 1)

        # Local t within segment
        local_t = (t * num_segments) - segment_idx
        local_t = np.clip(local_t, 0, 1)

        # Linear interpolation
        p1 = waypoints[segment_idx]
        p2 = waypoints[segment_idx + 1]

        return (1 - local_t) * p1 + local_t * p2


# =============================================================================
# Motion Planner Interface
# =============================================================================


class FreeSpaceMotionPlanner:
    """
    Motion planner for free-space segments.

    Uses RRT-style planning to find collision-free paths
    between skill segments.
    """

    def __init__(
        self,
        robot_type: str = "franka",
        verbose: bool = False,
    ):
        self.robot_type = robot_type
        self.verbose = verbose

        # Robot workspace limits
        self.workspace_min = np.array([-0.8, -0.8, 0.0])
        self.workspace_max = np.array([0.8, 0.8, 1.2])

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[MOTION-PLANNER] {msg}")

    def plan_path(
        self,
        start_waypoint: Waypoint,
        end_waypoint: Waypoint,
        obstacles: List[Dict[str, Any]],
        duration: float = 0.5,
    ) -> List[Waypoint]:
        """
        Plan a collision-free path between waypoints.

        Args:
            start_waypoint: Starting configuration
            end_waypoint: Goal configuration
            obstacles: List of obstacle objects with position and dimensions
            duration: Desired duration for the path

        Returns:
            List of intermediate waypoints (including start and end)
        """
        start_pos = start_waypoint.position
        end_pos = end_waypoint.position

        # Check if straight-line path is collision-free
        if self._is_path_collision_free(start_pos, end_pos, obstacles):
            # Simple linear interpolation
            return self._interpolate_straight(start_waypoint, end_waypoint, duration)

        # Use RRT-style planning for collision avoidance
        path = self._rrt_plan(start_pos, end_pos, obstacles)

        if path is None:
            self.log("RRT planning failed, using straight path")
            return self._interpolate_straight(start_waypoint, end_waypoint, duration)

        # Convert path to waypoints
        return self._path_to_waypoints(path, start_waypoint, end_waypoint, duration)

    def _is_path_collision_free(
        self,
        start: np.ndarray,
        end: np.ndarray,
        obstacles: List[Dict[str, Any]],
    ) -> bool:
        """Check if straight-line path is collision-free."""
        num_checks = 10
        for i in range(num_checks):
            t = i / (num_checks - 1)
            pos = (1 - t) * start + t * end

            for obs in obstacles:
                if self._point_in_obstacle(pos, obs):
                    return False

        return True

    def _point_in_obstacle(
        self,
        point: np.ndarray,
        obstacle: Dict[str, Any],
    ) -> bool:
        """Check if point is inside obstacle (with padding)."""
        obs_pos = np.array(obstacle.get("position", [0, 0, 0]))
        obs_dims = np.array(obstacle.get("dimensions", [0.1, 0.1, 0.1]))

        # Padding for safety
        padding = 0.05

        half_dims = obs_dims / 2 + padding

        # AABB collision check
        return np.all(np.abs(point - obs_pos) < half_dims)

    def _rrt_plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Dict[str, Any]],
        max_iters: int = 500,
        step_size: float = 0.1,
    ) -> Optional[List[np.ndarray]]:
        """
        Simple RRT planner.

        Returns path as list of positions, or None if planning fails.
        """
        # Tree nodes: list of (position, parent_index)
        tree = [(start.copy(), -1)]

        for _ in range(max_iters):
            # Sample random point (with goal bias)
            if np.random.random() < 0.2:
                sample = goal.copy()
            else:
                sample = np.random.uniform(self.workspace_min, self.workspace_max)

            # Find nearest node
            distances = [np.linalg.norm(n[0] - sample) for n in tree]
            nearest_idx = np.argmin(distances)
            nearest_pos = tree[nearest_idx][0]

            # Extend toward sample
            direction = sample - nearest_pos
            dist = np.linalg.norm(direction)
            if dist < 0.01:
                continue

            direction = direction / dist
            new_pos = nearest_pos + direction * min(step_size, dist)

            # Check collision
            if not self._is_path_collision_free(nearest_pos, new_pos, obstacles):
                continue

            # Add to tree
            tree.append((new_pos, nearest_idx))

            # Check if reached goal
            if np.linalg.norm(new_pos - goal) < step_size:
                # Reconstruct path
                path = [goal]
                idx = len(tree) - 1
                while idx >= 0:
                    path.append(tree[idx][0])
                    idx = tree[idx][1]
                return list(reversed(path))

        return None

    def _interpolate_straight(
        self,
        start: Waypoint,
        end: Waypoint,
        duration: float,
    ) -> List[Waypoint]:
        """Create straight-line path with intermediate waypoints."""
        num_points = max(3, int(duration * 10))  # ~10 Hz intermediate points
        waypoints = []

        for i in range(num_points):
            t = i / (num_points - 1)

            pos = (1 - t) * start.position + t * end.position
            ori = self._slerp_quat(start.orientation, end.orientation, t)
            gripper = (1 - t) * start.gripper_aperture + t * end.gripper_aperture

            wp = Waypoint(
                position=pos,
                orientation=ori,
                gripper_aperture=gripper,
                timestamp=start.timestamp + t * duration,
                duration_to_next=duration / num_points if i < num_points - 1 else 0,
                phase=start.phase if t < 0.5 else end.phase,
                max_velocity=start.max_velocity,
                max_acceleration=start.max_acceleration,
            )
            waypoints.append(wp)

        return waypoints

    def _path_to_waypoints(
        self,
        path: List[np.ndarray],
        start: Waypoint,
        end: Waypoint,
        duration: float,
    ) -> List[Waypoint]:
        """Convert RRT path to waypoints."""
        waypoints = []
        num_points = len(path)
        dt = duration / (num_points - 1) if num_points > 1 else 0

        for i, pos in enumerate(path):
            t = i / (num_points - 1) if num_points > 1 else 0

            ori = self._slerp_quat(start.orientation, end.orientation, t)
            gripper = (1 - t) * start.gripper_aperture + t * end.gripper_aperture

            wp = Waypoint(
                position=pos.copy(),
                orientation=ori,
                gripper_aperture=gripper,
                timestamp=start.timestamp + i * dt,
                duration_to_next=dt if i < num_points - 1 else 0,
                phase=start.phase if t < 0.5 else end.phase,
                max_velocity=start.max_velocity,
                max_acceleration=start.max_acceleration,
            )
            waypoints.append(wp)

        return waypoints

    def _slerp_quat(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation for quaternions."""
        # Ensure same hemisphere
        if np.dot(q1, q2) < 0:
            q2 = -q2

        # Compute angle
        dot = np.clip(np.dot(q1, q2), -1, 1)
        theta = np.arccos(dot)

        if theta < 0.001:
            return q1.copy()

        # SLERP
        sin_theta = np.sin(theta)
        result = (np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / sin_theta

        return result / np.linalg.norm(result)


# =============================================================================
# Constraint-Preserving Augmenter
# =============================================================================


class ConstraintPreservingAugmenter:
    """
    Main CP-Gen style augmentation engine.

    Takes a seed episode and generates variations by:
    1. Transforming object configurations
    2. Preserving skill segment constraints
    3. Replanning free-space motions

    This is the key to generating thousands of valid episodes
    from a single demonstration.
    """

    def __init__(
        self,
        robot_type: str = "franka",
        verbose: bool = True,
    ):
        self.robot_type = robot_type
        self.verbose = verbose

        self.constraint_solver = ConstraintSolver(verbose=verbose)
        self.motion_planner = FreeSpaceMotionPlanner(robot_type=robot_type, verbose=verbose)

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[CPGEN-AUGMENTER] [{level}] {msg}")

    def create_seed_episode(
        self,
        task_spec: TaskSpecification,
        motion_plan: MotionPlan,
        scene_objects: List[Dict[str, Any]],
    ) -> SeedEpisode:
        """
        Create a seed episode from a task specification and motion plan.

        Args:
            task_spec: Task specification with segments and constraints
            motion_plan: Original motion plan with waypoints
            scene_objects: Objects in the scene

        Returns:
            SeedEpisode ready for augmentation
        """
        self.log("Creating seed episode")

        # Decompose motion plan into segments
        segment_waypoints = self._decompose_motion_plan(
            motion_plan=motion_plan,
            segments=task_spec.segments,
        )

        # Extract reference object poses
        reference_poses = {}
        for obj in scene_objects:
            obj_id = obj.get("id", obj.get("name", ""))
            pos = np.array(obj.get("position", [0, 0, 0]))
            ori = np.array(obj.get("orientation", [1, 0, 0, 0]))
            reference_poses[obj_id] = (pos, ori)

        return SeedEpisode(
            episode_id=f"seed_{motion_plan.plan_id}",
            task_spec=task_spec,
            motion_plan=motion_plan,
            segment_waypoints=segment_waypoints,
            reference_object_poses=reference_poses,
        )

    def augment(
        self,
        seed: SeedEpisode,
        object_transforms: Dict[str, ObjectTransform],
        obstacles: List[Dict[str, Any]],
        variation_index: int = 0,
    ) -> AugmentedEpisode:
        """
        Generate an augmented episode from a seed.

        Args:
            seed: Seed episode to augment
            object_transforms: Transforms for each object
            obstacles: Updated obstacle list with new positions
            variation_index: Index of this variation

        Returns:
            AugmentedEpisode with transformed trajectory
        """
        import time
        start_time = time.time()

        self.log(f"Augmenting for variation {variation_index}")

        new_waypoints = []
        constraint_violations = 0
        total_constraints = 0

        # Process each segment
        prev_waypoint = None
        for segment in seed.task_spec.segments:
            seg_id = segment.segment_id
            seg_waypoints = seed.segment_waypoints.get(seg_id, [])

            if not seg_waypoints:
                continue

            if segment.segment_type == SegmentType.SKILL:
                # Skill segment: preserve constraints
                self.log(f"  Transforming skill segment: {segment.skill_name}")

                # Get constraints for this segment
                segment_constraints = segment.constraints + [
                    c for c in seed.task_spec.constraints
                    if c.start_time <= segment.end_time and c.end_time >= segment.start_time
                ]

                # Transform each waypoint
                transformed = []
                for wp in seg_waypoints:
                    new_wp = self.constraint_solver.solve_waypoint(
                        original_waypoint=wp,
                        constraints=segment_constraints,
                        keypoints=seed.task_spec.keypoints + segment.keypoints,
                        object_transforms=object_transforms,
                        original_object_poses=seed.reference_object_poses,
                    )
                    transformed.append(new_wp)

                    # Track constraint satisfaction
                    total_constraints += len(segment_constraints)

                # Add motion plan connection if needed
                if prev_waypoint is not None and transformed:
                    # Plan connection from previous segment
                    connection = self.motion_planner.plan_path(
                        start_waypoint=prev_waypoint,
                        end_waypoint=transformed[0],
                        obstacles=obstacles,
                        duration=0.1,
                    )
                    new_waypoints.extend(connection[:-1])  # Exclude last (it's the start of skill)

                new_waypoints.extend(transformed)

            else:
                # Free-space segment: replan
                self.log(f"  Replanning free-space segment: {segment.skill_name}")

                if len(seg_waypoints) >= 2:
                    start_wp = seg_waypoints[0]
                    end_wp = seg_waypoints[-1]

                    # Transform start/end based on adjacent segments
                    start_wp = self._transform_freespace_endpoint(
                        start_wp, segment, seed, object_transforms, "start"
                    )
                    end_wp = self._transform_freespace_endpoint(
                        end_wp, segment, seed, object_transforms, "end"
                    )

                    # Plan new path
                    planned = self.motion_planner.plan_path(
                        start_waypoint=start_wp,
                        end_waypoint=end_wp,
                        obstacles=obstacles,
                        duration=segment.end_time - segment.start_time,
                    )

                    # Connect to previous if needed
                    if prev_waypoint is not None and planned:
                        if np.linalg.norm(prev_waypoint.position - planned[0].position) > 0.01:
                            connection = self.motion_planner.plan_path(
                                start_waypoint=prev_waypoint,
                                end_waypoint=planned[0],
                                obstacles=obstacles,
                                duration=0.1,
                            )
                            new_waypoints.extend(connection[:-1])

                    new_waypoints.extend(planned)
                else:
                    # Single waypoint - transform it
                    if seg_waypoints:
                        wp = self._transform_freespace_endpoint(
                            seg_waypoints[0], segment, seed, object_transforms, "both"
                        )
                        new_waypoints.append(wp)

            if new_waypoints:
                prev_waypoint = new_waypoints[-1]

        # Calculate constraint satisfaction
        constraint_satisfaction = 1.0 - (constraint_violations / max(1, total_constraints))

        # Check for collisions
        collision_free = self._check_collision_free(new_waypoints, obstacles)

        # Create new motion plan
        new_plan = MotionPlan(
            plan_id=f"aug_{seed.motion_plan.plan_id}_v{variation_index}",
            task_name=seed.motion_plan.task_name,
            task_description=seed.motion_plan.task_description,
            waypoints=new_waypoints,
            target_object_id=seed.motion_plan.target_object_id,
            target_object_position=self._transform_position(
                seed.motion_plan.target_object_position,
                seed.motion_plan.target_object_id,
                object_transforms,
                seed.reference_object_poses,
            ) if seed.motion_plan.target_object_position is not None else None,
            place_position=self._transform_position(
                seed.motion_plan.place_position,
                seed.motion_plan.target_object_id,
                object_transforms,
                seed.reference_object_poses,
            ) if seed.motion_plan.place_position is not None else None,
            robot_type=seed.motion_plan.robot_type,
        )

        return AugmentedEpisode(
            episode_id=f"aug_{seed.episode_id}_v{variation_index}",
            seed_episode_id=seed.episode_id,
            variation_index=variation_index,
            motion_plan=new_plan,
            object_transforms=object_transforms,
            constraint_satisfaction=constraint_satisfaction,
            collision_free=collision_free,
            planning_success=len(new_waypoints) > 0,
            generation_time_seconds=time.time() - start_time,
        )

    def generate_variations(
        self,
        seed: SeedEpisode,
        scene_objects: List[Dict[str, Any]],
        num_variations: int = 100,
        position_noise_std: float = 0.05,
        rotation_noise_std: float = 0.1,
    ) -> List[AugmentedEpisode]:
        """
        Generate multiple variations from a seed episode.

        Args:
            seed: Seed episode
            scene_objects: Original scene objects
            num_variations: Number of variations to generate
            position_noise_std: Standard deviation for position noise (meters)
            rotation_noise_std: Standard deviation for rotation noise (radians)

        Returns:
            List of augmented episodes
        """
        self.log(f"Generating {num_variations} variations")

        augmented_episodes = []

        for var_idx in range(num_variations):
            # Generate random transforms for relevant objects
            object_transforms = {}
            updated_obstacles = []

            for obj in scene_objects:
                obj_id = obj.get("id", obj.get("name", ""))

                # Random position offset
                pos_offset = np.random.randn(3) * position_noise_std
                pos_offset[2] = 0  # Keep z (height) stable

                # Random rotation (around z-axis only for tabletop)
                angle = np.random.randn() * rotation_noise_std
                rot_quat = np.array([
                    np.cos(angle / 2),
                    0, 0,
                    np.sin(angle / 2)
                ])

                transform = ObjectTransform(
                    object_id=obj_id,
                    position_offset=pos_offset,
                    rotation_offset=rot_quat,
                )
                object_transforms[obj_id] = transform

                # Update obstacle for collision checking
                orig_pos = np.array(obj.get("position", [0, 0, 0]))
                new_obj = {
                    "id": obj_id,
                    "position": (orig_pos + pos_offset).tolist(),
                    "dimensions": obj.get("dimensions", [0.1, 0.1, 0.1]),
                }
                updated_obstacles.append(new_obj)

            # Generate augmented episode
            try:
                augmented = self.augment(
                    seed=seed,
                    object_transforms=object_transforms,
                    obstacles=updated_obstacles,
                    variation_index=var_idx,
                )
                augmented_episodes.append(augmented)
            except Exception as e:
                self.log(f"  Variation {var_idx} failed: {e}", "WARNING")

        successful = sum(1 for ep in augmented_episodes if ep.planning_success)
        self.log(f"Generated {successful}/{num_variations} successful variations")

        return augmented_episodes

    def _decompose_motion_plan(
        self,
        motion_plan: MotionPlan,
        segments: List[SkillSegment],
    ) -> Dict[str, List[Waypoint]]:
        """Assign waypoints to segments based on timing."""
        result = {}

        for segment in segments:
            seg_waypoints = []
            for wp in motion_plan.waypoints:
                # Check if waypoint falls within segment time range
                if segment.start_time <= wp.timestamp <= segment.end_time + 0.1:
                    seg_waypoints.append(wp)

            result[segment.segment_id] = seg_waypoints

        return result

    def _transform_freespace_endpoint(
        self,
        waypoint: Waypoint,
        segment: SkillSegment,
        seed: SeedEpisode,
        object_transforms: Dict[str, ObjectTransform],
        endpoint: str,
    ) -> Waypoint:
        """Transform a free-space segment endpoint based on adjacent skills."""

        # Find adjacent skill segments
        segments = seed.task_spec.segments
        seg_idx = None
        for i, seg in enumerate(segments):
            if seg.segment_id == segment.segment_id:
                seg_idx = i
                break

        if seg_idx is None:
            return waypoint

        # Get reference from adjacent skill segment
        if endpoint in ["start", "both"] and seg_idx > 0:
            prev_seg = segments[seg_idx - 1]
            if prev_seg.segment_type == SegmentType.SKILL and prev_seg.manipulated_object_id:
                transform = object_transforms.get(prev_seg.manipulated_object_id)
                if transform:
                    new_pos = transform.apply_to_position(waypoint.position)
                    return Waypoint(
                        position=new_pos,
                        orientation=waypoint.orientation.copy(),
                        gripper_aperture=waypoint.gripper_aperture,
                        timestamp=waypoint.timestamp,
                        duration_to_next=waypoint.duration_to_next,
                        phase=waypoint.phase,
                    )

        if endpoint in ["end", "both"] and seg_idx < len(segments) - 1:
            next_seg = segments[seg_idx + 1]
            if next_seg.segment_type == SegmentType.SKILL and next_seg.manipulated_object_id:
                transform = object_transforms.get(next_seg.manipulated_object_id)
                if transform:
                    new_pos = transform.apply_to_position(waypoint.position)
                    return Waypoint(
                        position=new_pos,
                        orientation=waypoint.orientation.copy(),
                        gripper_aperture=waypoint.gripper_aperture,
                        timestamp=waypoint.timestamp,
                        duration_to_next=waypoint.duration_to_next,
                        phase=waypoint.phase,
                    )

        return waypoint

    def _transform_position(
        self,
        position: Optional[np.ndarray],
        object_id: Optional[str],
        object_transforms: Dict[str, ObjectTransform],
        original_poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[np.ndarray]:
        """Transform a position based on object transform."""
        if position is None:
            return None

        if object_id and object_id in object_transforms:
            return object_transforms[object_id].apply_to_position(position)

        return position

    def _check_collision_free(
        self,
        waypoints: List[Waypoint],
        obstacles: List[Dict[str, Any]],
    ) -> bool:
        """Check if trajectory is collision-free."""
        for i in range(len(waypoints) - 1):
            if not self.motion_planner._is_path_collision_free(
                waypoints[i].position,
                waypoints[i + 1].position,
                obstacles,
            ):
                return False
        return True


# =============================================================================
# Convenience Functions
# =============================================================================


def augment_episode(
    task_spec: TaskSpecification,
    motion_plan: MotionPlan,
    scene_objects: List[Dict[str, Any]],
    num_variations: int = 10,
    robot_type: str = "franka",
) -> List[AugmentedEpisode]:
    """
    Convenience function to generate augmented episodes.

    Args:
        task_spec: Task specification
        motion_plan: Original motion plan
        scene_objects: Scene objects
        num_variations: Number of variations to generate
        robot_type: Robot type

    Returns:
        List of augmented episodes
    """
    augmenter = ConstraintPreservingAugmenter(robot_type=robot_type, verbose=False)

    seed = augmenter.create_seed_episode(
        task_spec=task_spec,
        motion_plan=motion_plan,
        scene_objects=scene_objects,
    )

    return augmenter.generate_variations(
        seed=seed,
        scene_objects=scene_objects,
        num_variations=num_variations,
    )


if __name__ == "__main__":
    from task_specifier import TaskSpecifier
    from motion_planner import AIMotionPlanner

    print("Testing CP-Gen Style Augmentation")
    print("=" * 60)

    # Create task specification
    specifier = TaskSpecifier(verbose=True)
    scene_objects = [
        {
            "id": "cup_001",
            "category": "cup",
            "position": [0.5, 0.1, 0.85],
            "dimensions": [0.08, 0.08, 0.12],
        },
        {
            "id": "plate_001",
            "category": "plate",
            "position": [0.3, -0.1, 0.82],
            "dimensions": [0.2, 0.2, 0.02],
        },
    ]

    spec = specifier.specify_task(
        task_name="pick_cup",
        task_description="Pick up the coffee cup and place it on the shelf",
        scene_objects=scene_objects,
        target_object_id="cup_001",
        place_position=[0.3, 0.2, 0.9],
    )

    # Create motion plan
    planner = AIMotionPlanner(robot_type="franka", use_llm=False, verbose=True)
    motion_plan = planner.plan_motion(
        task_name="pick_cup",
        task_description="Pick up the coffee cup",
        target_object={
            "id": "cup_001",
            "position": [0.5, 0.1, 0.85],
            "dimensions": [0.08, 0.08, 0.12],
        },
        place_position=[0.3, 0.2, 0.9],
    )

    # Generate augmented episodes
    print("\n" + "=" * 60)
    print("GENERATING AUGMENTED EPISODES")
    print("=" * 60)

    augmented = augment_episode(
        task_spec=spec,
        motion_plan=motion_plan,
        scene_objects=scene_objects,
        num_variations=5,
    )

    print(f"\nGenerated {len(augmented)} augmented episodes")
    for ep in augmented:
        print(f"  - {ep.episode_id}: "
              f"waypoints={ep.motion_plan.num_waypoints}, "
              f"constraint_sat={ep.constraint_satisfaction:.2f}, "
              f"collision_free={ep.collision_free}")
