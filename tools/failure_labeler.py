#!/usr/bin/env python3
"""
Failure Episode Labeler for Episode Generation.

Automatically labels failure episodes with structured failure information:
- Failure type (collision, drop, timeout, grasp_fail, placement_fail, etc.)
- Failure frame (when the failure occurred)
- Failure reason (human-readable explanation)
- Recovery possibility
- Severity level

These labels are critical for:
- Safety learning (what NOT to do)
- Recovery policy training
- Anomaly detection
- Robustness and edge case handling
- Contrastive learning (success vs failure)

Usage:
    labeler = FailureLabeler()
    failure_info = labeler.label_episode(episode_data, contact_events, task_result)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures in robot manipulation."""

    COLLISION = "collision"  # Unexpected collision with environment
    DROP = "drop"  # Dropped object during manipulation
    TIMEOUT = "timeout"  # Task took too long
    GRASP_FAIL = "grasp_fail"  # Failed to grasp object
    PLACEMENT_FAIL = "placement_fail"  # Failed to place object correctly
    SLIP = "slip"  # Object slipped during manipulation
    JOINT_LIMIT = "joint_limit"  # Hit joint limits
    VELOCITY_LIMIT = "velocity_limit"  # Exceeded velocity limits
    TORQUE_LIMIT = "torque_limit"  # Exceeded torque limits
    SELF_COLLISION = "self_collision"  # Robot collided with itself
    OUT_OF_REACH = "out_of_reach"  # Target was out of reach
    STABILITY_LOSS = "stability_loss"  # Lost balance (humanoids)
    CONTACT_LOSS = "contact_loss"  # Lost required contact
    UNKNOWN = "unknown"  # Unknown failure type


class FailureSeverity(Enum):
    """Severity levels for failures."""

    MINOR = "minor"  # Small deviation, easily recoverable
    MODERATE = "moderate"  # Significant deviation, requires intervention
    MAJOR = "major"  # Large deviation, task failed
    CATASTROPHIC = "catastrophic"  # Safety-critical failure


@dataclass
class FailureInfo:
    """
    Structured information about a failure episode.

    Compatible with standard robotics datasets and ML training pipelines.
    """

    # Core failure information
    is_failure: bool = True
    failure_type: FailureType = FailureType.UNKNOWN
    failure_frame: int = -1  # Frame where failure was detected
    failure_timestamp: float = -1.0  # Timestamp of failure
    failure_reason: str = ""  # Human-readable explanation

    # Recovery and severity
    recovery_possible: bool = False
    severity: FailureSeverity = FailureSeverity.MAJOR

    # Detailed information
    involved_objects: List[str] = field(default_factory=list)
    contact_forces: List[float] = field(default_factory=list)  # Forces at failure
    joint_positions: List[float] = field(default_factory=list)  # Robot state at failure
    ee_position: Optional[List[float]] = None  # End-effector position at failure

    # For contrastive learning
    nearest_success_distance: Optional[float] = None  # Distance to nearest success trajectory
    divergence_frame: Optional[int] = None  # Frame where trajectory diverged from success

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_failure": self.is_failure,
            "failure_type": self.failure_type.value,
            "failure_frame": self.failure_frame,
            "failure_timestamp": self.failure_timestamp,
            "failure_reason": self.failure_reason,
            "recovery_possible": self.recovery_possible,
            "severity": self.severity.value,
            "involved_objects": self.involved_objects,
            "contact_forces": self.contact_forces,
            "joint_positions": self.joint_positions,
            "ee_position": self.ee_position,
            "nearest_success_distance": self.nearest_success_distance,
            "divergence_frame": self.divergence_frame,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureInfo":
        """Create from dictionary."""
        return cls(
            is_failure=data.get("is_failure", True),
            failure_type=FailureType(data.get("failure_type", "unknown")),
            failure_frame=data.get("failure_frame", -1),
            failure_timestamp=data.get("failure_timestamp", -1.0),
            failure_reason=data.get("failure_reason", ""),
            recovery_possible=data.get("recovery_possible", False),
            severity=FailureSeverity(data.get("severity", "major")),
            involved_objects=data.get("involved_objects", []),
            contact_forces=data.get("contact_forces", []),
            joint_positions=data.get("joint_positions", []),
            ee_position=data.get("ee_position"),
            nearest_success_distance=data.get("nearest_success_distance"),
            divergence_frame=data.get("divergence_frame"),
        )

    @classmethod
    def success(cls) -> "FailureInfo":
        """Create a success (non-failure) info."""
        return cls(
            is_failure=False,
            failure_type=FailureType.UNKNOWN,
            failure_frame=-1,
            failure_reason="Task completed successfully",
            recovery_possible=True,
            severity=FailureSeverity.MINOR,
        )


class FailureLabeler:
    """
    Automatically labels failure episodes with structured failure information.

    Uses heuristics based on:
    - Contact events (unexpected collisions)
    - Object tracking (drops, slips)
    - Robot state (joint limits, torque limits)
    - Task success metrics
    """

    # Thresholds for failure detection
    COLLISION_FORCE_THRESHOLD = 50.0  # Newtons
    DROP_VELOCITY_THRESHOLD = 1.0  # m/s downward
    GRASP_FORCE_THRESHOLD = 0.5  # Newtons (minimum to consider grasped)
    PLACEMENT_ERROR_THRESHOLD = 0.05  # meters
    TIMEOUT_FRAMES = 1000  # Maximum frames for a task

    # Objects that shouldn't be contacted unexpectedly
    ENVIRONMENT_OBJECTS = ["table", "floor", "wall", "obstacle", "shelf", "cabinet"]

    def __init__(
        self,
        collision_threshold: float = 50.0,
        drop_threshold: float = 1.0,
        placement_threshold: float = 0.05,
        timeout_frames: int = 1000,
    ):
        """
        Initialize the failure labeler.

        Args:
            collision_threshold: Force threshold for collision detection (N)
            drop_threshold: Velocity threshold for drop detection (m/s)
            placement_threshold: Position error threshold for placement (m)
            timeout_frames: Maximum frames before timeout
        """
        self.collision_threshold = collision_threshold
        self.drop_threshold = drop_threshold
        self.placement_threshold = placement_threshold
        self.timeout_frames = timeout_frames

    def label_episode(
        self,
        task_success: bool,
        num_frames: int,
        contact_events: Optional[List[Dict[str, Any]]] = None,
        object_states: Optional[List[Dict[str, Any]]] = None,
        robot_states: Optional[List[Dict[str, Any]]] = None,
        quality_score: float = 1.0,
        grasp_events: Optional[List[Dict[str, Any]]] = None,
        target_position: Optional[List[float]] = None,
        final_object_position: Optional[List[float]] = None,
    ) -> FailureInfo:
        """
        Label an episode with failure information.

        Args:
            task_success: Whether the task was marked as successful
            num_frames: Number of frames in the episode
            contact_events: List of contact events [{frame, body_a, body_b, force, ...}]
            object_states: List of object states per frame [{object_id, position, velocity, ...}]
            robot_states: List of robot states per frame [{joint_positions, joint_velocities, ...}]
            quality_score: Quality score of the episode (0-1)
            grasp_events: List of grasp events [{frame, success, object_id, ...}]
            target_position: Target position for placement tasks
            final_object_position: Final position of manipulated object

        Returns:
            FailureInfo with detailed failure labels
        """
        # If task succeeded, return success info
        if task_success and quality_score >= 0.7:
            return FailureInfo.success()

        # Initialize failure info
        failure_info = FailureInfo(
            is_failure=True,
            failure_frame=num_frames - 1,  # Default to last frame
        )

        # Check for different failure types in order of priority

        # 1. Check for collisions
        collision_failure = self._check_collisions(contact_events)
        if collision_failure:
            return collision_failure

        # 2. Check for drops
        drop_failure = self._check_drops(object_states)
        if drop_failure:
            return drop_failure

        # 3. Check for grasp failures
        grasp_failure = self._check_grasp_failures(grasp_events, contact_events)
        if grasp_failure:
            return grasp_failure

        # 4. Check for placement failures
        placement_failure = self._check_placement_failures(
            target_position, final_object_position, task_success
        )
        if placement_failure:
            return placement_failure

        # 5. Check for timeout
        if num_frames >= self.timeout_frames:
            return FailureInfo(
                is_failure=True,
                failure_type=FailureType.TIMEOUT,
                failure_frame=num_frames - 1,
                failure_reason=f"Task timed out after {num_frames} frames",
                recovery_possible=True,
                severity=FailureSeverity.MODERATE,
            )

        # 6. Check for joint limit violations
        joint_failure = self._check_joint_limits(robot_states)
        if joint_failure:
            return joint_failure

        # 7. Check for stability loss (humanoids)
        stability_failure = self._check_stability_loss(robot_states)
        if stability_failure:
            return stability_failure

        # If no specific failure detected but task failed
        if not task_success:
            failure_info.failure_type = FailureType.UNKNOWN
            failure_info.failure_reason = f"Task failed with quality score {quality_score:.2f}"
            failure_info.severity = FailureSeverity.MODERATE if quality_score > 0.4 else FailureSeverity.MAJOR
            failure_info.recovery_possible = quality_score > 0.3

        return failure_info

    def _check_collisions(
        self,
        contact_events: Optional[List[Dict[str, Any]]],
    ) -> Optional[FailureInfo]:
        """Check for unexpected collision events."""
        if not contact_events:
            return None

        for event in contact_events:
            force = event.get("force_magnitude", event.get("force", 0))
            if force < self.collision_threshold:
                continue

            body_a = str(event.get("body_a", "")).lower()
            body_b = str(event.get("body_b", "")).lower()

            # Check if this is an unexpected collision with environment
            is_env_collision = any(
                env_obj in body_a or env_obj in body_b
                for env_obj in self.ENVIRONMENT_OBJECTS
            )

            # Check if this is robot-to-environment (not object-to-environment)
            robot_keywords = ["gripper", "finger", "link", "joint", "arm", "wrist"]
            involves_robot = any(kw in body_a or kw in body_b for kw in robot_keywords)

            if is_env_collision and involves_robot:
                # Check if it's a self-collision
                if "robot" in body_a and "robot" in body_b:
                    return FailureInfo(
                        is_failure=True,
                        failure_type=FailureType.SELF_COLLISION,
                        failure_frame=event.get("frame", -1),
                        failure_timestamp=event.get("timestamp", -1.0),
                        failure_reason=f"Robot self-collision between {body_a} and {body_b}",
                        recovery_possible=False,
                        severity=FailureSeverity.MAJOR,
                        contact_forces=[force],
                        involved_objects=[body_a, body_b],
                    )

                return FailureInfo(
                    is_failure=True,
                    failure_type=FailureType.COLLISION,
                    failure_frame=event.get("frame", -1),
                    failure_timestamp=event.get("timestamp", -1.0),
                    failure_reason=f"Unexpected collision: {body_a} hit {body_b} with {force:.1f}N",
                    recovery_possible=force < self.collision_threshold * 2,
                    severity=self._severity_from_force(force),
                    contact_forces=[force],
                    involved_objects=[body_a, body_b],
                )

        return None

    def _check_drops(
        self,
        object_states: Optional[List[Dict[str, Any]]],
    ) -> Optional[FailureInfo]:
        """Check for object drop events."""
        if not object_states:
            return None

        # Track objects that were being manipulated
        for i, state in enumerate(object_states):
            if not isinstance(state, dict):
                continue

            for obj_id, obj_state in state.items():
                if not isinstance(obj_state, dict):
                    continue

                velocity = obj_state.get("velocity", [0, 0, 0])
                if len(velocity) >= 3:
                    # Check for rapid downward motion
                    vz = velocity[2]
                    if vz < -self.drop_threshold:
                        return FailureInfo(
                            is_failure=True,
                            failure_type=FailureType.DROP,
                            failure_frame=i,
                            failure_reason=f"Object {obj_id} dropped (vz={vz:.2f} m/s)",
                            recovery_possible=False,
                            severity=FailureSeverity.MAJOR,
                            involved_objects=[obj_id],
                        )

        return None

    def _check_grasp_failures(
        self,
        grasp_events: Optional[List[Dict[str, Any]]],
        contact_events: Optional[List[Dict[str, Any]]],
    ) -> Optional[FailureInfo]:
        """Check for grasp failure events."""
        if grasp_events:
            for event in grasp_events:
                if not event.get("success", True):
                    return FailureInfo(
                        is_failure=True,
                        failure_type=FailureType.GRASP_FAIL,
                        failure_frame=event.get("frame", -1),
                        failure_reason=f"Failed to grasp {event.get('object_id', 'object')}",
                        recovery_possible=True,
                        severity=FailureSeverity.MODERATE,
                        involved_objects=[event.get("object_id", "unknown")],
                    )

        # Check contact events for slip during grasp
        if contact_events:
            gripper_contacts = []
            for event in contact_events:
                body_a = str(event.get("body_a", "")).lower()
                body_b = str(event.get("body_b", "")).lower()
                if "gripper" in body_a or "gripper" in body_b or "finger" in body_a or "finger" in body_b:
                    gripper_contacts.append(event)

            # Check for sudden loss of gripper contact (slip)
            if len(gripper_contacts) >= 2:
                forces = [c.get("force_magnitude", c.get("force", 0)) for c in gripper_contacts]
                # Large force drop indicates slip
                for i in range(1, len(forces)):
                    if forces[i-1] > self.GRASP_FORCE_THRESHOLD and forces[i] < self.GRASP_FORCE_THRESHOLD * 0.2:
                        return FailureInfo(
                            is_failure=True,
                            failure_type=FailureType.SLIP,
                            failure_frame=gripper_contacts[i].get("frame", -1),
                            failure_reason="Object slipped from gripper",
                            recovery_possible=False,
                            severity=FailureSeverity.MAJOR,
                        )

        return None

    def _check_placement_failures(
        self,
        target_position: Optional[List[float]],
        final_object_position: Optional[List[float]],
        task_success: bool,
    ) -> Optional[FailureInfo]:
        """Check for placement failure (object not at target)."""
        if target_position is None or final_object_position is None:
            return None

        if len(target_position) < 3 or len(final_object_position) < 3:
            return None

        # Calculate position error
        import math
        error = math.sqrt(sum(
            (t - f) ** 2
            for t, f in zip(target_position[:3], final_object_position[:3])
        ))

        if error > self.placement_threshold and not task_success:
            return FailureInfo(
                is_failure=True,
                failure_type=FailureType.PLACEMENT_FAIL,
                failure_frame=-1,  # End of episode
                failure_reason=f"Placement error: {error*100:.1f}cm from target",
                recovery_possible=error < self.placement_threshold * 3,
                severity=FailureSeverity.MODERATE if error < self.placement_threshold * 2 else FailureSeverity.MAJOR,
            )

        return None

    def _check_joint_limits(
        self,
        robot_states: Optional[List[Dict[str, Any]]],
    ) -> Optional[FailureInfo]:
        """Check for joint limit violations."""
        if not robot_states:
            return None

        for i, state in enumerate(robot_states):
            if not isinstance(state, dict):
                continue

            at_limit = state.get("at_joint_limit", False)
            if at_limit:
                joint_idx = state.get("limit_joint_idx", -1)
                return FailureInfo(
                    is_failure=True,
                    failure_type=FailureType.JOINT_LIMIT,
                    failure_frame=i,
                    failure_reason=f"Joint {joint_idx} hit limit",
                    recovery_possible=True,
                    severity=FailureSeverity.MODERATE,
                    joint_positions=state.get("joint_positions", []),
                )

        return None

    def _check_stability_loss(
        self,
        robot_states: Optional[List[Dict[str, Any]]],
    ) -> Optional[FailureInfo]:
        """Check for stability loss (humanoid robots)."""
        if not robot_states:
            return None

        for i, state in enumerate(robot_states):
            if not isinstance(state, dict):
                continue

            balance_state = state.get("balance_state", {})
            stability_margin = balance_state.get("stability_margin")

            if stability_margin is not None and stability_margin < 0:
                return FailureInfo(
                    is_failure=True,
                    failure_type=FailureType.STABILITY_LOSS,
                    failure_frame=i,
                    failure_reason=f"Lost balance (stability margin: {stability_margin:.3f})",
                    recovery_possible=False,
                    severity=FailureSeverity.CATASTROPHIC,
                )

        return None

    def _severity_from_force(self, force: float) -> FailureSeverity:
        """Determine severity based on collision force."""
        if force < self.collision_threshold:
            return FailureSeverity.MINOR
        elif force < self.collision_threshold * 2:
            return FailureSeverity.MODERATE
        elif force < self.collision_threshold * 5:
            return FailureSeverity.MAJOR
        else:
            return FailureSeverity.CATASTROPHIC


def label_episode_batch(
    episodes: List[Dict[str, Any]],
    labeler: Optional[FailureLabeler] = None,
) -> List[FailureInfo]:
    """
    Label a batch of episodes with failure information.

    Args:
        episodes: List of episode data dictionaries
        labeler: Optional pre-configured labeler

    Returns:
        List of FailureInfo for each episode
    """
    if labeler is None:
        labeler = FailureLabeler()

    results = []
    for episode in episodes:
        failure_info = labeler.label_episode(
            task_success=episode.get("success", episode.get("task_success", False)),
            num_frames=episode.get("num_frames", episode.get("length", 0)),
            contact_events=episode.get("contact_events", episode.get("contacts", [])),
            object_states=episode.get("object_states", []),
            robot_states=episode.get("robot_states", []),
            quality_score=episode.get("quality_score", 1.0),
            grasp_events=episode.get("grasp_events", []),
            target_position=episode.get("target_position"),
            final_object_position=episode.get("final_object_position"),
        )
        results.append(failure_info)

    return results
