#!/usr/bin/env python3
"""
Task Specifier - Gemini at the Top of the Stack.

This module uses Gemini (LLM) for high-level task specification:
- Goal definition and success criteria
- Constraint specification (contact, collision, timing)
- Keypoint trajectory definition
- Skill decomposition

Based on research findings (2025-2026):
- DemoGen: Skill segments + motion segments decomposition
- CP-Gen: Keypoint-trajectory constraints for geometry-aware generation
- AnyTask: Automated task and constraint generation

The key insight: LLMs are excellent at *specifying* what should happen,
but motion planning/optimization should produce the actual trajectories.

Reference:
- CP-Gen (CoRL 2025): https://cp-gen.github.io/
- DemoGen (RSS 2025): https://demo-generation.github.io/
- AnyTask: https://anytask.rai-inst.com/
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.metrics.pipeline_metrics import get_metrics

logger = logging.getLogger(__name__)

try:
    from tools.llm_client import create_llm_client, LLMResponse
    HAVE_LLM_CLIENT = True
except ImportError:
    HAVE_LLM_CLIENT = False
    create_llm_client = None

try:
    from tools.secrets import get_secret_or_env, SecretIds
    HAVE_SECRET_MANAGER = True
except ImportError:
    HAVE_SECRET_MANAGER = False
    get_secret_or_env = None
    SecretIds = None

SECRET_ID_GEMINI = SecretIds.GEMINI_API_KEY if SecretIds else "gemini-api-key"
SECRET_ID_OPENAI = SecretIds.OPENAI_API_KEY if SecretIds else "openai-api-key"
SECRET_ID_ANTHROPIC = SecretIds.ANTHROPIC_API_KEY if SecretIds else "anthropic-api-key"


def _load_secret_value(secret_id: str, env_var: str) -> Optional[str]:
    if HAVE_SECRET_MANAGER and get_secret_or_env is not None:
        try:
            return get_secret_or_env(secret_id, env_var=env_var)
        except Exception as exc:
            logger.warning(
                "[TASK-SPECIFIER] [WARN] Failed to fetch secret '%s', falling back to env var '%s': %s",
                secret_id,
                env_var,
                exc,
            )
            return os.environ.get(env_var)
    return os.environ.get(env_var)

# =============================================================================
# Data Models for Task Specification
# =============================================================================


class SegmentType(str, Enum):
    """Types of trajectory segments (DemoGen-style decomposition)."""
    SKILL = "skill"           # Contact manipulation - preserve exactly
    FREE_SPACE = "free_space"  # Motion in free space - can replan


class ConstraintType(str, Enum):
    """Types of constraints for keypoint trajectories (CP-Gen style)."""
    POSITION = "position"            # Keypoint must be at position
    RELATIVE_POSITION = "relative"   # Keypoint relative to object
    TRAJECTORY = "trajectory"        # Keypoint follows path
    ORIENTATION = "orientation"      # Maintain orientation
    CONTACT = "contact"              # Must be in contact
    CLEARANCE = "clearance"          # Must maintain distance


@dataclass
class Keypoint:
    """
    A keypoint on the robot or grasped object.

    CP-Gen uses keypoints to define constraints that must be preserved
    when generating demonstrations for new object configurations.
    """
    keypoint_id: str
    frame: str  # "gripper", "ee", "grasped_object", "wrist"
    local_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keypoint_id": self.keypoint_id,
            "frame": self.frame,
            "local_position": self.local_position.tolist(),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Keypoint":
        return cls(
            keypoint_id=data["keypoint_id"],
            frame=data["frame"],
            local_position=np.array(data.get("local_position", [0, 0, 0])),
            description=data.get("description", ""),
        )


@dataclass
class KeypointConstraint:
    """
    A constraint on a keypoint trajectory.

    Based on CP-Gen: keypoints on the robot/grasped object must track
    reference trajectories defined relative to task-relevant objects.
    """
    constraint_id: str
    keypoint_id: str
    constraint_type: ConstraintType

    # For RELATIVE_POSITION: reference object and offset
    reference_object_id: Optional[str] = None
    reference_offset: Optional[np.ndarray] = None

    # For TRAJECTORY: waypoints relative to reference
    trajectory_waypoints: List[np.ndarray] = field(default_factory=list)

    # For CONTACT: contact normal and force range
    contact_normal: Optional[np.ndarray] = None
    force_range: Tuple[float, float] = (0.0, 10.0)

    # For CLEARANCE: minimum distance
    clearance_distance: float = 0.02

    # Timing
    start_time: float = 0.0
    end_time: float = 1.0

    # Priority (for constraint solver)
    priority: int = 1  # Higher = more important

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "keypoint_id": self.keypoint_id,
            "constraint_type": self.constraint_type.value,
            "reference_object_id": self.reference_object_id,
            "reference_offset": self.reference_offset.tolist() if self.reference_offset is not None else None,
            "trajectory_waypoints": [w.tolist() for w in self.trajectory_waypoints],
            "contact_normal": self.contact_normal.tolist() if self.contact_normal is not None else None,
            "force_range": list(self.force_range),
            "clearance_distance": self.clearance_distance,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "priority": self.priority,
        }


@dataclass
class SkillSegment:
    """
    A skill segment involving contact manipulation.

    DemoGen insight: Skill segments are the hard part (contact physics).
    These should be preserved/transformed carefully, not replanned.
    """
    segment_id: str
    segment_type: SegmentType = SegmentType.SKILL

    # Skill description
    skill_name: str = ""
    description: str = ""

    # Timing
    start_time: float = 0.0
    end_time: float = 1.0

    # Objects involved
    manipulated_object_id: Optional[str] = None
    contact_objects: List[str] = field(default_factory=list)

    # Keypoints and constraints for this segment
    keypoints: List[Keypoint] = field(default_factory=list)
    constraints: List[KeypointConstraint] = field(default_factory=list)

    # Gripper state
    gripper_closed: bool = False

    # Success criteria
    success_criteria: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "segment_type": self.segment_type.value,
            "skill_name": self.skill_name,
            "description": self.description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "manipulated_object_id": self.manipulated_object_id,
            "contact_objects": self.contact_objects,
            "keypoints": [k.to_dict() for k in self.keypoints],
            "constraints": [c.to_dict() for c in self.constraints],
            "gripper_closed": self.gripper_closed,
            "success_criteria": self.success_criteria,
        }


@dataclass
class TaskSpecification:
    """
    Complete task specification from Gemini.

    This is the output of the "brain" (LLM) that tells the planner:
    - What to achieve (goal)
    - How to decompose it (segments)
    - What constraints to respect (keypoints)
    - How to verify success (criteria)
    """
    spec_id: str
    task_name: str
    task_description: str

    # Goal specification
    goal_object_id: Optional[str] = None
    goal_position: Optional[np.ndarray] = None
    goal_state: Dict[str, Any] = field(default_factory=dict)

    # Segment decomposition (DemoGen-style)
    segments: List[SkillSegment] = field(default_factory=list)

    # Global keypoints
    keypoints: List[Keypoint] = field(default_factory=list)

    # Global constraints
    constraints: List[KeypointConstraint] = field(default_factory=list)

    # Success criteria
    success_criteria: Dict[str, Any] = field(default_factory=dict)

    # Environment constraints
    obstacle_ids: List[str] = field(default_factory=list)
    collision_objects: List[str] = field(default_factory=list)

    # Robot configuration
    robot_type: str = "franka"

    # Estimated timing
    estimated_duration: float = 5.0

    # Confidence from LLM
    confidence: float = 1.0

    # Raw LLM response (for debugging)
    llm_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "task_name": self.task_name,
            "task_description": self.task_description,
            "goal_object_id": self.goal_object_id,
            "goal_position": self.goal_position.tolist() if self.goal_position is not None else None,
            "goal_state": self.goal_state,
            "segments": [s.to_dict() for s in self.segments],
            "keypoints": [k.to_dict() for k in self.keypoints],
            "constraints": [c.to_dict() for c in self.constraints],
            "success_criteria": self.success_criteria,
            "obstacle_ids": self.obstacle_ids,
            "collision_objects": self.collision_objects,
            "robot_type": self.robot_type,
            "estimated_duration": self.estimated_duration,
            "confidence": self.confidence,
        }


# =============================================================================
# Task Specifier (Gemini Integration)
# =============================================================================


class TaskSpecifier:
    """
    Uses Gemini to generate high-level task specifications.

    This is where the LLM's strength lies: understanding intent,
    decomposing tasks, and specifying constraints. The actual motion
    planning happens downstream.

    Usage:
        specifier = TaskSpecifier()
        spec = specifier.specify_task(
            task_name="pick_cup",
            task_description="Pick up the coffee cup and place it on the shelf",
            scene_objects=scene_manifest["objects"],
            robot_type="franka",
        )

        # spec now contains:
        # - Skill segments (grasp, transport, place)
        # - Keypoint constraints (gripper tip relative to cup)
        # - Success criteria (cup on shelf, stable)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._client = None

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[TASK-SPECIFIER] [%s] %s", level, msg)

    def _ensure_llm_credentials(self) -> None:
        providers = [
            ("gemini", SECRET_ID_GEMINI, "GEMINI_API_KEY"),
            ("openai", SECRET_ID_OPENAI, "OPENAI_API_KEY"),
            ("anthropic", SECRET_ID_ANTHROPIC, "ANTHROPIC_API_KEY"),
        ]
        available = []
        for provider, secret_id, env_var in providers:
            if _load_secret_value(secret_id, env_var):
                available.append(provider)

        if not available:
            message = (
                "No LLM API keys configured. "
                "Set Secret Manager IDs "
                f"'{SECRET_ID_GEMINI}', '{SECRET_ID_OPENAI}', "
                f"or '{SECRET_ID_ANTHROPIC}', "
                "or provide GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY env vars."
            )
            self.log(message, level="ERROR")
            raise ValueError(message)

    def _get_client(self):
        """Get or create LLM client."""
        if self._client is None:
            if not HAVE_LLM_CLIENT:
                raise ImportError("LLM client not available")
            self._ensure_llm_credentials()
            self._client = create_llm_client()
        return self._client

    def specify_task(
        self,
        task_name: str,
        task_description: str,
        scene_objects: List[Dict[str, Any]],
        robot_type: str = "franka",
        target_object_id: Optional[str] = None,
        place_position: Optional[List[float]] = None,
    ) -> TaskSpecification:
        """
        Generate a complete task specification using Gemini.

        Args:
            task_name: Name of the task
            task_description: Natural language description
            scene_objects: Objects in the scene (from manifest)
            robot_type: Robot type for planning
            target_object_id: ID of object to manipulate
            place_position: Target placement position

        Returns:
            TaskSpecification with segments, keypoints, constraints
        """
        self.log(f"Specifying task: {task_name}")

        # Find target object
        target_object = None
        if target_object_id:
            for obj in scene_objects:
                if obj.get("id") == target_object_id or obj.get("name") == target_object_id:
                    target_object = obj
                    break

        # Classify task type
        task_type = self._classify_task(task_name, task_description)
        self.log(f"  Task type: {task_type}")

        # Generate specification based on task type
        if task_type == "pick_place":
            spec = self._specify_pick_place(
                task_name=task_name,
                task_description=task_description,
                target_object=target_object,
                place_position=place_position,
                scene_objects=scene_objects,
                robot_type=robot_type,
            )
        elif task_type == "articulation":
            spec = self._specify_articulation(
                task_name=task_name,
                task_description=task_description,
                target_object=target_object,
                scene_objects=scene_objects,
                robot_type=robot_type,
            )
        elif task_type == "push":
            spec = self._specify_push(
                task_name=task_name,
                task_description=task_description,
                target_object=target_object,
                scene_objects=scene_objects,
                robot_type=robot_type,
            )
        else:
            spec = self._specify_generic(
                task_name=task_name,
                task_description=task_description,
                target_object=target_object,
                scene_objects=scene_objects,
                robot_type=robot_type,
            )

        # Optionally enhance with LLM
        if HAVE_LLM_CLIENT:
            try:
                spec = self._enhance_with_llm(spec, scene_objects)
            except Exception as e:
                self.log(f"  LLM enhancement failed: {e}", "WARNING")

        self.log(f"  Generated {len(spec.segments)} segments, {len(spec.constraints)} constraints")

        return spec

    def _classify_task(self, task_name: str, description: str) -> str:
        """Classify the task type."""
        text = f"{task_name} {description}".lower()

        if any(word in text for word in ["pick", "grasp", "grab", "lift", "place", "put", "move"]):
            return "pick_place"
        elif any(word in text for word in ["open", "close", "pull", "push drawer", "door", "articulate"]):
            return "articulation"
        elif any(word in text for word in ["push", "slide"]):
            return "push"
        else:
            return "generic"

    def _specify_pick_place(
        self,
        task_name: str,
        task_description: str,
        target_object: Optional[Dict[str, Any]],
        place_position: Optional[List[float]],
        scene_objects: List[Dict[str, Any]],
        robot_type: str,
    ) -> TaskSpecification:
        """Generate specification for pick-and-place task."""

        spec_id = f"spec_{task_name}_{id(self)}"

        # Get object properties
        obj_pos = np.array(target_object.get("position", [0.5, 0, 0.85])) if target_object else np.array([0.5, 0, 0.85])
        obj_dims = np.array(target_object.get("dimensions", [0.08, 0.08, 0.1])) if target_object else np.array([0.08, 0.08, 0.1])
        obj_id = target_object.get("id", "target") if target_object else "target"

        if place_position is None:
            place_position = [obj_pos[0] - 0.2, obj_pos[1] + 0.15, obj_pos[2]]
        place_pos = np.array(place_position)

        # Define keypoints (CP-Gen style)
        keypoints = [
            Keypoint(
                keypoint_id="gripper_tip",
                frame="gripper",
                local_position=np.array([0, 0, 0.1]),  # Tip of gripper fingers
                description="Gripper fingertip center",
            ),
            Keypoint(
                keypoint_id="grasp_center",
                frame="grasped_object",
                local_position=np.array([0, 0, obj_dims[2] / 2]),  # Center of object
                description="Grasp center point",
            ),
        ]

        # Define segments (DemoGen style decomposition)
        segments = [
            # Segment 1: Approach (free space)
            SkillSegment(
                segment_id=f"{spec_id}_approach",
                segment_type=SegmentType.FREE_SPACE,
                skill_name="approach",
                description="Move gripper above the object",
                start_time=0.0,
                end_time=0.3,
                gripper_closed=False,
                constraints=[
                    KeypointConstraint(
                        constraint_id=f"{spec_id}_approach_clearance",
                        keypoint_id="gripper_tip",
                        constraint_type=ConstraintType.CLEARANCE,
                        clearance_distance=0.05,
                        priority=2,
                    ),
                ],
            ),
            # Segment 2: Pre-grasp descent (skill - critical)
            SkillSegment(
                segment_id=f"{spec_id}_pregrasp",
                segment_type=SegmentType.SKILL,
                skill_name="pre_grasp",
                description="Align gripper above object for grasp",
                start_time=0.3,
                end_time=0.5,
                manipulated_object_id=obj_id,
                gripper_closed=False,
                keypoints=[keypoints[0]],
                constraints=[
                    KeypointConstraint(
                        constraint_id=f"{spec_id}_align",
                        keypoint_id="gripper_tip",
                        constraint_type=ConstraintType.RELATIVE_POSITION,
                        reference_object_id=obj_id,
                        reference_offset=np.array([0, 0, obj_dims[2] + 0.05]),
                        priority=3,
                    ),
                ],
            ),
            # Segment 3: Grasp (skill - critical)
            SkillSegment(
                segment_id=f"{spec_id}_grasp",
                segment_type=SegmentType.SKILL,
                skill_name="grasp",
                description="Close gripper on object",
                start_time=0.5,
                end_time=0.7,
                manipulated_object_id=obj_id,
                contact_objects=[obj_id],
                gripper_closed=True,
                keypoints=[keypoints[0]],
                constraints=[
                    KeypointConstraint(
                        constraint_id=f"{spec_id}_grasp_contact",
                        keypoint_id="gripper_tip",
                        constraint_type=ConstraintType.CONTACT,
                        reference_object_id=obj_id,
                        contact_normal=np.array([0, 0, -1]),
                        force_range=(1.0, 20.0),
                        priority=5,  # High priority for grasp
                    ),
                ],
                success_criteria={
                    "gripper_closed": True,
                    "object_grasped": True,
                    "stable_grasp": True,
                },
            ),
            # Segment 4: Lift (skill - maintaining grasp)
            SkillSegment(
                segment_id=f"{spec_id}_lift",
                segment_type=SegmentType.SKILL,
                skill_name="lift",
                description="Lift object while maintaining grasp",
                start_time=0.7,
                end_time=0.9,
                manipulated_object_id=obj_id,
                gripper_closed=True,
                keypoints=[keypoints[1]],
                constraints=[
                    KeypointConstraint(
                        constraint_id=f"{spec_id}_lift_traj",
                        keypoint_id="grasp_center",
                        constraint_type=ConstraintType.TRAJECTORY,
                        reference_object_id=obj_id,
                        trajectory_waypoints=[
                            np.array([0, 0, 0]),       # Start at grasp
                            np.array([0, 0, 0.15]),    # Lift up
                        ],
                        priority=4,
                    ),
                ],
            ),
            # Segment 5: Transport (free space)
            SkillSegment(
                segment_id=f"{spec_id}_transport",
                segment_type=SegmentType.FREE_SPACE,
                skill_name="transport",
                description="Move object to place position",
                start_time=0.9,
                end_time=1.3,
                manipulated_object_id=obj_id,
                gripper_closed=True,
                constraints=[
                    KeypointConstraint(
                        constraint_id=f"{spec_id}_transport_clearance",
                        keypoint_id="grasp_center",
                        constraint_type=ConstraintType.CLEARANCE,
                        clearance_distance=0.05,
                        priority=2,
                    ),
                ],
            ),
            # Segment 6: Place (skill - critical)
            SkillSegment(
                segment_id=f"{spec_id}_place",
                segment_type=SegmentType.SKILL,
                skill_name="place",
                description="Place object at target location",
                start_time=1.3,
                end_time=1.6,
                manipulated_object_id=obj_id,
                gripper_closed=True,
                keypoints=[keypoints[1]],
                constraints=[
                    KeypointConstraint(
                        constraint_id=f"{spec_id}_place_pos",
                        keypoint_id="grasp_center",
                        constraint_type=ConstraintType.POSITION,
                        reference_offset=place_pos + np.array([0, 0, obj_dims[2] / 2]),
                        priority=4,
                    ),
                ],
                success_criteria={
                    "object_at_target": True,
                    "object_stable": True,
                },
            ),
            # Segment 7: Release (skill)
            SkillSegment(
                segment_id=f"{spec_id}_release",
                segment_type=SegmentType.SKILL,
                skill_name="release",
                description="Open gripper to release object",
                start_time=1.6,
                end_time=1.8,
                manipulated_object_id=obj_id,
                gripper_closed=False,
            ),
            # Segment 8: Retract (free space)
            SkillSegment(
                segment_id=f"{spec_id}_retract",
                segment_type=SegmentType.FREE_SPACE,
                skill_name="retract",
                description="Retract gripper from object",
                start_time=1.8,
                end_time=2.0,
                gripper_closed=False,
            ),
        ]

        # Global constraints
        global_constraints = [
            KeypointConstraint(
                constraint_id=f"{spec_id}_collision_avoidance",
                keypoint_id="gripper_tip",
                constraint_type=ConstraintType.CLEARANCE,
                clearance_distance=0.02,
                start_time=0.0,
                end_time=2.0,
                priority=1,
            ),
        ]

        # Find obstacles
        obstacle_ids = [
            obj.get("id", obj.get("name", ""))
            for obj in scene_objects
            if obj.get("id") != obj_id and obj.get("sim_role") != "background"
        ]

        return TaskSpecification(
            spec_id=spec_id,
            task_name=task_name,
            task_description=task_description,
            goal_object_id=obj_id,
            goal_position=place_pos,
            goal_state={"object_placed": True, "stable": True},
            segments=segments,
            keypoints=keypoints,
            constraints=global_constraints,
            success_criteria={
                "object_at_goal": True,
                "object_stable": True,
                "no_collisions": True,
            },
            obstacle_ids=obstacle_ids,
            collision_objects=obstacle_ids,
            robot_type=robot_type,
            estimated_duration=2.0,
        )

    def _specify_articulation(
        self,
        task_name: str,
        task_description: str,
        target_object: Optional[Dict[str, Any]],
        scene_objects: List[Dict[str, Any]],
        robot_type: str,
    ) -> TaskSpecification:
        """Generate specification for articulation task (drawer/door)."""

        spec_id = f"spec_{task_name}_{id(self)}"

        # Determine if drawer or door
        is_drawer = "drawer" in task_name.lower() or "drawer" in task_description.lower()
        is_open = "open" in task_name.lower() or "open" in task_description.lower()

        obj_pos = np.array(target_object.get("position", [0.5, 0, 0.85])) if target_object else np.array([0.5, 0, 0.85])
        obj_id = target_object.get("id", "drawer") if target_object else "drawer"

        # Motion axis and distance
        if is_drawer:
            motion_axis = np.array([-1, 0, 0])  # Pull toward robot
            motion_dist = 0.3
        else:
            # Door rotation
            motion_axis = np.array([0, 1, 0])  # Rotate around Y
            motion_dist = 0.5  # radians

        if not is_open:
            motion_axis = -motion_axis

        # Keypoints
        keypoints = [
            Keypoint(
                keypoint_id="gripper_tip",
                frame="gripper",
                local_position=np.array([0, 0, 0.1]),
                description="Gripper fingertip",
            ),
            Keypoint(
                keypoint_id="handle_grip",
                frame="gripper",
                local_position=np.array([0, 0, 0]),
                description="Handle grip point",
            ),
        ]

        segments = [
            # Approach
            SkillSegment(
                segment_id=f"{spec_id}_approach",
                segment_type=SegmentType.FREE_SPACE,
                skill_name="approach",
                description="Move to handle",
                start_time=0.0,
                end_time=0.4,
                gripper_closed=False,
            ),
            # Grasp handle
            SkillSegment(
                segment_id=f"{spec_id}_grasp_handle",
                segment_type=SegmentType.SKILL,
                skill_name="grasp_handle",
                description="Grasp the handle",
                start_time=0.4,
                end_time=0.7,
                manipulated_object_id=obj_id,
                contact_objects=[obj_id],
                gripper_closed=True,
                keypoints=[keypoints[1]],
                constraints=[
                    KeypointConstraint(
                        constraint_id=f"{spec_id}_handle_contact",
                        keypoint_id="handle_grip",
                        constraint_type=ConstraintType.CONTACT,
                        reference_object_id=obj_id,
                        priority=5,
                    ),
                ],
            ),
            # Articulate
            SkillSegment(
                segment_id=f"{spec_id}_articulate",
                segment_type=SegmentType.SKILL,
                skill_name="articulate",
                description=f"{'Open' if is_open else 'Close'} the {'drawer' if is_drawer else 'door'}",
                start_time=0.7,
                end_time=1.5,
                manipulated_object_id=obj_id,
                gripper_closed=True,
                keypoints=[keypoints[1]],
                constraints=[
                    KeypointConstraint(
                        constraint_id=f"{spec_id}_articulation_traj",
                        keypoint_id="handle_grip",
                        constraint_type=ConstraintType.TRAJECTORY,
                        trajectory_waypoints=[
                            np.array([0, 0, 0]),
                            motion_axis * motion_dist,
                        ],
                        priority=4,
                    ),
                ],
                success_criteria={
                    "articulation_complete": True,
                    "target_angle_reached": True,
                },
            ),
            # Release
            SkillSegment(
                segment_id=f"{spec_id}_release",
                segment_type=SegmentType.SKILL,
                skill_name="release",
                description="Release handle",
                start_time=1.5,
                end_time=1.7,
                gripper_closed=False,
            ),
            # Retract
            SkillSegment(
                segment_id=f"{spec_id}_retract",
                segment_type=SegmentType.FREE_SPACE,
                skill_name="retract",
                description="Retract from handle",
                start_time=1.7,
                end_time=2.0,
                gripper_closed=False,
            ),
        ]

        return TaskSpecification(
            spec_id=spec_id,
            task_name=task_name,
            task_description=task_description,
            goal_object_id=obj_id,
            goal_state={"articulation_state": "open" if is_open else "closed"},
            segments=segments,
            keypoints=keypoints,
            success_criteria={
                "articulation_complete": True,
                "no_collisions": True,
            },
            robot_type=robot_type,
            estimated_duration=2.0,
        )

    def _specify_push(
        self,
        task_name: str,
        task_description: str,
        target_object: Optional[Dict[str, Any]],
        scene_objects: List[Dict[str, Any]],
        robot_type: str,
    ) -> TaskSpecification:
        """Generate specification for push task."""

        spec_id = f"spec_{task_name}_{id(self)}"

        obj_pos = np.array(target_object.get("position", [0.5, 0, 0.85])) if target_object else np.array([0.5, 0, 0.85])
        obj_id = target_object.get("id", "object") if target_object else "object"

        push_dir = np.array([1, 0, 0])  # Push forward
        push_dist = 0.2

        keypoints = [
            Keypoint(
                keypoint_id="push_point",
                frame="gripper",
                local_position=np.array([0, 0, 0.08]),
                description="Push contact point",
            ),
        ]

        segments = [
            SkillSegment(
                segment_id=f"{spec_id}_approach",
                segment_type=SegmentType.FREE_SPACE,
                skill_name="approach",
                start_time=0.0,
                end_time=0.3,
                gripper_closed=True,
            ),
            SkillSegment(
                segment_id=f"{spec_id}_contact",
                segment_type=SegmentType.SKILL,
                skill_name="contact",
                description="Make contact with object",
                start_time=0.3,
                end_time=0.5,
                manipulated_object_id=obj_id,
                contact_objects=[obj_id],
                gripper_closed=True,
                keypoints=[keypoints[0]],
                constraints=[
                    KeypointConstraint(
                        constraint_id=f"{spec_id}_push_contact",
                        keypoint_id="push_point",
                        constraint_type=ConstraintType.CONTACT,
                        reference_object_id=obj_id,
                        contact_normal=-push_dir,
                        priority=4,
                    ),
                ],
            ),
            SkillSegment(
                segment_id=f"{spec_id}_push",
                segment_type=SegmentType.SKILL,
                skill_name="push",
                description="Push object",
                start_time=0.5,
                end_time=1.2,
                manipulated_object_id=obj_id,
                contact_objects=[obj_id],
                gripper_closed=True,
                keypoints=[keypoints[0]],
                constraints=[
                    KeypointConstraint(
                        constraint_id=f"{spec_id}_push_traj",
                        keypoint_id="push_point",
                        constraint_type=ConstraintType.TRAJECTORY,
                        trajectory_waypoints=[
                            np.array([0, 0, 0]),
                            push_dir * push_dist,
                        ],
                        priority=4,
                    ),
                ],
                success_criteria={
                    "object_displaced": True,
                },
            ),
            SkillSegment(
                segment_id=f"{spec_id}_retract",
                segment_type=SegmentType.FREE_SPACE,
                skill_name="retract",
                start_time=1.2,
                end_time=1.5,
                gripper_closed=True,
            ),
        ]

        return TaskSpecification(
            spec_id=spec_id,
            task_name=task_name,
            task_description=task_description,
            goal_object_id=obj_id,
            goal_position=obj_pos + push_dir * push_dist,
            segments=segments,
            keypoints=keypoints,
            success_criteria={
                "object_at_goal": True,
            },
            robot_type=robot_type,
            estimated_duration=1.5,
        )

    def _specify_generic(
        self,
        task_name: str,
        task_description: str,
        target_object: Optional[Dict[str, Any]],
        scene_objects: List[Dict[str, Any]],
        robot_type: str,
    ) -> TaskSpecification:
        """Generate specification for generic reach/interaction task."""

        spec_id = f"spec_{task_name}_{id(self)}"

        obj_pos = np.array(target_object.get("position", [0.5, 0, 0.85])) if target_object else np.array([0.5, 0, 0.85])
        obj_id = target_object.get("id", "object") if target_object else "object"

        segments = [
            SkillSegment(
                segment_id=f"{spec_id}_approach",
                segment_type=SegmentType.FREE_SPACE,
                skill_name="approach",
                start_time=0.0,
                end_time=0.5,
                gripper_closed=False,
            ),
            SkillSegment(
                segment_id=f"{spec_id}_reach",
                segment_type=SegmentType.SKILL,
                skill_name="reach",
                start_time=0.5,
                end_time=1.0,
                manipulated_object_id=obj_id,
                gripper_closed=False,
            ),
            SkillSegment(
                segment_id=f"{spec_id}_retract",
                segment_type=SegmentType.FREE_SPACE,
                skill_name="retract",
                start_time=1.0,
                end_time=1.3,
                gripper_closed=False,
            ),
        ]

        return TaskSpecification(
            spec_id=spec_id,
            task_name=task_name,
            task_description=task_description,
            goal_object_id=obj_id,
            goal_position=obj_pos,
            segments=segments,
            robot_type=robot_type,
            estimated_duration=1.3,
        )

    def _enhance_with_llm(
        self,
        spec: TaskSpecification,
        scene_objects: List[Dict[str, Any]],
    ) -> TaskSpecification:
        """
        Use Gemini to enhance the task specification with:
        - Better constraint tuning
        - Additional safety constraints
        - Timing adjustments
        - Confidence scoring

        Includes timeout and retry logic for robustness.
        """
        max_retries = 3
        timeout_seconds = 30

        for attempt in range(max_retries):
            try:
                client = self._get_client()

                prompt = self._build_enhancement_prompt(spec, scene_objects)

                # Call LLM with timeout
                start_time = time.time()
                metrics = get_metrics()
                scene_id = os.getenv("SCENE_ID", "")
                with metrics.track_api_call("gemini", "task_specification", scene_id):
                    response = client.generate(
                        prompt=prompt,
                        json_output=True,
                        temperature=0.3,
                        max_tokens=4000,
                        timeout=timeout_seconds,  # Add timeout parameter
                    )
                elapsed = time.time() - start_time

                # Check for timeout
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"LLM call exceeded {timeout_seconds}s timeout")

                data = response.parse_json()

                # Apply enhancements
                if "timing_adjustments" in data:
                    for adj in data["timing_adjustments"]:
                        seg_id = adj.get("segment_id")
                        for seg in spec.segments:
                            if seg.segment_id == seg_id:
                                if "duration_multiplier" in adj:
                                    duration = seg.end_time - seg.start_time
                                    new_duration = duration * adj["duration_multiplier"]
                                    seg.end_time = seg.start_time + new_duration

                if "additional_constraints" in data:
                    for constraint_data in data["additional_constraints"]:
                        constraint = KeypointConstraint(
                            constraint_id=constraint_data.get("id", f"llm_constraint_{len(spec.constraints)}"),
                            keypoint_id=constraint_data.get("keypoint_id", "gripper_tip"),
                            constraint_type=ConstraintType(constraint_data.get("type", "clearance")),
                            clearance_distance=constraint_data.get("clearance", 0.02),
                            priority=constraint_data.get("priority", 2),
                        )
                        spec.constraints.append(constraint)

                if "confidence" in data:
                    spec.confidence = data["confidence"]

                spec.llm_response = json.dumps(data)

                self.log(f"  LLM enhancement applied: confidence={spec.confidence:.2f}")
                return spec  # Success!

            except (TimeoutError, ConnectionError, OSError) as e:
                # Retryable errors
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    self.log(f"  LLM call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}", "WARNING")
                    time.sleep(wait_time)
                else:
                    self.log(f"  LLM enhancement failed after {max_retries} attempts: {e}", "WARNING")

            except Exception as e:
                # Non-retryable errors
                self.log(f"  LLM enhancement failed: {e}", "WARNING")
                break

        return spec

    def _build_enhancement_prompt(
        self,
        spec: TaskSpecification,
        scene_objects: List[Dict[str, Any]],
    ) -> str:
        """Build prompt for LLM enhancement."""

        objects_summary = [
            {
                "id": obj.get("id", obj.get("name", "unknown")),
                "category": obj.get("category", "object"),
                "position": obj.get("position", [0, 0, 0]),
                "dimensions": obj.get("dimensions", [0.1, 0.1, 0.1]),
            }
            for obj in scene_objects[:10]  # Limit for prompt size
        ]

        return f"""You are a robotics task planning expert. Review and enhance this task specification.

## Task
Name: {spec.task_name}
Description: {spec.task_description}
Robot: {spec.robot_type}

## Scene Objects
{json.dumps(objects_summary, indent=2)}

## Current Segments
{json.dumps([s.to_dict() for s in spec.segments], indent=2)}

## Instructions
Analyze the task and suggest improvements:
1. Timing adjustments for safer execution
2. Additional collision avoidance constraints
3. Confidence score (0-1) for task success

Return ONLY valid JSON:
{{
  "timing_adjustments": [
    {{
      "segment_id": "segment_id_here",
      "duration_multiplier": 1.2,
      "reason": "Need more time for precision"
    }}
  ],
  "additional_constraints": [
    {{
      "id": "constraint_id",
      "keypoint_id": "gripper_tip",
      "type": "clearance",
      "clearance": 0.03,
      "priority": 2,
      "reason": "Extra clearance near fragile object"
    }}
  ],
  "confidence": 0.85,
  "notes": "Task looks feasible with standard approach"
}}
"""


# =============================================================================
# Convenience Functions
# =============================================================================


def specify_task(
    task_name: str,
    task_description: str,
    scene_objects: List[Dict[str, Any]],
    target_object_id: Optional[str] = None,
    place_position: Optional[List[float]] = None,
    robot_type: str = "franka",
) -> TaskSpecification:
    """Convenience function to specify a task."""
    specifier = TaskSpecifier(verbose=False)
    return specifier.specify_task(
        task_name=task_name,
        task_description=task_description,
        scene_objects=scene_objects,
        target_object_id=target_object_id,
        place_position=place_position,
        robot_type=robot_type,
    )


if __name__ == "__main__":
    # Test the task specifier
    from tools.logging_config import init_logging

    init_logging()
    logger.info("Testing Task Specifier")
    logger.info("%s", "=" * 60)

    specifier = TaskSpecifier(verbose=True)

    # Test scene
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

    logger.info("%s", "=" * 60)
    logger.info("TASK SPECIFICATION")
    logger.info("%s", "=" * 60)
    logger.info("Task: %s", spec.task_name)
    logger.info("Segments: %s", len(spec.segments))
    for seg in spec.segments:
        logger.info(
            "  - %s (%s): %.1f-%.1fs",
            seg.skill_name,
            seg.segment_type.value,
            seg.start_time,
            seg.end_time,
        )
    logger.info("Keypoints: %s", len(spec.keypoints))
    logger.info("Constraints: %s", len(spec.constraints))
    logger.info("Confidence: %s", spec.confidence)
