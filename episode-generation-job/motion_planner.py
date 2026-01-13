#!/usr/bin/env python3
"""
AI-Powered Motion Planner for Episode Generation.

Uses Gemini to generate robot motion plans based on task descriptions and scene context.
Converts high-level task descriptions into waypoint sequences that can be executed.

Key Features:
- Task-aware motion planning via LLM
- Waypoint generation with timing
- Collision-aware trajectory suggestions
- Multi-phase motion sequences (approach, grasp, transport, place)
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from policy_config_loader import load_motion_planner_timing

# Add parent to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tools.llm_client import create_llm_client, LLMResponse
    HAVE_LLM_CLIENT = True
except ImportError:
    HAVE_LLM_CLIENT = False
    create_llm_client = None


# =============================================================================
# Data Models
# =============================================================================


class MotionPhase(str, Enum):
    """Phases of robot motion."""
    HOME = "home"
    APPROACH = "approach"
    PRE_GRASP = "pre_grasp"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    PRE_PLACE = "pre_place"
    PLACE = "place"
    RELEASE = "release"
    RETRACT = "retract"
    ARTICULATE_APPROACH = "articulate_approach"
    ARTICULATE_GRASP = "articulate_grasp"
    ARTICULATE_MOTION = "articulate_motion"


@dataclass
class Waypoint:
    """A single waypoint in robot space."""

    # End-effector pose (position + quaternion)
    position: np.ndarray  # [x, y, z] in meters
    orientation: np.ndarray  # [qw, qx, qy, qz] quaternion

    # Gripper state (0.0 = closed, 1.0 = open)
    gripper_aperture: float = 1.0

    # Timing
    timestamp: float = 0.0
    duration_to_next: float = 0.5  # seconds to reach next waypoint

    # Motion phase
    phase: MotionPhase = MotionPhase.APPROACH

    # Velocity hints
    max_velocity: float = 1.0  # m/s
    max_acceleration: float = 2.0  # m/s^2

    # Optional: pre-computed joint positions (if IK solved)
    joint_positions: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
            "gripper_aperture": self.gripper_aperture,
            "timestamp": self.timestamp,
            "duration_to_next": self.duration_to_next,
            "phase": self.phase.value,
            "max_velocity": self.max_velocity,
            "max_acceleration": self.max_acceleration,
            "joint_positions": self.joint_positions.tolist() if self.joint_positions is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Waypoint":
        """Deserialize from dictionary."""
        return cls(
            position=np.array(data["position"]),
            orientation=np.array(data["orientation"]),
            gripper_aperture=data.get("gripper_aperture", 1.0),
            timestamp=data.get("timestamp", 0.0),
            duration_to_next=data.get("duration_to_next", 0.5),
            phase=MotionPhase(data.get("phase", "approach")),
            max_velocity=data.get("max_velocity", 1.0),
            max_acceleration=data.get("max_acceleration", 2.0),
            joint_positions=np.array(data["joint_positions"]) if data.get("joint_positions") else None,
        )


@dataclass
class MotionPlan:
    """A complete motion plan with waypoints."""

    plan_id: str
    task_name: str
    task_description: str

    # Waypoint sequence
    waypoints: List[Waypoint] = field(default_factory=list)

    # Object context
    target_object_id: Optional[str] = None
    target_object_position: Optional[np.ndarray] = None
    target_object_dimensions: Optional[np.ndarray] = None

    # Placement target (for pick-place tasks)
    place_position: Optional[np.ndarray] = None

    # Articulation context (for door/drawer tasks)
    articulation_axis: Optional[np.ndarray] = None
    articulation_range: Optional[Tuple[float, float]] = None
    handle_position: Optional[np.ndarray] = None

    # Timing
    total_duration: float = 0.0

    # Confidence score from LLM
    confidence: float = 1.0

    # Robot config used
    robot_type: str = "franka"

    # Planning metadata
    planning_backend: str = "heuristic"
    planning_success: bool = True
    planning_errors: List[str] = field(default_factory=list)

    # Collision checking + joint limits
    collision_checked: bool = False
    collision_free: Optional[bool] = None
    joint_limits_enforced: bool = False

    # Joint-space trajectory (optional, collision-checked)
    joint_trajectory: Optional[np.ndarray] = None  # [T, DOF]
    joint_trajectory_timestamps: Optional[np.ndarray] = None  # [T]

    def __post_init__(self):
        """Calculate total duration."""
        if self.waypoints:
            self.total_duration = sum(w.duration_to_next for w in self.waypoints[:-1])
            # Assign timestamps
            t = 0.0
            for w in self.waypoints:
                w.timestamp = t
                t += w.duration_to_next

    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plan_id": self.plan_id,
            "task_name": self.task_name,
            "task_description": self.task_description,
            "waypoints": [w.to_dict() for w in self.waypoints],
            "target_object_id": self.target_object_id,
            "target_object_position": self.target_object_position.tolist() if self.target_object_position is not None else None,
            "target_object_dimensions": self.target_object_dimensions.tolist() if self.target_object_dimensions is not None else None,
            "place_position": self.place_position.tolist() if self.place_position is not None else None,
            "articulation_axis": self.articulation_axis.tolist() if self.articulation_axis is not None else None,
            "articulation_range": list(self.articulation_range) if self.articulation_range else None,
            "handle_position": self.handle_position.tolist() if self.handle_position is not None else None,
            "total_duration": self.total_duration,
            "confidence": self.confidence,
            "robot_type": self.robot_type,
            "planning_backend": self.planning_backend,
            "planning_success": self.planning_success,
            "planning_errors": list(self.planning_errors),
            "collision_checked": self.collision_checked,
            "collision_free": self.collision_free,
            "joint_limits_enforced": self.joint_limits_enforced,
            "joint_trajectory": self.joint_trajectory.tolist() if self.joint_trajectory is not None else None,
            "joint_trajectory_timestamps": self.joint_trajectory_timestamps.tolist()
            if self.joint_trajectory_timestamps is not None else None,
        }


@dataclass
class SceneContext:
    """Context about the scene for motion planning."""

    scene_id: str
    environment_type: str

    # Objects in scene
    objects: List[Dict[str, Any]] = field(default_factory=list)

    # Robot placement
    robot_base_position: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    robot_base_orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))

    # Workspace bounds
    workspace_min: np.ndarray = field(default_factory=lambda: np.array([-1, -1, 0]))
    workspace_max: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1.5]))

    # Table/surface height (for placing)
    surface_height: float = 0.8


# =============================================================================
# Motion Planner
# =============================================================================


class AIMotionPlanner:
    """
    AI-powered motion planner using Gemini.

    Generates robot waypoint sequences for manipulation tasks by:
    1. Analyzing the task description and scene context
    2. Using LLM to determine optimal motion phases
    3. Generating waypoints with appropriate timing and gripper states

    Usage:
        planner = AIMotionPlanner(robot_type="franka")
        plan = planner.plan_motion(
            task_name="pick_cup",
            task_description="Pick up the coffee cup from the counter",
            target_object={
                "id": "cup_001",
                "position": [0.5, 0, 0.85],
                "dimensions": [0.08, 0.08, 0.12],
            },
            place_position=[0.3, 0.2, 0.85],
        )
    """

    # Robot configurations
    ROBOT_CONFIGS = {
        "franka": {
            "reach": 0.855,  # meters
            "gripper_max_width": 0.08,
            "default_height": 0.3,  # approach height above objects
            "ee_to_gripper": 0.107,  # distance from EE frame to gripper tips
            "home_position": np.array([0.3, 0, 0.5]),
            "home_orientation": np.array([0, 1, 0, 0]),  # pointing down
            "dof": 7,
            "default_joint_positions": np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
            "joint_limits_lower": np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            "joint_limits_upper": np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
        },
        "ur10": {
            "reach": 1.3,
            "gripper_max_width": 0.1,
            "default_height": 0.35,
            "ee_to_gripper": 0.15,
            "home_position": np.array([0.4, 0, 0.6]),
            "home_orientation": np.array([0, 1, 0, 0]),
            "dof": 6,
            "default_joint_positions": np.array([0.0, -1.571, 1.571, -1.571, -1.571, 0.0]),
            "joint_limits_lower": np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
            "joint_limits_upper": np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi]),
        },
        "fetch": {
            "reach": 1.1,
            "gripper_max_width": 0.1,
            "default_height": 0.25,
            "ee_to_gripper": 0.12,
            "home_position": np.array([0.5, 0, 0.8]),
            "home_orientation": np.array([0, 1, 0, 0]),
            "dof": 7,
            "default_joint_positions": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "joint_limits_lower": np.array([-1.605, -1.221, -np.inf, -2.251, -np.inf, -2.16, -np.inf]),
            "joint_limits_upper": np.array([1.605, 1.518, np.inf, 2.251, np.inf, 2.16, np.inf]),
        },
    }

    def __init__(
        self,
        robot_type: str = "franka",
        use_llm: bool = True,
        use_curobo: bool = True,
        curobo_device: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the motion planner.

        Args:
            robot_type: Type of robot (franka, ur10, fetch)
            use_llm: Whether to use LLM for enhanced planning
            use_curobo: Whether to use cuRobo for collision-checked planning
            curobo_device: CUDA device for cuRobo (e.g., "cuda:0")
            verbose: Print debug info
        """
        self.robot_type = robot_type
        self.robot_config = self.ROBOT_CONFIGS.get(robot_type, self.ROBOT_CONFIGS["franka"])
        self.use_llm = use_llm and HAVE_LLM_CLIENT
        self.verbose = verbose
        self._client = None
        self._curobo_planner = None
        self._curobo_device = curobo_device or os.getenv("CUROBO_DEVICE", "cuda:0")
        self._use_curobo = use_curobo and os.getenv("USE_CUROBO", "true").lower() in {"1", "true", "yes"}
        self._timing = load_motion_planner_timing()

    def log(self, msg: str, level: str = "INFO") -> None:
        """Log a message."""
        if self.verbose:
            print(f"[MOTION-PLANNER] [{level}] {msg}")

    def _get_client(self):
        """Get or create LLM client."""
        if self._client is None:
            if not HAVE_LLM_CLIENT:
                raise ImportError("LLM client not available")
            self._client = create_llm_client()
        return self._client

    def _get_curobo_planner(self):
        """Get or create cuRobo planner."""
        if self._curobo_planner is None:
            try:
                from curobo_planner import create_curobo_planner, is_curobo_available
            except ImportError:
                return None

            if not is_curobo_available():
                return None

            self._curobo_planner = create_curobo_planner(
                robot_type=self.robot_type,
                device=self._curobo_device,
            )
        return self._curobo_planner

    def _build_collision_objects(self, scene_context: Optional[SceneContext]) -> List[Any]:
        """Convert scene objects to cuRobo collision objects."""
        if scene_context is None or not scene_context.objects:
            return []

        try:
            from curobo_planner import CollisionObject, CollisionGeometryType
        except ImportError:
            return []

        obstacles = []
        for obj in scene_context.objects:
            obj_id = obj.get("id", obj.get("name", "obstacle"))
            position = np.array(obj.get("position", [0, 0, 0]))
            orientation = np.array(obj.get("orientation", [1, 0, 0, 0]))
            dimensions = np.array(obj.get("dimensions", [0.1, 0.1, 0.1]))

            obstacles.append(CollisionObject(
                object_id=obj_id,
                geometry_type=CollisionGeometryType.CUBOID,
                position=position,
                orientation=orientation,
                dimensions=dimensions,
                is_static=obj.get("is_static", True),
            ))

        return obstacles

    def _plan_joint_trajectory(
        self,
        waypoints: List[Waypoint],
        scene_context: Optional[SceneContext],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, Optional[bool], List[str]]:
        """Plan collision-checked joint trajectory using cuRobo."""
        if not self._use_curobo:
            return None, None, False, None, []

        planner = self._get_curobo_planner()
        if planner is None:
            return None, None, False, None, ["cuRobo planner unavailable"]

        try:
            from curobo_planner import CuRoboPlanRequest
        except ImportError:
            return None, None, False, None, ["cuRobo planner import failed"]

        obstacles = self._build_collision_objects(scene_context)
        current_joints = self.robot_config.get("default_joint_positions")
        if current_joints is None:
            current_joints = np.zeros(self.robot_config.get("dof", 7))
        current_joints = current_joints.copy()

        joint_segments: List[np.ndarray] = []
        time_segments: List[np.ndarray] = []
        time_offset = 0.0
        collision_free = True
        errors: List[str] = []

        for idx, wp in enumerate(waypoints):
            goal_pose = np.concatenate([wp.position, wp.orientation])
            request = CuRoboPlanRequest(
                start_joint_positions=current_joints,
                goal_pose=goal_pose,
                obstacles=obstacles,
            )
            result = planner.plan_to_pose(request)

            if not result.success or result.joint_trajectory is None:
                errors.append(f"cuRobo failed at waypoint {idx}: {result.error_message}")
                return None, None, True, None, errors

            if not result.is_collision_free:
                collision_free = False

            segment_traj = result.joint_trajectory
            segment_times = result.timesteps if result.timesteps is not None else np.arange(len(segment_traj))

            if joint_segments:
                segment_traj = segment_traj[1:]
                segment_times = segment_times[1:]

            if len(segment_traj) > 0:
                joint_segments.append(segment_traj)
                time_segments.append(segment_times + time_offset)
                time_offset = time_segments[-1][-1]
                current_joints = segment_traj[-1].copy()

            wp.joint_positions = current_joints.copy()

        if not joint_segments:
            return None, None, True, None, ["cuRobo produced empty trajectory"]

        joint_trajectory = np.vstack(joint_segments)
        joint_timestamps = np.concatenate(time_segments)

        limits_lower = self.robot_config.get("joint_limits_lower")
        limits_upper = self.robot_config.get("joint_limits_upper")
        if limits_lower is not None and limits_upper is not None:
            out_of_bounds = np.logical_or(
                joint_trajectory < limits_lower - 1e-5,
                joint_trajectory > limits_upper + 1e-5,
            )
            if np.any(out_of_bounds):
                errors.append("cuRobo trajectory violates joint limits")

        return joint_trajectory, joint_timestamps, True, collision_free, errors

    def plan_motion(
        self,
        task_name: str,
        task_description: str,
        target_object: Optional[Dict[str, Any]] = None,
        place_position: Optional[List[float]] = None,
        articulation_info: Optional[Dict[str, Any]] = None,
        scene_context: Optional[SceneContext] = None,
    ) -> MotionPlan:
        """
        Generate a motion plan for a manipulation task.

        Args:
            task_name: Name of the task (e.g., "pick_cup")
            task_description: Natural language description
            target_object: Object to manipulate (id, position, dimensions)
            place_position: Where to place the object (for pick-place)
            articulation_info: For articulated objects (axis, range, handle)
            scene_context: Full scene context

        Returns:
            MotionPlan with waypoint sequence
        """
        self.log(f"Planning motion: {task_name}")

        # Determine task type
        task_type = self._classify_task(task_name, task_description)
        self.log(f"  Task type: {task_type}")

        # Get target position
        target_pos = None
        target_dims = None
        if target_object:
            target_pos = np.array(target_object.get("position", [0.5, 0, 0.8]))
            target_dims = np.array(target_object.get("dimensions", [0.05, 0.05, 0.1]))

        # Generate waypoints based on task type
        if task_type == "pick_place":
            waypoints = self._plan_pick_place(
                target_pos=target_pos,
                target_dims=target_dims,
                place_pos=np.array(place_position) if place_position else None,
            )
        elif task_type == "articulation":
            waypoints = self._plan_articulation(
                handle_pos=np.array(articulation_info.get("handle_position", target_pos)) if articulation_info else target_pos,
                axis=np.array(articulation_info.get("axis", [0, 1, 0])) if articulation_info else np.array([0, 1, 0]),
                motion_range=articulation_info.get("range", [0, 0.5]) if articulation_info else [0, 0.5],
                motion_type=articulation_info.get("type", "prismatic") if articulation_info else "prismatic",
            )
        elif task_type == "push":
            waypoints = self._plan_push(
                target_pos=target_pos,
                push_direction=np.array([1, 0, 0]),  # Default push forward
                push_distance=0.2,
            )
        else:
            # Default: simple approach and reach
            waypoints = self._plan_simple_reach(target_pos=target_pos)

        # Optionally enhance with LLM
        if self.use_llm and target_object:
            waypoints = self._enhance_with_llm(
                waypoints=waypoints,
                task_name=task_name,
                task_description=task_description,
                target_object=target_object,
            )

        plan = MotionPlan(
            plan_id=f"plan_{task_name}_{id(self)}",
            task_name=task_name,
            task_description=task_description,
            waypoints=waypoints,
            target_object_id=target_object.get("id") if target_object else None,
            target_object_position=target_pos,
            target_object_dimensions=target_dims,
            place_position=np.array(place_position) if place_position else None,
            robot_type=self.robot_type,
        )

        joint_trajectory, joint_timestamps, collision_checked, collision_free, errors = (
            self._plan_joint_trajectory(waypoints=waypoints, scene_context=scene_context)
        )
        if collision_checked:
            plan.planning_backend = "curobo"
            plan.collision_checked = True
            plan.collision_free = collision_free
            plan.joint_limits_enforced = True
            plan.joint_trajectory = joint_trajectory
            plan.joint_trajectory_timestamps = joint_timestamps
            if errors:
                plan.planning_errors.extend(errors)
                plan.planning_success = False
            if collision_free is False:
                plan.planning_errors.append("cuRobo reported collision in planned trajectory")
                plan.planning_success = False

        if plan.planning_errors:
            self.log(f"  Planning errors: {plan.planning_errors}", "WARNING")

        self.log(f"  Generated {plan.num_waypoints} waypoints, duration: {plan.total_duration:.2f}s")

        return plan

    def _classify_task(self, task_name: str, description: str) -> str:
        """Classify the task type."""
        text = f"{task_name} {description}".lower()

        if any(word in text for word in ["pick", "grasp", "grab", "lift", "place", "put", "move"]):
            return "pick_place"
        elif any(word in text for word in ["open", "close", "pull", "push drawer", "door"]):
            return "articulation"
        elif any(word in text for word in ["push", "slide"]):
            return "push"
        else:
            return "reach"

    def _plan_pick_place(
        self,
        target_pos: np.ndarray,
        target_dims: Optional[np.ndarray] = None,
        place_pos: Optional[np.ndarray] = None,
    ) -> List[Waypoint]:
        """Generate waypoints for pick-and-place task."""
        waypoints = []

        if target_pos is None:
            target_pos = np.array([0.5, 0, 0.8])
        if target_dims is None:
            target_dims = np.array([0.05, 0.05, 0.1])

        # Grasp height adjustment (grasp at middle of object)
        grasp_height = target_pos[2] + target_dims[2] / 2
        approach_height = grasp_height + self.robot_config["default_height"]

        # Orientation: gripper pointing down
        down_orientation = np.array([0, 1, 0, 0])  # w, x, y, z

        # 1. Home position
        waypoints.append(Waypoint(
            position=self.robot_config["home_position"].copy(),
            orientation=self.robot_config["home_orientation"].copy(),
            gripper_aperture=1.0,
            phase=MotionPhase.HOME,
            duration_to_next=self._timing["home"],
        ))

        # 2. Approach above object
        waypoints.append(Waypoint(
            position=np.array([target_pos[0], target_pos[1], approach_height]),
            orientation=down_orientation.copy(),
            gripper_aperture=1.0,
            phase=MotionPhase.APPROACH,
            duration_to_next=self._timing["approach"],
        ))

        # 3. Pre-grasp (just above object)
        waypoints.append(Waypoint(
            position=np.array([target_pos[0], target_pos[1], grasp_height + 0.05]),
            orientation=down_orientation.copy(),
            gripper_aperture=1.0,
            phase=MotionPhase.PRE_GRASP,
            duration_to_next=self._timing["pre_grasp"],
        ))

        # 4. Grasp position
        waypoints.append(Waypoint(
            position=np.array([target_pos[0], target_pos[1], grasp_height]),
            orientation=down_orientation.copy(),
            gripper_aperture=1.0,
            phase=MotionPhase.GRASP,
            duration_to_next=self._timing["grasp"],
        ))

        # 5. Close gripper
        waypoints.append(Waypoint(
            position=np.array([target_pos[0], target_pos[1], grasp_height]),
            orientation=down_orientation.copy(),
            gripper_aperture=0.0,  # Closed
            phase=MotionPhase.GRASP,
            duration_to_next=self._timing["close_gripper"],
        ))

        # 6. Lift
        waypoints.append(Waypoint(
            position=np.array([target_pos[0], target_pos[1], approach_height]),
            orientation=down_orientation.copy(),
            gripper_aperture=0.0,
            phase=MotionPhase.LIFT,
            duration_to_next=self._timing["lift"],
        ))

        if place_pos is not None:
            place_height = place_pos[2] + 0.05  # Slightly above surface

            # 7. Transport to above place position
            waypoints.append(Waypoint(
                position=np.array([place_pos[0], place_pos[1], approach_height]),
                orientation=down_orientation.copy(),
                gripper_aperture=0.0,
                phase=MotionPhase.TRANSPORT,
                duration_to_next=self._timing["transport"],
            ))

            # 8. Pre-place
            waypoints.append(Waypoint(
                position=np.array([place_pos[0], place_pos[1], place_height + 0.05]),
                orientation=down_orientation.copy(),
                gripper_aperture=0.0,
                phase=MotionPhase.PRE_PLACE,
                duration_to_next=self._timing["pre_place"],
            ))

            # 9. Place
            waypoints.append(Waypoint(
                position=np.array([place_pos[0], place_pos[1], place_height]),
                orientation=down_orientation.copy(),
                gripper_aperture=0.0,
                phase=MotionPhase.PLACE,
                duration_to_next=self._timing["place"],
            ))

            # 10. Release
            waypoints.append(Waypoint(
                position=np.array([place_pos[0], place_pos[1], place_height]),
                orientation=down_orientation.copy(),
                gripper_aperture=1.0,  # Open
                phase=MotionPhase.RELEASE,
                duration_to_next=self._timing["release"],
            ))

            # 11. Retract
            waypoints.append(Waypoint(
                position=np.array([place_pos[0], place_pos[1], approach_height]),
                orientation=down_orientation.copy(),
                gripper_aperture=1.0,
                phase=MotionPhase.RETRACT,
                duration_to_next=self._timing["retract"],
            ))

        # 12. Return home
        waypoints.append(Waypoint(
            position=self.robot_config["home_position"].copy(),
            orientation=self.robot_config["home_orientation"].copy(),
            gripper_aperture=1.0,
            phase=MotionPhase.HOME,
            duration_to_next=self._timing["return_home"],
        ))

        return waypoints

    def _plan_articulation(
        self,
        handle_pos: np.ndarray,
        axis: np.ndarray,
        motion_range: List[float],
        motion_type: str = "prismatic",
    ) -> List[Waypoint]:
        """Generate waypoints for articulation task (drawer/door)."""
        waypoints = []

        # Normalize axis
        axis = axis / np.linalg.norm(axis)

        # Calculate approach direction (perpendicular to surface)
        if motion_type == "prismatic":
            # Drawer: approach from front
            approach_dir = -axis
        else:
            # Door: approach from side
            approach_dir = np.cross(axis, np.array([0, 0, 1]))
            if np.linalg.norm(approach_dir) < 0.1:
                approach_dir = np.array([1, 0, 0])
            approach_dir = approach_dir / np.linalg.norm(approach_dir)

        approach_height = handle_pos[2] + 0.15
        down_orientation = np.array([0, 1, 0, 0])

        # Motion distance
        motion_dist = motion_range[1] - motion_range[0]

        # 1. Home
        waypoints.append(Waypoint(
            position=self.robot_config["home_position"].copy(),
            orientation=self.robot_config["home_orientation"].copy(),
            gripper_aperture=1.0,
            phase=MotionPhase.HOME,
            duration_to_next=self._timing["home"],
        ))

        # 2. Approach above handle
        waypoints.append(Waypoint(
            position=np.array([handle_pos[0], handle_pos[1], approach_height]),
            orientation=down_orientation.copy(),
            gripper_aperture=1.0,
            phase=MotionPhase.ARTICULATE_APPROACH,
            duration_to_next=self._timing["articulation_approach"],
        ))

        # 3. Pre-grasp handle
        waypoints.append(Waypoint(
            position=handle_pos + approach_dir * 0.08,
            orientation=down_orientation.copy(),
            gripper_aperture=1.0,
            phase=MotionPhase.ARTICULATE_APPROACH,
            duration_to_next=self._timing["articulation_pre_grasp"],
        ))

        # 4. Grasp handle
        waypoints.append(Waypoint(
            position=handle_pos.copy(),
            orientation=down_orientation.copy(),
            gripper_aperture=1.0,
            phase=MotionPhase.ARTICULATE_GRASP,
            duration_to_next=self._timing["articulation_grasp"],
        ))

        # 5. Close gripper
        waypoints.append(Waypoint(
            position=handle_pos.copy(),
            orientation=down_orientation.copy(),
            gripper_aperture=0.0,
            phase=MotionPhase.ARTICULATE_GRASP,
            duration_to_next=self._timing["articulation_close"],
        ))

        # 6. Articulate motion (pull/push)
        end_pos = handle_pos + axis * motion_dist
        waypoints.append(Waypoint(
            position=end_pos,
            orientation=down_orientation.copy(),
            gripper_aperture=0.0,
            phase=MotionPhase.ARTICULATE_MOTION,
            duration_to_next=self._timing["articulation_motion"],  # Slow motion for articulation
        ))

        # 7. Release
        waypoints.append(Waypoint(
            position=end_pos,
            orientation=down_orientation.copy(),
            gripper_aperture=1.0,
            phase=MotionPhase.RELEASE,
            duration_to_next=self._timing["articulation_release"],
        ))

        # 8. Retract
        waypoints.append(Waypoint(
            position=end_pos + approach_dir * 0.15,
            orientation=down_orientation.copy(),
            gripper_aperture=1.0,
            phase=MotionPhase.RETRACT,
            duration_to_next=self._timing["articulation_retract"],
        ))

        # 9. Home
        waypoints.append(Waypoint(
            position=self.robot_config["home_position"].copy(),
            orientation=self.robot_config["home_orientation"].copy(),
            gripper_aperture=1.0,
            phase=MotionPhase.HOME,
            duration_to_next=self._timing["return_home"],
        ))

        return waypoints

    def _plan_push(
        self,
        target_pos: np.ndarray,
        push_direction: np.ndarray,
        push_distance: float = 0.2,
    ) -> List[Waypoint]:
        """Generate waypoints for pushing task."""
        waypoints = []

        # Normalize direction
        push_direction = push_direction / np.linalg.norm(push_direction)

        approach_height = target_pos[2] + 0.15
        contact_height = target_pos[2]
        down_orientation = np.array([0, 1, 0, 0])

        # Start position (behind object relative to push direction)
        start_pos = target_pos - push_direction * 0.1
        end_pos = target_pos + push_direction * push_distance

        waypoints = [
            # Home
            Waypoint(
                position=self.robot_config["home_position"].copy(),
                orientation=self.robot_config["home_orientation"].copy(),
                gripper_aperture=0.0,  # Closed for pushing
                phase=MotionPhase.HOME,
                duration_to_next=self._timing["push_home"],
            ),
            # Approach
            Waypoint(
                position=np.array([start_pos[0], start_pos[1], approach_height]),
                orientation=down_orientation.copy(),
                gripper_aperture=0.0,
                phase=MotionPhase.APPROACH,
                duration_to_next=self._timing["push_approach"],
            ),
            # Lower to contact
            Waypoint(
                position=np.array([start_pos[0], start_pos[1], contact_height]),
                orientation=down_orientation.copy(),
                gripper_aperture=0.0,
                phase=MotionPhase.PRE_GRASP,
                duration_to_next=self._timing["push_lower"],
            ),
            # Push
            Waypoint(
                position=np.array([end_pos[0], end_pos[1], contact_height]),
                orientation=down_orientation.copy(),
                gripper_aperture=0.0,
                phase=MotionPhase.TRANSPORT,
                duration_to_next=self._timing["push_motion"],
            ),
            # Lift
            Waypoint(
                position=np.array([end_pos[0], end_pos[1], approach_height]),
                orientation=down_orientation.copy(),
                gripper_aperture=0.0,
                phase=MotionPhase.RETRACT,
                duration_to_next=self._timing["push_lift"],
            ),
            # Home
            Waypoint(
                position=self.robot_config["home_position"].copy(),
                orientation=self.robot_config["home_orientation"].copy(),
                gripper_aperture=1.0,
                phase=MotionPhase.HOME,
                duration_to_next=self._timing["push_return_home"],
            ),
        ]

        return waypoints

    def _plan_simple_reach(self, target_pos: Optional[np.ndarray] = None) -> List[Waypoint]:
        """Generate waypoints for simple reach task."""
        if target_pos is None:
            target_pos = np.array([0.5, 0, 0.8])

        approach_height = target_pos[2] + 0.2
        down_orientation = np.array([0, 1, 0, 0])

        return [
            Waypoint(
                position=self.robot_config["home_position"].copy(),
                orientation=self.robot_config["home_orientation"].copy(),
                gripper_aperture=1.0,
                phase=MotionPhase.HOME,
                duration_to_next=self._timing["simple_home"],
            ),
            Waypoint(
                position=np.array([target_pos[0], target_pos[1], approach_height]),
                orientation=down_orientation.copy(),
                gripper_aperture=1.0,
                phase=MotionPhase.APPROACH,
                duration_to_next=self._timing["simple_approach"],
            ),
            Waypoint(
                position=target_pos.copy(),
                orientation=down_orientation.copy(),
                gripper_aperture=1.0,
                phase=MotionPhase.PRE_GRASP,
                duration_to_next=self._timing["simple_reach"],
            ),
            Waypoint(
                position=self.robot_config["home_position"].copy(),
                orientation=self.robot_config["home_orientation"].copy(),
                gripper_aperture=1.0,
                phase=MotionPhase.HOME,
                duration_to_next=self._timing["simple_return_home"],
            ),
        ]

    def _enhance_with_llm(
        self,
        waypoints: List[Waypoint],
        task_name: str,
        task_description: str,
        target_object: Dict[str, Any],
    ) -> List[Waypoint]:
        """Use LLM to enhance motion plan with task-specific adjustments."""

        if not self.use_llm:
            return waypoints

        try:
            client = self._get_client()

            prompt = self._build_enhancement_prompt(
                waypoints=waypoints,
                task_name=task_name,
                task_description=task_description,
                target_object=target_object,
            )

            response = client.generate(
                prompt=prompt,
                json_output=True,
                temperature=0.3,
                max_tokens=4000,
            )

            data = response.parse_json()
            adjustments = data.get("adjustments", [])

            # Apply adjustments
            for adj in adjustments:
                idx = adj.get("waypoint_index")
                if idx is not None and 0 <= idx < len(waypoints):
                    if "duration_to_next" in adj:
                        waypoints[idx].duration_to_next = adj["duration_to_next"]
                    if "gripper_aperture" in adj:
                        waypoints[idx].gripper_aperture = adj["gripper_aperture"]
                    if "max_velocity" in adj:
                        waypoints[idx].max_velocity = adj["max_velocity"]

            self.log(f"  Applied {len(adjustments)} LLM adjustments")

        except Exception as e:
            self.log(f"  LLM enhancement failed: {e}", "WARNING")

        return waypoints

    def _build_enhancement_prompt(
        self,
        waypoints: List[Waypoint],
        task_name: str,
        task_description: str,
        target_object: Dict[str, Any],
    ) -> str:
        """Build prompt for motion plan enhancement."""

        waypoint_summary = []
        for i, w in enumerate(waypoints):
            waypoint_summary.append({
                "index": i,
                "phase": w.phase.value,
                "position": w.position.tolist(),
                "gripper": w.gripper_aperture,
                "duration": w.duration_to_next,
            })

        return f"""You are a robotics motion planning expert. Review this motion plan and suggest timing/velocity adjustments.

## Task
Name: {task_name}
Description: {task_description}

## Target Object
{json.dumps(target_object, indent=2)}

## Current Waypoints
{json.dumps(waypoint_summary, indent=2)}

## Instructions
Suggest adjustments to improve the motion plan. Consider:
1. Approach speed (slower near objects for safety)
2. Grasp timing (allow time for gripper to close)
3. Lift/place smoothness (avoid jerky motions)
4. Object properties (fragile objects need gentler handling)

Return ONLY valid JSON:
{{
  "adjustments": [
    {{
      "waypoint_index": 3,
      "duration_to_next": 0.5,
      "reason": "Slower approach for fragile object"
    }}
  ],
  "notes": "Optional notes about the adjustments"
}}
"""


# =============================================================================
# Convenience Functions
# =============================================================================


def plan_pick_place(
    target_position: List[float],
    target_dimensions: List[float],
    place_position: List[float],
    robot_type: str = "franka",
    use_llm: bool = True,
) -> MotionPlan:
    """Convenience function for pick-and-place motion planning."""
    planner = AIMotionPlanner(robot_type=robot_type, use_llm=use_llm)
    return planner.plan_motion(
        task_name="pick_place",
        task_description="Pick up object and place it at target location",
        target_object={
            "id": "target_object",
            "position": target_position,
            "dimensions": target_dimensions,
        },
        place_position=place_position,
    )


def plan_drawer_open(
    handle_position: List[float],
    pull_distance: float = 0.3,
    robot_type: str = "franka",
) -> MotionPlan:
    """Convenience function for drawer opening."""
    planner = AIMotionPlanner(robot_type=robot_type, use_llm=False)
    return planner.plan_motion(
        task_name="open_drawer",
        task_description="Open a drawer by pulling the handle",
        target_object={
            "id": "drawer_handle",
            "position": handle_position,
            "dimensions": [0.1, 0.02, 0.02],
        },
        articulation_info={
            "handle_position": handle_position,
            "axis": [-1, 0, 0],  # Pull toward robot
            "range": [0, pull_distance],
            "type": "prismatic",
        },
    )


if __name__ == "__main__":
    # Test the motion planner
    planner = AIMotionPlanner(robot_type="franka", use_llm=False, verbose=True)

    # Test pick-place
    plan = planner.plan_motion(
        task_name="pick_cup",
        task_description="Pick up the coffee cup from the counter and place it on the shelf",
        target_object={
            "id": "cup_001",
            "position": [0.5, 0.1, 0.85],
            "dimensions": [0.08, 0.08, 0.12],
        },
        place_position=[0.3, -0.2, 0.9],
    )

    print(f"\nGenerated plan: {plan.num_waypoints} waypoints, {plan.total_duration:.2f}s")
    for i, w in enumerate(plan.waypoints):
        print(f"  {i}: {w.phase.value} @ {w.timestamp:.2f}s, gripper={w.gripper_aperture}")
