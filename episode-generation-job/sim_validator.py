#!/usr/bin/env python3
"""
Simulation Validator for Episode Generation.

This module validates generated episodes by:
1. Executing trajectories in ACTUAL physics simulation (PhysX via Isaac Sim)
2. Scoring success/failure based on real physics results
3. Computing quality metrics from simulation data
4. Filtering out failed episodes

This is what makes episodes "sellable" - verified, quality-scored data.

IMPORTANT: For production validation, this module should be run inside Isaac Sim
to use actual PhysX simulation. When running outside Isaac Sim, it falls back
to heuristic-based validation (less accurate but useful for testing).

Key Metrics (from research recommendations):
- Success/failure rate (from actual simulation)
- Collision count (from PhysX contact reports)
- Joint limit violations (checked during execution)
- Contact stability (gripper force monitoring)
- Time-to-completion distribution

Reference:
- AnyTask: Uses sim validation + success scoring
- CP-Gen: Reports success rates as key metric
- DemoGen: Emphasizes collision-free, feasible trajectories
"""

import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_planner import MotionPlan, Waypoint
from trajectory_solver import JointTrajectory, JointState, ROBOT_CONFIGS
from quality_constants import (
    MIN_QUALITY_SCORE,
    MAX_RETRIES,
    STABILITY_THRESHOLD,
    COLLISION_PENETRATION_THRESHOLD,
)

# Import Isaac Sim integration
try:
    from isaac_sim_integration import (
        is_isaac_sim_available,
        is_physx_available,
        PhysicsSimulator,
        PhysicsStepResult,
        get_isaac_sim_session,
    )
    _HAVE_PHYSICS_INTEGRATION = True
except ImportError:
    _HAVE_PHYSICS_INTEGRATION = False
    def is_isaac_sim_available() -> bool:
        return False
    def is_physx_available() -> bool:
        return False


# =============================================================================
# Data Models
# =============================================================================


class ValidationStatus(str, Enum):
    """Validation status for an episode."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_RETRY = "needs_retry"


class FailureReason(str, Enum):
    """Reasons for episode failure."""
    COLLISION = "collision"
    JOINT_LIMIT = "joint_limit"
    SELF_COLLISION = "self_collision"
    TASK_FAILURE = "task_failure"
    INSTABILITY = "instability"
    TIMEOUT = "timeout"
    IK_FAILURE = "ik_failure"
    EXCESSIVE_VELOCITY = "excessive_velocity"
    GRASP_FAILURE = "grasp_failure"
    PLACEMENT_FAILURE = "placement_failure"
    PLANNING_FAILURE = "planning_failure"


@dataclass
class CollisionEvent:
    """Record of a collision during execution."""
    frame_idx: int
    timestamp: float
    body_a: str
    body_b: str
    contact_point: np.ndarray
    contact_force: float
    is_expected: bool = False  # True if collision is part of the task (grasping)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "body_a": self.body_a,
            "body_b": self.body_b,
            "contact_point": self.contact_point.tolist(),
            "contact_force": self.contact_force,
            "is_expected": self.is_expected,
        }


@dataclass
class JointLimitEvent:
    """Record of a joint limit violation."""
    frame_idx: int
    timestamp: float
    joint_name: str
    joint_value: float
    limit_type: str  # "lower" or "upper"
    limit_value: float
    violation_amount: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "joint_name": self.joint_name,
            "joint_value": self.joint_value,
            "limit_type": self.limit_type,
            "limit_value": self.limit_value,
            "violation_amount": self.violation_amount,
        }


@dataclass
class QualityMetrics:
    """
    Quality metrics for an episode.

    These metrics are what customers care about when buying training data.
    """
    # Success
    task_success: bool = False
    grasp_success: bool = False
    placement_success: bool = False

    # Collision metrics
    total_collisions: int = 0
    unexpected_collisions: int = 0
    max_collision_force: float = 0.0

    # Joint metrics
    joint_limit_violations: int = 0
    max_joint_violation: float = 0.0
    max_joint_velocity: float = 0.0
    max_joint_acceleration: float = 0.0
    torque_limit_violations: int = 0  # P1-6 FIX: Count of torque violations

    # Stability metrics
    gripper_slip_events: int = 0
    object_dropped: bool = False
    object_stable_at_end: bool = True

    # Timing metrics
    total_duration: float = 0.0
    execution_time: float = 0.0  # Actual sim time

    # Trajectory smoothness
    path_length: float = 0.0
    jerk_integral: float = 0.0  # Lower is smoother
    velocity_smoothness: float = 1.0  # 0-1, higher is smoother

    # Overall score (0-1)
    overall_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_success": self.task_success,
            "grasp_success": self.grasp_success,
            "placement_success": self.placement_success,
            "total_collisions": self.total_collisions,
            "unexpected_collisions": self.unexpected_collisions,
            "max_collision_force": self.max_collision_force,
            "joint_limit_violations": self.joint_limit_violations,
            "max_joint_violation": self.max_joint_violation,
            "max_joint_velocity": self.max_joint_velocity,
            "max_joint_acceleration": self.max_joint_acceleration,
            "torque_limit_violations": self.torque_limit_violations,  # P1-6 FIX
            "gripper_slip_events": self.gripper_slip_events,
            "object_dropped": self.object_dropped,
            "object_stable_at_end": self.object_stable_at_end,
            "total_duration": self.total_duration,
            "execution_time": self.execution_time,
            "path_length": self.path_length,
            "jerk_integral": self.jerk_integral,
            "velocity_smoothness": self.velocity_smoothness,
            "overall_score": self.overall_score,
        }

    def compute_overall_score(self) -> float:
        """Compute overall quality score."""
        score = 0.0

        # Task success is primary (40%)
        if self.task_success:
            score += 0.4

        # Grasp and placement (20%)
        if self.grasp_success:
            score += 0.1
        if self.placement_success:
            score += 0.1

        # No unexpected collisions (15%)
        if self.unexpected_collisions == 0:
            score += 0.15

        # No joint violations (10%)
        if self.joint_limit_violations == 0:
            score += 0.1

        # Object stability (10%)
        if not self.object_dropped and self.object_stable_at_end:
            score += 0.1

        # Smoothness bonus (5%)
        score += 0.05 * self.velocity_smoothness

        self.overall_score = min(1.0, score)
        return self.overall_score


@dataclass
class ValidationResult:
    """Complete validation result for an episode."""

    episode_id: str
    status: ValidationStatus

    # Quality metrics
    metrics: QualityMetrics = field(default_factory=QualityMetrics)

    # Events
    collision_events: List[CollisionEvent] = field(default_factory=list)
    joint_limit_events: List[JointLimitEvent] = field(default_factory=list)

    # Failure info
    failure_reasons: List[FailureReason] = field(default_factory=list)
    failure_details: str = ""

    # Retry info
    retry_count: int = 0
    can_retry: bool = False
    retry_suggestion: str = ""

    # Timing
    validation_time_seconds: float = 0.0

    # P1-4 FIX: Physics backend used for validation
    physics_backend: str = "unknown"  # "isaac_sim", "heuristic", or "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "status": self.status.value,
            "metrics": self.metrics.to_dict(),
            "collision_events": [e.to_dict() for e in self.collision_events],
            "joint_limit_events": [e.to_dict() for e in self.joint_limit_events],
            "failure_reasons": [r.value for r in self.failure_reasons],
            "failure_details": self.failure_details,
            "retry_count": self.retry_count,
            "can_retry": self.can_retry,
            "retry_suggestion": self.retry_suggestion,
            "validation_time_seconds": self.validation_time_seconds,
            "physics_backend": self.physics_backend,  # P1-4 FIX
        }


@dataclass
class ValidationConfig:
    """
    Configuration for validation.

    LABS-BLOCKER-002 FIX: Raised quality thresholds to production levels.
    Previous thresholds were too lenient (70% quality, 80% collision-free).
    New thresholds prevent shipping low-quality data to labs.
    """

    # Thresholds
    max_unexpected_collisions: int = 0
    max_joint_violations: int = 0
    max_collision_force: float = 100.0  # Newtons
    max_joint_velocity: float = 5.0  # rad/s
    max_joint_acceleration: float = 20.0  # rad/s^2

    # Success criteria
    require_task_success: bool = True
    require_grasp_success: bool = True
    require_placement_success: bool = True

    # Stability
    require_object_stable: bool = True
    stability_threshold: float = STABILITY_THRESHOLD  # Unified constant

    # Quality thresholds - LABS-BLOCKER-002 FIX: Uses unified quality constants
    # Imported from quality_constants.py to ensure consistency across pipeline
    min_quality_score: float = MIN_QUALITY_SCORE  # 0.85 - unified threshold

    # Retry settings
    max_retries: int = MAX_RETRIES  # Unified constant
    retry_on_collision: bool = True
    retry_on_joint_limit: bool = True


# =============================================================================
# Simulation Validator
# =============================================================================


class SimulationValidator:
    """
    Validates episodes through physics simulation.

    This validator supports two modes:
    1. **Real Physics Mode** (Isaac Sim available): Runs actual PhysX simulation
       to validate trajectories, getting real collision data and physics feedback.
    2. **Heuristic Mode** (fallback): Uses geometric checks and kinematic analysis
       when Isaac Sim is not available.

    For production training data, always use Real Physics Mode.

    Usage:
        validator = SimulationValidator(robot_type="franka")

        # Check what mode we're in
        if validator.is_using_real_physics():
            print("Using Isaac Sim PhysX for validation")
        else:
            print("Using heuristic validation (less accurate)")

        # Validate
        result = validator.validate(trajectory, motion_plan, scene_objects)
    """

    def __init__(
        self,
        robot_type: str = "franka",
        config: Optional[ValidationConfig] = None,
        scene_usd_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the simulation validator.

        Args:
            robot_type: Robot type (franka, ur10, fetch)
            config: Validation configuration
            scene_usd_path: Path to USD scene for physics simulation
            verbose: Print debug info
        """
        self.robot_type = robot_type
        self.robot_config = ROBOT_CONFIGS.get(robot_type, ROBOT_CONFIGS["franka"])
        self.config = config or ValidationConfig()
        self.verbose = verbose
        self._scene_usd_path = scene_usd_path

        # Check physics availability
        self._physics_available = _HAVE_PHYSICS_INTEGRATION and is_physx_available()
        self._physics_sim: Optional["PhysicsSimulator"] = None

        if self._physics_available:
            self.log("PhysX available - using real physics validation")
            self._init_physics_simulator()
        else:
            self.log("PhysX not available - using heuristic validation", "WARNING")
            self.log("For production data, run with: /isaac-sim/python.sh", "WARNING")

    def _init_physics_simulator(self) -> None:
        """Initialize the physics simulator."""
        if not self._physics_available:
            return

        try:
            self._physics_sim = PhysicsSimulator(
                dt=1.0 / 240.0,  # 240 Hz physics
                substeps=4,
                verbose=self.verbose,
            )

            if self._scene_usd_path:
                self._physics_sim.load_scene(self._scene_usd_path)
                self.log(f"Loaded scene for physics: {self._scene_usd_path}")

        except Exception as e:
            self.log(f"Failed to initialize physics simulator: {e}", "ERROR")
            self._physics_available = False

    def is_using_real_physics(self) -> bool:
        """Check if using real PhysX simulation."""
        return self._physics_available and self._physics_sim is not None

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[SIM-VALIDATOR] [{level}] {msg}")

    def validate(
        self,
        trajectory: JointTrajectory,
        motion_plan: MotionPlan,
        scene_objects: List[Dict[str, Any]],
        task_success_checker: Optional[Callable] = None,
    ) -> ValidationResult:
        """
        Validate an episode trajectory.

        If PhysX is available, runs actual physics simulation.
        Otherwise, uses heuristic-based validation.

        Args:
            trajectory: Joint trajectory to validate
            motion_plan: Original motion plan for reference
            scene_objects: Objects in the scene
            task_success_checker: Optional callable to check task success

        Returns:
            ValidationResult with metrics and status
        """
        start_time = time.time()

        episode_id = trajectory.trajectory_id
        self.log(f"Validating episode: {episode_id}")
        self.log(f"  Mode: {'PhysX Simulation' if self.is_using_real_physics() else 'Heuristic'}")

        result = ValidationResult(
            episode_id=episode_id,
            status=ValidationStatus.PENDING,
        )

        if not getattr(motion_plan, "planning_success", True):
            result.failure_reasons.append(FailureReason.PLANNING_FAILURE)
            result.failure_details = "; ".join(getattr(motion_plan, "planning_errors", []))
            result.status = self._determine_status(result)
            result.validation_time_seconds = time.time() - start_time
            self._determine_retry_info(result)
            return result

        # Use real physics if available
        if self.is_using_real_physics():
            result.physics_backend = "isaac_sim"  # P1-4 FIX: Set physics backend
            result = self._validate_with_physics(
                trajectory, motion_plan, scene_objects, result, task_success_checker
            )
        else:
            # Fall back to heuristic validation
            result.physics_backend = "heuristic"  # P1-4 FIX: Set physics backend
            result = self._validate_heuristic(
                trajectory, motion_plan, scene_objects, result, task_success_checker
            )

        # Compute quality score
        result.metrics.compute_overall_score()

        # Determine final status
        result.status = self._determine_status(result)
        result.validation_time_seconds = time.time() - start_time

        # Determine if retry is possible
        self._determine_retry_info(result)

        self.log(f"  Status: {result.status.value}")
        self.log(f"  Score: {result.metrics.overall_score:.2f}")
        if result.failure_reasons:
            self.log(f"  Failures: {[r.value for r in result.failure_reasons]}")

        return result

    def _validate_with_physics(
        self,
        trajectory: JointTrajectory,
        motion_plan: MotionPlan,
        scene_objects: List[Dict[str, Any]],
        result: ValidationResult,
        task_success_checker: Optional[Callable],
    ) -> ValidationResult:
        """
        Validate using actual PhysX simulation.

        This is the gold standard for validation - runs the trajectory
        through real physics and captures actual collisions, forces, etc.
        """
        self.log("  Running PhysX simulation...")

        # Set up tracking for objects
        for obj in scene_objects:
            obj_id = obj.get("id", obj.get("name", ""))
            prim_path = obj.get("prim_path", f"/World/Objects/{obj_id}")
            self._physics_sim.add_tracked_object(obj_id, prim_path)

        # Set robot tracking
        robot_prim = f"/World/Robots/{self.robot_type}"
        self._physics_sim.set_robot(robot_prim)

        # Get joint trajectory as array
        joint_positions = trajectory.get_joint_positions_array()
        gripper_positions = trajectory.get_gripper_positions()

        # Run simulation
        try:
            physics_results = self._physics_sim.run_trajectory(
                joint_trajectory=joint_positions,
                dt=1.0 / trajectory.fps,
                gripper_trajectory=gripper_positions,
            )

            # Analyze physics results
            self._analyze_physics_results(
                physics_results, trajectory, motion_plan, scene_objects, result
            )

        except Exception as e:
            self.log(f"  Physics simulation failed: {e}", "ERROR")
            result.failure_reasons.append(FailureReason.TIMEOUT)
            result.failure_details = str(e)

        # Also run kinematic checks
        self._check_joint_limits(trajectory, result)
        self._check_velocities(trajectory, result)
        self._check_torques(trajectory, result)  # P1-6 FIX: Check torque limits
        self._check_trajectory_smoothness(trajectory, result)

        # Task success check
        if task_success_checker:
            result.metrics.task_success = task_success_checker(trajectory, motion_plan)
        else:
            # Infer from physics results
            result.metrics.task_success = (
                len(result.failure_reasons) == 0 and
                result.metrics.unexpected_collisions == 0
            )

        return result

    def _analyze_physics_results(
        self,
        physics_results: List["PhysicsStepResult"],
        trajectory: JointTrajectory,
        motion_plan: MotionPlan,
        scene_objects: List[Dict[str, Any]],
        result: ValidationResult,
    ) -> None:
        """Analyze results from physics simulation."""

        target_obj_id = motion_plan.target_object_id
        total_collisions = 0
        unexpected_collisions = 0
        max_collision_force = 0.0

        for step_result in physics_results:
            for contact in step_result.contacts:
                body_a = contact.get("body_a", "")
                body_b = contact.get("body_b", "")
                impulse = contact.get("impulse", 0)

                # Check if collision is expected (gripper-target during grasp)
                is_expected = (
                    ("gripper" in body_a.lower() or "gripper" in body_b.lower()) and
                    (target_obj_id in body_a or target_obj_id in body_b)
                )

                total_collisions += 1
                if not is_expected:
                    unexpected_collisions += 1

                    # Record collision event
                    event = CollisionEvent(
                        frame_idx=step_result.step_index,
                        timestamp=step_result.simulation_time,
                        body_a=body_a,
                        body_b=body_b,
                        contact_point=np.array(contact.get("position", [0, 0, 0])),
                        contact_force=impulse,
                        is_expected=is_expected,
                    )
                    result.collision_events.append(event)

                max_collision_force = max(max_collision_force, impulse)

        result.metrics.total_collisions = total_collisions
        result.metrics.unexpected_collisions = unexpected_collisions
        result.metrics.max_collision_force = max_collision_force

        if unexpected_collisions > self.config.max_unexpected_collisions:
            result.failure_reasons.append(FailureReason.COLLISION)

        # P1-5 FIX: Track grasp stability metrics
        gripper_slip_count = 0
        object_dropped = False
        object_stable_at_end = True

        if physics_results and target_obj_id:
            # Track gripper-object contact throughout episode
            grasp_active = False
            grasp_start_frame = -1
            object_max_height = 0.0

            for step_result in physics_results:
                # Check if gripper is in contact with target object
                gripper_in_contact = False
                for contact in step_result.contacts:
                    body_a = contact.get("body_a", "")
                    body_b = contact.get("body_b", "")
                    if (("gripper" in body_a.lower() or "finger" in body_a.lower()) and target_obj_id in body_b) or \
                       (("gripper" in body_b.lower() or "finger" in body_b.lower()) and target_obj_id in body_a):
                        gripper_in_contact = True
                        break

                # Detect grasp start
                if gripper_in_contact and not grasp_active:
                    grasp_active = True
                    grasp_start_frame = step_result.step_index

                # Detect slip (loss of contact during grasp)
                elif not gripper_in_contact and grasp_active:
                    # Check if object is still elevated (not placed)
                    if target_obj_id in step_result.object_states:
                        obj_state = step_result.object_states[target_obj_id]
                        obj_height = obj_state.get("position", [0, 0, 0])[2]

                        # If object is elevated, this is a slip
                        if obj_height > 0.1:  # 10cm threshold
                            gripper_slip_count += 1
                            grasp_active = False

                # Track object height during episode
                if target_obj_id in step_result.object_states:
                    obj_state = step_result.object_states[target_obj_id]
                    obj_height = obj_state.get("position", [0, 0, 0])[2]
                    object_max_height = max(object_max_height, obj_height)

            # Check if object was dropped (fell after being lifted)
            final_states = physics_results[-1].object_states
            if target_obj_id in final_states:
                obj_state = final_states[target_obj_id]
                final_height = obj_state.get("position", [0, 0, 0])[2]
                final_velocity = np.linalg.norm(obj_state.get("linear_velocity", [0, 0, 0]))

                # Object dropped if it was lifted but ended up significantly lower
                if object_max_height > 0.15 and final_height < object_max_height * 0.5:
                    object_dropped = True

                # Object stable at end if velocity is low
                object_stable_at_end = final_velocity < self.config.stability_threshold

        # P1-5 FIX: Set grasp stability metrics
        result.metrics.gripper_slip_events = gripper_slip_count
        result.metrics.object_dropped = object_dropped
        result.metrics.object_stable_at_end = object_stable_at_end

        self.log(f"  Physics: {total_collisions} contacts, {unexpected_collisions} unexpected")
        self.log(f"  Grasp stability: {gripper_slip_count} slips, dropped={object_dropped}, stable={object_stable_at_end}")

    def _validate_heuristic(
        self,
        trajectory: JointTrajectory,
        motion_plan: MotionPlan,
        scene_objects: List[Dict[str, Any]],
        result: ValidationResult,
        task_success_checker: Optional[Callable],
    ) -> ValidationResult:
        """
        Validate using heuristic checks (fallback when PhysX unavailable).

        This uses geometric collision checking and kinematic analysis.
        Less accurate than real physics but useful for testing.
        """
        self.log("  Running heuristic validation...")

        # Run all heuristic checks
        self._check_joint_limits(trajectory, result)
        self._check_velocities(trajectory, result)
        self._check_torques(trajectory, result)  # P1-6 FIX: Check torque limits
        self._check_collisions(trajectory, motion_plan, scene_objects, result)
        self._check_trajectory_smoothness(trajectory, result)
        self._check_task_success(trajectory, motion_plan, scene_objects, result, task_success_checker)

        # P1-5 FIX: Heuristic grasp stability estimation
        # Without physics, we can't detect actual slips, but we can estimate based on gripper motion
        gripper_positions = trajectory.get_gripper_positions()
        if len(gripper_positions) > 1:
            # Check for rapid gripper opening during trajectory (possible slip)
            slip_count = 0
            for i in range(1, len(gripper_positions)):
                delta = abs(gripper_positions[i] - gripper_positions[i - 1])
                # If gripper opens rapidly (>0.3 in one step), might be a slip recovery
                if delta > 0.3 and gripper_positions[i] > gripper_positions[i - 1]:
                    slip_count += 1

            result.metrics.gripper_slip_events = slip_count

            # Heuristic: assume object stable if final gripper is closed
            result.metrics.object_stable_at_end = gripper_positions[-1] < 0.3

            # Heuristic: object dropped if gripper opens fully near end
            final_gripper = gripper_positions[-1]
            result.metrics.object_dropped = final_gripper > 0.9  # Nearly full open

        self.log(f"  Heuristic grasp stability: {result.metrics.gripper_slip_events} estimated slips")

        return result

    def validate_batch(
        self,
        episodes: List[Tuple[JointTrajectory, MotionPlan]],
        scene_objects: List[Dict[str, Any]],
    ) -> List[ValidationResult]:
        """Validate a batch of episodes."""
        results = []
        for trajectory, motion_plan in episodes:
            result = self.validate(trajectory, motion_plan, scene_objects)
            results.append(result)
        return results

    def filter_valid(
        self,
        results: List[ValidationResult],
        min_score: Optional[float] = None,
    ) -> List[ValidationResult]:
        """Filter to only valid episodes above score threshold."""
        threshold = min_score or self.config.min_quality_score
        return [
            r for r in results
            if r.status == ValidationStatus.PASSED and r.metrics.overall_score >= threshold
        ]

    def _check_joint_limits(
        self,
        trajectory: JointTrajectory,
        result: ValidationResult,
    ) -> None:
        """Check for joint limit violations."""

        lower = self.robot_config.joint_limits_lower
        upper = self.robot_config.joint_limits_upper
        joint_names = self.robot_config.joint_names

        for state in trajectory.states:
            positions = state.joint_positions

            for j in range(len(positions)):
                if positions[j] < lower[j]:
                    violation = lower[j] - positions[j]
                    event = JointLimitEvent(
                        frame_idx=state.frame_idx,
                        timestamp=state.timestamp,
                        joint_name=joint_names[j],
                        joint_value=positions[j],
                        limit_type="lower",
                        limit_value=lower[j],
                        violation_amount=violation,
                    )
                    result.joint_limit_events.append(event)
                    result.metrics.max_joint_violation = max(
                        result.metrics.max_joint_violation, violation
                    )

                elif positions[j] > upper[j]:
                    violation = positions[j] - upper[j]
                    event = JointLimitEvent(
                        frame_idx=state.frame_idx,
                        timestamp=state.timestamp,
                        joint_name=joint_names[j],
                        joint_value=positions[j],
                        limit_type="upper",
                        limit_value=upper[j],
                        violation_amount=violation,
                    )
                    result.joint_limit_events.append(event)
                    result.metrics.max_joint_violation = max(
                        result.metrics.max_joint_violation, violation
                    )

        result.metrics.joint_limit_violations = len(result.joint_limit_events)

        if result.metrics.joint_limit_violations > 0:
            result.failure_reasons.append(FailureReason.JOINT_LIMIT)

    def _check_velocities(
        self,
        trajectory: JointTrajectory,
        result: ValidationResult,
    ) -> None:
        """Check for excessive velocities and accelerations."""

        if len(trajectory.states) < 2:
            return

        dt = 1.0 / trajectory.fps
        prev_vel = None

        for i in range(1, len(trajectory.states)):
            curr_pos = trajectory.states[i].joint_positions
            prev_pos = trajectory.states[i - 1].joint_positions

            # Velocity
            velocity = np.abs(curr_pos - prev_pos) / dt
            max_vel = np.max(velocity)
            result.metrics.max_joint_velocity = max(
                result.metrics.max_joint_velocity, max_vel
            )

            if max_vel > self.config.max_joint_velocity:
                result.failure_reasons.append(FailureReason.EXCESSIVE_VELOCITY)

            # Acceleration
            if prev_vel is not None:
                accel = np.abs(velocity - prev_vel) / dt
                max_accel = np.max(accel)
                result.metrics.max_joint_acceleration = max(
                    result.metrics.max_joint_acceleration, max_accel
                )

            prev_vel = velocity

    def _check_torques(
        self,
        trajectory: JointTrajectory,
        result: ValidationResult,
    ) -> None:
        """
        P1-6 FIX: Check for torque limit violations using inverse dynamics.

        Estimates required joint torques using simplified dynamics model:
        τ = I·α + C·v + G
        where I = inertia, α = acceleration, C = damping, v = velocity, G = gravity
        """
        if len(trajectory.states) < 3:
            return  # Need at least 3 states for acceleration

        dt = 1.0 / trajectory.fps

        # Simplified robot parameters (conservative estimates for 7-DOF arm)
        # These are approximate - real values depend on robot model
        joint_inertia = np.array([2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5])  # kg·m²
        damping_coeff = np.array([10.0, 10.0, 8.0, 8.0, 5.0, 5.0, 3.0])  # N·m·s/rad
        gravity_torque = np.array([15.0, 15.0, 10.0, 10.0, 5.0, 5.0, 2.0])  # N·m (config-dependent)

        # Maximum torque limits (conservative for safety)
        max_torque = np.array([80.0, 80.0, 60.0, 60.0, 40.0, 40.0, 20.0])  # N·m

        # Pad arrays if robot has fewer DOFs
        num_dof = len(trajectory.states[0].joint_positions)
        if num_dof < len(joint_inertia):
            joint_inertia = joint_inertia[:num_dof]
            damping_coeff = damping_coeff[:num_dof]
            gravity_torque = gravity_torque[:num_dof]
            max_torque = max_torque[:num_dof]
        elif num_dof > len(joint_inertia):
            # Extend with defaults for additional joints
            joint_inertia = np.pad(joint_inertia, (0, num_dof - len(joint_inertia)), constant_values=1.0)
            damping_coeff = np.pad(damping_coeff, (0, num_dof - len(damping_coeff)), constant_values=5.0)
            gravity_torque = np.pad(gravity_torque, (0, num_dof - len(gravity_torque)), constant_values=5.0)
            max_torque = np.pad(max_torque, (0, num_dof - len(max_torque)), constant_values=40.0)

        torque_violations = 0

        for i in range(2, len(trajectory.states)):
            prev_prev_pos = trajectory.states[i - 2].joint_positions
            prev_pos = trajectory.states[i - 1].joint_positions
            curr_pos = trajectory.states[i].joint_positions

            # Compute velocity and acceleration
            vel_prev = (prev_pos - prev_prev_pos) / dt
            vel_curr = (curr_pos - prev_pos) / dt
            accel = (vel_curr - vel_prev) / dt

            # Estimate torques: τ = I·α + C·v + G
            inertial_torque = joint_inertia * accel
            damping_torque = damping_coeff * vel_curr
            # Gravity torque depends on configuration - use conservative estimate
            total_torque = np.abs(inertial_torque) + np.abs(damping_torque) + np.abs(gravity_torque)

            # Check for violations
            for j, torque in enumerate(total_torque):
                if torque > max_torque[j]:
                    torque_violations += 1
                    if torque_violations == 1:  # Log first violation
                        self.log(
                            f"  Torque limit violation at frame {i}, joint {j}: "
                            f"{torque:.1f} N·m > {max_torque[j]:.1f} N·m limit",
                            "WARNING"
                        )

        # P1-6 FIX: Store torque violations in metrics
        result.metrics.torque_limit_violations = torque_violations
        if torque_violations > 0:
            self.log(f"  Total torque limit violations: {torque_violations}", "WARNING")

    def _check_collisions(
        self,
        trajectory: JointTrajectory,
        motion_plan: MotionPlan,
        scene_objects: List[Dict[str, Any]],
        result: ValidationResult,
    ) -> None:
        """Check for collisions with scene objects."""

        # Build obstacle list (excluding target object during manipulation)
        target_obj_id = motion_plan.target_object_id
        obstacles = [
            obj for obj in scene_objects
            if obj.get("id") != target_obj_id and obj.get("sim_role") != "background"
        ]

        for state in trajectory.states:
            ee_pos = state.ee_position
            if ee_pos is None:
                continue

            for obs in obstacles:
                obs_pos = np.array(obs.get("position", [0, 0, 0]))
                obs_dims = np.array(obs.get("dimensions", [0.1, 0.1, 0.1]))

                # Simple AABB collision check
                half_dims = obs_dims / 2 + 0.02  # Small padding
                if np.all(np.abs(ee_pos - obs_pos) < half_dims):
                    event = CollisionEvent(
                        frame_idx=state.frame_idx,
                        timestamp=state.timestamp,
                        body_a="gripper",
                        body_b=obs.get("id", "obstacle"),
                        contact_point=ee_pos.copy(),
                        contact_force=10.0,  # Estimated
                        is_expected=False,
                    )
                    result.collision_events.append(event)

        # Count unexpected collisions
        result.metrics.total_collisions = len(result.collision_events)
        result.metrics.unexpected_collisions = sum(
            1 for e in result.collision_events if not e.is_expected
        )

        if result.metrics.unexpected_collisions > self.config.max_unexpected_collisions:
            result.failure_reasons.append(FailureReason.COLLISION)

    def _check_trajectory_smoothness(
        self,
        trajectory: JointTrajectory,
        result: ValidationResult,
    ) -> None:
        """Compute trajectory smoothness metrics."""

        if len(trajectory.states) < 3:
            result.metrics.velocity_smoothness = 1.0
            return

        dt = 1.0 / trajectory.fps
        positions = np.array([s.joint_positions for s in trajectory.states])

        # Path length (sum of position changes)
        path_length = 0.0
        for i in range(1, len(positions)):
            path_length += np.linalg.norm(positions[i] - positions[i - 1])
        result.metrics.path_length = path_length

        # Jerk integral (smoothness)
        if len(positions) >= 4:
            velocities = np.diff(positions, axis=0) / dt
            accelerations = np.diff(velocities, axis=0) / dt
            jerks = np.diff(accelerations, axis=0) / dt

            jerk_integral = np.sum(np.abs(jerks))
            result.metrics.jerk_integral = jerk_integral

            # Normalize to 0-1 smoothness score
            # Lower jerk = higher smoothness
            max_expected_jerk = 1000.0  # Empirical threshold
            smoothness = 1.0 - min(1.0, jerk_integral / max_expected_jerk)
            result.metrics.velocity_smoothness = smoothness

    def _check_task_success(
        self,
        trajectory: JointTrajectory,
        motion_plan: MotionPlan,
        scene_objects: List[Dict[str, Any]],
        result: ValidationResult,
        task_success_checker: Optional[callable],
    ) -> None:
        """Check task success criteria."""

        # Use custom checker if provided
        if task_success_checker:
            result.metrics.task_success = task_success_checker(trajectory, motion_plan)
        else:
            # Default: check if trajectory completed without major issues
            result.metrics.task_success = (
                len(result.failure_reasons) == 0 and
                len(trajectory.states) >= 10
            )

        # Check grasp success (gripper closed during grasp phase)
        grasp_states = [
            s for s in trajectory.states
            if s.phase.value in ["grasp", "lift", "transport", "place"]
        ]
        if grasp_states:
            # Gripper should be closed (low position) during grasp
            avg_gripper = np.mean([s.gripper_position for s in grasp_states])
            result.metrics.grasp_success = avg_gripper < 0.02  # Less than 2cm = closed

        # Check placement success (object stable at end)
        if motion_plan.place_position is not None:
            # Final EE should be near place position then retracted
            final_state = trajectory.states[-1]
            if final_state.ee_position is not None:
                # Gripper should be open at end (released object)
                result.metrics.placement_success = final_state.gripper_position > 0.02

        # Update object stability
        result.metrics.object_stable_at_end = (
            result.metrics.placement_success or
            not motion_plan.place_position
        )
        result.metrics.object_dropped = not result.metrics.grasp_success

        # Set timing
        result.metrics.total_duration = trajectory.total_duration
        result.metrics.execution_time = trajectory.total_duration

        # Check if task requirements met
        if self.config.require_task_success and not result.metrics.task_success:
            result.failure_reasons.append(FailureReason.TASK_FAILURE)
        if self.config.require_grasp_success and not result.metrics.grasp_success:
            result.failure_reasons.append(FailureReason.GRASP_FAILURE)
        if self.config.require_placement_success and motion_plan.place_position and not result.metrics.placement_success:
            result.failure_reasons.append(FailureReason.PLACEMENT_FAILURE)

    def _determine_status(self, result: ValidationResult) -> ValidationStatus:
        """Determine final validation status."""

        if len(result.failure_reasons) == 0:
            if result.metrics.overall_score >= self.config.min_quality_score:
                return ValidationStatus.PASSED
            else:
                return ValidationStatus.NEEDS_RETRY

        # Check if failures are retryable
        retryable_failures = {
            FailureReason.COLLISION,
            FailureReason.JOINT_LIMIT,
            FailureReason.IK_FAILURE,
            FailureReason.PLANNING_FAILURE,
        }

        if all(r in retryable_failures for r in result.failure_reasons):
            return ValidationStatus.NEEDS_RETRY

        return ValidationStatus.FAILED

    def _determine_retry_info(self, result: ValidationResult) -> None:
        """Determine if and how to retry."""

        result.can_retry = (
            result.status == ValidationStatus.NEEDS_RETRY and
            result.retry_count < self.config.max_retries
        )

        if not result.can_retry:
            return

        suggestions = []

        if FailureReason.COLLISION in result.failure_reasons:
            suggestions.append("Replan with collision avoidance")
            result.can_retry = self.config.retry_on_collision

        if FailureReason.JOINT_LIMIT in result.failure_reasons:
            suggestions.append("Use different IK solution")
            result.can_retry = self.config.retry_on_joint_limit

        if FailureReason.EXCESSIVE_VELOCITY in result.failure_reasons:
            suggestions.append("Reduce motion speed")

        if FailureReason.GRASP_FAILURE in result.failure_reasons:
            suggestions.append("Adjust grasp position/timing")

        if FailureReason.PLANNING_FAILURE in result.failure_reasons:
            suggestions.append("Retry collision-checked motion planning")

        result.retry_suggestion = "; ".join(suggestions)


# =============================================================================
# Validation Report Generator
# =============================================================================


class ValidationReportGenerator:
    """Generates validation reports for episode batches."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def generate_report(
        self,
        results: List[ValidationResult],
        scene_id: str,
    ) -> Path:
        """Generate a validation report."""

        report = {
            "scene_id": scene_id,
            "summary": self._compute_summary(results),
            "quality_distribution": self._compute_quality_distribution(results),
            "failure_analysis": self._analyze_failures(results),
            "episodes": [r.to_dict() for r in results],
        }

        # Write report
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "validation_report.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report_path

    def _compute_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Compute summary statistics."""

        total = len(results)
        passed = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        needs_retry = sum(1 for r in results if r.status == ValidationStatus.NEEDS_RETRY)

        scores = [r.metrics.overall_score for r in results]

        return {
            "total_episodes": total,
            "passed": passed,
            "failed": failed,
            "needs_retry": needs_retry,
            "pass_rate": passed / total if total > 0 else 0,
            "average_score": np.mean(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "score_std": np.std(scores) if scores else 0,
        }

    def _compute_quality_distribution(
        self,
        results: List[ValidationResult],
    ) -> Dict[str, int]:
        """Compute quality score distribution."""

        buckets = {
            "excellent_0.9_1.0": 0,
            "good_0.7_0.9": 0,
            "acceptable_0.5_0.7": 0,
            "poor_0.3_0.5": 0,
            "failed_0.0_0.3": 0,
        }

        for r in results:
            score = r.metrics.overall_score
            if score >= 0.9:
                buckets["excellent_0.9_1.0"] += 1
            elif score >= 0.7:
                buckets["good_0.7_0.9"] += 1
            elif score >= 0.5:
                buckets["acceptable_0.5_0.7"] += 1
            elif score >= 0.3:
                buckets["poor_0.3_0.5"] += 1
            else:
                buckets["failed_0.0_0.3"] += 1

        return buckets

    def _analyze_failures(
        self,
        results: List[ValidationResult],
    ) -> Dict[str, Any]:
        """Analyze failure patterns."""

        failure_counts = {}
        for r in results:
            for reason in r.failure_reasons:
                failure_counts[reason.value] = failure_counts.get(reason.value, 0) + 1

        return {
            "failure_counts": failure_counts,
            "most_common_failure": (
                max(failure_counts, key=failure_counts.get)
                if failure_counts else None
            ),
            "total_collision_events": sum(
                len(r.collision_events) for r in results
            ),
            "total_joint_limit_events": sum(
                len(r.joint_limit_events) for r in results
            ),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_episode(
    trajectory: JointTrajectory,
    motion_plan: MotionPlan,
    scene_objects: List[Dict[str, Any]],
    robot_type: str = "franka",
) -> ValidationResult:
    """Convenience function to validate a single episode."""
    validator = SimulationValidator(robot_type=robot_type, verbose=False)
    return validator.validate(trajectory, motion_plan, scene_objects)


def filter_valid_episodes(
    episodes: List[Tuple[JointTrajectory, MotionPlan]],
    scene_objects: List[Dict[str, Any]],
    min_score: float = 0.7,
    robot_type: str = "franka",
) -> List[Tuple[JointTrajectory, MotionPlan, ValidationResult]]:
    """Filter episodes to only those passing validation."""

    validator = SimulationValidator(robot_type=robot_type, verbose=False)
    valid = []

    for trajectory, motion_plan in episodes:
        result = validator.validate(trajectory, motion_plan, scene_objects)
        if result.status == ValidationStatus.PASSED and result.metrics.overall_score >= min_score:
            valid.append((trajectory, motion_plan, result))

    return valid


if __name__ == "__main__":
    from motion_planner import AIMotionPlanner
    from trajectory_solver import TrajectorySolver

    print("Testing Simulation Validator")
    print("=" * 60)

    # Create test episode
    planner = AIMotionPlanner(robot_type="franka", use_llm=False, verbose=False)
    solver = TrajectorySolver(robot_type="franka", fps=30.0, verbose=False)

    motion_plan = planner.plan_motion(
        task_name="pick_cup",
        task_description="Pick up cup",
        target_object={
            "id": "cup_001",
            "position": [0.5, 0.1, 0.85],
            "dimensions": [0.08, 0.08, 0.12],
        },
        place_position=[0.3, 0.2, 0.9],
    )

    trajectory = solver.solve(motion_plan)

    # Validate
    scene_objects = [
        {
            "id": "cup_001",
            "position": [0.5, 0.1, 0.85],
            "dimensions": [0.08, 0.08, 0.12],
        },
        {
            "id": "table",
            "position": [0.4, 0, 0.4],
            "dimensions": [0.8, 0.6, 0.02],
            "sim_role": "background",
        },
    ]

    validator = SimulationValidator(robot_type="franka", verbose=True)
    result = validator.validate(trajectory, motion_plan, scene_objects)

    print("\n" + "=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)
    print(f"Status: {result.status.value}")
    print(f"Overall Score: {result.metrics.overall_score:.2f}")
    print(f"Task Success: {result.metrics.task_success}")
    print(f"Grasp Success: {result.metrics.grasp_success}")
    print(f"Collisions: {result.metrics.total_collisions}")
    print(f"Joint Violations: {result.metrics.joint_limit_violations}")
    print(f"Velocity Smoothness: {result.metrics.velocity_smoothness:.2f}")
    if result.failure_reasons:
        print(f"Failure Reasons: {[r.value for r in result.failure_reasons]}")
