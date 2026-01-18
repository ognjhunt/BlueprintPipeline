#!/usr/bin/env python3
"""
Isaac Sim Integration Module for Episode Generation.

This module provides unified access to Isaac Sim/Isaac Lab features:
- Physics simulation via PhysX
- Sensor data capture via Replicator
- Motion planning via cuRobo (when available)
- Scene management via USD

DEPLOYMENT ARCHITECTURE:
========================
PRODUCTION (Cloud Run / Docker):
  - Uses Dockerfile.isaacsim which extends NVIDIA Isaac Sim container
  - Isaac Sim IS available via omni.isaac.core imports
  - GPU acceleration enabled (L4/A100 GPUs)
  - docker-compose.isaacsim.yaml sets USE_MOCK_CAPTURE="false"

LOCAL DEVELOPMENT (without GPU/Isaac Sim):
  - Falls back to mock implementations for testing
  - Set USE_MOCK_CAPTURE="true" or run without Isaac Sim Python
  - Useful for CI/CD, unit tests, and development iteration

The production path is Isaac Sim. Mock mode is ONLY for development.

Usage:
    from isaac_sim_integration import (
        IsaacSimSession,
        PhysicsSimulator,
        get_isaac_sim_session,
        is_isaac_sim_available,
    )

    # Check availability
    if is_isaac_sim_available():
        session = get_isaac_sim_session()
        session.load_scene("/path/to/scene.usda")
        physics = session.get_physics_simulator()
        result = physics.step()
"""

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.logging_config import init_logging

# Configure logging
init_logging()
logger = logging.getLogger(__name__)

# Import timeout utilities
try:
    from tools.error_handling.timeout import timeout, TimeoutError as CustomTimeoutError, TimeoutManager
    HAVE_TIMEOUT_TOOLS = True
except ImportError:
    HAVE_TIMEOUT_TOOLS = False
    CustomTimeoutError = TimeoutError
    logger.warning("Timeout tools not available - Isaac Sim operations may hang")

# =============================================================================
# Isaac Sim Availability Detection
# =============================================================================

# Global flags for available features
_ISAAC_SIM_AVAILABLE = False
_OMNI_AVAILABLE = False
_PHYSX_AVAILABLE = False
_REPLICATOR_AVAILABLE = False
_CUROBO_AVAILABLE = False
_ISAAC_LAB_AVAILABLE = False

# Isaac Sim module references (set on successful import)
_omni = None
_rep = None
_physx = None
_usd_core = None
_curobo = None

def _check_isaac_sim_environment() -> Dict[str, bool]:
    """
    Check what Isaac Sim features are available.

    Returns:
        Dict with availability status for each feature
    """
    global _ISAAC_SIM_AVAILABLE, _OMNI_AVAILABLE, _PHYSX_AVAILABLE
    global _REPLICATOR_AVAILABLE, _CUROBO_AVAILABLE, _ISAAC_LAB_AVAILABLE
    global _omni, _rep, _physx, _usd_core, _curobo

    status = {
        "isaac_sim": False,
        "omniverse": False,
        "physx": False,
        "replicator": False,
        "curobo": False,
        "isaac_lab": False,
        "usd_core": False,
    }

    # Check for omni (indicates we're in Isaac Sim)
    try:
        import omni
        _omni = omni
        status["omniverse"] = True
        _OMNI_AVAILABLE = True

        # If omni is available, we're likely in Isaac Sim
        try:
            import omni.isaac.core
            status["isaac_sim"] = True
            _ISAAC_SIM_AVAILABLE = True
        except ImportError:
            pass

    except ImportError:
        pass

    # Check for PhysX
    try:
        import omni.physx
        _physx = omni.physx
        status["physx"] = True
        _PHYSX_AVAILABLE = True
    except ImportError:
        pass

    # Check for Replicator
    try:
        import omni.replicator.core as rep
        _rep = rep
        status["replicator"] = True
        _REPLICATOR_AVAILABLE = True
    except ImportError:
        pass

    # Check for cuRobo
    try:
        import curobo
        _curobo = curobo
        status["curobo"] = True
        _CUROBO_AVAILABLE = True
    except ImportError:
        pass

    # Check for Isaac Lab
    try:
        import omni.isaac.lab
        status["isaac_lab"] = True
        _ISAAC_LAB_AVAILABLE = True
    except ImportError:
        pass

    # Check for USD core (can work outside Isaac Sim)
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
        _usd_core = {"Usd": Usd, "UsdGeom": UsdGeom, "UsdPhysics": UsdPhysics}
        status["usd_core"] = True
    except ImportError:
        pass

    return status


# Run availability check on module import
_AVAILABILITY_STATUS = _check_isaac_sim_environment()


def is_isaac_sim_available() -> bool:
    """Check if running inside Isaac Sim environment."""
    return _ISAAC_SIM_AVAILABLE


def is_physx_available() -> bool:
    """Check if PhysX is available for physics simulation."""
    return _PHYSX_AVAILABLE


def is_replicator_available() -> bool:
    """Check if Replicator is available for sensor capture."""
    return _REPLICATOR_AVAILABLE


def is_curobo_available() -> bool:
    """Check if cuRobo is available for motion planning."""
    return _CUROBO_AVAILABLE


def is_isaac_lab_available() -> bool:
    """Check if Isaac Lab is available."""
    return _ISAAC_LAB_AVAILABLE


def get_availability_status() -> Dict[str, bool]:
    """Get detailed availability status."""
    return _AVAILABILITY_STATUS.copy()


def print_availability_report() -> None:
    """Print a human-readable availability report."""
    logger.info("%s", "=" * 60)
    logger.info("ISAAC SIM INTEGRATION - AVAILABILITY REPORT")
    logger.info("%s", "=" * 60)

    for feature, available in _AVAILABILITY_STATUS.items():
        status = "✅ AVAILABLE" if available else "❌ NOT AVAILABLE"
        logger.info("  %-20s: %s", feature, status)

    if not _ISAAC_SIM_AVAILABLE:
        logger.warning("⚠️  WARNING: Not running inside Isaac Sim!")
        logger.warning("   Features requiring Isaac Sim will use mock implementations.")
        logger.warning("   To run with full features, use: /isaac-sim/python.sh your_script.py")

    logger.info("%s", "=" * 60)


# =============================================================================
# Physics Simulation
# =============================================================================


class SimulationState(Enum):
    """State of the physics simulation."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class PhysicsStepResult:
    """Result of a single physics step."""
    step_index: int
    simulation_time: float
    dt: float

    # Collision data
    contacts: List[Dict[str, Any]] = field(default_factory=list)
    collision_count: int = 0

    # Object states
    object_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Robot state
    robot_joint_positions: Optional[np.ndarray] = None
    robot_joint_velocities: Optional[np.ndarray] = None
    robot_joint_efforts: Optional[np.ndarray] = None
    robot_ee_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # Success/error
    success: bool = True
    error_message: str = ""


@dataclass
class ContactInfo:
    """Information about a physics contact."""
    body_a: str
    body_b: str
    position: np.ndarray
    normal: np.ndarray
    impulse: float
    separation: float
    is_expected: bool = False  # For grasping contacts


class PhysicsSimulator:
    """
    Physics simulator using Isaac Sim's PhysX backend.

    Provides:
    - Scene loading and management
    - Physics stepping with configurable dt
    - Contact/collision detection
    - Object state tracking
    - Robot state tracking
    """

    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        substeps: int = 4,
        verbose: bool = True,
        physics_timeout: float = 10.0,
    ):
        """
        Initialize physics simulator.

        Args:
            dt: Physics timestep (default 1/60s = 60 Hz)
            substeps: Number of physics substeps per step
            verbose: Print debug info
            physics_timeout: Timeout for physics operations in seconds (default: 10.0)
        """
        self.dt = dt
        self.substeps = substeps
        self.verbose = verbose
        self.physics_timeout = physics_timeout

        self._simulation_time = 0.0
        self._step_count = 0
        self._state = SimulationState.STOPPED

        # Scene data
        self._scene_path: Optional[str] = None
        self._stage = None
        self._physics_context = None

        # Tracked objects
        self._tracked_objects: Dict[str, str] = {}  # id -> prim_path
        self._robot_prim_path: Optional[str] = None

        # Contact callbacks
        self._contact_callbacks: List[Callable] = []
        self._contacts_this_step: List[ContactInfo] = []

        # Check availability
        self._use_real_physics = _PHYSX_AVAILABLE and (_ISAAC_SIM_AVAILABLE or _ISAAC_LAB_AVAILABLE)

        if not self._use_real_physics:
            self.log("PhysX not available - using mock physics simulation", "WARNING")

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[PHYSICS-SIM] [%s] %s", level, msg)

    def load_scene(self, scene_path: str) -> bool:
        """
        Load a USD scene for physics simulation.

        Args:
            scene_path: Path to scene.usda file

        Returns:
            True if scene loaded successfully
        """
        self._scene_path = scene_path

        if not self._use_real_physics:
            self.log(f"Mock: Would load scene from {scene_path}")
            self._state = SimulationState.STOPPED
            return True

        stage = None
        try:
            import omni.usd
            from omni.isaac.core import World

            # Create or get world
            world = World.instance()
            if world is None:
                world = World(stage_units_in_meters=1.0)

            # Load scene
            omni.usd.get_context().open_stage(scene_path)
            stage = omni.usd.get_context().get_stage()
            self._stage = stage

            # Get physics context
            from omni.physx import get_physx_interface
            self._physics_context = get_physx_interface()

            # Register contact callback
            self._setup_contact_reporting()

            self._state = SimulationState.STOPPED
            self.log(f"Loaded scene: {scene_path}")
            return True

        except Exception as e:
            self.log(f"Failed to load scene: {e}", "ERROR")
            logger.error(f"Scene loading failed: {e}", exc_info=True)

            # Clean up resources on failure
            if stage is not None:
                try:
                    import omni.usd
                    omni.usd.get_context().close_stage()
                except Exception as cleanup_error:
                    self.log(f"Failed to cleanup stage after error: {cleanup_error}", "WARNING")

            self._stage = None
            self._physics_context = None
            self._state = SimulationState.ERROR
            return False

    def _setup_contact_reporting(self) -> None:
        """Set up PhysX contact reporting."""
        if not self._use_real_physics or self._physics_context is None:
            return

        try:
            # Enable contact reporting for all rigid bodies
            from pxr import UsdPhysics

            for prim in self._stage.Traverse():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    # Enable contact reporting
                    rigid_body = UsdPhysics.RigidBodyAPI(prim)
                    # Contact reporting is enabled by default in Isaac Sim

        except Exception as e:
            self.log(f"Contact reporting setup failed: {e}", "WARNING")

    def add_tracked_object(self, object_id: str, prim_path: str) -> None:
        """Add an object to track during simulation."""
        self._tracked_objects[object_id] = prim_path

    def set_robot(self, prim_path: str) -> None:
        """Set the robot to track."""
        self._robot_prim_path = prim_path

    def start(self) -> bool:
        """Start the physics simulation."""
        if self._state == SimulationState.RUNNING:
            return True

        self._simulation_time = 0.0
        self._step_count = 0

        if self._use_real_physics:
            try:
                from omni.isaac.core import World
                world = World.instance()
                if world:
                    world.reset()
                    world.play()
            except Exception as e:
                self.log(f"Failed to start simulation: {e}", "ERROR")
                return False

        self._state = SimulationState.RUNNING
        self.log("Simulation started")
        return True

    def stop(self) -> None:
        """Stop the physics simulation."""
        if self._use_real_physics:
            try:
                from omni.isaac.core import World
                world = World.instance()
                if world:
                    world.stop()
            except Exception as e:
                self.log(f"Failed to stop physics simulation: {e}", "WARNING")
                logger.warning(f"Physics simulation stop failed: {e}", exc_info=True)

        self._state = SimulationState.STOPPED
        self.log("Simulation stopped")

    def pause(self) -> None:
        """Pause the physics simulation."""
        if self._use_real_physics:
            try:
                from omni.isaac.core import World
                world = World.instance()
                if world:
                    world.pause()
            except Exception as e:
                self.log(f"Failed to pause physics simulation: {e}", "WARNING")
                logger.warning(f"Physics simulation pause failed: {e}", exc_info=True)

        self._state = SimulationState.PAUSED

    def step(self) -> PhysicsStepResult:
        """
        Perform one physics step.

        Returns:
            PhysicsStepResult with step data
        """
        self._contacts_this_step = []

        result = PhysicsStepResult(
            step_index=self._step_count,
            simulation_time=self._simulation_time,
            dt=self.dt,
        )

        if self._use_real_physics:
            result = self._step_real_physics(result)
        else:
            result = self._step_mock_physics(result)

        self._simulation_time += self.dt
        self._step_count += 1

        return result

    def _step_real_physics(self, result: PhysicsStepResult) -> PhysicsStepResult:
        """
        Perform a real physics step using PhysX.

        GAP-EH-005 FIX: Add timeout to prevent hung Isaac Sim operations.
        """
        try:
            from omni.isaac.core import World

            world = World.instance()
            if world is None:
                result.success = False
                result.error_message = "World not initialized"
                return result

            # GAP-EH-005 FIX: Step physics with configurable timeout
            if HAVE_TIMEOUT_TOOLS:
                try:
                    with timeout(self.physics_timeout, f"Physics step timed out after {self.physics_timeout}s"):
                        world.step(render=False)
                except CustomTimeoutError as e:
                    result.success = False
                    result.error_message = f"Physics step timeout: {e}"
                    logger.error(f"Physics step timed out: {e}")
                    return result
            else:
                # No timeout protection available - fall back to direct call
                logger.warning("Timeout tools not available - physics step may hang")
                world.step(render=False)

            # Get contacts
            result.contacts = self._get_contacts()
            result.collision_count = len(result.contacts)

            # Get object states
            result.object_states = self._get_object_states()

            # Get robot state
            if self._robot_prim_path:
                robot_state = self._get_robot_state()
                result.robot_joint_positions = robot_state.get("joint_positions")
                result.robot_joint_velocities = robot_state.get("joint_velocities")
                result.robot_joint_efforts = robot_state.get("joint_efforts")
                result.robot_ee_pose = robot_state.get("ee_pose")

            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.log(f"Physics step failed: {e}", "ERROR")

        return result

    def _step_mock_physics(self, result: PhysicsStepResult) -> PhysicsStepResult:
        """Perform a mock physics step (for testing outside Isaac Sim)."""
        # Generate mock object states
        for obj_id in self._tracked_objects:
            result.object_states[obj_id] = {
                "position": [0.5, 0.0, 0.85],
                "orientation": [1.0, 0.0, 0.0, 0.0],
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            }

        # No collisions in mock mode (unless explicitly set)
        result.contacts = []
        result.collision_count = 0
        result.success = True

        return result

    def _get_contacts(self) -> List[Dict[str, Any]]:
        """Get current contact information from PhysX."""
        contacts = []

        if not self._use_real_physics:
            return contacts

        try:
            from omni.physx import get_physx_interface
            physx = get_physx_interface()

            # Get contact report
            contact_data = physx.get_contact_report()

            for contact in contact_data:
                contacts.append({
                    "body_a": contact.get("actor0", "unknown"),
                    "body_b": contact.get("actor1", "unknown"),
                    "position": list(contact.get("position", [0, 0, 0])),
                    "normal": list(contact.get("normal", [0, 0, 1])),
                    "impulse": float(contact.get("impulse", 0)),
                    "separation": float(contact.get("separation", 0)),
                })

        except Exception as e:
            self.log(f"Failed to get contacts: {e}", "WARNING")

        return contacts

    def _get_object_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current state of tracked objects."""
        states = {}

        if not self._use_real_physics:
            return states

        try:
            from pxr import UsdGeom
            import omni.isaac.core.utils.stage as stage_utils

            for obj_id, prim_path in self._tracked_objects.items():
                try:
                    prim = self._stage.GetPrimAtPath(prim_path)
                    if not prim.IsValid():
                        continue

                    xformable = UsdGeom.Xformable(prim)
                    world_transform = xformable.ComputeLocalToWorldTransform(0)

                    # Extract position and rotation
                    translation = world_transform.ExtractTranslation()
                    rotation = world_transform.ExtractRotationQuat()

                    states[obj_id] = {
                        "position": [float(translation[0]), float(translation[1]), float(translation[2])],
                        "orientation": [float(rotation.GetReal()),
                                       float(rotation.GetImaginary()[0]),
                                       float(rotation.GetImaginary()[1]),
                                       float(rotation.GetImaginary()[2])],
                        "linear_velocity": [0, 0, 0],  # Would need ArticulationView for velocities
                        "angular_velocity": [0, 0, 0],
                    }
                except Exception as e:
                    self.log(f"Failed to get state for object {obj_name}: {e}", "DEBUG")
                    # Continue to next object instead of failing entire operation
                    continue

        except Exception as e:
            self.log(f"Failed to get object states: {e}", "WARNING")

        return states

    def _get_robot_state(self) -> Dict[str, Any]:
        """
        Get current robot state from Isaac Sim.

        Returns:
            Dict with joint_positions, joint_velocities, ee_pose, gripper_state
        """
        state = {}

        if not self._use_real_physics or not self._robot_prim_path:
            return state

        try:
            from omni.isaac.core.articulations import Articulation
            from omni.isaac.core.utils.stage import get_current_stage
            from pxr import UsdGeom

            stage = get_current_stage()
            if stage is None:
                self.log("No stage available", "WARNING")
                return state

            # Get robot articulation
            robot_prim = stage.GetPrimAtPath(self._robot_prim_path)
            if not robot_prim.IsValid():
                self.log(f"Robot prim not found at {self._robot_prim_path}", "WARNING")
                return state

            # Create articulation wrapper
            robot = Articulation(self._robot_prim_path)
            if not robot.initialized:
                robot.initialize()

            # Get joint positions and velocities
            joint_positions = robot.get_joint_positions()
            joint_velocities = robot.get_joint_velocities()
            joint_efforts = None

            if hasattr(robot, "get_joint_efforts"):
                joint_efforts = robot.get_joint_efforts()
            elif hasattr(robot, "get_measured_joint_efforts"):
                joint_efforts = robot.get_measured_joint_efforts()
            elif hasattr(robot, "get_applied_joint_efforts"):
                joint_efforts = robot.get_applied_joint_efforts()

            # Handle None cases
            if joint_positions is None:
                joint_positions = np.zeros(robot.num_dof if hasattr(robot, 'num_dof') else 7)
            if joint_velocities is None:
                joint_velocities = np.zeros_like(joint_positions)

            state["joint_positions"] = np.array(joint_positions, dtype=np.float64)
            state["joint_velocities"] = np.array(joint_velocities, dtype=np.float64)
            if joint_efforts is None:
                state["joint_efforts"] = np.zeros_like(state["joint_positions"])
            else:
                state["joint_efforts"] = np.array(joint_efforts, dtype=np.float64)

            # Get end-effector pose
            # Try common EE link names
            ee_link_names = [
                f"{self._robot_prim_path}/panda_hand",
                f"{self._robot_prim_path}/ee_link",
                f"{self._robot_prim_path}/tool0",
                f"{self._robot_prim_path}/gripper_link",
            ]

            ee_pose = None
            for ee_path in ee_link_names:
                ee_prim = stage.GetPrimAtPath(ee_path)
                if ee_prim.IsValid():
                    xformable = UsdGeom.Xformable(ee_prim)
                    world_transform = xformable.ComputeLocalToWorldTransform(0)
                    translation = world_transform.ExtractTranslation()
                    rotation = world_transform.ExtractRotationQuat()

                    ee_position = np.array([
                        float(translation[0]),
                        float(translation[1]),
                        float(translation[2])
                    ])
                    ee_orientation = np.array([
                        float(rotation.GetReal()),
                        float(rotation.GetImaginary()[0]),
                        float(rotation.GetImaginary()[1]),
                        float(rotation.GetImaginary()[2])
                    ])
                    ee_pose = (ee_position, ee_orientation)
                    break

            if ee_pose is None:
                # Fallback: use forward kinematics estimate
                ee_pose = (np.array([0.5, 0.0, 0.5]), np.array([1.0, 0.0, 0.0, 0.0]))
                self.log("Could not find EE link, using FK estimate", "WARNING")

            state["ee_pose"] = ee_pose

            # Get gripper state if available
            try:
                gripper_joints = robot.get_joint_positions()
                if gripper_joints is not None and len(gripper_joints) > 7:
                    # Assume last joints are gripper
                    state["gripper_state"] = float(gripper_joints[-1])
                else:
                    state["gripper_state"] = 0.04  # Default open position
            except Exception as e:
                self.log(f"Failed to extract gripper state: {e}", "WARNING")
                state["gripper_state"] = 0.04  # Default open position on error

        except Exception as e:
            self.log(f"Failed to get robot state: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "DEBUG")

        return state

    def apply_robot_command(
        self,
        joint_positions: Optional[np.ndarray] = None,
        joint_velocities: Optional[np.ndarray] = None,
        gripper_command: Optional[float] = None,
    ) -> bool:
        """
        Apply a command to the robot.

        Args:
            joint_positions: Target joint positions (radians)
            joint_velocities: Target joint velocities (rad/s)
            gripper_command: Gripper aperture (0=closed, 1=open)

        Returns:
            True if command applied successfully
        """
        if not self._use_real_physics:
            return True

        if not self._robot_prim_path:
            self.log("No robot path set", "WARNING")
            return False

        try:
            from omni.isaac.core.articulations import Articulation
            from omni.isaac.core.utils.types import ArticulationAction

            # Get robot articulation
            robot = Articulation(self._robot_prim_path)
            if not robot.initialized:
                robot.initialize()

            num_dof = robot.num_dof

            # Build action
            action = ArticulationAction()

            # Set joint position targets
            if joint_positions is not None:
                positions = np.array(joint_positions, dtype=np.float32)
                # Pad or truncate to match DOF count
                if len(positions) < num_dof:
                    # Pad with current positions for gripper joints
                    current = robot.get_joint_positions()
                    if current is not None:
                        full_positions = np.array(current, dtype=np.float32)
                        full_positions[:len(positions)] = positions
                        positions = full_positions
                    else:
                        positions = np.pad(positions, (0, num_dof - len(positions)))
                elif len(positions) > num_dof:
                    positions = positions[:num_dof]

                action.joint_positions = positions

            # Set joint velocity targets
            if joint_velocities is not None:
                velocities = np.array(joint_velocities, dtype=np.float32)
                if len(velocities) < num_dof:
                    velocities = np.pad(velocities, (0, num_dof - len(velocities)))
                elif len(velocities) > num_dof:
                    velocities = velocities[:num_dof]

                action.joint_velocities = velocities

            # Apply action
            robot.apply_action(action)

            # Handle gripper separately if specified
            if gripper_command is not None:
                self._apply_gripper_command(robot, gripper_command, num_dof)

            return True

        except Exception as e:
            self.log(f"Failed to apply robot command: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "DEBUG")
            return False

    def _apply_gripper_command(
        self,
        robot: Any,
        gripper_command: float,
        num_dof: int
    ) -> None:
        """
        Apply gripper command to robot.

        Args:
            robot: Articulation instance
            gripper_command: 0=closed, 1=open
            num_dof: Total number of DOFs
        """
        try:
            from omni.isaac.core.utils.types import ArticulationAction

            # Common gripper joint indices (after arm joints)
            # Franka: joints 7, 8 are gripper fingers
            # UR10: depends on gripper attached
            # Fetch: joints 7+ are gripper

            # Get current positions
            current = robot.get_joint_positions()
            if current is None or num_dof <= 7:
                return

            # Assume gripper joints are after arm joints
            gripper_positions = np.array(current, dtype=np.float32)

            # Map 0-1 to gripper range (typically 0-0.04m for Franka)
            # 0 = closed, 1 = open
            gripper_value = gripper_command * 0.04

            # Set gripper joint positions (typically last 1-2 joints)
            for i in range(7, num_dof):
                gripper_positions[i] = gripper_value

            action = ArticulationAction(joint_positions=gripper_positions)
            robot.apply_action(action)

        except Exception as e:
            self.log(f"Failed to apply gripper command: {e}", "WARNING")

    def run_trajectory(
        self,
        joint_trajectory: np.ndarray,
        dt: float = 1.0 / 30.0,
        gripper_trajectory: Optional[np.ndarray] = None,
        post_rollout_steps: int = 0,
    ) -> List[PhysicsStepResult]:
        """
        Run a complete trajectory through physics simulation.

        Args:
            joint_trajectory: Array of shape (num_frames, num_joints)
            dt: Time between trajectory frames
            gripper_trajectory: Optional gripper positions for each frame
            post_rollout_steps: Additional physics steps after last frame

        Returns:
            List of PhysicsStepResult for each frame
        """
        results = []

        num_frames = len(joint_trajectory)
        self.log(f"Running trajectory: {num_frames} frames")

        self.start()

        for i in range(num_frames):
            # Apply robot command
            gripper_cmd = gripper_trajectory[i] if gripper_trajectory is not None else None
            self.apply_robot_command(
                joint_positions=joint_trajectory[i],
                gripper_command=gripper_cmd,
            )

            # Step physics (may need multiple substeps for stability)
            steps_per_frame = max(1, int(dt / self.dt))
            for _ in range(steps_per_frame):
                result = self.step()

            results.append(result)

        for _ in range(max(0, post_rollout_steps)):
            result = self.step()
            results.append(result)

        self.stop()

        return results


# =============================================================================
# Isaac Sim Session
# =============================================================================


class IsaacSimSession:
    """
    Unified Isaac Sim session manager.

    Provides access to:
    - Physics simulation
    - Sensor capture
    - Motion planning
    - Scene management
    """

    _instance: Optional["IsaacSimSession"] = None
    _lock = threading.Lock()  # Thread-safe singleton lock

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._physics_sim: Optional[PhysicsSimulator] = None
        self._sensor_capture = None  # Set up when needed
        self._motion_planner = None  # Set up when needed

        self._scene_loaded = False
        self._scene_path: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "IsaacSimSession":
        """Get or create the singleton session instance (thread-safe)."""
        # Double-checked locking pattern for thread safety
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[ISAAC-SESSION] [%s] %s", level, msg)

    def load_scene(self, scene_path: str) -> bool:
        """Load a USD scene."""
        self._scene_path = scene_path

        # Initialize physics simulator
        if self._physics_sim is None:
            self._physics_sim = PhysicsSimulator(verbose=self.verbose)

        success = self._physics_sim.load_scene(scene_path)
        self._scene_loaded = success

        return success

    def get_physics_simulator(self) -> PhysicsSimulator:
        """Get the physics simulator."""
        if self._physics_sim is None:
            self._physics_sim = PhysicsSimulator(verbose=self.verbose)
        return self._physics_sim

    def is_real_simulation_available(self) -> bool:
        """Check if real physics simulation is available."""
        return _ISAAC_SIM_AVAILABLE and _PHYSX_AVAILABLE


def get_isaac_sim_session() -> IsaacSimSession:
    """Get the global Isaac Sim session."""
    return IsaacSimSession.get_instance()


# =============================================================================
# Module Initialization
# =============================================================================


if __name__ == "__main__":
    # Print availability report when run directly
    from tools.logging_config import init_logging

    init_logging()
    print_availability_report()

    # Test physics simulator
    logger.info("Testing Physics Simulator...")
    sim = PhysicsSimulator(verbose=True)
    sim.add_tracked_object("test_object", "/World/Objects/test")
    sim.start()

    for i in range(5):
        result = sim.step()
        logger.info(
            "  Step %s: success=%s, contacts=%s",
            i,
            result.success,
            result.collision_count,
        )

    sim.stop()
    logger.info("Done!")
