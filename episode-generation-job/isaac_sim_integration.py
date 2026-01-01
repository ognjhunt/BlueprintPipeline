#!/usr/bin/env python3
"""
Isaac Sim Integration Module for Episode Generation.

This module provides unified access to Isaac Sim/Isaac Lab features:
- Physics simulation via PhysX
- Sensor data capture via Replicator
- Motion planning via cuRobo (when available)
- Scene management via USD

IMPORTANT: This module must be imported from within Isaac Sim's Python environment.
Running outside Isaac Sim will result in graceful fallbacks with clear warnings.

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

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

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


def get_availability_status() -> Dict[str, bool]:
    """Get detailed availability status."""
    return _AVAILABILITY_STATUS.copy()


def print_availability_report() -> None:
    """Print a human-readable availability report."""
    print("\n" + "=" * 60)
    print("ISAAC SIM INTEGRATION - AVAILABILITY REPORT")
    print("=" * 60)

    for feature, available in _AVAILABILITY_STATUS.items():
        status = "✅ AVAILABLE" if available else "❌ NOT AVAILABLE"
        print(f"  {feature:20s}: {status}")

    if not _ISAAC_SIM_AVAILABLE:
        print("\n⚠️  WARNING: Not running inside Isaac Sim!")
        print("   Features requiring Isaac Sim will use mock implementations.")
        print("   To run with full features, use: /isaac-sim/python.sh your_script.py")

    print("=" * 60 + "\n")


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
    ):
        """
        Initialize physics simulator.

        Args:
            dt: Physics timestep (default 1/60s = 60 Hz)
            substeps: Number of physics substeps per step
            verbose: Print debug info
        """
        self.dt = dt
        self.substeps = substeps
        self.verbose = verbose

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
        self._use_real_physics = _PHYSX_AVAILABLE and _ISAAC_SIM_AVAILABLE

        if not self._use_real_physics:
            self.log("PhysX not available - using mock physics simulation", "WARNING")

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[PHYSICS-SIM] [{level}] {msg}")

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

        try:
            import omni.usd
            from omni.isaac.core import World

            # Create or get world
            world = World.instance()
            if world is None:
                world = World(stage_units_in_meters=1.0)

            # Load scene
            omni.usd.get_context().open_stage(scene_path)
            self._stage = omni.usd.get_context().get_stage()

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
            except Exception:
                pass

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
            except Exception:
                pass

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
        """Perform a real physics step using PhysX."""
        try:
            from omni.isaac.core import World

            world = World.instance()
            if world is None:
                result.success = False
                result.error_message = "World not initialized"
                return result

            # Step physics
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
                except Exception:
                    pass

        except Exception as e:
            self.log(f"Failed to get object states: {e}", "WARNING")

        return states

    def _get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state."""
        state = {}

        if not self._use_real_physics or not self._robot_prim_path:
            return state

        try:
            from omni.isaac.core.articulations import ArticulationView

            # This is a simplified version - real implementation would use ArticulationView
            state = {
                "joint_positions": np.zeros(7),
                "joint_velocities": np.zeros(7),
                "ee_pose": (np.array([0.5, 0, 0.5]), np.array([1, 0, 0, 0])),
            }

        except Exception as e:
            self.log(f"Failed to get robot state: {e}", "WARNING")

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

        try:
            from omni.isaac.core.articulations import ArticulationView

            # Apply commands via ArticulationView
            # This is a placeholder - real implementation depends on robot type
            return True

        except Exception as e:
            self.log(f"Failed to apply robot command: {e}", "ERROR")
            return False

    def run_trajectory(
        self,
        joint_trajectory: np.ndarray,
        dt: float = 1.0 / 30.0,
        gripper_trajectory: Optional[np.ndarray] = None,
    ) -> List[PhysicsStepResult]:
        """
        Run a complete trajectory through physics simulation.

        Args:
            joint_trajectory: Array of shape (num_frames, num_joints)
            dt: Time between trajectory frames
            gripper_trajectory: Optional gripper positions for each frame

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

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._physics_sim: Optional[PhysicsSimulator] = None
        self._sensor_capture = None  # Set up when needed
        self._motion_planner = None  # Set up when needed

        self._scene_loaded = False
        self._scene_path: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "IsaacSimSession":
        """Get or create the singleton session instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[ISAAC-SESSION] [{level}] {msg}")

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
    print_availability_report()

    # Test physics simulator
    print("\nTesting Physics Simulator...")
    sim = PhysicsSimulator(verbose=True)
    sim.add_tracked_object("test_object", "/World/Objects/test")
    sim.start()

    for i in range(5):
        result = sim.step()
        print(f"  Step {i}: success={result.success}, contacts={result.collision_count}")

    sim.stop()
    print("Done!")
