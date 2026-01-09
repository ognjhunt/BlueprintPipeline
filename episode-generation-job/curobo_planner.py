#!/usr/bin/env python3
"""
cuRobo GPU-Accelerated Motion Planner.

This module integrates NVIDIA cuRobo for high-performance motion planning with:
- GPU-accelerated collision checking
- Parallel trajectory optimization
- 10-100x faster than CPU-based planners
- Mesh-level collision detection
- Swept volume collision checking

cuRobo replaces simple AABB collision detection with production-quality
collision-aware motion planning suitable for real robots.

Dependencies:
    - NVIDIA cuRobo library
    - PyTorch with CUDA
    - Isaac Sim (optional, for scene import)

Environment Variables:
    USE_CUROBO: Enable cuRobo motion planning (default: true if available)
    CUROBO_BATCH_SIZE: Parallel trajectory optimization batch size (default: 32)
    CUROBO_MAX_ITERATIONS: Maximum optimization iterations (default: 1000)
"""

import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Try to import cuRobo
try:
    import torch
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import WorldConfig, Cuboid, Mesh
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose as CuRoboPose
    from curobo.types.robot import JointState as CuRoboJointState
    from curobo.types.state import JointState as CuRoboState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

    CUROBO_AVAILABLE = True
except ImportError:
    CUROBO_AVAILABLE = False
    print("[CUROBO] WARNING: cuRobo not available - falling back to CPU planning")

# Import local modules
from motion_planner import Waypoint, MotionPlan, MotionPhase
from trajectory_solver import ROBOT_CONFIGS


# =============================================================================
# Data Models
# =============================================================================


class CollisionGeometryType(str, Enum):
    """Types of collision geometry."""

    CUBOID = "cuboid"  # Axis-aligned bounding box
    MESH = "mesh"  # Triangle mesh
    SPHERE = "sphere"  # Sphere primitive
    CYLINDER = "cylinder"  # Cylinder primitive


@dataclass
class CollisionObject:
    """Collision geometry for obstacle."""

    object_id: str
    geometry_type: CollisionGeometryType

    # Pose
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [qw, qx, qy, qz]

    # Geometry parameters
    dimensions: Optional[np.ndarray] = None  # For cuboid: [length, width, height]
    radius: Optional[float] = None  # For sphere/cylinder
    height: Optional[float] = None  # For cylinder
    mesh_path: Optional[Path] = None  # For mesh

    # Properties
    is_static: bool = True  # Static obstacles vs movable objects


@dataclass
class CuRoboPlanRequest:
    """Request for cuRobo motion planning."""

    # Start and goal
    start_joint_positions: np.ndarray  # Joint positions
    goal_pose: np.ndarray  # End-effector pose [x, y, z, qw, qx, qy, qz]

    # Alternative: goal joint positions
    goal_joint_positions: Optional[np.ndarray] = None

    # Collision objects
    obstacles: List[CollisionObject] = field(default_factory=list)

    # Planning parameters
    max_iterations: int = 1000
    optimize_dt: bool = True  # Optimize trajectory timing
    parallel_finetune: bool = True  # Use parallel optimization
    batch_size: int = 32  # Number of parallel trajectories


@dataclass
class CuRoboPlanResult:
    """Result from cuRobo motion planning."""

    success: bool

    # Trajectory
    joint_trajectory: Optional[np.ndarray] = None  # [T, DOF]
    timesteps: Optional[np.ndarray] = None  # [T]

    # Metrics
    planning_time_ms: float = 0.0
    trajectory_duration_s: float = 0.0
    path_length: float = 0.0
    smoothness_score: float = 0.0

    # Collision info
    is_collision_free: bool = True
    collision_points: List[np.ndarray] = field(default_factory=list)

    # Error info
    error_message: str = ""


# =============================================================================
# cuRobo Motion Planner
# =============================================================================


class CuRoboMotionPlanner:
    """
    GPU-accelerated motion planner using NVIDIA cuRobo.

    This planner provides:
    - Fast collision-aware trajectory optimization
    - Parallel batch planning (32+ trajectories simultaneously)
    - Mesh-level collision detection
    - Trajectory smoothness optimization
    """

    def __init__(
        self,
        robot_type: str = "franka",
        device: str = "cuda:0",
        interpolation_dt: float = 0.02,
    ):
        """
        Initialize cuRobo planner.

        Args:
            robot_type: Robot type (franka, ur10, etc.)
            device: CUDA device
            interpolation_dt: Time step for interpolation (default 50Hz)
        """
        if not CUROBO_AVAILABLE:
            raise RuntimeError("cuRobo not available. Install with: pip install nvidia-curobo")

        self.robot_type = robot_type
        self.device = device
        self.interpolation_dt = interpolation_dt

        # Get robot config
        if robot_type not in ROBOT_CONFIGS:
            raise ValueError(f"Unsupported robot type: {robot_type}")

        self.robot_config = ROBOT_CONFIGS[robot_type]

        # Initialize cuRobo
        self._init_curobo()

    def _init_curobo(self):
        """Initialize cuRobo motion generator."""
        # Tensor device config
        self.tensor_args = TensorDeviceType(device=torch.device(self.device))

        # Map robot types to cuRobo config files
        robot_cfg_map = {
            "franka": "franka.yml",
            "ur10": "ur10e.yml",
            "ur5": "ur5e.yml",
            "fetch": "fetch.yml",
            "kinova": "kinova_gen3.yml",
        }

        robot_cfg_file = robot_cfg_map.get(self.robot_type, "franka.yml")

        # Create motion generator configuration
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg_file,
            self.tensor_args,
            trajopt_tsteps=32,  # Trajectory optimization time steps
            interpolation_dt=self.interpolation_dt,
        )

        # Initialize motion generator
        self.motion_gen = MotionGen(motion_gen_config)

        print(f"[CUROBO] ✅ Initialized for {self.robot_type} on {self.device}")
        print(f"[CUROBO]    DOF: {self.robot_config['dof']}")
        print(f"[CUROBO]    Interpolation dt: {self.interpolation_dt}s")

    def update_world(self, obstacles: List[CollisionObject]):
        """
        Update collision world with obstacles.

        Args:
            obstacles: List of collision objects
        """
        # Convert obstacles to cuRobo world config
        cuboids = []
        meshes = []

        for obj in obstacles:
            if obj.geometry_type == CollisionGeometryType.CUBOID:
                cuboid = Cuboid(
                    name=obj.object_id,
                    pose=self._numpy_pose_to_curobo(obj.position, obj.orientation),
                    dims=obj.dimensions.tolist(),
                )
                cuboids.append(cuboid)

            elif obj.geometry_type == CollisionGeometryType.MESH and obj.mesh_path:
                mesh = Mesh(
                    name=obj.object_id,
                    pose=self._numpy_pose_to_curobo(obj.position, obj.orientation),
                    file_path=str(obj.mesh_path),
                )
                meshes.append(mesh)

        # Create world config
        world_cfg = WorldConfig(cuboid=cuboids, mesh=meshes)

        # Update world
        self.motion_gen.update_world(world_cfg)

        print(f"[CUROBO] Updated world: {len(cuboids)} cuboids, {len(meshes)} meshes")

    def plan_to_pose(self, request: CuRoboPlanRequest) -> CuRoboPlanResult:
        """
        Plan trajectory to target end-effector pose.

        Args:
            request: Planning request

        Returns:
            CuRoboPlanResult with trajectory or error
        """
        start_time = time.time()

        try:
            # Update world with obstacles
            if request.obstacles:
                self.update_world(request.obstacles)

            # Create start state
            start_state = self._create_joint_state(request.start_joint_positions)

            # Create goal pose
            goal_pose = self._create_goal_pose(request.goal_pose)

            # Create planning config
            plan_config = MotionGenPlanConfig(
                enable_graph=True,
                enable_opt=True,
                max_attempts=request.max_iterations,
                enable_finetune_trajopt=request.parallel_finetune,
                parallel_finetune=request.parallel_finetune,
                finetune_attempts=request.batch_size if request.parallel_finetune else 1,
                timeout=30.0,  # 30 second timeout
            )

            # Plan!
            result = self.motion_gen.plan_single(
                start_state,
                goal_pose,
                plan_config,
            )

            planning_time = (time.time() - start_time) * 1000  # ms

            if result.success.item():
                # Extract trajectory
                joint_traj = result.get_interpolated_plan()  # Get smooth interpolated trajectory
                joint_positions = joint_traj.position.cpu().numpy()  # [T, DOF]
                timesteps = np.arange(len(joint_positions)) * self.interpolation_dt

                # Compute metrics
                path_length = self._compute_path_length(joint_positions)
                smoothness = self._compute_smoothness(joint_positions, timesteps)
                trajectory_duration = timesteps[-1] if len(timesteps) > 0 else 0.0

                # Check collision
                collision_free = not result.is_colliding()

                return CuRoboPlanResult(
                    success=True,
                    joint_trajectory=joint_positions,
                    timesteps=timesteps,
                    planning_time_ms=planning_time,
                    trajectory_duration_s=trajectory_duration,
                    path_length=path_length,
                    smoothness_score=smoothness,
                    is_collision_free=collision_free,
                )

            else:
                return CuRoboPlanResult(
                    success=False,
                    planning_time_ms=planning_time,
                    error_message="Planning failed - no collision-free path found",
                )

        except Exception as e:
            return CuRoboPlanResult(
                success=False,
                planning_time_ms=(time.time() - start_time) * 1000,
                error_message=f"Exception during planning: {str(e)}",
            )

    def plan_to_joint_config(self, request: CuRoboPlanRequest) -> CuRoboPlanResult:
        """
        Plan trajectory to target joint configuration.

        Args:
            request: Planning request (must have goal_joint_positions)

        Returns:
            CuRoboPlanResult with trajectory or error
        """
        if request.goal_joint_positions is None:
            return CuRoboPlanResult(
                success=False,
                error_message="goal_joint_positions required for joint config planning",
            )

        start_time = time.time()

        try:
            # Update world
            if request.obstacles:
                self.update_world(request.obstacles)

            # Create states
            start_state = self._create_joint_state(request.start_joint_positions)
            goal_state = self._create_joint_state(request.goal_joint_positions)

            # Plan
            plan_config = MotionGenPlanConfig(
                enable_graph=True,
                enable_opt=True,
                max_attempts=request.max_iterations,
                enable_finetune_trajopt=request.parallel_finetune,
                parallel_finetune=request.parallel_finetune,
                finetune_attempts=request.batch_size if request.parallel_finetune else 1,
                timeout=30.0,
            )

            result = self.motion_gen.plan_single_js(
                start_state,
                goal_state,
                plan_config,
            )

            planning_time = (time.time() - start_time) * 1000

            if result.success.item():
                joint_traj = result.get_interpolated_plan()
                joint_positions = joint_traj.position.cpu().numpy()
                timesteps = np.arange(len(joint_positions)) * self.interpolation_dt

                path_length = self._compute_path_length(joint_positions)
                smoothness = self._compute_smoothness(joint_positions, timesteps)
                trajectory_duration = timesteps[-1] if len(timesteps) > 0 else 0.0

                collision_free = not result.is_colliding()

                return CuRoboPlanResult(
                    success=True,
                    joint_trajectory=joint_positions,
                    timesteps=timesteps,
                    planning_time_ms=planning_time,
                    trajectory_duration_s=trajectory_duration,
                    path_length=path_length,
                    smoothness_score=smoothness,
                    is_collision_free=collision_free,
                )
            else:
                return CuRoboPlanResult(
                    success=False,
                    planning_time_ms=planning_time,
                    error_message="Planning failed",
                )

        except Exception as e:
            return CuRoboPlanResult(
                success=False,
                planning_time_ms=(time.time() - start_time) * 1000,
                error_message=f"Exception: {str(e)}",
            )

    def batch_plan(
        self,
        requests: List[CuRoboPlanRequest],
    ) -> List[CuRoboPlanResult]:
        """
        Plan multiple trajectories in parallel (GPU batch processing).

        This is one of cuRobo's key advantages - planning 32+ trajectories
        simultaneously on GPU.

        Args:
            requests: List of planning requests

        Returns:
            List of planning results (same order as requests)
        """
        # For now, process sequentially
        # TODO: Implement true batch planning with cuRobo batch API
        results = []
        for request in requests:
            result = self.plan_to_pose(request)
            results.append(result)

        return results

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _create_joint_state(self, joint_positions: np.ndarray) -> CuRoboState:
        """Create cuRobo joint state from numpy array."""
        position_tensor = torch.tensor(
            joint_positions,
            dtype=torch.float32,
            device=self.tensor_args.device,
        ).unsqueeze(0)  # Add batch dimension

        return CuRoboState(
            position=position_tensor,
            velocity=torch.zeros_like(position_tensor),
            acceleration=torch.zeros_like(position_tensor),
        )

    def _create_goal_pose(self, pose: np.ndarray) -> CuRoboPose:
        """
        Create cuRobo pose from numpy array.

        Args:
            pose: [x, y, z, qw, qx, qy, qz]

        Returns:
            CuRoboPose
        """
        position = torch.tensor(
            pose[:3],
            dtype=torch.float32,
            device=self.tensor_args.device,
        ).unsqueeze(0)

        quaternion = torch.tensor(
            pose[3:],
            dtype=torch.float32,
            device=self.tensor_args.device,
        ).unsqueeze(0)

        return CuRoboPose(position=position, quaternion=quaternion)

    def _numpy_pose_to_curobo(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> List[float]:
        """Convert numpy pose to cuRobo pose list."""
        return position.tolist() + orientation.tolist()

    def _compute_path_length(self, joint_trajectory: np.ndarray) -> float:
        """Compute total path length in joint space."""
        diffs = np.diff(joint_trajectory, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)

    def _compute_smoothness(
        self,
        joint_trajectory: np.ndarray,
        timesteps: np.ndarray,
    ) -> float:
        """
        Compute trajectory smoothness score (0-1, higher is smoother).

        Based on jerk (third derivative of position).
        """
        if len(joint_trajectory) < 3:
            return 1.0

        # Compute velocities
        dt = np.diff(timesteps)
        velocities = np.diff(joint_trajectory, axis=0) / dt[:, np.newaxis]

        # Compute accelerations
        accelerations = np.diff(velocities, axis=0) / dt[1:, np.newaxis]

        # Compute jerks
        jerks = np.diff(accelerations, axis=0) / dt[2:, np.newaxis]

        # Smoothness score: lower jerk = smoother
        mean_jerk = np.mean(np.abs(jerks))

        # Normalize to 0-1 (empirical threshold: 100 rad/s^3 = poor)
        smoothness = 1.0 / (1.0 + mean_jerk / 100.0)

        return smoothness


# =============================================================================
# Utility Functions
# =============================================================================


def is_curobo_available() -> bool:
    """Check if cuRobo is available."""
    return CUROBO_AVAILABLE


def create_curobo_planner(
    robot_type: str = "franka",
    device: str = "cuda:0",
) -> Optional[CuRoboMotionPlanner]:
    """
    Create cuRobo planner if available.

    Args:
        robot_type: Robot type
        device: CUDA device

    Returns:
        CuRoboMotionPlanner or None if unavailable
    """
    if not CUROBO_AVAILABLE:
        return None

    try:
        return CuRoboMotionPlanner(robot_type, device)
    except Exception as e:
        print(f"[CUROBO] Failed to create planner: {e}")
        return None


if __name__ == "__main__":
    # Test cuRobo availability
    if is_curobo_available():
        print("✅ cuRobo is available")

        # Create planner
        planner = create_curobo_planner("franka")

        if planner:
            print(f"✅ Created cuRobo planner for Franka")

            # Test planning
            request = CuRoboPlanRequest(
                start_joint_positions=np.zeros(7),
                goal_pose=np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
                obstacles=[],
            )

            result = planner.plan_to_pose(request)

            if result.success:
                print(f"✅ Planning succeeded in {result.planning_time_ms:.1f}ms")
                print(f"   Trajectory: {len(result.joint_trajectory)} waypoints")
                print(f"   Duration: {result.trajectory_duration_s:.2f}s")
                print(f"   Smoothness: {result.smoothness_score:.2f}")
            else:
                print(f"❌ Planning failed: {result.error_message}")
    else:
        print("❌ cuRobo not available")
        print("Install with: pip install nvidia-curobo")
