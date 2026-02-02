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

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.config.production_mode import resolve_production_mode

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
    CollisionCheckerType = Any
    WorldConfig = Any
    Cuboid = Any
    Mesh = Any
    TensorDeviceType = Any
    CuRoboPose = Any
    CuRoboJointState = Any
    CuRoboState = Any
    MotionGen = Any
    MotionGenConfig = Any
    MotionGenPlanConfig = Any
    CudaRobotModel = Any
    logger.warning("[CUROBO] cuRobo not available - falling back to CPU planning")

# Import local modules
from motion_planner import Waypoint, MotionPlan, MotionPhase
from trajectory_solver import ROBOT_CONFIGS


# =============================================================================
# Data Models
# =============================================================================


def _requires_curobo() -> bool:
    """Return True when cuRobo is required for production or labs staging runs."""
    return resolve_production_mode()


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
        self.robot_type = robot_type
        self.device = device
        self.interpolation_dt = interpolation_dt
        self.use_fallback = False

        # Get robot config
        if robot_type not in ROBOT_CONFIGS:
            raise ValueError(f"Unsupported robot type: {robot_type}")

        self.robot_config = ROBOT_CONFIGS[robot_type]

        # Initialize cuRobo (or fallback if not available)
        if not CUROBO_AVAILABLE:
            if _requires_curobo():
                raise RuntimeError(
                    "cuRobo is required for production/labs staging runs. "
                    "Install cuRobo with CUDA + PyTorch support before continuing."
                )
            logger.warning(
                "[CUROBO] cuRobo not available - using simple fallback planner"
            )
            logger.warning(
                "[CUROBO] For production quality, install cuRobo: pip install nvidia-curobo"
            )
            self.use_fallback = True
            self.motion_gen = None
        else:
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
            tensor_args=self.tensor_args,
            world_cfg=WorldConfig(),
            trajopt_tsteps=32,  # Trajectory optimization time steps
            interpolation_dt=self.interpolation_dt,
        )

        # Initialize motion generator
        self.motion_gen = MotionGen(motion_gen_config)

        logger.info(
            "[CUROBO] ✅ Initialized for %s on %s", self.robot_type, self.device
        )
        logger.info("[CUROBO]    DOF: %s", self.robot_config["dof"])
        logger.info("[CUROBO]    Interpolation dt: %ss", self.interpolation_dt)

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

        logger.info(
            "[CUROBO] Updated world: %s cuboids, %s meshes",
            len(cuboids),
            len(meshes),
        )

    def _fallback_plan_to_joint_config(
        self, request: CuRoboPlanRequest, start_time: float
    ) -> CuRoboPlanResult:
        """
        Fallback planner using simple linear interpolation.

        NOTE: This does NOT perform collision checking or optimization.
        For production use, install cuRobo: pip install nvidia-curobo

        Args:
            request: Planning request
            start_time: Planning start time

        Returns:
            CuRoboPlanResult with simple interpolated trajectory
        """
        # Linear interpolation between start and goal
        num_steps = 50  # Fixed number of waypoints
        start_pos = np.array(request.start_joint_positions)
        goal_pos = np.array(request.goal_joint_positions)

        # Generate linear interpolation
        alphas = np.linspace(0, 1, num_steps)
        joint_positions = np.array([
            start_pos + alpha * (goal_pos - start_pos)
            for alpha in alphas
        ])

        timesteps = np.arange(num_steps) * self.interpolation_dt
        planning_time = (time.time() - start_time) * 1000

        return CuRoboPlanResult(
            success=True,
            joint_positions=joint_positions,
            timesteps=timesteps,
            planning_time_ms=planning_time,
            num_iterations=1,
            collision_free=False,  # Unknown - no collision checking
            path_length=np.sum(np.linalg.norm(np.diff(joint_positions, axis=0), axis=1)),
            smoothness_cost=0.0,  # Not computed
            warnings=["Using fallback planner - NO collision checking performed. Install cuRobo for production use."],
        )

    def _fallback_plan_to_pose(
        self, request: CuRoboPlanRequest, start_time: float
    ) -> CuRoboPlanResult:
        """
        Fallback planner for pose targets (returns failure).

        NOTE: Pose-based planning requires inverse kinematics, which
        is not available in fallback mode. Use joint-config planning instead.

        Args:
            request: Planning request
            start_time: Planning start time

        Returns:
            CuRoboPlanResult indicating failure
        """
        return CuRoboPlanResult(
            success=False,
            planning_time_ms=(time.time() - start_time) * 1000,
            error_message=(
                "Pose-based planning requires cuRobo (inverse kinematics not available in fallback mode). "
                "Install cuRobo: pip install nvidia-curobo"
            ),
            warnings=["Fallback planner does not support pose-based planning"],
        )

    def plan_to_pose(self, request: CuRoboPlanRequest) -> CuRoboPlanResult:
        """
        Plan trajectory to target end-effector pose.

        Args:
            request: Planning request

        Returns:
            CuRoboPlanResult with trajectory or error
        """
        start_time = time.time()

        # Fallback mode: use simple linear interpolation
        if self.use_fallback:
            return self._fallback_plan_to_pose(request, start_time)

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

        # Fallback mode: use simple linear interpolation
        if self.use_fallback:
            return self._fallback_plan_to_joint_config(request, start_time)

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
        if not requests:
            return []

        # Group requests by world configuration (obstacles)
        # Requests with same obstacles can be batched together
        from collections import defaultdict
        world_groups = defaultdict(list)

        for idx, request in enumerate(requests):
            # Create hashable key from obstacles
            obstacle_key = tuple(sorted(
                f"{obj.object_id}_{obj.position[0]:.3f}_{obj.position[1]:.3f}_{obj.position[2]:.3f}"
                for obj in request.obstacles
            ))
            world_groups[obstacle_key].append((idx, request))

        # Process each world group in batch
        all_results = [None] * len(requests)

        for obstacle_key, group_requests in world_groups.items():
            indices, reqs = zip(*group_requests) if group_requests else ([], [])

            # Update world once for this group
            if reqs and reqs[0].obstacles:
                self.update_world(reqs[0].obstacles)

            try:
                # Prepare batch tensors
                batch_size = len(reqs)
                start_positions = []
                goal_poses = []
                goal_joint_configs = []
                use_pose_goals = []

                for req in reqs:
                    start_positions.append(req.start_joint_positions)

                    if req.goal_joint_positions is not None:
                        goal_joint_configs.append(req.goal_joint_positions)
                        use_pose_goals.append(False)
                    else:
                        goal_poses.append(req.goal_pose)
                        use_pose_goals.append(True)

                # Convert to tensors
                start_states_batch = torch.tensor(
                    start_positions,
                    dtype=torch.float32,
                    device=self.tensor_args.device,
                )

                # Plan batch (use common planning parameters from first request)
                plan_config = MotionGenPlanConfig(
                    enable_graph=True,
                    enable_opt=True,
                    max_attempts=reqs[0].max_iterations if reqs else 1000,
                    enable_finetune_trajopt=reqs[0].parallel_finetune if reqs else True,
                    parallel_finetune=reqs[0].parallel_finetune if reqs else True,
                    finetune_attempts=reqs[0].batch_size if reqs else 32,
                    timeout=30.0,
                )

                start_time = time.time()

                # Process based on goal type
                # For simplicity, if any request uses pose goal, process sequentially
                # Full batch implementation would separate pose vs joint goals
                if any(use_pose_goals):
                    # Mixed batch - process individually (fallback)
                    batch_results = []
                    for req in reqs:
                        if req.goal_joint_positions is not None:
                            result = self.plan_to_joint_config(req)
                        else:
                            result = self.plan_to_pose(req)
                        batch_results.append(result)
                else:
                    # All joint goals - can batch
                    goal_states_batch = torch.tensor(
                        goal_joint_configs,
                        dtype=torch.float32,
                        device=self.tensor_args.device,
                    )

                    # Create batch joint states
                    start_state = CuRoboState(
                        position=start_states_batch,
                        velocity=torch.zeros_like(start_states_batch),
                        acceleration=torch.zeros_like(start_states_batch),
                    )
                    goal_state = CuRoboState(
                        position=goal_states_batch,
                        velocity=torch.zeros_like(goal_states_batch),
                        acceleration=torch.zeros_like(goal_states_batch),
                    )

                    # Batch plan
                    result = self.motion_gen.plan_batch_js(
                        start_state,
                        goal_state,
                        plan_config,
                    )

                    planning_time = (time.time() - start_time) * 1000

                    # Extract results for each request
                    batch_results = []
                    for i in range(batch_size):
                        if result.success[i].item():
                            joint_traj = result.get_interpolated_plan()
                            joint_positions = joint_traj.position[i].cpu().numpy()
                            timesteps = np.arange(len(joint_positions)) * self.interpolation_dt

                            path_length = self._compute_path_length(joint_positions)
                            smoothness = self._compute_smoothness(joint_positions, timesteps)
                            trajectory_duration = timesteps[-1] if len(timesteps) > 0 else 0.0

                            collision_free = not result.is_colliding()[i].item()

                            batch_results.append(CuRoboPlanResult(
                                success=True,
                                joint_trajectory=joint_positions,
                                timesteps=timesteps,
                                planning_time_ms=planning_time / batch_size,  # Amortized time
                                trajectory_duration_s=trajectory_duration,
                                path_length=path_length,
                                smoothness_score=smoothness,
                                is_collision_free=collision_free,
                            ))
                        else:
                            batch_results.append(CuRoboPlanResult(
                                success=False,
                                planning_time_ms=planning_time / batch_size,
                                error_message="Batch planning failed",
                            ))

                # Store results in correct order
                for idx, result in zip(indices, batch_results):
                    all_results[idx] = result

            except Exception as e:
                # Fallback: process group sequentially on error
                logger.warning(
                    "[CUROBO] Batch planning failed, falling back to sequential: %s",
                    e,
                )
                for idx, req in zip(indices, reqs):
                    try:
                        result = self.plan_to_pose(req)
                        all_results[idx] = result
                    except Exception as seq_error:
                        all_results[idx] = CuRoboPlanResult(
                            success=False,
                            error_message=f"Sequential fallback failed: {seq_error}",
                        )

        return all_results

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
        if _requires_curobo():
            raise RuntimeError(
                "cuRobo is required for production/labs staging runs. "
                "Install cuRobo with CUDA + PyTorch support before continuing."
            )
        return None

    try:
        return CuRoboMotionPlanner(robot_type, device)
    except Exception as e:
        logger.error("[CUROBO] Failed to create planner: %s", e)
        return None


if __name__ == "__main__":
    # Test cuRobo availability
    from tools.logging_config import init_logging

    init_logging()
    if is_curobo_available():
        logger.info("✅ cuRobo is available")

        # Create planner
        planner = create_curobo_planner("franka")

        if planner:
            logger.info("✅ Created cuRobo planner for Franka")

            # Test planning
            request = CuRoboPlanRequest(
                start_joint_positions=np.zeros(7),
                goal_pose=np.array([0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
                obstacles=[],
            )

            result = planner.plan_to_pose(request)

            if result.success:
                logger.info(
                    "✅ Planning succeeded in %.1fms", result.planning_time_ms
                )
                logger.info(
                    "   Trajectory: %s waypoints", len(result.joint_trajectory)
                )
                logger.info(
                    "   Duration: %.2fs", result.trajectory_duration_s
                )
                logger.info(
                    "   Smoothness: %.2f", result.smoothness_score
                )
            else:
                logger.error("❌ Planning failed: %s", result.error_message)
    else:
        logger.error("❌ cuRobo not available")
        logger.error("Install with: pip install nvidia-curobo")
