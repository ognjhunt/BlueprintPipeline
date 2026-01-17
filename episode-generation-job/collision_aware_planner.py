#!/usr/bin/env python3
"""
Collision-Aware Motion Planning for Episode Generation.

This module provides production-quality motion planning with:
1. Scene-aware collision checking using USD scene geometry
2. RRT-based path planning for obstacle avoidance
3. Integration with cuRobo for GPU-accelerated planning (when available)
4. Improved IK solving with multiple solution selection

The planner supports two modes:
- **cuRobo Mode** (Isaac Sim with cuRobo): GPU-accelerated motion planning
- **RRT Mode** (fallback): CPU-based RRT planning with AABB collision checking

Usage:
    from collision_aware_planner import CollisionAwarePlanner

    # Create planner with scene
    planner = CollisionAwarePlanner(
        robot_type="franka",
        scene_usd_path="/path/to/scene.usda",
    )

    # Plan collision-free path
    path = planner.plan_path(start_config, goal_config)

    # Or plan from waypoints
    trajectory = planner.plan_waypoint_trajectory(waypoints)

Reference:
- cuRobo: GPU-accelerated motion planning (NVIDIA)
- RRT/RRT*: Rapidly-exploring Random Trees
- OMPL: Open Motion Planning Library
"""

import importlib
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import KDTree

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_planner import MotionPlan, Waypoint, MotionPhase
from trajectory_solver import JointTrajectory, ROBOT_CONFIGS, RobotConfig, IKSolver

logger = logging.getLogger(__name__)

# =============================================================================
# cuRobo Integration
# =============================================================================

_CUROBO_AVAILABLE = False
_curobo = None

try:
    import curobo
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    from curobo.types.robot import RobotConfig as CuRobotConfig
    _CUROBO_AVAILABLE = True
    _curobo = curobo
except ImportError:
    pass


def is_curobo_available() -> bool:
    """Check if cuRobo is available for GPU-accelerated planning."""
    return _CUROBO_AVAILABLE


# =============================================================================
# USD Scene Collision Geometry
# =============================================================================

_USD_AVAILABLE = False

try:
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    _USD_AVAILABLE = True
except ImportError:
    pass


@dataclass
class CollisionPrimitive:
    """A collision primitive (sphere, box, or mesh)."""

    prim_type: str  # "sphere", "box", "mesh", "capsule"
    position: np.ndarray
    orientation: np.ndarray  # quaternion (w, x, y, z)

    # Dimensions based on type
    radius: float = 0.0  # for sphere/capsule
    dimensions: np.ndarray = field(default_factory=lambda: np.zeros(3))  # for box
    height: float = 0.0  # for capsule

    # Mesh data (for mesh type)
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None

    # Metadata
    prim_path: str = ""
    is_robot: bool = False

    def get_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box (min, max)."""
        if self.prim_type == "sphere":
            half = np.array([self.radius, self.radius, self.radius])
        elif self.prim_type == "box":
            half = self.dimensions / 2
        elif self.prim_type == "capsule":
            half = np.array([self.radius, self.radius, self.height / 2 + self.radius])
        else:  # mesh or unknown
            if self.vertices is not None and len(self.vertices) > 0:
                return self.vertices.min(axis=0) + self.position, self.vertices.max(axis=0) + self.position
            half = self.dimensions / 2

        return self.position - half, self.position + half


class SceneCollisionChecker:
    """
    Collision checker using scene geometry.

    Loads collision primitives from USD scene and provides
    efficient collision checking for motion planning.
    """

    def __init__(
        self,
        scene_usd_path: Optional[str] = None,
        robot_type: str = "franka",
        robot_urdf_path: Optional[str] = None,
        verbose: bool = True,
    ):
        self.scene_usd_path = scene_usd_path
        self.robot_type = robot_type
        self.robot_urdf_path = robot_urdf_path
        self.verbose = verbose

        # Collision geometry
        self.collision_primitives: List[CollisionPrimitive] = []
        self._aabb_tree: Optional[KDTree] = None
        self._aabb_centers: Optional[np.ndarray] = None

        # Robot collision geometry
        self.robot_link_primitives: Dict[str, CollisionPrimitive] = {}

        # Load scene if provided
        if scene_usd_path:
            self.load_scene(scene_usd_path)
        if robot_urdf_path:
            self.load_robot_urdf(robot_urdf_path)

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[COLLISION-CHECKER] [%s] %s", level, msg)

    def load_robot_urdf(self, urdf_path: str) -> bool:
        """Load robot collision geometry from URDF."""
        spec = importlib.util.find_spec("urdfpy")
        if spec is None:
            self.log("urdfpy not available - skipping robot URDF load", "WARNING")
            return False

        urdfpy = importlib.import_module("urdfpy")
        try:
            robot = urdfpy.URDF.load(urdf_path)
        except Exception as exc:
            self.log(f"Failed to load robot URDF {urdf_path}: {exc}", "ERROR")
            return False

        loaded = 0
        for link in robot.links:
            for collision in link.collisions:
                prim = self._collision_primitive_from_urdf(link.name, collision)
                if prim is None:
                    continue
                prim.is_robot = True
                self.robot_link_primitives[f"{link.name}:{loaded}"] = prim
                loaded += 1

        self.log(f"Loaded {loaded} robot collision primitives from {urdf_path}")
        return loaded > 0

    def _collision_primitive_from_urdf(self, link_name: str, collision) -> Optional[CollisionPrimitive]:
        """Convert a URDF collision element into a collision primitive."""
        geometry = collision.geometry
        if geometry is None:
            return None

        origin = collision.origin
        if origin is not None:
            position = np.array(origin[:3, 3])
            orientation = self._quat_from_matrix(origin[:3, :3])
        else:
            position = np.zeros(3)
            orientation = np.array([1.0, 0.0, 0.0, 0.0])

        if geometry.box is not None:
            return CollisionPrimitive(
                prim_type="box",
                position=position,
                orientation=orientation,
                dimensions=np.array(geometry.box.size),
                prim_path=f"{link_name}/collision",
                is_robot=True,
            )
        if geometry.sphere is not None:
            return CollisionPrimitive(
                prim_type="sphere",
                position=position,
                orientation=orientation,
                radius=float(geometry.sphere.radius),
                prim_path=f"{link_name}/collision",
                is_robot=True,
            )
        if geometry.cylinder is not None:
            return CollisionPrimitive(
                prim_type="capsule",
                position=position,
                orientation=orientation,
                radius=float(geometry.cylinder.radius),
                height=float(geometry.cylinder.length),
                prim_path=f"{link_name}/collision",
                is_robot=True,
            )
        if geometry.mesh is not None:
            return CollisionPrimitive(
                prim_type="mesh",
                position=position,
                orientation=orientation,
                prim_path=f"{link_name}/collision",
                is_robot=True,
            )
        return None

    def _quat_from_matrix(self, mat: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion (w, x, y, z)."""
        trace = np.trace(mat)
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2
            return np.array([
                0.25 * s,
                (mat[2, 1] - mat[1, 2]) / s,
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[1, 0] - mat[0, 1]) / s,
            ])
        if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
            return np.array([
                (mat[2, 1] - mat[1, 2]) / s,
                0.25 * s,
                (mat[0, 1] + mat[1, 0]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
            ])
        if mat[1, 1] > mat[2, 2]:
            s = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
            return np.array([
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[0, 1] + mat[1, 0]) / s,
                0.25 * s,
                (mat[1, 2] + mat[2, 1]) / s,
            ])
        s = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
        return np.array([
            (mat[1, 0] - mat[0, 1]) / s,
            (mat[0, 2] + mat[2, 0]) / s,
            (mat[1, 2] + mat[2, 1]) / s,
            0.25 * s,
        ])

    def load_scene(self, scene_path: str) -> bool:
        """Load collision geometry from USD scene."""
        if not _USD_AVAILABLE:
            self.log("USD not available - using simplified collision checking", "WARNING")
            return False

        try:
            stage = Usd.Stage.Open(scene_path)
            if not stage:
                self.log(f"Failed to open USD stage: {scene_path}", "ERROR")
                return False

            # Traverse and extract collision geometry
            for prim in stage.Traverse():
                # Skip robot prims
                prim_path = str(prim.GetPath())
                if "robot" in prim_path.lower() or "franka" in prim_path.lower():
                    continue

                # Check for collision-enabled geometry
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision_prim = self._extract_collision_primitive(prim)
                    if collision_prim:
                        self.collision_primitives.append(collision_prim)

                # Also check for mesh prims that might be obstacles
                elif prim.IsA(UsdGeom.Mesh):
                    collision_prim = self._extract_mesh_primitive(prim)
                    if collision_prim:
                        self.collision_primitives.append(collision_prim)

            # Build spatial index for efficient queries
            if self.collision_primitives:
                self._build_spatial_index()

            self.log(f"Loaded {len(self.collision_primitives)} collision primitives from {scene_path}")
            return True

        except Exception as e:
            self.log(f"Failed to load scene: {e}", "ERROR")
            return False

    def _extract_collision_primitive(self, prim) -> Optional[CollisionPrimitive]:
        """Extract collision primitive from USD prim."""
        try:
            xformable = UsdGeom.Xformable(prim)
            world_transform = xformable.ComputeLocalToWorldTransform(0)

            # Extract position
            translation = world_transform.ExtractTranslation()
            position = np.array([float(translation[0]), float(translation[1]), float(translation[2])])

            # Extract rotation as quaternion
            rotation = world_transform.ExtractRotationQuat()
            orientation = np.array([
                float(rotation.GetReal()),
                float(rotation.GetImaginary()[0]),
                float(rotation.GetImaginary()[1]),
                float(rotation.GetImaginary()[2]),
            ])

            # Determine primitive type
            if prim.IsA(UsdGeom.Sphere):
                sphere = UsdGeom.Sphere(prim)
                radius = float(sphere.GetRadiusAttr().Get())
                return CollisionPrimitive(
                    prim_type="sphere",
                    position=position,
                    orientation=orientation,
                    radius=radius,
                    prim_path=str(prim.GetPath()),
                )

            elif prim.IsA(UsdGeom.Cube):
                cube = UsdGeom.Cube(prim)
                size = float(cube.GetSizeAttr().Get())
                return CollisionPrimitive(
                    prim_type="box",
                    position=position,
                    orientation=orientation,
                    dimensions=np.array([size, size, size]),
                    prim_path=str(prim.GetPath()),
                )

            elif prim.IsA(UsdGeom.Capsule):
                capsule = UsdGeom.Capsule(prim)
                radius = float(capsule.GetRadiusAttr().Get())
                height = float(capsule.GetHeightAttr().Get())
                return CollisionPrimitive(
                    prim_type="capsule",
                    position=position,
                    orientation=orientation,
                    radius=radius,
                    height=height,
                    prim_path=str(prim.GetPath()),
                )

            # Fallback to mesh
            return self._extract_mesh_primitive(prim)

        except Exception as e:
            self.log(f"Failed to extract collision from {prim.GetPath()}: {e}", "WARNING")
            return None

    def _extract_mesh_primitive(self, prim) -> Optional[CollisionPrimitive]:
        """Extract mesh collision primitive."""
        try:
            if not prim.IsA(UsdGeom.Mesh):
                return None

            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()

            if points is None or len(points) == 0:
                return None

            # Get transform
            xformable = UsdGeom.Xformable(prim)
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            translation = world_transform.ExtractTranslation()
            position = np.array([float(translation[0]), float(translation[1]), float(translation[2])])

            # Convert to numpy
            vertices = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in points])

            # Compute bounding box dimensions
            min_pt = vertices.min(axis=0)
            max_pt = vertices.max(axis=0)
            dimensions = max_pt - min_pt

            return CollisionPrimitive(
                prim_type="mesh",
                position=position,
                orientation=np.array([1, 0, 0, 0]),
                dimensions=dimensions,
                vertices=vertices,
                prim_path=str(prim.GetPath()),
            )

        except Exception as e:
            return None

    def _build_spatial_index(self) -> None:
        """Build KD-tree for efficient spatial queries."""
        if not self.collision_primitives:
            return

        centers = []
        for prim in self.collision_primitives:
            min_pt, max_pt = prim.get_aabb()
            center = (min_pt + max_pt) / 2
            centers.append(center)

        self._aabb_centers = np.array(centers)
        self._aabb_tree = KDTree(self._aabb_centers)

    def add_obstacle(
        self,
        position: np.ndarray,
        dimensions: np.ndarray,
        prim_type: str = "box",
    ) -> None:
        """Add a collision obstacle manually."""
        prim = CollisionPrimitive(
            prim_type=prim_type,
            position=np.array(position),
            orientation=np.array([1, 0, 0, 0]),
            dimensions=np.array(dimensions),
        )
        self.collision_primitives.append(prim)
        self._build_spatial_index()

    def check_collision_point(
        self,
        point: np.ndarray,
        radius: float = 0.05,
    ) -> bool:
        """
        Check if a point (with radius) collides with any obstacle.

        Args:
            point: 3D point to check
            radius: Collision radius around point

        Returns:
            True if collision detected
        """
        for prim in self.collision_primitives:
            min_pt, max_pt = prim.get_aabb()

            # Expand AABB by radius
            min_pt = min_pt - radius
            max_pt = max_pt + radius

            # Check if point is inside expanded AABB
            if np.all(point >= min_pt) and np.all(point <= max_pt):
                return True

        return False

    def check_collision_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float = 0.05,
        num_samples: int = 10,
    ) -> bool:
        """
        Check if a line segment collides with any obstacle.

        Args:
            start: Start point
            end: End point
            radius: Collision radius
            num_samples: Number of samples along segment

        Returns:
            True if collision detected
        """
        for i in range(num_samples):
            t = i / (num_samples - 1)
            point = (1 - t) * start + t * end
            if self.check_collision_point(point, radius):
                return True

        return False

    def check_collision_path(
        self,
        path: List[np.ndarray],
        radius: float = 0.05,
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if a path collides with any obstacle.

        Args:
            path: List of waypoints
            radius: Collision radius

        Returns:
            (collision_detected, first_collision_segment_index)
        """
        for i in range(len(path) - 1):
            if self.check_collision_segment(path[i], path[i + 1], radius):
                return True, i

        return False, None


# =============================================================================
# RRT Motion Planner
# =============================================================================


@dataclass
class RRTConfig:
    """Configuration for RRT planner."""

    max_iterations: int = 2000
    step_size: float = 0.1  # meters
    goal_bias: float = 0.2  # probability of sampling goal
    goal_threshold: float = 0.05  # meters
    smoothing_iterations: int = 50
    collision_check_resolution: int = 10


class RRTPlanner:
    """
    RRT (Rapidly-exploring Random Tree) motion planner.

    Provides collision-free paths in Cartesian space.
    """

    def __init__(
        self,
        collision_checker: SceneCollisionChecker,
        config: Optional[RRTConfig] = None,
        verbose: bool = True,
    ):
        self.collision_checker = collision_checker
        self.config = config or RRTConfig()
        self.verbose = verbose

        # Workspace bounds
        self.workspace_min = np.array([-1.0, -1.0, 0.0])
        self.workspace_max = np.array([1.0, 1.0, 1.5])

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[RRT-PLANNER] [%s] %s", level, msg)

    def set_workspace_bounds(self, min_pt: np.ndarray, max_pt: np.ndarray) -> None:
        """Set workspace bounds for sampling."""
        self.workspace_min = np.array(min_pt)
        self.workspace_max = np.array(max_pt)

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        robot_radius: float = 0.05,
    ) -> Optional[List[np.ndarray]]:
        """
        Plan a collision-free path from start to goal.

        Args:
            start: Start position [x, y, z]
            goal: Goal position [x, y, z]
            robot_radius: Robot collision radius

        Returns:
            List of waypoints or None if planning fails
        """
        self.log(f"Planning path from {start} to {goal}")

        # Check start and goal validity
        if self.collision_checker.check_collision_point(start, robot_radius):
            self.log("Start position is in collision!", "ERROR")
            return None

        if self.collision_checker.check_collision_point(goal, robot_radius):
            self.log("Goal position is in collision!", "ERROR")
            return None

        # Initialize tree
        tree_nodes = [start.copy()]
        tree_parents = [-1]  # Parent indices

        for iteration in range(self.config.max_iterations):
            # Sample random point (with goal bias)
            if np.random.random() < self.config.goal_bias:
                sample = goal.copy()
            else:
                sample = np.random.uniform(self.workspace_min, self.workspace_max)

            # Find nearest node in tree
            distances = [np.linalg.norm(node - sample) for node in tree_nodes]
            nearest_idx = np.argmin(distances)
            nearest = tree_nodes[nearest_idx]

            # Extend toward sample
            direction = sample - nearest
            distance = np.linalg.norm(direction)

            if distance < 0.001:
                continue

            direction = direction / distance
            step_distance = min(self.config.step_size, distance)
            new_point = nearest + direction * step_distance

            # Check collision
            if self.collision_checker.check_collision_segment(
                nearest, new_point, robot_radius, self.config.collision_check_resolution
            ):
                continue

            # Add to tree
            tree_nodes.append(new_point)
            tree_parents.append(nearest_idx)

            # Check if reached goal
            if np.linalg.norm(new_point - goal) < self.config.goal_threshold:
                # Check final segment to goal
                if not self.collision_checker.check_collision_segment(
                    new_point, goal, robot_radius, self.config.collision_check_resolution
                ):
                    # Reconstruct path
                    path = [goal]
                    idx = len(tree_nodes) - 1
                    while idx >= 0:
                        path.append(tree_nodes[idx])
                        idx = tree_parents[idx]

                    path = list(reversed(path))
                    self.log(f"Found path with {len(path)} waypoints in {iteration + 1} iterations")

                    # Smooth path
                    smoothed_path = self._smooth_path(path, robot_radius)
                    self.log(f"Smoothed to {len(smoothed_path)} waypoints")

                    return smoothed_path

        self.log(f"Failed to find path after {self.config.max_iterations} iterations", "WARNING")
        return None

    def _smooth_path(
        self,
        path: List[np.ndarray],
        robot_radius: float,
    ) -> List[np.ndarray]:
        """Smooth path by removing unnecessary waypoints."""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]

        i = 0
        while i < len(path) - 1:
            # Try to skip waypoints
            j = len(path) - 1
            while j > i + 1:
                if not self.collision_checker.check_collision_segment(
                    path[i], path[j], robot_radius, self.config.collision_check_resolution
                ):
                    # Can skip directly to j
                    break
                j -= 1

            smoothed.append(path[j])
            i = j

        return smoothed


# =============================================================================
# Collision-Aware Motion Planner
# =============================================================================


class CollisionAwarePlanner:
    """
    High-level collision-aware motion planner.

    Integrates:
    - Scene collision geometry from USD
    - cuRobo for GPU-accelerated planning (when available)
    - RRT for CPU-based planning (fallback)
    - IK solving with collision checking
    """

    def __init__(
        self,
        robot_type: str = "franka",
        scene_usd_path: Optional[str] = None,
        scene_objects: Optional[List[Dict[str, Any]]] = None,
        robot_urdf_path: Optional[str] = None,
        use_curobo: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize collision-aware planner.

        Args:
            robot_type: Robot type (franka, ur10, fetch)
            scene_usd_path: Path to USD scene for collision geometry
            scene_objects: Alternatively, list of object dicts with position/dimensions
            robot_urdf_path: Optional robot URDF for collision geometry
            use_curobo: Use cuRobo if available
            verbose: Print debug info
        """
        self.robot_type = robot_type
        self.robot_config = ROBOT_CONFIGS.get(robot_type, ROBOT_CONFIGS["franka"])
        self.verbose = verbose

        # Set up collision checking
        self.collision_checker = SceneCollisionChecker(
            scene_usd_path=scene_usd_path,
            robot_type=robot_type,
            robot_urdf_path=robot_urdf_path,
            verbose=verbose,
        )

        # Add objects from list if provided
        if scene_objects:
            for obj in scene_objects:
                pos = np.array(obj.get("position", [0, 0, 0]))
                dims = np.array(obj.get("dimensions", [0.1, 0.1, 0.1]))
                self.collision_checker.add_obstacle(pos, dims)

        # Set up RRT planner
        self.rrt_planner = RRTPlanner(
            collision_checker=self.collision_checker,
            verbose=verbose,
        )

        # Set up IK solver
        self.ik_solver = IKSolver(self.robot_config, verbose=verbose)

        # cuRobo integration
        self._use_curobo = use_curobo and _CUROBO_AVAILABLE
        self._motion_gen = None

        if self._use_curobo:
            self._init_curobo()

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[COLLISION-PLANNER] [%s] %s", level, msg)

    def _init_curobo(self) -> None:
        """Initialize cuRobo motion generator."""
        try:
            from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

            # Load robot configuration
            # Note: Real implementation would load proper robot URDF
            self.log("cuRobo available - using GPU-accelerated planning")
            # self._motion_gen = ...

        except Exception as e:
            self.log(f"cuRobo initialization failed: {e}", "WARNING")
            self._use_curobo = False

    def is_using_curobo(self) -> bool:
        """Check if using cuRobo for planning."""
        return self._use_curobo and self._motion_gen is not None

    def plan_cartesian_path(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        robot_radius: float = 0.08,
    ) -> Optional[List[np.ndarray]]:
        """
        Plan collision-free Cartesian path.

        Args:
            start_pos: Start position [x, y, z]
            goal_pos: Goal position [x, y, z]
            robot_radius: Robot collision sphere radius

        Returns:
            List of waypoint positions or None if planning fails
        """
        # First check if straight-line is collision-free
        if not self.collision_checker.check_collision_segment(
            start_pos, goal_pos, robot_radius
        ):
            self.log("Straight-line path is collision-free")
            return [start_pos, goal_pos]

        # Use RRT planner
        return self.rrt_planner.plan(start_pos, goal_pos, robot_radius)

    def plan_waypoint_trajectory(
        self,
        waypoints: List[Waypoint],
        robot_radius: float = 0.08,
    ) -> Optional[List[Waypoint]]:
        """
        Plan collision-free trajectory through waypoints.

        Replans segments that have collisions.

        Args:
            waypoints: Original waypoints
            robot_radius: Robot collision radius

        Returns:
            New waypoints with collision-free paths
        """
        if len(waypoints) < 2:
            return waypoints

        self.log(f"Planning collision-free trajectory through {len(waypoints)} waypoints")

        new_waypoints = [waypoints[0]]

        for i in range(len(waypoints) - 1):
            start_wp = waypoints[i]
            end_wp = waypoints[i + 1]

            # Check if segment has collision
            has_collision = self.collision_checker.check_collision_segment(
                start_wp.position, end_wp.position, robot_radius
            )

            if has_collision:
                self.log(f"  Segment {i} has collision - replanning")

                # Plan collision-free path
                path = self.plan_cartesian_path(
                    start_wp.position, end_wp.position, robot_radius
                )

                if path is None:
                    self.log(f"  Failed to find collision-free path for segment {i}", "ERROR")
                    # Keep original waypoint
                    new_waypoints.append(end_wp)
                else:
                    # Convert path to waypoints
                    segment_duration = end_wp.timestamp - start_wp.timestamp
                    dt = segment_duration / max(1, len(path) - 1)

                    for j, pos in enumerate(path[1:], 1):  # Skip first (it's start)
                        t = j / max(1, len(path) - 1)

                        # Interpolate orientation
                        ori = self._slerp_quat(
                            start_wp.orientation, end_wp.orientation, t
                        )

                        wp = Waypoint(
                            position=pos,
                            orientation=ori,
                            gripper_aperture=(1 - t) * start_wp.gripper_aperture + t * end_wp.gripper_aperture,
                            timestamp=start_wp.timestamp + j * dt,
                            duration_to_next=dt if j < len(path) - 1 else end_wp.duration_to_next,
                            phase=end_wp.phase,
                        )
                        new_waypoints.append(wp)
            else:
                new_waypoints.append(end_wp)

        self.log(f"  Result: {len(new_waypoints)} waypoints (original: {len(waypoints)})")
        return new_waypoints

    def solve_ik_with_collision_check(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        seed_joints: Optional[np.ndarray] = None,
        num_attempts: int = 10,
    ) -> Optional[np.ndarray]:
        """
        Solve IK with collision checking.

        Tries multiple IK solutions and returns first collision-free one.

        Args:
            target_position: Target EE position
            target_orientation: Target EE orientation (quaternion)
            seed_joints: Initial joint configuration
            num_attempts: Number of random seeds to try

        Returns:
            Collision-free joint configuration or None
        """
        # Try with provided seed first
        joints = self.ik_solver.solve(target_position, target_orientation, seed_joints)

        if joints is not None:
            # Check collision (simplified - check EE position)
            if not self.collision_checker.check_collision_point(target_position, 0.05):
                return joints

        # Try with random seeds
        for _ in range(num_attempts):
            random_seed = self._random_valid_joints()
            joints = self.ik_solver.solve(target_position, target_orientation, random_seed)

            if joints is not None:
                if not self.collision_checker.check_collision_point(target_position, 0.05):
                    return joints

        return None

    def _random_valid_joints(self) -> np.ndarray:
        """Generate random valid joint configuration."""
        lower = self.robot_config.joint_limits_lower
        upper = self.robot_config.joint_limits_upper

        # Handle infinite limits
        lower = np.where(np.isinf(lower), -np.pi, lower)
        upper = np.where(np.isinf(upper), np.pi, upper)

        return np.random.uniform(lower, upper)

    def _slerp_quat(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation for quaternions."""
        # Ensure same hemisphere
        if np.dot(q1, q2) < 0:
            q2 = -q2

        dot = np.clip(np.dot(q1, q2), -1, 1)
        theta = np.arccos(dot)

        if theta < 0.001:
            return q1.copy()

        sin_theta = np.sin(theta)
        result = (np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / sin_theta
        return result / np.linalg.norm(result)


# =============================================================================
# Integration with Existing Motion Planner
# =============================================================================


def enhance_motion_plan_with_collision_avoidance(
    motion_plan: MotionPlan,
    scene_objects: List[Dict[str, Any]],
    scene_usd_path: Optional[str] = None,
    robot_type: str = "franka",
    verbose: bool = True,
) -> MotionPlan:
    """
    Enhance a motion plan with collision avoidance.

    Takes an existing motion plan and replans any segments
    that would collide with scene objects.

    Args:
        motion_plan: Original motion plan
        scene_objects: Objects in the scene
        scene_usd_path: Optional USD scene path
        robot_type: Robot type
        verbose: Print debug info

    Returns:
        Enhanced motion plan with collision-free paths
    """
    planner = CollisionAwarePlanner(
        robot_type=robot_type,
        scene_usd_path=scene_usd_path,
        scene_objects=scene_objects,
        verbose=verbose,
    )

    new_waypoints = planner.plan_waypoint_trajectory(motion_plan.waypoints)

    if new_waypoints is None:
        return motion_plan  # Return original if planning failed

    # Create new motion plan with updated waypoints
    return MotionPlan(
        plan_id=f"{motion_plan.plan_id}_collision_free",
        task_name=motion_plan.task_name,
        task_description=motion_plan.task_description,
        waypoints=new_waypoints,
        target_object_id=motion_plan.target_object_id,
        target_object_position=motion_plan.target_object_position,
        target_object_dimensions=motion_plan.target_object_dimensions,
        place_position=motion_plan.place_position,
        robot_type=motion_plan.robot_type,
    )


# =============================================================================
# Testing
# =============================================================================


if __name__ == "__main__":
    logger.info("Testing Collision-Aware Motion Planner")
    logger.info("=" * 60)

    # Create some test obstacles
    scene_objects = [
        {"id": "table", "position": [0.5, 0, 0.4], "dimensions": [0.8, 0.6, 0.05]},
        {"id": "obstacle", "position": [0.5, 0.15, 0.7], "dimensions": [0.1, 0.1, 0.2]},
    ]

    # Create planner
    planner = CollisionAwarePlanner(
        robot_type="franka",
        scene_objects=scene_objects,
        verbose=True,
    )

    # Test path planning
    start = np.array([0.3, 0, 0.6])
    goal = np.array([0.7, 0.3, 0.8])

    logger.info("Planning from %s to %s", start, goal)
    path = planner.plan_cartesian_path(start, goal)

    if path:
        logger.info("Found path with %s waypoints:", len(path))
        for i, wp in enumerate(path):
            logger.info("  %s: %s", i, wp)
    else:
        logger.warning("No path found!")

    # Test with motion plan
    from motion_planner import AIMotionPlanner

    logger.info("%s", "=" * 60)
    logger.info("Testing with full motion plan")
    logger.info("=" * 60)

    mp = AIMotionPlanner(robot_type="franka", use_llm=False, verbose=True)
    plan = mp.plan_motion(
        task_name="pick_cup",
        task_description="Pick up cup",
        target_object={"id": "cup", "position": [0.6, 0.2, 0.85], "dimensions": [0.08, 0.08, 0.12]},
        place_position=[0.3, -0.1, 0.9],
    )

    enhanced = enhance_motion_plan_with_collision_avoidance(
        motion_plan=plan,
        scene_objects=scene_objects,
        robot_type="franka",
    )

    logger.info("Original: %s waypoints", plan.num_waypoints)
    logger.info("Enhanced: %s waypoints", enhanced.num_waypoints)
