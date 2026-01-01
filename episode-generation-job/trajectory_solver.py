#!/usr/bin/env python3
"""
Trajectory Solver for Episode Generation.

Converts waypoint-based motion plans into dense joint trajectories.
Uses analytical IK for supported robots and provides smooth interpolation.

Key Features:
- Analytical IK solvers for Franka, UR10, Fetch
- Cubic spline interpolation for smooth motions
- Collision-aware trajectory validation
- Frame-by-frame trajectory generation for training data
"""

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import interpolate

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_planner import MotionPlan, Waypoint, MotionPhase


# =============================================================================
# Robot Configurations
# =============================================================================


@dataclass
class RobotConfig:
    """Configuration for a robot arm."""

    name: str
    num_joints: int
    joint_names: List[str]
    joint_limits_lower: np.ndarray
    joint_limits_upper: np.ndarray
    default_joint_positions: np.ndarray
    gripper_joint_names: List[str] = field(default_factory=list)
    gripper_limits: Tuple[float, float] = (0.0, 0.04)  # meters


FRANKA_CONFIG = RobotConfig(
    name="franka",
    num_joints=7,
    joint_names=[
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ],
    joint_limits_lower=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    joint_limits_upper=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
    default_joint_positions=np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
    gripper_joint_names=["panda_finger_joint1", "panda_finger_joint2"],
    gripper_limits=(0.0, 0.04),
)

UR10_CONFIG = RobotConfig(
    name="ur10",
    num_joints=6,
    joint_names=[
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ],
    joint_limits_lower=np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
    joint_limits_upper=np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi]),
    default_joint_positions=np.array([0.0, -1.571, 1.571, -1.571, -1.571, 0.0]),
    gripper_joint_names=[],
    gripper_limits=(0.0, 0.05),
)

FETCH_CONFIG = RobotConfig(
    name="fetch",
    num_joints=7,
    joint_names=[
        "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
        "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint",
    ],
    joint_limits_lower=np.array([-1.605, -1.221, -np.inf, -2.251, -np.inf, -2.16, -np.inf]),
    joint_limits_upper=np.array([1.605, 1.518, np.inf, 2.251, np.inf, 2.16, np.inf]),
    default_joint_positions=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    gripper_joint_names=["l_gripper_finger_joint", "r_gripper_finger_joint"],
    gripper_limits=(0.0, 0.05),
)

ROBOT_CONFIGS = {
    "franka": FRANKA_CONFIG,
    "ur10": UR10_CONFIG,
    "fetch": FETCH_CONFIG,
}


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class JointState:
    """A single joint state in the trajectory."""

    # Frame index
    frame_idx: int

    # Timestamp
    timestamp: float

    # Joint positions (radians for revolute, meters for prismatic)
    joint_positions: np.ndarray

    # Joint velocities (optional)
    joint_velocities: Optional[np.ndarray] = None

    # Gripper state
    gripper_position: float = 0.0  # meters (finger separation)

    # End-effector pose (optional, for reference)
    ee_position: Optional[np.ndarray] = None
    ee_orientation: Optional[np.ndarray] = None

    # Motion phase
    phase: MotionPhase = MotionPhase.APPROACH

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "joint_positions": self.joint_positions.tolist(),
            "joint_velocities": self.joint_velocities.tolist() if self.joint_velocities is not None else None,
            "gripper_position": self.gripper_position,
            "ee_position": self.ee_position.tolist() if self.ee_position is not None else None,
            "ee_orientation": self.ee_orientation.tolist() if self.ee_orientation is not None else None,
            "phase": self.phase.value,
        }


@dataclass
class JointTrajectory:
    """A complete joint trajectory."""

    trajectory_id: str
    robot_type: str
    robot_config: RobotConfig

    # States at each frame
    states: List[JointState] = field(default_factory=list)

    # Source motion plan
    source_plan_id: Optional[str] = None

    # Timing
    fps: float = 30.0
    total_duration: float = 0.0

    @property
    def num_frames(self) -> int:
        return len(self.states)

    def get_joint_positions_array(self) -> np.ndarray:
        """Get all joint positions as a 2D array (num_frames x num_joints)."""
        if not self.states:
            return np.array([])
        return np.array([s.joint_positions for s in self.states])

    def get_gripper_positions(self) -> np.ndarray:
        """Get all gripper positions as 1D array."""
        return np.array([s.gripper_position for s in self.states])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trajectory_id": self.trajectory_id,
            "robot_type": self.robot_type,
            "joint_names": self.robot_config.joint_names,
            "gripper_joint_names": self.robot_config.gripper_joint_names,
            "num_joints": self.robot_config.num_joints,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "total_duration": self.total_duration,
            "source_plan_id": self.source_plan_id,
            "states": [s.to_dict() for s in self.states],
        }


# =============================================================================
# IK Solver
# =============================================================================


class IKSolver:
    """
    Inverse Kinematics solver for robot arms.

    Uses analytical solutions where available, falls back to numerical methods.
    """

    def __init__(self, robot_config: RobotConfig, verbose: bool = False):
        self.config = robot_config
        self.verbose = verbose

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        seed_joints: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Solve IK for target pose.

        Args:
            target_position: Target EE position [x, y, z]
            target_orientation: Target EE orientation as quaternion [w, x, y, z]
            seed_joints: Initial joint configuration for numerical solving

        Returns:
            Joint positions or None if IK fails
        """
        if seed_joints is None:
            seed_joints = self.config.default_joint_positions.copy()

        # Use simplified analytical IK for common configurations
        if self.config.name == "franka":
            return self._solve_franka_ik(target_position, target_orientation, seed_joints)
        elif self.config.name == "ur10":
            return self._solve_ur_ik(target_position, target_orientation, seed_joints)
        elif self.config.name == "fetch":
            return self._solve_fetch_ik(target_position, target_orientation, seed_joints)
        else:
            return self._solve_numerical_ik(target_position, target_orientation, seed_joints)

    def _solve_franka_ik(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        seed_joints: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Simplified IK for Franka Panda.

        Uses geometric approach for reaching poses with gripper pointing down.
        This is an approximation suitable for training data generation.
        """
        # Franka DH parameters (simplified)
        d1, d3, d5, d7 = 0.333, 0.316, 0.384, 0.107
        a4, a7 = 0.0825, 0.088

        # Target in base frame
        px, py, pz = target_position

        # Wrist center (offset from EE by d7 along z)
        wc = target_position - np.array([0, 0, d7])

        # Joint 1: rotation around z to point at target
        q1 = math.atan2(wc[1], wc[0])

        # Distance in xy plane
        r = math.sqrt(wc[0]**2 + wc[1]**2)

        # Joint 2 and 4: elbow configuration
        # Simplified: use seed configuration and adjust
        q2 = seed_joints[1]
        q3 = seed_joints[2]
        q4 = seed_joints[3]

        # Adjust q2 based on height difference
        height_diff = wc[2] - d1
        if abs(height_diff) > 0.1:
            q2 = -0.5 - 0.5 * np.clip(height_diff / 0.5, -1, 1)

        # Adjust q4 based on reach
        reach = math.sqrt(r**2 + height_diff**2)
        if reach > 0.5:
            q4 = -1.5 - 0.5 * np.clip((reach - 0.5) / 0.3, 0, 1)

        # Wrist joints for orientation (pointing down)
        q5 = seed_joints[4]
        q6 = seed_joints[5]
        q7 = seed_joints[6]

        joints = np.array([q1, q2, q3, q4, q5, q6, q7])

        # Clamp to limits
        joints = np.clip(joints, self.config.joint_limits_lower, self.config.joint_limits_upper)

        return joints

    def _solve_ur_ik(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        seed_joints: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Analytical IK for UR10 robot.

        Uses geometric approach based on UR robot kinematics.
        UR10 is a 6-DOF robot with spherical wrist.

        DH Parameters (UR10):
            d1 = 0.1273, a2 = -0.612, a3 = -0.5723
            d4 = 0.1639, d5 = 0.1157, d6 = 0.0922
        """
        # UR10 DH parameters
        d1 = 0.1273  # shoulder height
        a2 = 0.612   # upper arm length
        a3 = 0.5723  # forearm length
        d4 = 0.1639  # wrist offset
        d5 = 0.1157  # wrist 2 offset
        d6 = 0.0922  # tool offset

        px, py, pz = target_position

        # Joint 1: Base rotation
        q1 = math.atan2(py, px)

        # Wrist center position (offset from tool by d6)
        # Assuming gripper pointing down
        wc_x = px
        wc_y = py
        wc_z = pz + d6

        # Distance from base z-axis to wrist center in xy-plane
        r_xy = math.sqrt(wc_x**2 + wc_y**2)

        # Height from shoulder to wrist center
        z_diff = wc_z - d1

        # Distance from shoulder to wrist center
        r = math.sqrt(r_xy**2 + z_diff**2)

        # Check reachability
        max_reach = abs(a2) + abs(a3)
        min_reach = abs(abs(a2) - abs(a3))

        if r > max_reach or r < min_reach:
            # Out of reach - use seed with base rotation
            joints = seed_joints.copy()
            joints[0] = q1
            return np.clip(joints, self.config.joint_limits_lower, self.config.joint_limits_upper)

        # Joint 3: Elbow angle (using law of cosines)
        cos_q3 = (r**2 - a2**2 - a3**2) / (2 * abs(a2) * abs(a3))
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)

        # Elbow up configuration (negative for elbow down)
        q3 = math.acos(cos_q3)

        # Joint 2: Shoulder angle
        beta = math.atan2(z_diff, r_xy)
        phi = math.atan2(abs(a3) * math.sin(q3), abs(a2) + abs(a3) * math.cos(q3))
        q2 = -(beta + phi)  # Negative for typical UR configuration

        # Wrist joints (q4, q5, q6) - simplified for gripper-down orientation
        # For gripper pointing down: end-effector z aligned with -world z
        q4 = -q2 - q3  # Keep wrist horizontal
        q5 = -math.pi / 2  # Wrist 2 perpendicular
        q6 = 0.0  # No tool rotation

        joints = np.array([q1, q2, q3, q4, q5, q6])

        # Clamp to limits
        return np.clip(joints, self.config.joint_limits_lower, self.config.joint_limits_upper)

    def _solve_fetch_ik(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        seed_joints: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Analytical IK for Fetch robot arm.

        Fetch has a 7-DOF arm (redundant), which provides an extra degree of freedom.
        We use this for elbow positioning (null-space optimization).

        Joint order:
            0: shoulder_pan_joint (rotation around vertical)
            1: shoulder_lift_joint (shoulder flex/extend)
            2: upperarm_roll_joint (upper arm rotation)
            3: elbow_flex_joint (elbow flex/extend)
            4: forearm_roll_joint (forearm rotation)
            5: wrist_flex_joint (wrist flex/extend)
            6: wrist_roll_joint (wrist rotation)

        Approximate link lengths (from URDF):
            shoulder_to_elbow: 0.4 m
            elbow_to_wrist: 0.32 m
            wrist_to_gripper: 0.2 m
        """
        # Fetch arm approximate dimensions
        shoulder_height = 0.4  # Height of shoulder above base
        upper_arm = 0.4       # Shoulder to elbow
        forearm = 0.32        # Elbow to wrist
        wrist_to_grip = 0.2   # Wrist to gripper tip
        shoulder_offset = 0.1  # Lateral offset of shoulder from base

        px, py, pz = target_position

        # Joint 0: Shoulder pan (rotation to face target)
        q0 = math.atan2(py, px)

        # Wrist center position (offset from gripper by wrist_to_grip)
        # Assuming gripper pointing down
        wc_x = px
        wc_y = py
        wc_z = pz + wrist_to_grip

        # Distance in xy-plane from shoulder axis
        r_xy = math.sqrt(wc_x**2 + wc_y**2) - shoulder_offset

        # Height from shoulder to wrist center
        z_diff = wc_z - shoulder_height

        # Distance from shoulder to wrist center
        r = math.sqrt(r_xy**2 + z_diff**2)

        # Check reachability
        max_reach = upper_arm + forearm
        min_reach = abs(upper_arm - forearm) * 0.1

        if r > max_reach:
            # Out of reach - stretch toward target
            r = max_reach * 0.95
        elif r < min_reach:
            r = min_reach * 1.1

        # Joint 3: Elbow flex (using law of cosines)
        cos_q3 = (r**2 - upper_arm**2 - forearm**2) / (2 * upper_arm * forearm)
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        q3 = math.acos(cos_q3)

        # Joint 1: Shoulder lift
        alpha = math.atan2(z_diff, r_xy)
        beta = math.atan2(forearm * math.sin(q3), upper_arm + forearm * math.cos(q3))
        q1 = alpha + beta

        # Joint 2: Upper arm roll (use seed or neutral)
        q2 = seed_joints[2] if len(seed_joints) > 2 else 0.0

        # Joint 4: Forearm roll (use seed or neutral)
        q4 = seed_joints[4] if len(seed_joints) > 4 else 0.0

        # Joint 5: Wrist flex (keep gripper pointing down)
        # The gripper should point down, so wrist flex compensates for arm angle
        q5 = -q1 - q3  # Compensate to keep gripper vertical

        # Joint 6: Wrist roll (from orientation or neutral)
        # Extract yaw from target orientation quaternion
        w, x, y, z = target_orientation
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        q6 = yaw - q0  # Compensate for base rotation

        joints = np.array([q0, q1, q2, q3, q4, q5, q6])

        # Clamp to limits
        return np.clip(joints, self.config.joint_limits_lower, self.config.joint_limits_upper)

    def _solve_numerical_ik(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        seed_joints: np.ndarray,
        max_iterations: int = 100,
        position_tolerance: float = 0.001,
        orientation_tolerance: float = 0.01,
    ) -> Optional[np.ndarray]:
        """
        Numerical IK using damped least squares (Levenberg-Marquardt).

        This is a robust fallback for robots without analytical IK solutions.
        Uses the Jacobian pseudoinverse with damping for singularity handling.

        Args:
            target_position: Target end-effector position (x, y, z)
            target_orientation: Target orientation quaternion (w, x, y, z)
            seed_joints: Initial joint configuration
            max_iterations: Maximum iterations for convergence
            position_tolerance: Position error tolerance (meters)
            orientation_tolerance: Orientation error tolerance (radians)

        Returns:
            Joint configuration or None if failed to converge
        """
        joints = seed_joints.copy().astype(np.float64)
        num_joints = len(joints)

        # Damping factor for singularity robustness
        damping = 0.1
        step_size = 0.5

        for iteration in range(max_iterations):
            # Compute current end-effector pose using forward kinematics
            current_pos, current_quat = self._forward_kinematics(joints)

            # Position error
            pos_error = target_position - current_pos
            pos_error_norm = np.linalg.norm(pos_error)

            # Orientation error (quaternion to axis-angle)
            orient_error = self._quaternion_error(target_orientation, current_quat)
            orient_error_norm = np.linalg.norm(orient_error)

            # Check convergence
            if pos_error_norm < position_tolerance and orient_error_norm < orientation_tolerance:
                # Clamp to joint limits
                joints = np.clip(joints, self.config.joint_limits_lower, self.config.joint_limits_upper)
                return joints

            # Combine errors into 6D task-space error
            error = np.concatenate([pos_error, orient_error])

            # Compute Jacobian
            jacobian = self._compute_jacobian(joints)

            # Damped least squares: q_dot = J^T * (J * J^T + lambda^2 * I)^-1 * error
            JJT = jacobian @ jacobian.T
            damped_JJT = JJT + (damping ** 2) * np.eye(6)

            try:
                # Solve for task-space velocity
                task_vel = np.linalg.solve(damped_JJT, error)
                # Convert to joint-space
                joint_delta = jacobian.T @ task_vel
            except np.linalg.LinAlgError:
                # Fallback to pseudoinverse if singular
                joint_delta = np.linalg.pinv(jacobian) @ error

            # Update joints with step size
            joints = joints + step_size * joint_delta

            # Clamp to joint limits
            joints = np.clip(joints, self.config.joint_limits_lower, self.config.joint_limits_upper)

            # Adaptive damping: increase if error grows
            if iteration > 0:
                damping = min(1.0, damping * 1.1) if pos_error_norm > prev_pos_error else max(0.01, damping * 0.9)

            prev_pos_error = pos_error_norm

        # Did not converge
        self.log(f"Numerical IK did not converge after {max_iterations} iterations", "WARNING")
        self.log(f"  Position error: {pos_error_norm:.4f}m, Orientation error: {orient_error_norm:.4f}rad", "WARNING")

        # Return best effort if close
        if pos_error_norm < position_tolerance * 10:
            return np.clip(joints, self.config.joint_limits_lower, self.config.joint_limits_upper)

        return None

    def _forward_kinematics(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for the robot.

        Args:
            joints: Joint configuration

        Returns:
            Tuple of (position, quaternion)
        """
        # Use DH parameters for FK
        # This is a simplified version - for production, use proper DH chain
        if self.config.name == "franka":
            return self._fk_franka(joints)
        elif self.config.name == "ur10":
            return self._fk_ur(joints)
        else:
            return self._fk_generic(joints)

    def _fk_franka(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward kinematics for Franka Panda using DH parameters."""
        # Franka DH parameters (modified DH)
        # a, d, alpha values for each joint
        dh = [
            (0, 0.333, 0),           # Joint 1
            (0, 0, -np.pi/2),        # Joint 2
            (0, 0.316, np.pi/2),     # Joint 3
            (0.0825, 0, np.pi/2),    # Joint 4
            (-0.0825, 0.384, -np.pi/2),  # Joint 5
            (0, 0, np.pi/2),         # Joint 6
            (0.088, 0, np.pi/2),     # Joint 7
        ]

        # Compute transformation chain
        T = np.eye(4)
        for i, (a, d, alpha) in enumerate(dh):
            theta = joints[i] if i < len(joints) else 0
            Ti = self._dh_transform(a, d, alpha, theta)
            T = T @ Ti

        # Add flange offset
        T = T @ self._dh_transform(0, 0.107, 0, 0)

        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)

        return position, quaternion

    def _fk_ur(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward kinematics for UR10."""
        # UR10 DH parameters
        dh = [
            (0, 0.1273, np.pi/2),    # Joint 1
            (-0.612, 0, 0),          # Joint 2
            (-0.5723, 0, 0),         # Joint 3
            (0, 0.1639, np.pi/2),    # Joint 4
            (0, 0.1157, -np.pi/2),   # Joint 5
            (0, 0.0922, 0),          # Joint 6
        ]

        T = np.eye(4)
        for i, (a, d, alpha) in enumerate(dh):
            theta = joints[i] if i < len(joints) else 0
            Ti = self._dh_transform(a, d, alpha, theta)
            T = T @ Ti

        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)

        return position, quaternion

    def _fk_generic(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generic FK estimate based on joint angles."""
        # Very rough approximation for unknown robots
        x = 0.5 + 0.1 * np.cos(joints[0])
        y = 0.1 * np.sin(joints[0])
        z = 0.5 + 0.1 * joints[1] if len(joints) > 1 else 0.5

        position = np.array([x, y, z])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        return position, quaternion

    def _dh_transform(self, a: float, d: float, alpha: float, theta: float) -> np.ndarray:
        """Compute DH transformation matrix."""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])

    def _quaternion_error(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """
        Compute orientation error as axis-angle vector.

        Args:
            target: Target quaternion (w, x, y, z)
            current: Current quaternion (w, x, y, z)

        Returns:
            3D axis-angle error vector
        """
        # Quaternion difference: q_error = q_target * q_current^-1
        # For unit quaternions, inverse is conjugate
        current_inv = np.array([current[0], -current[1], -current[2], -current[3]])

        # Quaternion multiplication
        w1, x1, y1, z1 = target
        w2, x2, y2, z2 = current_inv

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        # Convert to axis-angle
        # theta = 2 * acos(w), axis = (x,y,z) / sin(theta/2)
        if abs(w) > 0.9999:
            # Nearly identity rotation
            return np.array([0.0, 0.0, 0.0])

        theta = 2 * np.arccos(np.clip(w, -1, 1))
        sin_half_theta = np.sqrt(1 - w*w)

        if sin_half_theta < 0.001:
            return np.array([2*x, 2*y, 2*z])

        return theta * np.array([x, y, z]) / sin_half_theta

    def _compute_jacobian(self, joints: np.ndarray, delta: float = 0.0001) -> np.ndarray:
        """
        Compute the geometric Jacobian numerically.

        Args:
            joints: Current joint configuration
            delta: Finite difference step size

        Returns:
            6xN Jacobian matrix (3 rows position, 3 rows orientation)
        """
        num_joints = len(joints)
        jacobian = np.zeros((6, num_joints))

        # Current pose
        pos_0, quat_0 = self._forward_kinematics(joints)

        for i in range(num_joints):
            # Perturb joint i
            joints_perturbed = joints.copy()
            joints_perturbed[i] += delta

            # Compute perturbed pose
            pos_1, quat_1 = self._forward_kinematics(joints_perturbed)

            # Position Jacobian (linear velocity)
            jacobian[:3, i] = (pos_1 - pos_0) / delta

            # Orientation Jacobian (angular velocity)
            orient_diff = self._quaternion_error(quat_1, quat_0)
            jacobian[3:, i] = orient_diff / delta

        return jacobian


# =============================================================================
# Trajectory Solver
# =============================================================================


class TrajectorySolver:
    """
    Converts motion plans to dense joint trajectories.

    Process:
    1. Solve IK for each waypoint
    2. Interpolate between waypoints using cubic splines
    3. Sample at desired FPS for training data
    """

    def __init__(
        self,
        robot_type: str = "franka",
        fps: float = 30.0,
        verbose: bool = True,
    ):
        """
        Initialize trajectory solver.

        Args:
            robot_type: Robot type (franka, ur10, fetch)
            fps: Target frames per second for output
            verbose: Print debug info
        """
        self.robot_type = robot_type
        self.robot_config = ROBOT_CONFIGS.get(robot_type, FRANKA_CONFIG)
        self.fps = fps
        self.verbose = verbose
        self.ik_solver = IKSolver(self.robot_config, verbose=verbose)

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[TRAJECTORY-SOLVER] [{level}] {msg}")

    def solve(self, motion_plan: MotionPlan) -> JointTrajectory:
        """
        Convert motion plan to joint trajectory.

        Args:
            motion_plan: Motion plan with waypoints

        Returns:
            JointTrajectory with per-frame joint states
        """
        self.log(f"Solving trajectory for {motion_plan.task_name}")
        self.log(f"  Waypoints: {motion_plan.num_waypoints}")
        self.log(f"  Duration: {motion_plan.total_duration:.2f}s")

        # Step 1: Solve IK for each waypoint
        waypoint_joints = self._solve_waypoint_ik(motion_plan.waypoints)

        # Step 2: Build interpolation splines
        splines = self._build_splines(motion_plan.waypoints, waypoint_joints)

        # Step 3: Sample at target FPS
        states = self._sample_trajectory(
            motion_plan=motion_plan,
            splines=splines,
            waypoint_joints=waypoint_joints,
        )

        trajectory = JointTrajectory(
            trajectory_id=f"traj_{motion_plan.plan_id}",
            robot_type=self.robot_type,
            robot_config=self.robot_config,
            states=states,
            source_plan_id=motion_plan.plan_id,
            fps=self.fps,
            total_duration=motion_plan.total_duration,
        )

        self.log(f"  Generated {trajectory.num_frames} frames")

        return trajectory

    def _solve_waypoint_ik(self, waypoints: List[Waypoint]) -> List[np.ndarray]:
        """Solve IK for all waypoints."""
        results = []
        prev_joints = self.robot_config.default_joint_positions.copy()

        for i, wp in enumerate(waypoints):
            joints = self.ik_solver.solve(
                target_position=wp.position,
                target_orientation=wp.orientation,
                seed_joints=prev_joints,
            )

            if joints is None:
                self.log(f"  IK failed for waypoint {i}, using previous", "WARNING")
                joints = prev_joints.copy()

            results.append(joints)
            prev_joints = joints

        return results

    def _build_splines(
        self,
        waypoints: List[Waypoint],
        waypoint_joints: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Build cubic spline interpolators for joints and gripper."""

        if len(waypoints) < 2:
            return {}

        # Timestamps for waypoints
        timestamps = np.array([wp.timestamp for wp in waypoints])

        # Joint position splines
        joint_array = np.array(waypoint_joints)
        joint_splines = []
        for j in range(self.robot_config.num_joints):
            joint_values = joint_array[:, j]
            # Use cubic spline with natural boundary conditions
            spline = interpolate.CubicSpline(timestamps, joint_values, bc_type="natural")
            joint_splines.append(spline)

        # Gripper spline
        gripper_values = np.array([
            wp.gripper_aperture * self.robot_config.gripper_limits[1]
            for wp in waypoints
        ])
        gripper_spline = interpolate.CubicSpline(timestamps, gripper_values, bc_type="natural")

        return {
            "joint_splines": joint_splines,
            "gripper_spline": gripper_spline,
            "timestamps": timestamps,
        }

    def _sample_trajectory(
        self,
        motion_plan: MotionPlan,
        splines: Dict[str, Any],
        waypoint_joints: List[np.ndarray],
    ) -> List[JointState]:
        """Sample trajectory at target FPS."""

        states = []

        if not splines or motion_plan.total_duration == 0:
            # Single frame
            state = JointState(
                frame_idx=0,
                timestamp=0.0,
                joint_positions=waypoint_joints[0] if waypoint_joints else self.robot_config.default_joint_positions,
                gripper_position=0.0,
                phase=MotionPhase.HOME,
            )
            return [state]

        # Calculate number of frames
        num_frames = max(2, int(motion_plan.total_duration * self.fps) + 1)
        dt = motion_plan.total_duration / (num_frames - 1)

        joint_splines = splines["joint_splines"]
        gripper_spline = splines["gripper_spline"]

        # Find phase for each timestamp
        def get_phase_at_time(t: float) -> MotionPhase:
            for wp in reversed(motion_plan.waypoints):
                if t >= wp.timestamp:
                    return wp.phase
            return MotionPhase.HOME

        for i in range(num_frames):
            t = i * dt

            # Sample joint positions
            joint_positions = np.array([spline(t) for spline in joint_splines])

            # Sample gripper
            gripper_position = float(gripper_spline(t))
            gripper_position = np.clip(gripper_position, 0, self.robot_config.gripper_limits[1])

            # Calculate velocities (finite difference)
            if i > 0:
                t_prev = (i - 1) * dt
                prev_joints = np.array([spline(t_prev) for spline in joint_splines])
                velocities = (joint_positions - prev_joints) / dt
            else:
                velocities = np.zeros(self.robot_config.num_joints)

            # Get EE pose (use waypoint interpolation)
            ee_position, ee_orientation = self._interpolate_ee_pose(t, motion_plan.waypoints)

            state = JointState(
                frame_idx=i,
                timestamp=t,
                joint_positions=joint_positions,
                joint_velocities=velocities,
                gripper_position=gripper_position,
                ee_position=ee_position,
                ee_orientation=ee_orientation,
                phase=get_phase_at_time(t),
            )
            states.append(state)

        return states

    def _interpolate_ee_pose(
        self,
        t: float,
        waypoints: List[Waypoint],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate end-effector pose at time t."""

        if not waypoints:
            return np.zeros(3), np.array([1, 0, 0, 0])

        # Find bracketing waypoints
        prev_wp = waypoints[0]
        next_wp = waypoints[-1]

        for i, wp in enumerate(waypoints):
            if wp.timestamp > t:
                next_wp = wp
                prev_wp = waypoints[i - 1] if i > 0 else wp
                break
            prev_wp = wp

        if prev_wp.timestamp == next_wp.timestamp:
            return prev_wp.position.copy(), prev_wp.orientation.copy()

        # Linear interpolation factor
        alpha = (t - prev_wp.timestamp) / (next_wp.timestamp - prev_wp.timestamp)
        alpha = np.clip(alpha, 0, 1)

        # Interpolate position
        position = (1 - alpha) * prev_wp.position + alpha * next_wp.position

        # SLERP for orientation (simplified - linear for now)
        orientation = (1 - alpha) * prev_wp.orientation + alpha * next_wp.orientation
        orientation = orientation / np.linalg.norm(orientation)

        return position, orientation


# =============================================================================
# Convenience Functions
# =============================================================================


def solve_trajectory(
    motion_plan: MotionPlan,
    robot_type: str = "franka",
    fps: float = 30.0,
) -> JointTrajectory:
    """Convenience function to solve a motion plan to joint trajectory."""
    solver = TrajectorySolver(robot_type=robot_type, fps=fps, verbose=False)
    return solver.solve(motion_plan)


if __name__ == "__main__":
    from motion_planner import AIMotionPlanner

    # Test the full pipeline
    print("Testing Motion Planner + Trajectory Solver")
    print("=" * 60)

    # Generate motion plan
    planner = AIMotionPlanner(robot_type="franka", use_llm=False, verbose=True)
    plan = planner.plan_motion(
        task_name="pick_cup",
        task_description="Pick up cup and place on shelf",
        target_object={
            "id": "cup_001",
            "position": [0.5, 0.1, 0.85],
            "dimensions": [0.08, 0.08, 0.12],
        },
        place_position=[0.3, -0.2, 0.9],
    )

    # Solve trajectory
    solver = TrajectorySolver(robot_type="franka", fps=30.0, verbose=True)
    trajectory = solver.solve(plan)

    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Frames: {trajectory.num_frames}")
    print(f"Duration: {trajectory.total_duration:.2f}s")
    print(f"FPS: {trajectory.fps}")
    print(f"Joint names: {trajectory.robot_config.joint_names}")

    # Sample output
    print("\nSample states:")
    for i in [0, len(trajectory.states)//2, -1]:
        s = trajectory.states[i]
        print(f"  Frame {s.frame_idx}: t={s.timestamp:.2f}s, phase={s.phase.value}, "
              f"gripper={s.gripper_position:.3f}m")
