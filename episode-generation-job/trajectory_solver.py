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
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.config.production_mode import resolve_production_mode

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

# =============================================================================
# Additional Robot Configurations (Multi-Robot Support)
# =============================================================================

UR5_CONFIG = RobotConfig(
    name="ur5",
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

UR5E_CONFIG = RobotConfig(
    name="ur5e",
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

UR10E_CONFIG = RobotConfig(
    name="ur10e",
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

KUKA_IIWA_CONFIG = RobotConfig(
    name="kuka_iiwa",
    num_joints=7,
    joint_names=[
        "iiwa_joint_1", "iiwa_joint_2", "iiwa_joint_3", "iiwa_joint_4",
        "iiwa_joint_5", "iiwa_joint_6", "iiwa_joint_7",
    ],
    joint_limits_lower=np.array([-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054]),
    joint_limits_upper=np.array([2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054]),
    default_joint_positions=np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0]),
    gripper_joint_names=[],
    gripper_limits=(0.0, 0.04),
)

KUKA_IIWA14_CONFIG = RobotConfig(
    name="kuka_iiwa14",
    num_joints=7,
    joint_names=[
        "iiwa_joint_1", "iiwa_joint_2", "iiwa_joint_3", "iiwa_joint_4",
        "iiwa_joint_5", "iiwa_joint_6", "iiwa_joint_7",
    ],
    joint_limits_lower=np.array([-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054]),
    joint_limits_upper=np.array([2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054]),
    default_joint_positions=np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0]),
    gripper_joint_names=[],
    gripper_limits=(0.0, 0.04),
)

SAWYER_CONFIG = RobotConfig(
    name="sawyer",
    num_joints=7,
    joint_names=[
        "right_j0", "right_j1", "right_j2", "right_j3",
        "right_j4", "right_j5", "right_j6",
    ],
    joint_limits_lower=np.array([-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124]),
    joint_limits_upper=np.array([3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124]),
    default_joint_positions=np.array([0.0, -1.18, 0.0, 2.18, 0.0, 0.57, 3.14]),
    gripper_joint_names=["right_gripper_l_finger_joint", "right_gripper_r_finger_joint"],
    gripper_limits=(0.0, 0.041),
)

BAXTER_LEFT_CONFIG = RobotConfig(
    name="baxter_left",
    num_joints=7,
    joint_names=[
        "left_s0", "left_s1", "left_e0", "left_e1",
        "left_w0", "left_w1", "left_w2",
    ],
    joint_limits_lower=np.array([-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.5707, -3.059]),
    joint_limits_upper=np.array([1.7016, 1.047, 3.0541, 2.618, 3.059, 2.094, 3.059]),
    default_joint_positions=np.array([0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0]),
    gripper_joint_names=["left_gripper_l_finger_joint", "left_gripper_r_finger_joint"],
    gripper_limits=(0.0, 0.04),
)

BAXTER_RIGHT_CONFIG = RobotConfig(
    name="baxter_right",
    num_joints=7,
    joint_names=[
        "right_s0", "right_s1", "right_e0", "right_e1",
        "right_w0", "right_w1", "right_w2",
    ],
    joint_limits_lower=np.array([-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.5707, -3.059]),
    joint_limits_upper=np.array([1.7016, 1.047, 3.0541, 2.618, 3.059, 2.094, 3.059]),
    default_joint_positions=np.array([0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0]),
    gripper_joint_names=["right_gripper_l_finger_joint", "right_gripper_r_finger_joint"],
    gripper_limits=(0.0, 0.04),
)

KINOVA_GEN3_CONFIG = RobotConfig(
    name="kinova_gen3",
    num_joints=7,
    joint_names=[
        "joint_1", "joint_2", "joint_3", "joint_4",
        "joint_5", "joint_6", "joint_7",
    ],
    joint_limits_lower=np.array([-np.inf, -2.41, -np.inf, -2.66, -np.inf, -2.23, -np.inf]),
    joint_limits_upper=np.array([np.inf, 2.41, np.inf, 2.66, np.inf, 2.23, np.inf]),
    default_joint_positions=np.array([0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]),
    gripper_joint_names=["finger_joint", "right_outer_knuckle_joint"],
    gripper_limits=(0.0, 0.04),
)

XARM7_CONFIG = RobotConfig(
    name="xarm7",
    num_joints=7,
    joint_names=[
        "joint1", "joint2", "joint3", "joint4",
        "joint5", "joint6", "joint7",
    ],
    joint_limits_lower=np.array([-6.28, -2.059, -6.28, -0.19, -6.28, -1.69, -6.28]),
    joint_limits_upper=np.array([6.28, 2.0944, 6.28, 3.927, 6.28, 3.14, 6.28]),
    default_joint_positions=np.array([0.0, 0.0, 0.0, 1.571, 0.0, 1.571, 0.0]),
    gripper_joint_names=["drive_joint"],
    gripper_limits=(0.0, 0.04),
)

XARM6_CONFIG = RobotConfig(
    name="xarm6",
    num_joints=6,
    joint_names=[
        "joint1", "joint2", "joint3", "joint4",
        "joint5", "joint6",
    ],
    joint_limits_lower=np.array([-6.28, -2.059, -6.28, -0.19, -6.28, -6.28]),
    joint_limits_upper=np.array([6.28, 2.0944, 6.28, 3.927, 6.28, 6.28]),
    default_joint_positions=np.array([0.0, 0.0, 0.0, 1.571, 0.0, 0.0]),
    gripper_joint_names=["drive_joint"],
    gripper_limits=(0.0, 0.04),
)

FANUC_LR_MATE_CONFIG = RobotConfig(
    name="fanuc_lr_mate",
    num_joints=6,
    joint_names=[
        "joint_1", "joint_2", "joint_3", "joint_4",
        "joint_5", "joint_6",
    ],
    joint_limits_lower=np.array([-2.96, -1.57, -2.36, -3.49, -2.09, -6.28]),
    joint_limits_upper=np.array([2.96, 2.79, 4.54, 3.49, 2.09, 6.28]),
    default_joint_positions=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    gripper_joint_names=[],
    gripper_limits=(0.0, 0.05),
)

ABB_IRB120_CONFIG = RobotConfig(
    name="abb_irb120",
    num_joints=6,
    joint_names=[
        "joint_1", "joint_2", "joint_3", "joint_4",
        "joint_5", "joint_6",
    ],
    joint_limits_lower=np.array([-2.87, -1.92, -1.22, -5.03, -2.09, -6.98]),
    joint_limits_upper=np.array([2.87, 1.57, 2.88, 5.03, 2.09, 6.98]),
    default_joint_positions=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    gripper_joint_names=[],
    gripper_limits=(0.0, 0.04),
)

YASKAWA_GP8_CONFIG = RobotConfig(
    name="yaskawa_gp8",
    num_joints=6,
    joint_names=[
        "joint_1_s", "joint_2_l", "joint_3_u", "joint_4_r",
        "joint_5_b", "joint_6_t",
    ],
    joint_limits_lower=np.array([-2.96, -1.05, -1.22, -3.32, -2.18, -6.28]),
    joint_limits_upper=np.array([2.96, 2.79, 2.44, 3.32, 2.18, 6.28]),
    default_joint_positions=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    gripper_joint_names=[],
    gripper_limits=(0.0, 0.05),
)

# Mobile manipulators
STRETCH_CONFIG = RobotConfig(
    name="stretch",
    num_joints=4,  # Arm joints only (excluding mobile base)
    joint_names=[
        "joint_lift", "joint_arm_l0", "joint_arm_l1", "joint_wrist_yaw",
    ],
    joint_limits_lower=np.array([0.0, 0.0, 0.0, -1.38]),
    joint_limits_upper=np.array([1.1, 0.13, 0.13, 4.58]),
    default_joint_positions=np.array([0.5, 0.0, 0.0, 0.0]),
    gripper_joint_names=["joint_gripper_finger_left", "joint_gripper_finger_right"],
    gripper_limits=(0.0, 0.05),
)

TIAGO_CONFIG = RobotConfig(
    name="tiago",
    num_joints=7,
    joint_names=[
        "arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint",
        "arm_5_joint", "arm_6_joint", "arm_7_joint",
    ],
    joint_limits_lower=np.array([0.0, -1.5, -3.46, -0.32, -2.07, -1.39, -2.07]),
    joint_limits_upper=np.array([2.68, 1.02, 1.57, 2.29, 2.07, 1.39, 2.07]),
    default_joint_positions=np.array([0.2, -1.34, -0.2, 1.94, -1.57, 1.37, 0.0]),
    gripper_joint_names=["gripper_left_finger_joint", "gripper_right_finger_joint"],
    gripper_limits=(0.0, 0.044),
)

# Humanoids (arm joints only)
G1_ARM_JOINT_LIMITS_LOWER = np.array([-2.9, -1.6, -2.9, -2.6, -2.9, -1.7, -2.9])
G1_ARM_JOINT_LIMITS_UPPER = np.array([2.9, 1.6, 2.9, 2.6, 2.9, 1.7, 2.9])
G1_RIGHT_ARM_DEFAULT = np.array([0.0, -0.2, 0.0, -1.2, 0.0, 0.6, 0.0])
G1_LEFT_ARM_DEFAULT = np.array([0.0, 0.2, 0.0, -1.2, 0.0, -0.6, 0.0])

G1_RIGHT_ARM_CONFIG = RobotConfig(
    name="g1_right_arm",
    num_joints=7,
    joint_names=[
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
    ],
    joint_limits_lower=G1_ARM_JOINT_LIMITS_LOWER,
    joint_limits_upper=G1_ARM_JOINT_LIMITS_UPPER,
    default_joint_positions=G1_RIGHT_ARM_DEFAULT,
    gripper_joint_names=["right_gripper_finger_joint1", "right_gripper_finger_joint2"],
    gripper_limits=(0.0, 0.08),
)

G1_LEFT_ARM_CONFIG = RobotConfig(
    name="g1_left_arm",
    num_joints=7,
    joint_names=[
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
    ],
    joint_limits_lower=G1_ARM_JOINT_LIMITS_LOWER,
    joint_limits_upper=G1_ARM_JOINT_LIMITS_UPPER,
    default_joint_positions=G1_LEFT_ARM_DEFAULT,
    gripper_joint_names=["left_gripper_finger_joint1", "left_gripper_finger_joint2"],
    gripper_limits=(0.0, 0.08),
)

G1_CONFIG = RobotConfig(
    name="g1",
    num_joints=7,
    joint_names=G1_RIGHT_ARM_CONFIG.joint_names,
    joint_limits_lower=G1_ARM_JOINT_LIMITS_LOWER,
    joint_limits_upper=G1_ARM_JOINT_LIMITS_UPPER,
    default_joint_positions=G1_RIGHT_ARM_DEFAULT,
    gripper_joint_names=G1_RIGHT_ARM_CONFIG.gripper_joint_names,
    gripper_limits=G1_RIGHT_ARM_CONFIG.gripper_limits,
)

# =============================================================================
# Robot Configuration Registry
# =============================================================================

ROBOT_CONFIGS = {
    # Franka Robotics
    "franka": FRANKA_CONFIG,
    "franka_panda": FRANKA_CONFIG,
    "panda": FRANKA_CONFIG,

    # Universal Robots
    "ur5": UR5_CONFIG,
    "ur5e": UR5E_CONFIG,
    "ur10": UR10_CONFIG,
    "ur10e": UR10E_CONFIG,

    # KUKA
    "kuka_iiwa": KUKA_IIWA_CONFIG,
    "kuka_iiwa7": KUKA_IIWA_CONFIG,
    "kuka_iiwa14": KUKA_IIWA14_CONFIG,
    "iiwa": KUKA_IIWA_CONFIG,
    "iiwa7": KUKA_IIWA_CONFIG,
    "iiwa14": KUKA_IIWA14_CONFIG,

    # Rethink Robotics
    "sawyer": SAWYER_CONFIG,
    "baxter_left": BAXTER_LEFT_CONFIG,
    "baxter_right": BAXTER_RIGHT_CONFIG,

    # Kinova
    "kinova_gen3": KINOVA_GEN3_CONFIG,
    "kinova_gen3_7dof": KINOVA_GEN3_CONFIG,
    "gen3": KINOVA_GEN3_CONFIG,

    # UFACTORY
    "xarm7": XARM7_CONFIG,
    "xarm6": XARM6_CONFIG,

    # Fanuc
    "fanuc_lr_mate": FANUC_LR_MATE_CONFIG,
    "lr_mate": FANUC_LR_MATE_CONFIG,

    # ABB
    "abb_irb120": ABB_IRB120_CONFIG,
    "irb120": ABB_IRB120_CONFIG,

    # Yaskawa
    "yaskawa_gp8": YASKAWA_GP8_CONFIG,
    "gp8": YASKAWA_GP8_CONFIG,

    # Mobile Manipulators
    "fetch": FETCH_CONFIG,
    "stretch": STRETCH_CONFIG,
    "hello_robot_stretch": STRETCH_CONFIG,
    "tiago": TIAGO_CONFIG,

    # Humanoids (arm-only control)
    "g1": G1_CONFIG,
    "g1_right_arm": G1_RIGHT_ARM_CONFIG,
    "g1_left_arm": G1_LEFT_ARM_CONFIG,
}


def get_robot_config(robot_type: str) -> RobotConfig:
    """
    Get robot configuration by name.

    Args:
        robot_type: Robot name (case-insensitive)

    Returns:
        RobotConfig for the specified robot

    Raises:
        ValueError: If robot type is not supported
    """
    robot_key = robot_type.lower().replace("-", "_").replace(" ", "_")

    if robot_key not in ROBOT_CONFIGS:
        available = sorted(set(ROBOT_CONFIGS.keys()))
        raise ValueError(
            f"Unknown robot type: '{robot_type}'. "
            f"Supported robots: {available}"
        )

    return ROBOT_CONFIGS[robot_key]


def list_supported_robots() -> List[str]:
    """Get list of supported robot types."""
    # Return unique robot names (removing aliases)
    unique_robots = set()
    for config in ROBOT_CONFIGS.values():
        unique_robots.add(config.name)
    return sorted(unique_robots)


def get_robot_info(robot_type: str) -> Dict[str, Any]:
    """
    Get detailed information about a robot.

    Args:
        robot_type: Robot name

    Returns:
        Dictionary with robot specifications
    """
    config = get_robot_config(robot_type)

    return {
        "name": config.name,
        "num_joints": config.num_joints,
        "joint_names": config.joint_names,
        "has_gripper": len(config.gripper_joint_names) > 0,
        "gripper_joints": config.gripper_joint_names,
        "gripper_max_width": config.gripper_limits[1],
        "joint_limits": {
            "lower": config.joint_limits_lower.tolist(),
            "upper": config.joint_limits_upper.tolist(),
        },
        "default_pose": config.default_joint_positions.tolist(),
    }


def robot_config_to_dict(config: "RobotConfig", urdf_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Serialize a RobotConfig to a JSON-safe dict for embodiment manifests.

    This is the canonical serialization for cross-embodiment metadata. Use this
    to write embodiment identity into HDF5 attributes and sidecar JSON files so
    that datasets are self-describing and usable for cross-embodiment training.
    """
    return {
        "robot_type": config.name,
        "num_joints": config.num_joints,
        "joint_names": list(config.joint_names),
        "joint_limits": {
            "lower": config.joint_limits_lower.tolist(),
            "upper": config.joint_limits_upper.tolist(),
        },
        "default_joint_positions": config.default_joint_positions.tolist(),
        "gripper_joint_names": list(config.gripper_joint_names),
        "gripper_limits": list(config.gripper_limits),
        "urdf_path": urdf_path or "",
    }


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class IMUReading:
    """
    IMU sensor reading for mobile robots.

    Contains accelerometer, gyroscope, and orientation data.
    Important for Fetch, TIAGo, Spot and other mobile manipulation platforms.
    """

    # Linear acceleration (m/s^2) in robot base frame
    linear_acceleration: np.ndarray  # [ax, ay, az]

    # Angular velocity (rad/s) in robot base frame
    angular_velocity: np.ndarray  # [wx, wy, wz]

    # Orientation quaternion (from IMU fusion) [w, x, y, z]
    orientation: Optional[np.ndarray] = None

    # Timestamp (may differ from main trajectory if IMU runs at higher rate)
    timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "linear_acceleration": self.linear_acceleration.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
        }
        if self.orientation is not None:
            result["orientation"] = self.orientation.tolist()
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IMUReading":
        """Deserialize from dictionary."""
        return cls(
            linear_acceleration=np.array(data["linear_acceleration"]),
            angular_velocity=np.array(data["angular_velocity"]),
            orientation=np.array(data["orientation"]) if "orientation" in data else None,
            timestamp=data.get("timestamp"),
        )


@dataclass
class EEWrench:
    """
    End-effector wrench (force/torque) reading.

    Critical for contact-rich manipulation tasks and sim-to-real.
    """

    # Force in end-effector frame (Newtons)
    force: np.ndarray  # [fx, fy, fz]

    # Torque in end-effector frame (Newton-meters)
    torque: np.ndarray  # [tx, ty, tz]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "force": self.force.tolist(),
            "torque": self.torque.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EEWrench":
        """Deserialize from dictionary."""
        return cls(
            force=np.array(data["force"]),
            torque=np.array(data["torque"]),
        )


@dataclass
class JointState:
    """
    A single joint state in the trajectory.

    Now includes full dynamics data (torques, efforts, accelerations)
    for contact-rich manipulation and sim-to-real transfer.
    """

    # Frame index
    frame_idx: int

    # Timestamp
    timestamp: float

    # Joint positions (radians for revolute, meters for prismatic)
    joint_positions: np.ndarray

    # Joint velocities (optional)
    joint_velocities: Optional[np.ndarray] = None

    # =========================================================================
    # NEW: Joint dynamics (P1 - Important for contact-rich tasks)
    # =========================================================================

    # Joint torques from physics simulation (Newton-meters)
    joint_torques: Optional[np.ndarray] = None

    # Commanded joint efforts (what was commanded, not what was achieved)
    joint_efforts: Optional[np.ndarray] = None

    # Joint accelerations (derived from velocities, or from physics)
    joint_accelerations: Optional[np.ndarray] = None

    # =========================================================================
    # Gripper state
    # =========================================================================

    # Gripper position (meters, finger separation)
    gripper_position: float = 0.0

    # Gripper force (Newtons, grasp force)
    gripper_force: Optional[float] = None

    # =========================================================================
    # End-effector state
    # =========================================================================

    # End-effector pose (optional, for reference)
    ee_position: Optional[np.ndarray] = None
    ee_orientation: Optional[np.ndarray] = None  # quaternion [w, x, y, z]

    # End-effector velocity (m/s for linear, rad/s for angular)
    ee_velocity: Optional[np.ndarray] = None  # [vx, vy, vz, wx, wy, wz]

    # End-effector wrench (force/torque)
    ee_wrench: Optional[EEWrench] = None

    # =========================================================================
    # Motion phase (for skill segmentation)
    # =========================================================================

    phase: MotionPhase = MotionPhase.APPROACH

    # =========================================================================
    # IMU data (for mobile robots)
    # =========================================================================

    imu: Optional[IMUReading] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "joint_positions": self.joint_positions.tolist(),
            "joint_velocities": self.joint_velocities.tolist() if self.joint_velocities is not None else None,
            # NEW: Joint dynamics
            "joint_torques": self.joint_torques.tolist() if self.joint_torques is not None else None,
            "joint_efforts": self.joint_efforts.tolist() if self.joint_efforts is not None else None,
            "joint_accelerations": self.joint_accelerations.tolist() if self.joint_accelerations is not None else None,
            # Gripper
            "gripper_position": self.gripper_position,
            "gripper_force": self.gripper_force,
            # End-effector
            "ee_position": self.ee_position.tolist() if self.ee_position is not None else None,
            "ee_orientation": self.ee_orientation.tolist() if self.ee_orientation is not None else None,
            "ee_velocity": self.ee_velocity.tolist() if self.ee_velocity is not None else None,
            "ee_wrench": self.ee_wrench.to_dict() if self.ee_wrench is not None else None,
            # Phase (for skill segmentation)
            "phase": self.phase.value,
            # IMU (for mobile robots)
            "imu": self.imu.to_dict() if self.imu is not None else None,
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

    # Provenance metadata
    provenance: Dict[str, str] = field(default_factory=lambda: {
        "action_source": "cubic_spline_interpolation",
        "velocity_source": "finite_difference",
    })

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
            "provenance": self.provenance,
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

    def log(self, msg: str, level: str = "INFO") -> None:
        """Log helper for IK solver."""
        if self.verbose:
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[IK-SOLVER] [%s] %s", level, msg)

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

    def _within_joint_limits(self, joints: np.ndarray, tolerance: float = 1e-4) -> bool:
        """Check whether joints are within configured limits."""
        lower = self.config.joint_limits_lower
        upper = self.config.joint_limits_upper
        finite_mask = np.isfinite(lower) & np.isfinite(upper)
        if not np.any(finite_mask):
            return True
        below = joints[finite_mask] < (lower[finite_mask] - tolerance)
        above = joints[finite_mask] > (upper[finite_mask] + tolerance)
        return not (np.any(below) or np.any(above))

    def _reject_if_out_of_bounds(self, joints: np.ndarray) -> Optional[np.ndarray]:
        """Return joints if within limits, otherwise log and return None."""
        if not self._within_joint_limits(joints):
            if self.verbose:
                self.log("IK solution violates joint limits", "WARNING")
            return None
        return joints

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

        return self._reject_if_out_of_bounds(joints)

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
            return self._reject_if_out_of_bounds(joints)

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

        return self._reject_if_out_of_bounds(joints)

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

        return self._reject_if_out_of_bounds(joints)

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
        clipped_during_solve = False
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
                if clipped_during_solve:
                    return None
                return self._reject_if_out_of_bounds(joints)

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

            # Clamp to joint limits for stability, track violations
            lower = self.config.joint_limits_lower
            upper = self.config.joint_limits_upper
            finite_mask = np.isfinite(lower) & np.isfinite(upper)
            if np.any(finite_mask):
                out_of_bounds = np.logical_or(
                    joints[finite_mask] < lower[finite_mask],
                    joints[finite_mask] > upper[finite_mask],
                )
                if np.any(out_of_bounds):
                    clipped_during_solve = True
                joints[finite_mask] = np.clip(joints[finite_mask], lower[finite_mask], upper[finite_mask])

            # Adaptive damping: increase if error grows
            if iteration > 0:
                damping = min(1.0, damping * 1.1) if pos_error_norm > prev_pos_error else max(0.01, damping * 0.9)

            prev_pos_error = pos_error_norm

        # Did not converge
        self.log(f"Numerical IK did not converge after {max_iterations} iterations", "WARNING")
        self.log(f"  Position error: {pos_error_norm:.4f}m, Orientation error: {orient_error_norm:.4f}rad", "WARNING")

        # Return best effort if close
        if pos_error_norm < position_tolerance * 10:
            if clipped_during_solve:
                return None
            return self._reject_if_out_of_bounds(joints)

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
        elif self.config.name in ("g1", "g1_right_arm", "g1_left_arm"):
            return self._fk_g1(joints)
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

    def _fk_g1(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward kinematics for Unitree G1 right arm (7-DOF).

        DH parameters derived from G1 URDF link offsets:
        shoulder_pitch -> shoulder_roll -> shoulder_yaw -> elbow_pitch
        -> wrist_yaw -> wrist_pitch -> wrist_roll -> EE
        """
        # Modified DH parameters (a, d, alpha) for each joint
        dh = [
            (0.0,     0.0,     -np.pi / 2),   # shoulder_pitch
            (0.0,     0.0,      np.pi / 2),    # shoulder_roll
            (0.0,    -0.2800,  -np.pi / 2),    # shoulder_yaw (upper arm length)
            (0.0,     0.0,      np.pi / 2),    # elbow_pitch
            (0.0,    -0.2430,  -np.pi / 2),    # wrist_yaw (forearm length)
            (0.0,     0.0,      np.pi / 2),    # wrist_pitch
            (0.0,    -0.0500,   0.0),           # wrist_roll (wrist to EE)
        ]

        # Base transform: right shoulder offset from torso origin
        T = np.eye(4)
        T[1, 3] = -0.1585   # y offset (right side)
        T[2, 3] = 0.3520    # z offset (shoulder height)

        for i, (a, d, alpha) in enumerate(dh):
            theta = joints[i] if i < len(joints) else 0.0
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


class TrajectoryIKError(RuntimeError):
    """Raised when IK fails for a waypoint during trajectory solving."""

    def __init__(
        self,
        message: str,
        waypoint_index: Optional[int] = None,
        phase: Optional[MotionPhase] = None,
    ):
        super().__init__(message)
        self.waypoint_index = waypoint_index
        self.phase = phase


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
            level_map = {
                "DEBUG": logger.debug,
                "INFO": logger.info,
                "WARNING": logger.warning,
                "ERROR": logger.error,
            }
            log_fn = level_map.get(level.upper(), logger.info)
            log_fn("[TRAJECTORY-SOLVER] [%s] %s", level, msg)

    def _is_production_mode(self) -> bool:
        """Detect production mode for fail-closed behavior."""
        return resolve_production_mode()

    def _within_joint_limits(self, joints: np.ndarray, tolerance: float = 1e-4) -> bool:
        """Check whether joints are within configured limits."""
        lower = self.robot_config.joint_limits_lower
        upper = self.robot_config.joint_limits_upper
        finite_mask = np.isfinite(lower) & np.isfinite(upper)
        if not np.any(finite_mask):
            return True
        below = joints[finite_mask] < (lower[finite_mask] - tolerance)
        above = joints[finite_mask] > (upper[finite_mask] + tolerance)
        return not (np.any(below) or np.any(above))

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

        if self._is_production_mode():
            if not motion_plan.planning_success:
                raise TrajectoryIKError(
                    f"Motion plan {motion_plan.plan_id} planning failed in production mode."
                )
            if motion_plan.joint_trajectory is None:
                raise TrajectoryIKError(
                    f"Motion plan {motion_plan.plan_id} missing joint trajectory in production mode."
                )
            if not motion_plan.waypoints:
                raise TrajectoryIKError(
                    f"Motion plan {motion_plan.plan_id} has no waypoints in production mode."
                )

        # Step 1: Solve IK for each waypoint
        try:
            waypoint_joints = self._solve_waypoint_ik(motion_plan.waypoints)
        except TrajectoryIKError as exc:
            raise TrajectoryIKError(
                f"IK failed for motion plan {motion_plan.plan_id}: {exc}",
                waypoint_index=exc.waypoint_index,
                phase=exc.phase,
            ) from exc

        # Step 2: Build interpolation splines
        splines = self._build_splines(motion_plan.waypoints, waypoint_joints)

        # Step 3: Sample at target FPS
        states = self._sample_trajectory(
            motion_plan=motion_plan,
            splines=splines,
            waypoint_joints=waypoint_joints,
        )

        # Step 4: Jerk limiting — detect and smooth jerk spikes
        states = self._apply_jerk_limiting(states)

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
            seed = wp.joint_positions if wp.joint_positions is not None else prev_joints
            joints = self.ik_solver.solve(
                target_position=wp.position,
                target_orientation=wp.orientation,
                seed_joints=seed,
            )

            if joints is None:
                phase_label = wp.phase.value if hasattr(wp.phase, "value") else str(wp.phase)
                message = (
                    f"IK failed for waypoint {i} (phase={phase_label}) "
                    f"pos={wp.position.tolist()} ori={wp.orientation.tolist()}"
                )
                self.log(f"  {message}", "ERROR")
                raise TrajectoryIKError(message, waypoint_index=i, phase=wp.phase)

            if not self._within_joint_limits(joints):
                phase_label = wp.phase.value if hasattr(wp.phase, "value") else str(wp.phase)
                message = (
                    f"IK joint limits violated for waypoint {i} (phase={phase_label}) "
                    f"joints={joints.tolist()}"
                )
                self.log(f"  {message}", "ERROR")
                raise TrajectoryIKError(message, waypoint_index=i, phase=wp.phase)

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
            # Use cubic spline with clamped BCs (zero velocity at start/end)
            # for physically realistic robot trajectories that start/stop at rest
            spline = interpolate.CubicSpline(
                timestamps, joint_values, bc_type="clamped"
            )
            joint_splines.append(spline)

        # Gripper spline
        gripper_values = np.array([
            wp.gripper_aperture * self.robot_config.gripper_limits[1]
            for wp in waypoints
        ])
        gripper_spline = interpolate.CubicSpline(
            timestamps, gripper_values, bc_type="clamped"
        )

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

            # Analytical velocities from spline 1st derivative (smooth, no noise)
            velocities = np.array([spline(t, 1) for spline in joint_splines])

            # Analytical accelerations from spline 2nd derivative
            accelerations = np.array([spline(t, 2) for spline in joint_splines])

            # Validate against Franka joint velocity limits (~2.175 rad/s)
            _v_max = getattr(self.robot_config, 'max_joint_velocities', None)
            if _v_max is not None:
                _v_violations = np.abs(velocities) > np.array(_v_max)
                if np.any(_v_violations):
                    # Clamp to limits (trajectory is still valid but note violation)
                    velocities = np.clip(velocities, -np.array(_v_max), np.array(_v_max))

            # Validate against acceleration limits (~15 rad/s²)
            _a_max = getattr(self.robot_config, 'max_joint_accelerations', None)
            if _a_max is not None:
                accelerations = np.clip(accelerations, -np.array(_a_max), np.array(_a_max))

            # Get EE pose (use waypoint interpolation)
            ee_position, ee_orientation = self._interpolate_ee_pose(t, motion_plan.waypoints)

            state = JointState(
                frame_idx=i,
                timestamp=t,
                joint_positions=joint_positions,
                joint_velocities=velocities,
                joint_accelerations=accelerations,
                gripper_position=gripper_position,
                ee_position=ee_position,
                ee_orientation=ee_orientation,
                phase=get_phase_at_time(t),
            )
            states.append(state)

        return states

    def _apply_jerk_limiting(
        self,
        states: List[JointState],
        max_jerk: float = 7500.0,
    ) -> List[JointState]:
        """Post-process trajectory to limit jerk spikes.

        Computes jerk (d(acceleration)/dt) between consecutive frames.
        When jerk exceeds max_jerk (default 7500 rad/s³ for Franka),
        smooths accelerations using a moving-average filter on the
        offending segments, then recomputes velocities via integration
        to maintain consistency.

        Args:
            states: Sampled trajectory states with velocities and accelerations.
            max_jerk: Maximum allowed jerk in rad/s³.

        Returns:
            States with smoothed accelerations/velocities where needed.
        """
        if len(states) < 3:
            return states

        # Compute jerk at each interior frame
        num_joints = len(states[0].joint_positions)
        dt = states[1].timestamp - states[0].timestamp
        if dt <= 0:
            return states

        # Gather accelerations into array for vectorized ops
        accels = np.array([
            s.joint_accelerations if s.joint_accelerations is not None
            else np.zeros(num_joints)
            for s in states
        ])

        # Compute jerk: (accel[i+1] - accel[i]) / dt
        jerks = np.diff(accels, axis=0) / dt
        max_observed_jerk = np.max(np.abs(jerks)) if jerks.size > 0 else 0.0

        if max_observed_jerk <= max_jerk:
            return states  # No smoothing needed

        logger.info(
            "Jerk limiting: max observed %.0f rad/s³ > limit %.0f rad/s³, smoothing",
            max_observed_jerk, max_jerk,
        )

        # Smooth accelerations with a small moving-average kernel
        # Kernel size chosen to bring jerk below limit
        kernel_size = min(7, max(3, int(np.ceil(max_observed_jerk / max_jerk)) | 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size

        smoothed_accels = np.copy(accels)
        for j in range(num_joints):
            smoothed_accels[:, j] = np.convolve(
                accels[:, j], kernel, mode="same"
            )

        # Recompute velocities by integrating smoothed accelerations
        # (preserve initial velocity from spline)
        smoothed_vels = np.zeros_like(smoothed_accels)
        smoothed_vels[0] = states[0].joint_velocities if states[0].joint_velocities is not None else np.zeros(num_joints)
        for i in range(1, len(states)):
            smoothed_vels[i] = smoothed_vels[i - 1] + smoothed_accels[i] * dt

        # Apply velocity limits if available
        _v_max = getattr(self.robot_config, 'max_joint_velocities', None)
        if _v_max is not None:
            _v_arr = np.array(_v_max)
            smoothed_vels = np.clip(smoothed_vels, -_v_arr, _v_arr)

        # Write back to states
        for i, state in enumerate(states):
            state.joint_accelerations = smoothed_accels[i]
            state.joint_velocities = smoothed_vels[i]

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
    from tools.logging_config import init_logging

    init_logging()
    # Test the full pipeline
    logger.info("Testing Motion Planner + Trajectory Solver")
    logger.info("%s", "=" * 60)

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

    logger.info("%s", "=" * 60)
    logger.info("TRAJECTORY SUMMARY")
    logger.info("%s", "=" * 60)
    logger.info("Frames: %s", trajectory.num_frames)
    logger.info("Duration: %.2fs", trajectory.total_duration)
    logger.info("FPS: %s", trajectory.fps)
    logger.info("Joint names: %s", trajectory.robot_config.joint_names)

    # Sample output
    logger.info("Sample states:")
    for i in [0, len(trajectory.states)//2, -1]:
        s = trajectory.states[i]
        logger.info(
            "  Frame %s: t=%.2fs, phase=%s, gripper=%.3fm",
            s.frame_idx,
            s.timestamp,
            s.phase.value,
            s.gripper_position,
        )
