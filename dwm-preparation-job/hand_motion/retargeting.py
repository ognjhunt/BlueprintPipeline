"""
Hand-to-robot retargeting utilities.

Maps MANO-format hand trajectories and associated camera poses into robot
end-effector targets and joint commands using a lightweight damped-least-squares
IK solver for a 6-DoF arm with a parallel gripper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from models import CameraTrajectory, HandPose, HandTrajectory, RobotAction


def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Create a standard DH transform."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to axis-angle vector."""
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = math.acos(trace)
    if abs(theta) < 1e-6:
        return np.zeros(3, dtype=np.float64)

    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=np.float64) / (2.0 * math.sin(theta))
    return axis * theta


@dataclass
class RobotConfig:
    """Basic robot configuration for retargeting."""

    name: str = "ur5e_parallel_gripper"
    # DH parameters for UR5e-style arm
    a: Sequence[float] = field(
        default_factory=lambda: [0.0, -0.425, -0.39225, 0.0, 0.0, 0.0]
    )
    alpha: Sequence[float] = field(
        default_factory=lambda: [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]
    )
    d: Sequence[float] = field(
        default_factory=lambda: [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823]
    )
    joint_limits: Sequence[tuple[float, float]] = field(
        default_factory=lambda: [(-2 * math.pi, 2 * math.pi)] * 6
    )
    joint_names: Sequence[str] = field(
        default_factory=lambda: [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ]
    )
    base_frame: str = "world"
    end_effector_frame: str = "tool0"
    gripper_open_aperture: float = 0.08
    gripper_closed_aperture: float = 0.0


class Simple6DofIK:
    """Minimal damped-least-squares IK for 6-DoF arms."""

    def __init__(self, config: RobotConfig):
        self.config = config

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        for a_i, alpha_i, d_i, q_i in zip(
            self.config.a, self.config.alpha, self.config.d, q
        ):
            T = T @ _dh_transform(a_i, alpha_i, d_i, q_i)
        return T

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        origins: List[np.ndarray] = [np.zeros(3)]
        z_axes: List[np.ndarray] = [np.array([0.0, 0.0, 1.0])]
        T = np.eye(4, dtype=np.float64)

        for a_i, alpha_i, d_i, q_i in zip(
            self.config.a, self.config.alpha, self.config.d, q
        ):
            T = T @ _dh_transform(a_i, alpha_i, d_i, q_i)
            origins.append(T[:3, 3])
            z_axes.append(T[:3, 2])

        o_n = origins[-1]
        J = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            z = z_axes[i]
            o = origins[i]
            J[:3, i] = np.cross(z, o_n - o)
            J[3:, i] = z
        return J

    def solve(
        self,
        target_T: np.ndarray,
        initial_q: Optional[np.ndarray] = None,
        max_iters: int = 200,
        step: float = 0.5,
        damping: float = 1e-3,
        tol: float = 1e-4,
    ) -> np.ndarray:
        q = np.zeros(6, dtype=np.float64) if initial_q is None else np.array(initial_q, dtype=np.float64)

        for _ in range(max_iters):
            current_T = self.forward_kinematics(q)
            pos_err = target_T[:3, 3] - current_T[:3, 3]
            rot_err = _rotation_to_axis_angle(target_T[:3, :3] @ current_T[:3, :3].T)
            error = np.concatenate([pos_err, rot_err])

            if np.linalg.norm(error) < tol:
                break

            J = self.jacobian(q)
            lhs = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(lhs, error)
            q += step * dq
            q = np.clip(
                q,
                [low for low, _ in self.config.joint_limits],
                [high for _, high in self.config.joint_limits],
            )

        return q


class HandRetargeter:
    """Retarget hand trajectories to robot joint targets."""

    def __init__(self, robot_config: Optional[RobotConfig] = None):
        self.robot_config = robot_config or RobotConfig()
        self.ik_solver = Simple6DofIK(self.robot_config)

    def retarget(
        self,
        hand_traj: HandTrajectory,
        camera_traj: Optional[CameraTrajectory] = None,
        base_pose_in_world: Optional[np.ndarray] = None,
    ) -> List[RobotAction]:
        """Compute per-frame robot actions for a hand trajectory."""
        if base_pose_in_world is None:
            if camera_traj and camera_traj.poses:
                base_pose_in_world = camera_traj.poses[0].transform
            else:
                base_pose_in_world = np.eye(4, dtype=np.float64)

        world_T_base = np.linalg.inv(base_pose_in_world)
        prev_q = np.zeros(6, dtype=np.float64)
        actions: List[RobotAction] = []

        for pose in hand_traj.poses:
            wrist_in_world = np.eye(4, dtype=np.float64)
            wrist_in_world[:3, :3] = pose.rotation
            wrist_in_world[:3, 3] = pose.position

            target_in_base = world_T_base @ wrist_in_world
            q = self.ik_solver.solve(target_in_base, initial_q=prev_q)
            prev_q = q

            gripper_aperture = (
                self.robot_config.gripper_closed_aperture
                if any(pose.contact_fingertips)
                else self.robot_config.gripper_open_aperture
            )

            actions.append(
                RobotAction(
                    frame_idx=pose.frame_idx,
                    wrist_pose=target_in_base,
                    joint_positions=q.tolist(),
                    joint_names=list(self.robot_config.joint_names),
                    gripper_aperture=gripper_aperture,
                    base_frame=self.robot_config.base_frame,
                    end_effector_frame=self.robot_config.end_effector_frame,
                )
            )

        return actions
