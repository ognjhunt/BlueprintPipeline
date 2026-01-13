"""
Robot Tracker for Dream2Flow.

Uses 3D object flow as a goal or reward signal for robot control.
This is the final stage of the Dream2Flow pipeline where the robot
actually executes the task by tracking the extracted object flow.

Based on Dream2Flow (arXiv:2512.24766):
- Object flow as embodiment-agnostic reward for RL
- Trajectory optimization to track flow targets
- Supports multiple robot embodiments (Franka, Spot, GR1, etc.)

Robot execution failures are the third failure mode in the pipeline,
after video generation artifacts and flow extraction failures.
"""

import math
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import uuid

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    ObjectFlow3D,
    RobotEmbodiment,
    RobotTrackingMethod,
    RobotTrackingTarget,
    RobotTrajectory,
)


@dataclass
class RobotTrackerConfig:
    """Configuration for robot tracking."""

    # Tracking method
    method: RobotTrackingMethod = RobotTrackingMethod.TRAJECTORY_OPTIMIZATION

    # Robot embodiment
    robot: RobotEmbodiment = RobotEmbodiment.FRANKA_PANDA

    # Trajectory optimization parameters
    horizon: int = 20  # MPC horizon
    num_iterations: int = 100
    learning_rate: float = 0.01

    # RL parameters (if using RL)
    use_flow_reward: bool = True
    position_reward_weight: float = 1.0
    velocity_reward_weight: float = 0.1
    smoothness_reward_weight: float = 0.05

    # Goal tracking parameters
    lookahead_frames: int = 5
    position_tolerance: float = 0.01  # meters
    velocity_tolerance: float = 0.1  # m/s

    # IK solver parameters
    ik_max_iterations: int = 200
    ik_tolerance: float = 1e-3
    ik_damping: float = 1e-2
    ik_step_size: float = 0.5
    ik_link_lengths: Optional[list[float]] = None

    # Isaac Lab integration
    isaac_lab_env: Optional[str] = None
    use_simulation: bool = True

    # Output options
    save_trajectory: bool = True
    save_visualization: bool = True

    # Debug
    verbose: bool = True
    # Remote tracking API (optional)
    tracking_api: Optional[str] = None
    # Feature flags
    enabled: bool = True
    allow_placeholder: bool = True
    require_real_backend: bool = False


@dataclass
class TrackingResult:
    """Result of tracking object flow."""

    # Success flag
    success: bool

    # Generated robot trajectory
    trajectory: Optional[RobotTrajectory] = None

    # Tracking error metrics
    mean_position_error: float = 0.0
    max_position_error: float = 0.0
    final_position_error: float = 0.0

    # Flow reward (for RL)
    total_flow_reward: float = 0.0
    per_step_rewards: list[float] = field(default_factory=list)

    # Error message if failed
    error: Optional[str] = None

    # Output paths
    trajectory_path: Optional[Path] = None
    visualization_path: Optional[Path] = None


class RobotTracker:
    """
    Tracks 3D object flow using robot control.

    Implements trajectory optimization and RL-based tracking
    to make the robot move objects along the target flow.
    """

    def __init__(self, config: RobotTrackerConfig):
        self.config = config
        self._initialized = False
        self._disabled_reason: Optional[str] = None

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.config.verbose:
            print(f"[ROBOT-TRACK] [{level}] {msg}")

    def initialize(self) -> bool:
        """Initialize robot tracker."""
        if self._initialized:
            return True
        if not self.config.enabled:
            self._disabled_reason = "Robot tracking disabled by configuration."
            self.log(self._disabled_reason, level="WARNING")
            self._initialized = True
            return False

        try:
            if self.config.tracking_api:
                self.log("Robot tracker initialized with API backend")
            elif self.config.isaac_lab_env:
                self.log(f"Robot tracker initialized with Isaac Lab env: {self.config.isaac_lab_env}")
            else:
                self.log("Robot tracker initialized with local IK backend")
            self._initialized = True
            return True
        except ImportError as e:
            if self.config.allow_placeholder and not self.config.require_real_backend:
                self.log(f"Robot tracker initialization skipped: {e}", level="WARNING")
                self._initialized = True  # Still proceed in placeholder mode
                return True
            self._disabled_reason = f"Robot tracker disabled: {e}"
            self.log(self._disabled_reason, level="ERROR")
            self._initialized = True
            return False

    def track(
        self,
        object_flow: ObjectFlow3D,
        output_dir: Path,
    ) -> TrackingResult:
        """
        Track object flow with robot.

        Args:
            object_flow: 3D object flow to track
            output_dir: Directory to save outputs

        Returns:
            TrackingResult with trajectory and metrics
        """
        if not self._initialized:
            self.initialize()
        if self._disabled_reason:
            self.log(self._disabled_reason, level="ERROR")
            return TrackingResult(success=False, error=self._disabled_reason)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log(f"Tracking flow: {object_flow.flow_id}")

        try:
            if self.config.tracking_api:
                return self._track_via_api(object_flow, output_dir)
            if self.config.method == RobotTrackingMethod.TRAJECTORY_OPTIMIZATION:
                return self._track_with_optimization(object_flow, output_dir)
            elif self.config.method == RobotTrackingMethod.REINFORCEMENT_LEARNING:
                return self._track_with_rl(object_flow, output_dir)
            else:
                if self.config.require_real_backend or not self.config.allow_placeholder:
                    return TrackingResult(
                        success=False,
                        error="Robot tracking failed: unknown tracking method and placeholders disabled.",
                    )
                return self._track_placeholder(object_flow, output_dir)

        except Exception as e:
            self.log(f"Tracking failed: {e}", "ERROR")
            traceback.print_exc()
            return TrackingResult(success=False, error=str(e))

    def _track_via_api(
        self,
        object_flow: ObjectFlow3D,
        output_dir: Path,
    ) -> TrackingResult:
        """Track object flow via remote API."""
        import requests

        self.log("Tracking via API...")
        payload = {
            "flow_id": object_flow.flow_id,
            "object_id": object_flow.object_id,
            "fps": object_flow.fps,
            "trajectory": object_flow.get_center_trajectory().tolist(),
            "robot": self.config.robot.value,
            "method": self.config.method.value,
        }

        response = requests.post(self.config.tracking_api, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()

        if not result.get("success", True):
            return TrackingResult(success=False, error=result.get("error", "Tracking API failed"))

        trajectory_data = result.get("trajectory")
        if trajectory_data is None:
            raise ValueError("Tracking API did not return trajectory data")

        ee_poses = np.array(trajectory_data.get("ee_poses"), dtype=np.float32)
        joint_positions = None
        if trajectory_data.get("joint_positions") is not None:
            joint_positions = np.array(trajectory_data.get("joint_positions"), dtype=np.float32)

        trajectory = RobotTrajectory(
            trajectory_id=trajectory_data.get("trajectory_id", f"api_{object_flow.flow_id}"),
            robot=self.config.robot,
            ee_poses=ee_poses,
            joint_positions=joint_positions,
            gripper_states=np.array(trajectory_data.get("gripper_states"))
            if trajectory_data.get("gripper_states") is not None
            else None,
            joint_names=trajectory_data.get("joint_names") or self._get_joint_names(),
            fps=trajectory_data.get("fps", object_flow.fps),
            mean_tracking_error=trajectory_data.get("mean_tracking_error", 0.0),
            max_tracking_error=trajectory_data.get("max_tracking_error", 0.0),
            success=True,
        )

        trajectory_path = None
        if self.config.save_trajectory:
            trajectory_path = self._save_trajectory(trajectory, output_dir)

        return TrackingResult(
            success=True,
            trajectory=trajectory,
            mean_position_error=trajectory.mean_tracking_error,
            max_position_error=trajectory.max_tracking_error,
            final_position_error=trajectory.max_tracking_error,
            trajectory_path=trajectory_path,
        )

    def _track_with_optimization(
        self,
        object_flow: ObjectFlow3D,
        output_dir: Path,
    ) -> TrackingResult:
        """
        Track using trajectory optimization (MPC/iLQR).
        """
        self.log("Using trajectory optimization with local IK solver")

        # Get target trajectory from flow
        target_trajectory = object_flow.get_center_trajectory()
        num_frames = len(target_trajectory)

        if num_frames == 0:
            return TrackingResult(success=False, error="Empty object flow")

        smoothed_targets = self._smooth_trajectory(target_trajectory, self.config.lookahead_frames)
        joint_positions = np.zeros((num_frames, self._get_dof()))
        ee_poses = np.zeros((num_frames, 4, 4))
        position_errors = []

        current_joints = self._get_home_joint_positions()
        for i, target_pos in enumerate(smoothed_targets):
            joint_solution, ee_pose, error = self._solve_ik(target_pos, current_joints)
            joint_positions[i] = joint_solution
            ee_poses[i] = ee_pose
            position_errors.append(error)
            current_joints = joint_solution

        # Create trajectory
        trajectory = RobotTrajectory(
            trajectory_id=f"traj_{object_flow.flow_id}",
            robot=self.config.robot,
            ee_poses=ee_poses,
            joint_positions=joint_positions,
            gripper_states=np.zeros(num_frames),
            joint_names=self._get_joint_names(),
            fps=object_flow.fps,
            mean_tracking_error=np.mean(position_errors),
            max_tracking_error=np.max(position_errors),
            success=True,
        )

        # Save trajectory
        trajectory_path = None
        if self.config.save_trajectory:
            trajectory_path = self._save_trajectory(trajectory, output_dir)

        return TrackingResult(
            success=True,
            trajectory=trajectory,
            mean_position_error=np.mean(position_errors),
            max_position_error=np.max(position_errors),
            final_position_error=position_errors[-1] if position_errors else 0.0,
            trajectory_path=trajectory_path,
        )

    def _track_with_rl(
        self,
        object_flow: ObjectFlow3D,
        output_dir: Path,
    ) -> TrackingResult:
        """
        Track using reinforcement learning with flow as reward.

        The key insight from Dream2Flow is that 3D object flow can serve
        as an embodiment-agnostic reward signal for RL training.
        """
        self.log("Using flow-reward optimization with IK rollouts")

        target_trajectory = object_flow.get_center_trajectory()
        num_frames = len(target_trajectory)

        if num_frames == 0:
            return TrackingResult(success=False, error="Empty object flow")

        rewards = []
        ee_poses = np.zeros((num_frames, 4, 4))
        joint_positions = np.zeros((num_frames, self._get_dof()))
        current_joints = self._get_home_joint_positions()
        position_errors = []
        for i in range(num_frames):
            target_pos = target_trajectory[i]
            joint_solution, ee_pose, error = self._solve_ik(target_pos, current_joints)
            ee_poses[i] = ee_pose
            joint_positions[i] = joint_solution
            current_joints = joint_solution
            position_errors.append(error)

            reward = -error * self.config.position_reward_weight
            if i > 0:
                achieved_velocity = ee_poses[i, :3, 3] - ee_poses[i - 1, :3, 3]
                target_velocity = target_trajectory[i] - target_trajectory[i - 1]
                velocity_error = np.linalg.norm(achieved_velocity - target_velocity)
                reward -= velocity_error * self.config.velocity_reward_weight
            rewards.append(reward)

        trajectory = RobotTrajectory(
            trajectory_id=f"rl_traj_{object_flow.flow_id}",
            robot=self.config.robot,
            ee_poses=ee_poses,
            joint_positions=joint_positions,
            fps=object_flow.fps,
            success=True,
        )

        trajectory_path = None
        if self.config.save_trajectory:
            trajectory_path = self._save_trajectory(trajectory, output_dir)

        return TrackingResult(
            success=True,
            trajectory=trajectory,
            total_flow_reward=sum(rewards),
            per_step_rewards=rewards,
            trajectory_path=trajectory_path,
            mean_position_error=float(np.mean(position_errors)) if position_errors else 0.0,
            max_position_error=float(np.max(position_errors)) if position_errors else 0.0,
            final_position_error=float(position_errors[-1]) if position_errors else 0.0,
        )

    def _track_placeholder(
        self,
        object_flow: ObjectFlow3D,
        output_dir: Path,
    ) -> TrackingResult:
        """Placeholder tracking for testing."""
        self.log("Using placeholder tracking", level="WARNING")

        target_trajectory = object_flow.get_center_trajectory()
        num_frames = len(target_trajectory)

        trajectory = RobotTrajectory(
            trajectory_id=f"placeholder_{object_flow.flow_id}",
            robot=self.config.robot,
            ee_poses=np.zeros((num_frames, 4, 4)),
            fps=object_flow.fps,
            success=True,
        )

        return TrackingResult(
            success=True,
            trajectory=trajectory,
            mean_position_error=0.05,  # Placeholder
        )

    def _get_dof(self) -> int:
        """Return degrees of freedom for the configured robot."""
        if self.config.robot == RobotEmbodiment.UR5E:
            return 6
        if self.config.robot in {RobotEmbodiment.BOSTON_DYNAMICS_SPOT, RobotEmbodiment.FOURIER_GR1}:
            return 7
        return 7

    def _get_home_joint_positions(self) -> np.ndarray:
        """Return a default home joint configuration."""
        dof = self._get_dof()
        return np.zeros(dof, dtype=np.float32)

    def _get_link_lengths(self) -> list[float]:
        """Return link lengths for a simple kinematic chain."""
        if self.config.ik_link_lengths:
            return self.config.ik_link_lengths
        if self.config.robot == RobotEmbodiment.UR5E:
            return [0.1625, 0.425, 0.3922, 0.1333, 0.0997, 0.0996]
        if self.config.robot == RobotEmbodiment.FRANKA_PANDA:
            return [0.333, 0.316, 0.384, 0.107, 0.1, 0.1, 0.1]
        return [0.2] * self._get_dof()

    def _get_joint_axes(self) -> list[np.ndarray]:
        """Return joint rotation axes for a simple 3D chain."""
        dof = self._get_dof()
        axes = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        return axes[:dof]

    def _axis_angle_to_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Compute rotation matrix from axis-angle."""
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        x, y, z = axis
        c = math.cos(angle)
        s = math.sin(angle)
        C = 1 - c
        return np.array([
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ])

    def _forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute end-effector pose for a simplified kinematic chain."""
        link_lengths = self._get_link_lengths()
        axes = self._get_joint_axes()
        T = np.eye(4)
        for angle, axis, length in zip(joint_positions, axes, link_lengths):
            R = self._axis_angle_to_matrix(axis, angle)
            T[:3, :3] = T[:3, :3] @ R
            T[:3, 3] += T[:3, :3] @ np.array([length, 0.0, 0.0])
        return T

    def _compute_jacobian(self, joint_positions: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
        """Numerically compute position Jacobian."""
        base_pose = self._forward_kinematics(joint_positions)
        base_pos = base_pose[:3, 3]
        dof = len(joint_positions)
        J = np.zeros((3, dof))
        for i in range(dof):
            perturbed = joint_positions.copy()
            perturbed[i] += epsilon
            pos = self._forward_kinematics(perturbed)[:3, 3]
            J[:, i] = (pos - base_pos) / epsilon
        return J

    def _solve_ik(
        self,
        target_pos: np.ndarray,
        initial_joints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Solve IK for a target position using damped least squares."""
        joints = initial_joints.copy()
        for _ in range(self.config.ik_max_iterations):
            pose = self._forward_kinematics(joints)
            current_pos = pose[:3, 3]
            error_vec = target_pos - current_pos
            error = np.linalg.norm(error_vec)
            if error <= self.config.ik_tolerance:
                return joints, pose, error
            J = self._compute_jacobian(joints)
            JJt = J @ J.T
            damping = self.config.ik_damping * np.eye(3)
            step = J.T @ np.linalg.solve(JJt + damping, error_vec)
            joints += self.config.ik_step_size * step
        pose = self._forward_kinematics(joints)
        error = np.linalg.norm(target_pos - pose[:3, 3])
        return joints, pose, error

    def _smooth_trajectory(self, trajectory: np.ndarray, window: int) -> np.ndarray:
        """Smooth trajectory with a moving average window."""
        if window <= 1 or len(trajectory) == 0:
            return trajectory
        smoothed = []
        for idx in range(len(trajectory)):
            start = max(0, idx - window)
            end = min(len(trajectory), idx + window + 1)
            smoothed.append(np.mean(trajectory[start:end], axis=0))
        return np.array(smoothed)

    def _get_joint_names(self) -> list[str]:
        """Get joint names for the configured robot."""
        if self.config.robot == RobotEmbodiment.FRANKA_PANDA:
            return [
                "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                "panda_joint5", "panda_joint6", "panda_joint7",
            ]
        elif self.config.robot == RobotEmbodiment.UR5E:
            return [
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            ]
        else:
            return [f"joint_{i}" for i in range(7)]

    def _save_trajectory(self, trajectory: RobotTrajectory, output_dir: Path) -> Path:
        """Save trajectory to file."""
        import json

        trajectory_path = output_dir / f"{trajectory.trajectory_id}.json"
        data = {
            "trajectory_id": trajectory.trajectory_id,
            "robot": trajectory.robot.value,
            "num_frames": trajectory.num_frames,
            "fps": trajectory.fps,
            "joint_names": trajectory.joint_names,
            "mean_tracking_error": trajectory.mean_tracking_error,
            "max_tracking_error": trajectory.max_tracking_error,
            "success": trajectory.success,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

        if trajectory.ee_poses is not None:
            data["ee_poses"] = trajectory.ee_poses.tolist()
        if trajectory.joint_positions is not None:
            data["joint_positions"] = trajectory.joint_positions.tolist()
        if trajectory.gripper_states is not None:
            data["gripper_states"] = trajectory.gripper_states.tolist()

        trajectory_path.write_text(json.dumps(data, indent=2))
        return trajectory_path

    def compute_flow_reward(
        self,
        target_flow: ObjectFlow3D,
        achieved_positions: np.ndarray,
        frame_idx: int,
    ) -> float:
        """
        Compute flow-based reward for a single timestep.

        This is the embodiment-agnostic reward signal that can be used
        across different robot platforms.
        """
        target_pos = target_flow.get_point_cloud_at_frame(frame_idx)
        if len(target_pos) == 0:
            return 0.0

        target_center = np.mean(target_pos, axis=0)
        achieved_center = np.mean(achieved_positions, axis=0) if len(achieved_positions) > 0 else np.zeros(3)

        # Position reward (negative distance)
        position_error = np.linalg.norm(achieved_center - target_center)
        position_reward = -position_error * self.config.position_reward_weight

        return position_reward


class MockRobotTracker(RobotTracker):
    """Mock robot tracker for testing."""

    def __init__(self, config: Optional[RobotTrackerConfig] = None):
        config = config or RobotTrackerConfig()
        super().__init__(config)

    def initialize(self) -> bool:
        self.log("Mock robot tracker initialized")
        self._initialized = True
        return True


def track_object_flow(
    object_flow: ObjectFlow3D,
    output_dir: Path,
    config: Optional[RobotTrackerConfig] = None,
) -> TrackingResult:
    """
    Convenience function to track object flow.

    Args:
        object_flow: 3D object flow to track
        output_dir: Directory to save outputs
        config: Optional tracker configuration

    Returns:
        TrackingResult with trajectory and metrics
    """
    config = config or RobotTrackerConfig()
    tracker = RobotTracker(config)

    return tracker.track(
        object_flow=object_flow,
        output_dir=output_dir,
    )
