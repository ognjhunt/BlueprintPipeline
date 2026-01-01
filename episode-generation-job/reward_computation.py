#!/usr/bin/env python3
"""
Reward Computation for Episode Generation.

Computes meaningful reward signals for robot manipulation episodes.
Rewards are essential for training policies with reinforcement learning.

Reward Types:
- Sparse Reward: Binary success/failure at episode end
- Dense Reward: Continuous signal based on progress toward goal
- Shaped Reward: Hand-crafted components for specific behaviors

Reward Components:
1. Task Completion: Did the robot achieve the goal?
2. Grasp Success: Did the robot successfully grasp the object?
3. Placement Accuracy: How close to target position?
4. Path Efficiency: Shorter/smoother paths get higher rewards
5. Safety: Penalties for collisions, joint limits, etc.

IMPORTANT: These rewards are computed during episode generation and stored
with the episode data. They can be used for:
- Filtering training data (only use high-reward episodes)
- Offline RL training
- Behavior cloning with reward weighting

References:
- RoboCasa: Environment-specific reward functions
- DemoGen: Success-based filtering
- AnyTask: Task-conditioned rewards
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class RewardType(Enum):
    """Types of reward signals."""
    SPARSE = "sparse"  # Binary success at end
    DENSE = "dense"  # Per-step progress
    SHAPED = "shaped"  # Multi-component hand-crafted


@dataclass
class RewardComponents:
    """Breakdown of reward components for interpretability."""

    # Task completion (primary)
    task_completion: float = 0.0  # 0-1, did task succeed?

    # Grasp phase
    grasp_success: float = 0.0  # 0-1, object grasped?
    grasp_stability: float = 0.0  # 0-1, stable grasp?

    # Placement phase
    placement_success: float = 0.0  # 0-1, object placed?
    placement_accuracy: float = 0.0  # 0-1, how close to target?

    # Efficiency
    path_efficiency: float = 0.0  # 0-1, optimal path?
    time_efficiency: float = 0.0  # 0-1, reasonable time?

    # Safety/quality
    collision_penalty: float = 0.0  # 0-1 (inverted, 1=no collisions)
    smoothness: float = 0.0  # 0-1, smooth motion?
    joint_limit_penalty: float = 0.0  # 0-1 (inverted, 1=no violations)

    def total(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted total reward."""
        if weights is None:
            weights = DEFAULT_REWARD_WEIGHTS

        total = 0.0
        for component, weight in weights.items():
            value = getattr(self, component, 0.0)
            total += weight * value

        return np.clip(total, 0.0, 1.0)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "task_completion": self.task_completion,
            "grasp_success": self.grasp_success,
            "grasp_stability": self.grasp_stability,
            "placement_success": self.placement_success,
            "placement_accuracy": self.placement_accuracy,
            "path_efficiency": self.path_efficiency,
            "time_efficiency": self.time_efficiency,
            "collision_penalty": self.collision_penalty,
            "smoothness": self.smoothness,
            "joint_limit_penalty": self.joint_limit_penalty,
        }


# Default weights for reward components
DEFAULT_REWARD_WEIGHTS = {
    "task_completion": 0.30,  # Primary objective
    "grasp_success": 0.15,
    "grasp_stability": 0.05,
    "placement_success": 0.15,
    "placement_accuracy": 0.10,
    "path_efficiency": 0.05,
    "time_efficiency": 0.05,
    "collision_penalty": 0.05,
    "smoothness": 0.05,
    "joint_limit_penalty": 0.05,
}


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    # Reward type
    reward_type: RewardType = RewardType.SHAPED

    # Component weights
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_REWARD_WEIGHTS.copy())

    # Thresholds
    placement_accuracy_threshold: float = 0.05  # meters
    time_bonus_threshold: float = 0.8  # fraction of expected time
    collision_tolerance: int = 0  # number of allowed collisions

    # Normalization
    normalize_rewards: bool = True
    reward_scale: float = 1.0


class RewardComputer:
    """
    Computes rewards for robot manipulation episodes.

    This class analyzes episode trajectories and validation results to compute
    meaningful reward signals for training.

    Usage:
        reward_computer = RewardComputer()

        # From trajectory and validation result
        reward, components = reward_computer.compute_episode_reward(
            trajectory=trajectory,
            motion_plan=motion_plan,
            validation_result=validation_result,
        )

        # Dense rewards (per step)
        dense_rewards = reward_computer.compute_dense_rewards(
            trajectory=trajectory,
            motion_plan=motion_plan,
        )
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        verbose: bool = False,
    ):
        self.config = config or RewardConfig()
        self.verbose = verbose

    def log(self, msg: str, level: str = "INFO") -> None:
        if self.verbose:
            print(f"[REWARD] [{level}] {msg}")

    def compute_episode_reward(
        self,
        trajectory: Any,  # JointTrajectory
        motion_plan: Any,  # MotionPlan
        validation_result: Optional[Any] = None,  # ValidationResult
        scene_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[float, RewardComponents]:
        """
        Compute total reward for an episode.

        Args:
            trajectory: The joint trajectory for the episode
            motion_plan: The original motion plan
            validation_result: Optional validation result with physics data
            scene_objects: Optional list of scene objects

        Returns:
            Tuple of (total_reward, components)
        """
        components = RewardComponents()

        # 1. Task completion
        components.task_completion = self._compute_task_completion(
            trajectory, motion_plan, validation_result
        )

        # 2. Grasp success
        components.grasp_success = self._compute_grasp_success(
            trajectory, motion_plan, validation_result
        )

        # 3. Grasp stability
        components.grasp_stability = self._compute_grasp_stability(
            trajectory, validation_result
        )

        # 4. Placement success
        components.placement_success = self._compute_placement_success(
            trajectory, motion_plan, validation_result
        )

        # 5. Placement accuracy
        components.placement_accuracy = self._compute_placement_accuracy(
            trajectory, motion_plan, scene_objects
        )

        # 6. Path efficiency
        components.path_efficiency = self._compute_path_efficiency(
            trajectory, motion_plan
        )

        # 7. Time efficiency
        components.time_efficiency = self._compute_time_efficiency(
            trajectory, motion_plan
        )

        # 8. Collision penalty (inverted: 1 = no collisions)
        components.collision_penalty = self._compute_collision_penalty(
            validation_result
        )

        # 9. Smoothness
        components.smoothness = self._compute_smoothness(trajectory)

        # 10. Joint limit penalty (inverted: 1 = no violations)
        components.joint_limit_penalty = self._compute_joint_limit_penalty(
            validation_result
        )

        # Compute weighted total
        total_reward = components.total(self.config.weights)

        if self.config.normalize_rewards:
            total_reward = np.clip(total_reward * self.config.reward_scale, 0.0, 1.0)

        self.log(f"Computed reward: {total_reward:.3f}")
        self.log(f"  Task completion: {components.task_completion:.2f}")
        self.log(f"  Grasp success: {components.grasp_success:.2f}")
        self.log(f"  Placement: {components.placement_success:.2f} (acc: {components.placement_accuracy:.2f})")

        return total_reward, components

    def compute_dense_rewards(
        self,
        trajectory: Any,
        motion_plan: Any,
        scene_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Compute per-step dense rewards for the trajectory.

        Dense rewards provide continuous feedback and are useful for RL training.

        Returns:
            Array of rewards, one per trajectory step
        """
        if not hasattr(trajectory, 'states') or len(trajectory.states) == 0:
            return np.array([0.0])

        num_steps = len(trajectory.states)
        rewards = np.zeros(num_steps)

        target_position = motion_plan.target_position
        place_position = motion_plan.place_position

        for i, state in enumerate(trajectory.states):
            step_reward = 0.0

            # Get end-effector position
            ee_pos = state.ee_position if hasattr(state, 'ee_position') else None
            if ee_pos is None:
                continue

            phase = state.phase.value if hasattr(state, 'phase') else "unknown"

            # Phase-dependent rewards
            if phase in ["approach", "pre_grasp"]:
                # Reward for getting closer to target
                if target_position is not None:
                    dist = np.linalg.norm(ee_pos - np.array(target_position))
                    step_reward += max(0, 1.0 - dist / 0.5)  # Normalize by 0.5m

            elif phase in ["grasp", "lift"]:
                # Reward for maintaining grasp
                gripper = state.gripper_position if hasattr(state, 'gripper_position') else 0
                if gripper < 0.02:  # Closed gripper
                    step_reward += 0.5

            elif phase in ["transport"]:
                # Reward for moving toward place position
                if place_position is not None:
                    dist = np.linalg.norm(ee_pos - np.array(place_position))
                    step_reward += max(0, 1.0 - dist / 0.5)

            elif phase in ["place", "release"]:
                # Reward for reaching place position
                if place_position is not None:
                    dist = np.linalg.norm(ee_pos - np.array(place_position))
                    step_reward += max(0, 1.0 - dist / 0.1)  # Tighter threshold

            elif phase in ["retract", "home"]:
                # Small reward for completing
                step_reward += 0.1

            rewards[i] = step_reward

        # Normalize
        if np.max(rewards) > 0:
            rewards = rewards / np.max(rewards)

        return rewards

    def compute_sparse_reward(
        self,
        trajectory: Any,
        motion_plan: Any,
        validation_result: Optional[Any] = None,
    ) -> float:
        """
        Compute sparse (binary) reward for episode.

        Returns 1.0 if task succeeded, 0.0 otherwise.
        """
        if validation_result is not None:
            # Use validation result
            if hasattr(validation_result, 'metrics'):
                return 1.0 if validation_result.metrics.task_success else 0.0
            if hasattr(validation_result, 'status'):
                return 1.0 if validation_result.status.value == "passed" else 0.0

        # Infer from trajectory
        task_completion = self._compute_task_completion(
            trajectory, motion_plan, validation_result
        )
        return 1.0 if task_completion > 0.5 else 0.0

    # =========================================================================
    # Component computation methods
    # =========================================================================

    def _compute_task_completion(
        self,
        trajectory: Any,
        motion_plan: Any,
        validation_result: Optional[Any],
    ) -> float:
        """Compute task completion reward."""
        if validation_result is not None:
            if hasattr(validation_result, 'metrics'):
                return 1.0 if validation_result.metrics.task_success else 0.0

        # Infer from trajectory end state
        if not hasattr(trajectory, 'states') or len(trajectory.states) == 0:
            return 0.0

        final_state = trajectory.states[-1]

        # Check if we reached a terminal phase
        if hasattr(final_state, 'phase'):
            terminal_phases = ["home", "retract", "release"]
            if final_state.phase.value in terminal_phases:
                return 1.0

        # Check if gripper released (for place tasks)
        if hasattr(final_state, 'gripper_position'):
            if final_state.gripper_position > 0.02:  # Open gripper
                return 0.8  # Partial credit

        return 0.3  # Minimal credit for completing trajectory

    def _compute_grasp_success(
        self,
        trajectory: Any,
        motion_plan: Any,
        validation_result: Optional[Any],
    ) -> float:
        """Compute grasp success reward."""
        if validation_result is not None:
            if hasattr(validation_result, 'metrics'):
                return 1.0 if validation_result.metrics.grasp_success else 0.0

        # Check gripper closed during grasp phase
        if not hasattr(trajectory, 'states'):
            return 0.0

        grasp_states = [
            s for s in trajectory.states
            if hasattr(s, 'phase') and s.phase.value in ["grasp", "lift", "transport"]
        ]

        if not grasp_states:
            return 0.0

        # Check if gripper is closed during grasp phases
        closed_count = sum(
            1 for s in grasp_states
            if hasattr(s, 'gripper_position') and s.gripper_position < 0.02
        )

        return closed_count / len(grasp_states)

    def _compute_grasp_stability(
        self,
        trajectory: Any,
        validation_result: Optional[Any],
    ) -> float:
        """Compute grasp stability reward."""
        if validation_result is not None:
            if hasattr(validation_result, 'metrics'):
                # Check for slip events
                slip_events = getattr(validation_result.metrics, 'gripper_slip_events', 0)
                return 1.0 if slip_events == 0 else max(0, 1.0 - slip_events * 0.2)

        # Assume stable if grasp phase has consistent gripper state
        if not hasattr(trajectory, 'states'):
            return 0.5

        grasp_states = [
            s for s in trajectory.states
            if hasattr(s, 'phase') and s.phase.value in ["lift", "transport"]
        ]

        if len(grasp_states) < 2:
            return 0.5

        # Check gripper position variance (lower is better)
        gripper_positions = [
            s.gripper_position for s in grasp_states
            if hasattr(s, 'gripper_position')
        ]

        if len(gripper_positions) < 2:
            return 0.5

        variance = np.var(gripper_positions)
        stability = 1.0 - min(1.0, variance * 100)  # Scale variance

        return stability

    def _compute_placement_success(
        self,
        trajectory: Any,
        motion_plan: Any,
        validation_result: Optional[Any],
    ) -> float:
        """Compute placement success reward."""
        if validation_result is not None:
            if hasattr(validation_result, 'metrics'):
                return 1.0 if validation_result.metrics.placement_success else 0.0

        # Check if place position was specified
        if not hasattr(motion_plan, 'place_position') or motion_plan.place_position is None:
            return 1.0  # No placement required

        # Check final gripper state (should be open after placement)
        if hasattr(trajectory, 'states') and len(trajectory.states) > 0:
            final_state = trajectory.states[-1]
            if hasattr(final_state, 'gripper_position'):
                return 1.0 if final_state.gripper_position > 0.02 else 0.0

        return 0.5

    def _compute_placement_accuracy(
        self,
        trajectory: Any,
        motion_plan: Any,
        scene_objects: Optional[List[Dict[str, Any]]],
    ) -> float:
        """Compute placement accuracy reward based on distance to target."""
        if not hasattr(motion_plan, 'place_position') or motion_plan.place_position is None:
            return 1.0  # No placement required

        target_pos = np.array(motion_plan.place_position)

        # Find EE position at place phase
        if hasattr(trajectory, 'states'):
            place_states = [
                s for s in trajectory.states
                if hasattr(s, 'phase') and s.phase.value in ["place", "release"]
            ]

            if place_states:
                # Use position when object was released
                for state in place_states:
                    if hasattr(state, 'ee_position') and state.ee_position is not None:
                        ee_pos = np.array(state.ee_position)
                        distance = np.linalg.norm(ee_pos - target_pos)

                        # Convert distance to reward (closer = higher)
                        threshold = self.config.placement_accuracy_threshold
                        if distance < threshold:
                            return 1.0
                        else:
                            return max(0, 1.0 - (distance - threshold) / 0.2)

        return 0.5  # Default partial credit

    def _compute_path_efficiency(
        self,
        trajectory: Any,
        motion_plan: Any,
    ) -> float:
        """Compute path efficiency reward (shorter paths preferred)."""
        if not hasattr(trajectory, 'states') or len(trajectory.states) < 2:
            return 0.5

        # Compute path length in joint space
        path_length = 0.0
        for i in range(1, len(trajectory.states)):
            prev = trajectory.states[i - 1].joint_positions
            curr = trajectory.states[i].joint_positions
            path_length += np.linalg.norm(curr - prev)

        # Compute straight-line distance (optimal path lower bound)
        if hasattr(trajectory.states[0], 'ee_position') and hasattr(trajectory.states[-1], 'ee_position'):
            start_pos = trajectory.states[0].ee_position
            end_pos = trajectory.states[-1].ee_position
            if start_pos is not None and end_pos is not None:
                direct_dist = np.linalg.norm(np.array(end_pos) - np.array(start_pos))

                # Efficiency: how close to optimal?
                # Optimal would be some minimum based on robot kinematics
                estimated_optimal = direct_dist * 5  # Rough estimate

                if path_length > 0:
                    efficiency = min(1.0, estimated_optimal / path_length)
                    return efficiency

        # Fallback: penalize very long paths
        expected_frames = 30 * 5  # 5 seconds at 30fps
        frame_count = len(trajectory.states)
        return min(1.0, expected_frames / max(1, frame_count))

    def _compute_time_efficiency(
        self,
        trajectory: Any,
        motion_plan: Any,
    ) -> float:
        """Compute time efficiency reward (faster completion preferred)."""
        if not hasattr(trajectory, 'total_duration'):
            return 0.5

        actual_duration = trajectory.total_duration

        # Expected duration based on motion plan
        if hasattr(motion_plan, 'total_duration'):
            expected_duration = motion_plan.total_duration
        else:
            expected_duration = 5.0  # Default 5 seconds

        # Faster is better, but don't reward being too fast
        ratio = expected_duration / max(0.1, actual_duration)

        if ratio > 1.0:
            # Faster than expected - cap the bonus
            return min(1.0, 0.8 + 0.2 * min(1.0, ratio - 1.0))
        else:
            # Slower than expected - penalize
            return max(0.0, ratio)

    def _compute_collision_penalty(
        self,
        validation_result: Optional[Any],
    ) -> float:
        """Compute collision penalty (inverted: 1 = no collisions)."""
        if validation_result is None:
            return 0.8  # Assume mostly okay

        if hasattr(validation_result, 'metrics'):
            unexpected = getattr(validation_result.metrics, 'unexpected_collisions', 0)
            tolerance = self.config.collision_tolerance

            if unexpected <= tolerance:
                return 1.0
            else:
                # Penalize each collision
                penalty_per_collision = 0.2
                return max(0.0, 1.0 - (unexpected - tolerance) * penalty_per_collision)

        return 0.8

    def _compute_smoothness(
        self,
        trajectory: Any,
    ) -> float:
        """Compute trajectory smoothness reward."""
        if not hasattr(trajectory, 'states') or len(trajectory.states) < 3:
            return 0.5

        # Compute jerk (derivative of acceleration)
        fps = trajectory.fps if hasattr(trajectory, 'fps') else 30.0
        dt = 1.0 / fps

        positions = np.array([s.joint_positions for s in trajectory.states])

        if len(positions) < 4:
            return 0.5

        velocities = np.diff(positions, axis=0) / dt
        accelerations = np.diff(velocities, axis=0) / dt
        jerks = np.diff(accelerations, axis=0) / dt

        # Compute mean absolute jerk
        mean_jerk = np.mean(np.abs(jerks))

        # Normalize: lower jerk = higher smoothness
        # Typical jerk values are 0-1000 rad/s^3
        max_expected_jerk = 500.0
        smoothness = 1.0 - min(1.0, mean_jerk / max_expected_jerk)

        return smoothness

    def _compute_joint_limit_penalty(
        self,
        validation_result: Optional[Any],
    ) -> float:
        """Compute joint limit penalty (inverted: 1 = no violations)."""
        if validation_result is None:
            return 0.8  # Assume mostly okay

        if hasattr(validation_result, 'metrics'):
            violations = getattr(validation_result.metrics, 'joint_limit_violations', 0)

            if violations == 0:
                return 1.0
            else:
                # Penalize each violation
                return max(0.0, 1.0 - violations * 0.1)

        return 0.8


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_episode_reward(
    trajectory: Any,
    motion_plan: Any,
    validation_result: Optional[Any] = None,
    reward_type: str = "shaped",
) -> float:
    """
    Convenience function to compute episode reward.

    Args:
        trajectory: Joint trajectory
        motion_plan: Motion plan
        validation_result: Optional validation result
        reward_type: "sparse", "dense", or "shaped"

    Returns:
        Total reward value (0-1)
    """
    config = RewardConfig(reward_type=RewardType(reward_type))
    computer = RewardComputer(config, verbose=False)

    if reward_type == "sparse":
        return computer.compute_sparse_reward(trajectory, motion_plan, validation_result)
    else:
        reward, _ = computer.compute_episode_reward(
            trajectory, motion_plan, validation_result
        )
        return reward


def compute_dense_rewards(
    trajectory: Any,
    motion_plan: Any,
) -> np.ndarray:
    """Convenience function to compute dense rewards."""
    computer = RewardComputer(verbose=False)
    return computer.compute_dense_rewards(trajectory, motion_plan)


if __name__ == "__main__":
    from motion_planner import AIMotionPlanner
    from trajectory_solver import TrajectorySolver

    print("Testing Reward Computation")
    print("=" * 60)

    # Create test episode
    planner = AIMotionPlanner(robot_type="franka", use_llm=False, verbose=False)
    solver = TrajectorySolver(robot_type="franka", fps=30.0, verbose=False)

    motion_plan = planner.plan_motion(
        task_name="pick_cup",
        task_description="Pick up cup and place on shelf",
        target_object={
            "id": "cup_001",
            "position": [0.5, 0.1, 0.85],
            "dimensions": [0.08, 0.08, 0.12],
        },
        place_position=[0.3, 0.2, 0.9],
    )

    trajectory = solver.solve(motion_plan)

    # Test reward computation
    computer = RewardComputer(verbose=True)

    print("\n--- Shaped Reward ---")
    reward, components = computer.compute_episode_reward(
        trajectory=trajectory,
        motion_plan=motion_plan,
    )
    print(f"Total reward: {reward:.3f}")
    print(f"Components: {components.to_dict()}")

    print("\n--- Sparse Reward ---")
    sparse = computer.compute_sparse_reward(trajectory, motion_plan)
    print(f"Sparse reward: {sparse:.1f}")

    print("\n--- Dense Rewards ---")
    dense = computer.compute_dense_rewards(trajectory, motion_plan)
    print(f"Dense rewards: {len(dense)} steps, mean={np.mean(dense):.3f}")
    print(f"  First 5: {dense[:5]}")
    print(f"  Last 5: {dense[-5:]}")
