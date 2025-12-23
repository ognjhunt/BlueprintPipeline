"""
Reward Function Generator for Isaac Lab tasks.

This module generates reward function implementations for different
policy types and task configurations.
"""

from typing import Any


class RewardFunctionGenerator:
    """
    Generates reward function code for Isaac Lab tasks.

    Reward functions are designed to be compatible with Isaac Lab's
    RewardTermCfg and can be used with the manager-based architecture.
    """

    # Reward function templates
    REWARD_TEMPLATES = {
        "grasp_success": '''
def reward_grasp_success(
    env: ManagerBasedEnv,
    grasp_threshold: float = 0.02,
    success_bonus: float = 10.0,
) -> torch.Tensor:
    """Reward for successful grasp."""
    robot = env.scene["robot"]
    # Check gripper closure and contact
    gripper_pos = robot.data.joint_pos[:, -2:]  # Last 2 DOFs are gripper
    gripper_closed = torch.all(gripper_pos < grasp_threshold, dim=-1)
    # Additional contact check would be needed
    return gripper_closed.float() * success_bonus
''',
        "placement_accuracy": '''
def reward_placement_accuracy(
    env: ManagerBasedEnv,
    target_pos_attr: str = "target_pos",
    threshold: float = 0.05,
) -> torch.Tensor:
    """Reward for accurate object placement."""
    target_pos = getattr(env, target_pos_attr)
    object_pos = env.scene["object"].data.root_pos_w
    dist = torch.norm(target_pos - object_pos, dim=-1)
    return torch.where(dist < threshold, torch.ones_like(dist), 1.0 - torch.tanh(5.0 * dist))
''',
        "collision_penalty": '''
def reward_collision_penalty(
    env: ManagerBasedEnv,
    penalty: float = -1.0,
    force_threshold: float = 50.0,
) -> torch.Tensor:
    """Penalty for collisions."""
    robot = env.scene["robot"]
    contact_forces = robot.data.net_contact_forces_w
    max_force = torch.max(torch.abs(contact_forces), dim=-1)[0]
    has_collision = max_force > force_threshold
    return has_collision.float() * penalty
''',
        "efficiency_bonus": '''
def reward_efficiency_bonus(
    env: ManagerBasedEnv,
    time_scale: float = 0.01,
) -> torch.Tensor:
    """Bonus for completing task efficiently."""
    # Penalize time spent - encourages faster completion
    return -time_scale * torch.ones(env.num_envs, device=env.device)
''',
        "joint_progress": '''
def reward_joint_progress(
    env: ManagerBasedEnv,
    target_pos: float = 1.0,
    articulation_name: str = "articulated_object",
) -> torch.Tensor:
    """Reward for progress on articulation joint."""
    articulation = env.scene[articulation_name]
    joint_pos = articulation.data.joint_pos[:, 0]  # First joint
    progress = joint_pos / target_pos
    return torch.clamp(progress, 0.0, 1.0)
''',
        "handle_grasp": '''
def reward_handle_grasp(
    env: ManagerBasedEnv,
    handle_name: str = "handle",
    grasp_dist_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward for grasping a handle."""
    robot = env.scene["robot"]
    handle = env.scene.get(handle_name)
    if handle is None:
        return torch.zeros(env.num_envs, device=env.device)
    ee_pos = robot.data.body_pos_w[:, robot.find_bodies("panda_hand")[0]]
    handle_pos = handle.data.root_pos_w
    dist = torch.norm(ee_pos - handle_pos, dim=-1)
    return torch.where(dist < grasp_dist_threshold, torch.ones_like(dist), 0.0)
''',
        "smooth_motion": '''
def reward_smooth_motion(
    env: ManagerBasedEnv,
    penalty_scale: float = 0.001,
) -> torch.Tensor:
    """Penalty for jerky motion (large accelerations)."""
    robot = env.scene["robot"]
    joint_acc = robot.data.joint_acc
    return -penalty_scale * torch.sum(joint_acc ** 2, dim=-1)
''',
        # ============================================================================
        # SIM2REAL TRANSFER: Jerk Penalty
        # ============================================================================
        # These rewards are CRITICAL for sim2real transfer. They prevent policies
        # from exploiting simulator artifacts by learning to "vibrate" or use
        # unrealistically high-frequency actions that real actuators cannot execute.
        "action_jerk_penalty": '''
def reward_action_jerk_penalty(
    env: ManagerBasedEnv,
    penalty_scale: float = 0.01,
) -> torch.Tensor:
    """
    Penalty for high-frequency action changes (jerk).

    This is CRITICAL for sim2real transfer because:
    1. Real actuators have rate limits and cannot change instantly
    2. Policies that "vibrate" exploit numerical solver artifacts
    3. Smooth actions transfer better to real hardware

    The penalty is computed as the squared difference between
    consecutive actions, encouraging smooth action trajectories.

    Args:
        env: The environment instance
        penalty_scale: Scale factor for the penalty (default: 0.01)

    Returns:
        Negative reward proportional to action jerk
    """
    # Get current and previous actions
    current_actions = env.action_manager.action
    prev_actions = getattr(env, "_prev_actions", current_actions)

    # Store current actions for next step
    env._prev_actions = current_actions.clone()

    # Compute action difference (jerk proxy)
    action_diff = current_actions - prev_actions

    # L2 norm of action change
    jerk = torch.sum(action_diff ** 2, dim=-1)

    return -penalty_scale * jerk
''',
        "action_rate_penalty": '''
def reward_action_rate_penalty(
    env: ManagerBasedEnv,
    penalty_scale: float = 0.005,
    rate_limit: float = 0.1,
) -> torch.Tensor:
    """
    Penalty for exceeding action rate limits.

    Encourages actions that stay within realistic actuator rate limits.
    This is complementary to jerk penalty - jerk penalizes acceleration
    while this penalizes exceeding a velocity threshold.

    Args:
        env: The environment instance
        penalty_scale: Scale factor for the penalty
        rate_limit: Maximum allowed action change per step

    Returns:
        Negative reward when action rate exceeds limit
    """
    current_actions = env.action_manager.action
    prev_actions = getattr(env, "_prev_actions_rate", current_actions)
    env._prev_actions_rate = current_actions.clone()

    action_diff = torch.abs(current_actions - prev_actions)

    # Penalize actions that exceed rate limit
    excess = torch.clamp(action_diff - rate_limit, min=0.0)
    penalty = torch.sum(excess ** 2, dim=-1)

    return -penalty_scale * penalty
''',
        "action_magnitude_penalty": '''
def reward_action_magnitude_penalty(
    env: ManagerBasedEnv,
    penalty_scale: float = 0.001,
) -> torch.Tensor:
    """
    Penalty for large action magnitudes.

    Encourages energy-efficient actions and prevents policies
    from learning to saturate actuators constantly.

    Args:
        env: The environment instance
        penalty_scale: Scale factor for the penalty

    Returns:
        Negative reward proportional to action magnitude
    """
    actions = env.action_manager.action
    magnitude = torch.sum(actions ** 2, dim=-1)
    return -penalty_scale * magnitude
''',
        "joint_acceleration_penalty": '''
def reward_joint_acceleration_penalty(
    env: ManagerBasedEnv,
    penalty_scale: float = 0.0001,
    max_acc: float = 100.0,
) -> torch.Tensor:
    """
    Penalty for high joint accelerations.

    High accelerations indicate jerky motion that:
    1. Causes wear on real robot hardware
    2. Can destabilize manipulation tasks
    3. Often indicates policy is exploiting simulator

    Args:
        env: The environment instance
        penalty_scale: Scale factor for the penalty
        max_acc: Maximum expected acceleration (for normalization)

    Returns:
        Negative reward proportional to joint acceleration
    """
    robot = env.scene["robot"]
    joint_acc = robot.data.joint_acc

    # Normalize by max expected acceleration
    normalized_acc = joint_acc / max_acc

    # Sum of squared accelerations
    acc_penalty = torch.sum(normalized_acc ** 2, dim=-1)

    return -penalty_scale * acc_penalty
''',
        "smooth_ee_velocity": '''
def reward_smooth_ee_velocity(
    env: ManagerBasedEnv,
    penalty_scale: float = 0.005,
    max_velocity: float = 1.0,
) -> torch.Tensor:
    """
    Penalty for high end-effector velocity changes.

    Encourages smooth end-effector motion which:
    1. Is safer around objects and humans
    2. Produces more reliable grasping
    3. Transfers better to real hardware

    Args:
        env: The environment instance
        penalty_scale: Scale factor for the penalty
        max_velocity: Maximum expected EE velocity (m/s)

    Returns:
        Negative reward for high EE velocity changes
    """
    robot = env.scene["robot"]

    # Get end-effector velocity
    ee_body_idx = robot.find_bodies("panda_hand")[0] if hasattr(robot, "find_bodies") else -1
    ee_vel = robot.data.body_vel_w[:, ee_body_idx, :3]  # Linear velocity

    # Store previous velocity
    prev_ee_vel = getattr(env, "_prev_ee_vel", ee_vel)
    env._prev_ee_vel = ee_vel.clone()

    # Compute velocity change (acceleration proxy)
    vel_change = ee_vel - prev_ee_vel
    vel_change_mag = torch.norm(vel_change, dim=-1)

    return -penalty_scale * vel_change_mag
''',
        "task_completion": '''
def reward_task_completion(
    env: ManagerBasedEnv,
    completion_bonus: float = 100.0,
) -> torch.Tensor:
    """Large bonus for task completion."""
    # This would check task-specific completion criteria
    completed = getattr(env, "task_completed", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return completed.float() * completion_bonus
''',
        "dish_placed": '''
def reward_dish_placed(
    env: ManagerBasedEnv,
    rack_region: tuple = ((-0.2, 0.2), (-0.2, 0.2), (0.0, 0.1)),
    reward_per_dish: float = 5.0,
) -> torch.Tensor:
    """Reward for placing dishes in dishwasher rack."""
    dishes = env.scene.get("dishes")
    if dishes is None:
        return torch.zeros(env.num_envs, device=env.device)
    dish_pos = dishes.data.root_pos_w
    in_rack = (
        (dish_pos[:, :, 0] > rack_region[0][0]) & (dish_pos[:, :, 0] < rack_region[0][1]) &
        (dish_pos[:, :, 1] > rack_region[1][0]) & (dish_pos[:, :, 1] < rack_region[1][1]) &
        (dish_pos[:, :, 2] > rack_region[2][0]) & (dish_pos[:, :, 2] < rack_region[2][1])
    )
    return torch.sum(in_rack.float(), dim=-1) * reward_per_dish
''',
        "sorting_accuracy": '''
def reward_sorting_accuracy(
    env: ManagerBasedEnv,
    correct_bin_reward: float = 5.0,
) -> torch.Tensor:
    """Reward for correct sorting of items."""
    # Would check item positions vs correct bin locations
    return torch.zeros(env.num_envs, device=env.device)
''',
        "rotation_accuracy": '''
def reward_rotation_accuracy(
    env: ManagerBasedEnv,
    target_rotation: float = 0.0,
    tolerance: float = 0.1,
) -> torch.Tensor:
    """Reward for accurate rotation of knob/dial."""
    articulation = env.scene.get("knob")
    if articulation is None:
        return torch.zeros(env.num_envs, device=env.device)
    joint_pos = articulation.data.joint_pos[:, 0]
    error = torch.abs(joint_pos - target_rotation)
    return torch.where(error < tolerance, torch.ones_like(error), 1.0 - torch.tanh(error))
''',
    }

    def __init__(self):
        pass

    def get_reward_function(self, component: str) -> str:
        """Get reward function code for a component."""
        return self.REWARD_TEMPLATES.get(component, self._generate_default_reward(component))

    def _generate_default_reward(self, component: str) -> str:
        """Generate a default reward function stub."""
        return f'''
def reward_{component}(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Reward for {component.replace('_', ' ')}."""
    # TODO: Implement {component} reward
    return torch.zeros(env.num_envs, device=env.device)
'''

    def generate_reward_module(
        self,
        components: list[str],
        weights: dict[str, float]
    ) -> str:
        """Generate a complete reward module with all components."""
        header = '''"""
Reward Functions Module
Generated by BlueprintRecipe

This module contains reward function implementations for the task.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

'''
        functions = []
        for component in components:
            func = self.get_reward_function(component)
            functions.append(func)

        # Add combined reward function
        combined = self._generate_combined_reward(components, weights)
        functions.append(combined)

        return header + "\n".join(functions)

    def _generate_combined_reward(
        self,
        components: list[str],
        weights: dict[str, float]
    ) -> str:
        """Generate combined reward function."""
        weight_lines = "\n".join(
            f'        "{c}": {weights.get(c, 1.0)},'
            for c in components
        )

        return f'''
def compute_combined_reward(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Compute weighted sum of all reward components."""
    weights = {{
{weight_lines}
    }}

    total_reward = torch.zeros(env.num_envs, device=env.device)

    for component, weight in weights.items():
        func = globals().get(f"reward_{{component}}")
        if func is not None:
            total_reward += weight * func(env)

    return total_reward
'''
