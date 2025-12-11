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
