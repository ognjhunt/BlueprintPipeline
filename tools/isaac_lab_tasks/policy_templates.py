"""
Policy Templates for Isaac Lab Task Generation.

This module provides policy-specific templates for:
- Reward functions (dense shaping + sparse bonuses)
- Success metrics
- Termination conditions
- Observation requirements

These templates align with the PolicyTarget enum from replicator-job.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class PolicyTarget(str, Enum):
    """Policy targets (mirrors replicator-job)."""
    DEXTEROUS_PICK_PLACE = "dexterous_pick_place"
    ARTICULATED_ACCESS = "articulated_access"
    PANEL_INTERACTION = "panel_interaction"
    MIXED_SKU_LOGISTICS = "mixed_sku_logistics"
    PRECISION_INSERTION = "precision_insertion"
    LAUNDRY_SORTING = "laundry_sorting"
    DISH_LOADING = "dish_loading"
    GROCERY_STOCKING = "grocery_stocking"
    TABLE_CLEARING = "table_clearing"
    DRAWER_MANIPULATION = "drawer_manipulation"
    DOOR_MANIPULATION = "door_manipulation"
    KNOB_MANIPULATION = "knob_manipulation"
    GENERAL_MANIPULATION = "general_manipulation"


@dataclass
class RewardTemplate:
    """Template for a reward function."""
    name: str
    description: str
    weight: float = 1.0
    is_sparse: bool = False  # Sparse vs dense reward
    function_code: str = ""


@dataclass
class TerminationTemplate:
    """Template for a termination condition."""
    name: str
    description: str
    is_success: bool = False  # Success termination vs failure
    time_based: bool = False
    function_code: str = ""


@dataclass
class SuccessMetric:
    """Template for a success metric."""
    name: str
    description: str
    threshold: float = 0.0
    duration_steps: int = 1  # How many steps condition must hold


@dataclass
class PolicyTemplate:
    """Complete template for a policy."""
    policy_id: str
    display_name: str
    description: str

    # Rewards
    rewards: List[RewardTemplate] = field(default_factory=list)

    # Success metrics
    success_metrics: List[SuccessMetric] = field(default_factory=list)

    # Terminations
    terminations: List[TerminationTemplate] = field(default_factory=list)

    # Required observations
    required_observations: List[str] = field(default_factory=list)

    # Control mode preference
    control_mode: str = "joint_velocity"

    # Episode settings
    episode_length: int = 500


# =============================================================================
# Policy-Specific Templates
# =============================================================================


DISH_LOADING_TEMPLATE = PolicyTemplate(
    policy_id="dish_loading",
    display_name="Dish Loading",
    description="Load dishes into a dishwasher rack",
    rewards=[
        RewardTemplate(
            name="reach_dish",
            description="Dense reward for reaching the dish",
            weight=0.5,
            function_code='''
def reward_reach_dish(env, ee_body: str = "panda_hand", distance_scale: float = 5.0):
    """Dense reaching reward towards dish."""
    robot = env.scene.get("robot")
    dish = env.scene.get("dish") or env.scene.get("target_object")
    if robot is None or dish is None:
        return torch.zeros(env.num_envs, device=env.device)
    ee_pos = robot.data.body_pos_w[:, robot.find_bodies(ee_body)[0]]
    dish_pos = dish.data.root_pos_w
    dist = torch.norm(dish_pos - ee_pos, dim=-1)
    return 1.0 - torch.tanh(distance_scale * dist)
'''
        ),
        RewardTemplate(
            name="grasp_dish",
            description="Reward for successfully grasping the dish",
            weight=2.0,
            is_sparse=True,
            function_code='''
def reward_grasp_dish(env, gripper_threshold: float = 0.02):
    """Sparse reward for grasping dish."""
    robot = env.scene.get("robot")
    dish = env.scene.get("dish") or env.scene.get("target_object")
    if robot is None or dish is None:
        return torch.zeros(env.num_envs, device=env.device)
    # Check gripper is closed and dish is lifted
    gripper_pos = robot.data.joint_pos[:, -2:]
    gripper_closed = torch.abs(gripper_pos).sum(dim=-1) < gripper_threshold
    dish_lifted = dish.data.root_pos_w[:, 2] > 0.1
    return (gripper_closed & dish_lifted).float() * 2.0
'''
        ),
        RewardTemplate(
            name="dish_in_rack",
            description="Dense reward for moving dish toward rack",
            weight=1.0,
            function_code='''
def reward_dish_in_rack(env, rack_position: tuple = (0.5, 0.0, 0.4), distance_scale: float = 3.0):
    """Dense reward for positioning dish in rack."""
    dish = env.scene.get("dish") or env.scene.get("target_object")
    if dish is None:
        return torch.zeros(env.num_envs, device=env.device)
    rack_pos = torch.tensor(rack_position, device=env.device).unsqueeze(0)
    dish_pos = dish.data.root_pos_w
    dist = torch.norm(dish_pos - rack_pos, dim=-1)
    return 1.0 - torch.tanh(distance_scale * dist)
'''
        ),
        RewardTemplate(
            name="task_success",
            description="Sparse bonus for completing the task",
            weight=10.0,
            is_sparse=True,
            function_code='''
def reward_task_success(env, bonus: float = 10.0):
    """Sparse success bonus."""
    success = getattr(env, "task_success", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return success.float() * bonus
'''
        ),
    ],
    success_metrics=[
        SuccessMetric(
            name="dish_in_rack_volume",
            description="Dish is within the rack volume",
            threshold=0.05,
            duration_steps=10,
        ),
    ],
    terminations=[
        TerminationTemplate(
            name="dish_dropped",
            description="Dish dropped below workspace",
            function_code='''
def termination_dish_dropped(env, threshold: float = -0.05):
    """Terminate if dish falls below floor."""
    dish = env.scene.get("dish") or env.scene.get("target_object")
    if dish is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return dish.data.root_pos_w[:, 2] < threshold
'''
        ),
        TerminationTemplate(
            name="task_success",
            description="Task completed successfully",
            is_success=True,
            function_code='''
def termination_task_success(env):
    """Terminate on success."""
    return getattr(env, "task_success", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
'''
        ),
    ],
    required_observations=["ee_pos", "dish_pos", "dish_quat", "rack_pos", "gripper_pos"],
    control_mode="ee_pose",
    episode_length=500,
)


DRAWER_MANIPULATION_TEMPLATE = PolicyTemplate(
    policy_id="drawer_manipulation",
    display_name="Drawer Manipulation",
    description="Open/close drawer using handle",
    rewards=[
        RewardTemplate(
            name="reach_handle",
            description="Dense reward for reaching drawer handle",
            weight=0.5,
            function_code='''
def reward_reach_handle(env, ee_body: str = "panda_hand", distance_scale: float = 5.0):
    """Dense reaching reward towards drawer handle."""
    robot = env.scene.get("robot")
    drawer = env.scene.get("drawer") or env.scene.get("articulated_object")
    if robot is None or drawer is None:
        return torch.zeros(env.num_envs, device=env.device)
    ee_pos = robot.data.body_pos_w[:, robot.find_bodies(ee_body)[0]]
    # Handle position is typically at front of drawer
    handle_pos = drawer.data.body_pos_w[:, -1]  # Last link is usually handle
    dist = torch.norm(handle_pos - ee_pos, dim=-1)
    return 1.0 - torch.tanh(distance_scale * dist)
'''
        ),
        RewardTemplate(
            name="drawer_progress",
            description="Reward for drawer movement progress",
            weight=2.0,
            function_code='''
def reward_drawer_progress(env, target_position: float = 0.4):
    """Dense reward for drawer opening progress."""
    drawer = env.scene.get("drawer") or env.scene.get("articulated_object")
    if drawer is None:
        return torch.zeros(env.num_envs, device=env.device)
    joint_pos = drawer.data.joint_pos[:, 0]  # First joint is drawer slide
    progress = joint_pos / target_position
    return torch.clamp(progress, 0.0, 1.0)
'''
        ),
        RewardTemplate(
            name="alignment_reward",
            description="Reward for gripper alignment with handle",
            weight=0.3,
            function_code='''
def reward_alignment(env, ee_body: str = "panda_hand"):
    """Reward for proper gripper alignment."""
    robot = env.scene.get("robot")
    if robot is None:
        return torch.zeros(env.num_envs, device=env.device)
    # Get end-effector orientation
    ee_quat = robot.data.body_quat_w[:, robot.find_bodies(ee_body)[0]]
    # Reward for gripper facing forward (z-axis alignment)
    z_axis = torch.tensor([0, 0, 1], device=env.device, dtype=torch.float32)
    # Simplified alignment check
    return torch.abs(ee_quat[:, 0])  # w component close to 1 means upright
'''
        ),
        RewardTemplate(
            name="task_success",
            description="Bonus for drawer fully open/closed",
            weight=10.0,
            is_sparse=True,
        ),
    ],
    success_metrics=[
        SuccessMetric(
            name="drawer_at_target",
            description="Drawer joint within epsilon of target",
            threshold=0.02,
            duration_steps=5,
        ),
    ],
    terminations=[
        TerminationTemplate(
            name="excessive_force",
            description="Applied force exceeds safe limit",
            function_code='''
def termination_excessive_force(env, force_limit: float = 100.0):
    """Terminate if excessive force applied."""
    robot = env.scene.get("robot")
    if robot is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Check joint torques
    torques = torch.abs(robot.data.applied_torque).sum(dim=-1)
    return torques > force_limit
'''
        ),
        TerminationTemplate(
            name="task_success",
            description="Drawer at target position",
            is_success=True,
        ),
    ],
    required_observations=["ee_pos", "handle_pos", "joint_pos", "joint_vel", "gripper_pos"],
    control_mode="ee_pose",
    episode_length=400,
)


DOOR_MANIPULATION_TEMPLATE = PolicyTemplate(
    policy_id="door_manipulation",
    display_name="Door Manipulation",
    description="Open/close cabinet or room doors",
    rewards=[
        RewardTemplate(
            name="reach_handle",
            description="Dense reward for reaching door handle",
            weight=0.5,
        ),
        RewardTemplate(
            name="door_angle_progress",
            description="Reward for door opening angle",
            weight=2.0,
            function_code='''
def reward_door_angle_progress(env, target_angle: float = 1.57):
    """Dense reward for door opening progress."""
    door = env.scene.get("door") or env.scene.get("articulated_object")
    if door is None:
        return torch.zeros(env.num_envs, device=env.device)
    joint_pos = door.data.joint_pos[:, 0]
    progress = joint_pos / target_angle
    return torch.clamp(progress, 0.0, 1.0)
'''
        ),
        RewardTemplate(
            name="task_success",
            description="Bonus for door fully open/closed",
            weight=10.0,
            is_sparse=True,
        ),
    ],
    success_metrics=[
        SuccessMetric(
            name="door_at_target_angle",
            description="Door angle within epsilon of target",
            threshold=0.05,
            duration_steps=5,
        ),
    ],
    terminations=[
        TerminationTemplate(name="task_success", is_success=True),
    ],
    required_observations=["ee_pos", "handle_pos", "joint_pos", "joint_vel"],
    control_mode="ee_pose",
    episode_length=400,
)


DEXTEROUS_PICK_PLACE_TEMPLATE = PolicyTemplate(
    policy_id="dexterous_pick_place",
    display_name="Dexterous Pick and Place",
    description="Pick up object and place at target location",
    rewards=[
        RewardTemplate(
            name="reach_object",
            description="Dense reward for reaching object",
            weight=0.5,
        ),
        RewardTemplate(
            name="grasp_success",
            description="Reward for successful grasp",
            weight=2.0,
            is_sparse=True,
        ),
        RewardTemplate(
            name="lift_object",
            description="Reward for lifting object",
            weight=1.0,
            function_code='''
def reward_lift_object(env, min_height: float = 0.1, max_height: float = 0.3):
    """Dense reward for lifting object."""
    obj = env.scene.get("target_object")
    if obj is None:
        return torch.zeros(env.num_envs, device=env.device)
    height = obj.data.root_pos_w[:, 2]
    normalized = (height - min_height) / (max_height - min_height)
    return torch.clamp(normalized, 0.0, 1.0)
'''
        ),
        RewardTemplate(
            name="reach_target",
            description="Dense reward for moving object to target",
            weight=1.5,
        ),
        RewardTemplate(
            name="place_success",
            description="Bonus for successful placement",
            weight=10.0,
            is_sparse=True,
        ),
    ],
    success_metrics=[
        SuccessMetric(
            name="object_at_target",
            description="Object within placement threshold",
            threshold=0.03,
            duration_steps=10,
        ),
    ],
    terminations=[
        TerminationTemplate(name="object_dropped", description="Object dropped"),
        TerminationTemplate(name="task_success", is_success=True),
    ],
    required_observations=["ee_pos", "object_pos", "object_quat", "target_pos", "gripper_pos"],
    control_mode="ee_pose",
    episode_length=500,
)


TABLE_CLEARING_TEMPLATE = PolicyTemplate(
    policy_id="table_clearing",
    display_name="Table Clearing",
    description="Clear objects from table surface",
    rewards=[
        RewardTemplate(name="reach_object", weight=0.3),
        RewardTemplate(name="grasp_success", weight=1.5, is_sparse=True),
        RewardTemplate(
            name="objects_cleared",
            description="Reward based on number of objects cleared",
            weight=3.0,
            function_code='''
def reward_objects_cleared(env, target_zone_bounds: tuple = (-0.5, -0.5, 0.5, 0.5)):
    """Reward for objects moved outside table zone."""
    objects = env.scene.get("objects")
    if objects is None:
        return torch.zeros(env.num_envs, device=env.device)
    positions = objects.data.root_pos_w
    in_zone_x = (positions[:, :, 0] > target_zone_bounds[0]) & (positions[:, :, 0] < target_zone_bounds[2])
    in_zone_y = (positions[:, :, 1] > target_zone_bounds[1]) & (positions[:, :, 1] < target_zone_bounds[3])
    in_zone = in_zone_x & in_zone_y
    cleared_count = (~in_zone).sum(dim=-1).float()
    return cleared_count / objects.num_instances
'''
        ),
        RewardTemplate(name="task_success", weight=10.0, is_sparse=True),
    ],
    success_metrics=[
        SuccessMetric(
            name="all_objects_cleared",
            description="All objects cleared from table",
            duration_steps=5,
        ),
    ],
    required_observations=["ee_pos", "object_positions", "gripper_pos"],
    episode_length=600,
)


GROCERY_STOCKING_TEMPLATE = PolicyTemplate(
    policy_id="grocery_stocking",
    display_name="Grocery Stocking",
    description="Stock grocery items on shelves",
    rewards=[
        RewardTemplate(name="reach_item", weight=0.3),
        RewardTemplate(name="grasp_item", weight=1.0, is_sparse=True),
        RewardTemplate(
            name="item_on_shelf",
            description="Reward for placing item on target shelf",
            weight=3.0,
            function_code='''
def reward_item_on_shelf(env, shelf_height: float = 1.2, tolerance: float = 0.1):
    """Reward for items placed on shelf."""
    item = env.scene.get("target_object")
    if item is None:
        return torch.zeros(env.num_envs, device=env.device)
    height = item.data.root_pos_w[:, 2]
    on_shelf = torch.abs(height - shelf_height) < tolerance
    return on_shelf.float() * 3.0
'''
        ),
        RewardTemplate(name="task_success", weight=10.0, is_sparse=True),
    ],
    success_metrics=[
        SuccessMetric(name="item_placed_on_shelf", threshold=0.05, duration_steps=10),
    ],
    required_observations=["ee_pos", "item_pos", "item_quat", "shelf_pos", "gripper_pos"],
    episode_length=500,
)


KNOB_MANIPULATION_TEMPLATE = PolicyTemplate(
    policy_id="knob_manipulation",
    display_name="Knob Manipulation",
    description="Rotate knobs/dials",
    rewards=[
        RewardTemplate(name="reach_knob", weight=0.5),
        RewardTemplate(
            name="knob_rotation_progress",
            description="Reward for knob rotation",
            weight=2.0,
            function_code='''
def reward_knob_rotation(env, target_rotation: float = 3.14):
    """Dense reward for knob rotation progress."""
    knob = env.scene.get("knob") or env.scene.get("articulated_object")
    if knob is None:
        return torch.zeros(env.num_envs, device=env.device)
    joint_pos = knob.data.joint_pos[:, 0]
    progress = torch.abs(joint_pos) / abs(target_rotation)
    return torch.clamp(progress, 0.0, 1.0)
'''
        ),
        RewardTemplate(name="task_success", weight=10.0, is_sparse=True),
    ],
    success_metrics=[
        SuccessMetric(name="knob_at_target", threshold=0.1, duration_steps=3),
    ],
    required_observations=["ee_pos", "knob_angle", "gripper_pos"],
    control_mode="joint_velocity",
    episode_length=300,
)


# Registry of all templates
POLICY_TEMPLATES: Dict[str, PolicyTemplate] = {
    "dish_loading": DISH_LOADING_TEMPLATE,
    "drawer_manipulation": DRAWER_MANIPULATION_TEMPLATE,
    "door_manipulation": DOOR_MANIPULATION_TEMPLATE,
    "dexterous_pick_place": DEXTEROUS_PICK_PLACE_TEMPLATE,
    "table_clearing": TABLE_CLEARING_TEMPLATE,
    "grocery_stocking": GROCERY_STOCKING_TEMPLATE,
    "knob_manipulation": KNOB_MANIPULATION_TEMPLATE,
}


def get_policy_template(policy_id: str) -> Optional[PolicyTemplate]:
    """Get template for a policy."""
    return POLICY_TEMPLATES.get(policy_id)


def get_reward_weights(policy_id: str) -> Dict[str, float]:
    """Get reward weights for a policy."""
    template = get_policy_template(policy_id)
    if template is None:
        return {"reaching": 1.0, "task_success": 10.0}
    return {r.name: r.weight for r in template.rewards}


def get_required_observations(policy_id: str) -> List[str]:
    """Get required observations for a policy."""
    template = get_policy_template(policy_id)
    if template is None:
        return ["ee_pos", "gripper_pos"]
    return template.required_observations


def generate_reward_functions_code(policy_id: str) -> str:
    """Generate reward functions code for a policy."""
    template = get_policy_template(policy_id)
    if template is None:
        return ""

    code_blocks = []
    for reward in template.rewards:
        if reward.function_code:
            code_blocks.append(reward.function_code.strip())

    return "\n\n".join(code_blocks)


def generate_termination_functions_code(policy_id: str) -> str:
    """Generate termination functions code for a policy."""
    template = get_policy_template(policy_id)
    if template is None:
        return ""

    code_blocks = []
    for term in template.terminations:
        if term.function_code:
            code_blocks.append(term.function_code.strip())

    return "\n\n".join(code_blocks)
