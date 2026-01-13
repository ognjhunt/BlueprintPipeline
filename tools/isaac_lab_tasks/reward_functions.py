"""
Reward Function Generator for Isaac Lab tasks.

This module generates reward function implementations for different
policy types and task configurations.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


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
        "rack_utilization": '''
def reward_rack_utilization(
    env: ManagerBasedEnv,
    rack_region: tuple = ((-0.25, 0.25), (-0.25, 0.25), (0.0, 0.15)),
    dish_entity: str = "dishes",
) -> torch.Tensor:
    """Reward for maximizing dish occupancy within the rack volume."""
    dishes = env.scene.get(dish_entity)
    if dishes is None:
        return torch.zeros(env.num_envs, device=env.device)
    dish_pos = dishes.data.root_pos_w
    in_rack = (
        (dish_pos[:, :, 0] > rack_region[0][0]) & (dish_pos[:, :, 0] < rack_region[0][1]) &
        (dish_pos[:, :, 1] > rack_region[1][0]) & (dish_pos[:, :, 1] < rack_region[1][1]) &
        (dish_pos[:, :, 2] > rack_region[2][0]) & (dish_pos[:, :, 2] < rack_region[2][1])
    )
    rack_ratio = torch.mean(in_rack.float(), dim=-1)
    return rack_ratio
''',
        "breakage_penalty": '''
def reward_breakage_penalty(
    env: ManagerBasedEnv,
    fragile_entity: str = "dishes",
    force_threshold: float = 75.0,
    penalty_scale: float = 0.02,
) -> torch.Tensor:
    """Penalty for excessive forces applied to fragile objects."""
    fragile = env.scene.get(fragile_entity)
    if fragile is None or not hasattr(fragile.data, "net_contact_forces_w"):
        return torch.zeros(env.num_envs, device=env.device)
    forces = fragile.data.net_contact_forces_w
    force_mag = torch.norm(forces, dim=-1)
    max_force = torch.max(force_mag, dim=-1)[0]
    excess = torch.clamp(max_force - force_threshold, min=0.0)
    return -penalty_scale * excess
''',
        "items_cleared": '''
def reward_items_cleared(
    env: ManagerBasedEnv,
    cleared_region: tuple = ((-0.4, 0.4), (-0.4, 0.4), (0.2, 1.0)),
    item_entity: str = "items",
) -> torch.Tensor:
    """Reward for moving items into a cleared/collection region."""
    items = env.scene.get(item_entity)
    if items is None:
        return torch.zeros(env.num_envs, device=env.device)
    item_pos = items.data.root_pos_w
    in_region = (
        (item_pos[:, :, 0] > cleared_region[0][0]) & (item_pos[:, :, 0] < cleared_region[0][1]) &
        (item_pos[:, :, 1] > cleared_region[1][0]) & (item_pos[:, :, 1] < cleared_region[1][1]) &
        (item_pos[:, :, 2] > cleared_region[2][0]) & (item_pos[:, :, 2] < cleared_region[2][1])
    )
    return torch.mean(in_region.float(), dim=-1)
''',
        "sorting_accuracy": '''
def reward_sorting_accuracy(
    env: ManagerBasedEnv,
    item_entity: str = "items",
    bin_entity: str = "bins",
    distance_scale: float = 10.0,
    success_threshold: float = 0.08,
) -> torch.Tensor:
    """Reward for placing items into their assigned bins."""
    items = env.scene.get(item_entity)
    bins = env.scene.get(bin_entity)
    item_bin_ids = getattr(env, "item_bin_ids", None)
    if items is None or bins is None or item_bin_ids is None:
        return torch.zeros(env.num_envs, device=env.device)
    item_pos = items.data.root_pos_w
    bin_pos = bins.data.root_pos_w
    if item_bin_ids.dim() == 1:
        item_bin_ids = item_bin_ids.unsqueeze(0).repeat(env.num_envs, 1)
    gather_idx = item_bin_ids.unsqueeze(-1).expand(-1, -1, bin_pos.shape[-1])
    target_bins = torch.gather(bin_pos, 1, gather_idx)
    dist = torch.norm(item_pos - target_bins, dim=-1)
    shaped = 1.0 - torch.tanh(distance_scale * dist)
    success = (dist < success_threshold).float()
    return torch.mean(shaped + success, dim=-1)
''',
        "fold_quality": '''
def reward_fold_quality(
    env: ManagerBasedEnv,
    keypoint_scale: float = 10.0,
) -> torch.Tensor:
    """Reward for aligning cloth keypoints to target folded positions."""
    keypoints = getattr(env, "cloth_keypoints", None)
    target_keypoints = getattr(env, "cloth_target_keypoints", None)
    if keypoints is None or target_keypoints is None:
        return torch.zeros(env.num_envs, device=env.device)
    error = torch.norm(keypoints - target_keypoints, dim=-1)
    return torch.mean(1.0 - torch.tanh(keypoint_scale * error), dim=-1)
''',
        "stacking_stability": '''
def reward_stacking_stability(
    env: ManagerBasedEnv,
    stack_entity: str = "stacked_items",
    velocity_threshold: float = 0.2,
    min_height: float = 0.05,
) -> torch.Tensor:
    """Reward stable stacks with low motion and sufficient height."""
    items = env.scene.get(stack_entity)
    if items is None:
        return torch.zeros(env.num_envs, device=env.device)
    pos = items.data.root_pos_w
    vel = getattr(items.data, "root_vel_w", None)
    if vel is None:
        vel = torch.zeros_like(pos)
    speed = torch.norm(vel[..., :3], dim=-1)
    stable = (speed < velocity_threshold) & (pos[..., 2] > min_height)
    return torch.mean(stable.float(), dim=-1)
''',
        "throughput": '''
def reward_throughput(
    env: ManagerBasedEnv,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward tasks completed per episode step."""
    completed = getattr(env, "tasks_completed", None)
    if completed is None:
        completed = getattr(env, "completed_count", torch.zeros(env.num_envs, device=env.device))
    return completed.float() * scale
''',
        "shelf_utilization": '''
def reward_shelf_utilization(
    env: ManagerBasedEnv,
    shelf_region: tuple = ((-0.5, 0.5), (-0.2, 0.2), (0.8, 1.6)),
    product_entity: str = "products",
) -> torch.Tensor:
    """Reward for placing products within shelf bounds."""
    products = env.scene.get(product_entity)
    if products is None:
        return torch.zeros(env.num_envs, device=env.device)
    pos = products.data.root_pos_w
    in_shelf = (
        (pos[:, :, 0] > shelf_region[0][0]) & (pos[:, :, 0] < shelf_region[0][1]) &
        (pos[:, :, 1] > shelf_region[1][0]) & (pos[:, :, 1] < shelf_region[1][1]) &
        (pos[:, :, 2] > shelf_region[2][0]) & (pos[:, :, 2] < shelf_region[2][1])
    )
    return torch.mean(in_shelf.float(), dim=-1)
''',
        "facing_accuracy": '''
def reward_facing_accuracy(
    env: ManagerBasedEnv,
    angle_scale: float = 3.0,
) -> torch.Tensor:
    """Reward for aligning product facing direction with target."""
    item_angles = getattr(env, "item_facing_angles", None)
    target_angles = getattr(env, "target_facing_angles", None)
    if item_angles is not None and target_angles is not None:
        error = torch.abs(item_angles - target_angles)
        return torch.mean(1.0 - torch.tanh(angle_scale * error), dim=-1)
    item_vecs = getattr(env, "item_facing_vectors", None)
    target_vecs = getattr(env, "target_facing_vectors", None)
    if item_vecs is None or target_vecs is None:
        return torch.zeros(env.num_envs, device=env.device)
    item_norm = torch.nn.functional.normalize(item_vecs, dim=-1)
    target_norm = torch.nn.functional.normalize(target_vecs, dim=-1)
    cos_sim = torch.sum(item_norm * target_norm, dim=-1)
    return torch.mean((1.0 + cos_sim) * 0.5, dim=-1)
''',
        "product_organization": '''
def reward_product_organization(
    env: ManagerBasedEnv,
    variance_scale: float = 5.0,
) -> torch.Tensor:
    """Reward for organized product placement (low spatial variance)."""
    organization_score = getattr(env, "organization_score", None)
    if organization_score is not None:
        return organization_score
    cluster_variance = getattr(env, "product_cluster_variance", None)
    if cluster_variance is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.exp(-variance_scale * cluster_variance)
''',
        "grasp_stability": '''
def reward_grasp_stability(
    env: ManagerBasedEnv,
    object_entity: str = "object",
    distance_threshold: float = 0.05,
    velocity_scale: float = 2.0,
) -> torch.Tensor:
    """Reward stable grasps with close distance and low relative motion."""
    robot = env.scene.get("robot")
    obj = env.scene.get(object_entity)
    if robot is None or obj is None:
        return torch.zeros(env.num_envs, device=env.device)
    ee_id = robot.find_bodies("panda_hand")[0] if hasattr(robot, "find_bodies") else 0
    ee_pos = robot.data.body_pos_w[:, ee_id]
    obj_pos = obj.data.root_pos_w
    dist = torch.norm(ee_pos - obj_pos, dim=-1)
    obj_vel = getattr(obj.data, "root_vel_w", torch.zeros_like(obj_pos))
    vel_mag = torch.norm(obj_vel[..., :3], dim=-1)
    proximity = torch.where(dist < distance_threshold, 1.0 - dist / distance_threshold, 0.0)
    stability = torch.exp(-velocity_scale * vel_mag)
    return proximity * stability
''',
        "insertion_success": '''
def reward_insertion_success(
    env: ManagerBasedEnv,
    success_bonus: float = 20.0,
) -> torch.Tensor:
    """Reward for successful insertion completion."""
    status = getattr(env, "insertion_success", None)
    if status is None:
        status = getattr(env, "task_success", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return status.float() * success_bonus
''',
        "alignment_accuracy": '''
def reward_alignment_accuracy(
    env: ManagerBasedEnv,
    pos_scale: float = 8.0,
    rot_scale: float = 4.0,
    target_pos_attr: str = "target_pos",
    target_quat_attr: str = "target_quat",
    object_entity: str = "object",
) -> torch.Tensor:
    """Reward for aligning object pose with target pose."""
    target_pos = getattr(env, target_pos_attr, None)
    target_quat = getattr(env, target_quat_attr, None)
    obj = env.scene.get(object_entity)
    if target_pos is None or target_quat is None or obj is None:
        return torch.zeros(env.num_envs, device=env.device)
    obj_pos = obj.data.root_pos_w
    obj_quat = obj.data.root_quat_w
    pos_error = torch.norm(target_pos - obj_pos, dim=-1)
    quat_dot = torch.sum(target_quat * obj_quat, dim=-1).abs()
    rot_error = 1.0 - quat_dot
    return 1.0 - torch.tanh(pos_scale * pos_error + rot_scale * rot_error)
''',
        "force_control": '''
def reward_force_control(
    env: ManagerBasedEnv,
    target_force: float = 20.0,
    penalty_scale: float = 0.02,
) -> torch.Tensor:
    """Reward for maintaining contact forces near a target level."""
    robot = env.scene.get("robot")
    if robot is None or not hasattr(robot.data, "net_contact_forces_w"):
        return torch.zeros(env.num_envs, device=env.device)
    forces = robot.data.net_contact_forces_w
    force_mag = torch.norm(forces, dim=-1)
    avg_force = torch.mean(force_mag, dim=-1)
    error = torch.abs(avg_force - target_force)
    return -penalty_scale * error
''',
        "efficiency": '''
def reward_efficiency(
    env: ManagerBasedEnv,
    time_scale: float = 0.01,
) -> torch.Tensor:
    """Penalty proportional to elapsed episode progress."""
    progress = getattr(env, "progress_buf", None)
    if progress is None:
        return -time_scale * torch.ones(env.num_envs, device=env.device)
    max_steps = getattr(env, "max_episode_length", None)
    if max_steps is None:
        return -time_scale * progress.float()
    return -time_scale * (progress.float() / max_steps)
''',
        "target_activation": '''
def reward_target_activation(
    env: ManagerBasedEnv,
    activation_bonus: float = 5.0,
) -> torch.Tensor:
    """Reward for activating a target switch/button."""
    status = getattr(env, "target_activated", None)
    if status is None:
        status = getattr(env, "target_activation", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return status.float() * activation_bonus
''',
        "precision": '''
def reward_precision(
    env: ManagerBasedEnv,
    error_scale: float = 10.0,
    target_pos_attr: str = "target_pos",
    object_entity: str = "object",
) -> torch.Tensor:
    """Reward precise positioning relative to a target."""
    target_pos = getattr(env, target_pos_attr, None)
    obj = env.scene.get(object_entity)
    if target_pos is None or obj is None:
        return torch.zeros(env.num_envs, device=env.device)
    obj_pos = obj.data.root_pos_w
    error = torch.norm(target_pos - obj_pos, dim=-1)
    return 1.0 - torch.tanh(error_scale * error)
''',
        "sequence_completion": '''
def reward_sequence_completion(
    env: ManagerBasedEnv,
    bonus_scale: float = 1.0,
) -> torch.Tensor:
    """Reward based on progress through a task sequence."""
    progress = getattr(env, "sequence_progress", None)
    if progress is not None:
        return progress.float() * bonus_scale
    step = getattr(env, "sequence_step", None)
    length = getattr(env, "sequence_length", None)
    if step is None or length is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (step.float() / length) * bonus_scale
''',
        "transport_success": '''
def reward_transport_success(
    env: ManagerBasedEnv,
    payload_entity: str = "payload",
    destination_region: tuple = ((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.2)),
    success_bonus: float = 10.0,
) -> torch.Tensor:
    """Reward for delivering payloads to destination region."""
    payload = env.scene.get(payload_entity)
    if payload is None:
        return torch.zeros(env.num_envs, device=env.device)
    pos = payload.data.root_pos_w
    in_dest = (
        (pos[:, 0] > destination_region[0][0]) & (pos[:, 0] < destination_region[0][1]) &
        (pos[:, 1] > destination_region[1][0]) & (pos[:, 1] < destination_region[1][1]) &
        (pos[:, 2] > destination_region[2][0]) & (pos[:, 2] < destination_region[2][1])
    )
    return in_dest.float() * success_bonus
''',
        "load_stability": '''
def reward_load_stability(
    env: ManagerBasedEnv,
    payload_entity: str = "payload",
    velocity_scale: float = 2.0,
    angular_scale: float = 1.0,
) -> torch.Tensor:
    """Reward stable payload motion during transport."""
    payload = env.scene.get(payload_entity)
    if payload is None:
        return torch.zeros(env.num_envs, device=env.device)
    vel = getattr(payload.data, "root_vel_w", None)
    if vel is None:
        return torch.ones(env.num_envs, device=env.device)
    lin_speed = torch.norm(vel[..., :3], dim=-1)
    ang_speed = torch.norm(vel[..., 3:], dim=-1) if vel.shape[-1] > 3 else torch.zeros_like(lin_speed)
    return torch.exp(-velocity_scale * lin_speed - angular_scale * ang_speed)
''',
        "navigation_efficiency": '''
def reward_navigation_efficiency(
    env: ManagerBasedEnv,
    scale: float = 1.0,
) -> torch.Tensor:
    """Reward navigation progress toward goal."""
    remaining = getattr(env, "nav_distance_remaining", None)
    start = getattr(env, "nav_distance_start", None)
    if remaining is None or start is None:
        return torch.zeros(env.num_envs, device=env.device)
    progress = 1.0 - (remaining / (start + 1e-6))
    return progress * scale
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

    def validate_components(self, components: list[str]) -> None:
        """Ensure reward components map to concrete implementations."""
        missing = self.get_missing_components(components)
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                "Reward components resolve to default stubs: "
                f"{missing_str}"
            )

    def get_missing_components(self, components: list[str]) -> list[str]:
        """Return reward components that would fall back to the default stub."""
        missing = set(components) - set(self.REWARD_TEMPLATES)
        return sorted(missing)

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
        self.validate_components(components)
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


def initialize_reward_state(env: ManagerBasedEnv) -> None:
    """
    Initialize state variables required by reward functions.

    This function should be called during environment initialization
    to prevent AttributeError on first timestep.

    Args:
        env: The environment instance to initialize
    """
    # Initialize action history for jerk/rate penalties
    if hasattr(env, "action_manager") and env.action_manager.action is not None:
        env._prev_actions = env.action_manager.action.clone()
        env._prev_actions_rate = env.action_manager.action.clone()
    else:
        # Fallback if action_manager not yet initialized
        action_dim = 7  # Default for most manipulation robots
        env._prev_actions = torch.zeros(env.num_envs, action_dim, device=env.device)
        env._prev_actions_rate = torch.zeros(env.num_envs, action_dim, device=env.device)

    # Initialize end-effector velocity history for smoothness penalties
    robot = env.scene.get("robot")
    if robot is not None and hasattr(robot, "data"):
        try:
            ee_body_idx = robot.find_bodies("panda_hand")[0] if hasattr(robot, "find_bodies") else 0
            env._prev_ee_vel = robot.data.body_vel_w[:, ee_body_idx, :3].clone()
        except (IndexError, AttributeError):
            # Fallback if end-effector not found
            env._prev_ee_vel = torch.zeros(env.num_envs, 3, device=env.device)
    else:
        env._prev_ee_vel = torch.zeros(env.num_envs, 3, device=env.device)


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
