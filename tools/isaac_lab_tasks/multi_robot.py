"""
Multi-Robot Support for Isaac Lab Tasks.

Extends the task generator to support multiple robots in a scene,
including:
- Robot team configuration
- Collision avoidance between robots
- Coordinated manipulation policies
- Task allocation strategies

This module enables scenarios like:
- Dual-arm manipulation
- Robot fleets in warehouse
- Collaborative assembly
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RobotRole(str, Enum):
    """Role of a robot in a multi-robot scenario."""
    PRIMARY = "primary"        # Main manipulator
    SECONDARY = "secondary"    # Support manipulator
    OBSERVER = "observer"      # Sensing only (no manipulation)
    TRANSPORTER = "transporter"  # Mobile base for transport


class CoordinationMode(str, Enum):
    """How robots coordinate their actions."""
    INDEPENDENT = "independent"    # Each robot acts independently
    LEADER_FOLLOWER = "leader_follower"  # One leads, others follow
    SYNCHRONIZED = "synchronized"  # All robots move together
    SEQUENTIAL = "sequential"      # Robots take turns
    COOPERATIVE = "cooperative"    # Shared reward, joint actions


@dataclass
class RobotInstance:
    """Configuration for a single robot in a multi-robot setup."""
    robot_id: str
    robot_type: str  # "franka", "ur10", "fetch", etc.
    role: RobotRole = RobotRole.PRIMARY
    base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz quaternion
    workspace_bounds: Optional[Dict[str, Tuple[float, float]]] = None  # {"x": (min, max), ...}
    collision_padding: float = 0.05  # Extra padding for collision avoidance
    is_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "robot_type": self.robot_type,
            "role": self.role.value,
            "base_position": list(self.base_position),
            "base_orientation": list(self.base_orientation),
            "workspace_bounds": self.workspace_bounds,
            "collision_padding": self.collision_padding,
            "is_enabled": self.is_enabled,
        }


@dataclass
class MultiRobotConfig:
    """Configuration for a multi-robot setup."""
    robots: List[RobotInstance] = field(default_factory=list)
    coordination_mode: CoordinationMode = CoordinationMode.INDEPENDENT
    shared_workspace: Optional[Dict[str, Tuple[float, float]]] = None
    collision_avoidance: bool = True
    collision_check_frequency: int = 10  # Check every N steps

    # Task allocation
    task_allocation_strategy: str = "fixed"  # "fixed", "dynamic", "auction"
    primary_robot_id: Optional[str] = None

    # Communication (for cooperative modes)
    enable_communication: bool = False
    communication_dim: int = 8  # Dimension of communication vector

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robots": [r.to_dict() for r in self.robots],
            "coordination_mode": self.coordination_mode.value,
            "shared_workspace": self.shared_workspace,
            "collision_avoidance": self.collision_avoidance,
            "collision_check_frequency": self.collision_check_frequency,
            "task_allocation_strategy": self.task_allocation_strategy,
            "primary_robot_id": self.primary_robot_id,
            "enable_communication": self.enable_communication,
            "communication_dim": self.communication_dim,
        }


# Extended robot configurations
def _omniverse_robot_root(isaac_version: str | None = None) -> str:
    if isaac_version is None:
        isaac_version = os.environ.get("ISAAC_SIM_VERSION", "2023.1.1")
    omniverse_host = os.environ.get("OMNIVERSE_HOST", "localhost")
    path_root = os.environ.get("OMNIVERSE_PATH_ROOT", "NVIDIA/Assets/Isaac").strip("/")
    return f"omniverse://{omniverse_host}/{path_root}/{isaac_version}/Isaac/Robots"


_DEFAULT_OMNIVERSE_ROBOT_ROOT = _omniverse_robot_root()

EXTENDED_ROBOT_CONFIGS = {
    "franka": {
        "num_dofs": 7,
        "gripper_dofs": 2,
        "ee_frame": "panda_hand",
        "default_joint_pos": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "reach": 0.855,  # meters
        "payload": 3.0,  # kg
        "usd_path": f"{_DEFAULT_OMNIVERSE_ROBOT_ROOT}/Franka/franka.usd",
    },
    "ur10": {
        "num_dofs": 6,
        "gripper_dofs": 0,
        "ee_frame": "tool0",
        "default_joint_pos": [0.0, -1.571, 1.571, -1.571, -1.571, 0.0],
        "reach": 1.3,
        "payload": 10.0,
        "usd_path": f"{_DEFAULT_OMNIVERSE_ROBOT_ROOT}/UR10/ur10.usd",
    },
    "fetch": {
        "num_dofs": 7,
        "gripper_dofs": 2,
        "ee_frame": "gripper_link",
        "default_joint_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "reach": 0.7,
        "payload": 2.3,
        "mobile_base": True,
        "base_dofs": 3,  # x, y, theta
        "usd_path": f"{_DEFAULT_OMNIVERSE_ROBOT_ROOT}/Fetch/fetch.usd",
    },
    "ur5": {
        "num_dofs": 6,
        "gripper_dofs": 0,
        "ee_frame": "tool0",
        "default_joint_pos": [0.0, -1.571, 1.571, -1.571, -1.571, 0.0],
        "reach": 0.85,
        "payload": 5.0,
        "usd_path": f"{_DEFAULT_OMNIVERSE_ROBOT_ROOT}/UR5/ur5.usd",
    },
    "kuka_iiwa": {
        "num_dofs": 7,
        "gripper_dofs": 0,
        "ee_frame": "iiwa_link_ee",
        "default_joint_pos": [0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0],
        "reach": 0.8,
        "payload": 14.0,
        "usd_path": f"{_DEFAULT_OMNIVERSE_ROBOT_ROOT}/Kuka/kuka_iiwa.usd",
    },
    "sawyer": {
        "num_dofs": 7,
        "gripper_dofs": 2,
        "ee_frame": "right_hand",
        "default_joint_pos": [0.0, -1.18, 0.0, 2.18, 0.0, 0.57, 3.14],
        "reach": 1.26,
        "payload": 4.0,
        "usd_path": f"{_DEFAULT_OMNIVERSE_ROBOT_ROOT}/Rethink/sawyer.usd",
    },
}


def create_dual_arm_config(
    left_robot: str = "franka",
    right_robot: str = "franka",
    separation: float = 1.5,  # meters between robots
    facing_inward: bool = True,
) -> MultiRobotConfig:
    """
    Create a dual-arm robot configuration.

    Args:
        left_robot: Type of left robot
        right_robot: Type of right robot
        separation: Distance between robot bases
        facing_inward: If True, robots face each other

    Returns:
        MultiRobotConfig for dual-arm setup
    """
    half_sep = separation / 2

    # Compute orientations
    if facing_inward:
        left_orient = (0.707, 0.0, 0.0, 0.707)   # Rotated 90 degrees to face right
        right_orient = (0.707, 0.0, 0.0, -0.707) # Rotated -90 degrees to face left
    else:
        left_orient = (1.0, 0.0, 0.0, 0.0)
        right_orient = (1.0, 0.0, 0.0, 0.0)

    left_config = EXTENDED_ROBOT_CONFIGS.get(left_robot, EXTENDED_ROBOT_CONFIGS["franka"])
    right_config = EXTENDED_ROBOT_CONFIGS.get(right_robot, EXTENDED_ROBOT_CONFIGS["franka"])

    robots = [
        RobotInstance(
            robot_id="left_arm",
            robot_type=left_robot,
            role=RobotRole.PRIMARY,
            base_position=(-half_sep, 0.0, 0.0),
            base_orientation=left_orient,
            workspace_bounds={
                "x": (-left_config["reach"], left_config["reach"] * 0.8),
                "y": (-left_config["reach"] * 0.5, left_config["reach"] * 0.5),
                "z": (0.0, left_config["reach"] * 0.8),
            },
        ),
        RobotInstance(
            robot_id="right_arm",
            robot_type=right_robot,
            role=RobotRole.SECONDARY,
            base_position=(half_sep, 0.0, 0.0),
            base_orientation=right_orient,
            workspace_bounds={
                "x": (-right_config["reach"] * 0.8, right_config["reach"]),
                "y": (-right_config["reach"] * 0.5, right_config["reach"] * 0.5),
                "z": (0.0, right_config["reach"] * 0.8),
            },
        ),
    ]

    return MultiRobotConfig(
        robots=robots,
        coordination_mode=CoordinationMode.COOPERATIVE,
        shared_workspace={
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (0.1, 0.5),
        },
        collision_avoidance=True,
        primary_robot_id="left_arm",
        enable_communication=True,
    )


def create_robot_fleet_config(
    robot_type: str = "fetch",
    num_robots: int = 4,
    formation: str = "grid",  # "grid", "line", "circle"
    spacing: float = 2.0,
) -> MultiRobotConfig:
    """
    Create a fleet of mobile robots.

    Args:
        robot_type: Type of robot for the fleet
        num_robots: Number of robots
        formation: Formation type
        spacing: Spacing between robots

    Returns:
        MultiRobotConfig for robot fleet
    """
    robots = []

    if formation == "grid":
        cols = math.ceil(math.sqrt(num_robots))
        for i in range(num_robots):
            row = i // cols
            col = i % cols
            x = col * spacing - (cols - 1) * spacing / 2
            y = row * spacing - ((num_robots - 1) // cols) * spacing / 2

            robots.append(RobotInstance(
                robot_id=f"robot_{i}",
                robot_type=robot_type,
                role=RobotRole.TRANSPORTER if i > 0 else RobotRole.PRIMARY,
                base_position=(x, y, 0.0),
            ))

    elif formation == "line":
        for i in range(num_robots):
            x = i * spacing - (num_robots - 1) * spacing / 2

            robots.append(RobotInstance(
                robot_id=f"robot_{i}",
                robot_type=robot_type,
                role=RobotRole.TRANSPORTER if i > 0 else RobotRole.PRIMARY,
                base_position=(x, 0.0, 0.0),
            ))

    elif formation == "circle":
        radius = spacing * num_robots / (2 * math.pi)
        for i in range(num_robots):
            angle = 2 * math.pi * i / num_robots
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            # Face center
            orient_angle = angle + math.pi
            qw = math.cos(orient_angle / 2)
            qz = math.sin(orient_angle / 2)

            robots.append(RobotInstance(
                robot_id=f"robot_{i}",
                robot_type=robot_type,
                role=RobotRole.TRANSPORTER if i > 0 else RobotRole.PRIMARY,
                base_position=(x, y, 0.0),
                base_orientation=(qw, 0.0, 0.0, qz),
            ))

    return MultiRobotConfig(
        robots=robots,
        coordination_mode=CoordinationMode.INDEPENDENT,
        collision_avoidance=True,
        task_allocation_strategy="dynamic",
        primary_robot_id="robot_0",
    )


def generate_multi_robot_env_config(
    config: MultiRobotConfig,
    base_env_code: str,
) -> str:
    """
    Generate Isaac Lab environment config code for multi-robot setup.

    Args:
        config: Multi-robot configuration
        base_env_code: Base environment config code to extend

    Returns:
        Modified environment config code with multi-robot support
    """
    robot_imports = []
    robot_cfgs = []
    robot_defs = []

    for robot in config.robots:
        robot_cfg = EXTENDED_ROBOT_CONFIGS.get(robot.robot_type, EXTENDED_ROBOT_CONFIGS["franka"])

        # Generate robot config class
        cfg_name = f"{robot.robot_id.title().replace('_', '')}Cfg"
        robot_cfgs.append(f"""
@configclass
class {cfg_name}:
    \"\"\"Configuration for {robot.robot_id}.\"\"\"
    robot_type: str = "{robot.robot_type}"
    base_position: tuple = {robot.base_position}
    base_orientation: tuple = {robot.base_orientation}
    num_dofs: int = {robot_cfg['num_dofs']}
    gripper_dofs: int = {robot_cfg['gripper_dofs']}
    ee_frame: str = "{robot_cfg['ee_frame']}"
    default_joint_pos: list = field(default_factory=lambda: {robot_cfg['default_joint_pos']})
    is_enabled: bool = {robot.is_enabled}
    role: str = "{robot.role.value}"
""")

        # Generate robot definition
        robot_defs.append(f"""
    # {robot.robot_id}
    {robot.robot_id}: ArticulationCfg = ArticulationCfg(
        prim_path="/World/{robot.robot_id}",
        spawn=sim_utils.UsdFileCfg(
            usd_path="{robot_cfg.get('usd_path', '')}",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos={robot.base_position},
            rot={robot.base_orientation},
            joint_pos={{".*": {robot_cfg['default_joint_pos']}}},
        ),
    )
""")

    # Generate collision avoidance code if enabled
    collision_code = ""
    if config.collision_avoidance:
        collision_code = """
    # Multi-robot collision avoidance
    def check_robot_collisions(self) -> torch.Tensor:
        \"\"\"Check for collisions between robots.\"\"\"
        collision_penalty = torch.zeros(self.num_envs, device=self.device)
        robot_positions = []

        for robot_id in self._robot_ids:
            ee_pos = getattr(self, f"{robot_id}_ee_pos")
            robot_positions.append(ee_pos)

        for i, pos_i in enumerate(robot_positions):
            for j, pos_j in enumerate(robot_positions):
                if i >= j:
                    continue
                dist = torch.norm(pos_i - pos_j, dim=-1)
                min_dist = self.cfg.collision_min_distance
                collision_penalty += torch.clamp(min_dist - dist, min=0.0) * 10.0

        return collision_penalty
"""

    # Generate coordination code based on mode
    coordination_code = ""
    if config.coordination_mode == CoordinationMode.COOPERATIVE:
        coordination_code = """
    # Cooperative control utilities
    def get_joint_observation(self) -> torch.Tensor:
        \"\"\"Get combined observation from all robots.\"\"\"
        obs_parts = []
        for robot_id in self._robot_ids:
            robot_obs = self._get_robot_observation(robot_id)
            obs_parts.append(robot_obs)
        return torch.cat(obs_parts, dim=-1)

    def apply_coordinated_action(self, actions: torch.Tensor) -> None:
        \"\"\"Apply coordinated actions to all robots.\"\"\"
        action_dim_per_robot = actions.shape[-1] // len(self._robot_ids)
        for i, robot_id in enumerate(self._robot_ids):
            start_idx = i * action_dim_per_robot
            end_idx = start_idx + action_dim_per_robot
            robot_action = actions[:, start_idx:end_idx]
            self._apply_robot_action(robot_id, robot_action)
"""

    # Combine into final code
    multi_robot_code = f"""
# ============================================================================
# Multi-Robot Configuration (Auto-generated)
# ============================================================================

from dataclasses import field

{chr(10).join(robot_cfgs)}

class MultiRobotSceneCfg:
    \"\"\"Scene configuration with multiple robots.\"\"\"

    coordination_mode: str = "{config.coordination_mode.value}"
    collision_avoidance: bool = {config.collision_avoidance}
    collision_check_frequency: int = {config.collision_check_frequency}
    collision_min_distance: float = 0.1  # meters

    # Robot configurations
{chr(10).join(robot_defs)}

{collision_code}
{coordination_code}
"""

    return multi_robot_code


def generate_multi_robot_reward_code(config: MultiRobotConfig) -> str:
    """
    Generate reward functions for multi-robot scenarios.

    Args:
        config: Multi-robot configuration

    Returns:
        Python code for multi-robot reward functions
    """
    code = '''
"""Multi-robot reward functions."""

import torch
from omni.isaac.lab.managers import RewardTermCfg


def collision_avoidance_reward(env) -> torch.Tensor:
    """Reward for maintaining safe distance between robots."""
    penalty = torch.zeros(env.num_envs, device=env.device)

    robot_positions = []
    for robot_id in env._robot_ids:
        ee_pos = getattr(env, f"{robot_id}_ee_pos")
        robot_positions.append(ee_pos)

    min_safe_dist = 0.15  # 15cm minimum distance

    for i, pos_i in enumerate(robot_positions):
        for j, pos_j in enumerate(robot_positions):
            if i >= j:
                continue
            dist = torch.norm(pos_i - pos_j, dim=-1)
            penalty += torch.clamp(min_safe_dist - dist, min=0.0) * 100.0

    return -penalty


def coordination_bonus_reward(env) -> torch.Tensor:
    """Bonus for coordinated robot movements."""
    bonus = torch.zeros(env.num_envs, device=env.device)

    if len(env._robot_ids) < 2:
        return bonus

    # Compute velocity alignment between robots
    velocities = []
    for robot_id in env._robot_ids:
        ee_vel = getattr(env, f"{robot_id}_ee_vel", None)
        if ee_vel is not None:
            velocities.append(ee_vel)

    if len(velocities) >= 2:
        # Reward parallel motion (same direction)
        v1_norm = velocities[0] / (torch.norm(velocities[0], dim=-1, keepdim=True) + 1e-8)
        v2_norm = velocities[1] / (torch.norm(velocities[1], dim=-1, keepdim=True) + 1e-8)
        alignment = torch.sum(v1_norm * v2_norm, dim=-1)
        bonus += torch.clamp(alignment, min=0.0) * 0.1

    return bonus


def task_completion_shared_reward(env) -> torch.Tensor:
    """Shared reward when any robot completes task."""
    shared_reward = torch.zeros(env.num_envs, device=env.device)

    for robot_id in env._robot_ids:
        robot_complete = getattr(env, f"{robot_id}_task_complete", None)
        if robot_complete is not None:
            shared_reward += robot_complete.float() * 10.0

    return shared_reward


def workspace_violation_penalty(env) -> torch.Tensor:
    """Penalty for robots leaving their designated workspace."""
    penalty = torch.zeros(env.num_envs, device=env.device)

    for robot_id, bounds in env._robot_workspace_bounds.items():
        if bounds is None:
            continue

        ee_pos = getattr(env, f"{robot_id}_ee_pos")

        for axis, (min_val, max_val) in bounds.items():
            axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
            pos_axis = ee_pos[:, axis_idx]

            below_min = torch.clamp(min_val - pos_axis, min=0.0)
            above_max = torch.clamp(pos_axis - max_val, min=0.0)

            penalty += (below_min + above_max) * 50.0

    return -penalty


# Reward term configurations for multi-robot
MULTI_ROBOT_REWARD_TERMS = {
    "collision_avoidance": RewardTermCfg(
        func=collision_avoidance_reward,
        weight=1.0,
    ),
    "coordination_bonus": RewardTermCfg(
        func=coordination_bonus_reward,
        weight=0.5,
    ),
    "task_completion_shared": RewardTermCfg(
        func=task_completion_shared_reward,
        weight=1.0,
    ),
    "workspace_violation": RewardTermCfg(
        func=workspace_violation_penalty,
        weight=1.0,
    ),
}
'''
    return code


class MultiRobotTaskGenerator:
    """
    Generates Isaac Lab task packages with multi-robot support.

    Usage:
        generator = MultiRobotTaskGenerator(policy_config)
        config = create_dual_arm_config()
        task = generator.generate(recipe, config, policy_id="dual_arm_manipulation")
    """

    def __init__(self, policy_config: Dict[str, Any]):
        self.policy_config = policy_config

    def generate(
        self,
        recipe: Dict[str, Any],
        multi_robot_config: MultiRobotConfig,
        policy_id: str,
        num_envs: int = 1024,
    ) -> Dict[str, str]:
        """
        Generate Isaac Lab task files for multi-robot scenario.

        Args:
            recipe: Scene recipe
            multi_robot_config: Multi-robot configuration
            policy_id: Policy identifier
            num_envs: Number of parallel environments

        Returns:
            Dictionary of filename -> content
        """
        files = {}

        # Generate multi-robot environment config
        files["multi_robot_cfg.py"] = generate_multi_robot_env_config(
            multi_robot_config, ""
        )

        # Generate multi-robot reward functions
        files["multi_robot_rewards.py"] = generate_multi_robot_reward_code(
            multi_robot_config
        )

        # Generate configuration JSON
        import json
        files["multi_robot_config.json"] = json.dumps(
            multi_robot_config.to_dict(), indent=2
        )

        return files
