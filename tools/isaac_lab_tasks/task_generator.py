"""
Isaac Lab Task Generator - Creates Isaac Lab task packages from recipes.

This module generates complete Isaac Lab task packages including:
- Environment configuration
- Task definitions (observations, actions, rewards)
- Domain randomization hooks (EventManager compatible)
- Training configuration

Note: Generated code is designed to work with Isaac Lab's ManagerBasedEnv
architecture and its event-driven randomization system.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .env_config import EnvConfigGenerator
from .reward_functions import RewardFunctionGenerator


@dataclass
class TaskConfig:
    """Configuration for a generated Isaac Lab task."""
    task_name: str
    policy_id: str
    scene_path: str
    robot_type: str = "franka"  # franka, ur10, fetch, etc.
    num_envs: int = 1024
    episode_length: int = 500
    primary_target: Optional[str] = None
    scene_entities: dict[str, str] = field(default_factory=dict)
    observation_space: dict[str, Any] = field(default_factory=dict)
    action_space: dict[str, Any] = field(default_factory=dict)
    task_metadata: dict[str, Any] = field(default_factory=dict)
    reward_weights: dict[str, float] = field(default_factory=dict)
    randomization_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedTask:
    """Result of task generation."""
    task_name: str
    files: dict[str, str]  # filename -> content
    config: TaskConfig


class IsaacLabTaskGenerator:
    """
    Generates Isaac Lab task packages from scene recipes.

    Usage:
        generator = IsaacLabTaskGenerator(policy_config)
        task = generator.generate(recipe, policy_id="drawer_manipulation")
        generator.save(task, "isaac_lab/")
    """

    ROBOT_CONFIGS = {
        "franka": {
            "num_dofs": 7,
            "gripper_dofs": 2,
            "ee_frame": "panda_hand",
            "default_joint_pos": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        },
        "ur10": {
            "num_dofs": 6,
            "gripper_dofs": 0,
            "ee_frame": "tool0",
            "default_joint_pos": [0.0, -1.571, 1.571, -1.571, -1.571, 0.0]
        },
        "fetch": {
            "num_dofs": 7,
            "gripper_dofs": 2,
            "ee_frame": "gripper_link",
            "default_joint_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }

    def __init__(self, policy_config: dict[str, Any]):
        self.policy_config = policy_config
        self.policies = policy_config.get("policies", {})
        self.environments = policy_config.get("environments", {})
        self.env_generator = EnvConfigGenerator()
        self.reward_generator = RewardFunctionGenerator()

    def generate(
        self,
        recipe: dict[str, Any],
        policy_id: str,
        robot_type: str = "franka",
        num_envs: int = 1024
    ) -> GeneratedTask:
        """
        Generate an Isaac Lab task package.

        Args:
            recipe: Scene recipe
            policy_id: ID of the training policy
            robot_type: Type of robot to use
            num_envs: Number of parallel environments

        Returns:
            GeneratedTask with all generated files
        """
        env_type = recipe.get("metadata", {}).get("environment_type", "unknown")
        scene_path = recipe.get("metadata", {}).get("scene_path", "scene.usda")
        policy = self.policies.get(policy_id, {})

        # Build task configuration
        task_config = self._build_task_config(
            recipe, policy, policy_id, robot_type, num_envs, scene_path
        )

        # Generate files
        files = {}

        # Environment config
        files["env_cfg.py"] = self._generate_env_config(task_config, recipe, policy)

        # Task implementation
        files[f"task_{policy_id}.py"] = self._generate_task_file(
            task_config, recipe, policy
        )

        # Reward implementations (manager-compatible)
        files["reward_functions.py"] = self.reward_generator.generate_reward_module(
            list(task_config.reward_weights.keys()), task_config.reward_weights
        )

        # Training config
        files["train_cfg.yaml"] = self._generate_train_config(task_config, policy)

        # Randomization hooks
        files["randomizations.py"] = self._generate_randomization_hooks(
            task_config, recipe, policy
        )

        # Package init
        files["__init__.py"] = self._generate_init_file(task_config, policy_id)

        return GeneratedTask(
            task_name=f"{env_type}_{policy_id}",
            files=files,
            config=task_config
        )

    def _build_task_config(
        self,
        recipe: dict[str, Any],
        policy: dict[str, Any],
        policy_id: str,
        robot_type: str,
        num_envs: int,
        scene_path: str
    ) -> TaskConfig:
        """Build task configuration from recipe and policy."""
        robot_config = self.ROBOT_CONFIGS.get(robot_type, self.ROBOT_CONFIGS["franka"])

        # Build observation space based on policy
        obs_space = self._build_observation_space(policy, robot_config, recipe)

        # Build action space based on robot
        action_space = self._build_action_space(robot_config, policy)

        # Determine primary manipulable target for scene-aware observations/rewards
        primary_target = self._select_primary_target(recipe)

        # Map logical object identifiers to prim paths in the stage
        scene_entities = self._build_scene_entity_map(recipe)
        if primary_target and primary_target not in scene_entities:
            scene_entities[primary_target] = f"/World/Scene/{primary_target}"

        # Get reward weights
        reward_components = policy.get("reward_components", [])
        reward_weights = {comp: 1.0 for comp in reward_components}

        # Get randomization config
        randomization_config = self._build_randomization_config(policy, recipe)

        return TaskConfig(
            task_name=f"{recipe['metadata']['environment_type']}_{policy_id}",
            policy_id=policy_id,
            scene_path=scene_path,
            robot_type=robot_type,
            num_envs=num_envs,
            episode_length=500,
            primary_target=primary_target,
            scene_entities=scene_entities,
            observation_space=obs_space,
            action_space=action_space,
            task_metadata={
                "workspace_origin": recipe.get("room", {}).get("origin", (0.0, 0.0, 0.0)),
                "target_object": primary_target,
            },
            reward_weights=reward_weights,
            randomization_config=randomization_config
        )

    def _build_observation_space(
        self,
        policy: dict[str, Any],
        robot_config: dict[str, Any],
        recipe: dict[str, Any]
    ) -> dict[str, Any]:
        """Build observation space configuration."""
        obs_space = {
            "robot": {
                "joint_pos": robot_config["num_dofs"],
                "joint_vel": robot_config["num_dofs"],
                "ee_pos": 3,
                "ee_quat": 4,
                "gripper_pos": robot_config["gripper_dofs"]
            },
            "task": {}
        }

        # Add task-specific observations based on policy
        policy_id = policy.get("isaac_lab_task", "")

        if "pick_place" in policy_id or "manipulation" in policy_id:
            obs_space["task"]["target_pos"] = 3
            obs_space["task"]["object_pos"] = 3
            obs_space["task"]["object_quat"] = 4

        if "drawer" in policy_id or "door" in policy_id:
            obs_space["task"]["joint_pos"] = 1
            obs_space["task"]["joint_vel"] = 1
            obs_space["task"]["handle_pos"] = 3

        if "articulated" in policy_id:
            obs_space["task"]["articulation_pos"] = 1
            obs_space["task"]["articulation_vel"] = 1

        return obs_space

    def _build_action_space(
        self, robot_config: dict[str, Any], policy: dict[str, Any]
    ) -> dict[str, Any]:
        """Build action space configuration with explicit control mode."""

        control_mode = policy.get("control_mode", "joint_velocity")
        action_scale = policy.get("action_scale", 0.1)
        ik_target = policy.get("ik_target", robot_config.get("ee_frame"))

        return {
            "type": "continuous",
            "arm_dofs": robot_config["num_dofs"],
            "gripper_dofs": robot_config["gripper_dofs"],
            "control_type": control_mode,
            "ik_target": ik_target,
            "action_scale": action_scale,
        }

    def _build_randomization_config(
        self,
        policy: dict[str, Any],
        recipe: dict[str, Any]
    ) -> dict[str, Any]:
        """Build domain randomization configuration."""
        config = {
            "on_reset": [],
            "on_step": []
        }

        for rand in policy.get("randomizers", []):
            if not rand.get("enabled", True):
                continue

            freq = rand.get("frequency", "per_episode")
            event = {
                "name": rand["name"],
                "params": rand.get("parameters", {})
            }

            if freq in ["per_episode", "once"]:
                config["on_reset"].append(event)
            else:
                config["on_step"].append(event)

        return config

    def _generate_env_config(
        self,
        config: TaskConfig,
        recipe: dict[str, Any],
        policy: dict[str, Any]
    ) -> str:
        """Generate environment configuration file."""
        primary_target = config.primary_target or "target_object"
        target_scene_entity = f'SceneEntityCfg("{primary_target}", prim_path=SCENE_ENTITY_MAP.get("{primary_target}"))'
        ee_frame = self.ROBOT_CONFIGS[config.robot_type]["ee_frame"]
        action_cfg = self._format_action_cfg(config.action_space)

        return f'''"""
Environment Configuration - {config.task_name}
Generated by BlueprintRecipe (Gemini-guided)

This file defines the environment configuration for Isaac Lab.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

import torch
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from omni.isaac.lab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
from omni.isaac.lab.utils import configclass
from . import reward_functions


SCENE_ENTITY_MAP = {self._format_scene_entity_map(config.scene_entities)}


def resolve_scene_entity(name: str, fallback: str | None = None) -> SceneEntityCfg:
    """Resolve a SceneEntityCfg using the recipe-aware prim mapping."""

    prim = SCENE_ENTITY_MAP.get(name, fallback)
    if prim:
        return SceneEntityCfg(name, prim_path=prim)
    return SceneEntityCfg(name)


def observation_target_relative(
    env,
    source_entity: SceneEntityCfg,
    target_entity: SceneEntityCfg,
):
    """Return target position relative to a source asset (EE or link)."""

    source = env.scene.get(source_entity.name)
    target = env.scene.get(target_entity.name)
    if source is None or target is None:
        return torch.zeros(env.num_envs, 3, device=env.device)

    source_pos = source.data.root_pos_w
    target_pos = target.data.root_pos_w
    return target_pos - source_pos


def observation_gripper_width(env, gripper_dof_indices: tuple[int, int] | None = None):
    """Compute gripper opening width from joint positions."""

    robot = env.scene.get("robot")
    if robot is None:
        return torch.zeros(env.num_envs, 1, device=env.device)

    if gripper_dof_indices is None:
        gripper_dof_indices = (-2, -1)
    joint_pos = robot.data.joint_pos[:, list(gripper_dof_indices)]
    width = torch.abs(joint_pos[:, 0]) + torch.abs(joint_pos[:, 1])
    return width.unsqueeze(-1)


def reward_reaching(
    env,
    target_entity: SceneEntityCfg = {target_scene_entity},
    ee_body: str = "{ee_frame}",
    distance_scale: float = 5.0,
    success_threshold: float = 0.05,
):
    """Dense reaching reward towards the scene-aware target."""

    robot = env.scene.get("robot")
    target = env.scene.get(target_entity.name)
    if robot is None or target is None:
        return torch.zeros(env.num_envs, device=env.device)

    ee_id = robot.find_bodies(ee_body)[0]
    ee_pos = robot.data.body_pos_w[:, ee_id]
    target_pos = target.data.root_pos_w
    dist = torch.norm(target_pos - ee_pos, dim=-1)

    shaped = 1.0 - torch.tanh(distance_scale * dist)
    success_bonus = (dist < success_threshold).float() * 2.0
    return shaped + success_bonus


def reward_task_success(env, success_flag: str = "task_success", bonus: float = 10.0):
    """Reward emitted when the task_success flag is true."""

    status = getattr(env, success_flag, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return status.float() * bonus


def termination_object_dropped(
    env,
    object_entity: SceneEntityCfg = {target_scene_entity},
    threshold: float = -0.05,
):
    """Terminate if the tracked object falls below the workspace floor."""

    target = env.scene.get(object_entity.name)
    if target is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    z_height = target.data.root_pos_w[:, 2]
    return z_height < threshold


def termination_task_success(env, success_attr: str = "task_success"):
    """Terminate successful rollouts early."""

    status = getattr(env, success_attr, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return status


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""

    # Ground plane (from recipe room)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Scene USD (from recipe)
    scene = AssetBaseCfg(
        prim_path="/World/Scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path="{config.scene_path}",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    # Robot configuration
    robot = ArticulationCfg(
        prim_path=SCENE_ENTITY_MAP.get("robot", "/World/Robot"),
        spawn=sim_utils.UsdFileCfg(
            usd_path="{{ROBOT_USD_PATH}}",  # Resolved at runtime
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={self._format_joint_pos(config.robot_type)},
        ),
    )

    # Primary task object (optional static registration for sensors)
    if "{primary_target}":
        {primary_target} = RigidObjectCfg(
            prim_path=SCENE_ENTITY_MAP.get("{primary_target}", "/World/Scene/{primary_target}"),
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(),
        )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1500.0,
            color=(1.0, 1.0, 1.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation configuration."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for the policy."""

        # Robot state observations
        joint_pos = ObservationTermCfg(
            func="omni.isaac.lab.envs.mdp.joint_pos",
            params={{"asset_cfg": resolve_scene_entity("robot")}},
        )
        joint_vel = ObservationTermCfg(
            func="omni.isaac.lab.envs.mdp.joint_vel",
            params={{"asset_cfg": resolve_scene_entity("robot")}},
        )
        ee_pos = ObservationTermCfg(
            func="omni.isaac.lab.envs.mdp.body_pos_w",
            params={{
                "asset_cfg": resolve_scene_entity("robot"),
                "body_name": "{ee_frame}"
            }},
        )
        target_rel_pos = ObservationTermCfg(
            func="observation_target_relative",
            params={{
                "source_entity": resolve_scene_entity("robot"),
                "target_entity": resolve_scene_entity("{primary_target}")
            }},
        )
        gripper_width = ObservationTermCfg(
            func="observation_gripper_width",
            params={{"gripper_dof_indices": (-2, -1)}},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action configuration."""

{action_cfg}


@configclass
class RewardsCfg:
    """Reward configuration."""

{self._generate_reward_terms(config, policy)}


@configclass
class TerminationsCfg:
    """Termination configuration."""

    time_out = TerminationTermCfg(
        func="omni.isaac.lab.envs.mdp.time_out",
        time_out=True,
    )

    object_dropped = TerminationTermCfg(
        func="termination_object_dropped",
    )

    task_success = TerminationTermCfg(
        func="termination_task_success",
    )


@configclass
class EventsCfg:
    """Event configuration for domain randomization."""

{self._generate_event_terms(config, recipe, policy)}


@configclass
class {self._to_class_name(config.task_name)}EnvCfg(ManagerBasedEnvCfg):
    """Configuration for {config.task_name} environment."""

    # Scene
    scene: SceneCfg = SceneCfg(num_envs={config.num_envs}, env_spacing=2.0)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Episode settings
    episode_length_s = {config.episode_length / 60.0:.1f}  # seconds

    def __post_init__(self):
        """Post initialization."""
        # Simulation settings
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = 1

        # Update scene settings
        self.scene.num_envs = {config.num_envs}
        self.scene_entity_map = SCENE_ENTITY_MAP
'''

    def _generate_task_file(
        self,
        config: TaskConfig,
        recipe: dict[str, Any],
        policy: dict[str, Any]
    ) -> str:
        """Generate task implementation file."""
        class_name = self._to_class_name(config.task_name)
        target_name = config.primary_target or "target_object"
        ee_frame = self.ROBOT_CONFIGS[config.robot_type]["ee_frame"]
        return f'''"""
Task Implementation - {config.task_name}
Generated by BlueprintRecipe

This file implements the task logic for {policy.get("display_name", config.policy_id)}.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.envs import ManagerBasedEnv

if TYPE_CHECKING:
    from .env_cfg import {class_name}EnvCfg


class {class_name}Task:
    """
    Task implementation for {config.task_name}.

    Policy: {policy.get("display_name", config.policy_id)}
    Description: {policy.get("description", "")}
    """

    def __init__(self, env: ManagerBasedEnv, cfg: {class_name}EnvCfg):
        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.num_envs = env.num_envs
        self.scene_entity_map = getattr(cfg, "scene_entity_map", {{}}) if hasattr(cfg, "scene_entity_map") else {config.scene_entities}
        self.target_name = "{target_name}"
        self.ee_frame = "{ee_frame}"

        # Task state
        self._setup_task_state()

    def _setup_task_state(self):
        """Initialize task-specific state tensors."""
        self.task_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor):
        """Reset task state for specified environments."""
        self.task_success[env_ids] = False
        self.object_dropped[env_ids] = False

    def compute_observations(self) -> dict[str, torch.Tensor]:
        """Compute task-specific observations and merge with manager outputs."""

        observations = self.env.observation_manager.compute()
        target = self._get_scene_entity(self.target_name)
        if target is not None:
            target_pose = torch.cat([target.data.root_pos_w, target.data.root_quat_w], dim=-1)
            observations["target_pose"] = target_pose

        return observations

    def compute_rewards(self) -> dict[str, torch.Tensor]:
        """Compute reward components and integrate with RewardManager."""

        reward_terms = self.env.reward_manager.compute()
        target = self._get_scene_entity(self.target_name)
        robot = self.env.scene.get("robot")

        if target is not None and robot is not None:
            ee_pos = self._get_ee_pos(robot)
            dist_to_target = torch.norm(target.data.root_pos_w - ee_pos, dim=-1)
            dense_reach = 1.0 - torch.tanh(4.0 * dist_to_target)
            close_enough = dist_to_target < 0.05

            reward_terms.setdefault("dense_reach", dense_reach)
            reward_terms.setdefault("proximity_success", close_enough.float() * 5.0)
            self.task_success = self.task_success | close_enough

        total = torch.zeros(self.num_envs, device=self.device)
        for value in reward_terms.values():
            total = total + value
        reward_terms["total"] = total
        return reward_terms

    def compute_terminations(self) -> dict[str, torch.Tensor]:
        """Compute termination conditions including manager-driven terms."""

        terminations = self.env.termination_manager.compute()
        terminations.setdefault("task_success", self.task_success)

        target = self._get_scene_entity(self.target_name)
        if target is not None:
            dropped = target.data.root_pos_w[:, 2] < -0.05
            self.object_dropped = self.object_dropped | dropped
            terminations.setdefault("object_dropped", dropped)

        return terminations

    def _get_ee_pos(self, robot=None) -> torch.Tensor:
        """Get end-effector position."""
        robot = robot or self.env.scene.get("robot")
        return robot.data.body_pos_w[:, robot.find_bodies(self.ee_frame)[0]]

    def _get_scene_entity(self, name: str):
        """Resolve a scene entity using the generated map for prim paths."""

        entity = self.env.scene.get(name)
        if entity is not None:
            return entity

        prim_path = self.scene_entity_map.get(name)
        if prim_path:
            return self.env.scene.get(prim_path)
        return None
'''

    def _generate_train_config(
        self,
        config: TaskConfig,
        policy: dict[str, Any]
    ) -> str:
        """Generate training configuration YAML."""
        return f'''# Training Configuration - {config.task_name}
# Generated by BlueprintRecipe

# Task settings
task_name: "{config.task_name}"
experiment_name: "{config.task_name}_training"

# Environment settings
env:
  num_envs: {config.num_envs}
  episode_length: {config.episode_length}

# PPO Algorithm settings
algo:
  name: "PPO"
  policy:
    class_name: "ActorCritic"
    init_noise_std: 1.0
    actor_hidden_dims: [256, 256, 128]
    critic_hidden_dims: [256, 256, 128]
    activation: "elu"

  # PPO specific
  clip_param: 0.2
  entropy_coef: 0.01
  value_loss_coef: 1.0
  max_grad_norm: 1.0

  # Learning rate
  learning_rate: 3.0e-4
  lr_schedule: "adaptive"

  # Batch settings
  num_learning_epochs: 5
  num_mini_batches: 4

  # Discount
  gamma: 0.99
  lam: 0.95

# Training settings
runner:
  max_iterations: 1500
  save_interval: 100
  log_interval: 10

  # Checkpointing
  checkpoint_path: null
  resume: false

# Logging
logging:
  wandb:
    enabled: false
    project: "blueprint_recipe"
    entity: null
  tensorboard:
    enabled: true

# Reward weights
rewards:
{self._format_reward_weights_yaml(config.reward_weights)}

# Domain randomization
randomization:
  enabled: true
  on_reset:
{self._format_randomization_yaml(config.randomization_config.get("on_reset", []))}
  on_step:
{self._format_randomization_yaml(config.randomization_config.get("on_step", []))}
'''

    def _generate_randomization_hooks(
        self,
        config: TaskConfig,
        recipe: dict[str, Any],
        policy: dict[str, Any]
    ) -> str:
        """Generate domain randomization hooks file."""
        return f'''"""
Domain Randomization Hooks - {config.task_name}
Generated by BlueprintRecipe

These hooks are designed to work with Isaac Lab's EventManager.
They write directly to PhysX for RL speed mode compatibility.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import EventTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnvCfg


def randomize_object_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float, float, float, float, float] = (-0.2, 0.2, -0.2, 0.2, 0.0, 0.1),
    rotation_range: tuple[float, float] = (0.0, 6.28),
):
    """
    Randomize object poses on reset.

    Note: This writes directly to PhysX buffers for RL speed mode compatibility.
    """
    num_envs = len(env_ids)

    # Get objects to randomize
    objects = env.scene.get("objects")
    if objects is None:
        return

    # Generate random positions
    pos = torch.zeros(num_envs, 3, device=env.device)
    pos[:, 0] = torch.rand(num_envs, device=env.device) * (position_range[1] - position_range[0]) + position_range[0]
    pos[:, 1] = torch.rand(num_envs, device=env.device) * (position_range[3] - position_range[2]) + position_range[2]
    pos[:, 2] = torch.rand(num_envs, device=env.device) * (position_range[5] - position_range[4]) + position_range[4]

    # Generate random rotations (around Z axis)
    yaw = torch.rand(num_envs, device=env.device) * (rotation_range[1] - rotation_range[0]) + rotation_range[0]
    quat = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw),
        torch.zeros_like(yaw),
        yaw
    )

    # Write to physics
    objects.write_root_pose_to_sim(
        torch.cat([pos, quat], dim=-1),
        env_ids
    )


def randomize_articulation_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    articulation_name: str = "articulated_object",
    joint_range: tuple[float, float] = (0.0, 1.0),
):
    """
    Randomize articulation joint states on reset.
    """
    num_envs = len(env_ids)

    articulation = env.scene.get(articulation_name)
    if articulation is None:
        return

    # Generate random joint positions
    num_joints = articulation.num_joints
    joint_pos = torch.rand(num_envs, num_joints, device=env.device)
    joint_pos = joint_pos * (joint_range[1] - joint_range[0]) + joint_range[0]

    # Respect joint limits
    joint_limits = articulation.data.joint_limits
    joint_pos = torch.clamp(
        joint_pos,
        joint_limits[..., 0],
        joint_limits[..., 1]
    )

    # Write to physics
    articulation.write_joint_state_to_sim(
        joint_pos,
        torch.zeros_like(joint_pos),  # zero velocity
        env_ids=env_ids
    )


def randomize_lighting(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (500.0, 2000.0),
):
    """
    Randomize lighting intensity and orientation using direct PhysX writes.

    """
    light = env.scene.get("light")
    if light is None:
        return

    num_envs = len(env_ids)
    intensity = torch.rand(num_envs, device=env.device)
    intensity = intensity * (intensity_range[1] - intensity_range[0]) + intensity_range[0]

    if hasattr(light, "write_attribute_to_sim"):
        light.write_attribute_to_sim("intensity", intensity, env_ids)

    # Randomize yaw for dome lights to alter shadows/reflections
    yaw = torch.rand(num_envs, device=env.device) * 6.28318
    quat = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
    )
    pose = torch.zeros(num_envs, 7, device=env.device)
    pose[:, 3:] = quat
    if hasattr(light, "write_root_pose_to_sim"):
        light.write_root_pose_to_sim(pose, env_ids)


def randomize_materials(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float] = (0.3, 1.0),
    restitution_range: tuple[float, float] = (0.0, 0.3),
):
    """
    Randomize physics material properties.
    """
    rigid_objects = env.scene.get("objects") or env.scene.get("rigid_objects")
    if rigid_objects is None:
        return

    num_envs = len(env_ids)
    friction = torch.rand(num_envs, device=env.device)
    friction = friction * (friction_range[1] - friction_range[0]) + friction_range[0]
    restitution = torch.rand(num_envs, device=env.device)
    restitution = restitution * (restitution_range[1] - restitution_range[0]) + restitution_range[0]

    if hasattr(rigid_objects, "write_material_properties_to_sim"):
        rigid_objects.write_material_properties_to_sim(
            friction_coefficients=friction,
            restitution_coefficients=restitution,
            env_ids=env_ids,
        )
    elif hasattr(rigid_objects, "write_rigid_body_properties_to_sim"):
        props = rigid_objects.data.rigid_body_properties
        props["friction"] = friction
        props["restitution"] = restitution
        rigid_objects.write_rigid_body_properties_to_sim(props, env_ids=env_ids)


# Event term configurations for EventManager
def get_reset_events() -> dict[str, EventTermCfg]:
    """Get reset event configurations."""
    return {{
        "randomize_objects": EventTermCfg(
            func=randomize_object_poses,
            mode="reset",
        ),
        "randomize_articulations": EventTermCfg(
            func=randomize_articulation_state,
            mode="reset",
        ),
    }}


def get_interval_events() -> dict[str, EventTermCfg]:
    """Get interval event configurations."""
    return {{
        # Add periodic randomization events here
    }}
'''

    def _generate_init_file(self, config: TaskConfig, policy_id: str) -> str:
        """Generate package __init__.py."""
        class_name = self._to_class_name(config.task_name)
        return f'''"""
Isaac Lab Task Package - {config.task_name}
Generated by BlueprintRecipe

This package provides:
- {class_name}EnvCfg: Environment configuration
- {class_name}Task: Task implementation
- Domain randomization hooks
"""

from .env_cfg import {class_name}EnvCfg
from .task_{policy_id} import {class_name}Task
from .randomizations import get_reset_events, get_interval_events

__all__ = [
    "{class_name}EnvCfg",
    "{class_name}Task",
    "get_reset_events",
    "get_interval_events",
]
'''

    def _format_joint_pos(self, robot_type: str) -> str:
        """Format default joint positions dict."""
        config = self.ROBOT_CONFIGS.get(robot_type, self.ROBOT_CONFIGS["franka"])
        positions = config["default_joint_pos"]
        return "{" + ", ".join(f'"joint_{i}": {p}' for i, p in enumerate(positions)) + "}"

    def _to_class_name(self, task_name: str) -> str:
        """Convert task name to class name."""
        return "".join(word.capitalize() for word in task_name.split("_"))

    def _generate_reward_terms(self, config: TaskConfig, policy: dict[str, Any]) -> str:
        """Generate reward term configurations."""
        terms = []
        for component, weight in config.reward_weights.items():
            # Built-in task-shaping terms live in env_cfg; others are generated in reward_functions.py
            if component in {"reaching", "task_success"}:
                func_name = f"reward_{component}"
            else:
                func_name = f"reward_functions.reward_{component}"

            terms.append(f'''    {component} = RewardTermCfg(
        func="{func_name}",
        weight={weight},
    )''')

        return "\n\n".join(terms) if terms else "    pass  # No reward terms configured"

    def _generate_event_terms(
        self,
        config: TaskConfig,
        recipe: dict[str, Any],
        policy: dict[str, Any]
    ) -> str:
        """Generate event term configurations."""
        terms = []

        # Reset events
        for event in config.randomization_config.get("on_reset", []):
            terms.append(f'''    {event["name"]} = EventTermCfg(
        func="randomize_{event["name"]}",
        mode="reset",
    )''')

        return "\n\n".join(terms) if terms else "    pass  # No event terms configured"

    def _format_reward_weights_yaml(self, weights: dict[str, float]) -> str:
        """Format reward weights for YAML."""
        if not weights:
            return "    # No reward weights configured"
        return "\n".join(f"  {k}: {v}" for k, v in weights.items())

    def _format_randomization_yaml(self, events: list[dict]) -> str:
        """Format randomization events for YAML."""
        if not events:
            return "    []"
        lines = []
        for event in events:
            lines.append(f"    - name: {event['name']}")
            if event.get("params"):
                lines.append("      params:")
                for k, v in event["params"].items():
                    lines.append(f"        {k}: {v}")
        return "\n".join(lines)

    def _format_scene_entity_map(self, entity_map: dict[str, str]) -> str:
        """Format a scene entity mapping into Python source."""

        if not entity_map:
            return "{}"

        lines = []
        for key, value in entity_map.items():
            lines.append(f'    "{key}": "{value}",')
        return "{\n" + "\n".join(lines) + "\n}"

    def _format_action_cfg(self, action_space: dict[str, Any]) -> str:
        """Generate the ActionsCfg block based on control mode."""

        control_type = action_space.get("control_type", "joint_velocity")
        scale = action_space.get("action_scale", 0.1)
        ik_target = action_space.get("ik_target", "ee_link")

        if control_type == "joint_position":
            return f'''    joint_position = {{
        "class_type": "omni.isaac.lab.envs.mdp.JointPositionActionCfg",
        "asset_name": "robot",
        "joint_names": [".*"],
        "scale": {scale},
    }}'''

        if control_type in ("ee_pose", "ik"):
            return f'''    ee_delta_pose = {{
        "class_type": "omni.isaac.lab.envs.mdp.EndEffectorPoseActionCfg",
        "asset_name": "robot",
        "body_name": "{ik_target}",
        "scale": {scale},
    }}'''

        # Default to velocity control
        return f'''    joint_vel = {{
        "class_type": "omni.isaac.lab.envs.mdp.JointVelocityActionCfg",
        "asset_name": "robot",
        "joint_names": [".*"],
        "scale": {scale},
    }}'''

    def _select_primary_target(self, recipe: dict[str, Any]) -> Optional[str]:
        """Select a primary target object for the task.

        Preference order:
        1) First manipulable object marked in the recipe
        2) First object with graspable affordance
        3) Fallback to the first object id
        """

        objects = recipe.get("objects", [])
        for obj in objects:
            if obj.get("manipulable"):
                return obj.get("id")

        for obj in objects:
            if "graspable" in obj.get("semantics", {}).get("affordances", []):
                return obj.get("id")

        return objects[0]["id"] if objects else None

    def _build_scene_entity_map(self, recipe: dict[str, Any]) -> dict[str, str]:
        """Map logical ids to prim paths using recipe context."""

        env_type = recipe.get("metadata", {}).get("environment_type", "Scene")
        room_prim = f"/World/{env_type.title()}"

        entity_map = {
            "robot": "/World/Robot",
            "scene_root": room_prim,
        }

        for obj in recipe.get("objects", []):
            prim_hint = obj.get("transform", {}).get("prim_path")
            if prim_hint:
                entity_map[obj["id"]] = prim_hint
                continue

            # Default placement inside scene hierarchy
            entity_map[obj["id"]] = f"{room_prim}/Objects/{obj['id']}"

        return entity_map

    def save(self, task: GeneratedTask, output_dir: str) -> dict[str, str]:
        """Save generated task to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}
        for filename, content in task.files.items():
            file_path = output_path / filename
            with open(file_path, "w") as f:
                f.write(content)
            saved_files[filename] = str(file_path)

        return saved_files
