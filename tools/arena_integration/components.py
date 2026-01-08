"""
Modular Arena Components - Isaac Lab-Arena Lego Architecture.

This module implements Arena's modular component-based design where
Objects, Scenes, Embodiments, and Tasks are separate composable units.

This enables:
- Mix-and-match any Object with any Scene
- Swap robot Embodiments without code changes
- Reuse Task definitions across scenes
- Dynamic environment compilation

Key Difference from Genie Sim:
- Genie Sim: Data GENERATION (episodes, trajectories)
- Arena Components: Policy EVALUATION infrastructure

Usage:
    from tools.arena_integration.components import (
        ArenaObject, ArenaScene, ArenaEmbodiment, ArenaTask,
        ArenaEnvironmentBuilder
    )

    # Create modular components
    obj = ArenaObject.from_manifest_object(manifest_obj)
    scene = ArenaScene.from_manifest(manifest)
    robot = ArenaEmbodiment.franka()
    task = ArenaTask.pick_and_place(obj)

    # Compose into evaluation environment
    builder = ArenaEnvironmentBuilder()
    env_cfg = builder.build(scene=scene, task=task, embodiment=robot)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from .affordances import AffordanceType, AffordanceParams, AffordanceDetector


# =============================================================================
# ARENA OBJECT - Modular Object Component
# =============================================================================

class ObjectCategory(str, Enum):
    """Standard Arena object categories."""
    MANIPULABLE = "manipulable"
    ARTICULATED = "articulated"
    INTERACTIVE = "interactive"
    CONTAINER = "container"
    SURFACE = "surface"
    TOOL = "tool"
    DEFORMABLE = "deformable"


@dataclass
class ArenaObjectConfig:
    """Configuration for an Arena Object component."""
    usd_path: str
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    collision_enabled: bool = True
    rigid_body: bool = True
    mass_kg: Optional[float] = None
    friction: float = 0.5
    restitution: float = 0.0


@dataclass
class ArenaObject:
    """
    Modular Arena Object Component.

    Represents a single object that can be placed in any Arena Scene.
    Objects are self-contained with affordances, physics, and metadata.
    """
    object_id: str
    name: str
    category: ObjectCategory
    affordances: list[AffordanceType]
    affordance_params: dict[str, AffordanceParams]
    config: ArenaObjectConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional articulation info
    articulation_joints: list[str] = field(default_factory=list)
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def from_manifest_object(
        cls,
        obj: dict[str, Any],
        affordance_detector: Optional[AffordanceDetector] = None
    ) -> "ArenaObject":
        """
        Create ArenaObject from Blueprint scene manifest object.

        Args:
            obj: Object dictionary from scene_manifest.json
            affordance_detector: Optional detector for affordance inference
        """
        detector = affordance_detector or AffordanceDetector(use_llm=False)

        # Detect affordances
        detected = detector.detect(obj)
        affordance_types = [aff.affordance_type for aff in detected]
        affordance_params = {
            aff.affordance_type.value: aff for aff in detected
        }

        # Determine category
        sim_role = obj.get("sim_role", "manipulable_object")
        category = cls._infer_category(sim_role, affordance_types)

        # Build config
        asset = obj.get("asset", {})
        dims = obj.get("dimensions_est", {})
        physics = obj.get("physics", {})

        config = ArenaObjectConfig(
            usd_path=asset.get("path", ""),
            scale=(
                dims.get("scale_x", 1.0),
                dims.get("scale_y", 1.0),
                dims.get("scale_z", 1.0),
            ),
            collision_enabled=physics.get("collision_enabled", True),
            rigid_body=sim_role in ["manipulable_object", "clutter"],
            mass_kg=physics.get("mass_kg"),
            friction=physics.get("friction", 0.5),
        )

        # Extract articulation info
        articulation = obj.get("articulation", {})
        joints = []
        joint_limits = {}
        if articulation.get("physx_endpoint"):
            joints.append(articulation["physx_endpoint"])
            if articulation.get("joint_limits"):
                joint_limits[articulation["physx_endpoint"]] = tuple(
                    articulation["joint_limits"]
                )

        return cls(
            object_id=obj.get("id", "unknown"),
            name=obj.get("name") or obj.get("category", "object"),
            category=category,
            affordances=affordance_types,
            affordance_params=affordance_params,
            config=config,
            metadata={
                "source": "blueprint_manifest",
                "original_sim_role": sim_role,
                "description": obj.get("description", ""),
            },
            articulation_joints=joints,
            joint_limits=joint_limits,
        )

    @staticmethod
    def _infer_category(
        sim_role: str,
        affordances: list[AffordanceType]
    ) -> ObjectCategory:
        """Infer object category from sim_role and affordances."""
        if sim_role in ["articulated_appliance", "articulated_furniture"]:
            return ObjectCategory.ARTICULATED
        if sim_role == "interactive":
            return ObjectCategory.INTERACTIVE
        if AffordanceType.CONTAINABLE in affordances:
            return ObjectCategory.CONTAINER
        if AffordanceType.PLACEABLE in affordances:
            return ObjectCategory.SURFACE
        if AffordanceType.FOLDABLE in affordances:
            return ObjectCategory.DEFORMABLE
        return ObjectCategory.MANIPULABLE

    def to_arena_dict(self) -> dict[str, Any]:
        """Export as Arena-compatible dictionary."""
        return {
            "id": self.object_id,
            "name": self.name,
            "category": self.category.value,
            "affordances": [aff.value for aff in self.affordances],
            "affordance_params": {
                k: self._affordance_params_to_dict(v)
                for k, v in self.affordance_params.items()
            },
            "config": {
                "usd_path": self.config.usd_path,
                "scale": list(self.config.scale),
                "collision_enabled": self.config.collision_enabled,
                "rigid_body": self.config.rigid_body,
                "mass_kg": self.config.mass_kg,
                "friction": self.config.friction,
                "restitution": self.config.restitution,
            },
            "articulation": {
                "joints": self.articulation_joints,
                "joint_limits": {k: list(v) for k, v in self.joint_limits.items()},
            },
            "metadata": self.metadata,
        }

    @staticmethod
    def _affordance_params_to_dict(params: AffordanceParams) -> dict[str, Any]:
        """Convert AffordanceParams to dictionary."""
        return {
            "type": params.affordance_type.value,
            "confidence": params.confidence,
            "source": params.source,
            "joint_type": params.joint_type,
            "open_angle": params.open_angle,
            "open_distance": params.open_distance,
            "grasp_width_range": list(params.grasp_width_range),
            "grasp_approach_direction": params.grasp_approach_direction,
        }


# =============================================================================
# ARENA SCENE - Modular Scene Component
# =============================================================================

class EnvironmentType(str, Enum):
    """Standard Arena environment types."""
    KITCHEN = "kitchen"
    WAREHOUSE = "warehouse"
    LAB = "lab"
    OFFICE = "office"
    HOME = "home"
    GROCERY = "grocery"
    INDUSTRIAL = "industrial"
    CUSTOM = "custom"


@dataclass
class ArenaSceneConfig:
    """Configuration for an Arena Scene component."""
    usd_path: str
    coordinate_frame: str = "z_up"
    meters_per_unit: float = 1.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    ground_plane: bool = True


@dataclass
class ArenaScene:
    """
    Modular Arena Scene Component.

    Represents a complete environment that can host any Objects.
    Scenes define the spatial structure and physics context.
    """
    scene_id: str
    environment_type: EnvironmentType
    objects: list[ArenaObject]
    config: ArenaSceneConfig

    # Placement regions for object spawning
    placement_regions: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Scene metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(
        cls,
        manifest: dict[str, Any],
        affordance_detector: Optional[AffordanceDetector] = None
    ) -> "ArenaScene":
        """
        Create ArenaScene from Blueprint scene manifest.

        Args:
            manifest: Full scene_manifest.json dictionary
            affordance_detector: Optional affordance detector
        """
        scene_meta = manifest.get("scene", {})
        scene_id = manifest.get("scene_id", "unknown")

        # Parse environment type
        env_type_str = scene_meta.get("environment_type", "custom")
        try:
            env_type = EnvironmentType(env_type_str.lower())
        except ValueError:
            env_type = EnvironmentType.CUSTOM

        # Create ArenaObjects from manifest objects
        detector = affordance_detector or AffordanceDetector(use_llm=False)
        objects = [
            ArenaObject.from_manifest_object(obj, detector)
            for obj in manifest.get("objects", [])
        ]

        # Build config
        config = ArenaSceneConfig(
            usd_path=scene_meta.get("usd_path", "scene.usda"),
            coordinate_frame=scene_meta.get("coordinate_frame", "z_up"),
            meters_per_unit=scene_meta.get("meters_per_unit", 1.0),
        )

        # Extract placement regions if available
        placement_regions = manifest.get("placement_regions", {})

        return cls(
            scene_id=scene_id,
            environment_type=env_type,
            objects=objects,
            config=config,
            placement_regions=placement_regions,
            metadata={
                "source": "blueprint_manifest",
                "object_count": len(objects),
            },
        )

    def get_objects_by_affordance(
        self,
        affordance: AffordanceType
    ) -> list[ArenaObject]:
        """Get all objects with a specific affordance."""
        return [obj for obj in self.objects if affordance in obj.affordances]

    def get_objects_by_category(
        self,
        category: ObjectCategory
    ) -> list[ArenaObject]:
        """Get all objects of a specific category."""
        return [obj for obj in self.objects if obj.category == category]

    def to_arena_dict(self) -> dict[str, Any]:
        """Export as Arena-compatible dictionary."""
        return {
            "scene_id": self.scene_id,
            "environment_type": self.environment_type.value,
            "config": {
                "usd_path": self.config.usd_path,
                "coordinate_frame": self.config.coordinate_frame,
                "meters_per_unit": self.config.meters_per_unit,
                "gravity": list(self.config.gravity),
                "ground_plane": self.config.ground_plane,
            },
            "objects": [obj.to_arena_dict() for obj in self.objects],
            "placement_regions": self.placement_regions,
            "metadata": self.metadata,
        }


# =============================================================================
# ARENA EMBODIMENT - Modular Robot Component
# =============================================================================

class EmbodimentType(str, Enum):
    """Standard Arena robot embodiments."""
    FRANKA_PANDA = "franka"
    UR10 = "ur10"
    UR5E = "ur5e"
    FETCH = "fetch"
    GR1 = "gr1"           # Fourier GR1 humanoid
    G1 = "g1"             # Unitree G1
    SPOT = "spot"         # Boston Dynamics Spot
    CUSTOM = "custom"


@dataclass
class ArenaEmbodimentConfig:
    """Configuration for an Arena Embodiment component."""
    usd_path: str
    urdf_path: Optional[str] = None
    dof: int = 7
    gripper_dof: int = 2
    ee_frame: str = "ee_link"
    base_frame: str = "base_link"
    control_frequency: float = 120.0
    action_scale: float = 1.0


@dataclass
class ArenaEmbodiment:
    """
    Modular Arena Embodiment (Robot) Component.

    Represents a robot that can operate in any Arena Scene.
    Embodiments are swappable without task code changes.
    """
    embodiment_type: EmbodimentType
    name: str
    config: ArenaEmbodimentConfig

    # Joint configuration
    joint_names: list[str] = field(default_factory=list)
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Action/observation space info
    action_dim: int = 7
    observation_keys: list[str] = field(default_factory=list)

    # Capabilities
    capabilities: list[str] = field(default_factory=list)

    @classmethod
    def franka(cls) -> "ArenaEmbodiment":
        """Create Franka Panda embodiment."""
        return cls(
            embodiment_type=EmbodimentType.FRANKA_PANDA,
            name="Franka Panda",
            config=ArenaEmbodimentConfig(
                usd_path="omniverse://localhost/NVIDIA/Robots/Franka/franka.usd",
                dof=7,
                gripper_dof=2,
                ee_frame="panda_hand",
                base_frame="panda_link0",
            ),
            joint_names=[f"panda_joint{i}" for i in range(1, 8)],
            action_dim=9,  # 7 joints + 2 gripper
            observation_keys=[
                "joint_pos", "joint_vel", "ee_pos", "ee_quat",
                "gripper_state", "gripper_force"
            ],
            capabilities=["manipulation", "precision_grasp", "force_control"],
        )

    @classmethod
    def ur10(cls) -> "ArenaEmbodiment":
        """Create UR10 embodiment."""
        return cls(
            embodiment_type=EmbodimentType.UR10,
            name="Universal Robots UR10",
            config=ArenaEmbodimentConfig(
                usd_path="omniverse://localhost/NVIDIA/Robots/UR10/ur10.usd",
                dof=6,
                gripper_dof=2,
                ee_frame="tool0",
                base_frame="base_link",
            ),
            joint_names=[f"shoulder_pan_joint", "shoulder_lift_joint",
                        "elbow_joint", "wrist_1_joint",
                        "wrist_2_joint", "wrist_3_joint"],
            action_dim=8,  # 6 joints + 2 gripper
            observation_keys=[
                "joint_pos", "joint_vel", "ee_pos", "ee_quat",
                "gripper_state"
            ],
            capabilities=["manipulation", "heavy_payload", "reach"],
        )

    @classmethod
    def gr1(cls) -> "ArenaEmbodiment":
        """Create Fourier GR1 humanoid embodiment."""
        return cls(
            embodiment_type=EmbodimentType.GR1,
            name="Fourier GR1",
            config=ArenaEmbodimentConfig(
                usd_path="omniverse://localhost/NVIDIA/Robots/GR1/gr1.usd",
                dof=32,
                gripper_dof=10,  # 5 per hand
                ee_frame="right_hand",
                base_frame="pelvis",
            ),
            action_dim=42,  # Full body
            observation_keys=[
                "joint_pos", "joint_vel", "body_pos", "body_quat",
                "left_hand_pos", "right_hand_pos",
                "left_gripper_state", "right_gripper_state"
            ],
            capabilities=[
                "bimanual", "mobile", "humanoid",
                "loco_manipulation", "dexterous"
            ],
        )

    @classmethod
    def g1(cls) -> "ArenaEmbodiment":
        """Create Unitree G1 embodiment."""
        return cls(
            embodiment_type=EmbodimentType.G1,
            name="Unitree G1",
            config=ArenaEmbodimentConfig(
                usd_path="omniverse://localhost/NVIDIA/Robots/G1/g1.usd",
                dof=29,
                gripper_dof=6,
                ee_frame="right_gripper",
                base_frame="pelvis",
            ),
            action_dim=35,
            observation_keys=[
                "joint_pos", "joint_vel", "body_pos", "body_quat",
                "imu_data", "foot_contacts"
            ],
            capabilities=["bimanual", "mobile", "humanoid", "loco_manipulation"],
        )

    @classmethod
    def from_type(cls, embodiment_type: str) -> "ArenaEmbodiment":
        """Factory method to create embodiment from type string."""
        factories = {
            "franka": cls.franka,
            "ur10": cls.ur10,
            "gr1": cls.gr1,
            "g1": cls.g1,
        }
        factory = factories.get(embodiment_type.lower())
        if factory:
            return factory()
        raise ValueError(f"Unknown embodiment type: {embodiment_type}")

    def to_arena_dict(self) -> dict[str, Any]:
        """Export as Arena-compatible dictionary."""
        return {
            "type": self.embodiment_type.value,
            "name": self.name,
            "config": {
                "usd_path": self.config.usd_path,
                "urdf_path": self.config.urdf_path,
                "dof": self.config.dof,
                "gripper_dof": self.config.gripper_dof,
                "ee_frame": self.config.ee_frame,
                "base_frame": self.config.base_frame,
                "control_frequency": self.config.control_frequency,
            },
            "joint_names": self.joint_names,
            "action_dim": self.action_dim,
            "observation_keys": self.observation_keys,
            "capabilities": self.capabilities,
        }


# =============================================================================
# ARENA TASK - Modular Task Component
# =============================================================================

class TaskDifficulty(str, Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class ArenaTaskConfig:
    """Configuration for an Arena Task component."""
    max_steps: int = 500
    success_threshold: float = 0.9
    reward_scale: float = 1.0
    early_termination: bool = True
    domain_randomization: bool = False


@dataclass
class ArenaTask:
    """
    Modular Arena Task Component.

    Represents a task that can be executed with any Object in any Scene.
    Tasks define objectives, rewards, and success criteria.
    """
    task_id: str
    name: str
    description: str
    required_affordances: list[AffordanceType]
    config: ArenaTaskConfig
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM

    # Task parameters
    params: dict[str, Any] = field(default_factory=dict)

    # Required observations
    observation_keys: list[str] = field(default_factory=list)

    # Compatible embodiments (empty = all)
    compatible_embodiments: list[EmbodimentType] = field(default_factory=list)

    # Success criteria function name
    success_fn: str = "default_success"

    # Reward function name
    reward_fn: str = "default_reward"

    @classmethod
    def pick_object(
        cls,
        target_object: Optional[ArenaObject] = None,
        lift_height: float = 0.15
    ) -> "ArenaTask":
        """Create pick object task."""
        return cls(
            task_id="pick_object",
            name="Pick Object",
            description="Grasp and lift target object",
            required_affordances=[AffordanceType.GRASPABLE],
            config=ArenaTaskConfig(max_steps=300),
            difficulty=TaskDifficulty.EASY,
            params={
                "target_object_id": target_object.object_id if target_object else None,
                "lift_height": lift_height,
                "grasp_force_range": (5.0, 50.0),
            },
            observation_keys=[
                "object_pos", "object_quat", "ee_pos", "gripper_state"
            ],
            success_fn="grasp_lift_success",
            reward_fn="grasp_lift_reward",
        )

    @classmethod
    def pick_and_place(
        cls,
        target_object: Optional[ArenaObject] = None,
        target_position: Optional[tuple[float, float, float]] = None
    ) -> "ArenaTask":
        """Create pick and place task."""
        return cls(
            task_id="pick_and_place",
            name="Pick and Place",
            description="Pick object and place at target location",
            required_affordances=[AffordanceType.GRASPABLE],
            config=ArenaTaskConfig(max_steps=500),
            difficulty=TaskDifficulty.MEDIUM,
            params={
                "target_object_id": target_object.object_id if target_object else None,
                "target_position": list(target_position) if target_position else None,
                "placement_tolerance": 0.03,
            },
            observation_keys=[
                "object_pos", "object_quat", "target_pos",
                "ee_pos", "gripper_state"
            ],
            success_fn="placement_success",
            reward_fn="placement_reward",
        )

    @classmethod
    def open_articulated(
        cls,
        target_object: Optional[ArenaObject] = None,
        target_openness: float = 0.9
    ) -> "ArenaTask":
        """Create open articulated object task (door/drawer)."""
        return cls(
            task_id="open_articulated",
            name="Open Articulated",
            description="Open door, drawer, or lid",
            required_affordances=[AffordanceType.OPENABLE],
            config=ArenaTaskConfig(max_steps=400),
            difficulty=TaskDifficulty.MEDIUM,
            params={
                "target_object_id": target_object.object_id if target_object else None,
                "target_openness": target_openness,
                "reset_openness": 0.1,
            },
            observation_keys=[
                "joint_pos", "joint_vel", "handle_pos", "ee_pos"
            ],
            success_fn="articulation_success",
            reward_fn="articulation_reward",
        )

    @classmethod
    def turn_knob(
        cls,
        target_object: Optional[ArenaObject] = None,
        target_rotation: float = 1.57  # 90 degrees
    ) -> "ArenaTask":
        """Create turn knob task."""
        return cls(
            task_id="turn_knob",
            name="Turn Knob",
            description="Rotate knob to target position",
            required_affordances=[AffordanceType.TURNABLE],
            config=ArenaTaskConfig(max_steps=300),
            difficulty=TaskDifficulty.MEDIUM,
            params={
                "target_object_id": target_object.object_id if target_object else None,
                "target_rotation": target_rotation,
                "rotation_tolerance": 0.1,
            },
            observation_keys=[
                "knob_angle", "ee_pos", "contact_force"
            ],
            success_fn="rotation_success",
            reward_fn="rotation_reward",
        )

    @classmethod
    def press_button(
        cls,
        target_object: Optional[ArenaObject] = None
    ) -> "ArenaTask":
        """Create press button task."""
        return cls(
            task_id="press_button",
            name="Press Button",
            description="Press button or toggle switch",
            required_affordances=[AffordanceType.PRESSABLE],
            config=ArenaTaskConfig(max_steps=200),
            difficulty=TaskDifficulty.EASY,
            params={
                "target_object_id": target_object.object_id if target_object else None,
                "press_force_threshold": 5.0,
                "hold_duration": 0.1,
            },
            observation_keys=[
                "button_state", "ee_pos", "contact_force"
            ],
            success_fn="button_press_success",
            reward_fn="button_press_reward",
        )

    def to_arena_dict(self) -> dict[str, Any]:
        """Export as Arena-compatible dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "required_affordances": [aff.value for aff in self.required_affordances],
            "config": {
                "max_steps": self.config.max_steps,
                "success_threshold": self.config.success_threshold,
                "reward_scale": self.config.reward_scale,
                "early_termination": self.config.early_termination,
                "domain_randomization": self.config.domain_randomization,
            },
            "difficulty": self.difficulty.value,
            "params": self.params,
            "observation_keys": self.observation_keys,
            "compatible_embodiments": [e.value for e in self.compatible_embodiments],
            "success_fn": self.success_fn,
            "reward_fn": self.reward_fn,
        }


# =============================================================================
# ARENA ENVIRONMENT BUILDER - Composes Components into Environments
# =============================================================================

@dataclass
class ArenaEnvironmentConfig:
    """Configuration for composed Arena Environment."""
    num_envs: int = 1
    device: str = "cuda:0"
    headless: bool = True
    render_mode: str = "rgb_array"
    physics_dt: float = 1 / 120.0
    rendering_dt: float = 1 / 60.0


@dataclass
class ArenaEnvironmentSpec:
    """
    Composed Arena Environment Specification.

    This is the output of ArenaEnvironmentBuilder - a complete
    specification that can be used to instantiate an Isaac Lab env.
    """
    scene: ArenaScene
    task: ArenaTask
    embodiment: ArenaEmbodiment
    config: ArenaEnvironmentConfig

    # Computed fields
    action_space_dim: int = 0
    observation_space_keys: list[str] = field(default_factory=list)

    def to_isaac_lab_cfg(self) -> dict[str, Any]:
        """
        Convert to Isaac Lab ManagerBasedEnvCfg compatible dict.

        This output can be used to create an Isaac Lab environment
        for policy evaluation.
        """
        return {
            "scene": {
                "scene_id": self.scene.scene_id,
                "usd_path": self.scene.config.usd_path,
                "objects": [obj.to_arena_dict() for obj in self.scene.objects],
            },
            "task": self.task.to_arena_dict(),
            "robot": self.embodiment.to_arena_dict(),
            "sim": {
                "num_envs": self.config.num_envs,
                "device": self.config.device,
                "headless": self.config.headless,
                "physics_dt": self.config.physics_dt,
                "rendering_dt": self.config.rendering_dt,
            },
            "observation_space": self.observation_space_keys,
            "action_space_dim": self.action_space_dim,
        }

    def to_json(self, path: Path) -> None:
        """Save environment spec to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_isaac_lab_cfg(), f, indent=2)


class ArenaEnvironmentBuilder:
    """
    Builder for composing Arena components into evaluation environments.

    This implements the "Lego-like" architecture from Isaac Lab-Arena,
    allowing mix-and-match of Objects, Scenes, Embodiments, and Tasks.

    Usage:
        builder = ArenaEnvironmentBuilder()
        env_spec = builder.build(
            scene=my_scene,
            task=ArenaTask.pick_and_place(),
            embodiment=ArenaEmbodiment.franka()
        )
    """

    def __init__(self, default_config: Optional[ArenaEnvironmentConfig] = None):
        self.default_config = default_config or ArenaEnvironmentConfig()

    def build(
        self,
        scene: ArenaScene,
        task: ArenaTask,
        embodiment: ArenaEmbodiment,
        config: Optional[ArenaEnvironmentConfig] = None,
    ) -> ArenaEnvironmentSpec:
        """
        Compose components into an environment specification.

        Args:
            scene: ArenaScene component
            task: ArenaTask component
            embodiment: ArenaEmbodiment component
            config: Optional environment configuration

        Returns:
            ArenaEnvironmentSpec ready for Isaac Lab instantiation
        """
        cfg = config or self.default_config

        # Validate compatibility
        self._validate_task_scene_compatibility(task, scene)
        self._validate_task_embodiment_compatibility(task, embodiment)

        # Compute observation space
        obs_keys = list(set(task.observation_keys + embodiment.observation_keys))

        # Compute action space
        action_dim = embodiment.action_dim

        return ArenaEnvironmentSpec(
            scene=scene,
            task=task,
            embodiment=embodiment,
            config=cfg,
            action_space_dim=action_dim,
            observation_space_keys=obs_keys,
        )

    def _validate_task_scene_compatibility(
        self,
        task: ArenaTask,
        scene: ArenaScene
    ) -> None:
        """Validate task can be executed in scene."""
        # Check if scene has objects with required affordances
        scene_affordances = set()
        for obj in scene.objects:
            scene_affordances.update(obj.affordances)

        missing = set(task.required_affordances) - scene_affordances
        if missing:
            missing_str = ", ".join(aff.value for aff in missing)
            raise ValueError(
                f"Scene missing required affordances for task: {missing_str}"
            )

    def _validate_task_embodiment_compatibility(
        self,
        task: ArenaTask,
        embodiment: ArenaEmbodiment
    ) -> None:
        """Validate embodiment can execute task."""
        if task.compatible_embodiments:
            if embodiment.embodiment_type not in task.compatible_embodiments:
                raise ValueError(
                    f"Embodiment {embodiment.embodiment_type.value} not compatible "
                    f"with task {task.task_id}"
                )

    def build_all_tasks_for_scene(
        self,
        scene: ArenaScene,
        embodiment: ArenaEmbodiment,
        config: Optional[ArenaEnvironmentConfig] = None,
    ) -> list[ArenaEnvironmentSpec]:
        """
        Build all possible task environments for a scene.

        Automatically generates tasks based on object affordances.
        """
        task_specs = []

        for obj in scene.objects:
            # Generate tasks based on affordances
            if AffordanceType.GRASPABLE in obj.affordances:
                task = ArenaTask.pick_object(target_object=obj)
                try:
                    spec = self.build(scene, task, embodiment, config)
                    task_specs.append(spec)
                except ValueError:
                    pass  # Skip incompatible combinations

                task = ArenaTask.pick_and_place(target_object=obj)
                try:
                    spec = self.build(scene, task, embodiment, config)
                    task_specs.append(spec)
                except ValueError:
                    pass

            if AffordanceType.OPENABLE in obj.affordances:
                task = ArenaTask.open_articulated(target_object=obj)
                try:
                    spec = self.build(scene, task, embodiment, config)
                    task_specs.append(spec)
                except ValueError:
                    pass

            if AffordanceType.TURNABLE in obj.affordances:
                task = ArenaTask.turn_knob(target_object=obj)
                try:
                    spec = self.build(scene, task, embodiment, config)
                    task_specs.append(spec)
                except ValueError:
                    pass

            if AffordanceType.PRESSABLE in obj.affordances:
                task = ArenaTask.press_button(target_object=obj)
                try:
                    spec = self.build(scene, task, embodiment, config)
                    task_specs.append(spec)
                except ValueError:
                    pass

        return task_specs


# =============================================================================
# COMPONENT REGISTRY - For Dynamic Component Discovery
# =============================================================================

class ArenaComponentRegistry:
    """
    Registry for Arena components enabling dynamic discovery.

    Similar to Arena's asset registry, this enables dynamic
    procurement of pre-built components.
    """

    def __init__(self):
        self._objects: dict[str, ArenaObject] = {}
        self._scenes: dict[str, ArenaScene] = {}
        self._embodiments: dict[str, ArenaEmbodiment] = {}
        self._tasks: dict[str, type] = {}

    def register_object(self, obj: ArenaObject) -> None:
        """Register an object component."""
        self._objects[obj.object_id] = obj

    def register_scene(self, scene: ArenaScene) -> None:
        """Register a scene component."""
        self._scenes[scene.scene_id] = scene

    def register_embodiment(self, embodiment: ArenaEmbodiment) -> None:
        """Register an embodiment component."""
        self._embodiments[embodiment.embodiment_type.value] = embodiment

    def get_object(self, object_id: str) -> Optional[ArenaObject]:
        """Get registered object by ID."""
        return self._objects.get(object_id)

    def get_scene(self, scene_id: str) -> Optional[ArenaScene]:
        """Get registered scene by ID."""
        return self._scenes.get(scene_id)

    def get_embodiment(self, embodiment_type: str) -> Optional[ArenaEmbodiment]:
        """Get registered embodiment by type."""
        return self._embodiments.get(embodiment_type)

    def list_objects(self) -> list[str]:
        """List all registered object IDs."""
        return list(self._objects.keys())

    def list_scenes(self) -> list[str]:
        """List all registered scene IDs."""
        return list(self._scenes.keys())

    def list_embodiments(self) -> list[str]:
        """List all registered embodiment types."""
        return list(self._embodiments.keys())

    def to_json(self) -> dict[str, Any]:
        """Export registry to JSON."""
        return {
            "objects": {k: v.to_arena_dict() for k, v in self._objects.items()},
            "scenes": {k: v.to_arena_dict() for k, v in self._scenes.items()},
            "embodiments": {k: v.to_arena_dict() for k, v in self._embodiments.items()},
        }

    def save(self, path: Path) -> None:
        """Save registry to file."""
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)


# Global registry instance
_global_registry = ArenaComponentRegistry()


def get_registry() -> ArenaComponentRegistry:
    """Get the global component registry."""
    return _global_registry
