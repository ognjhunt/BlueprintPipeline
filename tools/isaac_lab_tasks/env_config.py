"""
Environment Configuration Generator for Isaac Lab tasks.

This module generates environment configurations compatible with
Isaac Lab's ManagerBasedEnv architecture.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SceneConfig:
    """Scene configuration."""
    num_envs: int = 1024
    env_spacing: float = 2.0
    ground_plane: bool = True
    scene_usd_path: str = ""


@dataclass
class SimConfig:
    """Simulation configuration."""
    dt: float = 1.0 / 60.0
    render_interval: int = 1
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    enable_scene_query: bool = True
    use_gpu_pipeline: bool = True


class EnvConfigGenerator:
    """
    Generates Isaac Lab environment configurations.

    This generator creates configurations compatible with Isaac Lab's
    manager-based environment architecture.
    """

    def __init__(self):
        self.scene_config = SceneConfig()
        self.sim_config = SimConfig()

    def generate_scene_config(
        self,
        recipe: dict[str, Any],
        num_envs: int = 1024
    ) -> SceneConfig:
        """Generate scene configuration from recipe."""
        scene_path = recipe.get("metadata", {}).get("scene_path", "scene.usda")
        room = recipe.get("room", {})
        dims = room.get("dimensions", {})

        # Calculate env spacing based on room size
        max_dim = max(
            dims.get("width", 5.0),
            dims.get("depth", 5.0)
        )
        env_spacing = max_dim * 1.5  # Add buffer

        return SceneConfig(
            num_envs=num_envs,
            env_spacing=env_spacing,
            ground_plane=True,
            scene_usd_path=scene_path
        )

    def generate_sim_config(
        self,
        physics_dt: float = 1.0 / 60.0,
        render_interval: int = 1
    ) -> SimConfig:
        """Generate simulation configuration."""
        return SimConfig(
            dt=physics_dt,
            render_interval=render_interval,
            gravity=(0.0, 0.0, -9.81),
            enable_scene_query=True,
            use_gpu_pipeline=True
        )

    def get_robot_config(self, robot_type: str) -> dict[str, Any]:
        """Get configuration for a specific robot type."""
        configs = {
            "franka": {
                "usd_path": "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/Franka/franka_instanceable.usd",
                "num_dofs": 7,
                "gripper_dofs": 2,
                "ee_frame": "panda_hand",
                "base_frame": "panda_link0",
                "default_joint_pos": {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.571,
                    "panda_joint7": 0.785,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                }
            },
            "ur10": {
                "usd_path": "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/UniversalRobots/ur10/ur10.usd",
                "num_dofs": 6,
                "gripper_dofs": 0,
                "ee_frame": "tool0",
                "base_frame": "base_link",
                "default_joint_pos": {
                    "shoulder_pan_joint": 0.0,
                    "shoulder_lift_joint": -1.571,
                    "elbow_joint": 1.571,
                    "wrist_1_joint": -1.571,
                    "wrist_2_joint": -1.571,
                    "wrist_3_joint": 0.0,
                }
            },
            "fetch": {
                "usd_path": "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/Fetch/fetch.usd",
                "num_dofs": 7,
                "gripper_dofs": 2,
                "ee_frame": "gripper_link",
                "base_frame": "base_link",
                "default_joint_pos": {}  # Would need actual joint names
            }
        }
        return configs.get(robot_type, configs["franka"])

    def get_sensor_configs(
        self,
        include_camera: bool = True,
        include_contact: bool = True
    ) -> dict[str, dict[str, Any]]:
        """Get sensor configurations."""
        sensors = {}

        if include_camera:
            sensors["camera"] = {
                "type": "Camera",
                "prim_path": "/World/Robot/camera",
                "resolution": (640, 480),
                "focal_length": 24.0,
                "clipping_range": (0.1, 100.0),
                "output_types": ["rgb", "depth"]
            }

        if include_contact:
            sensors["contact"] = {
                "type": "ContactSensor",
                "prim_path": "/World/Robot/.*",
                "filter_prim_paths": ["/World/Scene/.*"],
                "update_period": 0.0,
                "track_air_time": False
            }

        return sensors
