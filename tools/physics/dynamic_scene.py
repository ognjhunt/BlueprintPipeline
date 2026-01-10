"""Dynamic Scene Changes Support.

Enable dynamic changes to scenes during episode generation:
- Moving obstacles (conveyor belts, moving platforms)
- Human models moving through the scene
- Dynamic lighting changes
- Object appearance/disappearance

This creates more realistic and challenging training scenarios.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

logger = logging.getLogger(__name__)


class DynamicObstacleType(str, Enum):
    """Types of dynamic obstacles."""
    MOVING_PLATFORM = "moving_platform"  # Platform moving linearly
    ROTATING_OBSTACLE = "rotating_obstacle"  # Rotating obstacle
    CONVEYOR_BELT = "conveyor_belt"  # Conveyor belt
    SWINGING_DOOR = "swinging_door"  # Door that swings
    ELEVATOR = "elevator"  # Vertical moving platform


class HumanMotionPattern(str, Enum):
    """Human motion patterns."""
    WALKING_STRAIGHT = "walking_straight"  # Walk in straight line
    WALKING_PATH = "walking_path"  # Follow predefined path
    STANDING_IDLE = "standing_idle"  # Standing with idle motion
    REACHING = "reaching"  # Reaching for objects
    RANDOM_WALK = "random_walk"  # Random walking


@dataclass
class DynamicObstacle:
    """Configuration for a dynamic obstacle."""

    # Identity
    obstacle_id: str
    obstacle_type: DynamicObstacleType

    # Initial state
    spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spawn_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    # Motion parameters
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # m/s
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # rad/s

    # Path/trajectory
    waypoints: List[Tuple[float, float, float]] = field(default_factory=list)
    loop_path: bool = True

    # Constraints
    min_position: Tuple[float, float, float] = (-2.0, -2.0, 0.0)
    max_position: Tuple[float, float, float] = (2.0, 2.0, 2.0)

    # Dimensions
    size: Tuple[float, float, float] = (0.5, 0.5, 0.1)

    # Active time range
    start_time: float = 0.0  # seconds
    end_time: Optional[float] = None  # None = always active

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obstacle_id": self.obstacle_id,
            "obstacle_type": self.obstacle_type.value,
            "spawn_position": list(self.spawn_position),
            "spawn_orientation": list(self.spawn_orientation),
            "velocity": list(self.velocity),
            "angular_velocity": list(self.angular_velocity),
            "waypoints": [list(wp) for wp in self.waypoints],
            "loop_path": self.loop_path,
            "min_position": list(self.min_position),
            "max_position": list(self.max_position),
            "size": list(self.size),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class HumanModel:
    """Configuration for a human model in the scene."""

    # Identity
    human_id: str

    # Appearance
    gender: str = "neutral"  # "male", "female", "neutral"
    height: float = 1.7  # meters
    clothing: str = "casual"  # "casual", "work", "formal"

    # Motion
    motion_pattern: HumanMotionPattern = HumanMotionPattern.WALKING_STRAIGHT

    # Initial state
    spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spawn_orientation: float = 0.0  # radians (yaw)

    # Motion parameters
    walking_speed: float = 1.2  # m/s
    path_waypoints: List[Tuple[float, float, float]] = field(default_factory=list)

    # Interaction
    can_interact_with_objects: bool = False
    interaction_probability: float = 0.1  # Probability per second

    # Active time range
    start_time: float = 0.0  # seconds
    end_time: Optional[float] = None  # None = always active

    def to_dict(self) -> Dict[str, Any]:
        return {
            "human_id": self.human_id,
            "gender": self.gender,
            "height": self.height,
            "clothing": self.clothing,
            "motion_pattern": self.motion_pattern.value,
            "spawn_position": list(self.spawn_position),
            "spawn_orientation": self.spawn_orientation,
            "walking_speed": self.walking_speed,
            "path_waypoints": [list(wp) for wp in self.path_waypoints],
            "can_interact_with_objects": self.can_interact_with_objects,
            "interaction_probability": self.interaction_probability,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class LightingDynamics:
    """Dynamic lighting configuration."""

    # Enable dynamic lighting
    enabled: bool = False

    # Lighting parameters
    ambient_light_range: Tuple[float, float] = (0.3, 1.0)  # Min, max intensity
    shadow_sharpness_range: Tuple[float, float] = (0.5, 1.0)

    # Change frequency
    change_interval_seconds: float = 5.0
    smooth_transition: bool = True
    transition_duration: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ambient_light_range": list(self.ambient_light_range),
            "shadow_sharpness_range": list(self.shadow_sharpness_range),
            "change_interval_seconds": self.change_interval_seconds,
            "smooth_transition": self.smooth_transition,
            "transition_duration": self.transition_duration,
        }


@dataclass
class SceneDynamicsConfig:
    """Complete scene dynamics configuration."""

    # Dynamic obstacles
    obstacles: List[DynamicObstacle] = field(default_factory=list)

    # Human models
    humans: List[HumanModel] = field(default_factory=list)

    # Lighting
    lighting: LightingDynamics = field(default_factory=LightingDynamics)

    # Object spawning/despawning
    dynamic_object_spawning: bool = False
    spawn_rate: float = 0.1  # Objects per second

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obstacles": [obs.to_dict() for obs in self.obstacles],
            "humans": [human.to_dict() for human in self.humans],
            "lighting": self.lighting.to_dict(),
            "dynamic_object_spawning": self.dynamic_object_spawning,
            "spawn_rate": self.spawn_rate,
        }


class DynamicSceneManager:
    """Manage dynamic scene changes during episode generation.

    Example:
        manager = DynamicSceneManager()

        # Add moving platform
        platform = DynamicObstacle(
            obstacle_id="platform_1",
            obstacle_type=DynamicObstacleType.MOVING_PLATFORM,
            spawn_position=(0.0, 0.5, 0.5),
            velocity=(0.2, 0.0, 0.0),
        )
        manager.add_obstacle(platform)

        # Add human walking through scene
        human = HumanModel(
            human_id="person_1",
            motion_pattern=HumanMotionPattern.WALKING_STRAIGHT,
            spawn_position=(-2.0, 0.0, 0.0),
            walking_speed=1.2,
        )
        manager.add_human(human)

        # Enable dynamic lighting
        manager.enable_dynamic_lighting(
            change_interval_seconds=10.0
        )

        # Export configuration
        config = manager.export_config()
    """

    def __init__(self, enable_logging: bool = True):
        """Initialize dynamic scene manager.

        Args:
            enable_logging: Whether to log events
        """
        self.enable_logging = enable_logging

        self.obstacles: List[DynamicObstacle] = []
        self.humans: List[HumanModel] = []
        self.lighting = LightingDynamics()

        self.dynamic_object_spawning = False
        self.spawn_rate = 0.1

    def log(self, msg: str) -> None:
        """Log if enabled."""
        if self.enable_logging:
            logger.info(f"[DYNAMIC_SCENE] {msg}")

    def add_obstacle(self, obstacle: DynamicObstacle) -> None:
        """Add a dynamic obstacle.

        Args:
            obstacle: Obstacle configuration
        """
        self.obstacles.append(obstacle)
        self.log(f"Added obstacle: {obstacle.obstacle_id} ({obstacle.obstacle_type.value})")

    def add_human(self, human: HumanModel) -> None:
        """Add a human model.

        Args:
            human: Human configuration
        """
        self.humans.append(human)
        self.log(f"Added human: {human.human_id} ({human.motion_pattern.value})")

    def enable_dynamic_lighting(
        self,
        change_interval_seconds: float = 5.0,
        ambient_range: Tuple[float, float] = (0.3, 1.0),
    ) -> None:
        """Enable dynamic lighting.

        Args:
            change_interval_seconds: How often to change lighting
            ambient_range: Range of ambient light intensity
        """
        self.lighting.enabled = True
        self.lighting.change_interval_seconds = change_interval_seconds
        self.lighting.ambient_light_range = ambient_range

        self.log("Enabled dynamic lighting")

    def create_conveyor_belt(
        self,
        obstacle_id: str,
        position: Tuple[float, float, float],
        length: float,
        speed: float,
        direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> DynamicObstacle:
        """Create a conveyor belt obstacle.

        Args:
            obstacle_id: Obstacle identifier
            position: Center position
            length: Belt length (meters)
            speed: Belt speed (m/s)
            direction: Movement direction (normalized)

        Returns:
            DynamicObstacle configuration
        """
        # Normalize direction
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0:
            direction = tuple(np.array(direction) / dir_norm)

        velocity = tuple(np.array(direction) * speed)

        obstacle = DynamicObstacle(
            obstacle_id=obstacle_id,
            obstacle_type=DynamicObstacleType.CONVEYOR_BELT,
            spawn_position=position,
            velocity=velocity,
            size=(length, 0.5, 0.1),
        )

        self.add_obstacle(obstacle)
        return obstacle

    def create_walking_human(
        self,
        human_id: str,
        start_position: Tuple[float, float, float],
        end_position: Tuple[float, float, float],
        walking_speed: float = 1.2,
    ) -> HumanModel:
        """Create a human walking from start to end.

        Args:
            human_id: Human identifier
            start_position: Starting position
            end_position: Ending position
            walking_speed: Walking speed (m/s)

        Returns:
            HumanModel configuration
        """
        # Calculate orientation toward end position
        direction = np.array(end_position) - np.array(start_position)
        yaw = math.atan2(direction[1], direction[0])

        human = HumanModel(
            human_id=human_id,
            motion_pattern=HumanMotionPattern.WALKING_STRAIGHT,
            spawn_position=start_position,
            spawn_orientation=yaw,
            walking_speed=walking_speed,
            path_waypoints=[end_position],
        )

        self.add_human(human)
        return human

    def create_circular_path_human(
        self,
        human_id: str,
        center: Tuple[float, float, float],
        radius: float,
        num_waypoints: int = 8,
        walking_speed: float = 1.0,
    ) -> HumanModel:
        """Create a human walking in circular path.

        Args:
            human_id: Human identifier
            center: Circle center
            radius: Circle radius
            num_waypoints: Number of waypoints
            walking_speed: Walking speed (m/s)

        Returns:
            HumanModel configuration
        """
        waypoints = []

        for i in range(num_waypoints):
            angle = 2 * math.pi * i / num_waypoints

            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = center[2]

            waypoints.append((x, y, z))

        # Start at first waypoint
        start_pos = waypoints[0]

        human = HumanModel(
            human_id=human_id,
            motion_pattern=HumanMotionPattern.WALKING_PATH,
            spawn_position=start_pos,
            walking_speed=walking_speed,
            path_waypoints=waypoints,
        )

        self.add_human(human)
        return human

    def detect_potential_collisions(
        self,
        robot_workspace: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        time_horizon: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Detect potential collisions with dynamic elements.

        Args:
            robot_workspace: (min_position, max_position) of robot workspace
            time_horizon: Time horizon for prediction (seconds)

        Returns:
            List of potential collision events
        """
        collisions = []

        workspace_min, workspace_max = robot_workspace

        # Check obstacles
        for obstacle in self.obstacles:
            # Predict position at future times
            for t in np.linspace(0, time_horizon, 20):
                pos = np.array(obstacle.spawn_position) + np.array(obstacle.velocity) * t

                # Check if in workspace
                if (workspace_min[0] <= pos[0] <= workspace_max[0] and
                    workspace_min[1] <= pos[1] <= workspace_max[1] and
                    workspace_min[2] <= pos[2] <= workspace_max[2]):

                    collisions.append({
                        "type": "obstacle",
                        "obstacle_id": obstacle.obstacle_id,
                        "time": t,
                        "position": tuple(pos),
                        "severity": "high",
                    })
                    break

        # Check humans
        for human in self.humans:
            # Predict position at future times
            if human.motion_pattern == HumanMotionPattern.WALKING_STRAIGHT and human.path_waypoints:
                start = np.array(human.spawn_position)
                end = np.array(human.path_waypoints[0])
                direction = end - start
                distance = np.linalg.norm(direction)

                if distance > 0:
                    direction = direction / distance

                    for t in np.linspace(0, min(time_horizon, distance / human.walking_speed), 20):
                        pos = start + direction * human.walking_speed * t

                        # Check if in workspace
                        if (workspace_min[0] <= pos[0] <= workspace_max[0] and
                            workspace_min[1] <= pos[1] <= workspace_max[1] and
                            workspace_min[2] <= pos[2] <= workspace_max[2]):

                            collisions.append({
                                "type": "human",
                                "human_id": human.human_id,
                                "time": t,
                                "position": tuple(pos),
                                "severity": "critical",  # Human collision is critical
                            })
                            break

        if collisions:
            self.log(f"Detected {len(collisions)} potential collisions")

        return collisions

    def export_config(self) -> SceneDynamicsConfig:
        """Export complete scene dynamics configuration.

        Returns:
            SceneDynamicsConfig
        """
        return SceneDynamicsConfig(
            obstacles=self.obstacles,
            humans=self.humans,
            lighting=self.lighting,
            dynamic_object_spawning=self.dynamic_object_spawning,
            spawn_rate=self.spawn_rate,
        )

    def export_isaac_sim_config(self) -> Dict[str, Any]:
        """Export configuration for Isaac Sim.

        Returns:
            Isaac Sim compatible configuration
        """
        config = {
            "dynamic_obstacles": [],
            "humans": [],
            "lighting": {},
        }

        # Obstacles
        for obstacle in self.obstacles:
            config["dynamic_obstacles"].append({
                "name": obstacle.obstacle_id,
                "type": obstacle.obstacle_type.value,
                "position": list(obstacle.spawn_position),
                "orientation": list(obstacle.spawn_orientation),
                "linear_velocity": list(obstacle.velocity),
                "angular_velocity": list(obstacle.angular_velocity),
                "size": list(obstacle.size),
            })

        # Humans
        for human in self.humans:
            config["humans"].append({
                "name": human.human_id,
                "motion_type": human.motion_pattern.value,
                "position": list(human.spawn_position),
                "orientation": human.spawn_orientation,
                "speed": human.walking_speed,
                "waypoints": [list(wp) for wp in human.path_waypoints],
                "height": human.height,
            })

        # Lighting
        if self.lighting.enabled:
            config["lighting"] = {
                "dynamic": True,
                "change_interval": self.lighting.change_interval_seconds,
                "ambient_range": list(self.lighting.ambient_light_range),
            }

        return config
