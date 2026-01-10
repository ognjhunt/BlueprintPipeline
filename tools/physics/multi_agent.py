"""Multi-Agent Scene Coordination.

Support for scenes with multiple robots working together on collaborative
manipulation tasks. Handles collision avoidance, task allocation, and
synchronized motion planning.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CollisionAvoidanceStrategy(str, Enum):
    """Collision avoidance strategies for multi-agent scenes."""
    NONE = "none"  # No collision avoidance (robots are far apart)
    SIMPLE_SPHERE = "simple_sphere"  # Simple bounding sphere checking
    VELOCITY_OBSTACLES = "velocity_obstacles"  # Velocity obstacle method
    RRT_CONNECT = "rrt_connect"  # Multi-agent RRT path planning
    PRIORITY_BASED = "priority_based"  # Priority-based conflict resolution


class TaskAllocationStrategy(str, Enum):
    """Task allocation strategies."""
    MANUAL = "manual"  # Manually specified
    NEAREST_NEIGHBOR = "nearest_neighbor"  # Assign nearest object to each robot
    OPTIMAL_ASSIGNMENT = "optimal_assignment"  # Optimal assignment (Hungarian algorithm)
    SEQUENTIAL = "sequential"  # Robots work sequentially
    SYNCHRONIZED = "synchronized"  # All robots work together on same object


@dataclass
class AgentConfiguration:
    """Configuration for a single agent in multi-agent scene."""

    # Agent identity
    agent_id: str
    robot_type: str  # e.g., "franka_panda", "ur5e"

    # Spawn configuration
    spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spawn_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # Quaternion

    # Workspace bounds (relative to spawn position)
    workspace_bounds: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Capabilities
    max_reach: float = 0.8  # meters
    max_payload: float = 5.0  # kg
    gripper_width: float = 0.08  # meters

    # Task assignment
    assigned_objects: List[str] = field(default_factory=list)
    role: str = "manipulator"  # "manipulator", "holder", "observer"

    # Priority (higher = gets right-of-way in conflicts)
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "robot_type": self.robot_type,
            "spawn_position": list(self.spawn_position),
            "spawn_orientation": list(self.spawn_orientation),
            "workspace_bounds": list(self.workspace_bounds),
            "max_reach": self.max_reach,
            "max_payload": self.max_payload,
            "gripper_width": self.gripper_width,
            "assigned_objects": self.assigned_objects,
            "role": self.role,
            "priority": self.priority,
        }


class MultiAgentCoordinator:
    """Coordinate multiple robots in shared workspace.

    Example:
        coordinator = MultiAgentCoordinator()

        # Add agents
        agent1 = AgentConfiguration(
            agent_id="robot_1",
            robot_type="franka_panda",
            spawn_position=(-0.5, 0.0, 0.0),
        )
        agent2 = AgentConfiguration(
            agent_id="robot_2",
            robot_type="franka_panda",
            spawn_position=(0.5, 0.0, 0.0),
        )

        coordinator.add_agent(agent1)
        coordinator.add_agent(agent2)

        # Allocate tasks
        objects = ["obj_001", "obj_002", "obj_003"]
        coordinator.allocate_tasks(objects, strategy="nearest_neighbor")

        # Check for workspace conflicts
        conflicts = coordinator.detect_workspace_conflicts()
    """

    def __init__(
        self,
        collision_avoidance: CollisionAvoidanceStrategy = CollisionAvoidanceStrategy.SIMPLE_SPHERE,
        min_separation_distance: float = 0.3,
        enable_logging: bool = True,
    ):
        """Initialize multi-agent coordinator.

        Args:
            collision_avoidance: Collision avoidance strategy
            min_separation_distance: Minimum distance between agents (meters)
            enable_logging: Whether to log coordination events
        """
        self.collision_avoidance = collision_avoidance
        self.min_separation_distance = min_separation_distance
        self.enable_logging = enable_logging

        self.agents: Dict[str, AgentConfiguration] = {}
        self.object_positions: Dict[str, Tuple[float, float, float]] = {}

    def log(self, msg: str) -> None:
        """Log if enabled."""
        if self.enable_logging:
            logger.info(f"[MULTI_AGENT] {msg}")

    def add_agent(self, agent: AgentConfiguration) -> None:
        """Add an agent to the coordinator.

        Args:
            agent: Agent configuration
        """
        self.agents[agent.agent_id] = agent
        self.log(f"Added agent: {agent.agent_id} ({agent.robot_type})")

    def set_object_positions(
        self,
        object_positions: Dict[str, Tuple[float, float, float]]
    ) -> None:
        """Set object positions for task allocation.

        Args:
            object_positions: Dict mapping object_id -> (x, y, z) position
        """
        self.object_positions = object_positions
        self.log(f"Set positions for {len(object_positions)} objects")

    def allocate_tasks(
        self,
        objects: List[str],
        strategy: TaskAllocationStrategy = TaskAllocationStrategy.NEAREST_NEIGHBOR,
    ) -> Dict[str, List[str]]:
        """Allocate objects to agents.

        Args:
            objects: List of object IDs to allocate
            strategy: Task allocation strategy

        Returns:
            Dict mapping agent_id -> list of assigned object IDs
        """
        self.log(f"Allocating {len(objects)} objects using {strategy.value} strategy")

        if strategy == TaskAllocationStrategy.MANUAL:
            # Use pre-assigned objects
            allocation = {
                agent_id: agent.assigned_objects
                for agent_id, agent in self.agents.items()
            }

        elif strategy == TaskAllocationStrategy.NEAREST_NEIGHBOR:
            allocation = self._allocate_nearest_neighbor(objects)

        elif strategy == TaskAllocationStrategy.OPTIMAL_ASSIGNMENT:
            allocation = self._allocate_optimal(objects)

        elif strategy == TaskAllocationStrategy.SEQUENTIAL:
            allocation = self._allocate_sequential(objects)

        elif strategy == TaskAllocationStrategy.SYNCHRONIZED:
            # All agents work on same object
            allocation = {
                agent_id: objects.copy()
                for agent_id in self.agents.keys()
            }

        else:
            raise ValueError(f"Unknown allocation strategy: {strategy}")

        # Update agent configurations
        for agent_id, assigned_objs in allocation.items():
            if agent_id in self.agents:
                self.agents[agent_id].assigned_objects = assigned_objs

        # Log allocation
        for agent_id, assigned_objs in allocation.items():
            self.log(f"Agent {agent_id}: {len(assigned_objs)} objects assigned")

        return allocation

    def _allocate_nearest_neighbor(self, objects: List[str]) -> Dict[str, List[str]]:
        """Allocate objects to nearest agent."""
        allocation = {agent_id: [] for agent_id in self.agents.keys()}

        for obj_id in objects:
            if obj_id not in self.object_positions:
                # Skip objects without known position
                continue

            obj_pos = np.array(self.object_positions[obj_id])

            # Find nearest agent
            min_dist = float('inf')
            nearest_agent = None

            for agent_id, agent in self.agents.items():
                agent_pos = np.array(agent.spawn_position)
                dist = np.linalg.norm(obj_pos - agent_pos)

                if dist < min_dist:
                    min_dist = dist
                    nearest_agent = agent_id

            if nearest_agent:
                allocation[nearest_agent].append(obj_id)

        return allocation

    def _allocate_optimal(self, objects: List[str]) -> Dict[str, List[str]]:
        """Optimal task allocation using Hungarian algorithm.

        Note: Simplified implementation. For production, use scipy.optimize.linear_sum_assignment
        """
        # Build cost matrix (distance from each agent to each object)
        agents_list = list(self.agents.keys())
        n_agents = len(agents_list)
        n_objects = len(objects)

        cost_matrix = np.zeros((n_agents, n_objects))

        for i, agent_id in enumerate(agents_list):
            agent_pos = np.array(self.agents[agent_id].spawn_position)

            for j, obj_id in enumerate(objects):
                if obj_id in self.object_positions:
                    obj_pos = np.array(self.object_positions[obj_id])
                    cost_matrix[i, j] = np.linalg.norm(agent_pos - obj_pos)
                else:
                    cost_matrix[i, j] = 1000.0  # High cost for unknown position

        # Simplified allocation: greedy assignment
        # For production, use Hungarian algorithm from scipy
        allocation = {agent_id: [] for agent_id in agents_list}

        assigned_objects = set()
        for _ in range(min(n_agents, n_objects)):
            # Find minimum cost assignment
            min_cost = float('inf')
            best_agent_idx = None
            best_obj_idx = None

            for i in range(n_agents):
                for j in range(n_objects):
                    if objects[j] not in assigned_objects:
                        if cost_matrix[i, j] < min_cost:
                            min_cost = cost_matrix[i, j]
                            best_agent_idx = i
                            best_obj_idx = j

            if best_agent_idx is not None and best_obj_idx is not None:
                agent_id = agents_list[best_agent_idx]
                obj_id = objects[best_obj_idx]
                allocation[agent_id].append(obj_id)
                assigned_objects.add(obj_id)

                # Mark as assigned
                cost_matrix[best_agent_idx, :] += 1000.0

        # Assign remaining objects to least loaded agent
        for obj_id in objects:
            if obj_id not in assigned_objects:
                # Find agent with fewest objects
                min_load = min(len(objs) for objs in allocation.values())
                for agent_id, objs in allocation.items():
                    if len(objs) == min_load:
                        objs.append(obj_id)
                        break

        return allocation

    def _allocate_sequential(self, objects: List[str]) -> Dict[str, List[str]]:
        """Allocate objects sequentially (round-robin)."""
        allocation = {agent_id: [] for agent_id in self.agents.keys()}

        agents_list = list(self.agents.keys())
        for i, obj_id in enumerate(objects):
            agent_id = agents_list[i % len(agents_list)]
            allocation[agent_id].append(obj_id)

        return allocation

    def detect_workspace_conflicts(self) -> List[Dict[str, Any]]:
        """Detect workspace conflicts between agents.

        Returns:
            List of conflicts, each with agent IDs and overlap info
        """
        conflicts = []

        agents_list = list(self.agents.values())

        for i, agent1 in enumerate(agents_list):
            for agent2 in agents_list[i+1:]:
                # Check if workspaces overlap
                overlap = self._check_workspace_overlap(agent1, agent2)

                if overlap:
                    conflicts.append({
                        "agent1": agent1.agent_id,
                        "agent2": agent2.agent_id,
                        "overlap_volume": overlap,
                        "severity": "high" if overlap > 0.5 else "medium",
                    })

        if conflicts:
            self.log(f"Detected {len(conflicts)} workspace conflicts")

        return conflicts

    def _check_workspace_overlap(
        self,
        agent1: AgentConfiguration,
        agent2: AgentConfiguration,
    ) -> float:
        """Check workspace overlap between two agents.

        Returns:
            Overlap volume (0 = no overlap)
        """
        # Compute workspace AABBs
        pos1 = np.array(agent1.spawn_position)
        bounds1 = np.array(agent1.workspace_bounds)
        min1 = pos1 - bounds1 / 2
        max1 = pos1 + bounds1 / 2

        pos2 = np.array(agent2.spawn_position)
        bounds2 = np.array(agent2.workspace_bounds)
        min2 = pos2 - bounds2 / 2
        max2 = pos2 + bounds2 / 2

        # Compute overlap
        overlap_min = np.maximum(min1, min2)
        overlap_max = np.minimum(max1, max2)

        overlap_size = overlap_max - overlap_min
        overlap_size = np.maximum(overlap_size, 0)  # Clamp to non-negative

        overlap_volume = np.prod(overlap_size)

        return float(overlap_volume)

    def generate_collision_avoidance_config(self) -> Dict[str, Any]:
        """Generate collision avoidance configuration for simulation.

        Returns:
            Configuration dict for Isaac Sim or other simulator
        """
        config = {
            "strategy": self.collision_avoidance.value,
            "min_separation_distance": self.min_separation_distance,
            "agents": {},
        }

        for agent_id, agent in self.agents.items():
            config["agents"][agent_id] = {
                "bounding_sphere_radius": agent.max_reach,
                "priority": agent.priority,
            }

        if self.collision_avoidance == CollisionAvoidanceStrategy.SIMPLE_SPHERE:
            config["check_frequency_hz"] = 30.0

        elif self.collision_avoidance == CollisionAvoidanceStrategy.VELOCITY_OBSTACLES:
            config["prediction_horizon_seconds"] = 2.0
            config["safety_margin_meters"] = 0.1

        return config

    def suggest_spawn_positions(
        self,
        n_agents: int,
        workspace_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        workspace_radius: float = 1.5,
    ) -> List[Tuple[float, float, float]]:
        """Suggest spawn positions for N agents around workspace.

        Args:
            n_agents: Number of agents
            workspace_center: Center of shared workspace
            workspace_radius: Radius around workspace

        Returns:
            List of suggested spawn positions
        """
        positions = []

        center = np.array(workspace_center)

        for i in range(n_agents):
            # Distribute agents evenly around workspace
            angle = 2 * math.pi * i / n_agents

            # Position on circle around workspace
            x = center[0] + workspace_radius * math.cos(angle)
            y = center[1] + workspace_radius * math.sin(angle)
            z = center[2]

            positions.append((x, y, z))

        self.log(f"Suggested {n_agents} spawn positions around workspace")

        return positions

    def export_multi_agent_config(self) -> Dict[str, Any]:
        """Export complete multi-agent configuration.

        Returns:
            Configuration dict for pipeline
        """
        return {
            "num_agents": len(self.agents),
            "agents": [agent.to_dict() for agent in self.agents.values()],
            "collision_avoidance": self.generate_collision_avoidance_config(),
            "task_allocation": {
                agent_id: agent.assigned_objects
                for agent_id, agent in self.agents.items()
            },
        }
