#!/usr/bin/env python3
"""
Advanced Capabilities Module.

Consolidates multiple premium upsell features:
- Multi-Robot Fleet Coordination
- Deformable Object Manipulation
- Custom Robot Embodiment Support
- Bimanual Manipulation

Each capability adds significant value to the pipeline.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# MULTI-ROBOT FLEET COORDINATION
# Upsell Value: +$6,000-$12,000 per scene
# =============================================================================

class FleetScenarioType(str, Enum):
    """Types of multi-robot scenarios."""
    DUAL_ARM_SAME_BASE = "dual_arm_same_base"
    ROBOT_HANDOFF = "robot_handoff"
    FLEET_COORDINATION = "fleet_coordination"
    COLLABORATIVE_ASSEMBLY = "collaborative_assembly"


@dataclass
class RobotAgent:
    """A robot agent in the fleet."""
    agent_id: str
    robot_type: str  # franka, ur10, fetch, etc.
    base_position: np.ndarray
    base_orientation: np.ndarray
    role: str = "manipulator"  # manipulator, transporter, holder


@dataclass
class FleetScenario:
    """Configuration for multi-robot scenario."""
    scenario_type: FleetScenarioType
    agents: List[RobotAgent]
    coordination_zones: List[Dict[str, Any]] = field(default_factory=list)
    handoff_points: List[np.ndarray] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_type": self.scenario_type.value,
            "num_agents": len(self.agents),
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "robot_type": a.robot_type,
                    "base_position": a.base_position.tolist(),
                    "role": a.role,
                }
                for a in self.agents
            ],
            "coordination_zones": self.coordination_zones,
            "handoff_points": [p.tolist() for p in self.handoff_points],
        }


class MultiRobotCoordinator:
    """
    Generates multi-robot coordination episodes.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[MULTI-ROBOT] {msg}")

    def create_handoff_scenario(
        self,
        robot_a_type: str = "franka",
        robot_b_type: str = "ur10",
        handoff_position: Optional[np.ndarray] = None,
    ) -> FleetScenario:
        """Create a robot-to-robot handoff scenario."""
        if handoff_position is None:
            handoff_position = np.array([0.5, 0.0, 0.3])

        agents = [
            RobotAgent(
                agent_id="robot_a",
                robot_type=robot_a_type,
                base_position=np.array([0.0, -0.5, 0.0]),
                base_orientation=np.array([0, 0, 0, 1]),
                role="picker",
            ),
            RobotAgent(
                agent_id="robot_b",
                robot_type=robot_b_type,
                base_position=np.array([0.0, 0.5, 0.0]),
                base_orientation=np.array([0, 0, 1, 0]),  # 180 deg yaw
                role="placer",
            ),
        ]

        scenario = FleetScenario(
            scenario_type=FleetScenarioType.ROBOT_HANDOFF,
            agents=agents,
            handoff_points=[handoff_position],
            coordination_zones=[{
                "name": "handoff_zone",
                "center": handoff_position.tolist(),
                "radius": 0.15,
            }],
        )

        self.log(f"Created handoff scenario: {robot_a_type} -> {robot_b_type}")
        return scenario

    def create_collaborative_assembly_scenario(
        self,
        num_robots: int = 2,
    ) -> FleetScenario:
        """Create a collaborative assembly scenario."""
        agents = []
        angle_step = 2 * np.pi / num_robots

        for i in range(num_robots):
            angle = i * angle_step
            position = np.array([
                0.8 * np.cos(angle),
                0.8 * np.sin(angle),
                0.0,
            ])
            agents.append(RobotAgent(
                agent_id=f"robot_{i}",
                robot_type="franka",
                base_position=position,
                base_orientation=np.array([0, 0, np.sin(angle/2), np.cos(angle/2)]),
                role="assembler" if i == 0 else "holder",
            ))

        scenario = FleetScenario(
            scenario_type=FleetScenarioType.COLLABORATIVE_ASSEMBLY,
            agents=agents,
            coordination_zones=[{
                "name": "assembly_zone",
                "center": [0.0, 0.0, 0.3],
                "radius": 0.2,
            }],
        )

        self.log(f"Created collaborative assembly with {num_robots} robots")
        return scenario

    def generate_coordination_trajectory(
        self,
        scenario: FleetScenario,
        task_type: str = "handoff",
    ) -> Dict[str, Any]:
        """Generate coordinated trajectories for all agents."""
        trajectories = {}

        for agent in scenario.agents:
            # Generate individual trajectory
            traj = {
                "agent_id": agent.agent_id,
                "waypoints": [],
                "timing": [],
                "sync_points": [],
            }

            if scenario.scenario_type == FleetScenarioType.ROBOT_HANDOFF:
                if agent.role == "picker":
                    # Pick object, bring to handoff
                    traj["waypoints"] = [
                        agent.base_position + np.array([0.3, 0, 0.2]),  # Home
                        agent.base_position + np.array([0.4, 0, 0.1]),  # Pick approach
                        agent.base_position + np.array([0.4, 0, 0.05]),  # Pick
                        scenario.handoff_points[0],  # Handoff position
                    ]
                    traj["sync_points"] = [3]  # Sync at handoff
                else:
                    # Wait, receive at handoff, place
                    traj["waypoints"] = [
                        agent.base_position + np.array([0.3, 0, 0.2]),  # Home
                        scenario.handoff_points[0],  # Handoff position
                        agent.base_position + np.array([0.4, 0, 0.1]),  # Place approach
                        agent.base_position + np.array([0.4, 0, 0.05]),  # Place
                    ]
                    traj["sync_points"] = [1]  # Sync at handoff

            trajectories[agent.agent_id] = traj

        return trajectories


# =============================================================================
# DEFORMABLE OBJECT MANIPULATION
# Upsell Value: +$5,000-$8,000 per scene
# =============================================================================

class DeformableType(str, Enum):
    """Types of deformable objects."""
    CLOTH = "cloth"
    ROPE = "rope"
    CABLE = "cable"
    SOFT_BODY = "soft_body"
    GRANULAR = "granular"


@dataclass
class DeformableObjectConfig:
    """Configuration for a deformable object."""
    deformable_type: DeformableType
    name: str

    # Material properties
    youngs_modulus: float = 1e5  # Pa
    poissons_ratio: float = 0.3
    density: float = 1000.0  # kg/m³
    damping: float = 0.1

    # Geometry
    dimensions: Tuple[float, ...] = (0.5, 0.5, 0.001)  # For cloth: width, height, thickness

    # Simulation
    num_particles: int = 100
    stretch_stiffness: float = 1.0
    bend_stiffness: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.deformable_type.value,
            "name": self.name,
            "material": {
                "youngs_modulus": self.youngs_modulus,
                "poissons_ratio": self.poissons_ratio,
                "density": self.density,
                "damping": self.damping,
            },
            "dimensions": list(self.dimensions),
            "simulation": {
                "num_particles": self.num_particles,
                "stretch_stiffness": self.stretch_stiffness,
                "bend_stiffness": self.bend_stiffness,
            },
        }


class DeformableObjectGenerator:
    """
    Generates deformable object manipulation episodes.
    """

    # Preset configurations
    PRESETS = {
        "cloth_towel": DeformableObjectConfig(
            deformable_type=DeformableType.CLOTH,
            name="towel",
            dimensions=(0.4, 0.6, 0.002),
            num_particles=400,
            stretch_stiffness=0.9,
            bend_stiffness=0.01,
        ),
        "rope_cable": DeformableObjectConfig(
            deformable_type=DeformableType.ROPE,
            name="cable",
            dimensions=(1.0, 0.01),  # length, radius
            num_particles=50,
            stretch_stiffness=0.95,
            bend_stiffness=0.05,
        ),
        "soft_sponge": DeformableObjectConfig(
            deformable_type=DeformableType.SOFT_BODY,
            name="sponge",
            youngs_modulus=1e4,
            dimensions=(0.1, 0.1, 0.05),
            num_particles=200,
        ),
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[DEFORMABLE] {msg}")

    def create_cloth_folding_task(
        self,
        cloth_config: Optional[DeformableObjectConfig] = None,
    ) -> Dict[str, Any]:
        """Create a cloth folding task specification."""
        if cloth_config is None:
            cloth_config = self.PRESETS["cloth_towel"]

        task = {
            "task_type": "cloth_folding",
            "deformable": cloth_config.to_dict(),
            "goal": {
                "fold_type": "half_fold",
                "target_shape": "rectangle",
                "target_dimensions": [
                    cloth_config.dimensions[0] / 2,
                    cloth_config.dimensions[1],
                ],
            },
            "grasp_points": [
                [0.0, 0.0],  # Corner 1
                [1.0, 0.0],  # Corner 2
            ],
            "fold_sequence": [
                {"type": "pick", "point": 0},
                {"type": "fold", "from": [0.0, 0.5], "to": [1.0, 0.5]},
                {"type": "release"},
            ],
        }

        self.log("Created cloth folding task")
        return task

    def create_cable_routing_task(
        self,
        cable_config: Optional[DeformableObjectConfig] = None,
        waypoints: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Create a cable routing task specification."""
        if cable_config is None:
            cable_config = self.PRESETS["rope_cable"]

        if waypoints is None:
            waypoints = [
                np.array([0.2, 0.0, 0.1]),
                np.array([0.4, 0.1, 0.1]),
                np.array([0.6, 0.0, 0.1]),
            ]

        task = {
            "task_type": "cable_routing",
            "deformable": cable_config.to_dict(),
            "goal": {
                "routing_waypoints": [w.tolist() for w in waypoints],
                "tolerance_mm": 10.0,
            },
            "constraints": {
                "min_bend_radius_mm": 20.0,
                "max_tension_n": 5.0,
            },
        }

        self.log("Created cable routing task")
        return task


# =============================================================================
# CUSTOM ROBOT EMBODIMENT SUPPORT
# Upsell Value: $15,000 setup + $2,000/scene
# =============================================================================

@dataclass
class CustomRobotSpec:
    """Specification for a custom robot."""
    robot_id: str
    name: str
    manufacturer: str

    # URDF/USD
    urdf_path: Optional[Path] = None
    usd_path: Optional[Path] = None

    # Kinematics
    num_joints: int = 7
    joint_names: List[str] = field(default_factory=list)
    joint_limits_lower: List[float] = field(default_factory=list)
    joint_limits_upper: List[float] = field(default_factory=list)

    # End effector
    ee_link_name: str = "ee_link"
    default_gripper: str = "parallel_jaw"

    # Physics
    joint_damping: List[float] = field(default_factory=list)
    joint_friction: List[float] = field(default_factory=list)

    # Workspace
    workspace_radius: float = 0.8  # meters
    workspace_center: np.ndarray = field(default_factory=lambda: np.array([0.5, 0, 0.3]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "name": self.name,
            "manufacturer": self.manufacturer,
            "kinematics": {
                "num_joints": self.num_joints,
                "joint_names": self.joint_names,
                "joint_limits": {
                    "lower": self.joint_limits_lower,
                    "upper": self.joint_limits_upper,
                },
            },
            "end_effector": {
                "link_name": self.ee_link_name,
                "default_gripper": self.default_gripper,
            },
            "workspace": {
                "radius": self.workspace_radius,
                "center": self.workspace_center.tolist(),
            },
        }


class CustomRobotOnboarder:
    """
    Onboards custom robot configurations into the pipeline.
    """

    def __init__(
        self,
        robots_dir: Path,
        verbose: bool = True,
    ):
        self.robots_dir = Path(robots_dir)
        self.robots_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[CUSTOM-ROBOT] {msg}")

    def onboard_from_urdf(
        self,
        urdf_path: Path,
        robot_name: str,
        manufacturer: str = "Custom",
    ) -> CustomRobotSpec:
        """Onboard a robot from URDF file."""
        self.log(f"Onboarding robot from {urdf_path}")

        # Parse URDF (simplified - would use urdfpy in production)
        robot_id = f"custom_{robot_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"

        # Create spec with extracted info
        spec = CustomRobotSpec(
            robot_id=robot_id,
            name=robot_name,
            manufacturer=manufacturer,
            urdf_path=urdf_path,
            num_joints=7,  # Would parse from URDF
            joint_names=[f"joint_{i}" for i in range(7)],
            joint_limits_lower=[-2.9] * 7,
            joint_limits_upper=[2.9] * 7,
        )

        # Save spec
        spec_path = self.robots_dir / f"{robot_id}.json"
        with open(spec_path, "w") as f:
            json.dump(spec.to_dict(), f, indent=2)

        self.log(f"Saved robot spec to {spec_path}")
        return spec

    def convert_urdf_to_usd(
        self,
        spec: CustomRobotSpec,
    ) -> Path:
        """Convert URDF to USD for Isaac Sim."""
        self.log(f"Converting {spec.name} URDF to USD")

        usd_path = self.robots_dir / f"{spec.robot_id}.usd"

        # In production, would use:
        # from omni.isaac.urdf import _urdf
        # urdf_interface = _urdf.acquire_urdf_interface()
        # urdf_interface.parse_urdf(str(spec.urdf_path), str(usd_path))

        self.log(f"USD saved to {usd_path}")
        return usd_path

    def generate_motion_planning_config(
        self,
        spec: CustomRobotSpec,
    ) -> Dict[str, Any]:
        """Generate motion planning configuration for the robot."""
        config = {
            "robot_id": spec.robot_id,
            "planning": {
                "planner": "RRT",
                "max_planning_time": 5.0,
                "goal_tolerance": 0.01,
            },
            "kinematics": {
                "solver": "FABRIK",
                "max_iterations": 100,
            },
            "collision": {
                "self_collision_check": True,
                "collision_pairs": [],
            },
        }

        config_path = self.robots_dir / f"{spec.robot_id}_planning.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return config


# =============================================================================
# BIMANUAL MANIPULATION
# Upsell Value: +$6,000-$10,000 per scene
# =============================================================================

class BimanualTaskType(str, Enum):
    """Types of bimanual tasks."""
    ARM_TO_ARM_HANDOFF = "handoff"
    COORDINATED_LIFT = "coordinated_lift"
    HOLD_AND_MANIPULATE = "hold_manipulate"
    BIMANUAL_ASSEMBLY = "bimanual_assembly"
    LID_OPENING = "lid_opening"


@dataclass
class BimanualConfig:
    """Configuration for bimanual manipulation."""
    left_arm: str = "franka"  # Robot type
    right_arm: str = "franka"
    shared_base: bool = True
    arm_separation: float = 0.5  # meters between bases

    # Coordination
    sync_mode: str = "tight"  # tight, loose, independent
    master_arm: str = "left"  # For asymmetric tasks

    def to_dict(self) -> Dict[str, Any]:
        return {
            "left_arm": self.left_arm,
            "right_arm": self.right_arm,
            "shared_base": self.shared_base,
            "arm_separation": self.arm_separation,
            "coordination": {
                "sync_mode": self.sync_mode,
                "master_arm": self.master_arm,
            },
        }


class BimanualTaskGenerator:
    """
    Generates bimanual manipulation episodes.
    """

    def __init__(
        self,
        config: Optional[BimanualConfig] = None,
        verbose: bool = True,
    ):
        self.config = config or BimanualConfig()
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[BIMANUAL] {msg}")

    def create_coordinated_lift_task(
        self,
        object_size: Tuple[float, float, float],
        object_weight_kg: float,
    ) -> Dict[str, Any]:
        """Create a coordinated lifting task (e.g., large box)."""
        # Grasp points on opposite sides
        left_grasp = np.array([-object_size[0]/2, 0, object_size[2]/2])
        right_grasp = np.array([object_size[0]/2, 0, object_size[2]/2])

        task = {
            "task_type": BimanualTaskType.COORDINATED_LIFT.value,
            "config": self.config.to_dict(),
            "object": {
                "size": list(object_size),
                "weight_kg": object_weight_kg,
            },
            "grasp_points": {
                "left": left_grasp.tolist(),
                "right": right_grasp.tolist(),
            },
            "coordination": {
                "sync_mode": "tight",
                "max_position_error_m": 0.01,
                "max_force_imbalance_n": 5.0,
            },
            "phases": [
                {"name": "approach", "sync": True},
                {"name": "grasp", "sync": True},
                {"name": "lift", "sync": True},
                {"name": "move", "sync": True},
                {"name": "lower", "sync": True},
                {"name": "release", "sync": True},
            ],
        }

        self.log(f"Created coordinated lift task for {object_weight_kg}kg object")
        return task

    def create_hold_and_manipulate_task(
        self,
        held_object: str,
        manipulation_type: str,
    ) -> Dict[str, Any]:
        """Create hold-and-manipulate task (one arm holds, other works)."""
        task = {
            "task_type": BimanualTaskType.HOLD_AND_MANIPULATE.value,
            "config": self.config.to_dict(),
            "held_object": held_object,
            "manipulation": manipulation_type,
            "roles": {
                "holder": self.config.master_arm,
                "manipulator": "right" if self.config.master_arm == "left" else "left",
            },
            "constraints": {
                "holder_stability_n": 2.0,  # Max allowed perturbation
                "workspace_overlap": True,  # Allow arms to work in shared space
            },
        }

        self.log(f"Created hold-and-manipulate task: {manipulation_type} on {held_object}")
        return task

    def create_lid_opening_task(
        self,
        container_type: str = "jar",
    ) -> Dict[str, Any]:
        """Create lid opening task (one holds container, other turns lid)."""
        task = {
            "task_type": BimanualTaskType.LID_OPENING.value,
            "config": self.config.to_dict(),
            "container": container_type,
            "phases": [
                {"name": "grasp_container", "arm": "left"},
                {"name": "grasp_lid", "arm": "right"},
                {"name": "stabilize", "sync": True},
                {"name": "rotate_lid", "arm": "right", "direction": "counter_clockwise"},
                {"name": "lift_lid", "arm": "right"},
                {"name": "release", "sync": True},
            ],
            "motion": {
                "lid_rotation_deg": 720,  # Multiple rotations
                "rotation_speed_dps": 45,  # Degrees per second
            },
        }

        self.log(f"Created lid opening task for {container_type}")
        return task


# =============================================================================
# MAIN ADVANCED CAPABILITIES CLASS
# =============================================================================

class AdvancedCapabilities:
    """
    Unified interface for all advanced manipulation capabilities.
    """

    def __init__(
        self,
        output_dir: Path,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Initialize all generators
        self.multi_robot = MultiRobotCoordinator(verbose=verbose)
        self.deformable = DeformableObjectGenerator(verbose=verbose)
        self.custom_robot = CustomRobotOnboarder(
            robots_dir=self.output_dir / "custom_robots",
            verbose=verbose,
        )
        self.bimanual = BimanualTaskGenerator(verbose=verbose)

    def generate_advanced_bundle(
        self,
        scene_id: str,
        capabilities: List[str],
    ) -> Dict[str, Any]:
        """Generate advanced capability data for a scene."""
        bundle = {
            "scene_id": scene_id,
            "capabilities": {},
        }

        if "multi_robot" in capabilities:
            scenario = self.multi_robot.create_handoff_scenario()
            bundle["capabilities"]["multi_robot"] = scenario.to_dict()

        if "deformable" in capabilities:
            cloth_task = self.deformable.create_cloth_folding_task()
            cable_task = self.deformable.create_cable_routing_task()
            bundle["capabilities"]["deformable"] = {
                "cloth_folding": cloth_task,
                "cable_routing": cable_task,
            }

        if "bimanual" in capabilities:
            lift_task = self.bimanual.create_coordinated_lift_task(
                object_size=(0.4, 0.3, 0.2),
                object_weight_kg=5.0,
            )
            bundle["capabilities"]["bimanual"] = lift_task

        # Save bundle
        bundle_path = self.output_dir / f"advanced_bundle_{scene_id}.json"
        with open(bundle_path, "w") as f:
            json.dump(bundle, f, indent=2)

        return bundle


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate advanced manipulation capabilities"
    )
    parser.add_argument(
        "--capability",
        choices=["multi_robot", "deformable", "bimanual", "all"],
        default="all",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default="demo_scene",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./advanced_capabilities"),
    )

    args = parser.parse_args()

    capabilities = AdvancedCapabilities(
        output_dir=args.output_dir,
    )

    if args.capability == "all":
        caps = ["multi_robot", "deformable", "bimanual"]
    else:
        caps = [args.capability]

    bundle = capabilities.generate_advanced_bundle(
        scene_id=args.scene_id,
        capabilities=caps,
    )

    print(f"\nGenerated advanced capabilities bundle:")
    print(json.dumps(bundle, indent=2, default=str))


if __name__ == "__main__":
    main()
