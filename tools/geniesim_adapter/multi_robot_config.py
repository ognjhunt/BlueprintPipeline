"""
Multi-Robot Configuration for Genie Sim 3.0 Integration.

Enables data generation across multiple robot embodiments by default,
not just single robot types. This is key for selling versatile training data.

Supported Robot Types:
- Humanoid: g2 (AGIBOT), gr1 (Fourier), figure (Figure AI)
- Arm: franka, ur10, ur5e, kuka_iiwa
- Mobile: fetch, tiago, spot

By default, generates data for MULTIPLE robot types per scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

ROBOT_ASSETS_DIR = Path(__file__).resolve().parent / "robot_assets"


class RobotCategory(str, Enum):
    """Categories of robot embodiments."""
    HUMANOID = "humanoid"
    ARM = "arm"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    DUAL_ARM = "dual_arm"


class RobotType(str, Enum):
    """Supported robot types."""
    # Humanoids
    G2 = "g2"                    # AGIBOT G2 (Genie Sim native)
    GR1 = "gr1"                  # Fourier GR1
    FIGURE_01 = "figure_01"     # Figure AI
    H1 = "h1"                   # Unitree H1

    # Arms
    FRANKA = "franka"           # Franka Emika Panda
    UR10 = "ur10"               # Universal Robots UR10
    UR5E = "ur5e"               # Universal Robots UR5e
    KUKA_IIWA = "kuka_iiwa"     # KUKA LBR iiwa

    # Mobile Manipulators
    FETCH = "fetch"             # Fetch Robotics
    TIAGO = "tiago"             # PAL Robotics TIAGo
    SPOT = "spot"               # Boston Dynamics Spot + Arm

    # Dual Arm
    YUMI = "yumi"               # ABB YuMi
    BAXTER = "baxter"           # Rethink Robotics Baxter


@dataclass
class RobotSpec:
    """Specification for a robot embodiment."""
    robot_type: RobotType
    category: RobotCategory

    # Kinematics
    num_joints: int
    action_dim: int  # Typically num_joints + gripper

    # End effector
    gripper_type: str  # parallel_jaw, dexterous, suction
    max_gripper_aperture: float  # meters

    # Workspace
    reach_radius: float  # meters
    base_height: float  # height of base from floor

    # Data format compatibility
    supported_formats: List[str] = field(default_factory=lambda: ["lerobot"])

    # URDF/USD paths (relative to robot assets)
    urdf_path: Optional[str] = None
    usd_path: Optional[str] = None

    # Genie Sim compatibility
    geniesim_native: bool = False  # True for G2
    requires_urdf_import: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_type": self.robot_type.value,
            "category": self.category.value,
            "num_joints": self.num_joints,
            "action_dim": self.action_dim,
            "gripper_type": self.gripper_type,
            "max_gripper_aperture": self.max_gripper_aperture,
            "reach_radius": self.reach_radius,
            "base_height": self.base_height,
            "geniesim_native": self.geniesim_native,
            "urdf_path": self.urdf_path,
            "usd_path": self.usd_path,
            "requires_urdf_import": self.requires_urdf_import,
        }


# Robot specifications database
ROBOT_SPECS: Dict[RobotType, RobotSpec] = {
    # Humanoids
    RobotType.G2: RobotSpec(
        robot_type=RobotType.G2,
        category=RobotCategory.HUMANOID,
        num_joints=32,  # Full body
        action_dim=33,
        gripper_type="dexterous",
        max_gripper_aperture=0.12,
        reach_radius=0.9,
        base_height=0.0,  # Legs
        geniesim_native=True,
        requires_urdf_import=False,
    ),
    RobotType.GR1: RobotSpec(
        robot_type=RobotType.GR1,
        category=RobotCategory.HUMANOID,
        num_joints=30,
        action_dim=31,
        gripper_type="dexterous",
        max_gripper_aperture=0.10,
        reach_radius=0.85,
        base_height=0.0,
        urdf_path="robots/gr1/gr1.urdf",
        usd_path="robots/gr1/gr1.usd",
    ),
    RobotType.FIGURE_01: RobotSpec(
        robot_type=RobotType.FIGURE_01,
        category=RobotCategory.HUMANOID,
        num_joints=28,
        action_dim=29,
        gripper_type="dexterous",
        max_gripper_aperture=0.10,
        reach_radius=0.9,
        base_height=0.0,
        urdf_path="robots/figure_01/figure_01.urdf",
        usd_path="robots/figure_01/figure_01.usd",
    ),
    RobotType.H1: RobotSpec(
        robot_type=RobotType.H1,
        category=RobotCategory.HUMANOID,
        num_joints=27,
        action_dim=28,
        gripper_type="dexterous",
        max_gripper_aperture=0.12,
        reach_radius=0.9,
        base_height=0.0,
        urdf_path="robots/h1/h1.urdf",
        usd_path="robots/h1/h1.usd",
    ),

    # Arms
    RobotType.FRANKA: RobotSpec(
        robot_type=RobotType.FRANKA,
        category=RobotCategory.ARM,
        num_joints=7,
        action_dim=8,  # 7 joints + 1 gripper
        gripper_type="parallel_jaw",
        max_gripper_aperture=0.08,
        reach_radius=0.855,
        base_height=0.0,
        urdf_path="robots/franka/panda.urdf",
    ),
    RobotType.UR10: RobotSpec(
        robot_type=RobotType.UR10,
        category=RobotCategory.ARM,
        num_joints=6,
        action_dim=7,
        gripper_type="parallel_jaw",
        max_gripper_aperture=0.085,
        reach_radius=1.3,
        base_height=0.0,
        urdf_path="robots/ur10/ur10.urdf",
    ),
    RobotType.UR5E: RobotSpec(
        robot_type=RobotType.UR5E,
        category=RobotCategory.ARM,
        num_joints=6,
        action_dim=7,
        gripper_type="parallel_jaw",
        max_gripper_aperture=0.085,
        reach_radius=0.85,
        base_height=0.0,
        urdf_path="robots/ur5e/ur5e.urdf",
    ),
    RobotType.KUKA_IIWA: RobotSpec(
        robot_type=RobotType.KUKA_IIWA,
        category=RobotCategory.ARM,
        num_joints=7,
        action_dim=8,
        gripper_type="parallel_jaw",
        max_gripper_aperture=0.08,
        reach_radius=0.8,
        base_height=0.0,
        urdf_path="robots/kuka_iiwa/iiwa14.urdf",
    ),

    # Mobile Manipulators
    RobotType.FETCH: RobotSpec(
        robot_type=RobotType.FETCH,
        category=RobotCategory.MOBILE_MANIPULATOR,
        num_joints=8,  # 7 arm + torso lift
        action_dim=10,  # + base (2) + gripper
        gripper_type="parallel_jaw",
        max_gripper_aperture=0.10,
        reach_radius=1.1,
        base_height=0.38,
        urdf_path="robots/fetch/fetch.urdf",
    ),
    RobotType.TIAGO: RobotSpec(
        robot_type=RobotType.TIAGO,
        category=RobotCategory.MOBILE_MANIPULATOR,
        num_joints=7,
        action_dim=10,
        gripper_type="parallel_jaw",
        max_gripper_aperture=0.08,
        reach_radius=0.9,
        base_height=0.35,
        urdf_path="robots/tiago/tiago.urdf",
    ),
    RobotType.SPOT: RobotSpec(
        robot_type=RobotType.SPOT,
        category=RobotCategory.MOBILE_MANIPULATOR,
        num_joints=6,  # Arm only
        action_dim=8,  # + gripper + base pose
        gripper_type="parallel_jaw",
        max_gripper_aperture=0.10,
        reach_radius=0.95,
        base_height=0.54,
        urdf_path="robots/spot/spot_arm.urdf",
    ),

    # Dual Arm
    RobotType.YUMI: RobotSpec(
        robot_type=RobotType.YUMI,
        category=RobotCategory.DUAL_ARM,
        num_joints=14,  # 7 per arm
        action_dim=16,  # + 2 grippers
        gripper_type="parallel_jaw",
        max_gripper_aperture=0.05,
        reach_radius=0.56,
        base_height=0.0,
        urdf_path="robots/yumi/yumi.urdf",
    ),
    RobotType.BAXTER: RobotSpec(
        robot_type=RobotType.BAXTER,
        category=RobotCategory.DUAL_ARM,
        num_joints=14,
        action_dim=16,
        gripper_type="parallel_jaw",
        max_gripper_aperture=0.08,
        reach_radius=1.2,
        base_height=0.9,
        urdf_path="robots/baxter/baxter.urdf",
    ),
}


@dataclass
class MultiRobotConfig:
    """Configuration for multi-robot data generation.

    By default, generates data for multiple robot types per scene.
    """

    # Primary robots to generate data for
    primary_robots: List[RobotType] = field(default_factory=lambda: [
        RobotType.FRANKA,   # Most common arm
        RobotType.G2,       # Genie Sim native humanoid
    ])

    # Additional robots (if enabled)
    secondary_robots: List[RobotType] = field(default_factory=lambda: [
        RobotType.UR10,
        RobotType.GR1,
        RobotType.FETCH,
    ])

    # Whether to generate for all robots or just primary
    generate_all: bool = False

    # Bimanual tasks
    enable_bimanual: bool = True
    bimanual_robots: List[RobotType] = field(default_factory=lambda: [
        RobotType.G2,      # Humanoid with two arms
        RobotType.H1,      # Humanoid with two arms
        RobotType.YUMI,    # Dedicated dual-arm
    ])

    # Multi-robot coordination
    enable_multi_robot_coordination: bool = True
    coordination_pairs: List[tuple] = field(default_factory=lambda: [
        (RobotType.FRANKA, RobotType.UR10),   # Arm-to-arm handoff
        (RobotType.FRANKA, RobotType.FETCH),  # Arm-to-mobile handoff
    ])

    def get_all_robots(self) -> List[RobotType]:
        """Get all robots to generate data for."""
        if self.generate_all:
            return list(ROBOT_SPECS.keys())
        return list(set(self.primary_robots + self.secondary_robots))

    def get_robots_by_category(self, category: RobotCategory) -> List[RobotType]:
        """Get robots in a specific category."""
        return [
            robot for robot in self.get_all_robots()
            if ROBOT_SPECS[robot].category == category
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_robots": [r.value for r in self.primary_robots],
            "secondary_robots": [r.value for r in self.secondary_robots],
            "generate_all": self.generate_all,
            "enable_bimanual": self.enable_bimanual,
            "bimanual_robots": [r.value for r in self.bimanual_robots],
            "enable_multi_robot_coordination": self.enable_multi_robot_coordination,
            "coordination_pairs": [
                (r1.value, r2.value) for r1, r2 in self.coordination_pairs
            ],
            "all_robots": [r.value for r in self.get_all_robots()],
            "robot_specs": {
                r.value: ROBOT_SPECS[r].to_dict()
                for r in self.get_all_robots()
            },
        }


# Default configuration: Multi-robot enabled by default
DEFAULT_MULTI_ROBOT_CONFIG = MultiRobotConfig(
    primary_robots=[
        RobotType.FRANKA,   # Industry standard arm
        RobotType.G2,       # Genie Sim native humanoid
    ],
    secondary_robots=[
        RobotType.UR10,     # Popular industrial arm
        RobotType.GR1,      # Alternative humanoid
        RobotType.H1,       # Unitree humanoid
    ],
    generate_all=False,
    enable_bimanual=True,
    enable_multi_robot_coordination=True,
)


# Full robot coverage configuration (for premium tiers)
FULL_ROBOT_CONFIG = MultiRobotConfig(
    primary_robots=list(ROBOT_SPECS.keys()),
    secondary_robots=[],
    generate_all=True,
    enable_bimanual=True,
    enable_multi_robot_coordination=True,
)


def get_robot_spec(robot_type: str | RobotType) -> RobotSpec:
    """Get specification for a robot type."""
    if isinstance(robot_type, str):
        robot_type = RobotType(robot_type)
    return ROBOT_SPECS[robot_type]


def resolve_robot_asset_path(asset_path: Optional[str]) -> Optional[Path]:
    """Resolve a robot asset path relative to the robot asset root."""
    if not asset_path:
        return None
    return (ROBOT_ASSETS_DIR / asset_path).resolve()


def validate_robot_assets(spec: RobotSpec) -> None:
    """Validate that robot assets referenced by the spec exist on disk."""
    if spec.requires_urdf_import and not spec.urdf_path:
        raise ValueError(
            f"URDF import required but no urdf_path is set for {spec.robot_type.value}."
        )
    for asset_path in (spec.urdf_path, spec.usd_path):
        resolved = resolve_robot_asset_path(asset_path)
        if resolved and not resolved.is_file():
            raise FileNotFoundError(
                f"Robot asset missing for {spec.robot_type.value}: {resolved}"
            )


def get_geniesim_robot_config(
    robot_type: RobotType,
    base_position: tuple = (0.0, 0.0, 0.0),
) -> Dict[str, Any]:
    """Get Genie Sim compatible robot configuration."""
    spec = ROBOT_SPECS[robot_type]
    validate_robot_assets(spec)

    return {
        "robot_type": robot_type.value,
        "category": spec.category.value,
        "base_position": list(base_position),
        "workspace_bounds": [
            [-spec.reach_radius, -spec.reach_radius, 0.0],
            [spec.reach_radius, spec.reach_radius, spec.reach_radius * 1.5],
        ],
        "action_dim": spec.action_dim,
        "gripper_type": spec.gripper_type,
        "urdf_import_required": spec.requires_urdf_import,
        "urdf_path": spec.urdf_path,
        "usd_path": spec.usd_path,
        "resolved_urdf_path": (
            str(resolve_robot_asset_path(spec.urdf_path))
            if spec.urdf_path
            else None
        ),
        "resolved_usd_path": (
            str(resolve_robot_asset_path(spec.usd_path))
            if spec.usd_path
            else None
        ),
        "asset_root": str(ROBOT_ASSETS_DIR),
        "geniesim_native": spec.geniesim_native,
    }


def save_multi_robot_config(
    config: MultiRobotConfig,
    output_path: Path,
) -> None:
    """Save multi-robot configuration to JSON."""
    with open(output_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
