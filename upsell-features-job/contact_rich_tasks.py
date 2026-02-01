#!/usr/bin/env python3
"""
Contact-Rich Task Specialization.

Specialized episode generation for precision assembly and insertion tasks:
- Peg-in-hole insertion
- Snap-fit assembly
- Screw driving
- Cable insertion
- Precision placement

These tasks require:
- High-precision contact physics
- Force/torque feedback
- Compliance control
- Tighter tolerances

Upsell Value: 3x base price ($7,500-$15,000 per scene)
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ContactRichTaskType(str, Enum):
    """Types of contact-rich manipulation tasks."""
    PEG_IN_HOLE = "peg_in_hole"
    SNAP_FIT = "snap_fit"
    SCREW_DRIVING = "screw_driving"
    CABLE_INSERTION = "cable_insertion"
    PRECISION_PLACEMENT = "precision_placement"
    KEY_INSERTION = "key_insertion"
    USB_INSERTION = "usb_insertion"
    CONNECTOR_MATING = "connector_mating"


class ToleranceClass(str, Enum):
    """Assembly tolerance classes."""
    LOOSE = "loose"          # > 2mm clearance
    MEDIUM = "medium"        # 0.5-2mm clearance
    TIGHT = "tight"          # 0.1-0.5mm clearance
    PRECISION = "precision"  # < 0.1mm clearance


@dataclass
class ContactRichTaskSpec:
    """Specification for a contact-rich task."""
    task_type: ContactRichTaskType
    tolerance_class: ToleranceClass

    # Geometry
    peg_geometry: str = "cylinder"  # cylinder, rectangular, custom
    hole_geometry: str = "cylinder"
    clearance_mm: float = 0.5
    insertion_depth_mm: float = 20.0

    # Physics
    friction_coefficient: float = 0.3
    stiffness: float = 1000.0  # N/m
    damping: float = 100.0  # Ns/m

    # Control
    force_threshold_n: float = 10.0
    torque_threshold_nm: float = 1.0
    max_force_n: float = 50.0
    use_compliance: bool = True

    # Timing
    approach_speed_mms: float = 50.0
    insertion_speed_mms: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "tolerance_class": self.tolerance_class.value,
            "geometry": {
                "peg": self.peg_geometry,
                "hole": self.hole_geometry,
                "clearance_mm": self.clearance_mm,
                "insertion_depth_mm": self.insertion_depth_mm,
            },
            "physics": {
                "friction": self.friction_coefficient,
                "stiffness": self.stiffness,
                "damping": self.damping,
            },
            "control": {
                "force_threshold_n": self.force_threshold_n,
                "torque_threshold_nm": self.torque_threshold_nm,
                "max_force_n": self.max_force_n,
                "use_compliance": self.use_compliance,
            },
            "timing": {
                "approach_speed_mms": self.approach_speed_mms,
                "insertion_speed_mms": self.insertion_speed_mms,
            },
        }


@dataclass
class InsertionPhase:
    """A phase of the insertion process."""
    name: str
    start_time: float
    end_time: float
    phase_type: str  # approach, search, align, insert, verify
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    stiffness_gains: np.ndarray = field(default_factory=lambda: np.ones(6) * 1000)
    damping_gains: np.ndarray = field(default_factory=lambda: np.ones(6) * 100)


@dataclass
class ContactRichTrajectory:
    """Trajectory with contact-rich phases and force profiles."""
    task_spec: ContactRichTaskSpec
    phases: List[InsertionPhase]

    # Force/torque profile
    force_profile: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    torque_profile: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    # Contact information
    contact_times: List[float] = field(default_factory=list)
    contact_forces: List[np.ndarray] = field(default_factory=list)

    # Quality metrics
    insertion_success: bool = False
    max_force_applied: float = 0.0
    insertion_time: float = 0.0
    alignment_error_mm: float = 0.0


class ContactRichTaskGenerator:
    """
    Generates specialized contact-rich manipulation episodes.
    """

    # Physics profiles for different task types
    PHYSICS_PROFILES = {
        ContactRichTaskType.PEG_IN_HOLE: {
            "friction": 0.3,
            "stiffness": 1000,
            "damping": 100,
            "force_profile": "linear_ramp",
        },
        ContactRichTaskType.SNAP_FIT: {
            "friction": 0.4,
            "stiffness": 2000,
            "damping": 200,
            "force_profile": "snap_click",
        },
        ContactRichTaskType.SCREW_DRIVING: {
            "friction": 0.5,
            "stiffness": 500,
            "damping": 50,
            "force_profile": "spiral_torque",
        },
        ContactRichTaskType.CABLE_INSERTION: {
            "friction": 0.2,
            "stiffness": 100,
            "damping": 20,
            "force_profile": "compliant",
        },
        ContactRichTaskType.USB_INSERTION: {
            "friction": 0.35,
            "stiffness": 800,
            "damping": 80,
            "force_profile": "search_insert",
        },
    }

    # Insertion strategies
    INSERTION_STRATEGIES = {
        "direct": {
            "phases": ["approach", "insert", "verify"],
            "search_pattern": None,
        },
        "spiral_search": {
            "phases": ["approach", "search", "align", "insert", "verify"],
            "search_pattern": "spiral",
            "search_radius_mm": 5.0,
        },
        "compliance_guided": {
            "phases": ["approach", "contact", "comply", "insert", "verify"],
            "search_pattern": "force_guided",
        },
        "visual_servoing": {
            "phases": ["approach", "visual_align", "insert", "verify"],
            "search_pattern": "visual",
        },
    }

    def __init__(
        self,
        verbose: bool = True,
    ):
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[CONTACT-RICH] {msg}")

    def create_task_spec(
        self,
        task_type: ContactRichTaskType,
        tolerance_class: ToleranceClass = ToleranceClass.MEDIUM,
        **kwargs,
    ) -> ContactRichTaskSpec:
        """Create a task specification with appropriate defaults."""
        profile = self.PHYSICS_PROFILES.get(task_type, {})

        # Clearance based on tolerance class
        clearance_map = {
            ToleranceClass.LOOSE: 2.5,
            ToleranceClass.MEDIUM: 1.0,
            ToleranceClass.TIGHT: 0.3,
            ToleranceClass.PRECISION: 0.05,
        }

        spec = ContactRichTaskSpec(
            task_type=task_type,
            tolerance_class=tolerance_class,
            clearance_mm=kwargs.get("clearance_mm", clearance_map[tolerance_class]),
            friction_coefficient=kwargs.get("friction", profile.get("friction", 0.3)),
            stiffness=kwargs.get("stiffness", profile.get("stiffness", 1000)),
            damping=kwargs.get("damping", profile.get("damping", 100)),
            **{k: v for k, v in kwargs.items() if k not in ["clearance_mm", "friction", "stiffness", "damping"]},
        )

        return spec

    def generate_insertion_phases(
        self,
        task_spec: ContactRichTaskSpec,
        start_position: np.ndarray,
        hole_position: np.ndarray,
        strategy: str = "spiral_search",
    ) -> List[InsertionPhase]:
        """Generate insertion phases based on strategy."""
        strategy_config = self.INSERTION_STRATEGIES.get(strategy, self.INSERTION_STRATEGIES["direct"])

        phases = []
        current_time = 0.0

        # Approach phase
        approach_distance = np.linalg.norm(hole_position[:2] - start_position[:2])
        approach_time = approach_distance / (task_spec.approach_speed_mms / 1000.0)

        phases.append(InsertionPhase(
            name="approach",
            start_time=current_time,
            end_time=current_time + approach_time,
            phase_type="approach",
            target_position=np.array([hole_position[0], hole_position[1], start_position[2]]),
            target_force=np.zeros(3),
            stiffness_gains=np.array([2000, 2000, 1000, 500, 500, 500]),
        ))
        current_time += approach_time

        # Search phase (if applicable)
        if "search" in strategy_config["phases"]:
            search_time = 3.0  # seconds
            phases.append(InsertionPhase(
                name="search",
                start_time=current_time,
                end_time=current_time + search_time,
                phase_type="search",
                target_position=hole_position + np.array([0, 0, 5e-3]),  # 5mm above
                target_force=np.array([0, 0, -5]),  # Light downward force
                stiffness_gains=np.array([500, 500, 100, 200, 200, 500]),
            ))
            current_time += search_time

        # Align phase
        if "align" in strategy_config["phases"]:
            align_time = 1.0
            phases.append(InsertionPhase(
                name="align",
                start_time=current_time,
                end_time=current_time + align_time,
                phase_type="align",
                target_position=hole_position + np.array([0, 0, 2e-3]),
                target_force=np.array([0, 0, -8]),
                stiffness_gains=np.array([1000, 1000, 200, 300, 300, 500]),
            ))
            current_time += align_time

        # Insert phase
        insertion_time = task_spec.insertion_depth_mm / task_spec.insertion_speed_mms
        phases.append(InsertionPhase(
            name="insert",
            start_time=current_time,
            end_time=current_time + insertion_time,
            phase_type="insert",
            target_position=hole_position - np.array([0, 0, task_spec.insertion_depth_mm * 1e-3]),
            target_force=np.array([0, 0, -task_spec.force_threshold_n]),
            stiffness_gains=np.array([1500, 1500, 500, 400, 400, 500]),
        ))
        current_time += insertion_time

        # Verify phase
        verify_time = 0.5
        phases.append(InsertionPhase(
            name="verify",
            start_time=current_time,
            end_time=current_time + verify_time,
            phase_type="verify",
            target_position=hole_position - np.array([0, 0, task_spec.insertion_depth_mm * 1e-3]),
            target_force=np.array([0, 0, -2]),  # Light hold
        ))

        return phases

    def generate_force_profile(
        self,
        phases: List[InsertionPhase],
        fps: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate force/torque profiles for the trajectory."""
        total_time = phases[-1].end_time
        num_frames = int(total_time * fps)

        force_profile = np.zeros((num_frames, 3))
        torque_profile = np.zeros((num_frames, 3))

        for phase in phases:
            start_frame = int(phase.start_time * fps)
            end_frame = int(phase.end_time * fps)

            for i in range(start_frame, min(end_frame, num_frames)):
                t = (i - start_frame) / max(1, end_frame - start_frame)

                if phase.phase_type == "insert":
                    # Ramping force during insertion
                    force_profile[i] = phase.target_force * t
                    # Add small torque for chamfer following
                    torque_profile[i] = np.array([0.1 * math.sin(t * 4 * math.pi), 0.1 * math.cos(t * 4 * math.pi), 0])
                elif phase.phase_type == "search":
                    # Spiral search pattern
                    angle = t * 4 * math.pi
                    radius = 0.002 * t  # Growing spiral
                    force_profile[i] = phase.target_force + np.array([
                        3 * math.cos(angle),
                        3 * math.sin(angle),
                        0,
                    ])
                else:
                    force_profile[i] = phase.target_force

        return force_profile, torque_profile

    def generate_contact_rich_episode(
        self,
        task_spec: ContactRichTaskSpec,
        start_position: np.ndarray,
        hole_position: np.ndarray,
        strategy: str = "spiral_search",
    ) -> ContactRichTrajectory:
        """Generate a complete contact-rich episode."""
        self.log(f"Generating {task_spec.task_type.value} episode with {strategy} strategy")

        # Generate phases
        phases = self.generate_insertion_phases(
            task_spec=task_spec,
            start_position=start_position,
            hole_position=hole_position,
            strategy=strategy,
        )

        # Generate force profile
        force_profile, torque_profile = self.generate_force_profile(phases)

        # Simulate contact events
        contact_times = []
        contact_forces = []

        # Contact at start of insert phase
        insert_phase = next((p for p in phases if p.phase_type == "insert"), None)
        if insert_phase:
            contact_times.append(insert_phase.start_time)
            contact_forces.append(np.array([0, 0, -task_spec.force_threshold_n * 0.5]))

            # Contact at bottom
            contact_times.append(insert_phase.end_time)
            contact_forces.append(np.array([0, 0, -task_spec.force_threshold_n]))

        trajectory = ContactRichTrajectory(
            task_spec=task_spec,
            phases=phases,
            force_profile=force_profile,
            torque_profile=torque_profile,
            contact_times=contact_times,
            contact_forces=contact_forces,
            insertion_success=True,
            max_force_applied=float(np.max(np.abs(force_profile))),
            insertion_time=phases[-1].end_time,
            alignment_error_mm=task_spec.clearance_mm * 0.1,  # 10% of clearance
        )

        return trajectory


class ContactRichDataAugmenter:
    """
    Augments contact-rich episodes with realistic variations.
    """

    def __init__(
        self,
        position_noise_mm: float = 0.5,
        angle_noise_deg: float = 2.0,
        force_noise_percent: float = 10.0,
    ):
        self.position_noise_mm = position_noise_mm
        self.angle_noise_deg = angle_noise_deg
        self.force_noise_percent = force_noise_percent

    def augment(
        self,
        trajectory: ContactRichTrajectory,
        num_variations: int = 10,
    ) -> List[ContactRichTrajectory]:
        """Generate augmented variations of a trajectory."""
        variations = []

        for i in range(num_variations):
            # Clone trajectory
            var_trajectory = ContactRichTrajectory(
                task_spec=trajectory.task_spec,
                phases=trajectory.phases.copy(),
                force_profile=trajectory.force_profile.copy(),
                torque_profile=trajectory.torque_profile.copy(),
                contact_times=trajectory.contact_times.copy(),
                contact_forces=[f.copy() for f in trajectory.contact_forces],
                insertion_success=trajectory.insertion_success,
                max_force_applied=trajectory.max_force_applied,
                insertion_time=trajectory.insertion_time,
            )

            # Add noise to force profile
            force_noise = np.random.normal(
                0,
                self.force_noise_percent / 100.0 * np.abs(var_trajectory.force_profile),
            )
            var_trajectory.force_profile += force_noise

            # Add noise to alignment error
            var_trajectory.alignment_error_mm = trajectory.alignment_error_mm + \
                np.random.normal(0, self.position_noise_mm * 0.1)

            variations.append(var_trajectory)

        return variations


def generate_contact_rich_isaac_lab_config(
    task_spec: ContactRichTaskSpec,
    robot_type: str = "franka",
) -> str:
    """Generate Isaac Lab configuration for contact-rich tasks."""
    config = f'''# Contact-Rich Task Configuration
# Generated by BlueprintPipeline

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg, RewardTermCfg
from isaaclab.sensors import ContactSensorCfg, ForceTorqueSensorCfg

@configclass
class ContactRichEnvCfg(ManagerBasedEnvCfg):
    """Configuration for contact-rich manipulation environment."""

    # Task configuration
    task_type: str = "{task_spec.task_type.value}"
    tolerance_class: str = "{task_spec.tolerance_class.value}"

    # Physics settings (critical for contact-rich)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 240.0,  # 240 Hz for contact accuracy
        substeps=4,
        physx=sim_utils.PhysxCfg(
            num_position_iterations=8,
            num_velocity_iterations=2,
            contact_offset=0.001,
            rest_offset=0.0,
        ),
    )

    # Contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{{ENV_REGEX_NS}}/Robot/.*_link",
        update_period=0.0,
        filter_prim_paths_expr=["{{ENV_REGEX_NS}}/Target/.*"],
    )

    # Force-torque sensor at wrist
    ft_sensor: ForceTorqueSensorCfg = ForceTorqueSensorCfg(
        prim_path="{{ENV_REGEX_NS}}/Robot/ee_link",
        update_period=0.0,
    )

    # Compliance control parameters
    stiffness_gains = [{task_spec.stiffness}, {task_spec.stiffness}, {task_spec.stiffness}, 500, 500, 500]
    damping_gains = [{task_spec.damping}, {task_spec.damping}, {task_spec.damping}, 50, 50, 50]

    # Force thresholds
    max_force = {task_spec.max_force_n}
    force_threshold = {task_spec.force_threshold_n}
    torque_threshold = {task_spec.torque_threshold_nm}

    # Insertion parameters
    insertion_depth = {task_spec.insertion_depth_mm / 1000.0}  # meters
    clearance = {task_spec.clearance_mm / 1000.0}  # meters


@configclass
class ContactRichRewardsCfg:
    """Reward configuration for contact-rich tasks."""

    # Insertion progress
    insertion_progress = RewardTermCfg(
        func=insertion_progress_reward,
        weight=5.0,
    )

    # Force alignment
    force_alignment = RewardTermCfg(
        func=force_alignment_reward,
        weight=2.0,
    )

    # Minimize lateral forces
    minimize_lateral_force = RewardTermCfg(
        func=lateral_force_penalty,
        weight=-1.0,
    )

    # Success bonus
    insertion_success = RewardTermCfg(
        func=insertion_success_reward,
        weight=100.0,
    )

    # Excessive force penalty
    excessive_force = RewardTermCfg(
        func=excessive_force_penalty,
        weight=-10.0,
        params={{"max_force": {task_spec.max_force_n}}},
    )
'''

    return config


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate contact-rich manipulation episodes"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=[t.value for t in ContactRichTaskType],
        default="peg_in_hole",
    )
    parser.add_argument(
        "--tolerance",
        type=str,
        choices=[t.value for t in ToleranceClass],
        default="medium",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./contact_rich_episodes"),
    )

    args = parser.parse_args()

    generator = ContactRichTaskGenerator()

    task_spec = generator.create_task_spec(
        task_type=ContactRichTaskType(args.task_type),
        tolerance_class=ToleranceClass(args.tolerance),
    )

    # Generate episodes
    episodes = []
    for i in range(args.num_episodes):
        start_pos = np.array([0.5, 0.0, 0.3]) + np.random.uniform(-0.02, 0.02, 3)
        hole_pos = np.array([0.5, 0.0, 0.1])

        trajectory = generator.generate_contact_rich_episode(
            task_spec=task_spec,
            start_position=start_pos,
            hole_position=hole_pos,
        )
        episodes.append(trajectory)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "task_spec": task_spec.to_dict(),
        "num_episodes": len(episodes),
        "generated_at": "2026-01-02",
    }

    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Save Isaac Lab config
    config = generate_contact_rich_isaac_lab_config(task_spec)
    with open(args.output_dir / "contact_rich_env_cfg.py", "w") as f:
        f.write(config)

    print(f"Generated {len(episodes)} contact-rich episodes")
    print(f"Task type: {args.task_type}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
