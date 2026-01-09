#!/usr/bin/env python3
"""
Embodiment Transfer Analysis for BlueprintPipeline.

Analyzes cross-robot compatibility and provides:
- Per-robot success rates
- Kinematic capability matrix
- Cross-embodiment transfer predictions
- Workspace coverage analysis
- Robot-specific recommendations

Upsell Value: $20,000-$100,000 per multi-robot dataset
- Labs with multiple robots can leverage single dataset
- Increases dataset value by 3-5x
- Answers: "Will data from robot X help train robot Y?"
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid


class RobotType(str, Enum):
    """Supported robot types."""
    FRANKA = "franka"
    UR5 = "ur5"
    UR10 = "ur10"
    FETCH = "fetch"
    KUKA_IIWA = "kuka_iiwa"
    SAWYER = "sawyer"
    GR1 = "gr1"
    G1 = "g1"
    UNITREE_H1 = "unitree_h1"
    CUSTOM = "custom"


class TransferCompatibility(str, Enum):
    """Cross-robot transfer compatibility levels."""
    EXCELLENT = "excellent"  # Direct transfer, minimal adaptation
    GOOD = "good"            # Minor adaptation needed
    MODERATE = "moderate"    # Significant adaptation/fine-tuning needed
    POOR = "poor"            # Major architectural changes needed
    INCOMPATIBLE = "incompatible"  # Fundamentally different


@dataclass
class RobotKinematics:
    """Robot kinematic properties."""
    robot_type: RobotType
    dof: int  # Degrees of freedom
    reach_m: float  # Maximum reach in meters
    payload_kg: float  # Maximum payload in kg

    # Joint limits (radians)
    joint_limits: List[Tuple[float, float]] = field(default_factory=list)

    # Workspace (approximate)
    workspace_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    workspace_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # End-effector capabilities
    gripper_type: str = "parallel_jaw"
    gripper_max_width_m: float = 0.08
    gripper_max_force_n: float = 70.0

    # Control properties
    max_velocity_rad_s: float = 2.0
    max_acceleration_rad_s2: float = 10.0
    control_frequency_hz: float = 1000.0

    # Action space
    action_dim: int = 7  # Joint positions + gripper

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_type": self.robot_type.value,
            "dof": self.dof,
            "reach_m": self.reach_m,
            "payload_kg": self.payload_kg,
            "workspace": {
                "min": self.workspace_min,
                "max": self.workspace_max,
            },
            "gripper": {
                "type": self.gripper_type,
                "max_width_m": self.gripper_max_width_m,
                "max_force_n": self.gripper_max_force_n,
            },
            "action_dim": self.action_dim,
        }


# Standard robot specifications
ROBOT_SPECS: Dict[RobotType, RobotKinematics] = {
    RobotType.FRANKA: RobotKinematics(
        robot_type=RobotType.FRANKA,
        dof=7,
        reach_m=0.855,
        payload_kg=3.0,
        workspace_min=(-0.8, -0.8, 0.0),
        workspace_max=(0.8, 0.8, 1.2),
        gripper_type="parallel_jaw",
        gripper_max_width_m=0.08,
        gripper_max_force_n=70.0,
        action_dim=8,  # 7 joints + 1 gripper
    ),
    RobotType.UR5: RobotKinematics(
        robot_type=RobotType.UR5,
        dof=6,
        reach_m=0.85,
        payload_kg=5.0,
        workspace_min=(-0.8, -0.8, 0.0),
        workspace_max=(0.8, 0.8, 1.0),
        gripper_type="parallel_jaw",
        action_dim=7,
    ),
    RobotType.UR10: RobotKinematics(
        robot_type=RobotType.UR10,
        dof=6,
        reach_m=1.3,
        payload_kg=10.0,
        workspace_min=(-1.2, -1.2, 0.0),
        workspace_max=(1.2, 1.2, 1.5),
        gripper_type="parallel_jaw",
        action_dim=7,
    ),
    RobotType.FETCH: RobotKinematics(
        robot_type=RobotType.FETCH,
        dof=7,
        reach_m=1.1,
        payload_kg=6.0,
        workspace_min=(-1.0, -1.0, 0.3),
        workspace_max=(1.0, 1.0, 2.0),
        gripper_type="parallel_jaw",
        action_dim=8,
    ),
    RobotType.KUKA_IIWA: RobotKinematics(
        robot_type=RobotType.KUKA_IIWA,
        dof=7,
        reach_m=0.8,
        payload_kg=14.0,
        workspace_min=(-0.75, -0.75, 0.0),
        workspace_max=(0.75, 0.75, 1.1),
        gripper_type="parallel_jaw",
        action_dim=8,
    ),
    RobotType.SAWYER: RobotKinematics(
        robot_type=RobotType.SAWYER,
        dof=7,
        reach_m=1.26,
        payload_kg=4.0,
        workspace_min=(-1.2, -1.2, 0.0),
        workspace_max=(1.2, 1.2, 1.4),
        gripper_type="parallel_jaw",
        action_dim=8,
    ),
    RobotType.GR1: RobotKinematics(
        robot_type=RobotType.GR1,
        dof=14,  # Two 7-DOF arms
        reach_m=0.7,
        payload_kg=3.0,
        workspace_min=(-0.8, -0.8, 0.5),
        workspace_max=(0.8, 0.8, 1.8),
        gripper_type="anthropomorphic",
        action_dim=16,  # 14 joints + 2 grippers
    ),
    RobotType.G1: RobotKinematics(
        robot_type=RobotType.G1,
        dof=14,
        reach_m=0.65,
        payload_kg=2.0,
        workspace_min=(-0.7, -0.7, 0.4),
        workspace_max=(0.7, 0.7, 1.6),
        gripper_type="anthropomorphic",
        action_dim=16,
    ),
    RobotType.UNITREE_H1: RobotKinematics(
        robot_type=RobotType.UNITREE_H1,
        dof=14,
        reach_m=0.6,
        payload_kg=2.5,
        workspace_min=(-0.6, -0.6, 0.5),
        workspace_max=(0.6, 0.6, 1.7),
        gripper_type="dexterous",
        action_dim=16,
    ),
}


@dataclass
class EmbodimentPerformance:
    """Performance metrics for a single robot embodiment."""
    robot_type: RobotType
    total_episodes: int
    successful_episodes: int
    success_rate: float

    # Per-task breakdown
    task_success_rates: Dict[str, float] = field(default_factory=dict)

    # Per-object breakdown
    object_success_rates: Dict[str, float] = field(default_factory=dict)

    # Kinematic analysis
    reachable_fraction: float = 1.0  # Fraction of targets reachable
    workspace_coverage: float = 0.0  # Fraction of workspace used

    # Quality metrics
    avg_quality_score: float = 0.0
    avg_completion_time: float = 0.0
    avg_path_efficiency: float = 0.0

    # Failure analysis
    failure_modes: Dict[str, int] = field(default_factory=dict)
    primary_limitation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_type": self.robot_type.value,
            "episodes": {
                "total": self.total_episodes,
                "successful": self.successful_episodes,
                "success_rate": f"{self.success_rate:.1%}",
            },
            "task_success_rates": {
                k: f"{v:.1%}" for k, v in self.task_success_rates.items()
            },
            "object_success_rates": {
                k: f"{v:.1%}" for k, v in self.object_success_rates.items()
            },
            "kinematic_analysis": {
                "reachable_fraction": f"{self.reachable_fraction:.1%}",
                "workspace_coverage": f"{self.workspace_coverage:.1%}",
            },
            "quality_metrics": {
                "avg_quality": self.avg_quality_score,
                "avg_completion_time": self.avg_completion_time,
                "avg_path_efficiency": self.avg_path_efficiency,
            },
            "failure_analysis": {
                "failure_modes": self.failure_modes,
                "primary_limitation": self.primary_limitation,
            },
        }


@dataclass
class TransferPrediction:
    """Prediction for cross-embodiment transfer."""
    source_robot: RobotType
    target_robot: RobotType
    compatibility: TransferCompatibility
    predicted_success_rate: float  # Expected success on target
    transfer_efficiency: float  # How much source data helps (0-1)

    # Factors
    kinematic_similarity: float
    action_space_compatibility: float
    workspace_overlap: float

    # Recommendations
    adaptation_required: List[str] = field(default_factory=list)
    not_transferable: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_robot.value,
            "target": self.target_robot.value,
            "compatibility": self.compatibility.value,
            "predicted_success_rate": f"{self.predicted_success_rate:.1%}",
            "transfer_efficiency": f"{self.transfer_efficiency:.1%}",
            "factors": {
                "kinematic_similarity": self.kinematic_similarity,
                "action_space_compatibility": self.action_space_compatibility,
                "workspace_overlap": self.workspace_overlap,
            },
            "recommendations": {
                "adaptation_required": self.adaptation_required,
                "not_transferable": self.not_transferable,
            },
        }


@dataclass
class EmbodimentTransferReport:
    """Complete embodiment transfer analysis report."""
    report_id: str
    scene_id: str
    created_at: str

    # Per-robot performance
    performances: Dict[str, EmbodimentPerformance] = field(default_factory=dict)

    # Transfer matrix
    transfer_matrix: Dict[str, Dict[str, TransferPrediction]] = field(default_factory=dict)

    # Best performers
    best_overall: Optional[RobotType] = None
    best_per_task: Dict[str, RobotType] = field(default_factory=dict)

    # Recommendations
    recommended_training_order: List[RobotType] = field(default_factory=list)
    multi_robot_strategy: str = ""

    # Value assessment
    data_multiplier: float = 1.0  # How much multi-robot data multiplies value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "scene_id": self.scene_id,
            "created_at": self.created_at,
            "performances": {
                k: v.to_dict() for k, v in self.performances.items()
            },
            "transfer_matrix": {
                source: {
                    target: pred.to_dict()
                    for target, pred in targets.items()
                }
                for source, targets in self.transfer_matrix.items()
            },
            "recommendations": {
                "best_overall": self.best_overall.value if self.best_overall else None,
                "best_per_task": {
                    k: v.value for k, v in self.best_per_task.items()
                },
                "training_order": [r.value for r in self.recommended_training_order],
                "strategy": self.multi_robot_strategy,
            },
            "value_assessment": {
                "data_multiplier": f"{self.data_multiplier:.1f}x",
            },
        }


class EmbodimentTransferAnalyzer:
    """
    Analyzes cross-robot transfer potential and generates compatibility matrix.

    Helps labs understand how well data from one robot will transfer to another.
    """

    # Transfer compatibility rules based on DOF and morphology
    DOF_TRANSFER_PENALTIES = {
        0: 0.0,    # Same DOF
        1: 0.1,    # 1 DOF difference
        2: 0.25,   # 2 DOF difference
        3: 0.4,    # 3 DOF difference
    }

    # Morphology compatibility
    MORPHOLOGY_GROUPS = {
        "arm_7dof": [RobotType.FRANKA, RobotType.KUKA_IIWA, RobotType.SAWYER, RobotType.FETCH],
        "arm_6dof": [RobotType.UR5, RobotType.UR10],
        "humanoid_bimanual": [RobotType.GR1, RobotType.G1, RobotType.UNITREE_H1],
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[EMBODIMENT-ANALYZER] {msg}")

    def get_robot_specs(self, robot_type: RobotType) -> RobotKinematics:
        """Get robot specifications."""
        return ROBOT_SPECS.get(robot_type, ROBOT_SPECS[RobotType.FRANKA])

    def compute_kinematic_similarity(
        self,
        source: RobotType,
        target: RobotType,
    ) -> float:
        """
        Compute kinematic similarity between two robots.

        Returns value 0-1 where 1 = identical kinematics.
        """
        source_spec = self.get_robot_specs(source)
        target_spec = self.get_robot_specs(target)

        # DOF similarity
        dof_diff = abs(source_spec.dof - target_spec.dof)
        dof_penalty = self.DOF_TRANSFER_PENALTIES.get(dof_diff, 0.6)
        dof_similarity = 1.0 - dof_penalty

        # Reach similarity
        reach_ratio = min(source_spec.reach_m, target_spec.reach_m) / \
                      max(source_spec.reach_m, target_spec.reach_m)

        # Morphology group similarity
        source_group = None
        target_group = None
        for group, robots in self.MORPHOLOGY_GROUPS.items():
            if source in robots:
                source_group = group
            if target in robots:
                target_group = group

        morphology_similarity = 1.0 if source_group == target_group else 0.5

        # Combined similarity
        similarity = (
            dof_similarity * 0.4 +
            reach_ratio * 0.3 +
            morphology_similarity * 0.3
        )

        return similarity

    def compute_action_space_compatibility(
        self,
        source: RobotType,
        target: RobotType,
    ) -> float:
        """
        Compute action space compatibility.

        Higher is better - considers if actions can be mapped directly.
        """
        source_spec = self.get_robot_specs(source)
        target_spec = self.get_robot_specs(target)

        # Same action dimension = high compatibility
        if source_spec.action_dim == target_spec.action_dim:
            return 0.95

        # Can target accommodate source actions?
        if target_spec.action_dim >= source_spec.action_dim:
            # Target has more DOF - can accommodate with null space
            return 0.85

        # Target has fewer DOF - may lose information
        ratio = target_spec.action_dim / source_spec.action_dim
        return ratio * 0.8

    def compute_workspace_overlap(
        self,
        source: RobotType,
        target: RobotType,
    ) -> float:
        """
        Compute workspace overlap between robots.

        Returns fraction of source workspace reachable by target.
        """
        source_spec = self.get_robot_specs(source)
        target_spec = self.get_robot_specs(target)

        # Simple overlap computation based on reach
        source_volume = 4/3 * math.pi * source_spec.reach_m ** 3
        target_volume = 4/3 * math.pi * target_spec.reach_m ** 3

        # Intersection (simplified - assume concentric workspaces)
        min_reach = min(source_spec.reach_m, target_spec.reach_m)
        intersection_volume = 4/3 * math.pi * min_reach ** 3

        overlap = intersection_volume / source_volume if source_volume > 0 else 0

        return min(1.0, overlap)

    def predict_transfer(
        self,
        source: RobotType,
        target: RobotType,
        source_success_rate: float = 0.8,
    ) -> TransferPrediction:
        """
        Predict transfer performance from source to target robot.

        Args:
            source: Source robot type
            target: Target robot type
            source_success_rate: Known success rate on source robot

        Returns:
            TransferPrediction with expected performance
        """
        kin_sim = self.compute_kinematic_similarity(source, target)
        action_compat = self.compute_action_space_compatibility(source, target)
        workspace_overlap = self.compute_workspace_overlap(source, target)

        # Transfer efficiency (how useful is source data)
        transfer_efficiency = (
            kin_sim * 0.4 +
            action_compat * 0.35 +
            workspace_overlap * 0.25
        )

        # Predicted success rate
        # Baseline: source rate degraded by transfer inefficiency
        predicted_success = source_success_rate * transfer_efficiency

        # Determine compatibility level
        if transfer_efficiency >= 0.85:
            compatibility = TransferCompatibility.EXCELLENT
        elif transfer_efficiency >= 0.7:
            compatibility = TransferCompatibility.GOOD
        elif transfer_efficiency >= 0.5:
            compatibility = TransferCompatibility.MODERATE
        elif transfer_efficiency >= 0.3:
            compatibility = TransferCompatibility.POOR
        else:
            compatibility = TransferCompatibility.INCOMPATIBLE

        # Generate recommendations
        adaptations = []
        not_transferable = []

        source_spec = self.get_robot_specs(source)
        target_spec = self.get_robot_specs(target)

        if source_spec.dof != target_spec.dof:
            adaptations.append(f"Adapt action space: {source_spec.dof}DOF -> {target_spec.dof}DOF")

        if source_spec.gripper_type != target_spec.gripper_type:
            adaptations.append(f"Adapt gripper control: {source_spec.gripper_type} -> {target_spec.gripper_type}")

        if workspace_overlap < 0.8:
            adaptations.append("Re-plan trajectories for different workspace")

        if kin_sim < 0.5:
            not_transferable.append("Joint-space policies may not transfer")

        if transfer_efficiency < 0.3:
            not_transferable.append("Consider collecting robot-specific data instead")

        return TransferPrediction(
            source_robot=source,
            target_robot=target,
            compatibility=compatibility,
            predicted_success_rate=predicted_success,
            transfer_efficiency=transfer_efficiency,
            kinematic_similarity=kin_sim,
            action_space_compatibility=action_compat,
            workspace_overlap=workspace_overlap,
            adaptation_required=adaptations,
            not_transferable=not_transferable,
        )

    def analyze_multi_robot_performance(
        self,
        episode_data: List[Dict[str, Any]],
    ) -> Dict[RobotType, EmbodimentPerformance]:
        """
        Analyze performance across multiple robot embodiments.

        Args:
            episode_data: List of episodes with robot_type and success fields

        Returns:
            Performance metrics per robot type
        """
        performances = {}

        # Group episodes by robot type
        episodes_by_robot: Dict[RobotType, List[Dict]] = {}
        for ep in episode_data:
            robot_str = ep.get("robot_type", "franka").lower()
            try:
                robot_type = RobotType(robot_str)
            except ValueError:
                robot_type = RobotType.CUSTOM

            if robot_type not in episodes_by_robot:
                episodes_by_robot[robot_type] = []
            episodes_by_robot[robot_type].append(ep)

        # Compute metrics per robot
        for robot_type, episodes in episodes_by_robot.items():
            total = len(episodes)
            successful = sum(1 for ep in episodes if ep.get("success", False))
            success_rate = successful / total if total > 0 else 0

            # Per-task success rates
            task_success: Dict[str, List[bool]] = {}
            for ep in episodes:
                task = ep.get("task_type", "unknown")
                if task not in task_success:
                    task_success[task] = []
                task_success[task].append(ep.get("success", False))

            task_rates = {
                task: sum(outcomes) / len(outcomes)
                for task, outcomes in task_success.items()
            }

            # Per-object success rates
            object_success: Dict[str, List[bool]] = {}
            for ep in episodes:
                obj = ep.get("target_object", ep.get("object_category", "unknown"))
                if obj not in object_success:
                    object_success[obj] = []
                object_success[obj].append(ep.get("success", False))

            object_rates = {
                obj: sum(outcomes) / len(outcomes)
                for obj, outcomes in object_success.items()
            }

            # Quality metrics
            quality_scores = [ep.get("quality_score", 0) for ep in episodes if ep.get("quality_score")]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

            completion_times = [ep.get("completion_time", 0) for ep in episodes if ep.get("completion_time")]
            avg_time = sum(completion_times) / len(completion_times) if completion_times else 0

            # Failure modes
            failure_modes: Dict[str, int] = {}
            for ep in episodes:
                if not ep.get("success", False):
                    mode = ep.get("failure_mode", "unknown")
                    failure_modes[mode] = failure_modes.get(mode, 0) + 1

            primary_limitation = max(failure_modes.items(), key=lambda x: x[1])[0] if failure_modes else None

            performances[robot_type] = EmbodimentPerformance(
                robot_type=robot_type,
                total_episodes=total,
                successful_episodes=successful,
                success_rate=success_rate,
                task_success_rates=task_rates,
                object_success_rates=object_rates,
                avg_quality_score=avg_quality,
                avg_completion_time=avg_time,
                failure_modes=failure_modes,
                primary_limitation=primary_limitation,
            )

        return performances

    def generate_transfer_matrix(
        self,
        performances: Dict[RobotType, EmbodimentPerformance],
        target_robots: Optional[List[RobotType]] = None,
    ) -> Dict[str, Dict[str, TransferPrediction]]:
        """
        Generate complete cross-robot transfer matrix.

        Args:
            performances: Performance metrics per robot
            target_robots: Optional list of target robots to consider

        Returns:
            Matrix of transfer predictions [source][target]
        """
        source_robots = list(performances.keys())
        if target_robots is None:
            # Include common robots
            target_robots = list(set(source_robots) | {
                RobotType.FRANKA, RobotType.UR10, RobotType.FETCH
            })

        matrix = {}
        for source in source_robots:
            source_success = performances[source].success_rate
            matrix[source.value] = {}

            for target in target_robots:
                prediction = self.predict_transfer(
                    source=source,
                    target=target,
                    source_success_rate=source_success,
                )
                matrix[source.value][target.value] = prediction

        return matrix

    def generate_report(
        self,
        episode_data: List[Dict[str, Any]],
        scene_id: str,
        target_robots: Optional[List[RobotType]] = None,
    ) -> EmbodimentTransferReport:
        """
        Generate complete embodiment transfer analysis report.

        Args:
            episode_data: List of episode data with robot_type and metrics
            scene_id: Scene identifier
            target_robots: Optional list of target robots

        Returns:
            Complete EmbodimentTransferReport
        """
        self.log(f"Analyzing {len(episode_data)} episodes for embodiment transfer...")

        # Analyze per-robot performance
        performances = self.analyze_multi_robot_performance(episode_data)

        # Generate transfer matrix
        transfer_matrix = self.generate_transfer_matrix(performances, target_robots)

        # Determine best performers
        best_overall = None
        best_rate = 0.0
        for robot, perf in performances.items():
            if perf.success_rate > best_rate:
                best_rate = perf.success_rate
                best_overall = robot

        # Best per task
        best_per_task = {}
        all_tasks = set()
        for perf in performances.values():
            all_tasks.update(perf.task_success_rates.keys())

        for task in all_tasks:
            best_robot = None
            best_task_rate = 0.0
            for robot, perf in performances.items():
                rate = perf.task_success_rates.get(task, 0)
                if rate > best_task_rate:
                    best_task_rate = rate
                    best_robot = robot
            if best_robot:
                best_per_task[task] = best_robot

        # Recommended training order (train on best first, then transfer)
        sorted_robots = sorted(
            performances.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        )
        recommended_order = [robot for robot, _ in sorted_robots]

        # Multi-robot strategy
        if len(performances) > 1:
            strategy = "Train on highest-performing robot first, then fine-tune for others using transfer learning"
        else:
            strategy = "Single robot dataset - collect data on target robots for better generalization"

        # Data multiplier (how valuable is multi-robot data)
        if len(performances) > 1:
            avg_transfer = sum(
                sum(p.transfer_efficiency for p in targets.values()) / len(targets)
                for targets in transfer_matrix.values()
            ) / len(transfer_matrix)
            data_multiplier = 1.0 + (len(performances) - 1) * avg_transfer
        else:
            data_multiplier = 1.0

        report = EmbodimentTransferReport(
            report_id=str(uuid.uuid4())[:12],
            scene_id=scene_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            performances={r.value: p for r, p in performances.items()},
            transfer_matrix=transfer_matrix,
            best_overall=best_overall,
            best_per_task=best_per_task,
            recommended_training_order=recommended_order,
            multi_robot_strategy=strategy,
            data_multiplier=data_multiplier,
        )

        self.log(f"Analysis complete: {len(performances)} robots analyzed")
        return report

    def save_report(
        self,
        report: EmbodimentTransferReport,
        output_path: Path,
    ) -> Path:
        """Save report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        self.log(f"Saved embodiment transfer report to {output_path}")
        return output_path


def analyze_embodiment_transfer(
    episodes_dir: Path,
    scene_id: str,
    output_dir: Optional[Path] = None,
) -> EmbodimentTransferReport:
    """
    Convenience function to analyze embodiment transfer.

    Args:
        episodes_dir: Path to episodes directory
        scene_id: Scene identifier
        output_dir: Optional output directory

    Returns:
        EmbodimentTransferReport
    """
    episodes_dir = Path(episodes_dir)

    # Load episode metadata
    episodes = []
    meta_file = episodes_dir / "meta" / "episodes.jsonl"
    if meta_file.exists():
        with open(meta_file) as f:
            for line in f:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not episodes:
        # Create placeholder data
        print(f"[EMBODIMENT-ANALYZER] No episode data found, using placeholder")
        episodes = [
            {"robot_type": "franka", "success": True, "task_type": "pick_place"},
            {"robot_type": "franka", "success": True, "task_type": "pick_place"},
            {"robot_type": "ur10", "success": True, "task_type": "pick_place"},
        ]

    analyzer = EmbodimentTransferAnalyzer(verbose=True)
    report = analyzer.generate_report(
        episode_data=episodes,
        scene_id=scene_id,
    )

    if output_dir:
        output_path = Path(output_dir) / "embodiment_transfer_report.json"
        analyzer.save_report(report, output_path)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze embodiment transfer")
    parser.add_argument("episodes_dir", type=Path, help="Path to episodes directory")
    parser.add_argument("--scene-id", required=True, help="Scene identifier")
    parser.add_argument("--output-dir", type=Path, help="Output directory")

    args = parser.parse_args()

    report = analyze_embodiment_transfer(
        episodes_dir=args.episodes_dir,
        scene_id=args.scene_id,
        output_dir=args.output_dir,
    )

    print(f"\n=== Embodiment Transfer Analysis ===")
    print(f"Robots Analyzed: {len(report.performances)}")
    for robot, perf in report.performances.items():
        print(f"  {robot}: {perf.success_rate:.1%} success ({perf.total_episodes} episodes)")
    print(f"\nBest Overall: {report.best_overall.value if report.best_overall else 'N/A'}")
    print(f"Data Multiplier: {report.data_multiplier:.1f}x")
    print(f"\nStrategy: {report.multi_robot_strategy}")
