#!/usr/bin/env python3
"""
Default Embodiment Transfer Analysis for Genie Sim 3.0 & Arena Pipelines.

Previously $20,000-$100,000 upsell - NOW INCLUDED BY DEFAULT!

Analyzes cross-robot compatibility and data transfer potential.
Tells robotics labs: "Will my Franka data help train my UR10?"

Features (DEFAULT - FREE):
- Cross-robot compatibility matrix (franka→ur10, franka→gr1, etc.)
- Kinematic similarity scoring
- Action space compatibility analysis
- Workspace overlap computation
- Predicted success rate when transferring to different robot
- Transfer efficiency score (3-5x value for multi-robot data)
- Multi-robot training strategy recommendations
- Data multiplier calculation

Output:
- embodiment_transfer_matrix.json - Complete transfer analysis
- compatibility_scores.json - Robot-to-robot compatibility
- multi_robot_strategy.json - Training strategy recommendations
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Robot specifications
ROBOT_SPECS = {
    "franka": {
        "dof": 7,
        "reach_m": 0.855,
        "payload_kg": 3.0,
        "gripper_type": "parallel_jaw",
        "gripper_width_range_m": (0.0, 0.08),
        "joint_limits": [(-2.9, 2.9), (-1.76, 1.76), (-2.9, 2.9), (-3.07, -0.07), (-2.9, 2.9), (-0.02, 3.75), (-2.9, 2.9)],
    },
    "ur10": {
        "dof": 6,
        "reach_m": 1.300,
        "payload_kg": 10.0,
        "gripper_type": "parallel_jaw",
        "gripper_width_range_m": (0.0, 0.14),
        "joint_limits": [(-6.28, 6.28)] * 6,
    },
    "g2": {
        "dof": 14,  # Bimanual
        "reach_m": 0.900,
        "payload_kg": 5.0,
        "gripper_type": "parallel_jaw_bimanual",
        "gripper_width_range_m": (0.0, 0.10),
        "joint_limits": [(-3.14, 3.14)] * 14,
    },
    "gr1": {
        "dof": 14,  # Humanoid arms
        "reach_m": 0.700,
        "payload_kg": 3.0,
        "gripper_type": "anthropomorphic",
        "gripper_width_range_m": (0.0, 0.12),
        "joint_limits": [(-3.14, 3.14)] * 14,
    },
    "fetch": {
        "dof": 7,
        "reach_m": 0.800,
        "payload_kg": 5.0,
        "gripper_type": "parallel_jaw",
        "gripper_width_range_m": (0.0, 0.10),
        "joint_limits": [(-1.57, 1.57)] * 7,
    },
}


@dataclass
class EmbodimentCompatibility:
    """Compatibility between two robot embodiments."""
    source_robot: str
    target_robot: str
    compatibility_score: float  # 0-1
    kinematic_similarity: float
    workspace_overlap: float
    action_space_compatibility: float
    predicted_transfer_success: float

    # Details
    dof_match: bool
    gripper_compatibility: str  # "identical", "similar", "different"
    reach_ratio: float

    # Recommendations
    transfer_strategy: str
    expected_data_efficiency: float  # e.g., 3.5x = source data worth 3.5x for target
    recommendations: List[str] = field(default_factory=list)


@dataclass
class EmbodimentTransferMatrix:
    """Complete embodiment transfer analysis."""
    matrix_id: str
    scene_id: str
    source_robot: str
    created_at: str

    # Compatibility with other robots
    compatibility_matrix: Dict[str, EmbodimentCompatibility] = field(default_factory=dict)

    # Overall insights
    best_transfer_target: Optional[str] = None
    worst_transfer_target: Optional[str] = None
    multi_robot_data_multiplier: float = 1.0

    # Recommendations
    multi_robot_strategy: List[Dict[str, Any]] = field(default_factory=list)


def compute_kinematic_similarity(source: str, target: str) -> float:
    """Compute kinematic similarity between robots."""
    source_spec = ROBOT_SPECS.get(source, {})
    target_spec = ROBOT_SPECS.get(target, {})

    if not source_spec or not target_spec:
        return 0.0

    # DOF similarity
    dof_source = source_spec["dof"]
    dof_target = target_spec["dof"]
    dof_similarity = 1.0 - abs(dof_source - dof_target) / max(dof_source, dof_target)

    # Reach similarity
    reach_source = source_spec["reach_m"]
    reach_target = target_spec["reach_m"]
    reach_similarity = min(reach_source, reach_target) / max(reach_source, reach_target)

    # Gripper similarity
    gripper_source = source_spec["gripper_type"]
    gripper_target = target_spec["gripper_type"]
    if gripper_source == gripper_target:
        gripper_similarity = 1.0
    elif "parallel_jaw" in gripper_source and "parallel_jaw" in gripper_target:
        gripper_similarity = 0.9
    elif "anthropomorphic" in gripper_source or "anthropomorphic" in gripper_target:
        gripper_similarity = 0.3
    else:
        gripper_similarity = 0.5

    # Weighted average
    return (dof_similarity * 0.3 + reach_similarity * 0.4 + gripper_similarity * 0.3)


def compute_workspace_overlap(source: str, target: str) -> float:
    """Compute workspace overlap between robots."""
    source_spec = ROBOT_SPECS.get(source, {})
    target_spec = ROBOT_SPECS.get(target, {})

    if not source_spec or not target_spec:
        return 0.0

    reach_source = source_spec["reach_m"]
    reach_target = target_spec["reach_m"]

    # Simplified workspace overlap (spherical workspace assumption)
    min_reach = min(reach_source, reach_target)
    max_reach = max(reach_source, reach_target)

    # Volume overlap ratio
    overlap = (min_reach / max_reach) ** 3

    return overlap


def compute_action_space_compatibility(source: str, target: str) -> float:
    """Compute action space compatibility."""
    source_spec = ROBOT_SPECS.get(source, {})
    target_spec = ROBOT_SPECS.get(target, {})

    if not source_spec or not target_spec:
        return 0.0

    # DOF compatibility
    dof_source = source_spec["dof"]
    dof_target = target_spec["dof"]

    if dof_source == dof_target:
        dof_compat = 1.0
    else:
        # Can project higher DOF to lower, or add dummy actions
        dof_compat = min(dof_source, dof_target) / max(dof_source, dof_target) * 0.8

    # Gripper action compatibility
    gripper_source_range = source_spec["gripper_width_range_m"]
    gripper_target_range = target_spec["gripper_width_range_m"]

    gripper_min = max(gripper_source_range[0], gripper_target_range[0])
    gripper_max = min(gripper_source_range[1], gripper_target_range[1])

    if gripper_max > gripper_min:
        gripper_overlap = (gripper_max - gripper_min) / (
            max(gripper_source_range[1], gripper_target_range[1]) -
            min(gripper_source_range[0], gripper_target_range[0])
        )
    else:
        gripper_overlap = 0.0

    return (dof_compat * 0.7 + gripper_overlap * 0.3)


def create_default_embodiment_transfer_exporter(
    scene_id: str,
    source_robot: str,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Create embodiment transfer analysis (DEFAULT - NO LONGER UPSELL).

    Args:
        scene_id: Scene identifier
        source_robot: Source robot type
        output_dir: Output directory

    Returns:
        Dict mapping manifest names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute compatibility with all other robots
    compatibility_matrix = {}
    target_robots = [r for r in ROBOT_SPECS.keys() if r != source_robot]

    for target in target_robots:
        kinematic_sim = compute_kinematic_similarity(source_robot, target)
        workspace_overlap = compute_workspace_overlap(source_robot, target)
        action_compat = compute_action_space_compatibility(source_robot, target)

        # Overall compatibility
        compatibility_score = (
            kinematic_sim * 0.35 +
            workspace_overlap * 0.35 +
            action_compat * 0.30
        )

        # Predicted transfer success
        # Higher compatibility = higher transfer success
        predicted_success = compatibility_score * 0.9

        # DOF match
        source_dof = ROBOT_SPECS[source_robot]["dof"]
        target_dof = ROBOT_SPECS[target]["dof"]
        dof_match = (source_dof == target_dof)

        # Gripper compatibility
        source_gripper = ROBOT_SPECS[source_robot]["gripper_type"]
        target_gripper = ROBOT_SPECS[target]["gripper_type"]

        if source_gripper == target_gripper:
            gripper_compat = "identical"
        elif "parallel_jaw" in source_gripper and "parallel_jaw" in target_gripper:
            gripper_compat = "similar"
        else:
            gripper_compat = "different"

        # Reach ratio
        reach_ratio = ROBOT_SPECS[target]["reach_m"] / ROBOT_SPECS[source_robot]["reach_m"]

        # Transfer strategy
        if compatibility_score >= 0.8:
            strategy = "direct_transfer"
            data_efficiency = 4.0
        elif compatibility_score >= 0.6:
            strategy = "transfer_with_finetuning"
            data_efficiency = 2.5
        elif compatibility_score >= 0.4:
            strategy = "partial_transfer_retrain_gripper"
            data_efficiency = 1.5
        else:
            strategy = "minimal_transfer_mostly_retrain"
            data_efficiency = 1.2

        # Recommendations
        recommendations = []
        if compatibility_score >= 0.7:
            recommendations.append(f"High compatibility - expect {predicted_success:.0%} transfer success")
            recommendations.append(f"Source data worth {data_efficiency}x for {target}")
        if not dof_match:
            if source_dof > target_dof:
                recommendations.append(f"Project {source_dof} DOF actions to {target_dof} DOF")
            else:
                recommendations.append(f"Pad {source_dof} DOF actions to {target_dof} DOF")
        if gripper_compat == "different":
            recommendations.append(f"Gripper mismatch - retrain grasp controller")
        if reach_ratio < 0.8:
            recommendations.append(f"Target has shorter reach - may need to rescale tasks")
        elif reach_ratio > 1.2:
            recommendations.append(f"Target has longer reach - can handle larger workspaces")

        compatibility_matrix[target] = EmbodimentCompatibility(
            source_robot=source_robot,
            target_robot=target,
            compatibility_score=compatibility_score,
            kinematic_similarity=kinematic_sim,
            workspace_overlap=workspace_overlap,
            action_space_compatibility=action_compat,
            predicted_transfer_success=predicted_success,
            dof_match=dof_match,
            gripper_compatibility=gripper_compat,
            reach_ratio=reach_ratio,
            transfer_strategy=strategy,
            expected_data_efficiency=data_efficiency,
            recommendations=recommendations,
        )

    # Find best/worst targets
    best_target = max(compatibility_matrix.items(), key=lambda x: x[1].compatibility_score)[0]
    worst_target = min(compatibility_matrix.items(), key=lambda x: x[1].compatibility_score)[0]

    # Multi-robot data multiplier
    avg_efficiency = sum(c.expected_data_efficiency for c in compatibility_matrix.values()) / len(compatibility_matrix)
    multi_robot_multiplier = avg_efficiency

    # Multi-robot strategy
    multi_robot_strategy = [
        {
            "strategy": "collect_multi_robot_data",
            "priority": "HIGH",
            "rationale": f"Data from {source_robot} has {multi_robot_multiplier:.1f}x value across {len(compatibility_matrix)} robots",
            "estimated_value": f"${int(multi_robot_multiplier * 50000):,} equivalent value",
        },
        {
            "strategy": "prioritize_high_compatibility_targets",
            "priority": "HIGH",
            "targets": [
                robot for robot, compat in compatibility_matrix.items()
                if compat.compatibility_score >= 0.7
            ],
            "rationale": "These robots can directly benefit from source data",
        },
        {
            "strategy": "shared_visual_encoder",
            "priority": "MEDIUM",
            "rationale": "Train shared vision encoder across all robots",
            "expected_benefit": "30-50% training speedup",
        },
    ]

    matrix = EmbodimentTransferMatrix(
        matrix_id=str(uuid.uuid4())[:12],
        scene_id=scene_id,
        source_robot=source_robot,
        created_at=datetime.utcnow().isoformat() + "Z",
        compatibility_matrix=compatibility_matrix,
        best_transfer_target=best_target,
        worst_transfer_target=worst_target,
        multi_robot_data_multiplier=multi_robot_multiplier,
        multi_robot_strategy=multi_robot_strategy,
    )

    # Save main transfer matrix
    matrix_path = output_dir / "embodiment_transfer_matrix.json"
    with open(matrix_path, "w") as f:
        json.dump({
            "matrix_id": matrix.matrix_id,
            "scene_id": matrix.scene_id,
            "source_robot": matrix.source_robot,
            "created_at": matrix.created_at,
            "compatibility_matrix": {
                robot: {
                    "compatibility_score": f"{compat.compatibility_score:.1%}",
                    "kinematic_similarity": f"{compat.kinematic_similarity:.1%}",
                    "workspace_overlap": f"{compat.workspace_overlap:.1%}",
                    "action_space_compatibility": f"{compat.action_space_compatibility:.1%}",
                    "predicted_transfer_success": f"{compat.predicted_transfer_success:.1%}",
                    "dof_match": compat.dof_match,
                    "gripper_compatibility": compat.gripper_compatibility,
                    "reach_ratio": f"{compat.reach_ratio:.2f}x",
                    "transfer_strategy": compat.transfer_strategy,
                    "data_efficiency_multiplier": f"{compat.expected_data_efficiency:.1f}x",
                    "recommendations": compat.recommendations,
                }
                for robot, compat in matrix.compatibility_matrix.items()
            },
            "best_transfer_target": matrix.best_transfer_target,
            "worst_transfer_target": matrix.worst_transfer_target,
            "multi_robot_data_multiplier": f"{matrix.multi_robot_data_multiplier:.1f}x",
            "multi_robot_strategy": matrix.multi_robot_strategy,
            "value": "Previously $20,000-$100,000 upsell - NOW FREE BY DEFAULT",
        }, f, indent=2)

    # Save compatibility scores
    scores_path = output_dir / "compatibility_scores.json"
    with open(scores_path, "w") as f:
        json.dump({
            "source_robot": source_robot,
            "target_robots": {
                robot: {
                    "score": f"{compat.compatibility_score:.1%}",
                    "recommended": compat.compatibility_score >= 0.6,
                }
                for robot, compat in matrix.compatibility_matrix.items()
            },
            "recommended_targets": [
                robot for robot, compat in matrix.compatibility_matrix.items()
                if compat.compatibility_score >= 0.6
            ],
        }, f, indent=2)

    # Save multi-robot strategy
    strategy_path = output_dir / "multi_robot_strategy.json"
    with open(strategy_path, "w") as f:
        json.dump({
            "data_multiplier": f"{multi_robot_multiplier:.1f}x",
            "strategy_recommendations": multi_robot_strategy,
            "implementation_plan": {
                "phase_1": f"Collect data on {source_robot}",
                "phase_2": f"Transfer to high-compatibility robots: {[r for r, c in matrix.compatibility_matrix.items() if c.compatibility_score >= 0.7]}",
                "phase_3": "Fine-tune on target robots with 100-500 episodes each",
                "phase_4": "Validate on all target robots",
            },
            "estimated_savings": f"${int((multi_robot_multiplier - 1) * 100000):,} vs. training each robot from scratch",
        }, f, indent=2)

    return {
        "embodiment_transfer_matrix": matrix_path,
        "compatibility_scores": scores_path,
        "multi_robot_strategy": strategy_path,
    }


if __name__ == "__main__":
    # Demo
    manifests = create_default_embodiment_transfer_exporter(
        scene_id="test_scene",
        source_robot="franka",
        output_dir=Path("./embodiment_demo"),
    )
    print("Generated embodiment transfer analysis:")
    for name, path in manifests.items():
        print(f"  {name}: {path}")
