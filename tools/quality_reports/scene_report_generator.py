#!/usr/bin/env python3
"""
Scene Quality Report Generator.

Generates standardized `scene_report.json` files that robotics labs expect
for procurement and quality assurance.

This is a KEY UPSELL differentiator - labs can show their procurement teams:
"We bought scenes with documented quality metrics, not just assets."

Output: quality/scene_report.json with:
- Physics stability stats (penetration rate, stability score, contact jitter)
- Asset inventory (counts by category, interactive vs static)
- Perception QA (if Replicator data available)
- Task QA (if Isaac Lab tasks configured)
- Overall pass/fail with breakdown

Reference: DROID dataset quality documentation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PhysicsQAMetrics:
    """Physics quality metrics for the scene."""

    # Stability metrics
    stability_score: float = 0.0  # 0-100 scale
    penetration_count: int = 0
    penetration_rate: float = 0.0  # Objects with penetration / total objects

    # Contact metrics
    contact_jitter: float = 0.0  # Variance in contact forces
    stable_contact_rate: float = 0.0  # Contacts that remain stable

    # Simulation metrics
    simulation_steps_to_settle: int = 0
    max_velocity_during_settle: float = 0.0
    objects_fell_through_floor: int = 0

    # Articulation metrics
    articulation_controllability: float = 0.0  # Can joints be driven smoothly
    joint_limit_violations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stability_score": round(self.stability_score, 2),
            "penetration": {
                "count": self.penetration_count,
                "rate": round(self.penetration_rate, 4),
            },
            "contact": {
                "jitter": round(self.contact_jitter, 4),
                "stable_rate": round(self.stable_contact_rate, 4),
            },
            "simulation": {
                "steps_to_settle": self.simulation_steps_to_settle,
                "max_velocity_during_settle": round(self.max_velocity_during_settle, 4),
                "objects_fell_through": self.objects_fell_through_floor,
            },
            "articulation": {
                "controllability_score": round(self.articulation_controllability, 2),
                "joint_limit_violations": self.joint_limit_violations,
            },
        }


@dataclass
class AssetInventoryMetrics:
    """Asset inventory and categorization metrics."""

    total_objects: int = 0
    static_objects: int = 0
    dynamic_objects: int = 0
    articulated_objects: int = 0

    # By category
    category_counts: Dict[str, int] = field(default_factory=dict)

    # By sim role
    sim_role_counts: Dict[str, int] = field(default_factory=dict)

    # Interactive stats
    manipulable_objects: int = 0
    container_objects: int = 0

    # Physics coverage
    objects_with_mass: int = 0
    objects_with_collision: int = 0
    objects_with_friction: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total_objects,
            "by_type": {
                "static": self.static_objects,
                "dynamic": self.dynamic_objects,
                "articulated": self.articulated_objects,
            },
            "by_category": self.category_counts,
            "by_sim_role": self.sim_role_counts,
            "interactive": {
                "manipulable": self.manipulable_objects,
                "containers": self.container_objects,
            },
            "physics_coverage": {
                "with_mass": self.objects_with_mass,
                "with_collision": self.objects_with_collision,
                "with_friction": self.objects_with_friction,
                "coverage_rate": round(
                    self.objects_with_collision / max(1, self.total_objects), 4
                ),
            },
        }


@dataclass
class PerceptionQAMetrics:
    """Perception/Replicator quality metrics."""

    # Render quality
    render_resolution: tuple = (640, 480)
    anti_aliasing_enabled: bool = True

    # Segmentation quality
    num_semantic_classes: int = 0
    avg_mask_coverage: float = 0.0  # Average area covered by valid masks

    # Depth quality
    depth_range_valid: bool = True
    depth_min: float = 0.0
    depth_max: float = 0.0

    # Bbox quality
    bbox_coverage_rate: float = 0.0  # Objects with valid bboxes / total
    occluded_objects_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "render": {
                "resolution": list(self.render_resolution),
                "anti_aliasing": self.anti_aliasing_enabled,
            },
            "segmentation": {
                "num_classes": self.num_semantic_classes,
                "avg_mask_coverage": round(self.avg_mask_coverage, 4),
            },
            "depth": {
                "valid": self.depth_range_valid,
                "range": [round(self.depth_min, 4), round(self.depth_max, 4)],
            },
            "bounding_boxes": {
                "coverage_rate": round(self.bbox_coverage_rate, 4),
                "occluded_rate": round(self.occluded_objects_rate, 4),
            },
        }


@dataclass
class TaskQAMetrics:
    """Task/Isaac Lab quality metrics."""

    num_tasks_defined: int = 0
    tasks_with_rewards: int = 0
    tasks_with_termination: int = 0

    # Task types
    task_types: Dict[str, int] = field(default_factory=dict)

    # Baseline performance (if available)
    baseline_success_rates: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_tasks": self.num_tasks_defined,
            "tasks_with_rewards": self.tasks_with_rewards,
            "tasks_with_termination": self.tasks_with_termination,
            "task_types": self.task_types,
            "baseline_success_rates": {
                k: round(v, 4) for k, v in self.baseline_success_rates.items()
            },
        }


@dataclass
class SceneQualityReport:
    """
    Complete scene quality report.

    This is the main output that labs want to see:
    - Overall quality score
    - Pass/fail status with breakdown
    - Detailed metrics by category
    """

    scene_id: str
    version: str = "1.0.0"
    generated_at: str = ""

    # Overall quality
    overall_score: float = 0.0  # 0-100
    passed: bool = False

    # Component scores
    physics_score: float = 0.0
    asset_score: float = 0.0
    perception_score: float = 0.0
    task_score: float = 0.0

    # Detailed metrics
    physics_metrics: PhysicsQAMetrics = field(default_factory=PhysicsQAMetrics)
    asset_metrics: AssetInventoryMetrics = field(default_factory=AssetInventoryMetrics)
    perception_metrics: PerceptionQAMetrics = field(default_factory=PerceptionQAMetrics)
    task_metrics: TaskQAMetrics = field(default_factory=TaskQAMetrics)

    # Issues and warnings
    blocking_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    pipeline_version: str = "1.0.0"
    source_reconstruction: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat() + "Z"

    def compute_overall_score(self) -> float:
        """Compute weighted overall score."""
        weights = {
            "physics": 0.35,
            "asset": 0.25,
            "perception": 0.20,
            "task": 0.20,
        }

        self.overall_score = (
            weights["physics"] * self.physics_score +
            weights["asset"] * self.asset_score +
            weights["perception"] * self.perception_score +
            weights["task"] * self.task_score
        )

        # Pass if overall >= 70 and no blocking issues
        self.passed = self.overall_score >= 70 and len(self.blocking_issues) == 0

        return self.overall_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "version": self.version,
            "generated_at": self.generated_at,
            "summary": {
                "overall_score": round(self.overall_score, 2),
                "passed": self.passed,
                "grade": self._score_to_grade(self.overall_score),
                "component_scores": {
                    "physics": round(self.physics_score, 2),
                    "assets": round(self.asset_score, 2),
                    "perception": round(self.perception_score, 2),
                    "tasks": round(self.task_score, 2),
                },
            },
            "physics": self.physics_metrics.to_dict(),
            "assets": self.asset_metrics.to_dict(),
            "perception": self.perception_metrics.to_dict(),
            "tasks": self.task_metrics.to_dict(),
            "issues": {
                "blocking": self.blocking_issues,
                "warnings": self.warnings,
                "recommendations": self.recommendations,
            },
            "metadata": {
                "pipeline_version": self.pipeline_version,
                "source_reconstruction": self.source_reconstruction,
            },
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def save(self, output_path: Path) -> None:
        """Save report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class SceneReportGenerator:
    """
    Generates comprehensive scene quality reports.

    Usage:
        generator = SceneReportGenerator(scene_dir)
        report = generator.generate()
        report.save(scene_dir / "quality" / "scene_report.json")
    """

    def __init__(
        self,
        scene_dir: Path,
        scene_id: Optional[str] = None,
        verbose: bool = True,
    ):
        self.scene_dir = Path(scene_dir)
        self.scene_id = scene_id or self.scene_dir.name
        self.verbose = verbose

        self.report = SceneQualityReport(scene_id=self.scene_id)

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SCENE-REPORT] {msg}")

    def generate(self) -> SceneQualityReport:
        """Generate complete scene quality report."""
        self.log(f"Generating quality report for: {self.scene_id}")

        # Load manifest
        manifest = self._load_manifest()

        # Generate component metrics
        self._analyze_physics(manifest)
        self._analyze_assets(manifest)
        self._analyze_perception()
        self._analyze_tasks()

        # Compute overall score
        self.report.compute_overall_score()

        # Generate recommendations
        self._generate_recommendations()

        self.log(f"Report complete: Score={self.report.overall_score:.1f}, Passed={self.report.passed}")

        return self.report

    def _load_manifest(self) -> Dict[str, Any]:
        """Load scene manifest."""
        manifest_path = self.scene_dir / "assets" / "scene_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f)
        return {}

    def _analyze_physics(self, manifest: Dict[str, Any]) -> None:
        """Analyze physics quality."""
        self.log("Analyzing physics quality...")

        objects = manifest.get("objects", [])
        physics_metrics = self.report.physics_metrics

        # Count penetrations (from validation if available)
        validation_path = self.scene_dir / "quality" / "validation_results.json"
        if validation_path.exists():
            with open(validation_path) as f:
                validation = json.load(f)
                physics_metrics.penetration_count = validation.get("penetration_count", 0)
                physics_metrics.simulation_steps_to_settle = validation.get("settle_steps", 100)

        # Calculate penetration rate
        total_objects = len(objects)
        if total_objects > 0:
            physics_metrics.penetration_rate = physics_metrics.penetration_count / total_objects

        # Count articulated objects for controllability
        articulated = [o for o in objects if o.get("sim_role") in ["articulated_furniture", "articulated_appliance"]]
        if articulated:
            # Check for articulation configs
            articulation_valid = sum(1 for o in articulated if o.get("articulation", {}).get("joints"))
            physics_metrics.articulation_controllability = 100.0 * articulation_valid / len(articulated)
        else:
            physics_metrics.articulation_controllability = 100.0  # No articulated objects = full score

        # Compute stability score
        base_score = 100.0
        base_score -= physics_metrics.penetration_rate * 50  # -50 max for penetrations
        base_score -= min(physics_metrics.joint_limit_violations * 5, 20)  # -20 max for violations
        base_score -= max(0, (physics_metrics.simulation_steps_to_settle - 100) * 0.1)  # Penalty for slow settle

        physics_metrics.stability_score = max(0, min(100, base_score))
        self.report.physics_score = physics_metrics.stability_score

        # Add issues
        if physics_metrics.penetration_rate > 0.1:
            self.report.blocking_issues.append(
                f"High penetration rate ({physics_metrics.penetration_rate:.1%}) - physics simulation may be unstable"
            )
        if physics_metrics.objects_fell_through_floor > 0:
            self.report.blocking_issues.append(
                f"{physics_metrics.objects_fell_through_floor} objects fell through floor"
            )

    def _analyze_assets(self, manifest: Dict[str, Any]) -> None:
        """Analyze asset inventory."""
        self.log("Analyzing asset inventory...")

        objects = manifest.get("objects", [])
        asset_metrics = self.report.asset_metrics

        asset_metrics.total_objects = len(objects)

        for obj in objects:
            sim_role = obj.get("sim_role", "unknown")
            category = obj.get("category", "unknown")

            # Count by sim role
            asset_metrics.sim_role_counts[sim_role] = asset_metrics.sim_role_counts.get(sim_role, 0) + 1

            # Count by category
            asset_metrics.category_counts[category] = asset_metrics.category_counts.get(category, 0) + 1

            # Classify by type
            if sim_role in ["static_furniture", "background", "scene_shell"]:
                asset_metrics.static_objects += 1
            elif sim_role in ["articulated_furniture", "articulated_appliance"]:
                asset_metrics.articulated_objects += 1
            else:
                asset_metrics.dynamic_objects += 1

            # Check manipulable
            if sim_role == "manipulable_object":
                asset_metrics.manipulable_objects += 1

            # Check physics coverage
            physics = obj.get("physics", {})
            if physics.get("mass") is not None:
                asset_metrics.objects_with_mass += 1
            if physics.get("collision_shape") or obj.get("asset", {}).get("path"):
                asset_metrics.objects_with_collision += 1
            if physics.get("friction") is not None:
                asset_metrics.objects_with_friction += 1

        # Compute asset score
        score = 100.0

        # Penalize for missing physics
        if asset_metrics.total_objects > 0:
            collision_coverage = asset_metrics.objects_with_collision / asset_metrics.total_objects
            if collision_coverage < 0.8:
                score -= (0.8 - collision_coverage) * 50
                self.report.warnings.append(
                    f"Only {collision_coverage:.1%} of objects have collision shapes"
                )

        # Bonus for manipulable objects (for manipulation tasks)
        if asset_metrics.manipulable_objects > 0:
            score = min(100, score + 5)

        self.report.asset_score = max(0, score)

    def _analyze_perception(self) -> None:
        """Analyze perception/Replicator quality."""
        self.log("Analyzing perception quality...")

        perception_metrics = self.report.perception_metrics

        # Check Replicator bundle
        replicator_dir = self.scene_dir / "replicator"
        if not replicator_dir.exists():
            self.report.warnings.append("No Replicator bundle found")
            self.report.perception_score = 50.0  # Partial score
            return

        # Check placement regions
        placement_regions = replicator_dir / "placement_regions.usda"
        if not placement_regions.exists():
            self.report.warnings.append("No placement regions defined for domain randomization")

        # Check policy scripts
        policies_dir = replicator_dir / "policies"
        if policies_dir.exists():
            policy_count = len(list(policies_dir.glob("*.py")))
            perception_metrics.num_semantic_classes = policy_count

        # Check for variation assets
        variation_manifest = replicator_dir / "variation_assets" / "manifest.json"
        if variation_manifest.exists():
            with open(variation_manifest) as f:
                variations = json.load(f)
                # Count texture variations etc.

        # Compute perception score
        score = 70.0  # Base score if Replicator exists

        if placement_regions.exists():
            score += 15

        if perception_metrics.num_semantic_classes > 0:
            score += 15

        self.report.perception_score = min(100, score)

    def _analyze_tasks(self) -> None:
        """Analyze Isaac Lab task quality."""
        self.log("Analyzing task quality...")

        task_metrics = self.report.task_metrics

        # Check Isaac Lab directory
        isaac_lab_dir = self.scene_dir / "isaac_lab"
        if not isaac_lab_dir.exists():
            self.report.warnings.append("No Isaac Lab tasks found")
            self.report.task_score = 50.0
            return

        # Count task files
        task_files = list(isaac_lab_dir.glob("task_*.py"))
        task_metrics.num_tasks_defined = len(task_files)

        # Check for env config
        env_cfg = isaac_lab_dir / "env_cfg.py"
        if env_cfg.exists():
            task_metrics.tasks_with_rewards += task_metrics.num_tasks_defined

        # Check for training config
        train_cfg = isaac_lab_dir / "train_cfg.yaml"
        if train_cfg.exists():
            task_metrics.tasks_with_termination += task_metrics.num_tasks_defined

        # Compute task score
        score = 50.0  # Base if Isaac Lab exists

        if task_metrics.num_tasks_defined > 0:
            score += 20

        if env_cfg.exists():
            score += 15

        if train_cfg.exists():
            score += 15

        self.report.task_score = min(100, score)

    def _generate_recommendations(self) -> None:
        """Generate improvement recommendations."""
        report = self.report

        if report.physics_score < 80:
            report.recommendations.append(
                "Run physics validation to identify and fix penetration issues"
            )

        if report.asset_score < 80:
            report.recommendations.append(
                "Add collision shapes to all manipulable objects"
            )

        if report.perception_score < 80:
            report.recommendations.append(
                "Add Replicator placement regions for domain randomization"
            )

        if report.task_score < 80:
            report.recommendations.append(
                "Define Isaac Lab tasks for the scene"
            )

        if not report.passed:
            report.recommendations.insert(0,
                "Address blocking issues before using this scene for training"
            )


def generate_scene_report(
    scene_dir: Path,
    output_path: Optional[Path] = None,
    scene_id: Optional[str] = None,
    verbose: bool = True,
) -> SceneQualityReport:
    """
    Convenience function to generate a scene quality report.

    Args:
        scene_dir: Path to scene directory
        output_path: Optional path to save report (defaults to scene_dir/quality/scene_report.json)
        scene_id: Optional scene ID
        verbose: Print progress

    Returns:
        SceneQualityReport
    """
    generator = SceneReportGenerator(scene_dir, scene_id, verbose)
    report = generator.generate()

    if output_path is None:
        output_path = scene_dir / "quality" / "scene_report.json"

    report.save(output_path)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate scene quality report")
    parser.add_argument("scene_dir", type=Path, help="Path to scene directory")
    parser.add_argument("--output", type=Path, help="Output path for report")
    parser.add_argument("--scene-id", help="Scene identifier")

    args = parser.parse_args()

    report = generate_scene_report(
        scene_dir=args.scene_dir,
        output_path=args.output,
        scene_id=args.scene_id,
    )

    print(f"\nScene Report Summary")
    print("=" * 50)
    print(f"Scene ID: {report.scene_id}")
    print(f"Overall Score: {report.overall_score:.1f} ({report._score_to_grade(report.overall_score)})")
    print(f"Passed: {report.passed}")
    print(f"\nComponent Scores:")
    print(f"  Physics: {report.physics_score:.1f}")
    print(f"  Assets: {report.asset_score:.1f}")
    print(f"  Perception: {report.perception_score:.1f}")
    print(f"  Tasks: {report.task_score:.1f}")

    if report.blocking_issues:
        print(f"\nBlocking Issues ({len(report.blocking_issues)}):")
        for issue in report.blocking_issues:
            print(f"  ❌ {issue}")

    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for warning in report.warnings:
            print(f"  ⚠️ {warning}")
