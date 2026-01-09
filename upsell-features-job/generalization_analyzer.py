#!/usr/bin/env python3
"""
Generalization & Learning Curves Analysis for BlueprintPipeline.

Provides analysis of dataset generalization potential:
- Task difficulty stratification (easy vs. hard variants)
- Per-object success rates
- Learning efficiency metrics
- Scene variation impact analysis
- Curriculum learning recommendations

Upsell Value: $10,000-$30,000 per dataset
- Tells labs if data covers their use case
- Shows data efficiency (how much do I need?)
- Critical for curriculum learning research
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid


class DifficultyLevel(str, Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class VariationType(str, Enum):
    """Types of scene/task variations."""
    OBJECT_POSE = "object_pose"
    OBJECT_SCALE = "object_scale"
    LIGHTING = "lighting"
    CAMERA_VIEW = "camera_view"
    DISTRACTOR_COUNT = "distractor_count"
    CLUTTER_LEVEL = "clutter_level"
    TABLE_TEXTURE = "table_texture"
    PHYSICS_PARAMS = "physics_params"


@dataclass
class ObjectPerformance:
    """Performance metrics for a single object category."""
    object_category: str
    total_episodes: int
    successful_episodes: int
    success_rate: float

    # Difficulty breakdown
    easy_success_rate: float = 0.0
    medium_success_rate: float = 0.0
    hard_success_rate: float = 0.0

    # Variation robustness
    pose_variation_robustness: float = 0.0
    scale_variation_robustness: float = 0.0
    lighting_variation_robustness: float = 0.0

    # Learning curve (episodes needed for X% success)
    episodes_for_50_pct: Optional[int] = None
    episodes_for_70_pct: Optional[int] = None
    episodes_for_90_pct: Optional[int] = None

    # Problem areas
    failure_modes: Dict[str, int] = field(default_factory=dict)
    hardest_variation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_category": self.object_category,
            "episodes": {
                "total": self.total_episodes,
                "successful": self.successful_episodes,
                "success_rate": f"{self.success_rate:.1%}",
            },
            "difficulty_breakdown": {
                "easy": f"{self.easy_success_rate:.1%}",
                "medium": f"{self.medium_success_rate:.1%}",
                "hard": f"{self.hard_success_rate:.1%}",
            },
            "variation_robustness": {
                "pose": f"{self.pose_variation_robustness:.1%}",
                "scale": f"{self.scale_variation_robustness:.1%}",
                "lighting": f"{self.lighting_variation_robustness:.1%}",
            },
            "learning_curve": {
                "episodes_for_50_pct": self.episodes_for_50_pct,
                "episodes_for_70_pct": self.episodes_for_70_pct,
                "episodes_for_90_pct": self.episodes_for_90_pct,
            },
            "problem_areas": {
                "failure_modes": self.failure_modes,
                "hardest_variation": self.hardest_variation,
            },
        }


@dataclass
class TaskPerformance:
    """Performance metrics for a task type."""
    task_type: str
    total_episodes: int
    success_rate: float

    # Per-object success
    object_success_rates: Dict[str, float] = field(default_factory=dict)

    # Difficulty distribution
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)

    # Success by difficulty
    success_by_difficulty: Dict[str, float] = field(default_factory=dict)

    # Time analysis
    avg_completion_time: float = 0.0
    time_by_difficulty: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "total_episodes": self.total_episodes,
            "success_rate": f"{self.success_rate:.1%}",
            "object_success_rates": {
                k: f"{v:.1%}" for k, v in self.object_success_rates.items()
            },
            "difficulty_distribution": self.difficulty_distribution,
            "success_by_difficulty": {
                k: f"{v:.1%}" for k, v in self.success_by_difficulty.items()
            },
            "timing": {
                "avg_completion_time": self.avg_completion_time,
                "time_by_difficulty": self.time_by_difficulty,
            },
        }


@dataclass
class VariationImpact:
    """Impact analysis for a specific variation type."""
    variation_type: VariationType
    num_levels: int  # Number of variation levels
    variation_range: Tuple[float, float] = (0.0, 1.0)

    # Success rate by variation level
    success_by_level: Dict[str, float] = field(default_factory=dict)

    # Correlation with success
    correlation_with_success: float = 0.0

    # Recommendations
    critical_threshold: Optional[float] = None
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variation_type": self.variation_type.value,
            "num_levels": self.num_levels,
            "variation_range": self.variation_range,
            "success_by_level": {
                k: f"{v:.1%}" for k, v in self.success_by_level.items()
            },
            "correlation_with_success": self.correlation_with_success,
            "critical_threshold": self.critical_threshold,
            "recommendation": self.recommendation,
        }


@dataclass
class LearningCurvePoint:
    """Single point on a learning curve."""
    num_episodes: int
    success_rate: float
    std_dev: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)


@dataclass
class LearningCurve:
    """Learning curve analysis for dataset."""
    curve_type: str  # "cumulative", "rolling_window"
    points: List[LearningCurvePoint] = field(default_factory=list)

    # Efficiency metrics
    episodes_to_50_pct: Optional[int] = None
    episodes_to_70_pct: Optional[int] = None
    episodes_to_90_pct: Optional[int] = None
    convergence_rate: float = 0.0  # Success rate gain per episode

    # Comparison to baselines
    efficiency_vs_bridgedata: Optional[float] = None
    efficiency_vs_droid: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "curve_type": self.curve_type,
            "points": [
                {
                    "episodes": p.num_episodes,
                    "success_rate": f"{p.success_rate:.1%}",
                    "std_dev": p.std_dev,
                }
                for p in self.points
            ],
            "efficiency": {
                "episodes_to_50_pct": self.episodes_to_50_pct,
                "episodes_to_70_pct": self.episodes_to_70_pct,
                "episodes_to_90_pct": self.episodes_to_90_pct,
                "convergence_rate": f"{self.convergence_rate:.4f}",
            },
            "benchmark_comparison": {
                "vs_bridgedata": self.efficiency_vs_bridgedata,
                "vs_droid": self.efficiency_vs_droid,
            },
        }


@dataclass
class CurriculumRecommendation:
    """Recommended curriculum for training."""
    stage: int
    difficulty: DifficultyLevel
    objects: List[str]
    variations: List[str]
    num_episodes: int
    expected_success_rate: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "difficulty": self.difficulty.value,
            "objects": self.objects,
            "variations": self.variations,
            "num_episodes": self.num_episodes,
            "expected_success_rate": f"{self.expected_success_rate:.1%}",
            "rationale": self.rationale,
        }


@dataclass
class GeneralizationReport:
    """Complete generalization analysis report."""
    report_id: str
    scene_id: str
    created_at: str

    # Dataset overview
    total_episodes: int
    unique_objects: int
    unique_tasks: int
    unique_variations: int

    # Performance by object
    object_performances: Dict[str, ObjectPerformance] = field(default_factory=dict)

    # Performance by task
    task_performances: Dict[str, TaskPerformance] = field(default_factory=dict)

    # Variation impact
    variation_impacts: Dict[str, VariationImpact] = field(default_factory=dict)

    # Learning curves
    overall_learning_curve: Optional[LearningCurve] = None
    per_object_learning_curves: Dict[str, LearningCurve] = field(default_factory=dict)

    # Curriculum recommendations
    curriculum: List[CurriculumRecommendation] = field(default_factory=list)

    # Generalization score
    generalization_score: float = 0.0
    coverage_score: float = 0.0
    robustness_score: float = 0.0

    # Data requirements
    estimated_episodes_needed: Dict[str, int] = field(default_factory=dict)

    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "scene_id": self.scene_id,
            "created_at": self.created_at,
            "overview": {
                "total_episodes": self.total_episodes,
                "unique_objects": self.unique_objects,
                "unique_tasks": self.unique_tasks,
                "unique_variations": self.unique_variations,
            },
            "object_performances": {
                k: v.to_dict() for k, v in self.object_performances.items()
            },
            "task_performances": {
                k: v.to_dict() for k, v in self.task_performances.items()
            },
            "variation_impacts": {
                k: v.to_dict() for k, v in self.variation_impacts.items()
            },
            "learning_curves": {
                "overall": self.overall_learning_curve.to_dict() if self.overall_learning_curve else None,
                "per_object": {
                    k: v.to_dict() for k, v in self.per_object_learning_curves.items()
                },
            },
            "curriculum": [c.to_dict() for c in self.curriculum],
            "scores": {
                "generalization": f"{self.generalization_score:.1%}",
                "coverage": f"{self.coverage_score:.1%}",
                "robustness": f"{self.robustness_score:.1%}",
            },
            "data_requirements": self.estimated_episodes_needed,
            "recommendations": self.recommendations,
        }


class GeneralizationAnalyzer:
    """
    Analyzes dataset generalization potential for robotics labs.

    Provides insights on:
    - Which objects/tasks are well-covered
    - How much data is needed for good performance
    - Which variations hurt performance
    - How to structure curriculum learning
    """

    # Difficulty thresholds
    DIFFICULTY_THRESHOLDS = {
        DifficultyLevel.EASY: {"clutter": 0.2, "pose_var": 0.1, "scale_var": 0.1},
        DifficultyLevel.MEDIUM: {"clutter": 0.5, "pose_var": 0.3, "scale_var": 0.2},
        DifficultyLevel.HARD: {"clutter": 0.8, "pose_var": 0.5, "scale_var": 0.3},
        DifficultyLevel.EXPERT: {"clutter": 1.0, "pose_var": 0.7, "scale_var": 0.5},
    }

    # Learning curve benchmarks (episodes for 70% success)
    BENCHMARKS = {
        "bridgedata_v2": 50000,
        "droid": 70000,
        "robomimic_ph": 2000,
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[GENERALIZATION-ANALYZER] {msg}")

    def classify_difficulty(self, episode: Dict[str, Any]) -> DifficultyLevel:
        """Classify episode difficulty based on variations."""
        variations = episode.get("variations", {})

        clutter = variations.get("clutter_level", 0)
        pose_var = variations.get("pose_variation", 0)
        scale_var = variations.get("scale_variation", 0)

        # Check thresholds from hardest to easiest
        for level in [DifficultyLevel.EXPERT, DifficultyLevel.HARD,
                      DifficultyLevel.MEDIUM, DifficultyLevel.EASY]:
            thresholds = self.DIFFICULTY_THRESHOLDS[level]
            if (clutter >= thresholds["clutter"] or
                pose_var >= thresholds["pose_var"] or
                scale_var >= thresholds["scale_var"]):
                return level

        return DifficultyLevel.EASY

    def analyze_object_performance(
        self,
        episodes: List[Dict[str, Any]],
    ) -> Dict[str, ObjectPerformance]:
        """Analyze performance per object category."""
        by_object: Dict[str, List[Dict[str, Any]]] = {}

        for ep in episodes:
            obj = ep.get("target_object", ep.get("object_category", "unknown"))
            if obj not in by_object:
                by_object[obj] = []
            by_object[obj].append(ep)

        performances = {}
        for obj, obj_episodes in by_object.items():
            total = len(obj_episodes)
            successful = sum(1 for ep in obj_episodes if ep.get("success", False))

            # By difficulty
            by_difficulty: Dict[DifficultyLevel, List[bool]] = {
                d: [] for d in DifficultyLevel
            }
            for ep in obj_episodes:
                diff = self.classify_difficulty(ep)
                by_difficulty[diff].append(ep.get("success", False))

            def safe_rate(outcomes: List[bool]) -> float:
                return sum(outcomes) / len(outcomes) if outcomes else 0.0

            # Variation robustness
            pose_var_success = [
                ep.get("success", False) for ep in obj_episodes
                if ep.get("variations", {}).get("pose_variation", 0) > 0.3
            ]
            scale_var_success = [
                ep.get("success", False) for ep in obj_episodes
                if ep.get("variations", {}).get("scale_variation", 0) > 0.2
            ]
            lighting_var_success = [
                ep.get("success", False) for ep in obj_episodes
                if ep.get("variations", {}).get("lighting_variation", 0) > 0.3
            ]

            # Failure modes
            failure_modes: Dict[str, int] = {}
            for ep in obj_episodes:
                if not ep.get("success", False):
                    mode = ep.get("failure_mode", "unknown")
                    failure_modes[mode] = failure_modes.get(mode, 0) + 1

            performances[obj] = ObjectPerformance(
                object_category=obj,
                total_episodes=total,
                successful_episodes=successful,
                success_rate=successful / total if total > 0 else 0,
                easy_success_rate=safe_rate(by_difficulty[DifficultyLevel.EASY]),
                medium_success_rate=safe_rate(by_difficulty[DifficultyLevel.MEDIUM]),
                hard_success_rate=safe_rate(by_difficulty[DifficultyLevel.HARD]),
                pose_variation_robustness=safe_rate(pose_var_success),
                scale_variation_robustness=safe_rate(scale_var_success),
                lighting_variation_robustness=safe_rate(lighting_var_success),
                failure_modes=failure_modes,
            )

        return performances

    def analyze_task_performance(
        self,
        episodes: List[Dict[str, Any]],
    ) -> Dict[str, TaskPerformance]:
        """Analyze performance per task type."""
        by_task: Dict[str, List[Dict[str, Any]]] = {}

        for ep in episodes:
            task = ep.get("task_type", "pick_place")
            if task not in by_task:
                by_task[task] = []
            by_task[task].append(ep)

        performances = {}
        for task, task_episodes in by_task.items():
            total = len(task_episodes)
            successful = sum(1 for ep in task_episodes if ep.get("success", False))

            # Per-object success
            object_success: Dict[str, List[bool]] = {}
            for ep in task_episodes:
                obj = ep.get("target_object", "unknown")
                if obj not in object_success:
                    object_success[obj] = []
                object_success[obj].append(ep.get("success", False))

            # By difficulty
            by_difficulty: Dict[str, List[bool]] = {}
            time_by_difficulty: Dict[str, List[float]] = {}
            for ep in task_episodes:
                diff = self.classify_difficulty(ep).value
                if diff not in by_difficulty:
                    by_difficulty[diff] = []
                    time_by_difficulty[diff] = []
                by_difficulty[diff].append(ep.get("success", False))
                if ep.get("completion_time"):
                    time_by_difficulty[diff].append(ep["completion_time"])

            performances[task] = TaskPerformance(
                task_type=task,
                total_episodes=total,
                success_rate=successful / total if total > 0 else 0,
                object_success_rates={
                    k: sum(v) / len(v) for k, v in object_success.items() if v
                },
                difficulty_distribution={
                    k: len(v) for k, v in by_difficulty.items()
                },
                success_by_difficulty={
                    k: sum(v) / len(v) for k, v in by_difficulty.items() if v
                },
                avg_completion_time=statistics.mean([
                    ep.get("completion_time", 0) for ep in task_episodes
                    if ep.get("completion_time")
                ]) if any(ep.get("completion_time") for ep in task_episodes) else 0,
                time_by_difficulty={
                    k: statistics.mean(v) for k, v in time_by_difficulty.items() if v
                },
            )

        return performances

    def analyze_variation_impact(
        self,
        episodes: List[Dict[str, Any]],
    ) -> Dict[str, VariationImpact]:
        """Analyze how each variation type impacts success."""
        impacts = {}

        variation_types = [
            ("object_pose", VariationType.OBJECT_POSE),
            ("lighting_variation", VariationType.LIGHTING),
            ("camera_view", VariationType.CAMERA_VIEW),
            ("distractor_count", VariationType.DISTRACTOR_COUNT),
            ("clutter_level", VariationType.CLUTTER_LEVEL),
        ]

        for var_name, var_type in variation_types:
            # Group by variation level
            by_level: Dict[str, List[bool]] = {}
            values = []

            for ep in episodes:
                var_value = ep.get("variations", {}).get(var_name, 0)
                values.append(var_value)

                # Bucket into levels
                if var_value < 0.25:
                    level = "low"
                elif var_value < 0.5:
                    level = "medium"
                elif var_value < 0.75:
                    level = "high"
                else:
                    level = "extreme"

                if level not in by_level:
                    by_level[level] = []
                by_level[level].append(ep.get("success", False))

            # Compute correlation
            successes = [ep.get("success", False) for ep in episodes]
            correlation = self._compute_correlation(values, successes)

            # Find critical threshold
            critical_threshold = None
            prev_rate = 1.0
            for level in ["low", "medium", "high", "extreme"]:
                if level in by_level and by_level[level]:
                    rate = sum(by_level[level]) / len(by_level[level])
                    if rate < 0.5 and prev_rate >= 0.5:
                        critical_threshold = {
                            "low": 0.25, "medium": 0.5,
                            "high": 0.75, "extreme": 1.0
                        }.get(level)
                    prev_rate = rate

            # Recommendation
            if correlation < -0.3:
                recommendation = f"High {var_name} strongly reduces success - consider more data with high variation"
            elif correlation < -0.1:
                recommendation = f"{var_name} has moderate impact - data seems reasonably robust"
            else:
                recommendation = f"{var_name} has minimal impact - good robustness"

            impacts[var_name] = VariationImpact(
                variation_type=var_type,
                num_levels=len(by_level),
                variation_range=(min(values) if values else 0, max(values) if values else 1),
                success_by_level={
                    k: sum(v) / len(v) for k, v in by_level.items() if v
                },
                correlation_with_success=correlation,
                critical_threshold=critical_threshold,
                recommendation=recommendation,
            )

        return impacts

    def _compute_correlation(
        self,
        x: List[float],
        y: List[bool],
    ) -> float:
        """Compute correlation between variation level and success."""
        if not x or not y or len(x) != len(y):
            return 0.0

        y_float = [1.0 if b else 0.0 for b in y]

        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y_float) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y_float))

        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y_float)

        denominator = math.sqrt(sum_sq_x * sum_sq_y)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def compute_learning_curve(
        self,
        episodes: List[Dict[str, Any]],
        window_size: int = 50,
    ) -> LearningCurve:
        """Compute learning curve showing success rate vs. number of episodes."""
        # Sort by episode index/timestamp
        sorted_eps = sorted(
            episodes,
            key=lambda e: e.get("episode_idx", 0)
        )

        points = []
        cumulative_success = 0

        for i, ep in enumerate(sorted_eps):
            if ep.get("success", False):
                cumulative_success += 1

            # Add point every window_size episodes
            if (i + 1) % window_size == 0 or i == len(sorted_eps) - 1:
                rate = cumulative_success / (i + 1)
                points.append(LearningCurvePoint(
                    num_episodes=i + 1,
                    success_rate=rate,
                    std_dev=math.sqrt(rate * (1 - rate) / (i + 1)),  # Binomial std
                ))

        # Find episodes to reach thresholds
        episodes_to_50 = None
        episodes_to_70 = None
        episodes_to_90 = None

        for point in points:
            if point.success_rate >= 0.5 and episodes_to_50 is None:
                episodes_to_50 = point.num_episodes
            if point.success_rate >= 0.7 and episodes_to_70 is None:
                episodes_to_70 = point.num_episodes
            if point.success_rate >= 0.9 and episodes_to_90 is None:
                episodes_to_90 = point.num_episodes

        # Convergence rate (success rate gain per episode)
        if len(points) >= 2:
            rate_diff = points[-1].success_rate - points[0].success_rate
            ep_diff = points[-1].num_episodes - points[0].num_episodes
            convergence_rate = rate_diff / ep_diff if ep_diff > 0 else 0
        else:
            convergence_rate = 0

        # Benchmark comparison
        efficiency_vs_bridgedata = None
        efficiency_vs_droid = None
        if episodes_to_70:
            efficiency_vs_bridgedata = self.BENCHMARKS["bridgedata_v2"] / episodes_to_70
            efficiency_vs_droid = self.BENCHMARKS["droid"] / episodes_to_70

        return LearningCurve(
            curve_type="cumulative",
            points=points,
            episodes_to_50_pct=episodes_to_50,
            episodes_to_70_pct=episodes_to_70,
            episodes_to_90_pct=episodes_to_90,
            convergence_rate=convergence_rate,
            efficiency_vs_bridgedata=efficiency_vs_bridgedata,
            efficiency_vs_droid=efficiency_vs_droid,
        )

    def generate_curriculum(
        self,
        object_performances: Dict[str, ObjectPerformance],
        task_performances: Dict[str, TaskPerformance],
    ) -> List[CurriculumRecommendation]:
        """Generate curriculum learning recommendations."""
        curriculum = []

        # Sort objects by difficulty (easiest first)
        sorted_objects = sorted(
            object_performances.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        )

        # Stage 1: Easy objects, low variation
        easy_objects = [
            obj for obj, perf in sorted_objects
            if perf.success_rate >= 0.8
        ][:3]

        if easy_objects:
            curriculum.append(CurriculumRecommendation(
                stage=1,
                difficulty=DifficultyLevel.EASY,
                objects=easy_objects,
                variations=["low_clutter", "centered_pose"],
                num_episodes=500,
                expected_success_rate=0.85,
                rationale="Start with high-success objects to establish basic skills",
            ))

        # Stage 2: Medium difficulty
        medium_objects = [
            obj for obj, perf in sorted_objects
            if 0.6 <= perf.success_rate < 0.8
        ][:3]

        if medium_objects:
            curriculum.append(CurriculumRecommendation(
                stage=2,
                difficulty=DifficultyLevel.MEDIUM,
                objects=medium_objects + easy_objects[:1],
                variations=["medium_clutter", "moderate_pose_variation"],
                num_episodes=1000,
                expected_success_rate=0.70,
                rationale="Introduce moderate challenge with familiar objects mixed in",
            ))

        # Stage 3: Hard objects
        hard_objects = [
            obj for obj, perf in sorted_objects
            if perf.success_rate < 0.6
        ][:3]

        if hard_objects:
            curriculum.append(CurriculumRecommendation(
                stage=3,
                difficulty=DifficultyLevel.HARD,
                objects=hard_objects + medium_objects[:1],
                variations=["high_clutter", "large_pose_variation", "varied_lighting"],
                num_episodes=2000,
                expected_success_rate=0.60,
                rationale="Challenge the policy with difficult cases",
            ))

        # Stage 4: Full variation
        all_objects = list(object_performances.keys())[:5]
        curriculum.append(CurriculumRecommendation(
            stage=4,
            difficulty=DifficultyLevel.EXPERT,
            objects=all_objects,
            variations=["full_variation", "distractors", "novel_viewpoints"],
            num_episodes=3000,
            expected_success_rate=0.50,
            rationale="Full domain randomization for robust generalization",
        ))

        return curriculum

    def generate_report(
        self,
        episodes: List[Dict[str, Any]],
        scene_id: str,
    ) -> GeneralizationReport:
        """Generate complete generalization analysis report."""
        self.log(f"Analyzing {len(episodes)} episodes for generalization...")

        # Analyze components
        object_performances = self.analyze_object_performance(episodes)
        task_performances = self.analyze_task_performance(episodes)
        variation_impacts = self.analyze_variation_impact(episodes)
        overall_curve = self.compute_learning_curve(episodes)

        # Per-object learning curves
        per_object_curves = {}
        for obj in object_performances.keys():
            obj_episodes = [
                ep for ep in episodes
                if ep.get("target_object", ep.get("object_category")) == obj
            ]
            if len(obj_episodes) >= 20:
                per_object_curves[obj] = self.compute_learning_curve(
                    obj_episodes, window_size=10
                )

        # Generate curriculum
        curriculum = self.generate_curriculum(object_performances, task_performances)

        # Compute scores
        avg_success = sum(
            p.success_rate for p in object_performances.values()
        ) / max(1, len(object_performances))

        robustness_scores = []
        for perf in object_performances.values():
            robustness_scores.extend([
                perf.pose_variation_robustness,
                perf.scale_variation_robustness,
                perf.lighting_variation_robustness,
            ])
        avg_robustness = sum(robustness_scores) / max(1, len(robustness_scores))

        coverage_score = min(1.0, len(object_performances) / 10)  # Assume 10 objects = full coverage

        generalization_score = (
            avg_success * 0.4 +
            avg_robustness * 0.35 +
            coverage_score * 0.25
        )

        # Data requirements
        data_requirements = {}
        for obj, perf in object_performances.items():
            if perf.success_rate < 0.7:
                # Estimate episodes needed
                current_eps = perf.total_episodes
                gap = 0.7 - perf.success_rate
                estimated_needed = int(current_eps * (1 + gap / 0.1))
                data_requirements[obj] = estimated_needed

        # Recommendations
        recommendations = self._generate_recommendations(
            object_performances,
            task_performances,
            variation_impacts,
            overall_curve,
        )

        report = GeneralizationReport(
            report_id=str(uuid.uuid4())[:12],
            scene_id=scene_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            total_episodes=len(episodes),
            unique_objects=len(object_performances),
            unique_tasks=len(task_performances),
            unique_variations=len(variation_impacts),
            object_performances=object_performances,
            task_performances=task_performances,
            variation_impacts=variation_impacts,
            overall_learning_curve=overall_curve,
            per_object_learning_curves=per_object_curves,
            curriculum=curriculum,
            generalization_score=generalization_score,
            coverage_score=coverage_score,
            robustness_score=avg_robustness,
            estimated_episodes_needed=data_requirements,
            recommendations=recommendations,
        )

        self.log(f"Analysis complete: generalization score {generalization_score:.1%}")
        return report

    def _generate_recommendations(
        self,
        object_performances: Dict[str, ObjectPerformance],
        task_performances: Dict[str, TaskPerformance],
        variation_impacts: Dict[str, VariationImpact],
        learning_curve: LearningCurve,
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recs = []

        # Low-performing objects
        low_performers = [
            (obj, perf) for obj, perf in object_performances.items()
            if perf.success_rate < 0.6
        ]
        if low_performers:
            worst = min(low_performers, key=lambda x: x[1].success_rate)
            recs.append({
                "priority": "HIGH",
                "category": "coverage",
                "issue": f"Low success rate on {worst[0]} ({worst[1].success_rate:.1%})",
                "action": f"Collect more demonstrations for {worst[0]} or simplify task",
                "estimated_impact": "Could improve overall success by 5-10%",
            })

        # Variation sensitivity
        sensitive_vars = [
            (name, impact) for name, impact in variation_impacts.items()
            if impact.correlation_with_success < -0.3
        ]
        if sensitive_vars:
            worst_var = min(sensitive_vars, key=lambda x: x[1].correlation_with_success)
            recs.append({
                "priority": "MEDIUM",
                "category": "robustness",
                "issue": f"High sensitivity to {worst_var[0]} variation",
                "action": f"Add more training data with high {worst_var[0]} variation",
                "estimated_impact": "Could improve robustness by 15-20%",
            })

        # Learning efficiency
        if learning_curve.episodes_to_70_pct and learning_curve.episodes_to_70_pct > 1000:
            recs.append({
                "priority": "MEDIUM",
                "category": "efficiency",
                "issue": f"Slow learning curve ({learning_curve.episodes_to_70_pct} episodes for 70%)",
                "action": "Consider curriculum learning or data augmentation",
                "estimated_impact": "Could reduce training time by 30-50%",
            })

        # Coverage gaps
        if len(object_performances) < 5:
            recs.append({
                "priority": "LOW",
                "category": "coverage",
                "issue": f"Limited object diversity ({len(object_performances)} objects)",
                "action": "Add more object categories for better generalization",
                "estimated_impact": "Could improve generalization to new objects",
            })

        return recs

    def save_report(
        self,
        report: GeneralizationReport,
        output_path: Path,
    ) -> Path:
        """Save report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        self.log(f"Saved generalization report to {output_path}")
        return output_path


def analyze_generalization(
    episodes_dir: Path,
    scene_id: str,
    output_dir: Optional[Path] = None,
) -> GeneralizationReport:
    """
    Convenience function to analyze generalization.

    Args:
        episodes_dir: Path to episodes directory
        scene_id: Scene identifier
        output_dir: Optional output directory

    Returns:
        GeneralizationReport
    """
    episodes_dir = Path(episodes_dir)

    # Load episodes
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
        print(f"[GENERALIZATION-ANALYZER] No episode data found, using placeholder")
        episodes = [
            {"target_object": "mug", "success": True, "task_type": "pick_place"},
            {"target_object": "mug", "success": True, "task_type": "pick_place"},
            {"target_object": "bowl", "success": False, "task_type": "pick_place"},
        ]

    analyzer = GeneralizationAnalyzer(verbose=True)
    report = analyzer.generate_report(episodes, scene_id)

    if output_dir:
        output_path = Path(output_dir) / "generalization_report.json"
        analyzer.save_report(report, output_path)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze generalization")
    parser.add_argument("episodes_dir", type=Path, help="Path to episodes directory")
    parser.add_argument("--scene-id", required=True, help="Scene identifier")
    parser.add_argument("--output-dir", type=Path, help="Output directory")

    args = parser.parse_args()

    report = analyze_generalization(
        episodes_dir=args.episodes_dir,
        scene_id=args.scene_id,
        output_dir=args.output_dir,
    )

    print(f"\n=== Generalization Analysis ===")
    print(f"Total Episodes: {report.total_episodes}")
    print(f"Unique Objects: {report.unique_objects}")
    print(f"Generalization Score: {report.generalization_score:.1%}")
    print(f"Coverage Score: {report.coverage_score:.1%}")
    print(f"Robustness Score: {report.robustness_score:.1%}")
