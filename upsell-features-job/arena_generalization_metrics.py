"""
Arena Generalization Metrics Capture Module
============================================

Captures object diversity coverage and generalization metrics from
Isaac Lab Arena evaluations - currently NOT captured in standard pipeline.

Premium Analytics Feature - Upsell Value: $15,000 - $35,000

Features:
- Object diversity coverage analysis
- Pose/lighting/clutter variation tracking
- Generalization score computation
- Coverage gap identification
- Training recommendation generation

Author: BlueprintPipeline Premium Analytics
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from datetime import datetime
from pathlib import Path
import statistics
import hashlib


class VariationType(Enum):
    """Types of variations in evaluation."""
    OBJECT_IDENTITY = "object_identity"
    OBJECT_POSE = "object_pose"
    OBJECT_SCALE = "object_scale"
    LIGHTING_INTENSITY = "lighting_intensity"
    LIGHTING_COLOR = "lighting_color"
    LIGHTING_DIRECTION = "lighting_direction"
    CLUTTER_LEVEL = "clutter_level"
    CLUTTER_OBJECTS = "clutter_objects"
    BACKGROUND = "background"
    CAMERA_POSE = "camera_pose"
    ROBOT_POSE = "robot_pose"


class DifficultyLevel(Enum):
    """Difficulty classification for variations."""
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class GeneralizationDimension(Enum):
    """Dimensions of generalization being tested."""
    OBJECT = "object"
    SPATIAL = "spatial"
    VISUAL = "visual"
    ENVIRONMENTAL = "environmental"
    COMPOSITIONAL = "compositional"


@dataclass
class ObjectVariation:
    """Tracks a specific object variation in evaluation."""
    object_id: str
    object_category: str
    object_name: str

    # Physical properties
    scale: float = 1.0
    mass_kg: float = 0.1
    friction: float = 0.5

    # Geometry properties
    geometry_type: str = "mesh"  # mesh, primitive, procedural
    vertex_count: int = 0
    has_texture: bool = True

    # Difficulty factors
    graspability_score: float = 0.5  # 0 = hard to grasp, 1 = easy
    stability_score: float = 0.5  # 0 = unstable, 1 = stable
    occlusion_score: float = 0.0  # 0 = fully visible, 1 = heavily occluded


@dataclass
class PoseVariation:
    """Tracks object pose variations."""
    variation_id: str
    position_x: float
    position_y: float
    position_z: float
    rotation_roll: float
    rotation_pitch: float
    rotation_yaw: float

    # Difficulty classification
    distance_from_center: float = 0.0
    height_above_surface: float = 0.0
    stability_risk: float = 0.0  # Risk of falling over
    reachability_score: float = 1.0  # 0 = hard to reach, 1 = easy


@dataclass
class LightingVariation:
    """Tracks lighting condition variations."""
    variation_id: str
    intensity: float  # 0-1
    color_temperature: int  # Kelvin
    direction: List[float]  # Unit vector
    ambient_ratio: float  # Ambient vs directional
    num_light_sources: int
    has_shadows: bool
    shadow_softness: float


@dataclass
class ClutterVariation:
    """Tracks scene clutter variations."""
    variation_id: str
    num_distractor_objects: int
    distractor_categories: List[str]
    occlusion_percentage: float
    workspace_coverage: float  # % of workspace occupied
    semantic_similarity: float  # How similar distractors are to target


@dataclass
class EpisodeGeneralizationData:
    """Generalization data for a single episode."""
    episode_id: str
    policy_id: str
    task_name: str

    # Variations applied
    object_variation: Optional[ObjectVariation] = None
    pose_variation: Optional[PoseVariation] = None
    lighting_variation: Optional[LightingVariation] = None
    clutter_variation: Optional[ClutterVariation] = None

    # Outcome
    success: bool = False
    reward: float = 0.0
    episode_length: int = 0

    # Difficulty assessment
    overall_difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    difficulty_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class CoverageMetrics:
    """Coverage metrics for a variation dimension."""
    dimension: str
    total_variations_tested: int
    unique_variations: int
    success_rate_by_variation: Dict[str, float] = field(default_factory=dict)
    coverage_percentage: float = 0.0  # % of possible variations covered
    coverage_gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class GeneralizationReport:
    """Complete generalization analysis report."""
    report_id: str
    policy_id: str
    task_name: str
    generated_at: datetime

    # Overall scores
    overall_generalization_score: float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)

    # Coverage by dimension
    object_coverage: Optional[CoverageMetrics] = None
    pose_coverage: Optional[CoverageMetrics] = None
    lighting_coverage: Optional[CoverageMetrics] = None
    clutter_coverage: Optional[CoverageMetrics] = None

    # Detailed analysis
    success_rate_by_difficulty: Dict[str, float] = field(default_factory=dict)
    failure_patterns: List[Dict[str, Any]] = field(default_factory=list)
    strength_areas: List[str] = field(default_factory=list)
    weakness_areas: List[str] = field(default_factory=list)

    # Recommendations
    training_recommendations: List[str] = field(default_factory=list)
    evaluation_recommendations: List[str] = field(default_factory=list)


class ArenaGeneralizationMetrics:
    """
    Captures and analyzes generalization metrics from Isaac Lab Arena.

    This module fills the gap of object diversity coverage analysis
    that is NOT captured in the standard pipeline output.
    """

    def __init__(self, output_dir: str = "./generalization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.episodes: List[EpisodeGeneralizationData] = []
        self.object_registry: Dict[str, ObjectVariation] = {}
        self.pose_registry: Dict[str, PoseVariation] = {}
        self.lighting_registry: Dict[str, LightingVariation] = {}
        self.clutter_registry: Dict[str, ClutterVariation] = {}

        # Known object categories for coverage calculation
        self.known_categories = {
            "household": ["mug", "bowl", "plate", "cup", "bottle", "can", "box", "container"],
            "tools": ["hammer", "screwdriver", "wrench", "pliers", "scissors"],
            "food": ["apple", "banana", "orange", "bread", "egg"],
            "office": ["pen", "stapler", "tape", "notebook", "phone"],
            "toys": ["ball", "cube", "cylinder", "cone", "block"]
        }

    def register_object_variation(
        self,
        object_id: str,
        category: str,
        name: str,
        scale: float = 1.0,
        mass_kg: float = 0.1,
        friction: float = 0.5,
        geometry_type: str = "mesh",
        graspability: float = 0.5,
        stability: float = 0.5
    ) -> ObjectVariation:
        """Register an object variation for tracking."""
        variation = ObjectVariation(
            object_id=object_id,
            object_category=category,
            object_name=name,
            scale=scale,
            mass_kg=mass_kg,
            friction=friction,
            geometry_type=geometry_type,
            graspability_score=graspability,
            stability_score=stability
        )
        self.object_registry[object_id] = variation
        return variation

    def register_pose_variation(
        self,
        variation_id: str,
        position: List[float],
        rotation: List[float],
        reachability: float = 1.0
    ) -> PoseVariation:
        """Register a pose variation for tracking."""
        # Compute difficulty factors
        distance = np.linalg.norm(position[:2])  # XY distance from origin
        height = position[2] if len(position) > 2 else 0.0

        # Stability risk based on rotation
        roll, pitch = rotation[0], rotation[1]
        stability_risk = abs(roll) / 90.0 + abs(pitch) / 90.0

        variation = PoseVariation(
            variation_id=variation_id,
            position_x=position[0],
            position_y=position[1],
            position_z=position[2] if len(position) > 2 else 0.0,
            rotation_roll=rotation[0],
            rotation_pitch=rotation[1],
            rotation_yaw=rotation[2] if len(rotation) > 2 else 0.0,
            distance_from_center=float(distance),
            height_above_surface=height,
            stability_risk=min(1.0, stability_risk),
            reachability_score=reachability
        )
        self.pose_registry[variation_id] = variation
        return variation

    def register_lighting_variation(
        self,
        variation_id: str,
        intensity: float,
        color_temperature: int = 5500,
        direction: List[float] = None,
        ambient_ratio: float = 0.3,
        num_sources: int = 1,
        has_shadows: bool = True
    ) -> LightingVariation:
        """Register a lighting variation for tracking."""
        variation = LightingVariation(
            variation_id=variation_id,
            intensity=intensity,
            color_temperature=color_temperature,
            direction=direction or [0.0, -1.0, 0.0],
            ambient_ratio=ambient_ratio,
            num_light_sources=num_sources,
            has_shadows=has_shadows,
            shadow_softness=0.5
        )
        self.lighting_registry[variation_id] = variation
        return variation

    def register_clutter_variation(
        self,
        variation_id: str,
        num_distractors: int,
        distractor_categories: List[str],
        occlusion_pct: float = 0.0,
        workspace_coverage: float = 0.1,
        semantic_similarity: float = 0.3
    ) -> ClutterVariation:
        """Register a clutter variation for tracking."""
        variation = ClutterVariation(
            variation_id=variation_id,
            num_distractor_objects=num_distractors,
            distractor_categories=distractor_categories,
            occlusion_percentage=occlusion_pct,
            workspace_coverage=workspace_coverage,
            semantic_similarity=semantic_similarity
        )
        self.clutter_registry[variation_id] = variation
        return variation

    def record_episode(
        self,
        episode_id: str,
        policy_id: str,
        task_name: str,
        object_id: Optional[str] = None,
        pose_id: Optional[str] = None,
        lighting_id: Optional[str] = None,
        clutter_id: Optional[str] = None,
        success: bool = False,
        reward: float = 0.0,
        episode_length: int = 0
    ) -> EpisodeGeneralizationData:
        """Record an episode with its generalization data."""
        episode = EpisodeGeneralizationData(
            episode_id=episode_id,
            policy_id=policy_id,
            task_name=task_name,
            object_variation=self.object_registry.get(object_id),
            pose_variation=self.pose_registry.get(pose_id),
            lighting_variation=self.lighting_registry.get(lighting_id),
            clutter_variation=self.clutter_registry.get(clutter_id),
            success=success,
            reward=reward,
            episode_length=episode_length
        )

        # Compute difficulty
        episode.overall_difficulty = self._compute_difficulty(episode)
        episode.difficulty_factors = self._get_difficulty_factors(episode)

        self.episodes.append(episode)
        return episode

    def _compute_difficulty(
        self,
        episode: EpisodeGeneralizationData
    ) -> DifficultyLevel:
        """Compute overall difficulty of episode configuration."""
        score = 0.5  # Base medium difficulty

        if episode.object_variation:
            score -= (episode.object_variation.graspability_score - 0.5) * 0.3
            score -= (episode.object_variation.stability_score - 0.5) * 0.2

        if episode.pose_variation:
            score += episode.pose_variation.stability_risk * 0.2
            score -= (episode.pose_variation.reachability_score - 0.5) * 0.2

        if episode.lighting_variation:
            # Extreme lighting is harder
            intensity = episode.lighting_variation.intensity
            if intensity < 0.3 or intensity > 0.9:
                score += 0.1

        if episode.clutter_variation:
            score += episode.clutter_variation.occlusion_percentage * 0.2
            score += episode.clutter_variation.workspace_coverage * 0.1

        if score < 0.2:
            return DifficultyLevel.TRIVIAL
        elif score < 0.4:
            return DifficultyLevel.EASY
        elif score < 0.6:
            return DifficultyLevel.MEDIUM
        elif score < 0.8:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.EXPERT

    def _get_difficulty_factors(
        self,
        episode: EpisodeGeneralizationData
    ) -> Dict[str, float]:
        """Extract individual difficulty factors."""
        factors = {}

        if episode.object_variation:
            factors["object_graspability"] = 1.0 - episode.object_variation.graspability_score
            factors["object_stability"] = 1.0 - episode.object_variation.stability_score

        if episode.pose_variation:
            factors["pose_stability_risk"] = episode.pose_variation.stability_risk
            factors["pose_reachability"] = 1.0 - episode.pose_variation.reachability_score

        if episode.lighting_variation:
            intensity = episode.lighting_variation.intensity
            factors["lighting_challenge"] = abs(intensity - 0.6) / 0.4

        if episode.clutter_variation:
            factors["clutter_occlusion"] = episode.clutter_variation.occlusion_percentage
            factors["clutter_density"] = episode.clutter_variation.workspace_coverage

        return factors

    def compute_object_coverage(
        self,
        policy_id: str
    ) -> CoverageMetrics:
        """
        Compute object diversity coverage metrics.

        UPSELL VALUE: This is the KEY metric not captured in standard output.
        """
        relevant_episodes = [
            ep for ep in self.episodes
            if ep.policy_id == policy_id and ep.object_variation is not None
        ]

        if not relevant_episodes:
            return CoverageMetrics(
                dimension="object",
                total_variations_tested=0,
                unique_variations=0
            )

        # Track unique objects and categories
        unique_objects: Set[str] = set()
        unique_categories: Set[str] = set()
        success_by_object: Dict[str, List[bool]] = {}
        success_by_category: Dict[str, List[bool]] = {}

        for ep in relevant_episodes:
            obj = ep.object_variation
            unique_objects.add(obj.object_id)
            unique_categories.add(obj.object_category)

            if obj.object_id not in success_by_object:
                success_by_object[obj.object_id] = []
            success_by_object[obj.object_id].append(ep.success)

            if obj.object_category not in success_by_category:
                success_by_category[obj.object_category] = []
            success_by_category[obj.object_category].append(ep.success)

        # Compute success rates
        success_rates = {}
        for obj_id, outcomes in success_by_object.items():
            success_rates[obj_id] = sum(outcomes) / len(outcomes) if outcomes else 0.0

        # Compute coverage
        total_known = sum(len(objs) for objs in self.known_categories.values())
        coverage_pct = len(unique_objects) / total_known if total_known > 0 else 0.0

        # Identify gaps
        gaps = []
        for category, objects in self.known_categories.items():
            if category not in unique_categories:
                gaps.append(f"No objects from category: {category}")
            else:
                tested_in_category = [
                    obj_id for obj_id, obj in self.object_registry.items()
                    if obj.object_category == category
                ]
                missing = set(objects) - set(obj.object_name for obj_id, obj in self.object_registry.items() if obj.object_category == category)
                if missing:
                    gaps.append(f"Missing objects in {category}: {list(missing)[:3]}")

        # Recommendations
        recommendations = []
        low_success_objects = [obj_id for obj_id, rate in success_rates.items() if rate < 0.5]
        if low_success_objects:
            recommendations.append(f"Add training data for low-success objects: {low_success_objects[:5]}")

        if coverage_pct < 0.3:
            recommendations.append("Expand object diversity - current coverage is below 30%")

        if len(unique_categories) < 3:
            recommendations.append("Add more object categories to test cross-category generalization")

        return CoverageMetrics(
            dimension="object",
            total_variations_tested=len(relevant_episodes),
            unique_variations=len(unique_objects),
            success_rate_by_variation=success_rates,
            coverage_percentage=coverage_pct,
            coverage_gaps=gaps,
            recommendations=recommendations
        )

    def compute_pose_coverage(
        self,
        policy_id: str
    ) -> CoverageMetrics:
        """Compute pose variation coverage metrics."""
        relevant_episodes = [
            ep for ep in self.episodes
            if ep.policy_id == policy_id and ep.pose_variation is not None
        ]

        if not relevant_episodes:
            return CoverageMetrics(
                dimension="pose",
                total_variations_tested=0,
                unique_variations=0
            )

        # Discretize pose space for coverage analysis
        pose_bins: Dict[str, List[bool]] = {}

        for ep in relevant_episodes:
            pose = ep.pose_variation

            # Bin by distance from center (near/mid/far)
            if pose.distance_from_center < 0.15:
                dist_bin = "near"
            elif pose.distance_from_center < 0.3:
                dist_bin = "mid"
            else:
                dist_bin = "far"

            # Bin by rotation (upright/tilted/inverted)
            max_tilt = max(abs(pose.rotation_roll), abs(pose.rotation_pitch))
            if max_tilt < 15:
                rot_bin = "upright"
            elif max_tilt < 45:
                rot_bin = "tilted"
            else:
                rot_bin = "extreme"

            bin_key = f"{dist_bin}_{rot_bin}"
            if bin_key not in pose_bins:
                pose_bins[bin_key] = []
            pose_bins[bin_key].append(ep.success)

        # Compute success rates per bin
        success_rates = {
            bin_key: sum(outcomes) / len(outcomes) if outcomes else 0.0
            for bin_key, outcomes in pose_bins.items()
        }

        # Expected bins (3 distance x 3 rotation = 9 bins)
        expected_bins = ["near_upright", "near_tilted", "near_extreme",
                        "mid_upright", "mid_tilted", "mid_extreme",
                        "far_upright", "far_tilted", "far_extreme"]

        coverage_pct = len(pose_bins) / len(expected_bins)

        # Identify gaps
        gaps = [b for b in expected_bins if b not in pose_bins]

        # Recommendations
        recommendations = []
        low_success_bins = [b for b, r in success_rates.items() if r < 0.5]
        if low_success_bins:
            recommendations.append(f"Focus training on pose configurations: {low_success_bins}")

        if "extreme" not in str(pose_bins.keys()):
            recommendations.append("Add extreme rotation variations to test robustness")

        if "far" not in str(pose_bins.keys()):
            recommendations.append("Add far-distance poses to test workspace limits")

        return CoverageMetrics(
            dimension="pose",
            total_variations_tested=len(relevant_episodes),
            unique_variations=len(pose_bins),
            success_rate_by_variation=success_rates,
            coverage_percentage=coverage_pct,
            coverage_gaps=gaps,
            recommendations=recommendations
        )

    def compute_lighting_coverage(
        self,
        policy_id: str
    ) -> CoverageMetrics:
        """Compute lighting variation coverage metrics."""
        relevant_episodes = [
            ep for ep in self.episodes
            if ep.policy_id == policy_id and ep.lighting_variation is not None
        ]

        if not relevant_episodes:
            return CoverageMetrics(
                dimension="lighting",
                total_variations_tested=0,
                unique_variations=0
            )

        # Bin lighting conditions
        lighting_bins: Dict[str, List[bool]] = {}

        for ep in relevant_episodes:
            light = ep.lighting_variation

            # Intensity bins
            if light.intensity < 0.3:
                int_bin = "dim"
            elif light.intensity < 0.7:
                int_bin = "normal"
            else:
                int_bin = "bright"

            # Color temperature bins
            if light.color_temperature < 4000:
                temp_bin = "warm"
            elif light.color_temperature < 6000:
                temp_bin = "neutral"
            else:
                temp_bin = "cool"

            bin_key = f"{int_bin}_{temp_bin}"
            if bin_key not in lighting_bins:
                lighting_bins[bin_key] = []
            lighting_bins[bin_key].append(ep.success)

        success_rates = {
            bin_key: sum(outcomes) / len(outcomes) if outcomes else 0.0
            for bin_key, outcomes in lighting_bins.items()
        }

        expected_bins = 9  # 3 intensity x 3 color temp
        coverage_pct = len(lighting_bins) / expected_bins

        gaps = []
        if "dim" not in str(lighting_bins.keys()):
            gaps.append("No low-light conditions tested")
        if "bright" not in str(lighting_bins.keys()):
            gaps.append("No high-light conditions tested")

        recommendations = []
        if coverage_pct < 0.5:
            recommendations.append("Expand lighting variations for visual robustness testing")

        dim_success = [r for b, r in success_rates.items() if "dim" in b]
        if dim_success and statistics.mean(dim_success) < 0.6:
            recommendations.append("Policy struggles in low-light - add dim lighting training data")

        return CoverageMetrics(
            dimension="lighting",
            total_variations_tested=len(relevant_episodes),
            unique_variations=len(lighting_bins),
            success_rate_by_variation=success_rates,
            coverage_percentage=coverage_pct,
            coverage_gaps=gaps,
            recommendations=recommendations
        )

    def compute_clutter_coverage(
        self,
        policy_id: str
    ) -> CoverageMetrics:
        """Compute clutter/distractor coverage metrics."""
        relevant_episodes = [
            ep for ep in self.episodes
            if ep.policy_id == policy_id and ep.clutter_variation is not None
        ]

        if not relevant_episodes:
            return CoverageMetrics(
                dimension="clutter",
                total_variations_tested=0,
                unique_variations=0
            )

        clutter_bins: Dict[str, List[bool]] = {}

        for ep in relevant_episodes:
            clutter = ep.clutter_variation

            # Number of distractors
            if clutter.num_distractor_objects == 0:
                count_bin = "none"
            elif clutter.num_distractor_objects <= 3:
                count_bin = "few"
            elif clutter.num_distractor_objects <= 7:
                count_bin = "moderate"
            else:
                count_bin = "heavy"

            # Occlusion level
            if clutter.occlusion_percentage < 0.1:
                occ_bin = "visible"
            elif clutter.occlusion_percentage < 0.3:
                occ_bin = "partial"
            else:
                occ_bin = "occluded"

            bin_key = f"{count_bin}_{occ_bin}"
            if bin_key not in clutter_bins:
                clutter_bins[bin_key] = []
            clutter_bins[bin_key].append(ep.success)

        success_rates = {
            bin_key: sum(outcomes) / len(outcomes) if outcomes else 0.0
            for bin_key, outcomes in clutter_bins.items()
        }

        expected_bins = 12  # 4 count x 3 occlusion
        coverage_pct = len(clutter_bins) / expected_bins

        gaps = []
        if "heavy" not in str(clutter_bins.keys()):
            gaps.append("No heavy clutter scenarios tested")
        if "occluded" not in str(clutter_bins.keys()):
            gaps.append("No significant occlusion scenarios tested")

        recommendations = []
        if "none_visible" in success_rates and success_rates.get("heavy_occluded", 0) < success_rates["none_visible"] * 0.5:
            recommendations.append("Policy degrades significantly with clutter - add cluttered training scenarios")

        return CoverageMetrics(
            dimension="clutter",
            total_variations_tested=len(relevant_episodes),
            unique_variations=len(clutter_bins),
            success_rate_by_variation=success_rates,
            coverage_percentage=coverage_pct,
            coverage_gaps=gaps,
            recommendations=recommendations
        )

    def compute_difficulty_analysis(
        self,
        policy_id: str
    ) -> Dict[str, Any]:
        """
        Analyze success rates by difficulty level.

        UPSELL VALUE: Understanding difficulty-performance relationship.
        """
        relevant_episodes = [ep for ep in self.episodes if ep.policy_id == policy_id]

        if not relevant_episodes:
            return {"error": "No episodes found for policy"}

        by_difficulty: Dict[str, List[bool]] = {}
        difficulty_factors: Dict[str, List[float]] = {}

        for ep in relevant_episodes:
            diff = ep.overall_difficulty.value
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(ep.success)

            for factor, value in ep.difficulty_factors.items():
                if factor not in difficulty_factors:
                    difficulty_factors[factor] = []
                difficulty_factors[factor].append((value, ep.success))

        success_by_difficulty = {
            diff: sum(outcomes) / len(outcomes) if outcomes else 0.0
            for diff, outcomes in by_difficulty.items()
        }

        # Compute correlation between difficulty factors and success
        factor_correlations = {}
        for factor, pairs in difficulty_factors.items():
            if len(pairs) > 10:
                values = [p[0] for p in pairs]
                successes = [1.0 if p[1] else 0.0 for p in pairs]
                # Simple correlation
                if np.std(values) > 0 and np.std(successes) > 0:
                    corr = np.corrcoef(values, successes)[0, 1]
                    factor_correlations[factor] = float(corr)

        return {
            "success_rate_by_difficulty": success_by_difficulty,
            "episode_count_by_difficulty": {
                diff: len(outcomes) for diff, outcomes in by_difficulty.items()
            },
            "factor_correlations": factor_correlations,
            "hardest_factors": sorted(
                factor_correlations.items(),
                key=lambda x: x[1] if not np.isnan(x[1]) else 0
            )[:3],
            "interpretation": self._interpret_difficulty_analysis(success_by_difficulty)
        }

    def _interpret_difficulty_analysis(
        self,
        success_by_difficulty: Dict[str, float]
    ) -> str:
        """Interpret difficulty analysis results."""
        if not success_by_difficulty:
            return "Insufficient data for difficulty analysis"

        trivial_easy = (success_by_difficulty.get("trivial", 0) + success_by_difficulty.get("easy", 0)) / 2
        hard_expert = (success_by_difficulty.get("hard", 0) + success_by_difficulty.get("expert", 0)) / 2

        if hard_expert >= trivial_easy * 0.8:
            return "Policy shows robust performance across difficulty levels"
        elif hard_expert >= trivial_easy * 0.5:
            return "Policy shows expected degradation with increased difficulty"
        else:
            return "Policy struggles significantly with harder variations - focused training recommended"

    def identify_failure_patterns(
        self,
        policy_id: str
    ) -> List[Dict[str, Any]]:
        """
        Identify patterns in failed episodes.

        UPSELL VALUE: Actionable insights for improvement.
        """
        failed_episodes = [
            ep for ep in self.episodes
            if ep.policy_id == policy_id and not ep.success
        ]

        if not failed_episodes:
            return []

        patterns = []

        # Analyze by difficulty factor
        factor_failure_rates: Dict[str, Dict[str, int]] = {}

        for ep in failed_episodes:
            for factor, value in ep.difficulty_factors.items():
                if factor not in factor_failure_rates:
                    factor_failure_rates[factor] = {"high": 0, "medium": 0, "low": 0}

                if value > 0.7:
                    factor_failure_rates[factor]["high"] += 1
                elif value > 0.3:
                    factor_failure_rates[factor]["medium"] += 1
                else:
                    factor_failure_rates[factor]["low"] += 1

        # Identify dominant patterns
        for factor, counts in factor_failure_rates.items():
            total = sum(counts.values())
            if counts["high"] / total > 0.5:
                patterns.append({
                    "pattern_type": "difficulty_factor",
                    "factor": factor,
                    "description": f"High {factor} associated with {counts['high']/total*100:.0f}% of failures",
                    "failure_count": counts["high"],
                    "recommendation": f"Focus training on scenarios with high {factor}"
                })

        # Analyze by object category
        category_failures: Dict[str, int] = {}
        for ep in failed_episodes:
            if ep.object_variation:
                cat = ep.object_variation.object_category
                category_failures[cat] = category_failures.get(cat, 0) + 1

        total_failures = len(failed_episodes)
        for cat, count in category_failures.items():
            if count / total_failures > 0.3:
                patterns.append({
                    "pattern_type": "object_category",
                    "category": cat,
                    "description": f"{count} failures ({count/total_failures*100:.0f}%) on {cat} objects",
                    "failure_count": count,
                    "recommendation": f"Add more training examples for {cat} category"
                })

        return patterns

    def generate_report(
        self,
        policy_id: str,
        task_name: str
    ) -> GeneralizationReport:
        """Generate comprehensive generalization analysis report."""
        report_id = hashlib.sha256(
            f"{policy_id}_{task_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        object_coverage = self.compute_object_coverage(policy_id)
        pose_coverage = self.compute_pose_coverage(policy_id)
        lighting_coverage = self.compute_lighting_coverage(policy_id)
        clutter_coverage = self.compute_clutter_coverage(policy_id)
        difficulty_analysis = self.compute_difficulty_analysis(policy_id)
        failure_patterns = self.identify_failure_patterns(policy_id)

        # Compute dimension scores
        dimension_scores = {
            "object": object_coverage.coverage_percentage if object_coverage else 0.0,
            "pose": pose_coverage.coverage_percentage if pose_coverage else 0.0,
            "lighting": lighting_coverage.coverage_percentage if lighting_coverage else 0.0,
            "clutter": clutter_coverage.coverage_percentage if clutter_coverage else 0.0
        }

        # Overall score
        overall_score = statistics.mean(dimension_scores.values()) if dimension_scores else 0.0

        # Identify strengths and weaknesses
        all_success_rates = []
        if object_coverage:
            all_success_rates.extend(object_coverage.success_rate_by_variation.items())
        if pose_coverage:
            all_success_rates.extend(pose_coverage.success_rate_by_variation.items())

        sorted_rates = sorted(all_success_rates, key=lambda x: x[1], reverse=True)
        strengths = [item[0] for item in sorted_rates[:5] if item[1] > 0.8]
        weaknesses = [item[0] for item in sorted_rates[-5:] if item[1] < 0.5]

        # Aggregate recommendations
        training_recs = []
        eval_recs = []

        for coverage in [object_coverage, pose_coverage, lighting_coverage, clutter_coverage]:
            if coverage:
                training_recs.extend(coverage.recommendations)
                if coverage.coverage_gaps:
                    eval_recs.append(f"Expand {coverage.dimension} testing: {coverage.coverage_gaps[0]}")

        report = GeneralizationReport(
            report_id=report_id,
            policy_id=policy_id,
            task_name=task_name,
            generated_at=datetime.now(),
            overall_generalization_score=overall_score,
            dimension_scores=dimension_scores,
            object_coverage=object_coverage,
            pose_coverage=pose_coverage,
            lighting_coverage=lighting_coverage,
            clutter_coverage=clutter_coverage,
            success_rate_by_difficulty=difficulty_analysis.get("success_rate_by_difficulty", {}),
            failure_patterns=failure_patterns,
            strength_areas=strengths,
            weakness_areas=weaknesses,
            training_recommendations=training_recs,
            evaluation_recommendations=eval_recs
        )

        return report

    def save_report(self, report: GeneralizationReport) -> str:
        """Save generalization report to JSON file."""
        output_file = self.output_dir / f"generalization_{report.report_id}.json"

        data = {
            "report_id": report.report_id,
            "policy_id": report.policy_id,
            "task_name": report.task_name,
            "generated_at": report.generated_at.isoformat(),
            "overall_generalization_score": report.overall_generalization_score,
            "dimension_scores": report.dimension_scores,
            "coverage_metrics": {
                "object": self._coverage_to_dict(report.object_coverage),
                "pose": self._coverage_to_dict(report.pose_coverage),
                "lighting": self._coverage_to_dict(report.lighting_coverage),
                "clutter": self._coverage_to_dict(report.clutter_coverage)
            },
            "success_rate_by_difficulty": report.success_rate_by_difficulty,
            "failure_patterns": report.failure_patterns,
            "strength_areas": report.strength_areas,
            "weakness_areas": report.weakness_areas,
            "training_recommendations": report.training_recommendations,
            "evaluation_recommendations": report.evaluation_recommendations
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        return str(output_file)

    def _coverage_to_dict(self, coverage: Optional[CoverageMetrics]) -> Optional[Dict[str, Any]]:
        """Convert coverage metrics to dict."""
        if not coverage:
            return None

        return {
            "dimension": coverage.dimension,
            "total_variations_tested": coverage.total_variations_tested,
            "unique_variations": coverage.unique_variations,
            "coverage_percentage": coverage.coverage_percentage,
            "success_rate_by_variation": coverage.success_rate_by_variation,
            "coverage_gaps": coverage.coverage_gaps,
            "recommendations": coverage.recommendations
        }

    def generate_premium_report(
        self,
        policy_id: str,
        task_name: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive premium generalization report.

        KEY UPSELL VALUE - Object diversity coverage analysis.
        """
        report = self.generate_report(policy_id, task_name)
        difficulty_analysis = self.compute_difficulty_analysis(policy_id)

        premium_report = {
            "report_type": "arena_generalization_premium",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {
                "policy_id": policy_id,
                "task_name": task_name,
                "overall_score": f"{report.overall_generalization_score * 100:.0f}%",
                "verdict": self._get_verdict(report.overall_generalization_score),
                "key_strengths": report.strength_areas[:3],
                "key_weaknesses": report.weakness_areas[:3]
            },
            "coverage_analysis": {
                "object_diversity": {
                    "score": f"{report.dimension_scores.get('object', 0) * 100:.0f}%",
                    "unique_objects_tested": report.object_coverage.unique_variations if report.object_coverage else 0,
                    "gaps": report.object_coverage.coverage_gaps[:3] if report.object_coverage else []
                },
                "pose_diversity": {
                    "score": f"{report.dimension_scores.get('pose', 0) * 100:.0f}%",
                    "configurations_tested": report.pose_coverage.unique_variations if report.pose_coverage else 0,
                    "gaps": report.pose_coverage.coverage_gaps[:3] if report.pose_coverage else []
                },
                "visual_diversity": {
                    "score": f"{report.dimension_scores.get('lighting', 0) * 100:.0f}%",
                    "conditions_tested": report.lighting_coverage.unique_variations if report.lighting_coverage else 0,
                    "gaps": report.lighting_coverage.coverage_gaps[:3] if report.lighting_coverage else []
                },
                "clutter_diversity": {
                    "score": f"{report.dimension_scores.get('clutter', 0) * 100:.0f}%",
                    "scenarios_tested": report.clutter_coverage.unique_variations if report.clutter_coverage else 0,
                    "gaps": report.clutter_coverage.coverage_gaps[:3] if report.clutter_coverage else []
                }
            },
            "difficulty_analysis": {
                "success_by_difficulty": difficulty_analysis.get("success_rate_by_difficulty", {}),
                "hardest_factors": difficulty_analysis.get("hardest_factors", []),
                "interpretation": difficulty_analysis.get("interpretation", "")
            },
            "failure_analysis": {
                "total_failures": len([ep for ep in self.episodes if ep.policy_id == policy_id and not ep.success]),
                "patterns_identified": len(report.failure_patterns),
                "top_patterns": report.failure_patterns[:3]
            },
            "recommendations": {
                "training": report.training_recommendations[:5],
                "evaluation": report.evaluation_recommendations[:5],
                "priority_actions": self._get_priority_actions(report)
            },
            "upsell_opportunities": [
                "Run failure mode analysis for detailed root cause investigation",
                "Perform embodiment transfer analysis for multi-robot deployment",
                "Generate sim2real fidelity matrix for real-world deployment confidence",
                "Conduct trajectory optimality analysis for efficiency improvements"
            ]
        }

        return premium_report

    def _get_verdict(self, score: float) -> str:
        """Get overall verdict based on generalization score."""
        if score >= 0.8:
            return "Excellent generalization - ready for diverse deployment"
        elif score >= 0.6:
            return "Good generalization - some gaps to address"
        elif score >= 0.4:
            return "Moderate generalization - significant training needed"
        else:
            return "Limited generalization - major improvements required"

    def _get_priority_actions(self, report: GeneralizationReport) -> List[str]:
        """Get top priority actions based on report."""
        actions = []

        # Lowest scoring dimension
        if report.dimension_scores:
            lowest = min(report.dimension_scores.items(), key=lambda x: x[1])
            actions.append(f"Priority: Improve {lowest[0]} coverage (currently {lowest[1]*100:.0f}%)")

        # Most impactful failure pattern
        if report.failure_patterns:
            top_pattern = report.failure_patterns[0]
            actions.append(f"Address failure pattern: {top_pattern.get('description', 'Unknown')}")

        # Worst performing variation
        if report.weakness_areas:
            actions.append(f"Focus on improving: {report.weakness_areas[0]}")

        return actions[:3]


def create_arena_generalization_metrics(
    output_dir: str = "./generalization"
) -> ArenaGeneralizationMetrics:
    """Factory function to create ArenaGeneralizationMetrics instance."""
    return ArenaGeneralizationMetrics(output_dir=output_dir)
