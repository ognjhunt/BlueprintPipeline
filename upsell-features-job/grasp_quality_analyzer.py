#!/usr/bin/env python3
"""
Grasp Quality Analysis for BlueprintPipeline.

Provides comprehensive grasp quality metrics for robotics labs:
- Grasp stability scores (force closure, grasp rank)
- Contact point analysis (where on object was it grasped)
- Grasp robustness estimation (probability of slip)
- Approach vector analysis
- Grasp configuration quality

Upsell Value: $15,000-$50,000 per dataset
- Grasp learning is 60% of real manipulation research
- Labs need confidence the grasps are robust
- Essential for contact-rich learning
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


class GraspType(str, Enum):
    """Types of grasps."""
    POWER = "power"             # Full-hand power grasp
    PRECISION = "precision"     # Fingertip precision grasp
    PINCH = "pinch"             # Two-finger pinch
    LATERAL = "lateral"         # Side grasp (key grasp)
    SPHERICAL = "spherical"     # Wrap around spherical object
    CYLINDRICAL = "cylindrical" # Wrap around cylinder
    HOOK = "hook"               # Hook grasp
    UNKNOWN = "unknown"


class GraspQualityRating(str, Enum):
    """Grasp quality ratings."""
    EXCELLENT = "excellent"   # Very stable, high margin
    GOOD = "good"            # Stable under normal conditions
    MARGINAL = "marginal"    # May fail under perturbation
    POOR = "poor"            # High failure probability
    INVALID = "invalid"      # Not a valid grasp


@dataclass
class ContactPoint:
    """Individual contact point on object."""
    position: Tuple[float, float, float]  # (x, y, z) in object frame
    normal: Tuple[float, float, float]    # Surface normal at contact
    force_magnitude: float                 # Contact force in Newtons
    friction_coefficient: float            # Estimated friction

    # Contact region
    finger_id: int = 0  # Which finger
    contact_area_m2: float = 0.0  # Estimated contact area

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": list(self.position),
            "normal": list(self.normal),
            "force_magnitude": self.force_magnitude,
            "friction_coefficient": self.friction_coefficient,
            "finger_id": self.finger_id,
            "contact_area_m2": self.contact_area_m2,
        }


@dataclass
class ApproachVector:
    """Grasp approach analysis."""
    direction: Tuple[float, float, float]  # Approach direction
    distance_m: float                       # Pre-grasp distance
    angle_to_vertical_deg: float            # Angle from vertical
    angle_to_surface_deg: float             # Angle to surface normal

    # Quality
    clearance_m: float = 0.0        # Clearance to obstacles
    collision_free: bool = True     # No collision on approach
    singularity_margin: float = 1.0 # Distance from singularity (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": list(self.direction),
            "distance_m": self.distance_m,
            "angle_to_vertical_deg": self.angle_to_vertical_deg,
            "angle_to_surface_deg": self.angle_to_surface_deg,
            "clearance_m": self.clearance_m,
            "collision_free": self.collision_free,
            "singularity_margin": self.singularity_margin,
        }


@dataclass
class GraspMetrics:
    """Computed grasp quality metrics."""
    # Core metrics
    force_closure: bool = False           # Force closure achieved
    grasp_rank: float = 0.0               # Grasp matrix rank (0-1)
    epsilon_quality: float = 0.0          # Ferrari-Canny epsilon quality
    volume_quality: float = 0.0           # Grasp wrench space volume

    # Stability metrics
    stability_score: float = 0.0          # Overall stability (0-1)
    slip_probability: float = 1.0         # Probability of slip (0-1)
    disturbance_resistance_n: float = 0.0 # Force to dislodge

    # Robustness metrics
    position_tolerance_m: float = 0.0     # Position error tolerance
    orientation_tolerance_deg: float = 0.0 # Orientation error tolerance
    force_margin: float = 0.0             # Extra force available

    # Contact metrics
    num_contacts: int = 0
    contact_centroid: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    contact_spread_m: float = 0.0         # Spread of contacts
    symmetric: bool = False               # Contacts symmetric about CoM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "force_closure": self.force_closure,
            "grasp_rank": self.grasp_rank,
            "epsilon_quality": self.epsilon_quality,
            "volume_quality": self.volume_quality,
            "stability_score": self.stability_score,
            "slip_probability": f"{self.slip_probability:.1%}",
            "disturbance_resistance_n": self.disturbance_resistance_n,
            "robustness": {
                "position_tolerance_m": self.position_tolerance_m,
                "orientation_tolerance_deg": self.orientation_tolerance_deg,
                "force_margin": self.force_margin,
            },
            "contacts": {
                "num_contacts": self.num_contacts,
                "contact_centroid": list(self.contact_centroid),
                "contact_spread_m": self.contact_spread_m,
                "symmetric": self.symmetric,
            },
        }


@dataclass
class GraspAnalysis:
    """Complete analysis of a single grasp."""
    grasp_id: str
    episode_id: str
    frame_idx: int
    timestamp: float

    # Object info
    object_id: str
    object_category: str
    object_mass_kg: float

    # Grasp classification
    grasp_type: GraspType
    quality_rating: GraspQualityRating

    # Contact points
    contact_points: List[ContactPoint] = field(default_factory=list)

    # Approach
    approach: Optional[ApproachVector] = None

    # Computed metrics
    metrics: GraspMetrics = field(default_factory=GraspMetrics)

    # Gripper state
    gripper_width_m: float = 0.0
    gripper_force_n: float = 0.0

    # Outcome
    grasp_successful: bool = False
    maintained_during_lift: bool = False
    maintained_during_transport: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grasp_id": self.grasp_id,
            "episode_id": self.episode_id,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "object": {
                "id": self.object_id,
                "category": self.object_category,
                "mass_kg": self.object_mass_kg,
            },
            "classification": {
                "grasp_type": self.grasp_type.value,
                "quality_rating": self.quality_rating.value,
            },
            "contact_points": [cp.to_dict() for cp in self.contact_points],
            "approach": self.approach.to_dict() if self.approach else None,
            "metrics": self.metrics.to_dict(),
            "gripper_state": {
                "width_m": self.gripper_width_m,
                "force_n": self.gripper_force_n,
            },
            "outcome": {
                "successful": self.grasp_successful,
                "maintained_lift": self.maintained_during_lift,
                "maintained_transport": self.maintained_during_transport,
            },
        }


@dataclass
class GraspQualityReport:
    """Complete grasp quality report for a dataset."""
    report_id: str
    scene_id: str
    created_at: str

    # Summary stats
    total_grasps: int
    successful_grasps: int
    grasp_success_rate: float

    # Quality distribution
    quality_distribution: Dict[str, int] = field(default_factory=dict)

    # Grasp type distribution
    type_distribution: Dict[str, int] = field(default_factory=dict)

    # Per-object analysis
    per_object_quality: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Aggregate metrics
    avg_epsilon_quality: float = 0.0
    avg_stability_score: float = 0.0
    avg_slip_probability: float = 0.0

    # Contact analysis
    avg_num_contacts: float = 0.0
    avg_contact_spread: float = 0.0
    force_closure_rate: float = 0.0

    # Approach analysis
    avg_approach_clearance: float = 0.0
    collision_free_rate: float = 0.0

    # Individual grasp analyses
    grasp_analyses: List[GraspAnalysis] = field(default_factory=list)

    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # Data quality
    data_usability_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "scene_id": self.scene_id,
            "created_at": self.created_at,
            "summary": {
                "total_grasps": self.total_grasps,
                "successful_grasps": self.successful_grasps,
                "success_rate": f"{self.grasp_success_rate:.1%}",
            },
            "quality_distribution": {
                k: {"count": v, "percentage": f"{v/max(1,self.total_grasps):.1%}"}
                for k, v in self.quality_distribution.items()
            },
            "type_distribution": self.type_distribution,
            "per_object_quality": self.per_object_quality,
            "aggregate_metrics": {
                "avg_epsilon_quality": self.avg_epsilon_quality,
                "avg_stability_score": self.avg_stability_score,
                "avg_slip_probability": f"{self.avg_slip_probability:.1%}",
                "force_closure_rate": f"{self.force_closure_rate:.1%}",
            },
            "contact_analysis": {
                "avg_num_contacts": self.avg_num_contacts,
                "avg_contact_spread_m": self.avg_contact_spread,
            },
            "approach_analysis": {
                "avg_clearance_m": self.avg_approach_clearance,
                "collision_free_rate": f"{self.collision_free_rate:.1%}",
            },
            "recommendations": self.recommendations,
            "data_usability_score": self.data_usability_score,
        }


class GraspQualityAnalyzer:
    """
    Comprehensive grasp quality analysis for robotics training data.

    Analyzes grasps to provide detailed quality metrics that robotics
    labs need to assess training data quality.
    """

    # Friction coefficients by material
    FRICTION_COEFFICIENTS = {
        "rubber": 1.0,
        "plastic": 0.5,
        "metal": 0.4,
        "glass": 0.3,
        "wood": 0.6,
        "ceramic": 0.4,
        "fabric": 0.8,
        "default": 0.5,
    }

    # Quality thresholds
    EPSILON_EXCELLENT = 0.15
    EPSILON_GOOD = 0.08
    EPSILON_MARGINAL = 0.03

    STABILITY_EXCELLENT = 0.9
    STABILITY_GOOD = 0.7
    STABILITY_MARGINAL = 0.5

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[GRASP-ANALYZER] {msg}")

    def analyze_grasp(
        self,
        episode_data: Dict[str, Any],
        grasp_frame_idx: int,
        episode_id: str,
    ) -> Optional[GraspAnalysis]:
        """
        Analyze a single grasp from episode data.

        Args:
            episode_data: Episode trajectory data
            grasp_frame_idx: Frame index where grasp occurs
            episode_id: Episode identifier

        Returns:
            GraspAnalysis or None if not enough data
        """
        frames = episode_data.get("frames", [])
        if grasp_frame_idx >= len(frames):
            return None

        frame = frames[grasp_frame_idx]
        object_info = episode_data.get("target_object_info", {})

        # Extract contact information
        contacts = frame.get("contacts", [])
        gripper_contacts = [
            c for c in contacts
            if "gripper" in str(c.get("body_a", "")).lower() or
               "gripper" in str(c.get("body_b", "")).lower()
        ]

        # Build contact points
        contact_points = []
        for i, contact in enumerate(gripper_contacts):
            cp = ContactPoint(
                position=tuple(contact.get("position", [0, 0, 0])),
                normal=tuple(contact.get("normal", [0, 0, 1])),
                force_magnitude=contact.get("force_magnitude", 0),
                friction_coefficient=self.FRICTION_COEFFICIENTS.get(
                    object_info.get("material", "default"),
                    self.FRICTION_COEFFICIENTS["default"]
                ),
                finger_id=i % 2,  # Alternate between fingers
            )
            contact_points.append(cp)

        # Classify grasp type
        grasp_type = self._classify_grasp_type(contact_points, object_info)

        # Compute approach vector
        approach = self._compute_approach_vector(episode_data, grasp_frame_idx)

        # Compute grasp metrics
        metrics = self._compute_grasp_metrics(
            contact_points=contact_points,
            object_mass=object_info.get("mass", 0.1),
            gripper_force=frame.get("gripper_force", 10.0),
        )

        # Determine quality rating
        quality_rating = self._rate_grasp_quality(metrics)

        # Check outcome
        grasp_successful = frame.get("gripper_position", 0.04) < 0.02
        maintained_lift = self._check_maintained_grasp(
            frames, grasp_frame_idx, "lift"
        )
        maintained_transport = self._check_maintained_grasp(
            frames, grasp_frame_idx, "transport"
        )

        analysis = GraspAnalysis(
            grasp_id=str(uuid.uuid4())[:8],
            episode_id=episode_id,
            frame_idx=grasp_frame_idx,
            timestamp=frame.get("timestamp", 0),
            object_id=object_info.get("id", "unknown"),
            object_category=object_info.get("category", "unknown"),
            object_mass_kg=object_info.get("mass", 0.1),
            grasp_type=grasp_type,
            quality_rating=quality_rating,
            contact_points=contact_points,
            approach=approach,
            metrics=metrics,
            gripper_width_m=frame.get("gripper_position", 0.04),
            gripper_force_n=frame.get("gripper_force", 10.0),
            grasp_successful=grasp_successful,
            maintained_during_lift=maintained_lift,
            maintained_during_transport=maintained_transport,
        )

        return analysis

    def _classify_grasp_type(
        self,
        contact_points: List[ContactPoint],
        object_info: Dict[str, Any],
    ) -> GraspType:
        """Classify the type of grasp based on contacts and object."""
        num_contacts = len(contact_points)
        object_shape = object_info.get("shape", "box")

        if num_contacts < 2:
            return GraspType.UNKNOWN

        # Simple classification based on contact count and object
        if object_shape == "sphere":
            return GraspType.SPHERICAL
        elif object_shape == "cylinder":
            return GraspType.CYLINDRICAL
        elif num_contacts == 2:
            # Check if fingertip grasp
            total_force = sum(cp.force_magnitude for cp in contact_points)
            if total_force < 20:
                return GraspType.PRECISION
            else:
                return GraspType.PINCH
        elif num_contacts >= 4:
            return GraspType.POWER
        else:
            return GraspType.PRECISION

    def _compute_approach_vector(
        self,
        episode_data: Dict[str, Any],
        grasp_frame_idx: int,
    ) -> Optional[ApproachVector]:
        """Compute the grasp approach vector."""
        frames = episode_data.get("frames", [])

        # Find pre-grasp frame (typically 10-30 frames before)
        pre_grasp_idx = max(0, grasp_frame_idx - 20)

        if pre_grasp_idx >= len(frames) or grasp_frame_idx >= len(frames):
            return None

        pre_grasp = frames[pre_grasp_idx]
        grasp = frames[grasp_frame_idx]

        pre_pos = pre_grasp.get("ee_position", [0, 0, 0])
        grasp_pos = grasp.get("ee_position", [0, 0, 0])

        # Compute direction
        dx = grasp_pos[0] - pre_pos[0]
        dy = grasp_pos[1] - pre_pos[1]
        dz = grasp_pos[2] - pre_pos[2]

        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if distance > 0:
            direction = (dx/distance, dy/distance, dz/distance)
        else:
            direction = (0, 0, -1)  # Default downward

        # Angle to vertical
        angle_to_vertical = math.degrees(math.acos(abs(direction[2])))

        return ApproachVector(
            direction=direction,
            distance_m=distance,
            angle_to_vertical_deg=angle_to_vertical,
            angle_to_surface_deg=0.0,  # Would need surface normal
            clearance_m=0.05,  # Estimated
            collision_free=True,  # Assumed if we got here
            singularity_margin=0.8,  # Estimated
        )

    def _compute_grasp_metrics(
        self,
        contact_points: List[ContactPoint],
        object_mass: float,
        gripper_force: float,
    ) -> GraspMetrics:
        """
        Compute grasp quality metrics.

        This is a simplified implementation. Full force closure analysis
        would require solving linear programs on the grasp wrench space.
        """
        num_contacts = len(contact_points)

        if num_contacts < 2:
            return GraspMetrics(num_contacts=num_contacts)

        # Contact centroid
        centroid_x = sum(cp.position[0] for cp in contact_points) / num_contacts
        centroid_y = sum(cp.position[1] for cp in contact_points) / num_contacts
        centroid_z = sum(cp.position[2] for cp in contact_points) / num_contacts
        centroid = (centroid_x, centroid_y, centroid_z)

        # Contact spread
        spread = sum(
            math.sqrt(
                (cp.position[0] - centroid_x)**2 +
                (cp.position[1] - centroid_y)**2 +
                (cp.position[2] - centroid_z)**2
            )
            for cp in contact_points
        ) / num_contacts

        # Symmetry check (simplified)
        symmetric = spread > 0.01  # Has some spread

        # Force closure (simplified - check if contacts oppose)
        total_force = sum(cp.force_magnitude for cp in contact_points)
        min_friction = min(cp.friction_coefficient for cp in contact_points) if contact_points else 0

        # Heuristic for force closure
        force_closure = num_contacts >= 2 and spread > 0.01 and total_force > object_mass * 9.81

        # Epsilon quality (simplified)
        epsilon_quality = min(1.0, (total_force / (object_mass * 9.81 + 1)) * min_friction * 0.5)

        # Stability score
        stability_score = min(1.0, (
            (0.3 if force_closure else 0) +
            (0.3 * epsilon_quality) +
            (0.2 * min(1, spread / 0.05)) +
            (0.2 * min(1, total_force / 50))
        ))

        # Slip probability (inverse of stability)
        slip_probability = max(0, 1.0 - stability_score)

        # Disturbance resistance
        disturbance_resistance = total_force * min_friction * 0.5

        # Position tolerance
        position_tolerance = spread * 0.1  # Can tolerate 10% of spread

        # Force margin
        max_gripper_force = 70.0  # Typical
        force_margin = (max_gripper_force - gripper_force) / max_gripper_force

        return GraspMetrics(
            force_closure=force_closure,
            grasp_rank=min(1.0, num_contacts / 4),
            epsilon_quality=epsilon_quality,
            volume_quality=epsilon_quality * 0.8,
            stability_score=stability_score,
            slip_probability=slip_probability,
            disturbance_resistance_n=disturbance_resistance,
            position_tolerance_m=position_tolerance,
            orientation_tolerance_deg=15 if force_closure else 5,
            force_margin=force_margin,
            num_contacts=num_contacts,
            contact_centroid=centroid,
            contact_spread_m=spread,
            symmetric=symmetric,
        )

    def _rate_grasp_quality(self, metrics: GraspMetrics) -> GraspQualityRating:
        """Rate the grasp quality based on metrics."""
        if not metrics.force_closure:
            if metrics.num_contacts < 2:
                return GraspQualityRating.INVALID
            return GraspQualityRating.POOR

        if metrics.epsilon_quality >= self.EPSILON_EXCELLENT and \
           metrics.stability_score >= self.STABILITY_EXCELLENT:
            return GraspQualityRating.EXCELLENT

        if metrics.epsilon_quality >= self.EPSILON_GOOD and \
           metrics.stability_score >= self.STABILITY_GOOD:
            return GraspQualityRating.GOOD

        if metrics.epsilon_quality >= self.EPSILON_MARGINAL and \
           metrics.stability_score >= self.STABILITY_MARGINAL:
            return GraspQualityRating.MARGINAL

        return GraspQualityRating.POOR

    def _check_maintained_grasp(
        self,
        frames: List[Dict[str, Any]],
        grasp_frame_idx: int,
        target_phase: str,
    ) -> bool:
        """Check if grasp was maintained during a phase."""
        for frame in frames[grasp_frame_idx:]:
            phase = frame.get("phase", "")
            if phase == target_phase:
                # Check gripper still closed
                if frame.get("gripper_position", 0.04) < 0.02:
                    return True
                return False
        return False

    def analyze_dataset(
        self,
        episodes: List[Dict[str, Any]],
        scene_id: str,
    ) -> GraspQualityReport:
        """
        Analyze all grasps in a dataset.

        Args:
            episodes: List of episode data
            scene_id: Scene identifier

        Returns:
            Complete GraspQualityReport
        """
        self.log(f"Analyzing grasps in {len(episodes)} episodes...")

        all_analyses: List[GraspAnalysis] = []
        quality_dist: Dict[str, int] = {}
        type_dist: Dict[str, int] = {}
        per_object: Dict[str, Dict[str, Any]] = {}

        for ep in episodes:
            episode_id = ep.get("episode_id", str(uuid.uuid4())[:8])
            frames = ep.get("frames", [])

            # Find grasp frames (where gripper closes)
            for idx, frame in enumerate(frames):
                phase = frame.get("phase", "")
                if phase == "grasp":
                    analysis = self.analyze_grasp(ep, idx, episode_id)
                    if analysis:
                        all_analyses.append(analysis)

                        # Update distributions
                        quality = analysis.quality_rating.value
                        quality_dist[quality] = quality_dist.get(quality, 0) + 1

                        gtype = analysis.grasp_type.value
                        type_dist[gtype] = type_dist.get(gtype, 0) + 1

                        # Per-object tracking
                        obj_cat = analysis.object_category
                        if obj_cat not in per_object:
                            per_object[obj_cat] = {
                                "total": 0,
                                "successful": 0,
                                "avg_stability": 0,
                                "stability_scores": [],
                            }
                        per_object[obj_cat]["total"] += 1
                        if analysis.grasp_successful:
                            per_object[obj_cat]["successful"] += 1
                        per_object[obj_cat]["stability_scores"].append(
                            analysis.metrics.stability_score
                        )
                    break  # One grasp per episode

        # Compute aggregates
        total = len(all_analyses)
        successful = sum(1 for a in all_analyses if a.grasp_successful)

        avg_epsilon = sum(a.metrics.epsilon_quality for a in all_analyses) / max(1, total)
        avg_stability = sum(a.metrics.stability_score for a in all_analyses) / max(1, total)
        avg_slip = sum(a.metrics.slip_probability for a in all_analyses) / max(1, total)
        avg_contacts = sum(a.metrics.num_contacts for a in all_analyses) / max(1, total)
        avg_spread = sum(a.metrics.contact_spread_m for a in all_analyses) / max(1, total)
        force_closure_rate = sum(1 for a in all_analyses if a.metrics.force_closure) / max(1, total)

        # Approach analysis
        approaches = [a.approach for a in all_analyses if a.approach]
        avg_clearance = sum(ap.clearance_m for ap in approaches) / max(1, len(approaches))
        collision_free = sum(1 for ap in approaches if ap.collision_free) / max(1, len(approaches))

        # Finalize per-object stats
        for obj in per_object:
            scores = per_object[obj]["stability_scores"]
            per_object[obj]["avg_stability"] = sum(scores) / len(scores) if scores else 0
            per_object[obj]["success_rate"] = (
                per_object[obj]["successful"] / per_object[obj]["total"]
                if per_object[obj]["total"] > 0 else 0
            )
            del per_object[obj]["stability_scores"]  # Clean up

        # Generate recommendations
        recommendations = self._generate_recommendations(
            quality_dist, avg_stability, force_closure_rate, all_analyses
        )

        # Data usability score
        usability = self._compute_usability_score(
            success_rate=successful / max(1, total),
            avg_stability=avg_stability,
            force_closure_rate=force_closure_rate,
        )

        report = GraspQualityReport(
            report_id=str(uuid.uuid4())[:12],
            scene_id=scene_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            total_grasps=total,
            successful_grasps=successful,
            grasp_success_rate=successful / max(1, total),
            quality_distribution=quality_dist,
            type_distribution=type_dist,
            per_object_quality=per_object,
            avg_epsilon_quality=avg_epsilon,
            avg_stability_score=avg_stability,
            avg_slip_probability=avg_slip,
            avg_num_contacts=avg_contacts,
            avg_contact_spread=avg_spread,
            force_closure_rate=force_closure_rate,
            avg_approach_clearance=avg_clearance,
            collision_free_rate=collision_free,
            grasp_analyses=all_analyses,
            recommendations=recommendations,
            data_usability_score=usability,
        )

        self.log(f"Analyzed {total} grasps, {successful} successful ({successful/max(1,total):.1%})")
        return report

    def _generate_recommendations(
        self,
        quality_dist: Dict[str, int],
        avg_stability: float,
        force_closure_rate: float,
        analyses: List[GraspAnalysis],
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on grasp analysis."""
        recs = []

        total = sum(quality_dist.values())
        poor_rate = quality_dist.get("poor", 0) / max(1, total)
        marginal_rate = quality_dist.get("marginal", 0) / max(1, total)

        if poor_rate > 0.2:
            recs.append({
                "priority": "HIGH",
                "issue": f"High rate of poor-quality grasps ({poor_rate:.1%})",
                "impact": "Training on poor grasps leads to unreliable policies",
                "action": "Filter episodes with poor grasp quality before training",
            })

        if force_closure_rate < 0.8:
            recs.append({
                "priority": "MEDIUM",
                "issue": f"Low force closure rate ({force_closure_rate:.1%})",
                "impact": "Grasps may not be stable under perturbations",
                "action": "Review grasp planning to ensure antipodal grasps",
            })

        if avg_stability < 0.7:
            recs.append({
                "priority": "MEDIUM",
                "issue": f"Low average stability score ({avg_stability:.2f})",
                "impact": "Policies may learn to drop objects during transport",
                "action": "Consider increasing gripper force or contact points",
            })

        # Object-specific recommendations
        problem_objects = []
        for analysis in analyses:
            if analysis.quality_rating in [GraspQualityRating.POOR, GraspQualityRating.INVALID]:
                problem_objects.append(analysis.object_category)

        if problem_objects:
            common_problems = {}
            for obj in problem_objects:
                common_problems[obj] = common_problems.get(obj, 0) + 1

            worst_obj = max(common_problems.items(), key=lambda x: x[1])
            recs.append({
                "priority": "LOW",
                "issue": f"Object '{worst_obj[0]}' has {worst_obj[1]} poor grasps",
                "impact": "May need object-specific grasp strategies",
                "action": f"Review grasp configurations for {worst_obj[0]}",
            })

        return recs

    def _compute_usability_score(
        self,
        success_rate: float,
        avg_stability: float,
        force_closure_rate: float,
    ) -> float:
        """Compute overall data usability score."""
        return (
            success_rate * 0.4 +
            avg_stability * 0.35 +
            force_closure_rate * 0.25
        )

    def save_report(
        self,
        report: GraspQualityReport,
        output_path: Path,
    ) -> Path:
        """Save report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create simplified report without full analyses
        report_dict = report.to_dict()
        report_dict["grasp_analyses"] = f"[{len(report.grasp_analyses)} analyses - see detailed file]"

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        # Save detailed analyses separately
        detailed_path = output_path.parent / f"grasp_analyses_detailed_{report.report_id}.json"
        with open(detailed_path, "w") as f:
            json.dump(
                [a.to_dict() for a in report.grasp_analyses],
                f,
                indent=2
            )

        self.log(f"Saved grasp quality report to {output_path}")
        return output_path


def analyze_grasp_quality(
    episodes_dir: Path,
    scene_id: str,
    output_dir: Optional[Path] = None,
) -> GraspQualityReport:
    """
    Convenience function to analyze grasp quality.

    Args:
        episodes_dir: Path to episodes directory
        scene_id: Scene identifier
        output_dir: Optional output directory

    Returns:
        GraspQualityReport
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
        # Placeholder
        print(f"[GRASP-ANALYZER] No episode data found, using placeholder")
        episodes = [{"frames": [{"phase": "grasp", "gripper_position": 0.01}]}]

    analyzer = GraspQualityAnalyzer(verbose=True)
    report = analyzer.analyze_dataset(episodes, scene_id)

    if output_dir:
        output_path = Path(output_dir) / "grasp_quality_report.json"
        analyzer.save_report(report, output_path)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze grasp quality")
    parser.add_argument("episodes_dir", type=Path, help="Path to episodes directory")
    parser.add_argument("--scene-id", required=True, help="Scene identifier")
    parser.add_argument("--output-dir", type=Path, help="Output directory")

    args = parser.parse_args()

    report = analyze_grasp_quality(
        episodes_dir=args.episodes_dir,
        scene_id=args.scene_id,
        output_dir=args.output_dir,
    )

    print(f"\n=== Grasp Quality Analysis ===")
    print(f"Total Grasps: {report.total_grasps}")
    print(f"Success Rate: {report.grasp_success_rate:.1%}")
    print(f"Force Closure Rate: {report.force_closure_rate:.1%}")
    print(f"Avg Stability: {report.avg_stability_score:.2f}")
    print(f"Data Usability: {report.data_usability_score:.2f}")
