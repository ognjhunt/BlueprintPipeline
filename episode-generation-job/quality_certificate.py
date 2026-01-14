#!/usr/bin/env python3
"""
Quality Certificate for Generated Episodes.

This module generates comprehensive quality certificates for all generated episodes,
making data quality transparent to customers and enabling quality-based filtering.

A quality certificate includes:
1. Data source information (sensor backend, physics backend)
2. Quality metrics (trajectory, visual, task, diversity)
3. Validation results
4. Training suitability assessment
5. Confidence scores

This enables customers to:
- Filter datasets by quality threshold
- Understand data limitations
- Make informed decisions about training
- Debug data quality issues
"""

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add parent to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quality_constants import (
    MIN_QUALITY_SCORE,
    PRODUCTION_TRAINING_THRESHOLD,
    FINE_TUNING_THRESHOLD,
    get_training_suitability_level,
)
from isaac_sim_enforcement import (
    DataQualityLevel,
    EnvironmentCapabilities,
    PhysicsValidationBackend,
    SensorSource,
)


# =============================================================================
# Quality Metrics
# =============================================================================


@dataclass
class TrajectoryQualityMetrics:
    """Metrics for trajectory quality assessment."""

    # Smoothness metrics
    smoothness_score: float = 0.0  # 0-1, higher is smoother (based on jerk)
    mean_jerk: float = 0.0  # Mean jerk across trajectory
    max_jerk: float = 0.0  # Maximum jerk value

    # Efficiency metrics
    path_efficiency: float = 0.0  # Actual path length / optimal path length
    time_efficiency: float = 0.0  # Actual time / minimum time
    trajectory_length_meters: float = 0.0  # Total end-effector path length

    # Dynamics feasibility
    dynamics_feasibility: float = 1.0  # 0-1, 1.0 if all limits satisfied
    joint_limit_violations: int = 0  # Count of joint limit violations
    velocity_limit_violations: int = 0  # Count of velocity violations
    torque_limit_violations: int = 0  # Count of torque violations

    # Safety metrics
    collision_count: int = 0  # Number of collisions detected
    self_collision_count: int = 0  # Number of self-collisions
    min_clearance_meters: float = 0.0  # Minimum clearance to obstacles


@dataclass
class VisualQualityMetrics:
    """Metrics for visual observation quality."""

    # Image quality (per camera)
    mean_sharpness: float = 0.0  # Mean Laplacian variance
    mean_brightness: float = 0.0  # Mean pixel intensity
    brightness_std: float = 0.0  # Brightness variation (exposure consistency)

    # Target visibility
    target_visibility_ratio: float = 0.0  # Fraction of frames where target visible
    mean_target_pixel_count: float = 0.0  # Mean pixels occupied by target
    occlusion_events: int = 0  # Count of occlusion events

    # Camera coverage
    viewpoint_diversity: float = 0.0  # Diversity of camera viewpoints (entropy)
    workspace_coverage: float = 0.0  # Fraction of workspace visible


@dataclass
class TaskQualityMetrics:
    """Metrics for task completion quality."""

    # Goal achievement
    goal_achievement_score: float = 0.0  # 0-1, how well goal was achieved
    final_state_error_meters: float = 0.0  # Distance from goal state
    final_orientation_error_rad: float = 0.0  # Orientation error

    # Skill segment correctness
    skill_segment_count: int = 0  # Number of skill segments
    skill_segments_correct: int = 0  # Number correctly executed
    skill_correctness_ratio: float = 0.0  # Correct / total

    # Constraint satisfaction
    constraint_violations: int = 0  # Number of constraint violations
    constraint_satisfaction_score: float = 1.0  # 0-1, 1.0 if all satisfied


@dataclass
class DiversityMetrics:
    """Metrics for episode diversity (within dataset)."""

    # Trajectory diversity
    trajectory_novelty: float = 0.0  # Distance from existing trajectories
    path_similarity_to_nearest: float = 0.0  # Similarity to most similar episode

    # State space coverage
    state_space_coverage: float = 0.0  # Fraction of reachable state space covered
    configuration_diversity: float = 0.0  # Joint configuration diversity


@dataclass
class Sim2RealMetrics:
    """Metrics for sim-to-real transfer assessment."""

    # Physics plausibility
    physics_plausibility_score: float = 1.0  # 0-1, 1.0 if plausible
    contact_forces_realistic: bool = True  # Contact forces within real-world bounds
    max_contact_force_newtons: float = 0.0  # Maximum contact force

    # Timing realism
    timing_realism_score: float = 1.0  # 0-1, based on comparison to human baseline
    episode_duration_seconds: float = 0.0  # Total episode duration
    human_baseline_ratio: float = 1.0  # Episode time / human baseline time


# =============================================================================
# Quality Certificate
# =============================================================================


@dataclass
class QualityWeights:
    """
    P2-2 FIX: Configurable weights for quality score computation.

    Allows tuning quality scoring for different use cases:
    - Production training (prioritize safety and task success)
    - Research (prioritize diversity and sim2real)
    - Visualization (prioritize visual quality)
    """
    # Overall quality weights (sum should be 1.0)
    trajectory: float = 0.30
    task: float = 0.30
    visual: float = 0.20
    sim2real: float = 0.15
    diversity: float = 0.05

    # Trajectory sub-weights
    smoothness: float = 0.4
    safety: float = 0.4
    feasibility: float = 0.2

    # Task sub-weights
    goal_achievement: float = 0.7
    skill_correctness: float = 0.2
    constraint_satisfaction: float = 0.1

    # Visual sub-weights
    target_visibility: float = 0.6
    image_sharpness: float = 0.25
    viewpoint_diversity: float = 0.15

    # Sim2Real sub-weights
    physics_plausibility: float = 0.5
    timing_realism: float = 0.5

    # Diversity sub-weights
    trajectory_novelty: float = 0.5
    state_space_coverage: float = 0.5

    def __post_init__(self):
        """Validate that weights sum to approximately 1.0."""
        total = self.trajectory + self.task + self.visual + self.sim2real + self.diversity
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Quality weights must sum to 1.0, got {total:.3f}")

    @classmethod
    def for_production_training(cls) -> "QualityWeights":
        """Weights optimized for production training datasets."""
        return cls(
            trajectory=0.35,  # Higher weight on trajectory quality
            task=0.35,       # Higher weight on task success
            visual=0.15,     # Moderate visual quality
            sim2real=0.10,   # Moderate sim2real
            diversity=0.05,  # Low diversity
            smoothness=0.35, # More emphasis on smoothness
            safety=0.50,     # Maximum emphasis on safety
            feasibility=0.15,
        )

    @classmethod
    def for_research(cls) -> "QualityWeights":
        """Weights optimized for research datasets."""
        return cls(
            trajectory=0.25,
            task=0.25,
            visual=0.20,
            sim2real=0.20,   # Higher sim2real weight
            diversity=0.10,  # Higher diversity weight
        )

    @classmethod
    def for_visualization(cls) -> "QualityWeights":
        """Weights optimized for visualization/demo datasets."""
        return cls(
            trajectory=0.25,
            task=0.25,
            visual=0.40,     # Highest visual quality
            sim2real=0.05,
            diversity=0.05,
        )


@dataclass
class QualityCertificate:
    """
    Comprehensive quality certificate for a generated episode.

    This certificate makes data quality transparent and enables informed
    decisions about dataset filtering and usage.
    """

    # Identification
    episode_id: str
    scene_id: str
    task_id: str
    generated_at: str  # ISO 8601 timestamp
    pipeline_version: str = "1.0.0"

    # Environment information
    sensor_source: str = SensorSource.MOCK.value
    physics_backend: str = PhysicsValidationBackend.HEURISTIC.value
    isaac_sim_version: Optional[str] = None
    gpu_type: Optional[str] = None

    # Quality level
    data_quality_level: str = DataQualityLevel.DEVELOPMENT.value
    training_suitability: str = "development_only"  # production | development_only
    recommended_use: str = "testing"  # production_training | fine_tuning | testing

    # Quality metrics
    trajectory_metrics: TrajectoryQualityMetrics = field(default_factory=TrajectoryQualityMetrics)
    visual_metrics: VisualQualityMetrics = field(default_factory=VisualQualityMetrics)
    task_metrics: TaskQualityMetrics = field(default_factory=TaskQualityMetrics)
    diversity_metrics: DiversityMetrics = field(default_factory=DiversityMetrics)
    sim2real_metrics: Sim2RealMetrics = field(default_factory=Sim2RealMetrics)

    # P2-2 FIX: Configurable quality weights
    quality_weights: QualityWeights = field(default_factory=QualityWeights)

    # Overall scores
    overall_quality_score: float = 0.0  # Weighted average of all metrics (0-1)
    confidence_score: float = 0.0  # Confidence in quality assessment (0-1)

    # Validation results
    validation_passed: bool = False
    validation_warnings: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)

    # Data integrity
    data_hash: Optional[str] = None  # SHA256 of episode data
    frame_count: int = 0
    camera_count: int = 0

    def compute_overall_quality_score(self) -> float:
        """
        P2-2 FIX: Compute overall quality score using configurable weights.

        Uses the quality_weights field to determine relative importance of
        different quality aspects. Allows tuning for different use cases.
        """
        w = self.quality_weights

        scores = {
            "trajectory": self._compute_trajectory_score(),
            "task": self._compute_task_score(),
            "visual": self._compute_visual_score(),
            "sim2real": self._compute_sim2real_score(),
            "diversity": self._compute_diversity_score(),
        }

        overall = (
            w.trajectory * scores["trajectory"] +
            w.task * scores["task"] +
            w.visual * scores["visual"] +
            w.sim2real * scores["sim2real"] +
            w.diversity * scores["diversity"]
        )
        self.overall_quality_score = overall
        return overall

    def _compute_trajectory_score(self) -> float:
        """P2-2 FIX: Compute trajectory quality score using configurable weights."""
        tm = self.trajectory_metrics
        w = self.quality_weights

        # Smoothness component
        smoothness = tm.smoothness_score

        # Safety component
        safety = 1.0
        if tm.collision_count > 0:
            safety *= 0.5  # Collisions are bad
        if tm.self_collision_count > 0:
            safety *= 0.7  # Self-collisions are worse

        # Feasibility component
        feasibility = tm.dynamics_feasibility

        return w.smoothness * smoothness + w.safety * safety + w.feasibility * feasibility

    def _compute_task_score(self) -> float:
        """P2-2 FIX: Compute task quality score using configurable weights."""
        tm = self.task_metrics
        w = self.quality_weights

        goal = tm.goal_achievement_score
        skill = tm.skill_correctness_ratio
        constraints = tm.constraint_satisfaction_score

        return w.goal_achievement * goal + w.skill_correctness * skill + w.constraint_satisfaction * constraints

    def _compute_visual_score(self) -> float:
        """P2-2 FIX: Compute visual quality score using configurable weights."""
        vm = self.visual_metrics
        w = self.quality_weights

        # Target visibility
        visibility = vm.target_visibility_ratio

        # Image quality (normalize sharpness - Laplacian variance typically 0-1000)
        sharpness = min(1.0, vm.mean_sharpness / 100.0)

        # Viewpoint diversity
        diversity = vm.viewpoint_diversity

        return w.target_visibility * visibility + w.image_sharpness * sharpness + w.viewpoint_diversity * diversity

    def _compute_sim2real_score(self) -> float:
        """P2-2 FIX: Compute sim-to-real quality score using configurable weights."""
        s2r = self.sim2real_metrics
        w = self.quality_weights
        return w.physics_plausibility * s2r.physics_plausibility_score + w.timing_realism * s2r.timing_realism_score

    def _compute_diversity_score(self) -> float:
        """P2-2 FIX: Compute diversity score using configurable weights."""
        dm = self.diversity_metrics
        w = self.quality_weights
        return w.trajectory_novelty * dm.trajectory_novelty + w.state_space_coverage * dm.state_space_coverage

    def assess_training_suitability(self) -> str:
        """Assess training suitability based on quality and source."""
        # Production data with high quality
        # LABS-BLOCKER-002 FIX: Uses unified quality thresholds from quality_constants.py
        if (
            self.sensor_source == SensorSource.ISAAC_SIM_REPLICATOR.value
            and self.physics_backend == PhysicsValidationBackend.PHYSX.value
            and self.overall_quality_score >= PRODUCTION_TRAINING_THRESHOLD  # 0.90
        ):
            return "production_training"

        # Fine-tuning: still uses real physics but may have minor issues
        if (
            self.sensor_source == SensorSource.ISAAC_SIM_REPLICATOR.value
            and self.physics_backend == PhysicsValidationBackend.PHYSX.value
            and self.overall_quality_score >= FINE_TUNING_THRESHOLD  # 0.80
        ):
            return "fine_tuning"

        # Mock data or low quality - NOT suitable for production use
        return "testing"

    def add_warning(self, warning: str):
        """Add a validation warning."""
        if warning not in self.validation_warnings:
            self.validation_warnings.append(warning)

    def add_error(self, error: str):
        """Add a validation error."""
        if error not in self.validation_errors:
            self.validation_errors.append(error)
        self.validation_passed = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Path):
        """Save certificate to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())


# =============================================================================
# Certificate Generator
# =============================================================================


class QualityCertificateGenerator:
    """
    Generates quality certificates for episodes.

    This class orchestrates the computation of all quality metrics and
    produces comprehensive certificates.
    """

    def __init__(self, capabilities: EnvironmentCapabilities):
        """
        Initialize generator.

        Args:
            capabilities: Environment capabilities
        """
        self.capabilities = capabilities

    def generate_certificate(
        self,
        episode_id: str,
        scene_id: str,
        task_id: str,
        trajectory_metrics: Optional[TrajectoryQualityMetrics] = None,
        visual_metrics: Optional[VisualQualityMetrics] = None,
        task_metrics: Optional[TaskQualityMetrics] = None,
        diversity_metrics: Optional[DiversityMetrics] = None,
        sim2real_metrics: Optional[Sim2RealMetrics] = None,
        validation_passed: bool = True,
        frame_count: int = 0,
        camera_count: int = 0,
        episode_data_hash: Optional[str] = None,
    ) -> QualityCertificate:
        """
        Generate quality certificate for an episode.

        Args:
            episode_id: Episode identifier
            scene_id: Scene identifier
            task_id: Task identifier
            trajectory_metrics: Trajectory quality metrics
            visual_metrics: Visual quality metrics
            task_metrics: Task quality metrics
            diversity_metrics: Diversity metrics
            sim2real_metrics: Sim2Real metrics
            validation_passed: Whether validation passed
            frame_count: Number of frames
            camera_count: Number of cameras
            episode_data_hash: SHA256 hash of episode data

        Returns:
            QualityCertificate
        """
        # Create certificate
        cert = QualityCertificate(
            episode_id=episode_id,
            scene_id=scene_id,
            task_id=task_id,
            generated_at=datetime.utcnow().isoformat() + "Z",
            sensor_source=self.capabilities.sensor_source.value,
            physics_backend=self.capabilities.physics_backend.value,
            data_quality_level=self._determine_quality_level(),
            training_suitability=self.capabilities.training_suitability,
            trajectory_metrics=trajectory_metrics or TrajectoryQualityMetrics(),
            visual_metrics=visual_metrics or VisualQualityMetrics(),
            task_metrics=task_metrics or TaskQualityMetrics(),
            diversity_metrics=diversity_metrics or DiversityMetrics(),
            sim2real_metrics=sim2real_metrics or Sim2RealMetrics(),
            validation_passed=validation_passed,
            frame_count=frame_count,
            camera_count=camera_count,
            data_hash=episode_data_hash,
        )

        # Compute overall quality score
        cert.compute_overall_quality_score()

        # Assess training suitability
        cert.recommended_use = cert.assess_training_suitability()

        # Compute confidence score
        cert.confidence_score = self._compute_confidence_score(cert)

        # Add warnings based on environment
        self._add_environment_warnings(cert)

        return cert

    def _determine_quality_level(self) -> str:
        """Determine data quality level from environment."""
        if self.capabilities.can_generate_production_data:
            return DataQualityLevel.PRODUCTION.value
        return DataQualityLevel.DEVELOPMENT.value

    def _compute_confidence_score(self, cert: QualityCertificate) -> float:
        """
        Compute confidence in quality assessment.

        Higher confidence when:
        - Using real physics (PhysX) vs heuristics
        - Using real rendering (Replicator) vs mock
        - More frames captured
        - More cameras used
        """
        confidence = 1.0

        # Physics backend confidence
        if cert.physics_backend == PhysicsValidationBackend.HEURISTIC.value:
            confidence *= 0.6  # Heuristics less reliable

        # Sensor source confidence
        if cert.sensor_source == SensorSource.MOCK.value:
            confidence *= 0.3  # Mock data very unreliable

        # Frame count confidence (more frames = higher confidence)
        if cert.frame_count < 10:
            confidence *= 0.7
        elif cert.frame_count > 100:
            confidence *= 1.1  # Bonus for long episodes

        # Camera count confidence (multi-view = higher confidence)
        if cert.camera_count >= 3:
            confidence *= 1.05

        return min(1.0, confidence)

    def _add_environment_warnings(self, cert: QualityCertificate):
        """Add warnings based on environment capabilities."""
        if not self.capabilities.can_generate_production_data:
            cert.add_warning(
                "Episode generated without Isaac Sim - not suitable for production training"
            )

        if cert.sensor_source == SensorSource.MOCK.value:
            cert.add_warning("Using mock sensor data - images are placeholder noise")

        if cert.physics_backend == PhysicsValidationBackend.HEURISTIC.value:
            cert.add_warning(
                "Using heuristic physics validation - may miss collisions and instabilities"
            )
            if cert.data_quality_level == DataQualityLevel.PRODUCTION.value:
                cert.add_error(
                    "Heuristic physics validation is not eligible for production packaging"
                )

        if cert.overall_quality_score < 0.5:
            cert.add_warning(f"Low quality score: {cert.overall_quality_score:.2f}")

        if not cert.validation_passed:
            cert.add_error("Episode failed validation checks")


# =============================================================================
# Utility Functions
# =============================================================================


def compute_episode_data_hash(episode_data: Dict[str, Any]) -> str:
    """Compute SHA256 hash of episode data for integrity checking."""
    # Convert to stable JSON representation
    json_str = json.dumps(episode_data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


if __name__ == "__main__":
    # Example usage
    from isaac_sim_enforcement import get_environment_capabilities

    capabilities = get_environment_capabilities()
    generator = QualityCertificateGenerator(capabilities)

    # Generate example certificate
    cert = generator.generate_certificate(
        episode_id="episode_000001",
        scene_id="kitchen_001",
        task_id="pick_apple",
        trajectory_metrics=TrajectoryQualityMetrics(
            smoothness_score=0.85,
            path_efficiency=0.92,
            dynamics_feasibility=1.0,
            collision_count=0,
        ),
        task_metrics=TaskQualityMetrics(
            goal_achievement_score=0.95,
            skill_correctness_ratio=1.0,
        ),
        visual_metrics=VisualQualityMetrics(
            mean_sharpness=85.0,
            target_visibility_ratio=0.98,
        ),
        frame_count=150,
        camera_count=3,
    )

    print(cert.to_json())
