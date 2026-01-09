#!/usr/bin/env python3
"""
Trajectory Optimality Analysis for BlueprintPipeline.

Provides comprehensive trajectory quality metrics:
- Energy efficiency (joint torque integral)
- Path straightness (deviation from optimal)
- Smoothness (jerk analysis)
- Joint limit margins
- Velocity profile analysis
- Singularity avoidance

These metrics are critical for robotics labs to:
- Validate trajectory quality for training
- Compare to optimal/reference trajectories
- Identify inefficient motion patterns
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid


@dataclass
class TrajectoryMetrics:
    """Comprehensive trajectory quality metrics."""
    # Energy metrics
    total_energy_j: float = 0.0           # Estimated energy consumption
    avg_torque_nm: float = 0.0            # Average joint torque
    peak_torque_nm: float = 0.0           # Maximum torque
    torque_variance: float = 0.0          # Torque smoothness

    # Path metrics
    path_length_m: float = 0.0            # Total end-effector path length
    straight_line_dist_m: float = 0.0     # Direct start-to-end distance
    path_efficiency: float = 0.0          # straight_line / path_length
    deviation_from_optimal_m: float = 0.0 # Max deviation from straight line

    # Smoothness metrics
    max_jerk_rad_s3: float = 0.0          # Maximum jerk
    avg_jerk_rad_s3: float = 0.0          # Average jerk
    jerk_rms: float = 0.0                 # RMS jerk
    smoothness_score: float = 0.0         # Overall smoothness (0-1)

    # Velocity metrics
    avg_velocity_m_s: float = 0.0         # Average EE velocity
    peak_velocity_m_s: float = 0.0        # Maximum EE velocity
    velocity_smoothness: float = 0.0      # Velocity profile smoothness

    # Joint limit metrics
    min_joint_limit_margin_rad: float = 0.0  # Closest approach to limits
    avg_joint_limit_margin_rad: float = 0.0  # Average margin
    joint_limit_violations: int = 0          # Number of violations

    # Singularity metrics
    min_manipulability: float = 0.0       # Closest to singularity
    avg_manipulability: float = 0.0       # Average manipulability
    singularity_warnings: int = 0         # Number of near-singularity points

    # Timing
    duration_s: float = 0.0               # Total trajectory duration
    time_efficiency: float = 0.0          # Compared to expected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "energy": {
                "total_energy_j": self.total_energy_j,
                "avg_torque_nm": self.avg_torque_nm,
                "peak_torque_nm": self.peak_torque_nm,
                "torque_variance": self.torque_variance,
            },
            "path": {
                "path_length_m": self.path_length_m,
                "straight_line_dist_m": self.straight_line_dist_m,
                "path_efficiency": f"{self.path_efficiency:.1%}",
                "deviation_from_optimal_m": self.deviation_from_optimal_m,
            },
            "smoothness": {
                "max_jerk_rad_s3": self.max_jerk_rad_s3,
                "avg_jerk_rad_s3": self.avg_jerk_rad_s3,
                "jerk_rms": self.jerk_rms,
                "smoothness_score": f"{self.smoothness_score:.1%}",
            },
            "velocity": {
                "avg_velocity_m_s": self.avg_velocity_m_s,
                "peak_velocity_m_s": self.peak_velocity_m_s,
                "velocity_smoothness": f"{self.velocity_smoothness:.1%}",
            },
            "joint_limits": {
                "min_margin_rad": self.min_joint_limit_margin_rad,
                "avg_margin_rad": self.avg_joint_limit_margin_rad,
                "violations": self.joint_limit_violations,
            },
            "singularity": {
                "min_manipulability": self.min_manipulability,
                "avg_manipulability": self.avg_manipulability,
                "warnings": self.singularity_warnings,
            },
            "timing": {
                "duration_s": self.duration_s,
                "time_efficiency": f"{self.time_efficiency:.1%}",
            },
        }


@dataclass
class TrajectoryQualityRating:
    """Overall trajectory quality rating."""
    overall_score: float = 0.0  # 0-100

    # Component scores
    energy_score: float = 0.0
    path_score: float = 0.0
    smoothness_score: float = 0.0
    safety_score: float = 0.0

    # Rating
    rating: str = "fair"  # excellent, good, fair, poor

    # Issues found
    issues: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "rating": self.rating,
            "component_scores": {
                "energy": self.energy_score,
                "path": self.path_score,
                "smoothness": self.smoothness_score,
                "safety": self.safety_score,
            },
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


@dataclass
class TrajectoryAnalysis:
    """Analysis of a single trajectory."""
    trajectory_id: str
    episode_id: str

    # Metrics
    metrics: TrajectoryMetrics
    quality: TrajectoryQualityRating

    # Phase breakdown
    phase_metrics: Dict[str, TrajectoryMetrics] = field(default_factory=dict)

    # Waypoint analysis
    num_waypoints: int = 0
    waypoint_spacing_avg_m: float = 0.0
    waypoint_spacing_std_m: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "episode_id": self.episode_id,
            "metrics": self.metrics.to_dict(),
            "quality": self.quality.to_dict(),
            "phase_metrics": {
                phase: m.to_dict() for phase, m in self.phase_metrics.items()
            },
            "waypoints": {
                "count": self.num_waypoints,
                "avg_spacing_m": self.waypoint_spacing_avg_m,
                "std_spacing_m": self.waypoint_spacing_std_m,
            },
        }


@dataclass
class TrajectoryOptimalityReport:
    """Complete trajectory optimality report."""
    report_id: str
    scene_id: str
    created_at: str

    # Summary
    total_trajectories: int
    avg_quality_score: float

    # Quality distribution
    quality_distribution: Dict[str, int] = field(default_factory=dict)

    # Aggregate metrics
    aggregate_metrics: TrajectoryMetrics = field(default_factory=TrajectoryMetrics)

    # Per-phase analysis
    phase_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Individual analyses
    trajectory_analyses: List[TrajectoryAnalysis] = field(default_factory=list)

    # Benchmark comparison
    benchmark_comparison: Dict[str, Any] = field(default_factory=dict)

    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # Training suitability
    training_suitability_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "scene_id": self.scene_id,
            "created_at": self.created_at,
            "summary": {
                "total_trajectories": self.total_trajectories,
                "avg_quality_score": self.avg_quality_score,
            },
            "quality_distribution": self.quality_distribution,
            "aggregate_metrics": self.aggregate_metrics.to_dict(),
            "phase_statistics": self.phase_statistics,
            "benchmark_comparison": self.benchmark_comparison,
            "recommendations": self.recommendations,
            "training_suitability_score": self.training_suitability_score,
        }


class TrajectoryOptimalityAnalyzer:
    """
    Analyzes trajectory optimality for robotics training data.

    Computes energy efficiency, path quality, smoothness, and safety metrics.
    """

    # Quality thresholds
    JERK_EXCELLENT = 100  # rad/s^3
    JERK_GOOD = 300
    JERK_FAIR = 500

    PATH_EFFICIENCY_EXCELLENT = 0.8
    PATH_EFFICIENCY_GOOD = 0.6
    PATH_EFFICIENCY_FAIR = 0.4

    MANIPULABILITY_WARNING = 0.05  # Near singularity
    JOINT_LIMIT_MARGIN_WARNING = 0.1  # radians

    # Robot parameters (Franka defaults)
    DEFAULT_JOINT_LIMITS = [
        (-2.9, 2.9),    # Joint 1
        (-1.76, 1.76),  # Joint 2
        (-2.9, 2.9),    # Joint 3
        (-3.07, -0.07), # Joint 4
        (-2.9, 2.9),    # Joint 5
        (-0.02, 3.75),  # Joint 6
        (-2.9, 2.9),    # Joint 7
    ]

    def __init__(
        self,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        verbose: bool = True,
    ):
        self.joint_limits = joint_limits or self.DEFAULT_JOINT_LIMITS
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[TRAJECTORY-ANALYZER] {msg}")

    def analyze_trajectory(
        self,
        episode_data: Dict[str, Any],
        episode_id: str,
    ) -> TrajectoryAnalysis:
        """
        Analyze a single trajectory from episode data.

        Args:
            episode_data: Episode with frames containing joint/EE data
            episode_id: Episode identifier

        Returns:
            TrajectoryAnalysis
        """
        frames = episode_data.get("frames", [])
        if not frames:
            return self._empty_analysis(episode_id)

        # Extract trajectories
        positions = []
        velocities = []
        ee_positions = []
        torques = []
        timestamps = []
        phases = []

        for frame in frames:
            positions.append(frame.get("joint_positions", [0] * 7))
            velocities.append(frame.get("joint_velocities", [0] * 7))
            ee_positions.append(frame.get("ee_position", [0, 0, 0]))
            torques.append(frame.get("joint_torques", [0] * 7))
            timestamps.append(frame.get("timestamp", 0))
            phases.append(frame.get("phase", "unknown"))

        # Compute metrics
        metrics = self._compute_metrics(
            positions, velocities, ee_positions, torques, timestamps
        )

        # Compute quality rating
        quality = self._compute_quality_rating(metrics)

        # Phase breakdown
        phase_metrics = self._compute_phase_metrics(
            positions, velocities, ee_positions, torques, timestamps, phases
        )

        # Waypoint analysis
        num_waypoints = len(ee_positions)
        spacings = []
        for i in range(1, len(ee_positions)):
            dist = self._euclidean_distance(ee_positions[i], ee_positions[i-1])
            spacings.append(dist)

        avg_spacing = sum(spacings) / len(spacings) if spacings else 0
        std_spacing = self._std_dev(spacings) if len(spacings) > 1 else 0

        return TrajectoryAnalysis(
            trajectory_id=str(uuid.uuid4())[:8],
            episode_id=episode_id,
            metrics=metrics,
            quality=quality,
            phase_metrics=phase_metrics,
            num_waypoints=num_waypoints,
            waypoint_spacing_avg_m=avg_spacing,
            waypoint_spacing_std_m=std_spacing,
        )

    def _compute_metrics(
        self,
        positions: List[List[float]],
        velocities: List[List[float]],
        ee_positions: List[List[float]],
        torques: List[List[float]],
        timestamps: List[float],
    ) -> TrajectoryMetrics:
        """Compute all trajectory metrics."""
        if not positions or len(positions) < 2:
            return TrajectoryMetrics()

        n_frames = len(positions)
        dt = (timestamps[-1] - timestamps[0]) / (n_frames - 1) if n_frames > 1 else 0.033

        # Energy metrics
        torque_magnitudes = [
            sum(abs(t) for t in torque) for torque in torques
        ]
        total_energy = sum(torque_magnitudes) * dt
        avg_torque = sum(torque_magnitudes) / len(torque_magnitudes) if torque_magnitudes else 0
        peak_torque = max(torque_magnitudes) if torque_magnitudes else 0
        torque_variance = self._variance(torque_magnitudes) if len(torque_magnitudes) > 1 else 0

        # Path metrics
        path_length = 0.0
        for i in range(1, len(ee_positions)):
            path_length += self._euclidean_distance(ee_positions[i], ee_positions[i-1])

        straight_line = self._euclidean_distance(ee_positions[0], ee_positions[-1])
        path_efficiency = straight_line / path_length if path_length > 0 else 1.0

        # Max deviation from straight line
        max_deviation = 0.0
        if len(ee_positions) > 2:
            for pos in ee_positions[1:-1]:
                dev = self._point_line_distance(pos, ee_positions[0], ee_positions[-1])
                max_deviation = max(max_deviation, dev)

        # Smoothness metrics (jerk)
        jerks = []
        if len(velocities) >= 3:
            for i in range(1, len(velocities) - 1):
                jerk = 0.0
                for j in range(len(velocities[i])):
                    acc_before = (velocities[i][j] - velocities[i-1][j]) / dt if dt > 0 else 0
                    acc_after = (velocities[i+1][j] - velocities[i][j]) / dt if dt > 0 else 0
                    jerk += abs(acc_after - acc_before) / dt if dt > 0 else 0
                jerks.append(jerk)

        max_jerk = max(jerks) if jerks else 0
        avg_jerk = sum(jerks) / len(jerks) if jerks else 0
        jerk_rms = math.sqrt(sum(j**2 for j in jerks) / len(jerks)) if jerks else 0

        # Smoothness score (inverse of normalized jerk)
        smoothness = max(0, 1 - min(1, avg_jerk / self.JERK_FAIR))

        # Velocity metrics
        ee_velocities = []
        for i in range(1, len(ee_positions)):
            dist = self._euclidean_distance(ee_positions[i], ee_positions[i-1])
            vel = dist / dt if dt > 0 else 0
            ee_velocities.append(vel)

        avg_velocity = sum(ee_velocities) / len(ee_velocities) if ee_velocities else 0
        peak_velocity = max(ee_velocities) if ee_velocities else 0
        velocity_smoothness = 1 - self._variance(ee_velocities) / (avg_velocity**2 + 0.001) if ee_velocities else 0
        velocity_smoothness = max(0, min(1, velocity_smoothness))

        # Joint limit metrics
        margins = []
        violations = 0
        for pos in positions:
            for j, (lower, upper) in enumerate(self.joint_limits[:len(pos)]):
                margin = min(pos[j] - lower, upper - pos[j])
                margins.append(margin)
                if margin < 0:
                    violations += 1

        min_margin = min(margins) if margins else 0
        avg_margin = sum(margins) / len(margins) if margins else 0

        # Singularity metrics (simplified - use manipulability)
        manipulabilities = []
        for vel in velocities:
            # Simplified manipulability based on joint velocities
            manip = 1.0 / (1.0 + sum(v**2 for v in vel))
            manipulabilities.append(manip)

        min_manip = min(manipulabilities) if manipulabilities else 0
        avg_manip = sum(manipulabilities) / len(manipulabilities) if manipulabilities else 0
        sing_warnings = sum(1 for m in manipulabilities if m < self.MANIPULABILITY_WARNING)

        # Timing
        duration = timestamps[-1] - timestamps[0] if timestamps else 0
        expected_duration = path_length / 0.5  # Assume 0.5 m/s avg speed
        time_efficiency = expected_duration / duration if duration > 0 else 1.0
        time_efficiency = min(1.0, time_efficiency)

        return TrajectoryMetrics(
            total_energy_j=total_energy,
            avg_torque_nm=avg_torque,
            peak_torque_nm=peak_torque,
            torque_variance=torque_variance,
            path_length_m=path_length,
            straight_line_dist_m=straight_line,
            path_efficiency=path_efficiency,
            deviation_from_optimal_m=max_deviation,
            max_jerk_rad_s3=max_jerk,
            avg_jerk_rad_s3=avg_jerk,
            jerk_rms=jerk_rms,
            smoothness_score=smoothness,
            avg_velocity_m_s=avg_velocity,
            peak_velocity_m_s=peak_velocity,
            velocity_smoothness=velocity_smoothness,
            min_joint_limit_margin_rad=min_margin,
            avg_joint_limit_margin_rad=avg_margin,
            joint_limit_violations=violations,
            min_manipulability=min_manip,
            avg_manipulability=avg_manip,
            singularity_warnings=sing_warnings,
            duration_s=duration,
            time_efficiency=time_efficiency,
        )

    def _compute_quality_rating(
        self,
        metrics: TrajectoryMetrics,
    ) -> TrajectoryQualityRating:
        """Compute overall quality rating."""
        issues = []
        recommendations = []

        # Energy score
        energy_score = max(0, 100 - metrics.torque_variance * 10)

        # Path score
        path_score = metrics.path_efficiency * 100

        # Smoothness score
        if metrics.avg_jerk_rad_s3 < self.JERK_EXCELLENT:
            smoothness_score = 100
        elif metrics.avg_jerk_rad_s3 < self.JERK_GOOD:
            smoothness_score = 80
        elif metrics.avg_jerk_rad_s3 < self.JERK_FAIR:
            smoothness_score = 60
        else:
            smoothness_score = 40
            issues.append(f"High jerk ({metrics.avg_jerk_rad_s3:.1f} rad/s^3)")
            recommendations.append("Consider trajectory smoothing or re-planning")

        # Safety score
        safety_score = 100
        if metrics.joint_limit_violations > 0:
            safety_score -= 30
            issues.append(f"{metrics.joint_limit_violations} joint limit violations")
            recommendations.append("Verify joint limits in simulation match robot")

        if metrics.singularity_warnings > 0:
            safety_score -= 20
            issues.append(f"{metrics.singularity_warnings} near-singularity points")

        if metrics.min_joint_limit_margin_rad < self.JOINT_LIMIT_MARGIN_WARNING:
            safety_score -= 10
            recommendations.append("Add margin to joint limits in planning")

        # Path issues
        if metrics.path_efficiency < self.PATH_EFFICIENCY_FAIR:
            issues.append(f"Low path efficiency ({metrics.path_efficiency:.1%})")
            recommendations.append("Path is significantly longer than optimal")

        # Overall score
        overall = (
            energy_score * 0.2 +
            path_score * 0.3 +
            smoothness_score * 0.25 +
            safety_score * 0.25
        )

        # Rating
        if overall >= 85:
            rating = "excellent"
        elif overall >= 70:
            rating = "good"
        elif overall >= 50:
            rating = "fair"
        else:
            rating = "poor"

        return TrajectoryQualityRating(
            overall_score=overall,
            energy_score=energy_score,
            path_score=path_score,
            smoothness_score=smoothness_score,
            safety_score=safety_score,
            rating=rating,
            issues=issues,
            recommendations=recommendations,
        )

    def _compute_phase_metrics(
        self,
        positions: List[List[float]],
        velocities: List[List[float]],
        ee_positions: List[List[float]],
        torques: List[List[float]],
        timestamps: List[float],
        phases: List[str],
    ) -> Dict[str, TrajectoryMetrics]:
        """Compute metrics per motion phase."""
        phase_data: Dict[str, Dict[str, List]] = {}

        for i, phase in enumerate(phases):
            if phase not in phase_data:
                phase_data[phase] = {
                    "positions": [],
                    "velocities": [],
                    "ee_positions": [],
                    "torques": [],
                    "timestamps": [],
                }
            phase_data[phase]["positions"].append(positions[i] if i < len(positions) else [])
            phase_data[phase]["velocities"].append(velocities[i] if i < len(velocities) else [])
            phase_data[phase]["ee_positions"].append(ee_positions[i] if i < len(ee_positions) else [])
            phase_data[phase]["torques"].append(torques[i] if i < len(torques) else [])
            phase_data[phase]["timestamps"].append(timestamps[i] if i < len(timestamps) else 0)

        phase_metrics = {}
        for phase, data in phase_data.items():
            if len(data["positions"]) >= 2:
                phase_metrics[phase] = self._compute_metrics(
                    data["positions"],
                    data["velocities"],
                    data["ee_positions"],
                    data["torques"],
                    data["timestamps"],
                )

        return phase_metrics

    def _empty_analysis(self, episode_id: str) -> TrajectoryAnalysis:
        """Return empty analysis for invalid data."""
        return TrajectoryAnalysis(
            trajectory_id=str(uuid.uuid4())[:8],
            episode_id=episode_id,
            metrics=TrajectoryMetrics(),
            quality=TrajectoryQualityRating(overall_score=0, rating="invalid"),
        )

    def _euclidean_distance(self, p1: List[float], p2: List[float]) -> float:
        """Compute Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _point_line_distance(
        self,
        point: List[float],
        line_start: List[float],
        line_end: List[float],
    ) -> float:
        """Compute distance from point to line segment."""
        # Simplified: project point onto line and compute distance
        if len(point) < 3 or len(line_start) < 3 or len(line_end) < 3:
            return 0.0

        line_vec = [e - s for s, e in zip(line_start, line_end)]
        line_len = math.sqrt(sum(v**2 for v in line_vec))

        if line_len == 0:
            return self._euclidean_distance(point, line_start)

        # Normalize
        line_unit = [v / line_len for v in line_vec]

        # Project
        point_vec = [p - s for p, s in zip(point, line_start)]
        proj_len = sum(pv * lu for pv, lu in zip(point_vec, line_unit))
        proj_len = max(0, min(line_len, proj_len))

        proj_point = [s + proj_len * lu for s, lu in zip(line_start, line_unit)]

        return self._euclidean_distance(point, proj_point)

    def _variance(self, values: List[float]) -> float:
        """Compute variance."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def _std_dev(self, values: List[float]) -> float:
        """Compute standard deviation."""
        return math.sqrt(self._variance(values))

    def analyze_dataset(
        self,
        episodes: List[Dict[str, Any]],
        scene_id: str,
    ) -> TrajectoryOptimalityReport:
        """Analyze all trajectories in dataset."""
        self.log(f"Analyzing {len(episodes)} trajectories...")

        analyses = []
        quality_dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0, "invalid": 0}

        for ep in episodes:
            episode_id = ep.get("episode_id", str(uuid.uuid4())[:8])
            analysis = self.analyze_trajectory(ep, episode_id)
            analyses.append(analysis)
            quality_dist[analysis.quality.rating] = quality_dist.get(analysis.quality.rating, 0) + 1

        # Aggregate metrics
        valid_analyses = [a for a in analyses if a.quality.rating != "invalid"]

        if valid_analyses:
            aggregate = TrajectoryMetrics(
                total_energy_j=sum(a.metrics.total_energy_j for a in valid_analyses) / len(valid_analyses),
                avg_torque_nm=sum(a.metrics.avg_torque_nm for a in valid_analyses) / len(valid_analyses),
                path_length_m=sum(a.metrics.path_length_m for a in valid_analyses) / len(valid_analyses),
                path_efficiency=sum(a.metrics.path_efficiency for a in valid_analyses) / len(valid_analyses),
                smoothness_score=sum(a.metrics.smoothness_score for a in valid_analyses) / len(valid_analyses),
                avg_jerk_rad_s3=sum(a.metrics.avg_jerk_rad_s3 for a in valid_analyses) / len(valid_analyses),
                avg_velocity_m_s=sum(a.metrics.avg_velocity_m_s for a in valid_analyses) / len(valid_analyses),
                joint_limit_violations=sum(a.metrics.joint_limit_violations for a in valid_analyses),
                duration_s=sum(a.metrics.duration_s for a in valid_analyses) / len(valid_analyses),
            )
            avg_quality = sum(a.quality.overall_score for a in valid_analyses) / len(valid_analyses)
        else:
            aggregate = TrajectoryMetrics()
            avg_quality = 0

        # Phase statistics
        phase_stats: Dict[str, Dict[str, List[float]]] = {}
        for analysis in valid_analyses:
            for phase, metrics in analysis.phase_metrics.items():
                if phase not in phase_stats:
                    phase_stats[phase] = {"duration": [], "smoothness": [], "path_efficiency": []}
                phase_stats[phase]["duration"].append(metrics.duration_s)
                phase_stats[phase]["smoothness"].append(metrics.smoothness_score)
                phase_stats[phase]["path_efficiency"].append(metrics.path_efficiency)

        phase_statistics = {}
        for phase, stats in phase_stats.items():
            phase_statistics[phase] = {
                "avg_duration": sum(stats["duration"]) / len(stats["duration"]) if stats["duration"] else 0,
                "avg_smoothness": sum(stats["smoothness"]) / len(stats["smoothness"]) if stats["smoothness"] else 0,
                "avg_path_efficiency": sum(stats["path_efficiency"]) / len(stats["path_efficiency"]) if stats["path_efficiency"] else 0,
            }

        # Benchmark comparison
        benchmark_comparison = {
            "path_efficiency_vs_optimal": f"{aggregate.path_efficiency:.1%}",
            "smoothness_vs_target": f"{aggregate.smoothness_score:.1%}",
            "note": "Optimal = straight-line path with zero jerk",
        }

        # Recommendations
        recommendations = []
        if avg_quality < 60:
            recommendations.append({
                "priority": "HIGH",
                "issue": "Low overall trajectory quality",
                "action": "Review motion planning configuration",
            })

        if aggregate.joint_limit_violations > 0:
            recommendations.append({
                "priority": "HIGH",
                "issue": f"{aggregate.joint_limit_violations} total joint limit violations",
                "action": "Verify joint limits match simulation configuration",
            })

        if aggregate.path_efficiency < 0.5:
            recommendations.append({
                "priority": "MEDIUM",
                "issue": "Inefficient paths (< 50% efficiency)",
                "action": "Consider RRT* or trajectory optimization",
            })

        # Training suitability
        training_suitability = (
            (quality_dist["excellent"] + quality_dist["good"]) /
            max(1, len(analyses))
        )

        report = TrajectoryOptimalityReport(
            report_id=str(uuid.uuid4())[:12],
            scene_id=scene_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            total_trajectories=len(analyses),
            avg_quality_score=avg_quality,
            quality_distribution=quality_dist,
            aggregate_metrics=aggregate,
            phase_statistics=phase_statistics,
            trajectory_analyses=analyses,
            benchmark_comparison=benchmark_comparison,
            recommendations=recommendations,
            training_suitability_score=training_suitability,
        )

        self.log(f"Analysis complete: avg quality {avg_quality:.1f}")
        return report

    def save_report(
        self,
        report: TrajectoryOptimalityReport,
        output_path: Path,
    ) -> Path:
        """Save report to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save summary report
        report_dict = report.to_dict()
        report_dict["trajectory_analyses"] = f"[{len(report.trajectory_analyses)} analyses - see detailed file]"

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        self.log(f"Saved trajectory optimality report to {output_path}")
        return output_path


def analyze_trajectory_optimality(
    episodes_dir: Path,
    scene_id: str,
    output_dir: Optional[Path] = None,
) -> TrajectoryOptimalityReport:
    """
    Convenience function to analyze trajectory optimality.

    Args:
        episodes_dir: Path to episodes directory
        scene_id: Scene identifier
        output_dir: Optional output directory

    Returns:
        TrajectoryOptimalityReport
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
        print(f"[TRAJECTORY-ANALYZER] No episode data found, using placeholder")
        episodes = [{"frames": [
            {"joint_positions": [0]*7, "ee_position": [0.4, 0, 0.5], "timestamp": 0},
            {"joint_positions": [0.1]*7, "ee_position": [0.5, 0, 0.4], "timestamp": 0.1},
        ]}]

    analyzer = TrajectoryOptimalityAnalyzer(verbose=True)
    report = analyzer.analyze_dataset(episodes, scene_id)

    if output_dir:
        output_path = Path(output_dir) / "trajectory_optimality_report.json"
        analyzer.save_report(report, output_path)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze trajectory optimality")
    parser.add_argument("episodes_dir", type=Path, help="Path to episodes directory")
    parser.add_argument("--scene-id", required=True, help="Scene identifier")
    parser.add_argument("--output-dir", type=Path, help="Output directory")

    args = parser.parse_args()

    report = analyze_trajectory_optimality(
        episodes_dir=args.episodes_dir,
        scene_id=args.scene_id,
        output_dir=args.output_dir,
    )

    print(f"\n=== Trajectory Optimality Analysis ===")
    print(f"Total Trajectories: {report.total_trajectories}")
    print(f"Avg Quality Score: {report.avg_quality_score:.1f}")
    print(f"Quality Distribution: {report.quality_distribution}")
    print(f"Training Suitability: {report.training_suitability_score:.1%}")
