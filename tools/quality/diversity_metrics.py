"""Episode Diversity Metrics for Training Data Quality.

Analyze diversity of generated episodes to ensure robust policy training.
High diversity in trajectories, visuals, and tasks leads to better generalization.
"""

from __future__ import annotations

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryDiversity:
    """Trajectory diversity metrics."""
    variance: float = 0.0  # Spatial variance of trajectories
    goal_position_coverage: float = 0.0  # Coverage of goal space (0-1)
    path_length_variance: float = 0.0  # Variance in path lengths
    unique_trajectory_ratio: float = 0.0  # Ratio of unique trajectories

    # Spatial coverage
    workspace_coverage_pct: float = 0.0  # Percentage of workspace covered
    underrepresented_regions: List[Tuple[float, float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variance": self.variance,
            "goal_position_coverage": self.goal_position_coverage,
            "path_length_variance": self.path_length_variance,
            "unique_trajectory_ratio": self.unique_trajectory_ratio,
            "workspace_coverage_pct": self.workspace_coverage_pct,
            "underrepresented_regions": self.underrepresented_regions,
        }


@dataclass
class VisualDiversity:
    """Visual diversity metrics."""
    viewpoint_coverage: float = 0.0  # Coverage of viewpoint space (0-1)
    lighting_variance: float = 0.0  # Variance in lighting conditions
    camera_pose_entropy: float = 0.0  # Entropy of camera poses

    # Color/appearance
    color_histogram_variance: float = 0.0
    brightness_variance: float = 0.0

    # Object visibility
    object_visibility_distribution: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "viewpoint_coverage": self.viewpoint_coverage,
            "lighting_variance": self.lighting_variance,
            "camera_pose_entropy": self.camera_pose_entropy,
            "color_histogram_variance": self.color_histogram_variance,
            "brightness_variance": self.brightness_variance,
            "object_visibility_distribution": self.object_visibility_distribution,
        }


@dataclass
class TaskDiversity:
    """Task diversity metrics."""
    object_interaction_distribution: Dict[str, int] = field(default_factory=dict)
    task_type_distribution: Dict[str, int] = field(default_factory=dict)
    failure_mode_coverage: Dict[str, int] = field(default_factory=dict)

    # Task complexity
    avg_task_complexity: float = 0.0
    complexity_variance: float = 0.0

    # Success patterns
    success_condition_diversity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_interaction_distribution": self.object_interaction_distribution,
            "task_type_distribution": self.task_type_distribution,
            "failure_mode_coverage": self.failure_mode_coverage,
            "avg_task_complexity": self.avg_task_complexity,
            "complexity_variance": self.complexity_variance,
            "success_condition_diversity": self.success_condition_diversity,
        }


@dataclass
class DiversityReport:
    """Comprehensive diversity report for episode batch."""
    episode_count: int = 0

    # Diversity metrics
    trajectory: TrajectoryDiversity = field(default_factory=TrajectoryDiversity)
    visual: VisualDiversity = field(default_factory=VisualDiversity)
    task: TaskDiversity = field(default_factory=TaskDiversity)

    # Overall score
    overall_diversity_score: float = 0.0  # 0-1, higher is better

    # Recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "trajectory": self.trajectory.to_dict(),
            "visual": self.visual.to_dict(),
            "task": self.task.to_dict(),
            "overall_diversity_score": self.overall_diversity_score,
            "recommendations": self.recommendations,
        }


class DiversityAnalyzer:
    """Analyze diversity of generated episodes.

    Example:
        analyzer = DiversityAnalyzer()

        # Analyze a batch of episodes
        report = analyzer.analyze_batch(episodes)

        print(f"Diversity score: {report.overall_diversity_score:.2f}")
        print(f"Trajectory variance: {report.trajectory.variance:.2f}")

        # Print recommendations
        for rec in report.recommendations:
            print(f"- {rec['message']}")
    """

    def __init__(
        self,
        workspace_bounds: Optional[Tuple[float, float, float]] = None,
        grid_resolution: float = 0.1,
        enable_logging: bool = True,
    ):
        """Initialize diversity analyzer.

        Args:
            workspace_bounds: Workspace bounds (x, y, z) in meters
            grid_resolution: Grid resolution for spatial coverage
            enable_logging: Whether to log analysis
        """
        self.workspace_bounds = workspace_bounds or (2.0, 2.0, 2.0)
        self.grid_resolution = grid_resolution
        self.enable_logging = enable_logging

        # Compute grid dimensions
        self.grid_dims = tuple(
            int(bound / grid_resolution) for bound in self.workspace_bounds
        )

    def log(self, msg: str) -> None:
        """Log if enabled."""
        if self.enable_logging:
            logger.info(f"[DIVERSITY] {msg}")

    def analyze_batch(self, episodes: List[Dict[str, Any]]) -> DiversityReport:
        """Analyze diversity of episode batch.

        Args:
            episodes: List of episode dictionaries with:
                - trajectory: List of (x, y, z) positions
                - camera_poses: List of camera poses
                - task_type: Task type string
                - object_interactions: List of interacted objects
                - success: Whether episode succeeded
                - metadata: Additional metadata

        Returns:
            DiversityReport
        """
        self.log(f"Analyzing diversity of {len(episodes)} episodes")

        report = DiversityReport(episode_count=len(episodes))

        if not episodes:
            return report

        # Analyze trajectory diversity
        report.trajectory = self._trajectory_diversity(episodes)

        # Analyze visual diversity
        report.visual = self._visual_diversity(episodes)

        # Analyze task diversity
        report.task = self._task_diversity(episodes)

        # Compute overall diversity score
        report.overall_diversity_score = self._compute_overall_score(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        self.log(f"Diversity score: {report.overall_diversity_score:.2f}")

        return report

    def _trajectory_diversity(self, episodes: List[Dict[str, Any]]) -> TrajectoryDiversity:
        """Analyze trajectory diversity."""
        trajectories = []
        goal_positions = []
        path_lengths = []

        for ep in episodes:
            traj = ep.get("trajectory", [])
            if traj:
                trajectories.append(traj)
                goal_positions.append(traj[-1])  # Last position is goal
                path_lengths.append(len(traj))

        if not trajectories:
            return TrajectoryDiversity()

        # Spatial variance
        all_positions = [pos for traj in trajectories for pos in traj]
        if all_positions:
            positions_array = np.array(all_positions)
            variance = float(np.mean(np.var(positions_array, axis=0)))
        else:
            variance = 0.0

        # Goal position coverage
        goal_coverage = self._compute_coverage(goal_positions)

        # Path length variance
        if path_lengths:
            path_variance = float(np.var(path_lengths))
        else:
            path_variance = 0.0

        # Unique trajectory ratio
        unique_ratio = self._compute_unique_trajectory_ratio(trajectories)

        # Workspace coverage
        workspace_coverage, underrepresented = self._compute_workspace_coverage(trajectories)

        return TrajectoryDiversity(
            variance=variance,
            goal_position_coverage=goal_coverage,
            path_length_variance=path_variance,
            unique_trajectory_ratio=unique_ratio,
            workspace_coverage_pct=workspace_coverage * 100,
            underrepresented_regions=underrepresented,
        )

    def _visual_diversity(self, episodes: List[Dict[str, Any]]) -> VisualDiversity:
        """Analyze visual diversity."""
        camera_poses = []
        lighting_values = []
        object_visibility = {}

        for ep in episodes:
            # Camera poses
            poses = ep.get("camera_poses", [])
            camera_poses.extend(poses)

            # Lighting (if available)
            lighting = ep.get("lighting_intensity")
            if lighting is not None:
                lighting_values.append(lighting)

            # Object visibility
            visible_objects = ep.get("visible_objects", [])
            for obj in visible_objects:
                object_visibility[obj] = object_visibility.get(obj, 0) + 1

        # Viewpoint coverage
        if camera_poses:
            # Compute viewpoint entropy
            viewpoint_coverage = self._compute_viewpoint_coverage(camera_poses)
            camera_entropy = self._compute_pose_entropy(camera_poses)
        else:
            viewpoint_coverage = 0.0
            camera_entropy = 0.0

        # Lighting variance
        if lighting_values:
            lighting_variance = float(np.var(lighting_values))
        else:
            lighting_variance = 0.0

        # Object visibility distribution
        if object_visibility:
            total_visibility = sum(object_visibility.values())
            visibility_dist = {
                obj: count / total_visibility
                for obj, count in object_visibility.items()
            }
        else:
            visibility_dist = {}

        return VisualDiversity(
            viewpoint_coverage=viewpoint_coverage,
            lighting_variance=lighting_variance,
            camera_pose_entropy=camera_entropy,
            object_visibility_distribution=visibility_dist,
        )

    def _task_diversity(self, episodes: List[Dict[str, Any]]) -> TaskDiversity:
        """Analyze task diversity."""
        object_interactions = {}
        task_types = {}
        failure_modes = {}
        task_complexities = []

        for ep in episodes:
            # Object interactions
            interactions = ep.get("object_interactions", [])
            for obj in interactions:
                object_interactions[obj] = object_interactions.get(obj, 0) + 1

            # Task types
            task_type = ep.get("task_type", "unknown")
            task_types[task_type] = task_types.get(task_type, 0) + 1

            # Failure modes (for failed episodes)
            if not ep.get("success", False):
                failure_mode = ep.get("failure_mode", "unknown")
                failure_modes[failure_mode] = failure_modes.get(failure_mode, 0) + 1

            # Task complexity
            complexity = ep.get("task_complexity", 0)
            task_complexities.append(complexity)

        # Task complexity stats
        if task_complexities:
            avg_complexity = float(np.mean(task_complexities))
            complexity_variance = float(np.var(task_complexities))
        else:
            avg_complexity = 0.0
            complexity_variance = 0.0

        # Success condition diversity
        success_conditions = set(
            ep.get("success_condition", "") for ep in episodes
        )
        success_diversity = len(success_conditions) / max(len(episodes), 1)

        return TaskDiversity(
            object_interaction_distribution=object_interactions,
            task_type_distribution=task_types,
            failure_mode_coverage=failure_modes,
            avg_task_complexity=avg_complexity,
            complexity_variance=complexity_variance,
            success_condition_diversity=success_diversity,
        )

    def _compute_coverage(self, positions: List[Tuple[float, float, float]]) -> float:
        """Compute spatial coverage of positions.

        Returns:
            Coverage score (0-1)
        """
        if not positions:
            return 0.0

        # Create 3D grid
        grid = np.zeros(self.grid_dims)

        # Mark occupied cells
        for pos in positions:
            # Convert to grid coordinates
            grid_pos = tuple(
                min(int(p / self.grid_resolution), dim - 1)
                for p, dim in zip(pos, self.grid_dims)
            )

            try:
                grid[grid_pos] = 1
            except IndexError:
                # Position outside grid
                pass

        # Coverage is fraction of occupied cells
        total_cells = np.prod(self.grid_dims)
        occupied_cells = np.sum(grid)

        return float(occupied_cells / total_cells)

    def _compute_unique_trajectory_ratio(self, trajectories: List[List]) -> float:
        """Compute ratio of unique trajectories.

        Returns:
            Ratio of unique trajectories (0-1)
        """
        if not trajectories:
            return 0.0

        # Hash trajectories (simplified - use first, middle, last positions)
        def traj_hash(traj):
            if len(traj) < 3:
                return tuple(traj[0]) if traj else ()

            first = tuple(traj[0])
            middle = tuple(traj[len(traj) // 2])
            last = tuple(traj[-1])

            # Round to grid resolution
            return tuple(
                round(p / self.grid_resolution) * self.grid_resolution
                for p in (first + middle + last)
            )

        hashes = [traj_hash(traj) for traj in trajectories]
        unique_count = len(set(hashes))

        return unique_count / len(trajectories)

    def _compute_workspace_coverage(
        self,
        trajectories: List[List]
    ) -> Tuple[float, List[Tuple[float, float, float]]]:
        """Compute workspace coverage and identify underrepresented regions.

        Returns:
            (coverage_fraction, underrepresented_regions)
        """
        # Create 3D grid
        grid = np.zeros(self.grid_dims)

        # Mark occupied cells
        for traj in trajectories:
            for pos in traj:
                grid_pos = tuple(
                    min(int(p / self.grid_resolution), dim - 1)
                    for p, dim in zip(pos, self.grid_dims)
                )

                try:
                    grid[grid_pos] += 1
                except IndexError:
                    pass

        # Coverage
        total_cells = np.prod(self.grid_dims)
        occupied_cells = np.sum(grid > 0)
        coverage = float(occupied_cells / total_cells)

        # Find underrepresented regions (visited < 5% of mean)
        mean_visits = np.mean(grid[grid > 0]) if occupied_cells > 0 else 0
        threshold = mean_visits * 0.05

        underrepresented = []
        if mean_visits > 0:
            for idx in np.ndindex(grid.shape):
                if 0 < grid[idx] < threshold:
                    # Convert back to world coordinates
                    world_pos = tuple(
                        (i + 0.5) * self.grid_resolution
                        for i in idx
                    )
                    underrepresented.append(world_pos)

        # Limit to top 10 most underrepresented
        underrepresented = underrepresented[:10]

        return coverage, underrepresented

    def _compute_viewpoint_coverage(self, camera_poses: List[Dict[str, Any]]) -> float:
        """Compute viewpoint coverage.

        Returns:
            Coverage score (0-1)
        """
        if not camera_poses:
            return 0.0

        # Extract camera positions and orientations
        positions = []
        orientations = []

        for pose in camera_poses:
            pos = pose.get("position", [0, 0, 0])
            positions.append(pos)

            # Orientation (as quaternion or euler angles)
            orientation = pose.get("orientation", [0, 0, 0, 1])
            orientations.append(orientation)

        # Compute position coverage
        position_coverage = self._compute_coverage(positions)

        # Compute orientation coverage (simplified - bin into octants)
        # For a more accurate measurement, use spherical coordinates
        orientation_bins = {}
        for ori in orientations:
            # Simple binning based on primary axis
            if len(ori) == 4:  # Quaternion
                # Extract yaw (simplified)
                yaw = math.atan2(2 * (ori[3] * ori[2] + ori[0] * ori[1]),
                                1 - 2 * (ori[1]**2 + ori[2]**2))
            else:  # Euler angles
                yaw = ori[2] if len(ori) > 2 else 0

            # Bin into 8 octants
            bin_idx = int((yaw + math.pi) / (2 * math.pi / 8))
            orientation_bins[bin_idx] = orientation_bins.get(bin_idx, 0) + 1

        orientation_coverage = len(orientation_bins) / 8  # 8 octants

        # Combined coverage
        return (position_coverage + orientation_coverage) / 2

    def _compute_pose_entropy(self, camera_poses: List[Dict[str, Any]]) -> float:
        """Compute entropy of camera poses.

        Returns:
            Entropy value
        """
        if len(camera_poses) < 2:
            return 0.0

        # Discretize poses into bins
        pose_bins = {}

        for pose in camera_poses:
            pos = pose.get("position", [0, 0, 0])

            # Discretize position
            bin_key = tuple(
                round(p / self.grid_resolution)
                for p in pos
            )

            pose_bins[bin_key] = pose_bins.get(bin_key, 0) + 1

        # Compute entropy
        total = len(camera_poses)
        entropy = 0.0

        for count in pose_bins.values():
            p = count / total
            entropy -= p * math.log2(p)

        return entropy

    def _compute_overall_score(self, report: DiversityReport) -> float:
        """Compute overall diversity score.

        Args:
            report: DiversityReport

        Returns:
            Overall score (0-1)
        """
        scores = []

        # Trajectory diversity (40% weight)
        traj_score = (
            report.trajectory.variance * 0.3 +
            report.trajectory.goal_position_coverage * 0.4 +
            report.trajectory.unique_trajectory_ratio * 0.3
        )
        scores.append(traj_score * 0.4)

        # Visual diversity (30% weight)
        visual_score = (
            report.visual.viewpoint_coverage * 0.5 +
            min(report.visual.camera_pose_entropy / 5.0, 1.0) * 0.3 +
            min(report.visual.lighting_variance / 100.0, 1.0) * 0.2
        )
        scores.append(visual_score * 0.3)

        # Task diversity (30% weight)
        task_score = (
            report.task.success_condition_diversity * 0.5 +
            min(len(report.task.task_type_distribution) / 10.0, 1.0) * 0.3 +
            min(len(report.task.object_interaction_distribution) / 20.0, 1.0) * 0.2
        )
        scores.append(task_score * 0.3)

        return sum(scores)

    def _generate_recommendations(self, report: DiversityReport) -> List[Dict[str, Any]]:
        """Generate recommendations based on diversity report.

        Args:
            report: DiversityReport

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check trajectory diversity
        if report.trajectory.workspace_coverage_pct < 50:
            recommendations.append({
                "category": "trajectory",
                "severity": "high",
                "message": f"Low workspace coverage ({report.trajectory.workspace_coverage_pct:.1f}%). "
                          "Consider varying start/goal positions across full workspace.",
                "target_metric": "workspace_coverage_pct",
                "target_value": 70.0,
            })

        if report.trajectory.unique_trajectory_ratio < 0.5:
            recommendations.append({
                "category": "trajectory",
                "severity": "medium",
                "message": f"Low trajectory uniqueness ({report.trajectory.unique_trajectory_ratio:.1%}). "
                          "Many episodes follow similar paths. Vary obstacles or constraints.",
                "target_metric": "unique_trajectory_ratio",
                "target_value": 0.7,
            })

        # Check visual diversity
        if report.visual.viewpoint_coverage < 0.3:
            recommendations.append({
                "category": "visual",
                "severity": "high",
                "message": f"Low viewpoint coverage ({report.visual.viewpoint_coverage:.1%}). "
                          "Vary camera positions and orientations more.",
                "target_metric": "viewpoint_coverage",
                "target_value": 0.6,
            })

        if report.visual.lighting_variance < 10.0:
            recommendations.append({
                "category": "visual",
                "severity": "low",
                "message": "Low lighting variance. Consider varying lighting conditions "
                          "(brightness, color temperature, shadows).",
                "target_metric": "lighting_variance",
                "target_value": 50.0,
            })

        # Check task diversity
        if len(report.task.task_type_distribution) < 3:
            recommendations.append({
                "category": "task",
                "severity": "medium",
                "message": f"Only {len(report.task.task_type_distribution)} task types. "
                          "Include more task variations (pick, place, push, pull, etc.).",
                "target_metric": "task_type_count",
                "target_value": 5,
            })

        if len(report.task.failure_mode_coverage) < 2:
            recommendations.append({
                "category": "task",
                "severity": "low",
                "message": "Limited failure mode coverage. Include challenging scenarios "
                          "(collisions, grasp failures, etc.) for robust training.",
                "target_metric": "failure_mode_count",
                "target_value": 5,
            })

        # Overall diversity
        if report.overall_diversity_score < 0.4:
            recommendations.append({
                "category": "overall",
                "severity": "critical",
                "message": f"Overall diversity score is low ({report.overall_diversity_score:.2f}). "
                          "Training data may not generalize well. Review all diversity metrics.",
                "target_metric": "overall_diversity_score",
                "target_value": 0.6,
            })

        return recommendations
