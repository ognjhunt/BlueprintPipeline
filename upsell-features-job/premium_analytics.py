#!/usr/bin/env python3
"""
Premium Analytics Integration for BlueprintPipeline.

This module integrates all premium analytics capabilities into a unified
service that generates comprehensive data quality validation packages for
robotics labs.

Premium Analytics Package includes:
1. Failure Mode Analysis - Root cause tracking & filtering recommendations
2. Sim-to-Real Fidelity Matrix - Physics validation & trust assessment
3. Embodiment Transfer Analysis - Cross-robot compatibility matrix
4. Grasp Quality Metrics - Grasp stability & robustness analysis
5. Generalization Analysis - Learning curves & curriculum recommendations
6. Trajectory Optimality - Path quality & energy efficiency

Isaac Lab Arena Premium Analytics (NEW):
7. Arena Telemetry Capture - Episode-level metrics, grasp events, collisions
8. Policy Leaderboard - Multi-policy comparison with confidence intervals
9. Parallel Eval Capture - GPU-accelerated benchmark metrics
10. Arena Generalization - Object diversity coverage analysis

Upsell Value: $50,000-$200,000 for complete validation package
Arena Add-ons: Additional $25,000-$75,000 for Arena-specific analytics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

# Import all analytics modules
from .failure_mode_analyzer import (
    FailureModeAnalyzer,
    FailureAnalysisReport,
    analyze_episode_failures,
)
from .sim2real_fidelity_matrix import (
    Sim2RealFidelityAnalyzer,
    FidelityMatrix,
    generate_fidelity_matrix,
)
from .embodiment_transfer_analyzer import (
    EmbodimentTransferAnalyzer,
    EmbodimentTransferReport,
    analyze_embodiment_transfer,
)
from .grasp_quality_analyzer import (
    GraspQualityAnalyzer,
    GraspQualityReport,
    analyze_grasp_quality,
)
from .generalization_analyzer import (
    GeneralizationAnalyzer,
    GeneralizationReport,
    analyze_generalization,
)
from .trajectory_optimality_analyzer import (
    TrajectoryOptimalityAnalyzer,
    TrajectoryOptimalityReport,
    analyze_trajectory_optimality,
)

# Isaac Lab Arena Premium Analytics Modules
from .arena_telemetry_capture import (
    ArenaTelemetryCapture,
    ParallelEvalBatch,
    EpisodeTelemetry,
    create_arena_telemetry_capture,
)
from .policy_leaderboard import (
    PolicyLeaderboardGenerator,
    PolicyEvalResult,
    Leaderboard,
    RankingMetric,
    create_policy_leaderboard_generator,
)
from .parallel_eval_capture import (
    ParallelEvalCapture,
    ParallelEvalResults,
    create_parallel_eval_capture,
)
from .arena_generalization_metrics import (
    ArenaGeneralizationMetrics,
    GeneralizationReport as ArenaGeneralizationReport,
    create_arena_generalization_metrics,
)


class AnalyticsTier:
    """Premium analytics tier definitions."""
    QUICK_INSIGHTS = "quick_insights"      # $5k - Basic analysis
    STANDARD = "standard"                  # $15k - Core metrics
    COMPREHENSIVE = "comprehensive"        # $35k - Full analysis
    ENTERPRISE = "enterprise"              # $75k+ - Everything + custom
    ARENA_BENCHMARK = "arena_benchmark"    # $50k - Arena evaluation analytics
    ARENA_COMPLETE = "arena_complete"      # $100k+ - Full Arena + Premium


@dataclass
class AnalyticsTierConfig:
    """Configuration for each analytics tier."""
    name: str
    price_usd: int
    included_analyses: List[str]
    description: str


TIER_CONFIGS = {
    AnalyticsTier.QUICK_INSIGHTS: AnalyticsTierConfig(
        name="Quick Insights",
        price_usd=5000,
        included_analyses=["failure_summary", "success_rates"],
        description="Basic success/failure analysis with recommendations",
    ),
    AnalyticsTier.STANDARD: AnalyticsTierConfig(
        name="Standard Analytics",
        price_usd=15000,
        included_analyses=[
            "failure_analysis",
            "grasp_quality",
            "generalization",
        ],
        description="Core metrics for training data quality assessment",
    ),
    AnalyticsTier.COMPREHENSIVE: AnalyticsTierConfig(
        name="Comprehensive Analytics",
        price_usd=35000,
        included_analyses=[
            "failure_analysis",
            "grasp_quality",
            "generalization",
            "trajectory_optimality",
            "fidelity_matrix",
        ],
        description="Full analysis for production deployment preparation",
    ),
    AnalyticsTier.ENTERPRISE: AnalyticsTierConfig(
        name="Enterprise Analytics",
        price_usd=75000,
        included_analyses=[
            "failure_analysis",
            "grasp_quality",
            "generalization",
            "trajectory_optimality",
            "fidelity_matrix",
            "embodiment_transfer",
            "custom_reports",
        ],
        description="Complete validation package with multi-robot analysis",
    ),
    AnalyticsTier.ARENA_BENCHMARK: AnalyticsTierConfig(
        name="Arena Benchmark Analytics",
        price_usd=50000,
        included_analyses=[
            "arena_telemetry",
            "policy_leaderboard",
            "parallel_eval",
            "arena_generalization",
            "confidence_intervals",
        ],
        description="Isaac Lab Arena evaluation analytics with statistical rigor",
    ),
    AnalyticsTier.ARENA_COMPLETE: AnalyticsTierConfig(
        name="Arena Complete Package",
        price_usd=125000,
        included_analyses=[
            # Core analytics
            "failure_analysis",
            "grasp_quality",
            "generalization",
            "trajectory_optimality",
            "fidelity_matrix",
            "embodiment_transfer",
            # Arena analytics
            "arena_telemetry",
            "policy_leaderboard",
            "parallel_eval",
            "arena_generalization",
            "confidence_intervals",
            "timeout_collision_breakdown",
            "custom_reports",
        ],
        description="Complete validation + Arena benchmark analytics package",
    ),
}


@dataclass
class PremiumAnalyticsReport:
    """Complete premium analytics report."""
    report_id: str
    scene_id: str
    created_at: str
    tier: str

    # Individual reports - Core Analytics
    failure_analysis: Optional[FailureAnalysisReport] = None
    fidelity_matrix: Optional[FidelityMatrix] = None
    embodiment_transfer: Optional[EmbodimentTransferReport] = None
    grasp_quality: Optional[GraspQualityReport] = None
    generalization: Optional[GeneralizationReport] = None
    trajectory_optimality: Optional[TrajectoryOptimalityReport] = None

    # Individual reports - Isaac Lab Arena Analytics
    arena_telemetry: Optional[Dict[str, Any]] = None
    policy_leaderboard: Optional[Dict[str, Any]] = None
    parallel_eval: Optional[Dict[str, Any]] = None
    arena_generalization: Optional[Dict[str, Any]] = None
    timeout_collision_breakdown: Optional[Dict[str, Any]] = None

    # Executive summary
    executive_summary: Dict[str, Any] = field(default_factory=dict)

    # Overall scores
    data_quality_score: float = 0.0
    training_readiness_score: float = 0.0
    deployment_readiness_score: float = 0.0

    # Key findings
    key_findings: List[Dict[str, Any]] = field(default_factory=list)

    # Prioritized recommendations
    recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # Value assessment
    estimated_value: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "scene_id": self.scene_id,
            "created_at": self.created_at,
            "tier": self.tier,
            "executive_summary": self.executive_summary,
            "scores": {
                "data_quality": f"{self.data_quality_score:.1%}",
                "training_readiness": f"{self.training_readiness_score:.1%}",
                "deployment_readiness": f"{self.deployment_readiness_score:.1%}",
            },
            "key_findings": self.key_findings,
            "recommendations": self.recommendations,
            "estimated_value": self.estimated_value,
            "individual_reports": {
                # Core Analytics
                "failure_analysis": "included" if self.failure_analysis else "not_included",
                "fidelity_matrix": "included" if self.fidelity_matrix else "not_included",
                "embodiment_transfer": "included" if self.embodiment_transfer else "not_included",
                "grasp_quality": "included" if self.grasp_quality else "not_included",
                "generalization": "included" if self.generalization else "not_included",
                "trajectory_optimality": "included" if self.trajectory_optimality else "not_included",
                # Arena Analytics
                "arena_telemetry": "included" if self.arena_telemetry else "not_included",
                "policy_leaderboard": "included" if self.policy_leaderboard else "not_included",
                "parallel_eval": "included" if self.parallel_eval else "not_included",
                "arena_generalization": "included" if self.arena_generalization else "not_included",
                "timeout_collision_breakdown": "included" if self.timeout_collision_breakdown else "not_included",
            },
            # Arena Analytics Data (if available)
            "arena_analytics": {
                "telemetry": self.arena_telemetry,
                "leaderboard": self.policy_leaderboard,
                "parallel_eval": self.parallel_eval,
                "generalization": self.arena_generalization,
                "timeout_collision": self.timeout_collision_breakdown,
            } if any([
                self.arena_telemetry,
                self.policy_leaderboard,
                self.parallel_eval,
                self.arena_generalization,
                self.timeout_collision_breakdown
            ]) else None,
        }


class PremiumAnalyticsService:
    """
    Unified premium analytics service for robotics training data.

    Runs comprehensive analysis and generates customer-facing reports
    that justify premium pricing for robotics labs.
    """

    def __init__(
        self,
        scene_dir: Path,
        tier: str = AnalyticsTier.COMPREHENSIVE,
        robot_type: str = "franka",
        verbose: bool = True,
    ):
        self.scene_dir = Path(scene_dir)
        self.scene_id = self.scene_dir.name
        self.tier = tier
        self.robot_type = robot_type
        self.verbose = verbose

        self.tier_config = TIER_CONFIGS.get(tier, TIER_CONFIGS[AnalyticsTier.COMPREHENSIVE])

        # Output directory
        self.output_dir = self.scene_dir / "premium_analytics"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Episodes directory
        self.episodes_dir = self.scene_dir / "episodes"

        # Arena analytics modules (lazy initialization)
        self._arena_telemetry: Optional[ArenaTelemetryCapture] = None
        self._policy_leaderboard: Optional[PolicyLeaderboardGenerator] = None
        self._parallel_eval: Optional[ParallelEvalCapture] = None
        self._arena_generalization: Optional[ArenaGeneralizationMetrics] = None

    @property
    def arena_telemetry(self) -> ArenaTelemetryCapture:
        """Get or create Arena telemetry capture instance."""
        if self._arena_telemetry is None:
            self._arena_telemetry = create_arena_telemetry_capture(
                output_dir=str(self.output_dir / "arena_telemetry")
            )
        return self._arena_telemetry

    @property
    def policy_leaderboard(self) -> PolicyLeaderboardGenerator:
        """Get or create policy leaderboard generator instance."""
        if self._policy_leaderboard is None:
            self._policy_leaderboard = create_policy_leaderboard_generator(
                output_dir=str(self.output_dir / "leaderboards")
            )
        return self._policy_leaderboard

    @property
    def parallel_eval(self) -> ParallelEvalCapture:
        """Get or create parallel eval capture instance."""
        if self._parallel_eval is None:
            self._parallel_eval = create_parallel_eval_capture(
                output_dir=str(self.output_dir / "parallel_eval")
            )
        return self._parallel_eval

    @property
    def arena_generalization(self) -> ArenaGeneralizationMetrics:
        """Get or create Arena generalization metrics instance."""
        if self._arena_generalization is None:
            self._arena_generalization = create_arena_generalization_metrics(
                output_dir=str(self.output_dir / "arena_generalization")
            )
        return self._arena_generalization

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[PREMIUM-ANALYTICS] {msg}")

    def run_all_analyses(self) -> PremiumAnalyticsReport:
        """
        Run all analyses included in the selected tier.

        Returns:
            Complete PremiumAnalyticsReport
        """
        self.log(f"Running {self.tier_config.name} for scene {self.scene_id}")
        self.log(f"Included analyses: {', '.join(self.tier_config.included_analyses)}")

        report = PremiumAnalyticsReport(
            report_id=str(uuid.uuid4())[:12],
            scene_id=self.scene_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            tier=self.tier,
        )

        # Run each included analysis
        if "failure_analysis" in self.tier_config.included_analyses:
            self.log("Running failure mode analysis...")
            report.failure_analysis = self._run_failure_analysis()

        if "fidelity_matrix" in self.tier_config.included_analyses:
            self.log("Running sim-to-real fidelity analysis...")
            report.fidelity_matrix = self._run_fidelity_analysis()

        if "embodiment_transfer" in self.tier_config.included_analyses:
            self.log("Running embodiment transfer analysis...")
            report.embodiment_transfer = self._run_embodiment_analysis()

        if "grasp_quality" in self.tier_config.included_analyses:
            self.log("Running grasp quality analysis...")
            report.grasp_quality = self._run_grasp_analysis()

        if "generalization" in self.tier_config.included_analyses:
            self.log("Running generalization analysis...")
            report.generalization = self._run_generalization_analysis()

        if "trajectory_optimality" in self.tier_config.included_analyses:
            self.log("Running trajectory optimality analysis...")
            report.trajectory_optimality = self._run_trajectory_analysis()

        # Isaac Lab Arena Analytics
        if "arena_telemetry" in self.tier_config.included_analyses:
            self.log("Running Arena telemetry analysis...")
            report.arena_telemetry = self._run_arena_telemetry_analysis()

        if "policy_leaderboard" in self.tier_config.included_analyses:
            self.log("Running policy leaderboard analysis...")
            report.policy_leaderboard = self._run_policy_leaderboard_analysis()

        if "parallel_eval" in self.tier_config.included_analyses:
            self.log("Running parallel evaluation analysis...")
            report.parallel_eval = self._run_parallel_eval_analysis()

        if "arena_generalization" in self.tier_config.included_analyses:
            self.log("Running Arena generalization analysis...")
            report.arena_generalization = self._run_arena_generalization_analysis()

        if "timeout_collision_breakdown" in self.tier_config.included_analyses:
            self.log("Running timeout/collision breakdown analysis...")
            report.timeout_collision_breakdown = self._run_timeout_collision_analysis()

        # Generate summary and recommendations
        self.log("Generating executive summary...")
        report.executive_summary = self._generate_executive_summary(report)
        report.key_findings = self._extract_key_findings(report)
        report.recommendations = self._prioritize_recommendations(report)

        # Compute overall scores
        report.data_quality_score = self._compute_data_quality_score(report)
        report.training_readiness_score = self._compute_training_readiness(report)
        report.deployment_readiness_score = self._compute_deployment_readiness(report)

        # Value assessment
        report.estimated_value = self._compute_value_assessment(report)

        # Save report
        self._save_report(report)

        self.log(f"Analysis complete. Data quality: {report.data_quality_score:.1%}")
        return report

    def _run_failure_analysis(self) -> Optional[FailureAnalysisReport]:
        """Run failure mode analysis."""
        try:
            return analyze_episode_failures(
                episodes_dir=self.episodes_dir,
                scene_id=self.scene_id,
                output_dir=self.output_dir / "failure_analysis",
            )
        except Exception as e:
            self.log(f"Failure analysis error: {e}")
            return None

    def _run_fidelity_analysis(self) -> Optional[FidelityMatrix]:
        """Run sim-to-real fidelity analysis."""
        try:
            return generate_fidelity_matrix(
                scene_dir=self.scene_dir,
                robot_type=self.robot_type,
                output_dir=self.output_dir / "fidelity_matrix",
            )
        except Exception as e:
            self.log(f"Fidelity analysis error: {e}")
            return None

    def _run_embodiment_analysis(self) -> Optional[EmbodimentTransferReport]:
        """Run embodiment transfer analysis."""
        try:
            return analyze_embodiment_transfer(
                episodes_dir=self.episodes_dir,
                scene_id=self.scene_id,
                output_dir=self.output_dir / "embodiment_transfer",
            )
        except Exception as e:
            self.log(f"Embodiment analysis error: {e}")
            return None

    def _run_grasp_analysis(self) -> Optional[GraspQualityReport]:
        """Run grasp quality analysis."""
        try:
            return analyze_grasp_quality(
                episodes_dir=self.episodes_dir,
                scene_id=self.scene_id,
                output_dir=self.output_dir / "grasp_quality",
            )
        except Exception as e:
            self.log(f"Grasp analysis error: {e}")
            return None

    def _run_generalization_analysis(self) -> Optional[GeneralizationReport]:
        """Run generalization analysis."""
        try:
            return analyze_generalization(
                episodes_dir=self.episodes_dir,
                scene_id=self.scene_id,
                output_dir=self.output_dir / "generalization",
            )
        except Exception as e:
            self.log(f"Generalization analysis error: {e}")
            return None

    def _run_trajectory_analysis(self) -> Optional[TrajectoryOptimalityReport]:
        """Run trajectory optimality analysis."""
        try:
            return analyze_trajectory_optimality(
                episodes_dir=self.episodes_dir,
                scene_id=self.scene_id,
                output_dir=self.output_dir / "trajectory_optimality",
            )
        except Exception as e:
            self.log(f"Trajectory analysis error: {e}")
            return None

    def _run_arena_telemetry_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Run Arena telemetry analysis.

        Captures episode-level telemetry including per-step rewards,
        collisions, and grasp events - NOT captured in standard output.
        """
        try:
            # Load Arena evaluation data if available
            arena_data_path = self.scene_dir / "arena_eval" / "telemetry.json"
            if not arena_data_path.exists():
                arena_data_path = self.scene_dir / "arena_telemetry.json"

            if arena_data_path.exists():
                with open(arena_data_path) as f:
                    arena_data = json.load(f)

                # Generate premium report from loaded data
                return {
                    "source": str(arena_data_path),
                    "episodes_analyzed": arena_data.get("total_episodes", 0),
                    "grasp_events_captured": arena_data.get("grasp_events", 0),
                    "collision_events_captured": arena_data.get("collision_events", 0),
                    "telemetry_metrics": {
                        "per_step_rewards": "captured",
                        "collision_detection": "captured",
                        "grasp_timeline": "captured",
                        "reward_decomposition": "captured",
                    },
                    "upsell_value": "$15,000 - $30,000",
                    "raw_data": arena_data,
                }
            else:
                # Return placeholder indicating capability
                return {
                    "status": "arena_data_not_found",
                    "message": "Arena telemetry capture ready - run Arena evaluation to generate data",
                    "capabilities": {
                        "per_step_rewards": "Available with Arena eval",
                        "collision_detection": "Available with Arena eval",
                        "grasp_timeline": "Available with Arena eval",
                        "reward_decomposition": "Available with Arena eval",
                    },
                    "upsell_value": "$15,000 - $30,000",
                }
        except Exception as e:
            self.log(f"Arena telemetry analysis error: {e}")
            return None

    def _run_policy_leaderboard_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Run policy leaderboard analysis with confidence intervals.

        Provides multi-policy comparison rankings with statistical
        significance testing - NOT captured in standard output.
        """
        try:
            # Check for policy evaluation results
            policies_dir = self.scene_dir / "policies"
            arena_results_path = self.scene_dir / "arena_eval" / "policy_results.json"

            if arena_results_path.exists():
                with open(arena_results_path) as f:
                    results_data = json.load(f)

                # Add policies to leaderboard generator
                for policy_data in results_data.get("policies", []):
                    result = PolicyEvalResult(
                        policy_id=policy_data.get("policy_id", "unknown"),
                        policy_name=policy_data.get("name", "Unknown Policy"),
                        policy_version=policy_data.get("version", "1.0"),
                        task_name=policy_data.get("task", self.scene_id),
                        total_episodes=policy_data.get("total_episodes", 0),
                        successful_episodes=policy_data.get("successful_episodes", 0),
                        success_rate=policy_data.get("success_rate", 0.0),
                        mean_reward=policy_data.get("mean_reward", 0.0),
                        std_reward=policy_data.get("std_reward", 0.0),
                        episode_rewards=policy_data.get("episode_rewards", []),
                        episode_outcomes=policy_data.get("episode_outcomes", []),
                    )
                    self.policy_leaderboard.add_policy_result(result)

                # Generate leaderboard
                leaderboard = self.policy_leaderboard.generate_leaderboard(
                    task_name=self.scene_id,
                    metric=RankingMetric.SUCCESS_RATE,
                    run_significance_tests=True,
                )

                # Generate premium report
                return self.policy_leaderboard.generate_premium_report(self.scene_id)
            else:
                return {
                    "status": "policy_data_not_found",
                    "message": "Policy leaderboard ready - run multi-policy Arena evaluation to generate rankings",
                    "capabilities": {
                        "confidence_intervals": "Wilson score & bootstrap methods",
                        "significance_testing": "t-test & Mann-Whitney U",
                        "rank_stability": "Bootstrap estimation",
                        "pairwise_comparison": "Full comparison matrix",
                    },
                    "upsell_value": "$20,000 - $40,000",
                }
        except Exception as e:
            self.log(f"Policy leaderboard analysis error: {e}")
            return None

    def _run_parallel_eval_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Run parallel evaluation analysis.

        Captures GPU-accelerated parallel evaluation metrics for
        1000+ environment benchmarks - NOT captured in standard output.
        """
        try:
            parallel_results_path = self.scene_dir / "arena_eval" / "parallel_results.json"

            if parallel_results_path.exists():
                with open(parallel_results_path) as f:
                    results_data = json.load(f)

                return {
                    "source": str(parallel_results_path),
                    "num_environments": results_data.get("num_environments", 0),
                    "gpus_used": results_data.get("num_gpus", 1),
                    "throughput": {
                        "episodes_per_second": results_data.get("episodes_per_second", 0),
                        "steps_per_second": results_data.get("steps_per_second", 0),
                        "realtime_factor": results_data.get("sim_to_real_ratio", 1.0),
                    },
                    "reproducibility": {
                        "score": results_data.get("reproducibility_score", 0),
                        "inter_env_variance": results_data.get("inter_env_reward_variance", 0),
                    },
                    "gpu_efficiency": {
                        "utilization": results_data.get("gpu_utilization_mean", 0),
                        "memory_efficiency": results_data.get("gpu_memory_efficiency", 0),
                        "power_efficiency": results_data.get("power_efficiency", 0),
                    },
                    "upsell_value": "$25,000 - $50,000",
                    "raw_data": results_data,
                }
            else:
                return {
                    "status": "parallel_eval_not_found",
                    "message": "Parallel eval capture ready - run GPU-accelerated Arena benchmark to generate metrics",
                    "capabilities": {
                        "parallel_environments": "Up to 4096+ environments",
                        "throughput_analysis": "Episodes/steps per second tracking",
                        "reproducibility_scoring": "Cross-environment variance analysis",
                        "gpu_efficiency": "Utilization & power metrics",
                    },
                    "upsell_value": "$25,000 - $50,000",
                }
        except Exception as e:
            self.log(f"Parallel eval analysis error: {e}")
            return None

    def _run_arena_generalization_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Run Arena generalization analysis.

        Captures object diversity coverage and generalization metrics
        NOT captured in standard output.
        """
        try:
            gen_results_path = self.scene_dir / "arena_eval" / "generalization.json"

            if gen_results_path.exists():
                with open(gen_results_path) as f:
                    results_data = json.load(f)

                return {
                    "source": str(gen_results_path),
                    "overall_score": results_data.get("overall_generalization_score", 0),
                    "dimension_scores": results_data.get("dimension_scores", {}),
                    "coverage_analysis": {
                        "object_diversity": results_data.get("object_coverage", {}),
                        "pose_diversity": results_data.get("pose_coverage", {}),
                        "lighting_diversity": results_data.get("lighting_coverage", {}),
                        "clutter_diversity": results_data.get("clutter_coverage", {}),
                    },
                    "failure_patterns": results_data.get("failure_patterns", []),
                    "training_recommendations": results_data.get("training_recommendations", []),
                    "upsell_value": "$15,000 - $35,000",
                }
            else:
                return {
                    "status": "generalization_data_not_found",
                    "message": "Arena generalization analysis ready - run diversity evaluation to generate metrics",
                    "capabilities": {
                        "object_coverage": "Track unique objects and categories tested",
                        "pose_coverage": "Analyze position/rotation variation coverage",
                        "visual_coverage": "Lighting and appearance variation tracking",
                        "difficulty_analysis": "Success rate by difficulty level",
                    },
                    "upsell_value": "$15,000 - $35,000",
                }
        except Exception as e:
            self.log(f"Arena generalization analysis error: {e}")
            return None

    def _run_timeout_collision_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Run timeout vs collision failure breakdown analysis.

        Provides detailed breakdown of failure types - NOT captured
        in standard output.
        """
        try:
            telemetry_path = self.scene_dir / "arena_eval" / "telemetry.json"

            if telemetry_path.exists():
                with open(telemetry_path) as f:
                    telemetry_data = json.load(f)

                # Compute breakdown if raw data available
                termination_counts = telemetry_data.get("termination_counts", {})
                total_failures = sum(
                    count for term_type, count in termination_counts.items()
                    if term_type != "success"
                )

                return {
                    "source": str(telemetry_path),
                    "total_failures": total_failures,
                    "breakdown": {
                        "timeout": {
                            "count": termination_counts.get("timeout", 0),
                            "percentage": (termination_counts.get("timeout", 0) / total_failures * 100) if total_failures > 0 else 0,
                            "by_phase": telemetry_data.get("timeout_by_phase", {}),
                        },
                        "collision": {
                            "count": termination_counts.get("collision", 0),
                            "percentage": (termination_counts.get("collision", 0) / total_failures * 100) if total_failures > 0 else 0,
                            "by_type": telemetry_data.get("collision_by_type", {}),
                        },
                        "other": {
                            "count": sum(
                                count for term_type, count in termination_counts.items()
                                if term_type not in ["success", "timeout", "collision"]
                            ),
                            "types": {
                                k: v for k, v in termination_counts.items()
                                if k not in ["success", "timeout", "collision"]
                            },
                        },
                    },
                    "insights": self._generate_failure_insights(termination_counts, total_failures),
                    "upsell_value": "Included in Arena Telemetry package",
                }
            else:
                return {
                    "status": "telemetry_not_found",
                    "message": "Timeout/collision breakdown ready - run Arena evaluation to generate failure analysis",
                    "capabilities": {
                        "timeout_analysis": "Phase-by-phase timeout breakdown",
                        "collision_types": "Self/table/object/environment collision tracking",
                        "failure_insights": "Actionable optimization recommendations",
                    },
                    "upsell_value": "Included in Arena Telemetry package",
                }
        except Exception as e:
            self.log(f"Timeout/collision analysis error: {e}")
            return None

    def _generate_failure_insights(
        self,
        termination_counts: Dict[str, int],
        total_failures: int
    ) -> List[str]:
        """Generate insights from failure breakdown."""
        insights = []

        if total_failures == 0:
            return ["No failures recorded - excellent performance!"]

        timeout_pct = (termination_counts.get("timeout", 0) / total_failures * 100) if total_failures > 0 else 0
        collision_pct = (termination_counts.get("collision", 0) / total_failures * 100) if total_failures > 0 else 0

        if timeout_pct > 50:
            insights.append(
                f"High timeout rate ({timeout_pct:.1f}%) - consider trajectory optimization or increasing episode length"
            )

        if collision_pct > 30:
            insights.append(
                f"Significant collision rate ({collision_pct:.1f}%) - motion planning refinement recommended"
            )

        if timeout_pct < 20 and collision_pct < 20:
            insights.append(
                "Diverse failure modes - investigate individual failure types for targeted improvements"
            )

        return insights

    def _generate_executive_summary(
        self,
        report: PremiumAnalyticsReport,
    ) -> Dict[str, Any]:
        """Generate executive summary from all analyses."""
        summary = {
            "scene_id": self.scene_id,
            "analysis_date": report.created_at,
            "tier": self.tier_config.name,
        }

        # Episode statistics
        if report.failure_analysis:
            summary["episodes"] = {
                "total": report.failure_analysis.total_episodes,
                "successful": report.failure_analysis.total_episodes - report.failure_analysis.failed_episodes,
                "success_rate": f"{report.failure_analysis.success_rate:.1%}",
            }
            summary["failure_rate"] = f"{1 - report.failure_analysis.success_rate:.1%}"
            summary["top_failure_mode"] = report.failure_analysis.top_failure_mode

        # Sim-to-real assessment
        if report.fidelity_matrix:
            summary["sim2real"] = {
                "overall_grade": report.fidelity_matrix.overall_grade.value,
                "transfer_confidence": f"{report.fidelity_matrix.transfer_confidence:.0%}",
                "physics_grade": report.fidelity_matrix.physics_grade.value,
            }

        # Grasp quality
        if report.grasp_quality:
            summary["grasp_quality"] = {
                "success_rate": f"{report.grasp_quality.grasp_success_rate:.1%}",
                "avg_stability": f"{report.grasp_quality.avg_stability_score:.2f}",
                "usability_score": f"{report.grasp_quality.data_usability_score:.2f}",
            }

        # Generalization
        if report.generalization:
            summary["generalization"] = {
                "score": f"{report.generalization.generalization_score:.1%}",
                "robustness": f"{report.generalization.robustness_score:.1%}",
                "unique_objects": report.generalization.unique_objects,
            }

        # Multi-robot
        if report.embodiment_transfer:
            summary["multi_robot"] = {
                "robots_analyzed": len(report.embodiment_transfer.performances),
                "data_multiplier": f"{report.embodiment_transfer.data_multiplier:.1f}x",
                "best_performer": (
                    report.embodiment_transfer.best_overall.value
                    if report.embodiment_transfer.best_overall else "N/A"
                ),
            }

        # Isaac Lab Arena Analytics
        if report.arena_telemetry:
            summary["arena_telemetry"] = {
                "status": report.arena_telemetry.get("status", "captured"),
                "episodes_analyzed": report.arena_telemetry.get("episodes_analyzed", 0),
                "grasp_events": report.arena_telemetry.get("grasp_events_captured", 0),
                "collision_events": report.arena_telemetry.get("collision_events_captured", 0),
            }

        if report.policy_leaderboard:
            summary["policy_leaderboard"] = {
                "status": report.policy_leaderboard.get("status", "generated"),
                "policies_compared": report.policy_leaderboard.get("summary", {}).get("total_policies", 0),
                "best_policy": report.policy_leaderboard.get("summary", {}).get("best_policy_by_success"),
            }

        if report.parallel_eval:
            summary["parallel_eval"] = {
                "status": report.parallel_eval.get("status", "captured"),
                "environments": report.parallel_eval.get("num_environments", 0),
                "throughput_eps": report.parallel_eval.get("throughput", {}).get("episodes_per_second", 0),
                "reproducibility": report.parallel_eval.get("reproducibility", {}).get("score", 0),
            }

        if report.arena_generalization:
            summary["arena_generalization"] = {
                "status": report.arena_generalization.get("status", "analyzed"),
                "overall_score": report.arena_generalization.get("overall_score", 0),
                "dimension_scores": report.arena_generalization.get("dimension_scores", {}),
            }

        if report.timeout_collision_breakdown:
            breakdown = report.timeout_collision_breakdown.get("breakdown", {})
            summary["failure_breakdown"] = {
                "timeout_pct": breakdown.get("timeout", {}).get("percentage", 0),
                "collision_pct": breakdown.get("collision", {}).get("percentage", 0),
                "insights": report.timeout_collision_breakdown.get("insights", []),
            }

        return summary

    def _extract_key_findings(
        self,
        report: PremiumAnalyticsReport,
    ) -> List[Dict[str, Any]]:
        """Extract key findings from all analyses."""
        findings = []

        # Failure analysis findings
        if report.failure_analysis:
            if report.failure_analysis.success_rate >= 0.9:
                findings.append({
                    "category": "success",
                    "finding": f"High success rate ({report.failure_analysis.success_rate:.1%})",
                    "impact": "Data is suitable for training",
                    "priority": "info",
                })
            elif report.failure_analysis.success_rate < 0.7:
                findings.append({
                    "category": "concern",
                    "finding": f"Low success rate ({report.failure_analysis.success_rate:.1%})",
                    "impact": "May need data filtering or collection",
                    "priority": "high",
                })

            if report.failure_analysis.top_failure_mode:
                findings.append({
                    "category": "failure_mode",
                    "finding": f"Primary failure mode: {report.failure_analysis.top_failure_mode}",
                    "impact": "Focus debugging on this failure type",
                    "priority": "medium",
                })

        # Fidelity findings
        if report.fidelity_matrix:
            if report.fidelity_matrix.transfer_confidence < 0.5:
                findings.append({
                    "category": "concern",
                    "finding": f"Low sim-to-real confidence ({report.fidelity_matrix.transfer_confidence:.0%})",
                    "impact": "Real-world validation strongly recommended",
                    "priority": "high",
                })
            elif report.fidelity_matrix.transfer_confidence >= 0.8:
                findings.append({
                    "category": "success",
                    "finding": f"High sim-to-real confidence ({report.fidelity_matrix.transfer_confidence:.0%})",
                    "impact": "Data likely transfers well to real robots",
                    "priority": "info",
                })

        # Grasp findings
        if report.grasp_quality:
            if report.grasp_quality.force_closure_rate < 0.7:
                findings.append({
                    "category": "concern",
                    "finding": f"Low force closure rate ({report.grasp_quality.force_closure_rate:.1%})",
                    "impact": "Grasps may not be robust",
                    "priority": "medium",
                })

        # Generalization findings
        if report.generalization:
            if report.generalization.coverage_score < 0.5:
                findings.append({
                    "category": "concern",
                    "finding": "Limited object diversity",
                    "impact": "May not generalize to new objects",
                    "priority": "medium",
                })

        return findings

    def _prioritize_recommendations(
        self,
        report: PremiumAnalyticsReport,
    ) -> List[Dict[str, Any]]:
        """Consolidate and prioritize all recommendations."""
        all_recs = []

        # Gather recommendations from all reports
        if report.failure_analysis:
            for rec in report.failure_analysis.recommendations:
                all_recs.append({
                    "source": "failure_analysis",
                    "priority": rec.get("priority", "medium"),
                    "action": rec.get("message", rec.get("action", "")),
                    "impact": rec.get("training_impact", ""),
                })

        if report.fidelity_matrix:
            for dim in report.fidelity_matrix.validate_before_deploy:
                all_recs.append({
                    "source": "fidelity_matrix",
                    "priority": "medium",
                    "action": f"Validate {dim} before deployment",
                    "impact": "Reduce sim-to-real gap",
                })

        if report.grasp_quality:
            for rec in report.grasp_quality.recommendations:
                all_recs.append({
                    "source": "grasp_quality",
                    "priority": rec.get("priority", "medium").lower(),
                    "action": rec.get("action", ""),
                    "impact": rec.get("impact", ""),
                })

        if report.generalization:
            for rec in report.generalization.recommendations:
                all_recs.append({
                    "source": "generalization",
                    "priority": rec.get("priority", "medium").lower(),
                    "action": rec.get("action", ""),
                    "impact": rec.get("estimated_impact", ""),
                })

        if report.trajectory_optimality:
            for rec in report.trajectory_optimality.recommendations:
                all_recs.append({
                    "source": "trajectory_optimality",
                    "priority": rec.get("priority", "medium").lower(),
                    "action": rec.get("action", ""),
                    "impact": rec.get("issue", ""),
                })

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_recs.sort(key=lambda x: priority_order.get(x["priority"], 2))

        return all_recs[:10]  # Top 10 recommendations

    def _compute_data_quality_score(
        self,
        report: PremiumAnalyticsReport,
    ) -> float:
        """Compute overall data quality score."""
        scores = []

        if report.failure_analysis:
            scores.append(report.failure_analysis.data_quality_score)

        if report.grasp_quality:
            scores.append(report.grasp_quality.data_usability_score)

        if report.trajectory_optimality:
            scores.append(report.trajectory_optimality.training_suitability_score)

        return sum(scores) / len(scores) if scores else 0.5

    def _compute_training_readiness(
        self,
        report: PremiumAnalyticsReport,
    ) -> float:
        """Compute training readiness score."""
        factors = []

        if report.failure_analysis:
            factors.append(report.failure_analysis.success_rate)

        if report.grasp_quality:
            factors.append(report.grasp_quality.grasp_success_rate)

        if report.generalization:
            factors.append(report.generalization.generalization_score)

        return sum(factors) / len(factors) if factors else 0.5

    def _compute_deployment_readiness(
        self,
        report: PremiumAnalyticsReport,
    ) -> float:
        """Compute deployment readiness score."""
        factors = []

        if report.fidelity_matrix:
            factors.append(report.fidelity_matrix.transfer_confidence)

        if report.grasp_quality:
            factors.append(report.grasp_quality.avg_stability_score)

        if report.trajectory_optimality:
            factors.append(report.trajectory_optimality.avg_quality_score / 100)

        return sum(factors) / len(factors) if factors else 0.5

    def _compute_value_assessment(
        self,
        report: PremiumAnalyticsReport,
    ) -> Dict[str, Any]:
        """Compute value assessment for the customer."""
        base_value = self.tier_config.price_usd

        # Multipliers based on analysis results
        multiplier = 1.0

        if report.data_quality_score >= 0.85:
            multiplier *= 1.2  # High quality data is worth more

        if report.embodiment_transfer:
            multiplier *= report.embodiment_transfer.data_multiplier

        if report.fidelity_matrix and report.fidelity_matrix.transfer_confidence >= 0.8:
            multiplier *= 1.15  # Validated for transfer

        estimated_dataset_value = base_value * multiplier

        return {
            "analytics_tier": self.tier_config.name,
            "analytics_cost": f"${self.tier_config.price_usd:,}",
            "data_quality_multiplier": f"{multiplier:.2f}x",
            "estimated_dataset_value": f"${estimated_dataset_value:,.0f}",
            "value_drivers": [
                "High success rate" if report.failure_analysis and report.failure_analysis.success_rate >= 0.8 else None,
                "Sim2Real validated" if report.fidelity_matrix and report.fidelity_matrix.transfer_confidence >= 0.7 else None,
                "Multi-robot" if report.embodiment_transfer and len(report.embodiment_transfer.performances) > 1 else None,
                "High grasp quality" if report.grasp_quality and report.grasp_quality.grasp_success_rate >= 0.8 else None,
            ],
        }

    def _save_report(self, report: PremiumAnalyticsReport) -> Path:
        """Save the complete report."""
        output_path = self.output_dir / "premium_analytics_report.json"

        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Also save a markdown summary
        md_path = self.output_dir / "EXECUTIVE_SUMMARY.md"
        self._save_markdown_summary(report, md_path)

        self.log(f"Saved premium analytics report to {output_path}")
        return output_path

    def _save_markdown_summary(
        self,
        report: PremiumAnalyticsReport,
        output_path: Path,
    ) -> None:
        """Generate markdown executive summary."""
        md = f"""# Premium Analytics Report

**Scene:** {report.scene_id}
**Generated:** {report.created_at}
**Tier:** {self.tier_config.name}

---

## Executive Summary

| Metric | Score |
|--------|-------|
| **Data Quality** | {report.data_quality_score:.1%} |
| **Training Readiness** | {report.training_readiness_score:.1%} |
| **Deployment Readiness** | {report.deployment_readiness_score:.1%} |

"""

        if report.executive_summary.get("episodes"):
            eps = report.executive_summary["episodes"]
            md += f"""
### Episode Statistics

- **Total Episodes:** {eps["total"]}
- **Successful:** {eps["successful"]}
- **Success Rate:** {eps["success_rate"]}

"""

        if report.key_findings:
            md += "## Key Findings\n\n"
            for finding in report.key_findings:
                icon = "checkmark" if finding["category"] == "success" else "warning"
                md += f"- **[{finding['priority'].upper()}]** {finding['finding']}\n"
                md += f"  - Impact: {finding['impact']}\n\n"

        if report.recommendations:
            md += "## Priority Recommendations\n\n"
            for i, rec in enumerate(report.recommendations[:5], 1):
                md += f"{i}. **[{rec['priority'].upper()}]** {rec['action']}\n"
                if rec.get("impact"):
                    md += f"   - {rec['impact']}\n"
                md += "\n"

        if report.estimated_value:
            md += f"""
## Value Assessment

- **Analytics Cost:** {report.estimated_value.get('analytics_cost', 'N/A')}
- **Data Quality Multiplier:** {report.estimated_value.get('data_quality_multiplier', 'N/A')}
- **Estimated Dataset Value:** {report.estimated_value.get('estimated_dataset_value', 'N/A')}

"""

        md += """
---

*Report generated by BlueprintPipeline Premium Analytics*
*Contact: support@tryblueprint.io*
"""

        with open(output_path, "w") as f:
            f.write(md)


def run_premium_analytics(
    scene_dir: Path,
    tier: str = AnalyticsTier.COMPREHENSIVE,
    robot_type: str = "franka",
    verbose: bool = True,
) -> PremiumAnalyticsReport:
    """
    Run premium analytics on a scene.

    Args:
        scene_dir: Path to scene directory
        tier: Analytics tier (quick_insights, standard, comprehensive, enterprise)
        robot_type: Robot type for analysis
        verbose: Print progress

    Returns:
        PremiumAnalyticsReport
    """
    service = PremiumAnalyticsService(
        scene_dir=scene_dir,
        tier=tier,
        robot_type=robot_type,
        verbose=verbose,
    )

    return service.run_all_analyses()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run premium analytics")
    parser.add_argument("scene_dir", type=Path, help="Path to scene directory")
    parser.add_argument(
        "--tier",
        choices=[
            "quick_insights",
            "standard",
            "comprehensive",
            "enterprise",
            "arena_benchmark",
            "arena_complete",
        ],
        default="comprehensive",
        help="Analytics tier (arena_benchmark and arena_complete include Isaac Lab Arena analytics)",
    )
    parser.add_argument("--robot-type", default="franka", help="Robot type")

    args = parser.parse_args()

    report = run_premium_analytics(
        scene_dir=args.scene_dir,
        tier=args.tier,
        robot_type=args.robot_type,
    )

    print(f"\n{'='*60}")
    print("PREMIUM ANALYTICS REPORT")
    print(f"{'='*60}")
    print(f"Scene: {report.scene_id}")
    print(f"Tier: {report.tier}")
    print(f"\nScores:")
    print(f"  Data Quality: {report.data_quality_score:.1%}")
    print(f"  Training Readiness: {report.training_readiness_score:.1%}")
    print(f"  Deployment Readiness: {report.deployment_readiness_score:.1%}")
    print(f"\nKey Findings: {len(report.key_findings)}")
    print(f"Recommendations: {len(report.recommendations)}")
    print(f"\nValue: {report.estimated_value.get('estimated_dataset_value', 'N/A')}")

    # Arena Analytics Summary
    if any([report.arena_telemetry, report.policy_leaderboard,
            report.parallel_eval, report.arena_generalization]):
        print(f"\n{'='*60}")
        print("ISAAC LAB ARENA ANALYTICS")
        print(f"{'='*60}")

        if report.arena_telemetry:
            status = report.arena_telemetry.get("status", "captured")
            if status == "captured":
                print(f"  Telemetry: {report.arena_telemetry.get('episodes_analyzed', 0)} episodes analyzed")
            else:
                print(f"  Telemetry: Ready (run Arena eval to generate)")

        if report.policy_leaderboard:
            status = report.policy_leaderboard.get("status", "generated")
            if "summary" in report.policy_leaderboard:
                print(f"  Leaderboard: {report.policy_leaderboard['summary'].get('total_policies', 0)} policies compared")
            else:
                print(f"  Leaderboard: Ready (run multi-policy eval to generate)")

        if report.parallel_eval:
            status = report.parallel_eval.get("status", "captured")
            if status == "captured":
                eps = report.parallel_eval.get("throughput", {}).get("episodes_per_second", 0)
                print(f"  Parallel Eval: {report.parallel_eval.get('num_environments', 0)} envs @ {eps:.1f} eps/sec")
            else:
                print(f"  Parallel Eval: Ready (run GPU benchmark to generate)")

        if report.arena_generalization:
            status = report.arena_generalization.get("status", "analyzed")
            if status == "analyzed":
                score = report.arena_generalization.get("overall_score", 0)
                print(f"  Generalization: {score*100:.0f}% coverage score")
            else:
                print(f"  Generalization: Ready (run diversity eval to generate)")
