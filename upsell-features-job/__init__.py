"""
Upsell Features Module for BlueprintPipeline.

This module provides automated upsell capabilities that can be integrated
into the main pipeline to increase bundle value:

Tier 1 - High Impact (Immediate):
- VLA Fine-Tuning Packages (+$3k-$8k/scene)
- Language-Conditioned Data (+$1.5k/scene)
- Sim2Real Validation Service ($5k-$25k)
- Contact-Rich Task Premium (3x pricing)

Tier 2 - Strategic (Medium-term):
- Tactile Sensor Simulation (+$2.5k-$4k)
- Multi-Robot Fleet Coordination (+$6k-$12k)
- Deformable Object Manipulation (+$5k-$8k)
- Custom Robot Embodiment ($15k setup + $2k/scene)
- Bimanual Manipulation (+$6k-$10k)

Premium Analytics (NEW - High Value):
- Failure Mode Analysis ($10k-$50k) - Root cause tracking & debugging
- Sim-to-Real Fidelity Matrix ($20k-$50k) - Physics validation
- Embodiment Transfer Analysis ($20k-$100k) - Cross-robot compatibility
- Grasp Quality Metrics ($15k-$50k) - Grasp stability analysis
- Generalization Analysis ($10k-$30k) - Learning curves & coverage
- Trajectory Optimality ($10k-$25k) - Path quality metrics

Bundle Tiers:
- Standard: $5,499 (base)
- Pro: $12,499 (language + VLA + enhanced quality)
- Enterprise: $25,000 (sim2real + tactile + advanced)
- Foundation: $500k-$2M+/year (everything + custom scale)
- Premium Analytics: $50k-$200k (complete validation package)

Usage:
    from upsell_features_job import run_enhanced_pipeline, BundleTier

    result = run_enhanced_pipeline(
        scene_dir=Path("./scenes/kitchen_001"),
        tier="pro",
        robot_type="franka",
    )

    # For premium analytics
    from upsell_features_job import run_premium_analytics, AnalyticsTier

    report = run_premium_analytics(
        scene_dir=Path("./scenes/kitchen_001"),
        tier=AnalyticsTier.COMPREHENSIVE,
        robot_type="franka",
    )
"""

from .bundle_config import (
    BundleTier,
    BundleFeatures,
    BundlePricing,
    BundleConfigManager,
    BUNDLE_CONFIGS,
    BUNDLE_PRICING,
)

from .vla_finetuning_generator import (
    VLAModel,
    VLAFinetuningConfig,
    VLAFinetuningGenerator,
)

from .language_annotator import (
    LanguageAnnotator,
    AnnotationStyle,
    LanguageTemplates,
    integrate_with_lerobot_export,
)

from .sim2real_service import (
    Sim2RealService,
    ValidationTier,
    ValidationReport,
)

from .contact_rich_tasks import (
    ContactRichTaskType,
    ToleranceClass,
    ContactRichTaskSpec,
    ContactRichTaskGenerator,
)

from .tactile_sensor_sim import (
    TactileSensorType,
    TactileSensorConfig,
    TactileSensorSimulator,
    DualGripperTactileSimulator,
    SENSOR_CONFIGS,
)

from .advanced_capabilities import (
    AdvancedCapabilities,
    MultiRobotCoordinator,
    FleetScenarioType,
    DeformableObjectGenerator,
    DeformableType,
    BimanualTaskGenerator,
    BimanualTaskType,
    CustomRobotOnboarder,
)

from .pipeline_integration import (
    EnhancedPipeline,
    PipelineResult,
    run_enhanced_pipeline,
)

# Premium Analytics - NEW HIGH-VALUE MODULES
from .failure_mode_analyzer import (
    FailureModeAnalyzer,
    FailureAnalysisReport,
    FailureCategory,
    FailureSeverity,
    FailureTaxonomy,
    analyze_episode_failures,
)

from .sim2real_fidelity_matrix import (
    Sim2RealFidelityAnalyzer,
    FidelityMatrix,
    FidelityGrade,
    FidelityDimension,
    DomainRandomizationCoverage,
    generate_fidelity_matrix,
)

from .embodiment_transfer_analyzer import (
    EmbodimentTransferAnalyzer,
    EmbodimentTransferReport,
    EmbodimentPerformance,
    TransferPrediction,
    TransferCompatibility,
    RobotType as EmbodimentRobotType,
    analyze_embodiment_transfer,
)

from .grasp_quality_analyzer import (
    GraspQualityAnalyzer,
    GraspQualityReport,
    GraspAnalysis,
    GraspMetrics,
    GraspType,
    GraspQualityRating,
    analyze_grasp_quality,
)

from .generalization_analyzer import (
    GeneralizationAnalyzer,
    GeneralizationReport,
    ObjectPerformance,
    TaskPerformance,
    LearningCurve,
    CurriculumRecommendation,
    DifficultyLevel,
    analyze_generalization,
)

from .trajectory_optimality_analyzer import (
    TrajectoryOptimalityAnalyzer,
    TrajectoryOptimalityReport,
    TrajectoryAnalysis,
    TrajectoryMetrics,
    TrajectoryQualityRating,
    analyze_trajectory_optimality,
)

from .premium_analytics import (
    PremiumAnalyticsService,
    PremiumAnalyticsReport,
    AnalyticsTier,
    TIER_CONFIGS,
    run_premium_analytics,
)

__all__ = [
    # Bundle configuration
    "BundleTier",
    "BundleFeatures",
    "BundlePricing",
    "BundleConfigManager",
    "BUNDLE_CONFIGS",
    "BUNDLE_PRICING",
    # VLA
    "VLAModel",
    "VLAFinetuningConfig",
    "VLAFinetuningGenerator",
    # Language
    "LanguageAnnotator",
    "AnnotationStyle",
    "LanguageTemplates",
    "integrate_with_lerobot_export",
    # Sim2Real
    "Sim2RealService",
    "ValidationTier",
    "ValidationReport",
    # Contact-rich
    "ContactRichTaskType",
    "ToleranceClass",
    "ContactRichTaskSpec",
    "ContactRichTaskGenerator",
    # Tactile
    "TactileSensorType",
    "TactileSensorConfig",
    "TactileSensorSimulator",
    "DualGripperTactileSimulator",
    "SENSOR_CONFIGS",
    # Advanced
    "AdvancedCapabilities",
    "MultiRobotCoordinator",
    "FleetScenarioType",
    "DeformableObjectGenerator",
    "DeformableType",
    "BimanualTaskGenerator",
    "BimanualTaskType",
    "CustomRobotOnboarder",
    # Pipeline
    "EnhancedPipeline",
    "PipelineResult",
    "run_enhanced_pipeline",
    # Premium Analytics - Failure Mode Analysis
    "FailureModeAnalyzer",
    "FailureAnalysisReport",
    "FailureCategory",
    "FailureSeverity",
    "FailureTaxonomy",
    "analyze_episode_failures",
    # Premium Analytics - Sim2Real Fidelity
    "Sim2RealFidelityAnalyzer",
    "FidelityMatrix",
    "FidelityGrade",
    "FidelityDimension",
    "DomainRandomizationCoverage",
    "generate_fidelity_matrix",
    # Premium Analytics - Embodiment Transfer
    "EmbodimentTransferAnalyzer",
    "EmbodimentTransferReport",
    "EmbodimentPerformance",
    "TransferPrediction",
    "TransferCompatibility",
    "EmbodimentRobotType",
    "analyze_embodiment_transfer",
    # Premium Analytics - Grasp Quality
    "GraspQualityAnalyzer",
    "GraspQualityReport",
    "GraspAnalysis",
    "GraspMetrics",
    "GraspType",
    "GraspQualityRating",
    "analyze_grasp_quality",
    # Premium Analytics - Generalization
    "GeneralizationAnalyzer",
    "GeneralizationReport",
    "ObjectPerformance",
    "TaskPerformance",
    "LearningCurve",
    "CurriculumRecommendation",
    "DifficultyLevel",
    "analyze_generalization",
    # Premium Analytics - Trajectory Optimality
    "TrajectoryOptimalityAnalyzer",
    "TrajectoryOptimalityReport",
    "TrajectoryAnalysis",
    "TrajectoryMetrics",
    "TrajectoryQualityRating",
    "analyze_trajectory_optimality",
    # Premium Analytics - Unified Service
    "PremiumAnalyticsService",
    "PremiumAnalyticsReport",
    "AnalyticsTier",
    "TIER_CONFIGS",
    "run_premium_analytics",
]

__version__ = "2.0.0"
