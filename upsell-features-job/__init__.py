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

Bundle Tiers:
- Standard: $5,499 (base)
- Pro: $12,499 (language + VLA + enhanced quality)
- Enterprise: $25,000 (sim2real + tactile + advanced)
- Foundation: $500k-$2M+/year (everything + custom scale)

Usage:
    from upsell_features_job import run_enhanced_pipeline, BundleTier

    result = run_enhanced_pipeline(
        scene_dir=Path("./scenes/kitchen_001"),
        tier="pro",
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
]

__version__ = "1.0.0"
