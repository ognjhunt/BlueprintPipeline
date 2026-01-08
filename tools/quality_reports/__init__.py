"""
Quality Reports Package for BlueprintPipeline.

Provides tools for generating quality reports that robotics labs require
for procurement and quality assurance:

- scene_report.json: Comprehensive quality metrics for a scene
- asset_provenance.json: Legal/licensing audit trail for all assets

These are KEY UPSELL differentiators:
- Labs can justify purchases to procurement with documented quality
- Legal teams can verify licensing before commercial use
"""

from .scene_report_generator import (
    SceneQualityReport,
    SceneReportGenerator,
    generate_scene_report,
    PhysicsQAMetrics,
    AssetInventoryMetrics,
    PerceptionQAMetrics,
    TaskQAMetrics,
)

from .asset_provenance_generator import (
    SceneProvenance,
    AssetProvenance,
    AssetProvenanceGenerator,
    generate_asset_provenance,
    LicenseType,
    COMMERCIAL_OK_LICENSES,
)

__all__ = [
    # Scene reports
    "SceneQualityReport",
    "SceneReportGenerator",
    "generate_scene_report",
    "PhysicsQAMetrics",
    "AssetInventoryMetrics",
    "PerceptionQAMetrics",
    "TaskQAMetrics",
    # Asset provenance
    "SceneProvenance",
    "AssetProvenance",
    "AssetProvenanceGenerator",
    "generate_asset_provenance",
    "LicenseType",
    "COMMERCIAL_OK_LICENSES",
]
