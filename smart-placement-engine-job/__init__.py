"""Smart Placement Engine - Intelligent asset-to-placement-region matching with collision awareness.

This module provides:
- Scene-to-asset compatibility matrix
- AI-powered intelligent region detection using Gemini 3.1 Pro Preview
- Collision-aware smart placement logic
- Physics validation for placements
"""

from .compatibility_matrix import (
    CompatibilityMatrix,
    AssetCategory,
    PlacementContext,
    get_compatible_assets,
    get_suitable_regions,
)
from .intelligent_region_detector import (
    IntelligentRegionDetector,
    DetectedRegion,
    RegionDetectionResult,
)
from .placement_engine import (
    SmartPlacementEngine,
    PlacementResult,
    PlacementCandidate,
    CollisionCheckResult,
)
from .physics_validator import (
    PhysicsValidator,
    ValidationResult,
    StabilityCheck,
)

__all__ = [
    # Compatibility Matrix
    "CompatibilityMatrix",
    "AssetCategory",
    "PlacementContext",
    "get_compatible_assets",
    "get_suitable_regions",
    # Region Detection
    "IntelligentRegionDetector",
    "DetectedRegion",
    "RegionDetectionResult",
    # Placement Engine
    "SmartPlacementEngine",
    "PlacementResult",
    "PlacementCandidate",
    "CollisionCheckResult",
    # Physics Validation
    "PhysicsValidator",
    "ValidationResult",
    "StabilityCheck",
]
