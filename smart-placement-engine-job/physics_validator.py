"""Physics Validator for Placement Validation.

This module validates that placements are physically plausible:
- Stability checks (objects won't fall/tip)
- Weight constraints (surfaces can support objects)
- Friction validation (objects won't slide)
- Support polygon analysis
"""

import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
    from google.genai import types
    HAVE_GENAI = True
except ImportError:
    HAVE_GENAI = False

from .compatibility_matrix import AssetCategory, RegionType
from .intelligent_region_detector import DetectedRegion
from .placement_engine import (
    AssetInstance,
    BoundingBox,
    PlacementCandidate,
    PlacementResult,
    PlacementPlan,
)


class ValidationStatus(str, Enum):
    """Status of a physics validation."""
    VALID = "valid"
    UNSTABLE = "unstable"
    WEIGHT_EXCEEDED = "weight_exceeded"
    FRICTION_INSUFFICIENT = "friction_insufficient"
    SUPPORT_INSUFFICIENT = "support_insufficient"
    WARNING = "warning"


class StabilityLevel(str, Enum):
    """Level of stability for an object."""
    STABLE = "stable"
    MARGINALLY_STABLE = "marginally_stable"
    UNSTABLE = "unstable"


@dataclass
class StabilityCheck:
    """Result of a stability check for a single object."""
    asset_id: str
    level: StabilityLevel
    tip_angle_deg: float = 0.0  # Angle at which object would tip
    support_ratio: float = 1.0  # 0-1, how much of base is supported
    center_of_mass_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    details: str = ""


@dataclass
class WeightCheck:
    """Result of a weight constraint check."""
    region_id: str
    total_weight_kg: float
    max_weight_kg: float
    is_exceeded: bool
    margin_kg: float = 0.0
    heaviest_object: Optional[str] = None


@dataclass
class FrictionCheck:
    """Result of a friction validation."""
    asset_id: str
    static_friction: float
    required_friction: float
    is_sufficient: bool
    slip_angle_deg: float = 0.0  # Angle at which object would slip


@dataclass
class ValidationResult:
    """Complete validation result for a placement or plan."""
    status: ValidationStatus
    stability_checks: List[StabilityCheck] = field(default_factory=list)
    weight_checks: List[WeightCheck] = field(default_factory=list)
    friction_checks: List[FrictionCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    score: float = 1.0  # 0-1 overall physics validity score
    ai_analysis: str = ""


# Default physics properties by asset category
PHYSICS_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "dishes": {
        "bulk_density_kg_m3": 2400,  # Ceramic
        "static_friction": 0.4,
        "dynamic_friction": 0.35,
        "restitution": 0.1,
        "center_of_mass_height_ratio": 0.3,  # CoM is 30% up from base
    },
    "utensils": {
        "bulk_density_kg_m3": 7800,  # Steel
        "static_friction": 0.3,
        "dynamic_friction": 0.25,
        "restitution": 0.15,
        "center_of_mass_height_ratio": 0.4,
    },
    "cookware": {
        "bulk_density_kg_m3": 2700,  # Aluminum
        "static_friction": 0.35,
        "dynamic_friction": 0.3,
        "restitution": 0.1,
        "center_of_mass_height_ratio": 0.35,
    },
    "groceries": {
        "bulk_density_kg_m3": 800,
        "static_friction": 0.5,
        "dynamic_friction": 0.4,
        "restitution": 0.05,
        "center_of_mass_height_ratio": 0.5,
    },
    "bottles": {
        "bulk_density_kg_m3": 1200,  # Glass + liquid
        "static_friction": 0.35,
        "dynamic_friction": 0.3,
        "restitution": 0.1,
        "center_of_mass_height_ratio": 0.4,
    },
    "boxes": {
        "bulk_density_kg_m3": 400,  # Cardboard
        "static_friction": 0.5,
        "dynamic_friction": 0.4,
        "restitution": 0.05,
        "center_of_mass_height_ratio": 0.5,
    },
    "clothing": {
        "bulk_density_kg_m3": 300,
        "static_friction": 0.7,
        "dynamic_friction": 0.6,
        "restitution": 0.02,
        "center_of_mass_height_ratio": 0.5,
    },
    "tools": {
        "bulk_density_kg_m3": 3500,
        "static_friction": 0.5,
        "dynamic_friction": 0.4,
        "restitution": 0.1,
        "center_of_mass_height_ratio": 0.45,
    },
    "lab_equipment": {
        "bulk_density_kg_m3": 2000,
        "static_friction": 0.4,
        "dynamic_friction": 0.35,
        "restitution": 0.1,
        "center_of_mass_height_ratio": 0.4,
    },
    "default": {
        "bulk_density_kg_m3": 600,
        "static_friction": 0.5,
        "dynamic_friction": 0.4,
        "restitution": 0.1,
        "center_of_mass_height_ratio": 0.5,
    },
}

# Surface weight limits by region type (kg)
SURFACE_WEIGHT_LIMITS: Dict[str, float] = {
    "counter": 100.0,
    "table": 80.0,
    "shelf": 30.0,
    "drawer": 15.0,
    "cabinet": 40.0,
    "refrigerator": 50.0,
    "dishwasher": 25.0,
    "rack_level": 200.0,
    "pallet_position": 1000.0,
    "bench": 100.0,
    "desk": 50.0,
    "floor": float("inf"),  # No limit
    "default": 50.0,
}

# Surface friction coefficients
SURFACE_FRICTION: Dict[str, float] = {
    "counter": 0.4,  # Granite/laminate
    "table": 0.35,  # Wood/laminate
    "shelf": 0.3,  # Metal/wood
    "drawer": 0.25,  # Smooth drawer liner
    "cabinet": 0.35,
    "refrigerator": 0.2,  # Cold, can be condensation
    "dishwasher": 0.15,  # Wet, plastic racks
    "rack_level": 0.4,  # Metal wire
    "pallet_position": 0.5,  # Rough wood
    "bench": 0.35,
    "desk": 0.3,
    "floor": 0.5,
    "default": 0.35,
}


class PhysicsValidator:
    """Validates physics plausibility of placements.

    Checks for:
    - Object stability (won't tip over)
    - Weight constraints (surfaces can hold objects)
    - Friction (objects won't slide)
    - Stacking safety
    """

    DEFAULT_MODEL = "gemini-3-pro-preview"
    GRAVITY = 9.81  # m/s²

    def __init__(
        self,
        api_key: Optional[str] = None,
        strict_mode: bool = False,
        surface_angle_tolerance_deg: float = 5.0,
    ):
        """Initialize the physics validator.

        Args:
            api_key: Gemini API key for AI-powered analysis
            strict_mode: If True, warnings become errors
            surface_angle_tolerance_deg: Max surface angle before slip check
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.strict_mode = strict_mode
        self.surface_angle_tolerance = surface_angle_tolerance_deg
        self._client: Optional["genai.Client"] = None

    def _get_client(self) -> Optional["genai.Client"]:
        """Get or create the Gemini client."""
        if not HAVE_GENAI:
            return None

        if self._client is None and self.api_key:
            self._client = genai.Client(api_key=self.api_key)

        return self._client

    def _get_physics_properties(
        self,
        asset: AssetInstance,
    ) -> Dict[str, Any]:
        """Get physics properties for an asset.

        Args:
            asset: The asset to get properties for

        Returns:
            Dictionary of physics properties
        """
        category = asset.category.value.lower()
        return PHYSICS_DEFAULTS.get(category, PHYSICS_DEFAULTS["default"])

    def _get_surface_properties(
        self,
        region: DetectedRegion,
    ) -> Tuple[float, float]:
        """Get surface properties for a region.

        Args:
            region: The placement region

        Returns:
            Tuple of (weight_limit_kg, friction_coefficient)
        """
        region_type = region.region_type.value.lower()

        weight_limit = SURFACE_WEIGHT_LIMITS.get(
            region_type, SURFACE_WEIGHT_LIMITS["default"]
        )
        friction = SURFACE_FRICTION.get(
            region_type, SURFACE_FRICTION["default"]
        )

        # Adjust for wet surfaces
        if "wet" in region.semantic_tags:
            friction *= 0.7

        # Adjust for refrigerated/cold surfaces
        if region.region_type == RegionType.REFRIGERATOR:
            friction *= 0.9  # Slight condensation

        return weight_limit, friction

    def check_stability(
        self,
        asset: AssetInstance,
        placement: PlacementCandidate,
        region: DetectedRegion,
    ) -> StabilityCheck:
        """Check if an object placement is stable.

        Args:
            asset: The placed asset
            placement: The placement candidate
            region: The placement region

        Returns:
            StabilityCheck result
        """
        physics = self._get_physics_properties(asset)
        box = asset.bounding_box

        # Calculate center of mass
        com_height_ratio = physics.get("center_of_mass_height_ratio", 0.5)
        com_offset = [0.0, 0.0, (com_height_ratio - 0.5) * box.size[2]]

        # Calculate base dimensions
        base_width = box.size[0]
        base_depth = box.size[1]
        height = box.size[2]

        # Calculate tip angle (simplified)
        # Object tips when CoM projects outside base
        com_height = height * com_height_ratio
        min_base_dim = min(base_width, base_depth)

        if com_height > 0:
            tip_angle = math.degrees(math.atan(min_base_dim / (2 * com_height)))
        else:
            tip_angle = 90.0

        # Check if stacked
        support_ratio = 1.0
        if placement.is_stacked and placement.supporting_asset_id:
            # Stacked objects have reduced support
            support_ratio = 0.8  # Assume 80% base support when stacked

        # Determine stability level
        if tip_angle > 45 and support_ratio > 0.7:
            level = StabilityLevel.STABLE
            details = "Object has a wide base relative to height"
        elif tip_angle > 25 and support_ratio > 0.5:
            level = StabilityLevel.MARGINALLY_STABLE
            details = "Object is stable but may tip under disturbance"
        else:
            level = StabilityLevel.UNSTABLE
            details = "Object has high center of mass or insufficient support"

        return StabilityCheck(
            asset_id=asset.asset_id,
            level=level,
            tip_angle_deg=tip_angle,
            support_ratio=support_ratio,
            center_of_mass_offset=com_offset,
            details=details,
        )

    def check_weight_constraints(
        self,
        placements: List[Tuple[AssetInstance, PlacementCandidate]],
        regions: List[DetectedRegion],
    ) -> List[WeightCheck]:
        """Check weight constraints for all regions.

        Args:
            placements: List of (asset, placement) tuples
            regions: Available regions

        Returns:
            List of WeightCheck results
        """
        # Group placements by region
        region_weights: Dict[str, List[AssetInstance]] = {}
        for asset, placement in placements:
            region_id = placement.region_id
            if region_id not in region_weights:
                region_weights[region_id] = []
            region_weights[region_id].append(asset)

        # Check each region
        results: List[WeightCheck] = []
        region_map = {r.id: r for r in regions}

        for region_id, assets in region_weights.items():
            region = region_map.get(region_id)
            if not region:
                continue

            weight_limit, _ = self._get_surface_properties(region)
            total_weight = sum(a.mass_kg for a in assets)

            # Find heaviest object
            heaviest = max(assets, key=lambda a: a.mass_kg) if assets else None

            results.append(WeightCheck(
                region_id=region_id,
                total_weight_kg=total_weight,
                max_weight_kg=weight_limit,
                is_exceeded=total_weight > weight_limit,
                margin_kg=weight_limit - total_weight,
                heaviest_object=heaviest.asset_id if heaviest else None,
            ))

        return results

    def check_friction(
        self,
        asset: AssetInstance,
        placement: PlacementCandidate,
        region: DetectedRegion,
    ) -> FrictionCheck:
        """Check if friction is sufficient to prevent sliding.

        Args:
            asset: The placed asset
            placement: The placement candidate
            region: The placement region

        Returns:
            FrictionCheck result
        """
        physics = self._get_physics_properties(asset)
        _, surface_friction = self._get_surface_properties(region)

        object_friction = physics.get("static_friction", 0.5)

        # Effective friction is minimum of object and surface
        effective_friction = min(object_friction, surface_friction)

        # Calculate slip angle
        # Object slips when tan(angle) > friction coefficient
        slip_angle = math.degrees(math.atan(effective_friction))

        # Required friction based on surface angle
        surface_angle = abs(placement.rotation[0]) + abs(placement.rotation[1])
        required_friction = math.tan(math.radians(surface_angle))

        is_sufficient = effective_friction >= required_friction

        return FrictionCheck(
            asset_id=asset.asset_id,
            static_friction=effective_friction,
            required_friction=required_friction,
            is_sufficient=is_sufficient,
            slip_angle_deg=slip_angle,
        )

    def validate_placement(
        self,
        result: PlacementResult,
        region: DetectedRegion,
    ) -> ValidationResult:
        """Validate a single placement result.

        Args:
            result: The placement result to validate
            region: The placement region

        Returns:
            ValidationResult with all checks
        """
        if not result.final_position:
            return ValidationResult(
                status=ValidationStatus.VALID,
                warnings=["Placement has no position (may have failed)"],
                score=0.0,
            )

        asset = result.asset
        placement = PlacementCandidate(
            position=result.final_position,
            rotation=result.final_rotation or [0, 0, 0],
            region_id=result.region_id or "",
        )

        # Run checks
        stability = self.check_stability(asset, placement, region)
        friction = self.check_friction(asset, placement, region)

        stability_checks = [stability]
        friction_checks = [friction]
        warnings: List[str] = []
        errors: List[str] = []

        # Evaluate stability
        if stability.level == StabilityLevel.UNSTABLE:
            errors.append(f"{asset.asset_name}: {stability.details}")
        elif stability.level == StabilityLevel.MARGINALLY_STABLE:
            warnings.append(f"{asset.asset_name}: {stability.details}")

        # Evaluate friction
        if not friction.is_sufficient:
            warnings.append(
                f"{asset.asset_name} may slide (friction {friction.static_friction:.2f} "
                f"< required {friction.required_friction:.2f})"
            )

        # Calculate score
        score = 1.0
        if stability.level == StabilityLevel.UNSTABLE:
            score -= 0.5
        elif stability.level == StabilityLevel.MARGINALLY_STABLE:
            score -= 0.2

        if not friction.is_sufficient:
            score -= 0.3

        score = max(0.0, score)

        # Determine status
        if errors and self.strict_mode:
            status = ValidationStatus.UNSTABLE
        elif errors:
            status = ValidationStatus.WARNING
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        return ValidationResult(
            status=status,
            stability_checks=stability_checks,
            friction_checks=friction_checks,
            warnings=warnings,
            errors=errors,
            score=score,
        )

    def validate_plan(
        self,
        plan: PlacementPlan,
        regions: List[DetectedRegion],
    ) -> ValidationResult:
        """Validate an entire placement plan.

        Args:
            plan: The placement plan to validate
            regions: Available regions

        Returns:
            ValidationResult for the entire plan
        """
        region_map = {r.id: r for r in regions}

        all_stability: List[StabilityCheck] = []
        all_friction: List[FrictionCheck] = []
        all_warnings: List[str] = []
        all_errors: List[str] = []

        # Collect successful placements for weight check
        successful_placements: List[Tuple[AssetInstance, PlacementCandidate]] = []

        for result in plan.placements:
            if not result.final_position or not result.region_id:
                continue

            region = region_map.get(result.region_id)
            if not region:
                continue

            # Create placement candidate
            placement = PlacementCandidate(
                position=result.final_position,
                rotation=result.final_rotation or [0, 0, 0],
                region_id=result.region_id,
            )

            successful_placements.append((result.asset, placement))

            # Run stability check
            stability = self.check_stability(result.asset, placement, region)
            all_stability.append(stability)

            if stability.level == StabilityLevel.UNSTABLE:
                all_errors.append(f"{result.asset.asset_name}: {stability.details}")
            elif stability.level == StabilityLevel.MARGINALLY_STABLE:
                all_warnings.append(f"{result.asset.asset_name}: {stability.details}")

            # Run friction check
            friction = self.check_friction(result.asset, placement, region)
            all_friction.append(friction)

            if not friction.is_sufficient:
                all_warnings.append(
                    f"{result.asset.asset_name} may slide "
                    f"(friction {friction.static_friction:.2f})"
                )

        # Run weight checks
        weight_checks = self.check_weight_constraints(successful_placements, regions)

        for wc in weight_checks:
            if wc.is_exceeded:
                all_errors.append(
                    f"Region {wc.region_id}: weight exceeded "
                    f"({wc.total_weight_kg:.1f}kg > {wc.max_weight_kg:.1f}kg)"
                )

        # Calculate overall score
        total_checks = len(all_stability) + len(all_friction) + len(weight_checks)
        if total_checks > 0:
            issues = len(all_errors) + len(all_warnings) * 0.5
            score = max(0.0, 1.0 - issues / total_checks)
        else:
            score = 1.0

        # Generate AI analysis if available
        ai_analysis = self._generate_ai_analysis(
            plan, all_stability, weight_checks, all_warnings, all_errors
        )

        # Determine status
        if all_errors:
            status = ValidationStatus.UNSTABLE if self.strict_mode else ValidationStatus.WARNING
        elif all_warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        return ValidationResult(
            status=status,
            stability_checks=all_stability,
            weight_checks=weight_checks,
            friction_checks=all_friction,
            warnings=all_warnings,
            errors=all_errors,
            score=score,
            ai_analysis=ai_analysis,
        )

    def _generate_ai_analysis(
        self,
        plan: PlacementPlan,
        stability_checks: List[StabilityCheck],
        weight_checks: List[WeightCheck],
        warnings: List[str],
        errors: List[str],
    ) -> str:
        """Generate AI analysis of physics validation.

        Args:
            plan: The placement plan
            stability_checks: All stability checks
            weight_checks: All weight checks
            warnings: Warning messages
            errors: Error messages

        Returns:
            AI-generated analysis
        """
        client = self._get_client()
        if not client:
            return ""

        # Summarize issues
        unstable_count = sum(
            1 for s in stability_checks if s.level == StabilityLevel.UNSTABLE
        )
        marginal_count = sum(
            1 for s in stability_checks if s.level == StabilityLevel.MARGINALLY_STABLE
        )
        overweight_count = sum(1 for w in weight_checks if w.is_exceeded)

        prompt = f"""Analyze these physics validation results and provide recommendations.

Scene: {plan.scene_id}
Total placements: {len(plan.placements)}
Successful: {plan.total_assets_placed}

Physics Issues:
- Unstable objects: {unstable_count}
- Marginally stable: {marginal_count}
- Overweight regions: {overweight_count}

Warnings:
{json.dumps(warnings[:5], indent=2)}

Errors:
{json.dumps(errors[:5], indent=2)}

Provide 2-3 sentences with:
1. Overall physics plausibility assessment
2. Specific recommendations for fixing issues
3. Any sim2real concerns
"""

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]

            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            )

            response_text = ""
            for chunk in client.models.generate_content_stream(
                model=self.DEFAULT_MODEL,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    response_text += chunk.text

            return response_text.strip()

        except Exception as e:
            return f"AI analysis unavailable: {e}"

    def suggest_improvements(
        self,
        validation: ValidationResult,
    ) -> List[str]:
        """Suggest improvements based on validation results.

        Args:
            validation: The validation result

        Returns:
            List of improvement suggestions
        """
        suggestions: List[str] = []

        # Stability suggestions
        for check in validation.stability_checks:
            if check.level == StabilityLevel.UNSTABLE:
                suggestions.append(
                    f"Consider placing {check.asset_id} on a larger surface "
                    f"or reduce stacking height (tip angle: {check.tip_angle_deg:.1f}°)"
                )
            elif check.level == StabilityLevel.MARGINALLY_STABLE:
                suggestions.append(
                    f"Monitor {check.asset_id} - may need stabilization "
                    f"(support ratio: {check.support_ratio:.1%})"
                )

        # Weight suggestions
        for check in validation.weight_checks:
            if check.is_exceeded:
                suggestions.append(
                    f"Redistribute objects from {check.region_id} - "
                    f"over by {-check.margin_kg:.1f}kg"
                )

        # Friction suggestions
        for check in validation.friction_checks:
            if not check.is_sufficient:
                suggestions.append(
                    f"Add grip mat or adjust surface angle for {check.asset_id} "
                    f"(slip angle: {check.slip_angle_deg:.1f}°)"
                )

        return suggestions


def create_physics_validator(
    api_key: Optional[str] = None,
    strict_mode: bool = False,
) -> PhysicsValidator:
    """Factory function to create a physics validator.

    Args:
        api_key: Optional Gemini API key
        strict_mode: Whether to treat warnings as errors

    Returns:
        Configured PhysicsValidator instance
    """
    return PhysicsValidator(api_key=api_key, strict_mode=strict_mode)


if __name__ == "__main__":
    from .placement_engine import AssetInstance, BoundingBox, PlacementCandidate

    # Test the physics validator
    validator = create_physics_validator()

    # Create test asset
    test_asset = AssetInstance(
        asset_id="tall_bottle_01",
        asset_name="Tall Wine Bottle",
        category=AssetCategory.BOTTLES,
        bounding_box=BoundingBox(
            center=[0, 0, 0],
            size=[0.08, 0.08, 0.35],  # Tall and narrow
        ),
        mass_kg=1.2,
    )

    # Create test region
    test_region = DetectedRegion(
        id="counter_01",
        name="Kitchen Counter",
        region_type=RegionType.COUNTER,
        position=[0.0, 0.0, 0.9],
        size=[2.0, 0.6, 0.05],
        surface_type="horizontal",
        clearance_above_m=0.5,
        semantic_tags=[],
        suitable_for=["bottles"],
    )

    # Create test placement
    test_placement = PlacementCandidate(
        position=[0.5, 0.3, 0.95],
        rotation=[0.0, 0.0, 0.0],
        region_id="counter_01",
    )

    # Run stability check
    stability = validator.check_stability(test_asset, test_placement, test_region)
    print(f"Stability: {stability.level.value}")
    print(f"  Tip angle: {stability.tip_angle_deg:.1f}°")
    print(f"  Details: {stability.details}")

    # Run friction check
    friction = validator.check_friction(test_asset, test_placement, test_region)
    print(f"\nFriction: {'sufficient' if friction.is_sufficient else 'insufficient'}")
    print(f"  Static friction: {friction.static_friction:.2f}")
    print(f"  Slip angle: {friction.slip_angle_deg:.1f}°")
