"""Scale Authority implementation.

Provides centralized metric scale computation and validation for BlueprintPipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np


class ScaleSource(str, Enum):
    """Source of scale authority."""
    USER_ANCHOR = "user_anchor"          # User provided known dimension
    SCALE_JOB = "scale_job"              # Computed by scale-job with calibration
    STAGE1 = "stage1"                    # From Stage 1 generation output
    LAYOUT_JOB = "layout_job"            # Historic: from legacy reconstruction
    REFERENCE_OBJECT = "reference_object" # Inferred from known object types
    DEFAULT = "default"                   # Default heuristic


@dataclass
class ScaleConfig:
    """Configuration for scene scale.

    This is the authoritative scale information that should be propagated
    to all downstream jobs.
    """
    meters_per_unit: float = 1.0         # Meters per scene unit (for USD)
    scale_factor: float = 1.0            # Multiplier to convert raw coords to meters
    source: ScaleSource = ScaleSource.DEFAULT
    confidence: float = 0.5              # 0-1, how confident we are in the scale
    reference_object: Optional[str] = None  # Object used for reference (if any)
    reference_dimension: Optional[str] = None  # e.g., "height", "width"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meters_per_unit": self.meters_per_unit,
            "scale_factor": self.scale_factor,
            "source": self.source.value,
            "confidence": self.confidence,
            "reference_object": self.reference_object,
            "reference_dimension": self.reference_dimension,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScaleConfig":
        return cls(
            meters_per_unit=data.get("meters_per_unit", 1.0),
            scale_factor=data.get("scale_factor", 1.0),
            source=ScaleSource(data.get("source", "default")),
            confidence=data.get("confidence", 0.5),
            reference_object=data.get("reference_object"),
            reference_dimension=data.get("reference_dimension"),
            notes=data.get("notes", ""),
        )


# Known reference object dimensions (in meters)
REFERENCE_DIMENSIONS: Dict[str, Dict[str, float]] = {
    # Doors
    "door": {"height": 2.0, "width": 0.9},
    "interior_door": {"height": 2.0, "width": 0.8},
    "exterior_door": {"height": 2.1, "width": 0.9},

    # Kitchen surfaces
    "countertop": {"height": 0.9, "depth": 0.6},
    "kitchen_counter": {"height": 0.9, "depth": 0.6},
    "kitchen_island": {"height": 0.9},
    "stove": {"height": 0.9, "width": 0.76},
    "oven": {"height": 0.9, "width": 0.76},

    # Kitchen appliances
    "refrigerator": {"height": 1.8, "width": 0.9, "depth": 0.7},
    "fridge": {"height": 1.8, "width": 0.9},
    "dishwasher": {"height": 0.85, "width": 0.6},
    "microwave": {"height": 0.3, "width": 0.5},

    # Furniture
    "dining_table": {"height": 0.75, "width": 1.5},
    "desk": {"height": 0.75, "width": 1.2},
    "office_desk": {"height": 0.73, "width": 1.5},
    "coffee_table": {"height": 0.45, "width": 1.0},
    "dining_chair": {"height": 0.9, "seat_height": 0.45},
    "office_chair": {"height": 1.0, "seat_height": 0.45},
    "sofa": {"height": 0.85, "depth": 0.9},
    "couch": {"height": 0.85, "depth": 0.9},
    "bed": {"height": 0.5, "width": 1.5, "length": 2.0},
    "nightstand": {"height": 0.6, "width": 0.5},

    # Bathroom
    "toilet": {"height": 0.45, "depth": 0.7},
    "sink": {"height": 0.85},
    "bathroom_sink": {"height": 0.85},
    "bathtub": {"height": 0.55, "length": 1.7},
    "shower": {"height": 2.0, "width": 0.9},

    # Laundry
    "washer": {"height": 0.85, "width": 0.6},
    "dryer": {"height": 0.85, "width": 0.6},
    "washing_machine": {"height": 0.85, "width": 0.6},

    # Cabinets
    "cabinet": {"height": 0.9},
    "upper_cabinet": {"height": 0.7, "bottom_height": 1.5},
    "lower_cabinet": {"height": 0.85},
    "bookshelf": {"height": 1.8, "width": 0.8},
    "shelf": {"depth": 0.3},

    # Common objects for scale
    "standard_doorway": {"height": 2.0, "width": 0.9},
    "window": {"height": 1.2},
    "ceiling": {"height": 2.7},  # Typical residential
}


class ScaleAuthority:
    """Centralized scale authority for BlueprintPipeline.

    Computes and validates metric scale from multiple sources,
    selecting the most authoritative and reliable.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SCALE-AUTHORITY] {msg}")

    def compute_scale(
        self,
        layout: Optional[Dict[str, Any]] = None,
        manifest: Optional[Dict[str, Any]] = None,
        user_anchor: Optional[Dict[str, float]] = None,
        stage1_scale: Optional[float] = None,
        reference_objects: Optional[List[str]] = None,
        trust_stage1: bool = False,
    ) -> ScaleConfig:
        """Compute authoritative scale configuration.

        Args:
            layout: Scene layout data (from Stage 1 or legacy sources)
            manifest: Scene manifest data
            user_anchor: User-provided scale anchor {object_id: dimension_m}
            stage1_scale: Scale factor from Stage 1 output (if available)
            reference_objects: List of object types to use for scale inference
            trust_stage1: If True, prefer Stage 1 scale

        Returns:
            ScaleConfig with authoritative scale information
        """
        # 1. User anchor is most authoritative
        if user_anchor:
            config = self._compute_from_user_anchor(user_anchor, layout, manifest)
            if config:
                return config

        # 2. If we have a previous scale-job result, use that
        if layout and "scale_factor" in layout:
            scale = layout.get("scale_factor", 1.0)
            if scale != 1.0:
                return ScaleConfig(
                    meters_per_unit=scale,
                    scale_factor=scale,
                    source=ScaleSource.SCALE_JOB,
                    confidence=0.8,
                    notes="From previous scale-job calibration",
                )

        # 3. Stage 1 scale (if trusted)
        if trust_stage1 and stage1_scale is not None:
            return ScaleConfig(
                meters_per_unit=stage1_scale,
                scale_factor=stage1_scale,
                source=ScaleSource.STAGE1,
                confidence=0.7,
                notes="From Stage 1 generation output",
            )

        # 4. Infer from reference objects
        if reference_objects and (layout or manifest):
            config = self._compute_from_reference_objects(
                reference_objects, layout, manifest
            )
            if config:
                return config

        # 5. Auto-detect reference objects and compute scale
        config = self._auto_compute_from_objects(layout, manifest)
        if config:
            return config

        # 6. Default
        self.log("Using default scale (1.0 meters_per_unit)")
        return ScaleConfig(
            meters_per_unit=1.0,
            scale_factor=1.0,
            source=ScaleSource.DEFAULT,
            confidence=0.3,
            notes="Default scale - no reference objects found",
        )

    def _compute_from_user_anchor(
        self,
        user_anchor: Dict[str, float],
        layout: Optional[Dict],
        manifest: Optional[Dict],
    ) -> Optional[ScaleConfig]:
        """Compute scale from user-provided anchor."""
        for obj_id, expected_dim_m in user_anchor.items():
            # Find object in layout or manifest
            obj_data = self._find_object(obj_id, layout, manifest)
            if not obj_data:
                continue

            # Get current dimension
            current_dim = self._get_largest_dimension(obj_data)
            if current_dim is None or current_dim < 1e-6:
                continue

            scale = expected_dim_m / current_dim
            self.log(f"User anchor: {obj_id} -> scale={scale:.4f}")

            return ScaleConfig(
                meters_per_unit=scale,
                scale_factor=scale,
                source=ScaleSource.USER_ANCHOR,
                confidence=0.95,
                reference_object=obj_id,
                reference_dimension="user_specified",
                notes=f"User specified {obj_id} = {expected_dim_m}m",
            )

        return None

    def _compute_from_reference_objects(
        self,
        reference_objects: List[str],
        layout: Optional[Dict],
        manifest: Optional[Dict],
    ) -> Optional[ScaleConfig]:
        """Compute scale from specified reference object types."""
        scale_estimates = []

        objects = self._get_all_objects(layout, manifest)

        for obj in objects:
            category = (obj.get("class_name") or obj.get("category") or "").lower()
            obj_id = obj.get("id")

            for ref_type in reference_objects:
                if ref_type.lower() in category:
                    ref_dims = REFERENCE_DIMENSIONS.get(ref_type.lower())
                    if not ref_dims:
                        continue

                    current_dim = self._get_largest_dimension(obj)
                    if current_dim is None or current_dim < 1e-6:
                        continue

                    # Use height if available, otherwise largest known dimension
                    expected = ref_dims.get("height") or max(ref_dims.values())
                    scale = expected / current_dim

                    scale_estimates.append({
                        "scale": scale,
                        "object_id": obj_id,
                        "reference_type": ref_type,
                        "expected_m": expected,
                        "current": current_dim,
                    })
                    break

        if not scale_estimates:
            return None

        # Use median scale estimate for robustness
        scales = [e["scale"] for e in scale_estimates]
        median_scale = float(np.median(scales))

        # Find the estimate closest to median
        best = min(scale_estimates, key=lambda e: abs(e["scale"] - median_scale))

        self.log(f"Reference object scale: {best['reference_type']} -> scale={median_scale:.4f}")

        return ScaleConfig(
            meters_per_unit=median_scale,
            scale_factor=median_scale,
            source=ScaleSource.REFERENCE_OBJECT,
            confidence=min(0.7, 0.5 + 0.1 * len(scale_estimates)),
            reference_object=best["object_id"],
            reference_dimension="height",
            notes=f"From {len(scale_estimates)} reference objects: {[e['reference_type'] for e in scale_estimates]}",
        )

    def _auto_compute_from_objects(
        self,
        layout: Optional[Dict],
        manifest: Optional[Dict],
    ) -> Optional[ScaleConfig]:
        """Automatically detect reference objects and compute scale."""
        # Priority order for auto-detection
        priority_refs = [
            "door", "refrigerator", "fridge", "countertop", "kitchen_counter",
            "dishwasher", "oven", "stove", "washer", "dryer",
            "desk", "dining_table", "bed", "sofa", "couch",
            "cabinet", "bookshelf", "toilet", "sink",
        ]

        return self._compute_from_reference_objects(priority_refs, layout, manifest)

    def _find_object(
        self,
        obj_id: str,
        layout: Optional[Dict],
        manifest: Optional[Dict],
    ) -> Optional[Dict]:
        """Find object by ID in layout or manifest."""
        if layout:
            for obj in layout.get("objects", []):
                if str(obj.get("id")) == str(obj_id):
                    return obj

        if manifest:
            for obj in manifest.get("objects", []):
                if str(obj.get("id")) == str(obj_id):
                    return obj

        return None

    def _get_all_objects(
        self,
        layout: Optional[Dict],
        manifest: Optional[Dict],
    ) -> List[Dict]:
        """Get all objects from layout and manifest."""
        objects = []
        seen_ids = set()

        if layout:
            for obj in layout.get("objects", []):
                obj_id = obj.get("id")
                if obj_id and obj_id not in seen_ids:
                    objects.append(obj)
                    seen_ids.add(obj_id)

        if manifest:
            for obj in manifest.get("objects", []):
                obj_id = obj.get("id")
                if obj_id and obj_id not in seen_ids:
                    objects.append(obj)
                    seen_ids.add(obj_id)

        return objects

    def _get_largest_dimension(self, obj: Dict) -> Optional[float]:
        """Get the largest dimension of an object."""
        # Try OBB extents first (half-sizes)
        obb = obj.get("obb", {})
        extents = obb.get("extents")
        if extents and len(extents) == 3:
            return 2.0 * max(extents)  # Convert half-extents to full size

        # Try bounds
        bounds = obj.get("bounds", {})
        min_pt = bounds.get("min")
        max_pt = bounds.get("max")
        if min_pt and max_pt and len(min_pt) == 3 and len(max_pt) == 3:
            sizes = [max_pt[i] - min_pt[i] for i in range(3)]
            return max(sizes)

        # Try dimensions_est
        dims = obj.get("dimensions_est", {})
        if dims:
            values = [dims.get("width", 0), dims.get("height", 0), dims.get("depth", 0)]
            if max(values) > 0:
                return max(values)

        return None


# =============================================================================
# Utility Functions
# =============================================================================


def apply_scale_to_manifest(
    manifest: Dict[str, Any],
    scale_config: ScaleConfig,
) -> Dict[str, Any]:
    """Apply scale configuration to a manifest.

    Updates:
    - scene.meters_per_unit
    - metadata.scale_authority
    """
    manifest = dict(manifest)  # Shallow copy

    if "scene" not in manifest:
        manifest["scene"] = {}

    manifest["scene"]["meters_per_unit"] = scale_config.meters_per_unit

    if "metadata" not in manifest:
        manifest["metadata"] = {}

    manifest["metadata"]["scale_authority"] = scale_config.to_dict()

    return manifest


def apply_scale_to_layout(
    layout: Dict[str, Any],
    scale_config: ScaleConfig,
) -> Dict[str, Any]:
    """Apply scale configuration to a layout.

    Updates:
    - scale_factor
    - meters_per_unit
    - metadata.scale_authority
    - Optionally scales all object positions/sizes
    """
    layout = dict(layout)  # Shallow copy

    layout["scale_factor"] = scale_config.scale_factor
    layout["meters_per_unit"] = scale_config.meters_per_unit

    if "metadata" not in layout:
        layout["metadata"] = {}

    layout["metadata"]["scale_authority"] = scale_config.to_dict()

    return layout


def validate_scale(
    layout: Optional[Dict] = None,
    manifest: Optional[Dict] = None,
    tolerance: float = 0.2,
) -> Tuple[bool, List[str]]:
    """Validate scale by checking object dimensions against expected ranges.

    Returns:
        (is_valid, list of warnings)
    """
    warnings = []
    authority = ScaleAuthority(verbose=False)

    objects = authority._get_all_objects(layout, manifest)

    for obj in objects:
        category = (obj.get("class_name") or obj.get("category") or "").lower()
        largest_dim = authority._get_largest_dimension(obj)

        if largest_dim is None:
            continue

        # Check against known reference dimensions
        for ref_type, ref_dims in REFERENCE_DIMENSIONS.items():
            if ref_type in category:
                expected_height = ref_dims.get("height")
                if expected_height:
                    ratio = largest_dim / expected_height
                    if ratio < (1 - tolerance) or ratio > (1 + tolerance):
                        warnings.append(
                            f"{obj.get('id')}: {category} height={largest_dim:.2f}m "
                            f"(expected ~{expected_height:.2f}m, ratio={ratio:.2f})"
                        )
                break

    # General sanity checks
    for obj in objects:
        largest_dim = authority._get_largest_dimension(obj)
        if largest_dim is not None:
            # Objects shouldn't be microscopic or building-sized
            if largest_dim < 0.01:
                warnings.append(f"{obj.get('id')}: suspiciously small ({largest_dim:.4f}m)")
            elif largest_dim > 10:
                warnings.append(f"{obj.get('id')}: suspiciously large ({largest_dim:.2f}m)")

    is_valid = len(warnings) == 0
    return is_valid, warnings
