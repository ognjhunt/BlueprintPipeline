"""Scale Authority - Centralized metric scale management for BlueprintPipeline.

This module provides a single source of truth for metric scale throughout the pipeline.
All jobs that deal with physical dimensions should use this module to ensure consistency.

Scale Authority Hierarchy (from most to least authoritative):
1. User-provided scale anchor (e.g., known reference object dimension)
2. Calibrated scale factor from scale-job
3. Stage 1 scale (if trusted)
4. Default heuristics (e.g., door height ~2m, countertop ~0.9m)

Key Outputs:
- meters_per_unit: The authoritative scale factor (written to manifest + USD)
- scale_factor: Multiplier to convert from raw coordinates to meters
- scale_source: Documentation of where scale came from

Usage:
    from tools.scale_authority import ScaleAuthority, ScaleConfig

    authority = ScaleAuthority()
    config = authority.compute_scale(
        layout=layout_data,
        manifest=manifest_data,
        reference_objects=["door", "countertop"],
    )

    # Apply to manifest
    manifest["scene"]["meters_per_unit"] = config.meters_per_unit

    # Apply to layout
    layout["scale_factor"] = config.scale_factor
"""

from .authority import (
    ScaleAuthority,
    ScaleConfig,
    ScaleSource,
    apply_scale_to_manifest,
    apply_scale_to_layout,
    validate_scale,
)

__all__ = [
    "ScaleAuthority",
    "ScaleConfig",
    "ScaleSource",
    "apply_scale_to_manifest",
    "apply_scale_to_layout",
    "validate_scale",
]
