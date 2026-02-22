"""Variation Assets Contract - Standardized naming and metadata for domain randomization.

This module defines the standard contract for variation assets used in Replicator
domain randomization. It ensures consistent:

1. Asset naming and directory structure
2. Metadata format (physics, materials, semantics)
3. Replicator script references

Contract Structure:
    scenes/{scene_id}/variation_assets/
        manifest.json              # Master list of variation assets
        {asset_name}/
            reference.png          # Reference image (from variation-gen-job)
            asset.glb              # 3D model (from Stage 1 text generation)
            asset.usdz             # USD-converted model (from usd-assembly)
            simready.usda          # Physics-ready wrapper (from simready-job)
            metadata.json          # Physics + semantic metadata

Manifest Structure:
    {
        "version": "1.0.0",
        "scene_id": "...",
        "scene_type": "kitchen",
        "environment_type": "kitchen",
        "policies": ["dish_loading", "table_clearing"],
        "assets": [
            {
                "name": "dirty_plate_01",
                "category": "dishes",
                "semantic_class": "dish",
                "description": "...",
                "priority": "required|recommended|optional",
                "source_hint": "generate|catalog|import",
                "example_variants": ["..."],
                "physics_hints": {
                    "mass_range_kg": [0.3, 0.5],
                    "material_type": "ceramic",
                    "collision_shape": "box",
                    "graspable": true
                },
                "material_hint": "ceramic, porcelain",
                "style_hint": "...",
                "generation_prompt_hint": "..."
            }
        ],
        "metadata": {
            "generated_at": "...",
            "source_pipeline": "replicator-job"
        }
    }
"""

from .contract import (
    VariationAssetContract,
    VariationAssetSpec,
    VariationManifest,
    validate_variation_manifest,
    create_variation_manifest,
    standardize_asset_name,
    get_asset_paths,
)

__all__ = [
    "VariationAssetContract",
    "VariationAssetSpec",
    "VariationManifest",
    "validate_variation_manifest",
    "create_variation_manifest",
    "standardize_asset_name",
    "get_asset_paths",
]
