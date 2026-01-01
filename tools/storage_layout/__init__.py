"""Storage Layout - Canonical directory structure for BlueprintPipeline.

This module defines the standardized storage layout used across all jobs.
All jobs should use these constants to ensure interoperability.

Canonical Layout:
    gs://{BUCKET}/scenes/{scene_id}/
        input/
            room.jpg                      # Original uploaded image
        seg/
            inventory.json                # Semantic inventory (categories, roles, hints)
        assets/
            scene_manifest.json           # Canonical manifest (Definition of Done requirement)
            scene_assets.json             # Legacy format (fallback)
            obj_{id}/                     # Per-object asset directories
                asset.glb                 # Original mesh
                asset.usdz                # Converted USD
                simready.usda             # Physics-ready wrapper
                metadata.json             # Asset metadata
            interactive/                  # Interactive/articulated assets
                obj_{id}/
                    {id}.urdf             # URDF with joints
                    articulated.usda      # USD articulation
        layout/
            scene_layout_scaled.json      # Scaled layout with transforms
        usd/
            scene.usda                    # Final assembled scene (Definition of Done requirement)
        replicator/                       # Replicator bundle (Definition of Done requirement)
            placement_regions.usda        # Placement regions
            bundle_metadata.json          # Bundle config
            policies/                     # Policy scripts
                {policy}.py
            variation_assets/
                manifest.json             # Variation asset manifest
        variation_assets/                 # Generated variation assets
            {asset_name}/
                reference.png             # Reference image
                asset.glb                 # 3D model
                simready.usda             # Physics wrapper
                metadata.json
        isaac_lab/                        # Isaac Lab tasks (Definition of Done requirement)
            env_cfg.py                    # Environment config
            task_{policy}.py              # Task implementation
            train_cfg.yaml                # Training config
            randomizations.py             # Domain randomization hooks
            reward_functions.py           # Reward implementations

Pipeline Jobs and Their Output Locations:
    regen3d-job      → assets/scene_manifest.json, layout/scene_layout_scaled.json, seg/inventory.json
    scale-job        → layout/scene_layout_scaled.json (updated)
    interactive-job  → assets/interactive/obj_{id}/*.urdf
    simready-job     → assets/obj_{id}/simready.usda
    usd-assembly-job → usd/scene.usda
    replicator-job   → replicator/*
    variation-gen-job → variation_assets/*
    isaac-lab-job    → isaac_lab/*
"""

from .paths import (
    STORAGE_VERSION,
    StorageLayout,
    ScenePaths,
    get_scene_paths,
    get_asset_path,
    get_manifest_path,
    get_layout_path,
    get_usd_path,
    get_replicator_path,
    get_isaac_lab_path,
    validate_scene_structure,
)

__all__ = [
    "STORAGE_VERSION",
    "StorageLayout",
    "ScenePaths",
    "get_scene_paths",
    "get_asset_path",
    "get_manifest_path",
    "get_layout_path",
    "get_usd_path",
    "get_replicator_path",
    "get_isaac_lab_path",
    "validate_scene_structure",
]
