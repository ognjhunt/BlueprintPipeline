"""Articulation Wiring - Connects interactive-job outputs to USD assembly.

This module provides utilities to wire articulated assets from interactive-job
into the final scene.usda. It handles:

1. Finding articulated assets (URDF, USD) from interactive-job outputs
2. Converting URDF to USD articulation if needed
3. Updating manifest with articulation metadata
4. Wiring articulated references into scene.usda

Pipeline Position:
    interactive-job → [THIS MODULE] → scene.usda with articulated objects

Expected interactive-job outputs:
    assets/interactive/obj_{id}/
        {id}.urdf                  # URDF with joint definitions
        part.glb or mesh.glb      # Segmented mesh parts
        interactive_manifest.json  # Articulation metadata

This module produces:
    - Updated scene.usda with articulated object references
    - Manifest updates with articulation metadata
    - Validation that joints are accessible in PhysX
"""

from .wiring import (
    ArticulationWiring,
    find_articulated_assets,
    wire_articulation_to_scene,
    update_manifest_with_articulation,
    ArticulatedAsset,
    JointInfo,
)

__all__ = [
    "ArticulationWiring",
    "find_articulated_assets",
    "wire_articulation_to_scene",
    "update_manifest_with_articulation",
    "ArticulatedAsset",
    "JointInfo",
]
