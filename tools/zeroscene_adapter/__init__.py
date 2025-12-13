"""ZeroScene adapter for BlueprintPipeline integration.

This module provides adapters to convert ZeroScene outputs into the formats
expected by the BlueprintPipeline jobs (usd-assembly, simready, replicator, etc.).

ZeroScene (https://arxiv.org/html/2509.23607v1) provides:
- Instance segmentation + depth extraction
- Object pose optimization using 3D/2D projection losses
- Foreground + background mesh reconstruction
- PBR material estimation for improved rendering realism
- Triangle mesh outputs for each object

This adapter converts those outputs into:
- scene_manifest.json (canonical manifest)
- scene_layout_scaled.json (layout with transforms)
- Per-object asset folders following SimReady conventions
"""

from .adapter import (
    ZeroSceneAdapter,
    manifest_from_zeroscene,
    layout_from_zeroscene,
    ZeroSceneOutput,
    ZeroSceneObject,
    ZeroScenePose,
    ZeroSceneMaterial,
)

__all__ = [
    "ZeroSceneAdapter",
    "manifest_from_zeroscene",
    "layout_from_zeroscene",
    "ZeroSceneOutput",
    "ZeroSceneObject",
    "ZeroScenePose",
    "ZeroSceneMaterial",
]
