"""Source-pipeline request and mapping utilities."""

from .request import (
    PipelineSourceMode,
    QualityTier,
    SceneRequestV1,
    SceneRequestImage,
    SceneRequestFallback,
    normalize_scene_request,
    build_seed_scene_ids,
    should_fallback_to_image,
    build_variants_index,
    scene_request_to_dict,
    choose_primary_source_mode,
)
from .adapter import (
    build_manifest_layout_inventory,
    materialize_placeholder_assets,
)
from .generator import generate_text_scene_package

__all__ = [
    "PipelineSourceMode",
    "QualityTier",
    "SceneRequestV1",
    "SceneRequestImage",
    "SceneRequestFallback",
    "normalize_scene_request",
    "build_seed_scene_ids",
    "should_fallback_to_image",
    "build_variants_index",
    "scene_request_to_dict",
    "choose_primary_source_mode",
    "build_manifest_layout_inventory",
    "materialize_placeholder_assets",
    "generate_text_scene_package",
]
