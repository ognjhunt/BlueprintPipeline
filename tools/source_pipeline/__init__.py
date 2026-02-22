"""Source-pipeline request and mapping utilities.

Keep import-time dependencies light. Heavy modules (adapter/retrieval) are
loaded lazily through ``__getattr__`` to avoid requiring optional runtime
packages when callers only need request/generator helpers.
"""

from .request import (
    PipelineSourceMode,
    QualityTier,
    TextBackend,
    SceneRequestV1,
    normalize_scene_request,
    build_seed_scene_ids,
    build_variants_index,
    scene_request_to_dict,
)
from .generator import generate_text_scene_package
from .prompt_engine import (
    PromptGenerationResult,
    build_prompt_constraints_metadata,
    compute_novelty_score,
    generate_prompt,
    load_prompt_matrix,
)

__all__ = [
    "PipelineSourceMode",
    "QualityTier",
    "TextBackend",
    "SceneRequestV1",
    "normalize_scene_request",
    "build_seed_scene_ids",
    "build_variants_index",
    "scene_request_to_dict",
    "build_manifest_layout_inventory",
    "materialize_placeholder_assets",
    "generate_text_scene_package",
    "PromptGenerationResult",
    "build_prompt_constraints_metadata",
    "compute_novelty_score",
    "generate_prompt",
    "load_prompt_matrix",
    "AssetQuerySpec",
    "RetrievalCandidate",
    "RetrievalDecision",
    "AssetRetrievalService",
    "effective_retrieval_mode",
    "update_rollout_state",
]


def __getattr__(name: str):
    if name in {"build_manifest_layout_inventory", "materialize_placeholder_assets"}:
        from .adapter import build_manifest_layout_inventory, materialize_placeholder_assets

        return {
            "build_manifest_layout_inventory": build_manifest_layout_inventory,
            "materialize_placeholder_assets": materialize_placeholder_assets,
        }[name]

    if name in {"AssetQuerySpec", "RetrievalCandidate", "RetrievalDecision", "AssetRetrievalService"}:
        from .asset_retrieval import (
            AssetQuerySpec,
            RetrievalCandidate,
            RetrievalDecision,
            AssetRetrievalService,
        )

        return {
            "AssetQuerySpec": AssetQuerySpec,
            "RetrievalCandidate": RetrievalCandidate,
            "RetrievalDecision": RetrievalDecision,
            "AssetRetrievalService": AssetRetrievalService,
        }[name]

    if name in {"effective_retrieval_mode", "update_rollout_state"}:
        from .asset_retrieval_rollout import effective_retrieval_mode, update_rollout_state

        return {
            "effective_retrieval_mode": effective_retrieval_mode,
            "update_rollout_state": update_rollout_state,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
