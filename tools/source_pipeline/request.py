from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping


class PipelineSourceMode(str, Enum):
    """Source selection mode for Stage 1 generation."""

    TEXT = "text"


class QualityTier(str, Enum):
    """Cost/quality preset for source generation."""

    STANDARD = "standard"
    PREMIUM = "premium"


class TextBackend(str, Enum):
    """Stage 1 text backend selector."""

    SCENESMITH = "scenesmith"
    SAGE = "sage"
    HYBRID_SERIAL = "hybrid_serial"


@dataclass(frozen=True)
class SceneRequestV1:
    """Canonical v1 text-only request payload."""

    schema_version: str
    scene_id: str
    source_mode: PipelineSourceMode = PipelineSourceMode.TEXT
    text_backend: TextBackend = TextBackend.HYBRID_SERIAL
    prompt: str = ""
    quality_tier: QualityTier = QualityTier.STANDARD
    seed_count: int = 1
    constraints: Dict[str, Any] = field(default_factory=dict)
    provider_policy: str = "openrouter_qwen_primary"


_ALLOWED_SCHEMA_VERSIONS = {"v1"}
_ALLOWED_PROVIDER_POLICIES = {"openai_primary", "openrouter_qwen_primary"}


def _parse_source_mode(raw: Any, default_source_mode: PipelineSourceMode) -> PipelineSourceMode:
    if raw is None:
        return default_source_mode
    try:
        source_mode = PipelineSourceMode(str(raw).strip().lower())
    except ValueError as exc:
        raise ValueError("source_mode must be 'text'") from exc
    if source_mode != PipelineSourceMode.TEXT:
        raise ValueError("source_mode must be 'text'")
    return source_mode


def _parse_quality_tier(raw: Any) -> QualityTier:
    if raw is None:
        return QualityTier.STANDARD
    try:
        return QualityTier(str(raw).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"quality_tier must be one of {[t.value for t in QualityTier]}, got {raw!r}"
        ) from exc


def _parse_text_backend(
    raw: Any,
    *,
    default_text_backend: TextBackend,
) -> TextBackend:
    if raw is None:
        return default_text_backend
    text_backend_raw = str(raw).strip().lower()
    if text_backend_raw == "":
        return default_text_backend
    try:
        return TextBackend(text_backend_raw)
    except ValueError as exc:
        raise ValueError(
            f"text_backend must be one of {[t.value for t in TextBackend]}, got {raw!r}"
        ) from exc


def _parse_seed_count(raw: Any, max_seeds: int) -> int:
    if raw is None:
        return 1
    try:
        seed_count = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed_count must be an integer >= 1, got {raw!r}") from exc
    if seed_count < 1:
        raise ValueError(f"seed_count must be >= 1, got {seed_count}")
    if seed_count > max_seeds:
        raise ValueError(f"seed_count exceeds max supported ({max_seeds}), got {seed_count}")
    return seed_count


def normalize_scene_request(
    payload: Mapping[str, Any],
    *,
    default_source_mode: PipelineSourceMode = PipelineSourceMode.TEXT,
    default_text_backend: TextBackend = TextBackend.HYBRID_SERIAL,
    max_seeds: int = 16,
) -> SceneRequestV1:
    """Validate and normalize scene_request.json payload into SceneRequestV1."""

    if default_source_mode != PipelineSourceMode.TEXT:
        raise ValueError("default_source_mode must be 'text'")

    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version not in _ALLOWED_SCHEMA_VERSIONS:
        raise ValueError(
            f"schema_version must be one of {sorted(_ALLOWED_SCHEMA_VERSIONS)}, got {schema_version!r}"
        )

    scene_id = str(payload.get("scene_id", "")).strip()
    if not scene_id:
        raise ValueError("scene_id is required")

    source_mode = _parse_source_mode(payload.get("source_mode"), default_source_mode)
    text_backend = _parse_text_backend(
        payload.get("text_backend"),
        default_text_backend=default_text_backend,
    )
    quality_tier = _parse_quality_tier(payload.get("quality_tier"))
    seed_count = _parse_seed_count(payload.get("seed_count"), max_seeds)

    prompt_raw = payload.get("prompt")
    prompt = str(prompt_raw).strip() if prompt_raw is not None else ""
    if not prompt:
        raise ValueError("prompt is required when source_mode is text")

    constraints_raw = payload.get("constraints")
    if constraints_raw is None:
        constraints: Dict[str, Any] = {}
    elif isinstance(constraints_raw, Mapping):
        constraints = dict(constraints_raw)
    else:
        raise ValueError("constraints must be an object when provided")

    provider_policy = (
        str(payload.get("provider_policy", "openrouter_qwen_primary")).strip()
        or "openrouter_qwen_primary"
    )
    if provider_policy not in _ALLOWED_PROVIDER_POLICIES:
        raise ValueError(
            f"provider_policy must be one of {sorted(_ALLOWED_PROVIDER_POLICIES)}, got {provider_policy!r}"
        )

    return SceneRequestV1(
        schema_version=schema_version,
        scene_id=scene_id,
        source_mode=source_mode,
        text_backend=text_backend,
        prompt=prompt,
        quality_tier=quality_tier,
        seed_count=seed_count,
        constraints=constraints,
        provider_policy=provider_policy,
    )


def build_seed_scene_ids(scene_id: str, seed_count: int) -> List[str]:
    """Build stable child scene IDs for multi-seed fanout."""

    if seed_count < 1:
        raise ValueError("seed_count must be >= 1")
    child_ids: List[str] = []
    for idx in range(1, seed_count + 1):
        child_ids.append(f"{scene_id}-s{idx:03d}")
    return child_ids


def build_variants_index(
    request: SceneRequestV1,
    *,
    parent_scene_id: str,
    child_scene_ids: List[str],
) -> Dict[str, Any]:
    """Build parent variants index payload for multi-seed fanout."""

    return {
        "schema_version": "v1",
        "scene_id": parent_scene_id,
        "source_mode": request.source_mode.value,
        "text_backend": request.text_backend.value,
        "quality_tier": request.quality_tier.value,
        "provider_policy": request.provider_policy,
        "seed_count": len(child_scene_ids),
        "variants": [
            {
                "scene_id": child_scene_id,
                "seed": idx + 1,
            }
            for idx, child_scene_id in enumerate(child_scene_ids)
        ],
    }


def scene_request_to_dict(request: SceneRequestV1) -> Dict[str, Any]:
    """Serialize a normalized request for downstream job artifacts."""

    return {
        "schema_version": request.schema_version,
        "scene_id": request.scene_id,
        "source_mode": request.source_mode.value,
        "text_backend": request.text_backend.value,
        "prompt": request.prompt,
        "quality_tier": request.quality_tier.value,
        "seed_count": request.seed_count,
        "constraints": dict(request.constraints),
        "provider_policy": request.provider_policy,
    }
