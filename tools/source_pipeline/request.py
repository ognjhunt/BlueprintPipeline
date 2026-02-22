from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class PipelineSourceMode(str, Enum):
    """Source selection mode for Stage 1 generation."""

    TEXT = "text"
    IMAGE = "image"
    AUTO = "auto"


class QualityTier(str, Enum):
    """Cost/quality preset for source generation."""

    STANDARD = "standard"
    PREMIUM = "premium"


class TextBackend(str, Enum):
    """Stage 1 text backend selector."""

    INTERNAL = "internal"
    SCENESMITH = "scenesmith"
    SAGE = "sage"
    HYBRID_SERIAL = "hybrid_serial"


@dataclass(frozen=True)
class SceneRequestImage:
    """Optional image fallback payload."""

    gcs_uri: str
    generation: Optional[str] = None


@dataclass(frozen=True)
class SceneRequestFallback:
    """Fallback options for request execution."""

    allow_image_fallback: bool = True


@dataclass(frozen=True)
class SceneRequestV1:
    """Canonical v1 request payload for text/image source selection."""

    schema_version: str
    scene_id: str
    source_mode: PipelineSourceMode = PipelineSourceMode.TEXT
    text_backend: TextBackend = TextBackend.SCENESMITH
    prompt: Optional[str] = None
    quality_tier: QualityTier = QualityTier.STANDARD
    seed_count: int = 1
    image: Optional[SceneRequestImage] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    provider_policy: str = "openrouter_qwen_primary"
    fallback: SceneRequestFallback = field(default_factory=SceneRequestFallback)


_ALLOWED_SCHEMA_VERSIONS = {"v1"}
_ALLOWED_PROVIDER_POLICIES = {"openai_primary", "openrouter_qwen_primary"}


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _parse_source_mode(raw: Any, default_source_mode: PipelineSourceMode) -> PipelineSourceMode:
    if raw is None:
        return default_source_mode
    try:
        return PipelineSourceMode(str(raw).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"source_mode must be one of {[m.value for m in PipelineSourceMode]}, got {raw!r}"
        ) from exc


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
    strict: bool,
) -> TextBackend:
    if raw is None:
        return default_text_backend
    text_backend_raw = str(raw).strip().lower()
    if text_backend_raw == "":
        return default_text_backend
    try:
        return TextBackend(text_backend_raw)
    except ValueError as exc:
        if not strict:
            return default_text_backend
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


def _parse_image(raw: Any) -> Optional[SceneRequestImage]:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("image must be an object with gcs_uri and optional generation")
    gcs_uri = str(raw.get("gcs_uri", "")).strip()
    if not gcs_uri:
        raise ValueError("image.gcs_uri is required when image is provided")
    generation_raw = raw.get("generation")
    generation = str(generation_raw).strip() if generation_raw is not None else None
    if generation == "":
        generation = None
    return SceneRequestImage(gcs_uri=gcs_uri, generation=generation)


def normalize_scene_request(
    payload: Mapping[str, Any],
    *,
    default_source_mode: PipelineSourceMode = PipelineSourceMode.TEXT,
    default_text_backend: TextBackend = TextBackend.SCENESMITH,
    max_seeds: int = 16,
) -> SceneRequestV1:
    """Validate and normalize scene_request.json payload into SceneRequestV1."""

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
        strict=source_mode != PipelineSourceMode.IMAGE,
    )
    quality_tier = _parse_quality_tier(payload.get("quality_tier"))
    seed_count = _parse_seed_count(payload.get("seed_count"), max_seeds)
    image = _parse_image(payload.get("image"))

    prompt_raw = payload.get("prompt")
    prompt = str(prompt_raw).strip() if prompt_raw is not None else None
    if prompt == "":
        prompt = None

    if source_mode in {PipelineSourceMode.TEXT, PipelineSourceMode.AUTO} and not prompt:
        raise ValueError("prompt is required when source_mode is text or auto")

    if source_mode == PipelineSourceMode.IMAGE and image is None:
        raise ValueError("image payload is required when source_mode is image")

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

    fallback_raw = payload.get("fallback")
    allow_fallback = True
    if isinstance(fallback_raw, Mapping):
        allow_fallback = _as_bool(fallback_raw.get("allow_image_fallback"), default=True)
    elif fallback_raw is not None:
        raise ValueError("fallback must be an object when provided")

    return SceneRequestV1(
        schema_version=schema_version,
        scene_id=scene_id,
        source_mode=source_mode,
        text_backend=text_backend,
        prompt=prompt,
        quality_tier=quality_tier,
        seed_count=seed_count,
        image=image,
        constraints=constraints,
        provider_policy=provider_policy,
        fallback=SceneRequestFallback(allow_image_fallback=allow_fallback),
    )


def build_seed_scene_ids(scene_id: str, seed_count: int) -> List[str]:
    """Build stable child scene IDs for multi-seed fanout."""

    if seed_count < 1:
        raise ValueError("seed_count must be >= 1")
    child_ids: List[str] = []
    for idx in range(1, seed_count + 1):
        child_ids.append(f"{scene_id}-s{idx:03d}")
    return child_ids


def choose_primary_source_mode(request: SceneRequestV1) -> PipelineSourceMode:
    """Resolve the primary source for execution. AUTO resolves to TEXT first."""

    if request.source_mode == PipelineSourceMode.AUTO:
        return PipelineSourceMode.TEXT
    return request.source_mode


def should_fallback_to_image(
    request: SceneRequestV1,
    *,
    text_stage_failed: bool,
) -> bool:
    """Return whether the orchestrator should fallback to image execution."""

    if not text_stage_failed:
        return False
    if not request.fallback.allow_image_fallback:
        return False
    if request.image is None:
        return False
    if request.source_mode == PipelineSourceMode.IMAGE:
        return False
    return True


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

    image_payload = None
    if request.image is not None:
        image_payload = {
            "gcs_uri": request.image.gcs_uri,
            "generation": request.image.generation,
        }

    return {
        "schema_version": request.schema_version,
        "scene_id": request.scene_id,
        "source_mode": request.source_mode.value,
        "text_backend": request.text_backend.value,
        "prompt": request.prompt,
        "quality_tier": request.quality_tier.value,
        "seed_count": request.seed_count,
        "image": image_payload,
        "constraints": dict(request.constraints),
        "provider_policy": request.provider_policy,
        "fallback": {
            "allow_image_fallback": request.fallback.allow_image_fallback,
        },
    }
