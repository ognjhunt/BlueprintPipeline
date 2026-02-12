from __future__ import annotations

import pytest

from tools.source_pipeline.request import (
    PipelineSourceMode,
    QualityTier,
    TextBackend,
    build_seed_scene_ids,
    build_variants_index,
    choose_primary_source_mode,
    normalize_scene_request,
    scene_request_to_dict,
    should_fallback_to_image,
)


def test_normalize_scene_request_defaults_to_text_standard() -> None:
    request = normalize_scene_request(
        {
            "schema_version": "v1",
            "scene_id": "scene_123",
            "prompt": "A realistic kitchen with manipulable objects",
        }
    )

    assert request.source_mode == PipelineSourceMode.TEXT
    assert request.text_backend == TextBackend.SCENESMITH
    assert request.quality_tier == QualityTier.STANDARD
    assert request.seed_count == 1
    assert request.provider_policy == "openai_primary"
    assert request.fallback.allow_image_fallback is True


def test_normalize_scene_request_rejects_missing_prompt_for_text_modes() -> None:
    with pytest.raises(ValueError, match="prompt is required"):
        normalize_scene_request(
            {
                "schema_version": "v1",
                "scene_id": "scene_123",
                "source_mode": "text",
            }
        )

    with pytest.raises(ValueError, match="prompt is required"):
        normalize_scene_request(
            {
                "schema_version": "v1",
                "scene_id": "scene_123",
                "source_mode": "auto",
            }
        )


def test_normalize_scene_request_rejects_missing_image_for_image_mode() -> None:
    with pytest.raises(ValueError, match="image payload is required"):
        normalize_scene_request(
            {
                "schema_version": "v1",
                "scene_id": "scene_123",
                "source_mode": "image",
            }
        )


def test_normalize_scene_request_accepts_text_backend() -> None:
    request = normalize_scene_request(
        {
            "schema_version": "v1",
            "scene_id": "scene_123",
            "source_mode": "text",
            "text_backend": "hybrid_serial",
            "prompt": "A warehouse aisle",
        }
    )
    assert request.text_backend == TextBackend.HYBRID_SERIAL


def test_normalize_scene_request_image_mode_ignores_invalid_text_backend() -> None:
    request = normalize_scene_request(
        {
            "schema_version": "v1",
            "scene_id": "scene_img",
            "source_mode": "image",
            "text_backend": "invalid_backend",
            "image": {"gcs_uri": "gs://bucket/scenes/scene_img/images/input.png"},
        }
    )
    assert request.text_backend == TextBackend.SCENESMITH


def test_build_seed_scene_ids_uses_stable_suffixes() -> None:
    assert build_seed_scene_ids("scene_abc", 3) == [
        "scene_abc-s001",
        "scene_abc-s002",
        "scene_abc-s003",
    ]


def test_should_fallback_to_image_requires_policy_image_and_failure() -> None:
    request = normalize_scene_request(
        {
            "schema_version": "v1",
            "scene_id": "scene_123",
            "source_mode": "auto",
            "prompt": "A warehouse aisle with bins",
            "image": {
                "gcs_uri": "gs://bucket/scenes/scene_123/images/source.png",
            },
            "fallback": {"allow_image_fallback": True},
        }
    )

    assert should_fallback_to_image(request, text_stage_failed=True) is True
    assert should_fallback_to_image(request, text_stage_failed=False) is False


def test_choose_primary_source_mode_resolves_auto_to_text() -> None:
    request = normalize_scene_request(
        {
            "schema_version": "v1",
            "scene_id": "scene_auto",
            "source_mode": "auto",
            "prompt": "Some scene",
        }
    )
    assert choose_primary_source_mode(request) == PipelineSourceMode.TEXT


def test_choose_primary_source_mode_preserves_text_and_image() -> None:
    text_req = normalize_scene_request(
        {
            "schema_version": "v1",
            "scene_id": "scene_t",
            "source_mode": "text",
            "prompt": "A kitchen",
        }
    )
    assert choose_primary_source_mode(text_req) == PipelineSourceMode.TEXT

    image_req = normalize_scene_request(
        {
            "schema_version": "v1",
            "scene_id": "scene_i",
            "source_mode": "image",
            "image": {"gcs_uri": "gs://b/scenes/s/images/img.png"},
        }
    )
    assert choose_primary_source_mode(image_req) == PipelineSourceMode.IMAGE


def test_build_variants_index_structure() -> None:
    request = normalize_scene_request(
        {
            "schema_version": "v1",
            "scene_id": "scene_v",
            "source_mode": "text",
            "prompt": "A kitchen",
            "seed_count": 3,
        }
    )
    child_ids = build_seed_scene_ids("scene_v", 3)
    index = build_variants_index(request, parent_scene_id="scene_v", child_scene_ids=child_ids)

    assert index["schema_version"] == "v1"
    assert index["scene_id"] == "scene_v"
    assert index["source_mode"] == "text"
    assert index["text_backend"] == "scenesmith"
    assert index["seed_count"] == 3
    assert len(index["variants"]) == 3
    assert index["variants"][0] == {"scene_id": "scene_v-s001", "seed": 1}
    assert index["variants"][2] == {"scene_id": "scene_v-s003", "seed": 3}


def test_scene_request_to_dict_roundtrip() -> None:
    original = {
        "schema_version": "v1",
        "scene_id": "scene_rt",
        "source_mode": "text",
        "prompt": "A lab bench",
        "quality_tier": "premium",
        "seed_count": 2,
        "provider_policy": "openai_primary",
        "fallback": {"allow_image_fallback": False},
        "constraints": {"room_type": "lab"},
    }
    request = normalize_scene_request(original)
    serialized = scene_request_to_dict(request)

    assert serialized["schema_version"] == "v1"
    assert serialized["scene_id"] == "scene_rt"
    assert serialized["source_mode"] == "text"
    assert serialized["text_backend"] == "scenesmith"
    assert serialized["prompt"] == "A lab bench"
    assert serialized["quality_tier"] == "premium"
    assert serialized["seed_count"] == 2
    assert serialized["constraints"] == {"room_type": "lab"}
    assert serialized["fallback"]["allow_image_fallback"] is False
    assert serialized["image"] is None


def test_normalize_rejects_seed_count_zero() -> None:
    with pytest.raises(ValueError, match="seed_count must be >= 1"):
        normalize_scene_request(
            {
                "schema_version": "v1",
                "scene_id": "scene_z",
                "source_mode": "text",
                "prompt": "A room",
                "seed_count": 0,
            }
        )


def test_normalize_rejects_seed_count_exceeds_max() -> None:
    with pytest.raises(ValueError, match="seed_count exceeds max"):
        normalize_scene_request(
            {
                "schema_version": "v1",
                "scene_id": "scene_big",
                "source_mode": "text",
                "prompt": "A room",
                "seed_count": 100,
            },
            max_seeds=16,
        )


def test_normalize_rejects_invalid_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        normalize_scene_request(
            {
                "schema_version": "v99",
                "scene_id": "scene_bad",
                "source_mode": "text",
                "prompt": "A room",
            }
        )


def test_normalize_rejects_empty_scene_id() -> None:
    with pytest.raises(ValueError, match="scene_id is required"):
        normalize_scene_request(
            {
                "schema_version": "v1",
                "scene_id": "",
                "source_mode": "text",
                "prompt": "A room",
            }
        )
