from __future__ import annotations

import pytest

from tools.source_pipeline.request import (
    PipelineSourceMode,
    QualityTier,
    build_seed_scene_ids,
    normalize_scene_request,
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
