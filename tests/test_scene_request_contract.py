from __future__ import annotations

import pytest

from tests.contract_utils import load_schema, validate_json_schema


def test_scene_request_schema_accepts_valid_text_request() -> None:
    schema = load_schema("scene_request_v1.schema.json")
    payload = {
        "schema_version": "v1",
        "scene_id": "scene_123",
        "source_mode": "text",
        "text_backend": "scenesmith",
        "prompt": "A modern kitchen for robot pick-and-place tasks",
        "quality_tier": "standard",
        "seed_count": 2,
        "provider_policy": "openai_primary",
        "fallback": {"allow_image_fallback": True},
    }
    validate_json_schema(payload, schema)


def test_scene_request_schema_accepts_openrouter_policy() -> None:
    schema = load_schema("scene_request_v1.schema.json")
    payload = {
        "schema_version": "v1",
        "scene_id": "scene_124",
        "source_mode": "text",
        "text_backend": "scenesmith",
        "prompt": "A modern kitchen for robot pick-and-place tasks",
        "quality_tier": "standard",
        "seed_count": 1,
        "provider_policy": "openrouter_qwen_primary",
    }
    validate_json_schema(payload, schema)


def test_scene_request_schema_rejects_unknown_source_mode() -> None:
    schema = load_schema("scene_request_v1.schema.json")
    payload = {
        "schema_version": "v1",
        "scene_id": "scene_123",
        "source_mode": "video",
        "prompt": "invalid mode",
    }
    with pytest.raises((ValueError, Exception)):
        validate_json_schema(payload, schema)


def test_scene_request_schema_accepts_valid_image_request() -> None:
    schema = load_schema("scene_request_v1.schema.json")
    payload = {
        "schema_version": "v1",
        "scene_id": "scene_img_001",
        "source_mode": "image",
        "image": {
            "gcs_uri": "gs://bucket/scenes/scene_img_001/images/photo.png",
        },
    }
    validate_json_schema(payload, schema)


def test_scene_request_schema_accepts_auto_mode() -> None:
    schema = load_schema("scene_request_v1.schema.json")
    payload = {
        "schema_version": "v1",
        "scene_id": "scene_auto_001",
        "source_mode": "auto",
        "text_backend": "hybrid_serial",
        "prompt": "A warehouse for robotic bin picking",
        "image": {
            "gcs_uri": "gs://bucket/scenes/scene_auto_001/images/scan.jpg",
        },
    }
    validate_json_schema(payload, schema)


def test_scene_request_schema_declares_no_additional_properties() -> None:
    """Verify the schema itself declares additionalProperties: false.

    The minimal validator fallback (when jsonschema is not installed) does not
    enforce additionalProperties at runtime, so we validate the schema
    declaration directly to ensure it will be enforced in production where
    jsonschema is available.
    """
    schema = load_schema("scene_request_v1.schema.json")
    assert schema.get("additionalProperties") is False, (
        "scene_request_v1 schema should set additionalProperties: false"
    )


def test_scene_request_schema_rejects_missing_scene_id() -> None:
    schema = load_schema("scene_request_v1.schema.json")
    payload = {
        "schema_version": "v1",
        "source_mode": "text",
        "prompt": "A room",
    }
    with pytest.raises((ValueError, Exception)):
        validate_json_schema(payload, schema)


def test_scene_request_schema_rejects_unknown_text_backend() -> None:
    schema = load_schema("scene_request_v1.schema.json")
    payload = {
        "schema_version": "v1",
        "scene_id": "scene_backend_001",
        "source_mode": "text",
        "text_backend": "unknown_backend",
        "prompt": "A room",
    }
    with pytest.raises((ValueError, Exception)):
        validate_json_schema(payload, schema)
