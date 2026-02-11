from __future__ import annotations

import pytest

from tests.contract_utils import load_schema, validate_json_schema


def test_scene_request_schema_accepts_valid_text_request() -> None:
    schema = load_schema("scene_request_v1.schema.json")
    payload = {
        "schema_version": "v1",
        "scene_id": "scene_123",
        "source_mode": "text",
        "prompt": "A modern kitchen for robot pick-and-place tasks",
        "quality_tier": "standard",
        "seed_count": 2,
        "provider_policy": "openai_primary",
        "fallback": {"allow_image_fallback": True},
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
    with pytest.raises(Exception):
        validate_json_schema(payload, schema)
