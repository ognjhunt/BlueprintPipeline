import hashlib
import importlib.util
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def submit_module() -> types.ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "genie-sim-submit-job" / "submit_to_geniesim.py"
    spec = importlib.util.spec_from_file_location("submit_to_geniesim", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_canary_tag_match_triggers_assignment(submit_module: types.ModuleType) -> None:
    result = submit_module._resolve_canary_assignment(
        scene_id="scene-123",
        scene_tags=["fast", "indoor"],
        canary_enabled=True,
        canary_tags=["fast", "night"],
        canary_scene_ids=[],
        canary_percent=0,
    )

    assert result["is_canary"] is True
    assert result["matched_tags"] == ["fast"]
    assert "tag_match" in result["match_reasons"]


def test_canary_scene_id_allowlist_triggers_assignment(submit_module: types.ModuleType) -> None:
    result = submit_module._resolve_canary_assignment(
        scene_id="scene-allow",
        scene_tags=[],
        canary_enabled=True,
        canary_tags=[],
        canary_scene_ids=["scene-allow"],
        canary_percent=0,
    )

    assert result["is_canary"] is True
    assert "scene_id_allowlist" in result["match_reasons"]


def test_canary_percentage_rollout_uses_hash_bucket(submit_module: types.ModuleType) -> None:
    scene_id = "scene-percent"
    bucket = submit_module._scene_hash_percentage(scene_id)

    no_hit = submit_module._resolve_canary_assignment(
        scene_id=scene_id,
        scene_tags=[],
        canary_enabled=True,
        canary_tags=[],
        canary_scene_ids=[],
        canary_percent=bucket,
    )
    assert no_hit["is_canary"] is False
    assert "percentage_rollout" not in no_hit["match_reasons"]

    hit = submit_module._resolve_canary_assignment(
        scene_id=scene_id,
        scene_tags=[],
        canary_enabled=True,
        canary_tags=[],
        canary_scene_ids=[],
        canary_percent=bucket + 1,
    )
    assert hit["is_canary"] is True
    assert "percentage_rollout" in hit["match_reasons"]


def test_canary_disabled_overrides_matches(submit_module: types.ModuleType) -> None:
    result = submit_module._resolve_canary_assignment(
        scene_id="scene-disabled",
        scene_tags=["fast"],
        canary_enabled=False,
        canary_tags=["fast"],
        canary_scene_ids=["scene-disabled"],
        canary_percent=100,
    )

    assert result["is_canary"] is False
    assert set(result["match_reasons"]) >= {"tag_match", "scene_id_allowlist", "percentage_rollout"}


def test_scene_hash_percentage_is_deterministic(submit_module: types.ModuleType) -> None:
    scene_id = "scene-deterministic"
    expected = int(hashlib.sha256(scene_id.encode("utf-8")).hexdigest(), 16) % 100

    first = submit_module._scene_hash_percentage(scene_id)
    second = submit_module._scene_hash_percentage(scene_id)

    assert first == expected
    assert second == expected
