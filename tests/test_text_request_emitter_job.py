from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_text_request_emitter_emits_scene_request_and_updates_state(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "text_request_emitter_job_test",
        repo_root / "text-request-emitter-job" / "emit_text_requests.py",
    )

    gcs_root = tmp_path / "gcs"
    monkeypatch.setattr(module, "GCS_ROOT", gcs_root)

    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("TEXT_AUTONOMY_STORAGE_MODE", "filesystem")
    monkeypatch.setenv("TEXT_AUTONOMY_STATE_PREFIX", "automation/text_daily")
    monkeypatch.setenv("TEXT_AUTONOMY_TIMEZONE", "America/New_York")
    monkeypatch.setenv("TEXT_AUTONOMY_RUN_DATE", "2026-02-11")
    monkeypatch.setenv("TEXT_DAILY_QUOTA", "1")
    monkeypatch.setenv("TEXT_AUTONOMY_PROVIDER_POLICY", "openai_primary")
    monkeypatch.setenv("TEXT_AUTONOMY_QUALITY_TIER", "premium")
    monkeypatch.setenv("TEXT_AUTONOMY_ALLOW_IMAGE_FALLBACK", "false")
    monkeypatch.setenv("TEXT_AUTONOMY_SEED_COUNT", "1")
    monkeypatch.setenv("TEXT_PROMPT_USE_LLM", "false")

    assert module.main() == 0

    emitted_index = json.loads(
        (gcs_root / "automation/text_daily/runs/2026-02-11/emitted_requests.json").read_text(encoding="utf-8")
    )
    assert len(emitted_index["scene_ids"]) == 1

    scene_id = emitted_index["scene_ids"][0]
    request_path = gcs_root / f"scenes/{scene_id}/prompts/scene_request.json"
    assert request_path.is_file()

    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["schema_version"] == "v1"
    assert request["scene_id"] == scene_id
    assert request["source_mode"] == "text"
    assert request["quality_tier"] == "premium"
    assert request["provider_policy"] == "openai_primary"
    assert request["seed_count"] == 1
    assert request["fallback"]["allow_image_fallback"] is False
    assert "prompt_diversity" in request["constraints"]

    state_payload = json.loads((gcs_root / "automation/text_daily/state.json").read_text(encoding="utf-8"))
    assert state_payload["last_emit_run_date"] == "2026-02-11"
    assert len(state_payload["recent_prompts"]) == 1


def test_text_request_emitter_daily_lock_prevents_duplicate_emits(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "text_request_emitter_job_lock_test",
        repo_root / "text-request-emitter-job" / "emit_text_requests.py",
    )

    gcs_root = tmp_path / "gcs"
    monkeypatch.setattr(module, "GCS_ROOT", gcs_root)

    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("TEXT_AUTONOMY_STORAGE_MODE", "filesystem")
    monkeypatch.setenv("TEXT_AUTONOMY_STATE_PREFIX", "automation/text_daily")
    monkeypatch.setenv("TEXT_AUTONOMY_RUN_DATE", "2026-02-12")
    monkeypatch.setenv("TEXT_DAILY_QUOTA", "1")
    monkeypatch.setenv("TEXT_AUTONOMY_PROVIDER_POLICY", "openai_primary")
    monkeypatch.setenv("TEXT_AUTONOMY_QUALITY_TIER", "premium")
    monkeypatch.setenv("TEXT_PROMPT_USE_LLM", "false")

    assert module.main() == 0
    first_index = json.loads(
        (gcs_root / "automation/text_daily/runs/2026-02-12/emitted_requests.json").read_text(encoding="utf-8")
    )
    first_scene_ids = list(first_index["scene_ids"])

    # Second run with same day should no-op because lock already exists.
    assert module.main() == 0
    second_index = json.loads(
        (gcs_root / "automation/text_daily/runs/2026-02-12/emitted_requests.json").read_text(encoding="utf-8")
    )

    assert second_index["scene_ids"] == first_scene_ids
    lock_file = gcs_root / "automation/text_daily/locks/2026-02-12.lock"
    assert lock_file.is_file()
