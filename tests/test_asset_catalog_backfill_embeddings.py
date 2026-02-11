from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "asset_catalog"
    / "backfill_embeddings.py"
)
SPEC = importlib.util.spec_from_file_location("backfill_embeddings", MODULE_PATH)
backfill_embeddings = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(backfill_embeddings)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_backfill_enqueues_queue_objects_and_updates_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backfill_embeddings, "GCS_ROOT", tmp_path)
    monkeypatch.setenv("BUCKET", "unit-bucket")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_QUEUE_PREFIX", "automation/asset_embedding/queue")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_BACKFILL_STATE_PREFIX", "automation/asset_embedding/backfill")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_BACKFILL_PAGE_SIZE", "2")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_BACKFILL_MAX_ENQUEUE", "10")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_BACKFILL_QUEUE_CHUNK", "2")

    calls = []

    def _fake_iter_firestore_assets(*, page_size: int, cursor_asset_id: str):
        calls.append(cursor_asset_id)
        if not cursor_asset_id:
            return (
                [
                    {"asset_id": "asset_001", "class_name": "mug", "description": "mug"},
                    {"asset_id": "asset_002", "class_name": "plate", "description": "plate"},
                ],
                "asset_002",
            )
        return ([], cursor_asset_id)

    monkeypatch.setattr(backfill_embeddings, "_iter_firestore_assets", _fake_iter_firestore_assets)
    assert backfill_embeddings.main() == 0

    state_path = tmp_path / "automation/asset_embedding/backfill/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["cursor_asset_id"] == "asset_002"
    assert state["last_enqueued"] == 2
    assert calls[0] == ""

    queue_files = sorted((tmp_path / "automation/asset_embedding/queue").glob("*.json"))
    assert len(queue_files) == 1
    payload = json.loads(queue_files[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "v1"
    assert len(payload["assets"]) == 2


def test_backfill_resumes_from_cursor(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backfill_embeddings, "GCS_ROOT", tmp_path)
    monkeypatch.setenv("BUCKET", "unit-bucket")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_QUEUE_PREFIX", "automation/asset_embedding/queue")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_BACKFILL_STATE_PREFIX", "automation/asset_embedding/backfill")

    _write_json(
        tmp_path / "automation/asset_embedding/backfill/state.json",
        {
            "schema_version": "v1",
            "cursor_asset_id": "asset_050",
            "enqueued_total": 0,
            "skipped_total": 0,
        },
    )

    seen_cursors = []

    def _fake_iter_firestore_assets(*, page_size: int, cursor_asset_id: str):
        seen_cursors.append(cursor_asset_id)
        return ([], cursor_asset_id)

    monkeypatch.setattr(backfill_embeddings, "_iter_firestore_assets", _fake_iter_firestore_assets)
    assert backfill_embeddings.main() == 0
    assert seen_cursors[0] == "asset_050"

