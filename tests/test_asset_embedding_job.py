from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "asset-embedding-job"
    / "index_asset_embeddings.py"
)
SPEC = importlib.util.spec_from_file_location("index_asset_embeddings", MODULE_PATH)
index_asset_embeddings = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(index_asset_embeddings)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_asset_embedding_job_processes_queue_item_dry_run_with_idempotency(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(index_asset_embeddings, "GCS_ROOT", tmp_path)
    monkeypatch.setenv("BUCKET", "unit-bucket")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_DRY_RUN", "true")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_FAIL_ON_ERROR", "true")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_BACKEND", "deterministic")
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "sqlite")
    monkeypatch.setenv("VECTOR_STORE_CONNECTION_URI", str(tmp_path / "vectors.db"))
    monkeypatch.setenv("VECTOR_STORE_COLLECTION", "asset_vectors")
    monkeypatch.setenv("VECTOR_STORE_DIMENSION", "16")

    queue_object = "automation/asset_embedding/queue/scene_1.json"
    queue_payload = {
        "scene_id": "scene_1",
        "embedding_model": "text-embedding-3-small",
        "assets": [
            {
                "asset_id": "text::scene_1::obj_001",
                "descriptor_text": "mug manipulable object",
                "descriptor_hash": "abc",
                "idempotency_key": "dup-key",
                "class_name": "mug",
                "sim_roles": ["manipulable_object"],
                "usd_path": "scenes/scene_1/assets/obj_001/model.usd",
            },
            {
                "asset_id": "text::scene_1::obj_001",
                "descriptor_text": "mug manipulable object",
                "descriptor_hash": "abc",
                "idempotency_key": "dup-key",
                "class_name": "mug",
                "sim_roles": ["manipulable_object"],
                "usd_path": "scenes/scene_1/assets/obj_001/model.usd",
            },
        ],
    }
    _write_json(tmp_path / queue_object, queue_payload)
    monkeypatch.setenv("QUEUE_OBJECT", queue_object)

    rc = index_asset_embeddings.main()
    assert rc == 0

    assert not (tmp_path / queue_object).exists()
    processed = tmp_path / "automation/asset_embedding/processed/scene_1.json"
    assert processed.is_file()
    summary = json.loads(processed.read_text(encoding="utf-8"))
    assert summary["status"] == "succeeded"
    assert summary["result"]["upserted"] == 1
    assert summary["result"]["skipped"] == 1


def test_asset_embedding_job_moves_failed_item(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(index_asset_embeddings, "GCS_ROOT", tmp_path)
    monkeypatch.setenv("BUCKET", "unit-bucket")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_DRY_RUN", "false")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_FAIL_ON_ERROR", "true")
    monkeypatch.setenv("TEXT_ASSET_EMBEDDING_BACKEND", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "sqlite")
    monkeypatch.setenv("VECTOR_STORE_CONNECTION_URI", str(tmp_path / "vectors.db"))
    monkeypatch.setenv("VECTOR_STORE_COLLECTION", "asset_vectors")
    monkeypatch.setenv("VECTOR_STORE_DIMENSION", "16")

    queue_object = "automation/asset_embedding/queue/scene_fail.json"
    queue_payload = {
        "scene_id": "scene_fail",
        "assets": [
            {
                "asset_id": "text::scene_fail::obj_001",
                "descriptor_text": "mug manipulable object",
                "class_name": "mug",
                "sim_roles": ["manipulable_object"],
                "usd_path": "scenes/scene_fail/assets/obj_001/model.usd",
            }
        ],
    }
    _write_json(tmp_path / queue_object, queue_payload)
    monkeypatch.setenv("QUEUE_OBJECT", queue_object)

    rc = index_asset_embeddings.main()
    assert rc == 1

    assert not (tmp_path / queue_object).exists()
    failed = tmp_path / "automation/asset_embedding/failed/scene_fail.json"
    assert failed.is_file()
    summary = json.loads(failed.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["result"]["errors"]

