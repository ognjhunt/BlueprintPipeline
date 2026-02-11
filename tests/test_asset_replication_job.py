from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "asset-replication-job"
    / "process_replication_queue.py"
)
SPEC = importlib.util.spec_from_file_location("process_replication_queue", MODULE_PATH)
process_replication_queue = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(process_replication_queue)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_asset_replication_job_processes_queue_item_dry_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(process_replication_queue, "GCS_ROOT", tmp_path)
    monkeypatch.setenv("BUCKET", "unit-bucket")
    monkeypatch.setenv("TEXT_ASSET_REPLICATION_DRY_RUN", "true")
    monkeypatch.setenv("TEXT_ASSET_REPLICATION_FAIL_ON_ERROR", "true")

    asset_file = tmp_path / "scenes/scene_1/assets/obj_001/model.usd"
    asset_file.parent.mkdir(parents=True, exist_ok=True)
    asset_file.write_text("#usda 1.0\n", encoding="utf-8")

    queue_object = "automation/asset_replication/queue/scene_1.json"
    queue_payload = {
        "scene_id": "scene_1",
        "assets": [
            {
                "asset_id": "text::scene_1::obj_001",
                "files": [
                    {
                        "path": "scenes/scene_1/assets/obj_001/model.usd",
                        "target_key": "assets/scenes/scene_1/assets/obj_001/model.usd",
                    }
                ],
            }
        ],
    }
    _write_json(tmp_path / queue_object, queue_payload)

    monkeypatch.setenv("QUEUE_OBJECT", queue_object)
    rc = process_replication_queue.main()
    assert rc == 0

    assert not (tmp_path / queue_object).exists()
    processed = tmp_path / "automation/asset_replication/processed/scene_1.json"
    assert processed.is_file()
    summary = json.loads(processed.read_text(encoding="utf-8"))
    assert summary["status"] == "succeeded"
    assert summary["result"]["uploaded"] == 1


def test_asset_replication_job_moves_failed_item(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(process_replication_queue, "GCS_ROOT", tmp_path)
    monkeypatch.setenv("BUCKET", "unit-bucket")
    monkeypatch.setenv("TEXT_ASSET_REPLICATION_DRY_RUN", "true")
    monkeypatch.setenv("TEXT_ASSET_REPLICATION_FAIL_ON_ERROR", "true")

    queue_object = "automation/asset_replication/queue/scene_missing.json"
    queue_payload = {
        "scene_id": "scene_missing",
        "assets": [
            {
                "asset_id": "text::scene_missing::obj_001",
                "files": [
                    {
                        "path": "scenes/scene_missing/assets/obj_001/model.usd",
                        "target_key": "assets/scenes/scene_missing/assets/obj_001/model.usd",
                    }
                ],
            }
        ],
    }
    _write_json(tmp_path / queue_object, queue_payload)

    monkeypatch.setenv("QUEUE_OBJECT", queue_object)
    rc = process_replication_queue.main()
    assert rc == 1

    assert not (tmp_path / queue_object).exists()
    failed = tmp_path / "automation/asset_replication/failed/scene_missing.json"
    assert failed.is_file()
    summary = json.loads(failed.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["result"]["errors"]

