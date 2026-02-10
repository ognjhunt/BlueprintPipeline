from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _reset_google_cloud_modules() -> None:
    for module_name in list(sys.modules):
        if (
            module_name == "google"
            or module_name.startswith("google.")
            or module_name == "firebase_admin"
            or module_name.startswith("firebase_admin.")
        ):
            sys.modules.pop(module_name, None)


def _build_payload(tmp_path: Path, robot_type: str) -> dict:
    output_dir = tmp_path / robot_type
    output_dir.mkdir(parents=True, exist_ok=True)
    result = SimpleNamespace(
        success=True,
        output_dir=output_dir,
        import_manifest_path=tmp_path / f"{robot_type}_manifest.json",
        episodes_passed_validation=1,
        episodes_filtered=0,
        episode_content_hashes={"episode-1": "hash"},
    )
    entry = {
        "robot_type": robot_type,
        "errors": [],
        "firebase_upload": None,
        "gcs_output_path": f"gs://bucket/{robot_type}",
    }
    return {"robot_type": robot_type, "result": result, "entry": entry}


def test_firebase_upload_allows_partial_failures(monkeypatch, load_job_module, tmp_path):
    _reset_google_cloud_modules()
    module = load_job_module("geniesim_import", "import_from_geniesim.py")
    monkeypatch.setattr(module, "_persist_episode_hash_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module, "_publish_dataset_catalog_document", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(module, "_alert_firebase_upload_failure", lambda *args, **kwargs: None)

    def fake_upload(*, payload, **_kwargs):
        if payload["robot_type"] == "robot-fail":
            raise RuntimeError("boom")
        return (
            payload["robot_type"],
            {
                "uploaded": 1,
                "skipped": 0,
                "reuploaded": 0,
                "failed": 0,
                "total_files": 1,
                "deduplicated_episode_ids": [],
            },
            None,
        )

    monkeypatch.setattr(module, "_upload_robot_payload_to_firebase", fake_upload)

    payload_success = _build_payload(tmp_path, "robot-ok")
    payload_failure = _build_payload(tmp_path, "robot-fail")
    log = logging.LoggerAdapter(logging.getLogger("test"), {})

    overall_success = module._run_firebase_uploads_for_robot_payloads(
        robot_results=[payload_success, payload_failure],
        scene_id="scene-1",
        job_id="job-1",
        firebase_upload_prefix="datasets",
        firebase_upload_max_workers=2,
        allow_partial_firebase_uploads=True,
        fail_on_partial_error=False,
        log=log,
        overall_success=True,
    )

    assert overall_success is False
    assert payload_success["entry"]["firebase_upload"] is not None
    assert payload_failure["entry"]["errors"] == ["Firebase upload failed: boom"]


def test_firebase_upload_blocks_partial_failures(monkeypatch, load_job_module, tmp_path):
    _reset_google_cloud_modules()
    module = load_job_module("geniesim_import", "import_from_geniesim.py")
    monkeypatch.setattr(module, "_persist_episode_hash_index", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module, "_publish_dataset_catalog_document", lambda *args, **kwargs: None
    )

    def fake_upload(*, payload, **_kwargs):
        if payload["robot_type"] == "robot-fail":
            return payload["robot_type"], None, RuntimeError("boom")
        return (
            payload["robot_type"],
            {
                "uploaded": 1,
                "skipped": 0,
                "reuploaded": 0,
                "failed": 0,
                "total_files": 1,
                "deduplicated_episode_ids": [],
            },
            None,
        )

    monkeypatch.setattr(module, "_upload_robot_payload_to_firebase", fake_upload)

    payload_success = _build_payload(tmp_path, "robot-ok")
    payload_failure = _build_payload(tmp_path, "robot-fail")
    log = logging.LoggerAdapter(logging.getLogger("test"), {})

    with pytest.raises(module.FirebaseUploadOrchestratorError) as excinfo:
        module._run_firebase_uploads_for_robot_payloads(
            robot_results=[payload_success, payload_failure],
            scene_id="scene-1",
            job_id="job-1",
            firebase_upload_prefix="datasets",
            firebase_upload_max_workers=2,
            allow_partial_firebase_uploads=False,
            fail_on_partial_error=True,
            log=log,
            overall_success=True,
        )

    assert "robot-fail" in str(excinfo.value)
    assert payload_success["entry"]["firebase_upload"] is not None
    assert payload_failure["entry"]["errors"] == ["Firebase upload failed: boom"]
