import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class FakeUploadResult:
    success: bool = True
    error: str | None = None


class FakeBlob:
    def __init__(self, bucket: "FakeBucket", name: str) -> None:
        self._bucket = bucket
        self.name = name

    def exists(self) -> bool:
        return self.name in self._bucket._data

    def download_as_text(self) -> str:
        payload = self._bucket._data[self.name]
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    def upload_from_string(self, payload: str, content_type: str | None = None) -> None:
        self._bucket._data[self.name] = payload


class FakeBucket:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)


class FakeStorageClient:
    def __init__(self, buckets: dict[str, dict[str, object]]) -> None:
        self._buckets = buckets

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self._buckets.setdefault(name, {}))


@pytest.fixture(scope="module")
def submit_module() -> types.ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "genie-sim-submit-job" / "submit_to_geniesim.py"
    spec = importlib.util.spec_from_file_location("submit_to_geniesim", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def minimal_scene_graph() -> dict[str, object]:
    return {
        "scene_id": "scene-1",
        "coordinate_system": "rhs",
        "meters_per_unit": 1.0,
        "nodes": [
            {
                "asset_id": "asset-1",
                "semantic": "chair",
                "size": [1.0, 1.0, 1.0],
                "pose": {
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "task_tag": ["grasp"],
                "usd_path": "scenes/scene-1/usd/scene.usd",
            }
        ],
        "edges": [],
    }


@pytest.fixture()
def minimal_asset_index() -> dict[str, object]:
    return {
        "assets": [
            {
                "asset_id": "asset-1",
                "usd_path": "assets/chair.usd",
                "semantic_description": "chair",
                "categories": ["furniture"],
            }
        ]
    }


@pytest.fixture()
def minimal_task_config() -> dict[str, object]:
    return {
        "scene_id": "scene-1",
        "environment_type": "kitchen",
        "suggested_tasks": [
            {
                "task_type": "pick",
                "target_object": "asset-1",
                "difficulty": "easy",
                "priority": 1,
            }
        ],
        "robot_config": {
            "type": "franka",
            "base_position": [0.0, 0.0, 0.0],
            "workspace_bounds": [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]],
        },
    }


@pytest.fixture()
def export_marker() -> dict[str, object]:
    return {
        "export_schema_version": "1.0.0",
        "geniesim_schema_version": "3.0.0",
        "blueprintpipeline_version": "test",
        "export_timestamp": "2024-01-01T00:00:00Z",
        "schema_compatibility": {
            "min_geniesim_version": "3.0.0",
            "max_geniesim_version": "3.x",
        },
    }


def _prepare_common_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", "scene-1")
    monkeypatch.setenv("EPISODES_PER_TASK", "1")
    monkeypatch.setenv("NUM_VARIATIONS", "1")


def _install_firebase_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_firebase = types.ModuleType("tools.firebase_upload")

    class FirebaseUploadError(RuntimeError):
        pass

    def upload_episodes_to_firebase(*_args: object, **_kwargs: object) -> dict[str, int]:
        return {"uploaded": 0, "skipped": 0, "reuploaded": 0, "failed": 0, "total_files": 0}

    def upload_firebase_files(*_args: object, **_kwargs: object) -> dict[str, int]:
        return {"uploaded": 0, "skipped": 0, "reuploaded": 0, "failed": 0, "total_files": 0}

    fake_firebase.FirebaseUploadError = FirebaseUploadError
    fake_firebase.upload_episodes_to_firebase = upload_episodes_to_firebase
    fake_firebase.upload_firebase_files = upload_firebase_files
    monkeypatch.setitem(sys.modules, "tools.firebase_upload", fake_firebase)


def _run_submit_job(
    *,
    monkeypatch: pytest.MonkeyPatch,
    submit_module: types.ModuleType,
    blob_payloads: dict[str, object],
    env_overrides: dict[str, str],
) -> dict[str, object]:
    _prepare_common_env(monkeypatch)
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)

    fake_client = FakeStorageClient({"test-bucket": blob_payloads})
    monkeypatch.setattr(submit_module.storage, "Client", lambda: fake_client)
    monkeypatch.setattr(submit_module, "run_geniesim_preflight_or_exit", lambda *_a, **_k: {})
    monkeypatch.setattr(submit_module, "_run_geniesim_ik_gate", lambda **_k: True)
    monkeypatch.setattr(submit_module, "send_alert", lambda **_k: None)
    monkeypatch.setattr(submit_module, "_preflight_firebase_upload", lambda: None)
    monkeypatch.setattr(
        submit_module,
        "_run_local_data_collection_with_handshake",
        lambda **_k: submit_module.DataCollectionResult(
            success=True,
            task_name="test-task",
            episodes_collected=1,
            episodes_passed=1,
        ),
    )
    monkeypatch.setattr(submit_module, "upload_blob_from_filename", lambda *_a, **_k: FakeUploadResult())
    monkeypatch.setattr(submit_module, "verify_blob_upload", lambda *_a, **_k: (True, None))

    class FakeMetrics:
        def __init__(self) -> None:
            self.backend = types.SimpleNamespace(value="test")

        def get_stats(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(submit_module, "get_metrics", lambda: FakeMetrics())
    _install_firebase_stub(monkeypatch)

    submit_module.main()
    job_path = "scenes/scene-1/geniesim/job.json"
    job_payload = json.loads(fake_client.bucket("test-bucket")._data[job_path])
    return job_payload


def _build_blob_payloads(
    *,
    minimal_scene_graph: dict[str, object],
    minimal_asset_index: dict[str, object],
    minimal_task_config: dict[str, object],
    export_marker: dict[str, object],
    asset_provenance: dict[str, object] | None,
) -> dict[str, object]:
    payloads: dict[str, object] = {
        "scenes/scene-1/geniesim/scene_graph.json": minimal_scene_graph,
        "scenes/scene-1/geniesim/asset_index.json": minimal_asset_index,
        "scenes/scene-1/geniesim/task_config.json": minimal_task_config,
        "scenes/scene-1/geniesim/export_manifest.json": {"exported": "ok"},
        "scenes/scene-1/geniesim/_GENIESIM_EXPORT_COMPLETE": export_marker,
    }
    if asset_provenance is not None:
        payloads["scenes/scene-1/geniesim/legal/asset_provenance.json"] = asset_provenance
    return payloads


def test_provenance_blockers_fail_without_override(
    monkeypatch: pytest.MonkeyPatch,
    submit_module: types.ModuleType,
    minimal_scene_graph: dict[str, object],
    minimal_asset_index: dict[str, object],
    minimal_task_config: dict[str, object],
    export_marker: dict[str, object],
) -> None:
    asset_provenance = {"license": {"commercial_ok": False, "blockers": ["non-commercial"]}}
    blobs = _build_blob_payloads(
        minimal_scene_graph=minimal_scene_graph,
        minimal_asset_index=minimal_asset_index,
        minimal_task_config=minimal_task_config,
        export_marker=export_marker,
        asset_provenance=asset_provenance,
    )
    job_payload = _run_submit_job(
        monkeypatch=monkeypatch,
        submit_module=submit_module,
        blob_payloads=blobs,
        env_overrides={"ALLOW_NONCOMMERCIAL_DATA": "false"},
    )

    assert job_payload["status"] == "failed"
    assert job_payload["failure_reason"] == "Asset provenance blocked submission"
    assert job_payload["provenance_gate"]["status"] == "blocked"


def test_provenance_blockers_override_with_allow_noncommercial(
    monkeypatch: pytest.MonkeyPatch,
    submit_module: types.ModuleType,
    minimal_scene_graph: dict[str, object],
    minimal_asset_index: dict[str, object],
    minimal_task_config: dict[str, object],
    export_marker: dict[str, object],
) -> None:
    asset_provenance = {"license": {"commercial_ok": False, "blockers": ["non-commercial"]}}
    blobs = _build_blob_payloads(
        minimal_scene_graph=minimal_scene_graph,
        minimal_asset_index=minimal_asset_index,
        minimal_task_config=minimal_task_config,
        export_marker=export_marker,
        asset_provenance=asset_provenance,
    )
    job_payload = _run_submit_job(
        monkeypatch=monkeypatch,
        submit_module=submit_module,
        blob_payloads=blobs,
        env_overrides={"ALLOW_NONCOMMERCIAL_DATA": "true"},
    )

    assert job_payload["status"] == "completed"
    assert job_payload.get("failure_reason") is None
    assert job_payload["provenance_gate"]["status"] == "override"


def test_provenance_missing_fails_without_allow_missing(
    monkeypatch: pytest.MonkeyPatch,
    submit_module: types.ModuleType,
    minimal_scene_graph: dict[str, object],
    minimal_asset_index: dict[str, object],
    minimal_task_config: dict[str, object],
    export_marker: dict[str, object],
) -> None:
    blobs = _build_blob_payloads(
        minimal_scene_graph=minimal_scene_graph,
        minimal_asset_index=minimal_asset_index,
        minimal_task_config=minimal_task_config,
        export_marker=export_marker,
        asset_provenance=None,
    )
    job_payload = _run_submit_job(
        monkeypatch=monkeypatch,
        submit_module=submit_module,
        blob_payloads=blobs,
        env_overrides={"ALLOW_MISSING_ASSET_PROVENANCE": "false"},
    )

    assert job_payload["status"] == "failed"
    assert job_payload["failure_reason"] == "Asset provenance missing"
    assert job_payload["provenance_gate"]["status"] == "missing"


def test_provenance_missing_allowed(
    monkeypatch: pytest.MonkeyPatch,
    submit_module: types.ModuleType,
    minimal_scene_graph: dict[str, object],
    minimal_asset_index: dict[str, object],
    minimal_task_config: dict[str, object],
    export_marker: dict[str, object],
) -> None:
    blobs = _build_blob_payloads(
        minimal_scene_graph=minimal_scene_graph,
        minimal_asset_index=minimal_asset_index,
        minimal_task_config=minimal_task_config,
        export_marker=export_marker,
        asset_provenance=None,
    )
    job_payload = _run_submit_job(
        monkeypatch=monkeypatch,
        submit_module=submit_module,
        blob_payloads=blobs,
        env_overrides={"ALLOW_MISSING_ASSET_PROVENANCE": "true"},
    )

    assert job_payload["status"] == "completed"
    assert job_payload.get("failure_reason") is None
    assert job_payload["provenance_gate"]["status"] == "missing"


def test_provenance_missing_fails_in_production_even_with_override(
    monkeypatch: pytest.MonkeyPatch,
    submit_module: types.ModuleType,
    minimal_scene_graph: dict[str, object],
    minimal_asset_index: dict[str, object],
    minimal_task_config: dict[str, object],
    export_marker: dict[str, object],
) -> None:
    blobs = _build_blob_payloads(
        minimal_scene_graph=minimal_scene_graph,
        minimal_asset_index=minimal_asset_index,
        minimal_task_config=minimal_task_config,
        export_marker=export_marker,
        asset_provenance=None,
    )
    job_payload = _run_submit_job(
        monkeypatch=monkeypatch,
        submit_module=submit_module,
        blob_payloads=blobs,
        env_overrides={
            "ALLOW_MISSING_ASSET_PROVENANCE": "true",
            "PRODUCTION_MODE": "true",
        },
    )

    assert job_payload["status"] == "failed"
    assert job_payload["failure_reason"] == "Asset provenance missing"
    assert job_payload["provenance_gate"]["status"] == "missing"
