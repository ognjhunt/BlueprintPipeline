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
def minimal_payloads() -> dict[str, object]:
    return {
        "scenes/scene-1/geniesim/scene_graph.json": {
            "scene_id": "scene-1",
            "coordinate_system": "rhs",
            "meters_per_unit": 1.0,
            "nodes": [],
            "edges": [],
        },
        "scenes/scene-1/geniesim/asset_index.json": {"assets": []},
        "scenes/scene-1/geniesim/task_config.json": {
            "scene_id": "scene-1",
            "environment_type": "kitchen",
            "suggested_tasks": [],
            "robot_config": {"type": "franka", "base_position": [0.0, 0.0, 0.0], "workspace_bounds": [[-1, -1, 0], [1, 1, 1]]},
        },
        "scenes/scene-1/geniesim/export_manifest.json": {
            "schema_version": "3.0",
            "schema_definition": {"version": "3.0", "description": "test", "fields": {}},
            "export_info": {"timestamp": "2025-01-01T00:00:00Z", "exporter_version": "1.0", "source_pipeline": "test"},
            "asset_provenance_path": None,
            "config": {
                "robot_type": "franka", "generate_embeddings": False,
                "embedding_model": None, "require_embeddings": False,
                "filter_commercial_only": False, "max_tasks": 5,
                "lerobot_export_format": "lerobot_v3",
            },
            "result": {
                "success": True, "scene_id": "scene-1", "output_dir": "geniesim",
                "outputs": {"scene_graph": "scene_graph.json", "asset_index": "asset_index.json", "task_config": "task_config.json", "scene_config": "scene_config.yaml"},
                "statistics": {"nodes": 0, "edges": 0, "assets": 0, "tasks": 0},
                "errors": [], "warnings": [],
            },
            "geniesim_compatibility": {"version": "3.0.0", "isaac_sim_version": "4.0", "formats": {"scene_graph": "json", "asset_index": "json", "task_config": "json", "scene_config": "yaml"}},
            "file_inventory": [],
            "checksums": {"files": {}},
        },
        "scenes/scene-1/geniesim/_GENIESIM_EXPORT_COMPLETE": {
            "export_schema_version": "1.0.0",
            "geniesim_schema_version": "3.0.0",
            "blueprintpipeline_version": "test",
            "export_timestamp": "2024-01-01T00:00:00Z",
            "schema_compatibility": {"min_geniesim_version": "3.0.0", "max_geniesim_version": "3.99.99"},
        },
    }


def _install_firebase_retry_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    retry_calls: list[list[Path]],
) -> None:
    fake_firebase = types.ModuleType("tools.firebase_upload")

    class FirebaseUploadError(RuntimeError):
        def __init__(self, summary: dict, message: str) -> None:
            super().__init__(message)
            self.summary = summary

    def upload_episodes_to_firebase(episodes_dir: Path, scene_id: str, prefix: str) -> dict:
        failure_path = episodes_dir / "episode_1" / "data.txt"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text("data")
        summary = {
            "total_files": 1,
            "uploaded": 0,
            "skipped": 0,
            "reuploaded": 0,
            "failed": 1,
            "file_statuses": [],
            "failures": [
                {
                    "local_path": str(failure_path),
                    "remote_path": f"{prefix}/{scene_id}/episode_1/data.txt",
                    "status": "failed",
                    "error": "upload failed",
                }
            ],
            "verification_failed": [],
            "verification_strategy": "sha256_metadata+md5_base64",
        }
        raise FirebaseUploadError(summary, "Firebase upload failed for retry test")

    def upload_firebase_files(paths: list[Path], prefix: str, scene_id: str) -> dict:
        retry_calls.append(paths)
        return {
            "total_files": len(paths),
            "uploaded": len(paths),
            "skipped": 0,
            "reuploaded": 0,
            "failed": 0,
            "file_statuses": [],
            "failures": [],
            "verification_failed": [],
            "verification_strategy": "sha256_metadata+md5_base64",
        }

    fake_firebase.FirebaseUploadError = FirebaseUploadError
    fake_firebase.upload_episodes_to_firebase = upload_episodes_to_firebase
    fake_firebase.upload_firebase_files = upload_firebase_files
    monkeypatch.setitem(sys.modules, "tools.firebase_upload", fake_firebase)


def test_firebase_retry_second_pass(
    monkeypatch: pytest.MonkeyPatch,
    submit_module: types.ModuleType,
    minimal_payloads: dict[str, object],
) -> None:
    import tempfile as _tmpmod
    monkeypatch.setenv("QUALITY_APPROVAL_PATH", _tmpmod.mkdtemp(prefix="bp_approval_"))
    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", "scene-1")
    monkeypatch.setenv("EPISODES_PER_TASK", "1")
    monkeypatch.setenv("NUM_VARIATIONS", "1")
    monkeypatch.setenv("FIREBASE_UPLOAD_SECOND_PASS_MAX", "1")
    monkeypatch.setenv("ALLOW_MISSING_ASSET_PROVENANCE", "1")

    fake_client = FakeStorageClient({"test-bucket": minimal_payloads})
    monkeypatch.setattr(submit_module.storage, "Client", lambda *args, **kwargs: fake_client)
    monkeypatch.setattr(submit_module, "run_geniesim_preflight_or_exit", lambda *_a, **_k: {})
    monkeypatch.setattr(submit_module, "_run_geniesim_ik_gate", lambda **_k: True)
    monkeypatch.setattr(submit_module, "send_alert", lambda **_k: None)
    monkeypatch.setattr(submit_module, "_run_firebase_preflight", lambda **_k: None)
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
    monkeypatch.setattr(submit_module, "verify_blob_upload", lambda *_a, **_k: (True, None))

    uploaded_paths: list[str] = []

    def _record_upload(_blob, file_path, _gcs_uri, **_kwargs):
        uploaded_paths.append(str(file_path))
        return FakeUploadResult()

    monkeypatch.setattr(submit_module, "upload_blob_from_filename", _record_upload)

    class FakeMetrics:
        def __init__(self) -> None:
            self.backend = types.SimpleNamespace(value="test")

        def get_stats(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(submit_module, "get_metrics", lambda: FakeMetrics())

    retry_calls: list[list[Path]] = []
    _install_firebase_retry_stub(monkeypatch, retry_calls=retry_calls)

    submit_module.main()

    job_path = "scenes/scene-1/geniesim/job.json"
    job_payload = json.loads(fake_client.bucket("test-bucket")._data[job_path])

    assert retry_calls
    assert job_payload["firebase_upload_retry"]["retry_attempted"] is True
    assert job_payload["firebase_upload_retry"]["retry_failed_count"] == 0
    assert job_payload["firebase_upload_retry"]["retry_manifest_path"].endswith(
        "upload_retry_manifest.json"
    )
    assert any(path.endswith("upload_retry_manifest.json") for path in uploaded_paths)
