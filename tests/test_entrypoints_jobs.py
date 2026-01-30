from __future__ import annotations

import hmac
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "entrypoints"


class FakeBlob:
    def __init__(self, name: str, data_map: dict[str, str], uploads: dict[str, str]) -> None:
        self.name = name
        self._data_map = data_map
        self._uploads = uploads
        self.md5_hash: str | None = None
        self.crc32c: str | None = None
        self.content_type: str | None = None

    def exists(self) -> bool:
        return self.name in self._data_map

    def download_as_text(self) -> str:
        if self.name not in self._data_map:
            raise FileNotFoundError(self.name)
        return self._data_map[self.name]

    def download_to_filename(self, filename: str) -> None:
        if self.name not in self._data_map:
            raise FileNotFoundError(self.name)
        Path(filename).write_text(self._data_map[self.name])

    def reload(self) -> None:
        if self.name in self._data_map:
            import base64
            data = self._data_map[self.name]
            raw = data.encode("utf-8") if isinstance(data, str) else data
            self.md5_hash = base64.b64encode(hashlib.md5(raw).digest()).decode("utf-8")

    @property
    def size(self) -> int:
        if self.name in self._data_map:
            return len(self._data_map[self.name].encode("utf-8") if isinstance(self._data_map[self.name], str) else self._data_map[self.name])
        return 0

    def upload_from_string(self, data: str, content_type: str | None = None) -> None:
        self._uploads[self.name] = data
        self._data_map[self.name] = data

    def upload_from_filename(self, filename: str, content_type: str | None = None) -> None:
        content = Path(filename).read_text()
        self._uploads[self.name] = content
        self._data_map[self.name] = content


class FakeBucket:
    def __init__(self, data_map: dict[str, str], uploads: dict[str, str]) -> None:
        self._data_map = data_map
        self._uploads = uploads

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(name, self._data_map, self._uploads)


class FakeStorageClient:
    def __init__(self, data_map: dict[str, str], uploads: dict[str, str]) -> None:
        self._data_map = data_map
        self._uploads = uploads

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self._data_map, self._uploads)


class FakeFirestoreDocumentSnapshot:
    def __init__(self, data: dict | None) -> None:
        self._data = data or {}
        self.exists = data is not None

    def to_dict(self) -> dict:
        return self._data


class FakeFirestoreDocument:
    def __init__(self, store: dict, key: str) -> None:
        self._store = store
        self._key = key

    def create(self, data: dict) -> None:
        if self._key in self._store:
            from google.api_core import exceptions as google_exceptions

            raise google_exceptions.AlreadyExists("exists")
        self._store[self._key] = data

    def get(self) -> FakeFirestoreDocumentSnapshot:
        data = self._store.get(self._key)
        return FakeFirestoreDocumentSnapshot(data)

    def update(self, data: dict) -> None:
        current = self._store.get(self._key, {})
        current.update(data)
        self._store[self._key] = current

    def delete(self) -> None:
        self._store.pop(self._key, None)


class FakeFirestoreCollection:
    def __init__(self, store: dict) -> None:
        self._store = store

    def document(self, key: str) -> FakeFirestoreDocument:
        return FakeFirestoreDocument(self._store, key)


class FakeFirestoreClient:
    def __init__(self) -> None:
        self._collections: dict[str, dict] = {}

    def collection(self, name: str) -> FakeFirestoreCollection:
        collection = self._collections.setdefault(name, {})
        return FakeFirestoreCollection(collection)


class FakeFirestoreModule(ModuleType):
    SERVER_TIMESTAMP = object()

    def __init__(self, client: FakeFirestoreClient) -> None:
        super().__init__("google.cloud.firestore")
        self._client = client

    def Client(self, project: str | None = None) -> FakeFirestoreClient:  # noqa: N802
        return self._client


class FakeExecutionResponse:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeExecutionsClient:
    def __init__(self, name: str) -> None:
        self._name = name

    def create_execution(self, request: dict, retry=None) -> FakeExecutionResponse:
        return FakeExecutionResponse(self._name)


def _install_fake_google_cloud(
    monkeypatch: pytest.MonkeyPatch,
    *,
    storage_module: ModuleType | None = None,
    firestore_module: ModuleType | None = None,
    workflows_module: ModuleType | None = None,
) -> None:
    google_module = sys.modules.get("google") or ModuleType("google")
    cloud_module = sys.modules.get("google.cloud") or ModuleType("google.cloud")
    if not hasattr(google_module, "__path__"):
        google_module.__path__ = []
    if not hasattr(cloud_module, "__path__"):
        cloud_module.__path__ = []
    if storage_module is not None:
        cloud_module.storage = storage_module
        monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)
    if firestore_module is not None:
        cloud_module.firestore = firestore_module
        monkeypatch.setitem(sys.modules, "google.cloud.firestore", firestore_module)
    if workflows_module is not None:
        cloud_module.workflows = workflows_module
        monkeypatch.setitem(sys.modules, "google.cloud.workflows", workflows_module)
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_dream2flow_entrypoint_parses_env_and_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_storage_module = ModuleType("google.cloud.storage")
    data_map = {
        "scenes/scene_1/assets/scene_manifest.json": (
            FIXTURES_DIR / "dream2flow_scene_manifest.json"
        ).read_text(),
        "scenes/scene_1/usd/scene.usda": "#usda 1.0",
    }
    uploads: dict[str, str] = {}
    fake_storage_module.Client = lambda: FakeStorageClient(data_map, uploads)
    fake_storage_module.Bucket = FakeBucket
    _install_fake_google_cloud(monkeypatch, storage_module=fake_storage_module)

    entrypoint = _load_module(
        "dream2flow_entrypoint",
        REPO_ROOT / "dream2flow-preparation-job" / "entrypoint.py",
    )

    captured: dict[str, object] = {}

    class FakeJob:
        def __init__(self, config) -> None:
            captured["config"] = config

        def run(self):
            output_dir = captured["config"].output_dir
            output_dir.joinpath("bundle.json").write_text("{}")
            return SimpleNamespace(
                bundles=[{"id": "bundle-1"}],
                num_successful_videos=1,
                num_successful_flows=1,
                generation_time_seconds=0.1,
                errors=[],
                success=True,
            )

    class FakeUploadResult:
        def __init__(self) -> None:
            self.success = True
            self.error = None

    monkeypatch.setattr(entrypoint, "Dream2FlowPreparationJob", FakeJob)
    monkeypatch.setattr(entrypoint, "upload_blob_from_filename", lambda *args, **kwargs: FakeUploadResult())
    monkeypatch.setattr(entrypoint, "send_alert", lambda **kwargs: None)

    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", "scene_1")
    monkeypatch.setenv("NUM_TASKS", "3")
    monkeypatch.setenv("ROBOT", "ur5e")

    with pytest.raises(SystemExit) as excinfo:
        entrypoint.main()

    assert excinfo.value.code == 0
    config = captured["config"]
    assert config.num_tasks == 3
    assert config.robot_embodiment.value == "ur5e"


def test_geniesim_submit_entrypoint_validates_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_storage_module = ModuleType("google.cloud.storage")
    geniesim_prefix = "scenes/scene_1/geniesim"
    data_map = {
        f"{geniesim_prefix}/scene_graph.json": (
            FIXTURES_DIR / "geniesim_scene_graph.json"
        ).read_text(),
        f"{geniesim_prefix}/asset_index.json": (
            FIXTURES_DIR / "geniesim_asset_index.json"
        ).read_text(),
        f"{geniesim_prefix}/task_config.json": (
            FIXTURES_DIR / "geniesim_task_config.json"
        ).read_text(),
        f"{geniesim_prefix}/_GENIESIM_EXPORT_COMPLETE": (
            FIXTURES_DIR / "geniesim_export_marker.json"
        ).read_text(),
        f"{geniesim_prefix}/export_manifest.json": json.dumps({
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
                "success": True, "scene_id": "scene_1", "output_dir": "geniesim",
                "outputs": {"scene_graph": "scene_graph.json", "asset_index": "asset_index.json", "task_config": "task_config.json", "scene_config": "scene_config.yaml"},
                "statistics": {"nodes": 1, "edges": 0, "assets": 1, "tasks": 1},
                "errors": [], "warnings": [],
            },
            "geniesim_compatibility": {"version": "3.0.0", "isaac_sim_version": "4.0", "formats": {"scene_graph": "json", "asset_index": "json", "task_config": "json", "scene_config": "yaml"}},
            "file_inventory": [],
            "checksums": {"files": {}},
        }),
    }
    uploads: dict[str, str] = {}
    fake_storage_module.Client = lambda: FakeStorageClient(data_map, uploads)
    fake_storage_module.Bucket = FakeBucket
    _install_fake_google_cloud(monkeypatch, storage_module=fake_storage_module)

    module = _load_module(
        "geniesim_submit_entrypoint",
        REPO_ROOT / "genie-sim-submit-job" / "submit_to_geniesim.py",
    )

    import tempfile as _tmpmod
    _approval_dir = _tmpmod.mkdtemp(prefix="bp_approval_")
    monkeypatch.setenv("QUALITY_APPROVAL_PATH", _approval_dir)
    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", "scene_1")
    monkeypatch.setenv("GENIESIM_PREFIX", geniesim_prefix)
    monkeypatch.setenv("EPISODES_PER_TASK", "7")
    monkeypatch.setenv("NUM_VARIATIONS", "2")
    monkeypatch.setenv("MIN_QUALITY_SCORE", "0.9")

    monkeypatch.setattr(module, "run_geniesim_preflight_or_exit", lambda *args, **kwargs: {"ok": True})

    def _fake_run_local(*args, **kwargs):
        return module.DataCollectionResult(
            success=True,
            task_name="mock",
            episodes_collected=7,
            episodes_passed=7,
        )

    monkeypatch.setattr(module, "_run_local_data_collection_with_handshake", _fake_run_local)
    monkeypatch.setattr(module, "upload_blob_from_filename", lambda *args, **kwargs: SimpleNamespace(success=True, error=None))

    exit_code = module.main()

    assert exit_code == 0
    job_payload = json.loads(uploads[f"{geniesim_prefix}/job.json"])
    assert job_payload["generation_params"]["episodes_per_task"] == 7
    assert job_payload["generation_params"]["num_variations"] == 2
    assert job_payload["generation_params"]["min_quality_score"] == 0.9
    assert job_payload["export_schema"]["geniesim_schema_version"] == "3.0.0"


@pytest.mark.skipif(
    not importlib.util.find_spec("flask"),
    reason="flask not installed",
)
def test_geniesim_import_webhook_persists_firestore(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_firestore_client = FakeFirestoreClient()
    fake_firestore_module = FakeFirestoreModule(fake_firestore_client)
    fake_workflows_module = ModuleType("google.cloud.workflows")
    fake_executions_module = ModuleType("google.cloud.workflows.executions_v1")
    fake_executions_module.ExecutionsClient = lambda: FakeExecutionsClient("execution-1")
    fake_executions_module.Execution = SimpleNamespace
    fake_workflows_module.executions_v1 = fake_executions_module

    _install_fake_google_cloud(
        monkeypatch,
        firestore_module=fake_firestore_module,
        workflows_module=fake_workflows_module,
    )
    monkeypatch.setitem(sys.modules, "google.cloud.workflows.executions_v1", fake_executions_module)

    module = _load_module(
        "geniesim_import_webhook",
        REPO_ROOT / "genie-sim-import-webhook" / "main.py",
    )

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("WEBHOOK_HMAC_SECRET", "secret")

    payload = json.loads((FIXTURES_DIR / "geniesim_webhook_payload.json").read_text())
    body = json.dumps(payload).encode("utf-8")
    signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()

    client = module.app.test_client()
    response = client.post(
        "/webhooks/geniesim/job-complete",
        data=body,
        headers={"X-Webhook-Signature": f"sha256={signature}"},
        content_type="application/json",
    )

    assert response.status_code == 202
    response_payload = response.get_json()
    assert response_payload["execution"] == "execution-1"
    assert fake_firestore_client._collections
