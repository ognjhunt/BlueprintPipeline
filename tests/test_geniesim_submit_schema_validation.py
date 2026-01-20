import importlib.util
import json
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


def _build_export_manifest() -> dict:
    return {
        "schema_version": "2.0",
        "schema_definition": {
            "version": "2.0",
            "description": "Test schema",
            "fields": {"schema_version": "Schema version"},
            "notes": ["note"],
        },
        "export_info": {
            "timestamp": "2024-01-01T00:00:00Z",
            "exporter_version": "1.0.0",
            "source_pipeline": "blueprintpipeline",
        },
        "asset_provenance_path": None,
        "config": {
            "robot_type": "franka",
            "generate_embeddings": False,
            "embedding_model": None,
            "require_embeddings": False,
            "filter_commercial_only": True,
            "max_tasks": 5,
            "lerobot_export_format": "lerobot_v2",
        },
        "result": {
            "success": True,
            "scene_id": "scene-1",
            "output_dir": "/tmp/output",
            "outputs": {
                "scene_graph": "scene_graph.json",
                "asset_index": "asset_index.json",
                "task_config": "task_config.json",
                "scene_config": None,
            },
            "statistics": {"nodes": 1, "edges": 2, "assets": 3, "tasks": 4},
            "errors": [],
            "warnings": [],
        },
        "geniesim_compatibility": {
            "version": "3.0",
            "isaac_sim_version": "5.1.0",
            "formats": {
                "scene_graph": "json",
                "asset_index": "json",
                "task_config": "json",
                "scene_config": "yaml",
            },
        },
        "file_inventory": [{"path": "scene_graph.json", "size_bytes": 123}],
        "checksums": {
            "files": {
                "scene_graph.json": {"sha256": "abc", "size_bytes": 123},
                "export_manifest.json": {"sha256": "def"},
            }
        },
    }


def test_export_manifest_schema_accepts_valid_payload(submit_module: types.ModuleType) -> None:
    payload = _build_export_manifest()

    submit_module._validate_bundle_schemas({"export_manifest": payload})


def test_export_manifest_schema_missing_required_field(submit_module: types.ModuleType) -> None:
    payload = _build_export_manifest()
    payload.pop("export_info")

    with pytest.raises(RuntimeError) as excinfo:
        submit_module._validate_bundle_schemas({"export_manifest": payload})

    message = str(excinfo.value)
    assert "export_manifest.json" in message
    assert "export_info" in message


class FakeBlob:
    def __init__(self, text: str) -> None:
        self._text = text

    def exists(self) -> bool:
        return True

    def download_as_text(self) -> str:
        return self._text


class FakeBucket:
    def __init__(self, blobs: dict[str, FakeBlob]) -> None:
        self._blobs = blobs

    def blob(self, name: str) -> FakeBlob:
        return self._blobs[name]


class FakeClient:
    def __init__(self, blobs: dict[str, FakeBlob]) -> None:
        self._bucket = FakeBucket(blobs)

    def bucket(self, name: str) -> FakeBucket:
        return self._bucket


def test_optional_json_blob_schema_rejects_invalid_payloads(
    submit_module: types.ModuleType,
) -> None:
    blobs = {
        "job.json": FakeBlob(json.dumps(["not", "an", "object"])),
        "job_idempotency.json": FakeBlob(json.dumps({"key": "only"})),
    }
    client = FakeClient(blobs)

    with pytest.raises(RuntimeError):
        submit_module._read_optional_json_blob(
            client,
            "bucket",
            "job.json",
            schema_name="job_output.schema.json",
        )

    with pytest.raises(RuntimeError):
        submit_module._read_optional_json_blob(
            client,
            "bucket",
            "job_idempotency.json",
            schema_name="job_idempotency.schema.json",
        )
