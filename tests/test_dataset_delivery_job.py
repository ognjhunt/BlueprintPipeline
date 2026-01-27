import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


class FakeBlob:
    def __init__(self, bucket, name, data=b""):
        self.bucket = bucket
        self.name = name
        self.data = data
        self.size = len(data)

    def download_as_text(self):
        return self.data.decode("utf-8")

    def download_to_filename(self, filename):
        Path(filename).write_bytes(self.data)
        self.size = len(self.data)

    def upload_from_filename(self, filename):
        self.data = Path(filename).read_bytes()
        self.size = len(self.data)
        self.bucket.blobs[self.name] = self

    def upload_from_string(self, data, content_type=None):
        if isinstance(data, str):
            self.data = data.encode("utf-8")
        else:
            self.data = data
        self.size = len(self.data)
        self.bucket.blobs[self.name] = self

    def exists(self):
        return self.name in self.bucket.blobs

    def delete(self):
        self.bucket.blobs.pop(self.name, None)


class FakeBucket:
    def __init__(self, name):
        self.name = name
        self.blobs = {}

    def blob(self, name):
        if name not in self.blobs:
            self.blobs[name] = FakeBlob(self, name, b"")
        return self.blobs[name]

    def copy_blob(self, source_blob, dest_bucket, new_name=None):
        blob_name = new_name or source_blob.name
        copied = FakeBlob(dest_bucket, blob_name, source_blob.data)
        dest_bucket.blobs[blob_name] = copied
        return copied


class FakeStorageClient:
    def __init__(self, buckets=None):
        self._buckets = buckets or {}

    def bucket(self, name):
        if name not in self._buckets:
            self._buckets[name] = FakeBucket(name)
        return self._buckets[name]


class TempDirFactory:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1
        path = self.base_path / f"tmp-{self.counter}"
        path.mkdir()
        return _TempDir(path)


class _TempDir:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        return str(self.path)

    def __exit__(self, exc_type, exc, tb):
        return False


def _load_module(name: str):
    module_path = Path(__file__).resolve().parents[1] / "dataset-delivery-job/dataset_delivery.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load dataset_delivery module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_json(bucket, object_name, payload):
    data = json.dumps(payload).encode("utf-8")
    bucket.blobs[object_name] = FakeBlob(bucket, object_name, data)


def _seed_bytes(bucket, object_name, payload: bytes):
    bucket.blobs[object_name] = FakeBlob(bucket, object_name, payload)


def _setup_env(monkeypatch, manifest_uri: str):
    monkeypatch.setenv("BUCKET", "source-bucket")
    monkeypatch.setenv("SCENE_ID", "scene-1")
    monkeypatch.setenv("JOB_ID", "job-1")
    monkeypatch.setenv("IMPORT_MANIFEST_PATH", manifest_uri)


def _setup_tempdir(monkeypatch, module, tmp_path):
    temp_factory = TempDirFactory(tmp_path)
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", temp_factory)
    return temp_factory


def _setup_requests(monkeypatch, module):
    calls = []

    def _post(*args, **kwargs):
        calls.append((args, kwargs))
        class Response:
            status_code = 200
            text = "ok"
        return Response()

    monkeypatch.setattr(module.requests, "post", _post)
    return calls


def _setup_storage(monkeypatch, module, client):
    monkeypatch.setattr(module.storage, "Client", lambda: client)


def _build_manifest():
    return {
        "scene_id": "scene-1",
        "job_id": "job-1",
        "gcs_output_path": "gs://source-bucket/scenes/scene-1/geniesim/output",
        "package": {"path": "bundle/package.zip"},
        "checksums_path": "checksums.json",
        "asset_provenance_path": "legal/asset_provenance.json",
        "provenance": {"scene_id": "scene-1"},
    }


def _seed_common_objects(client, manifest, package_bytes, checksums, asset_provenance):
    source_bucket = client.bucket("source-bucket")
    _seed_json(source_bucket, "scenes/scene-1/geniesim/import_manifest.json", manifest)
    _seed_json(source_bucket, "scenes/scene-1/geniesim/output/checksums.json", checksums)
    _seed_json(source_bucket, "scenes/scene-1/geniesim/output/legal/asset_provenance.json", asset_provenance)
    _seed_bytes(source_bucket, "scenes/scene-1/geniesim/output/bundle/package.zip", package_bytes)


def test_dataset_delivery_happy_path(monkeypatch, tmp_path):
    module = _load_module("dataset_delivery_happy")
    manifest = _build_manifest()
    package_bytes = b"package-bytes"
    checksum = hashlib.sha256(package_bytes).hexdigest()
    checksums = {
        "bundle_files": {
            "bundle/package.zip": {
                "sha256": checksum,
                "size_bytes": len(package_bytes),
            }
        }
    }
    asset_provenance = {"assets": [{"asset_id": "asset-1", "license": "cc-by"}]}
    client = FakeStorageClient()
    _seed_common_objects(client, manifest, package_bytes, checksums, asset_provenance)

    _setup_env(monkeypatch, "gs://source-bucket/scenes/scene-1/geniesim/import_manifest.json")
    monkeypatch.setenv("LAB_DELIVERY_BUCKETS", "lab1=dest-bucket")
    monkeypatch.setenv("LAB_WEBHOOK_URLS", "lab1=https://example.com/hook")

    _setup_storage(monkeypatch, module, client)
    _setup_tempdir(monkeypatch, module, tmp_path)
    calls = _setup_requests(monkeypatch, module)

    result = module.main()

    assert result == 0
    assert calls
    dest_bucket = client.bucket("dest-bucket")
    assert "deliveries/scene-1/job-1/bundle/package.zip" in dest_bucket.blobs
    assert "deliveries/scene-1/job-1/dataset_card.json" in dest_bucket.blobs


@pytest.mark.parametrize("missing_field", ["package", "checksums_path"])
def test_dataset_delivery_invalid_manifest(monkeypatch, tmp_path, missing_field):
    module = _load_module(f"dataset_delivery_invalid_{missing_field}")
    manifest = _build_manifest()
    if missing_field == "package":
        manifest["package"] = {}
    else:
        manifest.pop("checksums_path", None)

    client = FakeStorageClient()
    _seed_json(
        client.bucket("source-bucket"),
        "scenes/scene-1/geniesim/import_manifest.json",
        manifest,
    )

    _setup_env(monkeypatch, "gs://source-bucket/scenes/scene-1/geniesim/import_manifest.json")
    monkeypatch.setenv("LAB_DELIVERY_BUCKETS", "lab1=dest-bucket")

    _setup_storage(monkeypatch, module, client)
    _setup_tempdir(monkeypatch, module, tmp_path)
    _setup_requests(monkeypatch, module)

    assert module.main() == 1


@pytest.mark.parametrize("license_value", ["cc-by-nc", "unknown"])
def test_dataset_delivery_license_failure(monkeypatch, tmp_path, license_value):
    module = _load_module(f"dataset_delivery_license_{license_value}")
    manifest = _build_manifest()
    package_bytes = b"package-bytes"
    checksum = hashlib.sha256(package_bytes).hexdigest()
    checksums = {"bundle_files": {"bundle/package.zip": {"sha256": checksum}}}
    asset_provenance = {"assets": [{"asset_id": "asset-1", "license": license_value}]}
    client = FakeStorageClient()
    _seed_common_objects(client, manifest, package_bytes, checksums, asset_provenance)

    _setup_env(monkeypatch, "gs://source-bucket/scenes/scene-1/geniesim/import_manifest.json")
    monkeypatch.setenv("LAB_DELIVERY_BUCKETS", "lab1=dest-bucket")

    _setup_storage(monkeypatch, module, client)
    _setup_tempdir(monkeypatch, module, tmp_path)
    _setup_requests(monkeypatch, module)

    assert module.main() == 1


def test_dataset_delivery_missing_delivery_config(monkeypatch, tmp_path):
    module = _load_module("dataset_delivery_missing_delivery")
    manifest = _build_manifest()
    package_bytes = b"package-bytes"
    checksum = hashlib.sha256(package_bytes).hexdigest()
    checksums = {"bundle_files": {"bundle/package.zip": {"sha256": checksum}}}
    asset_provenance = {"assets": [{"asset_id": "asset-1", "license": "cc-by"}]}
    client = FakeStorageClient()
    _seed_common_objects(client, manifest, package_bytes, checksums, asset_provenance)

    _setup_env(monkeypatch, "gs://source-bucket/scenes/scene-1/geniesim/import_manifest.json")
    monkeypatch.delenv("LAB_DELIVERY_BUCKETS", raising=False)
    monkeypatch.delenv("DEFAULT_DELIVERY_BUCKET", raising=False)

    _setup_storage(monkeypatch, module, client)
    _setup_tempdir(monkeypatch, module, tmp_path)
    _setup_requests(monkeypatch, module)

    assert module.main() == 1
