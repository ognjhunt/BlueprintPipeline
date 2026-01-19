"""Tests for Firebase upload helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.firebase_upload import uploader


class FakeBlob:
    """Simple blob stub for Firebase Storage."""

    def __init__(self, name: str, *, exists: bool = False, md5_hash: str | None = None, metadata=None):
        self.name = name
        self._exists = exists
        self.md5_hash = md5_hash
        self.metadata = metadata or {}
        self.upload_calls = []
        self.delete_calls = 0

    def exists(self) -> bool:
        return self._exists

    def upload_from_filename(self, file_path: str, content_type: str | None = None) -> None:
        self.upload_calls.append({"file_path": file_path, "content_type": content_type})
        local_hashes = uploader._calculate_file_hashes(Path(file_path))
        self.md5_hash = local_hashes["md5_base64"]
        self._exists = True

    def reload(self) -> None:
        return None

    def delete(self) -> None:
        self.delete_calls += 1
        self._exists = False


class FakeBucket:
    """Bucket stub that manages blobs by name."""

    def __init__(self, blobs: dict[str, FakeBlob] | None = None):
        self.blobs = blobs or {}

    def blob(self, name: str) -> FakeBlob:
        if name not in self.blobs:
            self.blobs[name] = FakeBlob(name)
        return self.blobs[name]


@pytest.fixture(autouse=True)
def _reset_firebase_app(monkeypatch):
    monkeypatch.setattr(uploader, "_FIREBASE_APP", None)


def _setup_firebase_env(monkeypatch) -> None:
    monkeypatch.setenv("FIREBASE_STORAGE_BUCKET", "test-bucket")
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_JSON", "{}")
    monkeypatch.setenv("FIREBASE_UPLOAD_CONCURRENCY", "1")


def _setup_firebase_stubs(monkeypatch, bucket: FakeBucket) -> None:
    monkeypatch.setattr(uploader.credentials, "Certificate", lambda *args, **kwargs: object())
    monkeypatch.setattr(uploader.firebase_admin, "initialize_app", lambda *args, **kwargs: object())
    monkeypatch.setattr(uploader.storage, "bucket", lambda: bucket)


def _write_episode_file(episodes_dir: Path, relative_path: str, content: bytes) -> Path:
    file_path = episodes_dir / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(content)
    return file_path


def test_init_firebase_missing_bucket_raises(monkeypatch):
    monkeypatch.delenv("FIREBASE_STORAGE_BUCKET", raising=False)
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_JSON", "{}")

    with pytest.raises(ValueError, match="FIREBASE_STORAGE_BUCKET"):
        uploader.init_firebase()


def test_init_firebase_invalid_service_json_raises(monkeypatch):
    monkeypatch.setenv("FIREBASE_STORAGE_BUCKET", "test-bucket")
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_JSON", "not-json")

    with pytest.raises(ValueError, match="FIREBASE_SERVICE_ACCOUNT_JSON"):
        uploader.init_firebase()


def test_init_firebase_missing_service_account_path_raises(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing.json"
    monkeypatch.setenv("FIREBASE_STORAGE_BUCKET", "test-bucket")
    monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_PATH", str(missing_path))

    with pytest.raises(FileNotFoundError, match="FIREBASE_SERVICE_ACCOUNT_PATH not found"):
        uploader.init_firebase()


def test_upload_episodes_to_firebase_uploads(monkeypatch, tmp_path):
    _setup_firebase_env(monkeypatch)
    bucket = FakeBucket()
    _setup_firebase_stubs(monkeypatch, bucket)

    episodes_dir = tmp_path / "episodes"
    file_path = _write_episode_file(episodes_dir, "episode_1/data.txt", b"hello")

    summary = uploader.upload_episodes_to_firebase(episodes_dir, scene_id="scene-1", prefix="datasets")

    assert summary["uploaded"] == 1
    assert summary["skipped"] == 0
    assert summary["reuploaded"] == 0
    assert summary["failed"] == 0
    remote_path = "datasets/scene-1/episode_1/data.txt"
    assert bucket.blobs[remote_path].upload_calls == [
        {"file_path": str(file_path), "content_type": "text/plain"}
    ]


def test_upload_episodes_to_firebase_skips_matching_blob(monkeypatch, tmp_path):
    _setup_firebase_env(monkeypatch)
    episodes_dir = tmp_path / "episodes"
    file_path = _write_episode_file(episodes_dir, "episode_1/data.txt", b"hello")
    local_hashes = uploader._calculate_file_hashes(file_path)

    remote_path = "datasets/scene-1/episode_1/data.txt"
    bucket = FakeBucket(
        {
            remote_path: FakeBlob(
                remote_path,
                exists=True,
                md5_hash=local_hashes["md5_base64"],
                metadata={"sha256": local_hashes["sha256_hex"]},
            )
        }
    )
    _setup_firebase_stubs(monkeypatch, bucket)

    summary = uploader.upload_episodes_to_firebase(episodes_dir, scene_id="scene-1", prefix="datasets")

    assert summary["uploaded"] == 0
    assert summary["skipped"] == 1
    assert summary["reuploaded"] == 0
    assert summary["failed"] == 0
    assert bucket.blobs[remote_path].upload_calls == []


def test_upload_episodes_to_firebase_reuploads_mismatched_blob(monkeypatch, tmp_path):
    _setup_firebase_env(monkeypatch)
    episodes_dir = tmp_path / "episodes"
    file_path = _write_episode_file(episodes_dir, "episode_1/data.txt", b"hello")

    remote_path = "datasets/scene-1/episode_1/data.txt"
    bucket = FakeBucket(
        {
            remote_path: FakeBlob(
                remote_path,
                exists=True,
                md5_hash="wrong",
                metadata={"sha256": "wrong"},
            )
        }
    )
    _setup_firebase_stubs(monkeypatch, bucket)

    summary = uploader.upload_episodes_to_firebase(episodes_dir, scene_id="scene-1", prefix="datasets")

    assert summary["uploaded"] == 0
    assert summary["skipped"] == 0
    assert summary["reuploaded"] == 1
    assert summary["failed"] == 0
    blob = bucket.blobs[remote_path]
    assert blob.delete_calls == 1
    assert blob.upload_calls == [{"file_path": str(file_path), "content_type": "text/plain"}]


def test_upload_episodes_to_firebase_raises_on_verification_failure(monkeypatch, tmp_path):
    _setup_firebase_env(monkeypatch)
    episodes_dir = tmp_path / "episodes"
    _write_episode_file(episodes_dir, "episode_1/data.txt", b"hello")

    bucket = FakeBucket()
    _setup_firebase_stubs(monkeypatch, bucket)

    def _failed_verify(*args, **kwargs):
        return False, {
            "expected_md5": "expected",
            "actual_md5": "actual",
            "expected_sha256": "expected",
            "actual_sha256": "actual",
            "hash_strategy": "sha256_metadata",
        }

    monkeypatch.setattr(uploader, "_verify_blob_checksum", _failed_verify)

    with pytest.raises(uploader.FirebaseUploadError) as exc_info:
        uploader.upload_episodes_to_firebase(episodes_dir, scene_id="scene-1", prefix="datasets")

    summary = exc_info.value.summary
    assert summary["failed"] == 1
    assert summary["failures"]
    assert summary["verification_failed"]


def test_upload_episodes_to_firebase_raises_on_upload_failure(monkeypatch, tmp_path):
    _setup_firebase_env(monkeypatch)
    episodes_dir = tmp_path / "episodes"
    _write_episode_file(episodes_dir, "episode_1/data.txt", b"hello")

    bucket = FakeBucket()
    _setup_firebase_stubs(monkeypatch, bucket)

    def _boom(*args, **kwargs):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(uploader, "_upload_file", _boom)

    with pytest.raises(uploader.FirebaseUploadError) as exc_info:
        uploader.upload_episodes_to_firebase(episodes_dir, scene_id="scene-1", prefix="datasets")

    summary = exc_info.value.summary
    assert summary["failed"] == 1
    assert summary["failures"][0]["error"] == "upload failed"
