from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from tools import gcs_sync as gcs_sync_module


@dataclass
class _StoredBlob:
    content: bytes
    generation: str = "0"


class _FakeBlob:
    def __init__(self, name: str, store: Dict[str, _StoredBlob]) -> None:
        self.name = name
        self._store = store

    @property
    def generation(self) -> Optional[str]:
        payload = self._store.get(self.name)
        return payload.generation if payload else None

    def exists(self) -> bool:
        return self.name in self._store

    def download_to_filename(self, filename: str) -> None:
        payload = self._store[self.name]
        Path(filename).write_bytes(payload.content)

    def upload_from_filename(self, filename: str, content_type=None) -> None:
        self._store[self.name] = _StoredBlob(content=Path(filename).read_bytes(), generation="1")

    def upload_from_string(self, payload: str, content_type=None) -> None:
        encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
        self._store[self.name] = _StoredBlob(content=encoded, generation="1")


class _FakeBucket:
    def __init__(self, store: Dict[str, _StoredBlob]) -> None:
        self._store = store

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(name, self._store)

    def list_blobs(self, prefix: str):
        for name in sorted(self._store):
            if name.startswith(prefix):
                yield _FakeBlob(name, self._store)


class _FakeClient:
    def __init__(self, bucket: _FakeBucket) -> None:
        self._bucket = bucket

    def bucket(self, _bucket_name: str) -> _FakeBucket:
        return self._bucket


class _FakeStorageModule:
    def __init__(self, bucket: _FakeBucket) -> None:
        self._bucket = bucket

    def Client(self) -> _FakeClient:
        return _FakeClient(self._bucket)


def _make_sync(monkeypatch, local_scene_dir: Path, store: Dict[str, _StoredBlob], **kwargs):
    fake_bucket = _FakeBucket(store)
    monkeypatch.setattr(gcs_sync_module, "gcs_storage", _FakeStorageModule(fake_bucket))
    monkeypatch.setattr(gcs_sync_module, "upload_blob_from_filename", None)
    return gcs_sync_module.GCSSync(
        bucket_name="test-bucket",
        scene_id="scene-1",
        local_scene_dir=local_scene_dir,
        **kwargs,
    )


def test_download_inputs_prefers_explicit_object(monkeypatch, tmp_path):
    scene_dir = tmp_path / "scene"
    store = {
        "scenes/scene-1/images/custom_input.jpeg": _StoredBlob(
            content=b"jpeg-bytes",
            generation="1739220750732768",
        ),
    }
    sync = _make_sync(
        monkeypatch,
        scene_dir,
        store,
        input_object="scenes/scene-1/images/custom_input.jpeg",
    )

    downloaded = sync.download_inputs()

    assert downloaded.name == "custom_input.jpeg"
    assert downloaded.read_bytes() == b"jpeg-bytes"
    assert sync.input_object == "scenes/scene-1/images/custom_input.jpeg"
    assert sync.input_generation == "1739220750732768"


def test_download_inputs_falls_back_to_latest_generation(monkeypatch, tmp_path):
    scene_dir = tmp_path / "scene"
    store = {
        "scenes/scene-1/images/older.png": _StoredBlob(content=b"old", generation="10"),
        "scenes/scene-1/images/newer.jpg": _StoredBlob(content=b"new", generation="12"),
    }
    sync = _make_sync(monkeypatch, scene_dir, store)

    downloaded = sync.download_inputs()

    assert downloaded.name == "newer.jpg"
    assert downloaded.read_bytes() == b"new"
    assert sync.input_object == "scenes/scene-1/images/newer.jpg"
    assert sync.input_generation == "12"


def test_upload_all_outputs_flattens_directory_mapping(monkeypatch, tmp_path):
    scene_dir = tmp_path / "scene"
    (scene_dir / "assets").mkdir(parents=True)
    (scene_dir / "seg").mkdir(parents=True)
    (scene_dir / "assets" / "scene_manifest.json").write_text("{}")
    (scene_dir / "seg" / "inventory.json").write_text("{}")

    store: Dict[str, _StoredBlob] = {}
    sync = _make_sync(monkeypatch, scene_dir, store)

    results = sync.upload_all_outputs()

    assert "assets" in results
    assert "seg" in results
    assert results["assets"].files_synced == 1
    assert results["seg"].files_synced == 1
    assert "scenes/scene-1/assets/scene_manifest.json" in store
    assert "scenes/scene-1/seg/inventory.json" in store
