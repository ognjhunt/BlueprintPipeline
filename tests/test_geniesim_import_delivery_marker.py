from __future__ import annotations

import json
from pathlib import Path

import pytest


class FakePreconditionFailed(Exception):
    code = 412


class FakeBlob:
    def __init__(self, store: dict[str, str], name: str) -> None:
        self._store = store
        self._name = name

    def exists(self) -> bool:
        return self._name in self._store

    def upload_from_string(
        self,
        data: str,
        *,
        content_type: str | None = None,
        if_generation_match: int | None = None,
    ) -> None:
        if if_generation_match == 0 and self.exists():
            raise FakePreconditionFailed("precondition failed")
        self._store[self._name] = data

    def download_as_text(self) -> str:
        return self._store[self._name]


class FakeBucket:
    def __init__(self, store: dict[str, str]) -> None:
        self._store = store

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self._store, name)


class FakeStorageClient:
    def __init__(self) -> None:
        self._buckets: dict[str, dict[str, str]] = {}

    def bucket(self, name: str) -> FakeBucket:
        store = self._buckets.setdefault(name, {})
        return FakeBucket(store)


def test_delivery_marker_double_write_fails_without_override(
    load_job_module,
    temp_test_dir: Path,
) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")
    storage_client = FakeStorageClient()
    bucket = "test-bucket"
    scene_id = "scene-123"
    run_id = "run-abc"
    job_id = "job-456"

    first = module._write_delivery_marker(
        bucket=bucket,
        scene_id=scene_id,
        job_id=job_id,
        run_id=run_id,
        idempotency_key="idem-key",
        allow_idempotent_retry=False,
        log=module.logging.LoggerAdapter(module.logger, {}),
        storage_client=storage_client,
        local_root=temp_test_dir,
    )

    assert first.payload["scene_id"] == scene_id
    assert first.gcs_uri == f"gs://{bucket}/scenes/{scene_id}/geniesim/delivery/{run_id}.json"
    assert first.local_path == temp_test_dir / bucket / "scenes" / scene_id / "geniesim" / "delivery" / f"{run_id}.json"
    assert json.loads(first.local_path.read_text())["job_id"] == job_id

    with pytest.raises(module.DeliveryMarkerExistsError):
        module._write_delivery_marker(
            bucket=bucket,
            scene_id=scene_id,
            job_id=job_id,
            run_id=run_id,
            idempotency_key="idem-key",
            allow_idempotent_retry=False,
            log=module.logging.LoggerAdapter(module.logger, {}),
            storage_client=storage_client,
            local_root=temp_test_dir,
        )
