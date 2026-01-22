from __future__ import annotations

import types
import sys
from pathlib import Path


def test_download_recordings_fallback(monkeypatch, tmp_path, load_job_module):
    module = load_job_module("geniesim_import", "import_from_geniesim.py")

    class FakeBlob:
        def __init__(self, name: str, payload: bytes) -> None:
            self.name = name
            self.payload = payload
            self.size = len(payload)

        def download_to_filename(self, filename: str) -> None:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.payload)

    class FakeClient:
        last_bucket = None
        last_prefix = None

        def list_blobs(self, bucket: str, prefix: str):
            FakeClient.last_bucket = bucket
            FakeClient.last_prefix = prefix
            return [FakeBlob(f"{prefix}episode_000.json", b"{}")]

    storage_mod = types.ModuleType("google.cloud.storage")
    storage_mod.Client = FakeClient
    cloud_mod = types.ModuleType("google.cloud")
    cloud_mod.storage = storage_mod
    google_mod = types.ModuleType("google")
    google_mod.cloud = cloud_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_mod)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_mod)

    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("OUTPUT_PREFIX", "scenes/test/episodes")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = module._create_import_config(
        {
            "job_id": "job123",
            "output_dir": output_dir,
        }
    )

    log = module.logging.LoggerAdapter(module.logger, {"job_id": "test", "scene_id": "scene"})
    recordings_dir = module._resolve_recordings_dir(
        config,
        bucket="test-bucket",
        output_prefix="scenes/test/episodes",
        log=log,
    )

    assert recordings_dir.joinpath("episode_000.json").exists()
    assert FakeClient.last_bucket == "test-bucket"
    assert (
        FakeClient.last_prefix
        == "scenes/test/episodes/geniesim_job123/recordings/"
    )
