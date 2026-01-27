from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _ensure_asset_catalog_client():
    import tools.asset_catalog as asset_catalog

    if not hasattr(asset_catalog, "AssetCatalogClient"):
        asset_catalog.AssetCatalogClient = SimpleNamespace


def test_simready_load_helpers(load_job_module, tmp_path: Path):
    _ensure_asset_catalog_client()
    module = load_job_module("simready", "prepare_simready_assets.py")

    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({"value": 1}))
    assert module.load_json(payload_path) == {"value": 1}

    root = Path("/root")
    assert module.safe_path_join(root, "/foo/bar") == root / "foo/bar"


def test_simready_main_env_validation(load_job_module, monkeypatch):
    _ensure_asset_catalog_client()
    module = load_job_module("simready", "prepare_simready_assets.py")

    monkeypatch.delenv("BUCKET", raising=False)
    monkeypatch.delenv("SCENE_ID", raising=False)
    monkeypatch.delenv("ASSETS_PREFIX", raising=False)

    with pytest.raises(SystemExit):
        module.main()


def test_simready_main_writes_complete_marker(load_job_module, tmp_path: Path, monkeypatch):
    _ensure_asset_catalog_client()
    module = load_job_module("simready", "prepare_simready_assets.py")

    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", "scene-1")
    monkeypatch.setenv("ASSETS_PREFIX", "scenes/scene-1/assets")

    assets_root = tmp_path / "scenes/scene-1/assets"
    assets_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module, "GCS_ROOT", tmp_path)
    monkeypatch.setattr(module, "validate_required_env_vars", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "validate_scene_manifest", lambda *args, **kwargs: None)

    def _fake_run_from_env(root):
        marker_path = root / "scenes/scene-1/assets/.simready_complete"
        marker_path.write_text("complete")
        return 0

    import blueprint_sim.simready as simready_module

    monkeypatch.setattr(simready_module, "run_from_env", _fake_run_from_env)

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 0
    assert (assets_root / ".simready_complete").is_file()


def test_simready_main_failure_marker(load_job_module, tmp_path: Path, monkeypatch):
    _ensure_asset_catalog_client()
    module = load_job_module("simready", "prepare_simready_assets.py")

    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", "scene-1")
    monkeypatch.setenv("ASSETS_PREFIX", "scenes/scene-1/assets")

    monkeypatch.setattr(module, "GCS_ROOT", tmp_path)
    monkeypatch.setattr(module, "validate_required_env_vars", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "validate_scene_manifest", lambda *args, **kwargs: None)

    import blueprint_sim.simready as simready_module

    def _fake_run_from_env(root):
        raise RuntimeError("boom")

    monkeypatch.setattr(simready_module, "run_from_env", _fake_run_from_env)

    calls = {}

    class _FakeFailureMarkerWriter:
        def __init__(self, bucket, scene_id, job_name):
            self.bucket = bucket
            self.scene_id = scene_id
            self.job_name = job_name

        def write_failure(self, **kwargs):
            calls["called"] = True
            calls["kwargs"] = kwargs
            return tmp_path / "failed.json"

    monkeypatch.setattr(module, "FailureMarkerWriter", _FakeFailureMarkerWriter)

    with pytest.raises(RuntimeError):
        module.main()

    assert calls.get("called") is True
