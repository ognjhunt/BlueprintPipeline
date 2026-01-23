import importlib.util
import json
from pathlib import Path

import pytest


def _load_entrypoint_checks_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "tools" / "validation" / "entrypoint_checks.py"
    spec = importlib.util.spec_from_file_location("entrypoint_checks", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_minimal_manifest(path: Path) -> None:
    manifest = {
        "version": "1.0.0",
        "scene_id": "scene_1",
        "scene": {"coordinate_frame": "y_up", "meters_per_unit": 1.0},
        "objects": [
            {
                "id": "object_1",
                "sim_role": "static",
                "transform": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation_euler": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "asset": {"path": "assets/object_1.usd"},
            }
        ],
    }
    path.write_text(json.dumps(manifest))


def test_validate_required_env_vars_success(monkeypatch: pytest.MonkeyPatch) -> None:
    entrypoint_checks = _load_entrypoint_checks_module()
    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", "scene_1")
    monkeypatch.setenv("ASSETS_PREFIX", "assets")
    monkeypatch.setenv("GENIESIM_PREFIX", "geniesim")

    entrypoint_checks.validate_required_env_vars(
        {"BUCKET": "Bucket name", "SCENE_ID": "Scene identifier"}, label="[TEST]"
    )


def test_validate_required_env_vars_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    entrypoint_checks = _load_entrypoint_checks_module()
    monkeypatch.delenv("BUCKET", raising=False)
    monkeypatch.setenv("SCENE_ID", "scene_1")

    with pytest.raises(SystemExit):
        entrypoint_checks.validate_required_env_vars(
            {"BUCKET": "Bucket name", "SCENE_ID": "Scene identifier"}, label="[TEST]"
        )


def test_validate_required_env_vars_webhook_missing_url(monkeypatch: pytest.MonkeyPatch) -> None:
    entrypoint_checks = _load_entrypoint_checks_module()
    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", "scene_1")
    monkeypatch.setenv("ALERT_BACKEND", "webhook")
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)

    with pytest.raises(SystemExit):
        entrypoint_checks.validate_required_env_vars(
            {"BUCKET": "Bucket name", "SCENE_ID": "Scene identifier"}, label="[TEST]"
        )


def test_validate_scene_manifest_success(tmp_path: Path) -> None:
    entrypoint_checks = _load_entrypoint_checks_module()
    manifest_path = tmp_path / "scene_manifest.json"
    _write_minimal_manifest(manifest_path)

    entrypoint_checks.validate_scene_manifest(manifest_path, label="[TEST]")


def test_validate_scene_manifest_missing(tmp_path: Path) -> None:
    entrypoint_checks = _load_entrypoint_checks_module()
    missing_path = tmp_path / "missing_manifest.json"

    with pytest.raises(SystemExit):
        entrypoint_checks.validate_scene_manifest(missing_path, label="[TEST]")
