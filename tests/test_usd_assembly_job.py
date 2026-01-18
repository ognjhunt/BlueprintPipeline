import importlib.util
import json
import os
from pathlib import Path

import pytest


def load_assemble_scene_module() -> object:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "usd-assembly-job"
        / "assemble_scene.py"
    )
    spec = importlib.util.spec_from_file_location(
        "usd_assembly_job_assemble_scene",
        module_path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_usd_assembly_job_stubs(tmp_path, monkeypatch):
    assemble_scene = load_assemble_scene_module()

    def fake_assemble_from_env() -> int:
        usd_prefix = os.environ["USD_PREFIX"]
        usd_path = tmp_path / usd_prefix / "scene.usda"
        usd_path.parent.mkdir(parents=True, exist_ok=True)
        usd_path.write_text("#usda 1.0")
        return 0

    class DummyQualityGateRegistry:
        def __init__(self, *args, **kwargs):
            self.verbose = kwargs.get("verbose")

        def run_checkpoint(self, *args, **kwargs) -> None:
            return None

        def save_report(self, *args, **kwargs) -> None:
            return None

        def can_proceed(self) -> bool:
            return True

    def patched_path(*args, **kwargs):
        if args and args[0] == "/mnt/gcs":
            return tmp_path
        return Path(*args, **kwargs)

    monkeypatch.setattr(assemble_scene, "GCS_ROOT", tmp_path)
    monkeypatch.setattr(assemble_scene, "assemble_from_env", fake_assemble_from_env)
    monkeypatch.setattr(assemble_scene, "QualityGateRegistry", DummyQualityGateRegistry)
    monkeypatch.setattr(assemble_scene, "Path", patched_path)

    scene_id = "scene-123"
    assets_prefix = f"scenes/{scene_id}/assets"
    layout_prefix = f"scenes/{scene_id}/layout"
    usd_prefix = f"scenes/{scene_id}/usd"

    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", scene_id)
    monkeypatch.setenv("LAYOUT_PREFIX", layout_prefix)
    monkeypatch.setenv("ASSETS_PREFIX", assets_prefix)
    monkeypatch.setenv("USD_PREFIX", usd_prefix)

    assets_root = tmp_path / assets_prefix
    assets_root.mkdir(parents=True, exist_ok=True)
    manifest_path = assets_root / "scene_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "scene_id": scene_id,
                "scene": {
                    "coordinate_frame": "y_up",
                    "meters_per_unit": 1.0,
                },
                "objects": [
                    {
                        "id": "object-1",
                        "sim_role": "static",
                        "transform": {
                            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                        },
                        "asset": {"path": "assets/object-1.usd"},
                    }
                ],
            }
        )
    )

    with pytest.raises(SystemExit) as excinfo:
        assemble_scene.main()

    assert excinfo.value.code == 0

    completion_manifest_path = tmp_path / usd_prefix / "usd_assembly_manifest.json"
    assert completion_manifest_path.exists()

    completion_manifest = json.loads(completion_manifest_path.read_text())
    assert completion_manifest["status"]
    assert completion_manifest["completed_at"]
    assert completion_manifest["usd_prefix"] == usd_prefix
