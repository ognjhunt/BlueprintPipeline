import importlib.util
import json
import os
from pathlib import Path

from fixtures.generate_mock_regen3d import generate_mock_regen3d
from tools.validation.entrypoint_checks import validate_scene_manifest


def load_regen3d_adapter_job() -> object:
    module_path = Path(__file__).resolve().parents[1] / "regen3d-job" / "regen3d_adapter_job.py"
    spec = importlib.util.spec_from_file_location("regen3d_adapter_job", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_regen3d_adapter_job_outputs(tmp_path, monkeypatch) -> None:
    module = load_regen3d_adapter_job()

    scene_id = "test_kitchen_scene"
    generate_mock_regen3d(tmp_path, scene_id, environment_type="kitchen")

    monkeypatch.setattr(module, "GCS_ROOT", tmp_path)
    monkeypatch.setenv("BUCKET", "local-test-bucket")
    monkeypatch.setenv("SCENE_ID", scene_id)
    monkeypatch.setenv("REGEN3D_PREFIX", f"scenes/{scene_id}/regen3d")
    monkeypatch.setenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    monkeypatch.setenv("LAYOUT_PREFIX", f"scenes/{scene_id}/layout")
    monkeypatch.setenv("ENVIRONMENT_TYPE", "kitchen")
    monkeypatch.setenv("SCALE_FACTOR", "1.0")
    monkeypatch.setenv("BYPASS_QUALITY_GATES", "true")

    real_run = module.run_regen3d_adapter_job
    call_state = {"count": 0}

    def run_once(*args, **kwargs):
        call_state["count"] += 1
        if call_state["count"] > 1:
            return 0
        if "bucket" not in kwargs and len(args) < 2:
            kwargs["bucket"] = os.environ["BUCKET"]
        return real_run(*args, **kwargs)

    monkeypatch.setattr(module, "run_regen3d_adapter_job", run_once)

    exit_code = module.main()
    assert exit_code == 0

    assets_dir = tmp_path / f"scenes/{scene_id}/assets"
    layout_dir = tmp_path / f"scenes/{scene_id}/layout"
    seg_dir = tmp_path / f"scenes/{scene_id}/seg"

    manifest_path = assets_dir / "scene_manifest.json"
    layout_path = layout_dir / "scene_layout_scaled.json"
    inventory_path = seg_dir / "inventory.json"

    assert manifest_path.exists()
    assert layout_path.exists()
    assert inventory_path.exists()

    manifest = json.loads(manifest_path.read_text())
    layout = json.loads(layout_path.read_text())
    inventory = json.loads(inventory_path.read_text())

    validate_scene_manifest(manifest_path, label="[TEST]")

    assert layout["scene_id"] == scene_id
    assert layout["objects"]

    assert manifest["objects"]
    assert all(obj.get("id") for obj in manifest["objects"])
    assert all(obj.get("sim_role") for obj in manifest["objects"])

    inventory_objects = {obj["id"]: obj for obj in inventory["objects"]}
    manifest_objects = {obj["id"]: obj for obj in manifest["objects"]}

    assert inventory_objects["fridge"]["is_floor_contact"] is True
    assert inventory_objects["mug_0"]["is_floor_contact"] is False

    assert manifest_objects["fridge"]["physics_hints"]["is_floor_contact"] is True
    assert manifest_objects["mug_0"]["physics_hints"]["is_floor_contact"] is False
