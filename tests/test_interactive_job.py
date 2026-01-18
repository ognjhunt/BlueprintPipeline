from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from fixtures.generate_mock_regen3d import create_minimal_glb

MODULE_PATH = REPO_ROOT / "interactive-job" / "run_interactive_assets.py"
SPEC = importlib.util.spec_from_file_location("run_interactive_assets", MODULE_PATH)
run_interactive_assets = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_interactive_assets)


def write_scene_manifest(assets_root: Path, scene_id: str, objects: list[dict]) -> None:
    manifest = {
        "version": "1.0.0",
        "scene_id": scene_id,
        "scene": {
            "environment_type": "kitchen",
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
            "room": {
                "bounds": {
                    "width": 2.0,
                    "depth": 2.0,
                    "height": 2.0,
                }
            },
        },
        "objects": objects,
    }
    (assets_root / "scene_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_scene_assets(assets_prefix: str, object_ids: list[str]) -> Path:
    root = Path("/mnt/gcs")
    root.mkdir(parents=True, exist_ok=True)
    assets_root = root / assets_prefix
    regen3d_root = assets_root / "regen3d"
    regen3d_root.mkdir(parents=True, exist_ok=True)

    objects = []
    for obj_id in object_ids:
        obj_name = f"obj_{obj_id}"
        obj_dir = regen3d_root / obj_name
        obj_dir.mkdir(parents=True, exist_ok=True)
        glb_path = obj_dir / f"obj_{obj_id}.glb"
        glb_path.write_bytes(create_minimal_glb())

        objects.append(
            {
                "id": obj_id,
                "name": obj_id,
                "category": "mug",
                "description": "coffee mug",
                "sim_role": "manipulable_object",
                "dimensions_est": {
                    "width": 0.08,
                    "depth": 0.08,
                    "height": 0.1,
                },
                "transform": {
                    "position": {"x": 0.1, "y": 0.2, "z": 0.8},
                    "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "asset": {"path": f"regen3d/{obj_name}/obj_{obj_id}.glb"},
                "physics": {"mass": 0.2},
                "physics_hints": {"material_type": "ceramic"},
                "semantics": {"affordances": ["Graspable", "Containable"]},
                "relationships": [],
            }
        )

    write_scene_manifest(assets_root, "interactive_scene", objects)
    return assets_root


def run_job(monkeypatch, assets_prefix: str, disallow_placeholder: bool, mock_placeholder: bool) -> None:
    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", "interactive_scene")
    monkeypatch.setenv("ASSETS_PREFIX", assets_prefix)
    monkeypatch.setenv("REGEN3D_PREFIX", f"{assets_prefix}/regen3d")
    monkeypatch.setenv("INTERACTIVE_MODE", "glb")
    monkeypatch.setenv("PARTICULATE_MODE", "mock")
    monkeypatch.setenv("DISALLOW_PLACEHOLDER_URDF", str(disallow_placeholder).lower())
    monkeypatch.setenv("PARTICULATE_MOCK_PLACEHOLDER", str(mock_placeholder).lower())
    monkeypatch.delenv("PARTICULATE_ENDPOINT", raising=False)
    monkeypatch.delenv("PARTICULATE_LOCAL_ENDPOINT", raising=False)
    monkeypatch.delenv("PARTICULATE_LOCAL_MODEL", raising=False)
    monkeypatch.delenv("APPROVED_PARTICULATE_MODELS", raising=False)
    monkeypatch.delenv("PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("LABS_MODE", raising=False)
    monkeypatch.delenv("MULTIVIEW_PREFIX", raising=False)

    run_interactive_assets.main()


def test_interactive_job_mock_glb_outputs(tmp_path, monkeypatch) -> None:
    assets_prefix = f"interactive-{tmp_path.name}"
    assets_root = build_scene_assets(assets_prefix, ["mug_0", "mug_1"])

    run_job(monkeypatch, assets_prefix, disallow_placeholder=False, mock_placeholder=False)

    assert (assets_root / ".interactive_complete").is_file()

    results_path = assets_root / "interactive" / "interactive_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["ok_count"] == 2
    assert results["error_count"] == 0

    for entry in results["objects"]:
        output_dir = Path(entry["output_dir"])
        assert output_dir.is_dir()
        assert Path(entry["mesh_path"]).is_file()
        assert Path(entry["urdf_path"]).is_file()


def test_disallow_placeholder_urdf_blocks_mock(tmp_path, monkeypatch) -> None:
    assets_prefix = f"interactive-disallow-{tmp_path.name}"
    assets_root = build_scene_assets(assets_prefix, ["mug_0"])

    run_job(monkeypatch, assets_prefix, disallow_placeholder=True, mock_placeholder=True)

    complete_payload = json.loads((assets_root / ".interactive_complete").read_text(encoding="utf-8"))
    assert complete_payload["status"] == "failure"
    assert (assets_root / ".interactive_failed").is_file()

    results_path = assets_root / "interactive" / "interactive_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["error_count"] == 1
    assert "Placeholder URDF generation blocked" in results["objects"][0]["error"]
