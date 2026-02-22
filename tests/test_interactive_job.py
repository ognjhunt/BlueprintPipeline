from __future__ import annotations

import importlib.util
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from fixtures.generate_mock_stage1 import create_minimal_glb

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


def build_scene_assets(assets_prefix: str, object_ids: list[str], gcs_root: Path | None = None) -> Path:
    root = gcs_root if gcs_root is not None else Path("/mnt/gcs")
    root.mkdir(parents=True, exist_ok=True)
    assets_root = root / assets_prefix
    stage1_root = assets_root / "stage1"
    stage1_root.mkdir(parents=True, exist_ok=True)

    objects = []
    for obj_id in object_ids:
        obj_name = f"obj_{obj_id}"
        obj_dir = stage1_root / obj_name
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
                "asset": {"path": f"stage1/{obj_name}/obj_{obj_id}.glb"},
                "physics": {"mass": 0.2},
                "physics_hints": {"material_type": "ceramic"},
                "articulation": {"type": "revolute"},
                "semantics": {"affordances": ["Graspable", "Containable"]},
                "relationships": [],
            }
        )

    write_scene_manifest(assets_root, "interactive_scene", objects)
    return assets_root


def run_job(
    monkeypatch,
    assets_prefix: str,
    disallow_placeholder: bool,
    mock_placeholder: bool,
    gcs_root: Path | None = None,
    particulate_mode: str = "mock",
    articulation_backend: str = "auto",
    process_all: bool | None = None,
) -> None:
    monkeypatch.setenv("BUCKET", "test-bucket")
    monkeypatch.setenv("SCENE_ID", "interactive_scene")
    monkeypatch.setenv("ASSETS_PREFIX", assets_prefix)
    monkeypatch.setenv("STAGE1_PREFIX", f"{assets_prefix}/stage1")
    monkeypatch.setenv("INTERACTIVE_MODE", "glb")
    monkeypatch.setenv("PARTICULATE_MODE", particulate_mode)
    monkeypatch.setenv("ARTICULATION_BACKEND", articulation_backend)
    monkeypatch.setenv("DISALLOW_PLACEHOLDER_URDF", str(disallow_placeholder).lower())
    monkeypatch.setenv("PARTICULATE_MOCK_PLACEHOLDER", str(mock_placeholder).lower())
    monkeypatch.delenv("PARTICULATE_ENDPOINT", raising=False)
    monkeypatch.delenv("PARTICULATE_LOCAL_ENDPOINT", raising=False)
    monkeypatch.delenv("PARTICULATE_LOCAL_MODEL", raising=False)
    monkeypatch.delenv("APPROVED_PARTICULATE_MODELS", raising=False)
    monkeypatch.delenv("PRODUCTION_MODE", raising=False)
    monkeypatch.delenv("LABS_MODE", raising=False)
    monkeypatch.delenv("MULTIVIEW_PREFIX", raising=False)
    if process_all is None:
        monkeypatch.delenv("INTERACTIVE_PROCESS_ALL", raising=False)
    else:
        monkeypatch.setenv("INTERACTIVE_PROCESS_ALL", str(process_all).lower())

    if gcs_root is not None:
        _real_path = Path

        def _patched_path(*args, **kwargs):
            if args and args[0] == "/mnt/gcs":
                return gcs_root
            return _real_path(*args, **kwargs)

        monkeypatch.setattr(run_interactive_assets, "Path", _patched_path)

    run_interactive_assets.main()


def test_interactive_job_mock_glb_outputs(tmp_path, monkeypatch) -> None:
    assets_prefix = f"interactive-{tmp_path.name}"
    assets_root = build_scene_assets(assets_prefix, ["mug_0", "mug_1"], gcs_root=tmp_path)

    run_job(monkeypatch, assets_prefix, disallow_placeholder=False, mock_placeholder=False, gcs_root=tmp_path)

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
    assets_root = build_scene_assets(assets_prefix, ["mug_0"], gcs_root=tmp_path)

    run_job(monkeypatch, assets_prefix, disallow_placeholder=True, mock_placeholder=True, gcs_root=tmp_path)

    complete_payload = json.loads((assets_root / ".interactive_complete").read_text(encoding="utf-8"))
    assert complete_payload["status"] == "failure"
    assert (assets_root / ".interactive_failed").is_file()

    results_path = assets_root / "interactive" / "interactive_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["error_count"] == 1
    assert "Placeholder URDF generation blocked" in results["objects"][0]["error"]


def test_backend_dispatcher_resolves_particulate_heuristic_and_auto() -> None:
    resolve = run_interactive_assets.resolve_articulation_backend

    assert (
        resolve(
            run_interactive_assets.ARTICULATION_BACKEND_PARTICULATE,
            run_interactive_assets.PARTICULATE_MODE_REMOTE,
            "",
        )
        == run_interactive_assets.ARTICULATION_BACKEND_PARTICULATE
    )
    assert (
        resolve(
            run_interactive_assets.ARTICULATION_BACKEND_HEURISTIC,
            run_interactive_assets.PARTICULATE_MODE_LOCAL,
            "http://localhost:8080",
        )
        == run_interactive_assets.ARTICULATION_BACKEND_HEURISTIC
    )
    assert (
        resolve(
            run_interactive_assets.ARTICULATION_BACKEND_AUTO,
            run_interactive_assets.PARTICULATE_MODE_REMOTE,
            "https://particulate.example",
        )
        == run_interactive_assets.ARTICULATION_BACKEND_PARTICULATE
    )
    assert (
        resolve(
            run_interactive_assets.ARTICULATION_BACKEND_AUTO,
            run_interactive_assets.PARTICULATE_MODE_REMOTE,
            "",
        )
        == run_interactive_assets.ARTICULATION_BACKEND_HEURISTIC
    )
    assert (
        resolve(
            run_interactive_assets.ARTICULATION_BACKEND_AUTO,
            run_interactive_assets.PARTICULATE_MODE_LOCAL,
            "",
        )
        == run_interactive_assets.ARTICULATION_BACKEND_PARTICULATE
    )


def test_required_articulation_defaults_to_explicit_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = {
        "sim_role": "articulated_furniture",
        "articulation_required": False,
        "articulation": {"required": False},
    }
    monkeypatch.delenv("INTERACTIVE_REQUIRE_SIM_ROLE_DEFAULTS", raising=False)
    assert run_interactive_assets.is_required_articulation_object(obj) is False

    monkeypatch.setenv("INTERACTIVE_REQUIRE_SIM_ROLE_DEFAULTS", "true")
    assert run_interactive_assets.is_required_articulation_object(obj) is True


def test_candidate_articulation_explicit_flag_overrides_sim_role(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = {
        "sim_role": "articulated_furniture",
        "articulation_candidate": False,
        "articulation": {"candidate": False, "required": False},
    }
    monkeypatch.delenv("INTERACTIVE_REQUIRE_SIM_ROLE_DEFAULTS", raising=False)
    monkeypatch.setenv("INTERACTIVE_CANDIDATE_FROM_SIM_ROLE", "true")
    assert run_interactive_assets.is_articulation_candidate_object(obj) is False


def test_required_articulation_object_fails_closed_on_non_articulated_output(tmp_path, monkeypatch) -> None:
    assets_prefix = f"interactive-required-{tmp_path.name}"
    assets_root = build_scene_assets(assets_prefix, ["desk_0"], gcs_root=tmp_path)

    manifest_path = assets_root / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objects"][0]["category"] = "desk with drawers"
    manifest["objects"][0]["articulation"] = {"required": True, "type": "prismatic"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        run_job(
            monkeypatch,
            assets_prefix,
            disallow_placeholder=False,
            mock_placeholder=False,
            gcs_root=tmp_path,
            particulate_mode="mock",
            articulation_backend="particulate",
        )

    assert exit_info.value.code == 1
    assert (assets_root / ".interactive_failed").is_file()

    complete_payload = json.loads((assets_root / ".interactive_complete").read_text(encoding="utf-8"))
    assert complete_payload["status"] == "failure"
    assert complete_payload["summary"]["required_failures"]

    failed_payload = json.loads((assets_root / ".interactive_failed").read_text(encoding="utf-8"))
    assert failed_payload["reason"] == "required_articulation_unmet"


def test_candidate_filter_skips_non_articulation_objects_without_override(tmp_path, monkeypatch) -> None:
    assets_prefix = f"interactive-filter-{tmp_path.name}"
    assets_root = build_scene_assets(assets_prefix, ["mug_0"], gcs_root=tmp_path)

    manifest_path = assets_root / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objects"][0].pop("articulation", None)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_job(
        monkeypatch,
        assets_prefix,
        disallow_placeholder=False,
        mock_placeholder=False,
        gcs_root=tmp_path,
    )

    complete_payload = json.loads((assets_root / ".interactive_complete").read_text(encoding="utf-8"))
    assert complete_payload["summary"]["total_objects"] == 0

    summary_payload = json.loads((assets_root / ".interactive_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["candidate_interactive_count"] == 0
    assert summary_payload["interactive_objects_in_manifest"] == 1


def test_candidate_filter_process_all_override(tmp_path, monkeypatch) -> None:
    assets_prefix = f"interactive-filter-all-{tmp_path.name}"
    assets_root = build_scene_assets(assets_prefix, ["mug_0"], gcs_root=tmp_path)

    manifest_path = assets_root / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objects"][0].pop("articulation", None)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_job(
        monkeypatch,
        assets_prefix,
        disallow_placeholder=False,
        mock_placeholder=False,
        gcs_root=tmp_path,
        process_all=True,
    )

    complete_payload = json.loads((assets_root / ".interactive_complete").read_text(encoding="utf-8"))
    assert complete_payload["summary"]["total_objects"] == 1

    summary_payload = json.loads((assets_root / ".interactive_summary.json").read_text(encoding="utf-8"))
    assert summary_payload["candidate_interactive_count"] == 1
    assert summary_payload["process_all_interactive"] is True


def test_generate_multiview_scaffold_writes_synthetic_views(tmp_path, monkeypatch) -> None:
    glb_path = tmp_path / "obj_drawer.glb"
    glb_path.write_bytes(create_minimal_glb())
    output_dir = tmp_path / "interactive" / "obj_drawer"

    class _FakeImageClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def generate_image(self, prompt: str, size: str) -> SimpleNamespace:
            self.calls.append((prompt, size))
            return SimpleNamespace(images=[b"fake-image-bytes"])

    fake_client = _FakeImageClient()
    llm_client_module = __import__("tools.llm_client.client", fromlist=["create_llm_client"])
    monkeypatch.setattr(
        llm_client_module,
        "create_llm_client",
        lambda provider, model, fallback_enabled: fake_client,
    )

    metadata = run_interactive_assets.generate_multiview_scaffold(
        obj_name="obj_drawer",
        obj_class="desk with drawers",
        glb_path=glb_path,
        output_dir=output_dir,
        view_count=3,
        model="gemini-3-pro-image-preview",
    )

    assert metadata["status"] == "success"
    assert metadata["generated_count"] == 3
    assert len(fake_client.calls) == 3

    scaffold_metadata_path = output_dir / "multiview_scaffold.json"
    assert scaffold_metadata_path.is_file()
    scaffold_metadata = json.loads(scaffold_metadata_path.read_text(encoding="utf-8"))
    assert scaffold_metadata["model"] == "gemini-3-pro-image-preview"
    assert scaffold_metadata["generated_count"] == 3
    assert len(scaffold_metadata["generated_files"]) == 3

    scaffold_dir = output_dir / "multiview_synth"
    for file_name in scaffold_metadata["generated_files"]:
        assert (scaffold_dir / file_name).is_file()
