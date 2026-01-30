from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from fixtures.generate_mock_regen3d import generate_mock_regen3d
from tools.geniesim_adapter import GenieSimExportResult


def _install_fake_google_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    google_module = sys.modules.get("google") or ModuleType("google")
    cloud_module = sys.modules.get("google.cloud") or ModuleType("google.cloud")
    if not hasattr(google_module, "__path__"):
        google_module.__path__ = []
    if not hasattr(cloud_module, "__path__"):
        cloud_module.__path__ = []

    storage_module = ModuleType("google.cloud.storage")
    storage_module.Client = lambda *args, **kwargs: None
    import importlib
    firestore_module = ModuleType("google.cloud.firestore")
    firestore_module.Client = lambda *args, **kwargs: None
    firestore_module.__spec__ = importlib.machinery.ModuleSpec("google.cloud.firestore", None)

    cloud_module.storage = storage_module
    cloud_module.firestore = firestore_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)
    monkeypatch.setitem(sys.modules, "google.cloud.firestore", firestore_module)


def _write_marker(marker_path: Path, payload: dict) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(payload, indent=2))


class DummyQualityGateRegistry:
    def __init__(self, *args, **kwargs) -> None:
        self.verbose = kwargs.get("verbose")

    def run_checkpoint(self, *args, **kwargs) -> None:
        return None

    def save_report(self, *args, **kwargs) -> None:
        return None

    def can_proceed(self) -> bool:
        return True


class DummyMetrics:
    def __init__(self) -> None:
        self.backend = SimpleNamespace(value="local")

    @contextmanager
    def track_job(self, *args, **kwargs):
        yield

    def get_stats(self) -> dict:
        return {}


def test_pipeline_data_flow_entrypoints(
    tmp_path: Path,
    load_job_module,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_google_cloud(monkeypatch)

    scene_id = "pipeline-scene"
    regen3d_root = generate_mock_regen3d(tmp_path, scene_id, environment_type="kitchen")

    regen3d_job = load_job_module("regen3d", "regen3d_adapter_job.py")
    monkeypatch.setattr(regen3d_job, "GCS_ROOT", tmp_path)
    monkeypatch.setattr(regen3d_job, "get_metrics", lambda: DummyMetrics())
    monkeypatch.setenv("BUCKET", "local-bucket")
    monkeypatch.setenv("SCENE_ID", scene_id)
    monkeypatch.setenv("REGEN3D_PREFIX", f"scenes/{scene_id}/regen3d")
    monkeypatch.setenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    monkeypatch.setenv("LAYOUT_PREFIX", f"scenes/{scene_id}/layout")
    monkeypatch.setenv("ENVIRONMENT_TYPE", "kitchen")
    monkeypatch.setenv("SCALE_FACTOR", "1.0")
    monkeypatch.setenv("BYPASS_QUALITY_GATES", "true")

    real_regen_run = regen3d_job.run_regen3d_adapter_job
    call_state = {"count": 0}

    def run_once(*args, **kwargs):
        call_state["count"] += 1
        if call_state["count"] > 1:
            return 0
        return real_regen_run(*args, **kwargs)

    monkeypatch.setattr(regen3d_job, "run_regen3d_adapter_job", run_once)

    assert regen3d_root.exists()
    assert regen3d_job.main() == 0

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

    assert manifest["scene_id"] == scene_id
    assert manifest["objects"]
    assert layout["scene_id"] == scene_id
    assert layout["objects"]
    assert inventory["objects"]

    simready_job = load_job_module("simready", "prepare_simready_assets.py")
    monkeypatch.setattr(simready_job, "GCS_ROOT", tmp_path)

    # Install stub modules so blueprint_sim.assembly / usd-assembly-job can import
    if "pxr" not in sys.modules:
        pxr_stub = ModuleType("pxr")
        for attr in ("Sdf", "Usd", "UsdGeom", "UsdPhysics", "UsdShade", "UsdUtils", "Gf", "Vt"):
            setattr(pxr_stub, attr, SimpleNamespace())
        monkeypatch.setitem(sys.modules, "pxr", pxr_stub)
    if "pygltflib" not in sys.modules:
        pygltflib_stub = ModuleType("pygltflib")
        pygltflib_stub.GLTF2 = type("GLTF2", (), {})
        monkeypatch.setitem(sys.modules, "pygltflib", pygltflib_stub)

    import blueprint_sim.simready as simready_runner

    def fake_run_from_env(root: Path) -> int:
        simready_marker = root / f"scenes/{scene_id}/assets/.simready_complete"
        _write_marker(simready_marker, {"status": "completed", "scene_id": scene_id})
        return 0

    monkeypatch.setattr(simready_runner, "run_from_env", fake_run_from_env)

    with pytest.raises(SystemExit) as simready_exit:
        simready_job.main()
    assert simready_exit.value.code == 0

    simready_marker = assets_dir / ".simready_complete"
    assert simready_marker.exists()
    simready_payload = json.loads(simready_marker.read_text())
    assert simready_payload["status"] == "completed"

    usd_job = load_job_module("usd", "assemble_scene.py")
    monkeypatch.setattr(usd_job, "GCS_ROOT", tmp_path)
    monkeypatch.setattr(usd_job, "QualityGateRegistry", DummyQualityGateRegistry)
    monkeypatch.setattr(usd_job, "get_metrics", lambda: DummyMetrics())

    usd_prefix = f"scenes/{scene_id}/usd"
    monkeypatch.setenv("USD_PREFIX", usd_prefix)

    def fake_assemble_from_env() -> int:
        usd_path = tmp_path / usd_prefix / "scene.usda"
        usd_path.parent.mkdir(parents=True, exist_ok=True)
        usd_path.write_text("#usda 1.0")
        _write_marker(assets_dir / ".usd_assembly_complete", {"status": "completed"})
        return 0

    monkeypatch.setattr(usd_job, "assemble_from_env", fake_assemble_from_env)

    with pytest.raises(SystemExit) as usd_exit:
        usd_job.main()
    assert usd_exit.value.code == 0

    usd_manifest_path = tmp_path / usd_prefix / "usd_assembly_manifest.json"
    assert usd_manifest_path.exists()
    usd_manifest = json.loads(usd_manifest_path.read_text())
    assert usd_manifest["scene_id"] == scene_id
    assert usd_manifest["usd_prefix"] == usd_prefix

    usd_marker = assets_dir / ".usd_assembly_complete"
    assert usd_marker.exists()

    replicator_job = load_job_module("replicator", "generate_replicator_bundle.py")
    monkeypatch.setattr(replicator_job, "GCS_ROOT", tmp_path)
    monkeypatch.setenv("SEG_PREFIX", f"scenes/{scene_id}/seg")
    monkeypatch.setenv("REPLICATOR_PREFIX", f"scenes/{scene_id}/replicator")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv(
        "LLM_MOCK_RESPONSE_PATH",
        str(repo_root / "tests" / "fixtures" / "replicator" / "mock_analysis.json"),
    )

    assert replicator_job.main() == 0

    replicator_dir = tmp_path / f"scenes/{scene_id}/replicator"
    variation_manifest = replicator_dir / "variation_assets" / "manifest.json"
    assert variation_manifest.exists()
    variation_payload = json.loads(variation_manifest.read_text())
    assert "assets" in variation_payload

    _write_marker(replicator_dir / ".replicator_complete", {"status": "completed"})

    geniesim_job = load_job_module("geniesim_export", "export_to_geniesim.py")
    monkeypatch.setattr(geniesim_job, "HAVE_QUALITY_GATES", False)
    monkeypatch.setattr(geniesim_job, "PREMIUM_ANALYTICS_AVAILABLE", False)
    monkeypatch.setattr(geniesim_job, "SIM2REAL_AVAILABLE", False)
    monkeypatch.setattr(geniesim_job, "EMBODIMENT_TRANSFER_AVAILABLE", False)
    monkeypatch.setattr(geniesim_job, "TRAJECTORY_OPTIMALITY_AVAILABLE", False)
    monkeypatch.setattr(geniesim_job, "POLICY_LEADERBOARD_AVAILABLE", False)
    monkeypatch.setattr(geniesim_job, "TACTILE_SENSOR_AVAILABLE", False)
    monkeypatch.setattr(geniesim_job, "LANGUAGE_ANNOTATIONS_AVAILABLE", False)
    monkeypatch.setattr(geniesim_job, "GENERALIZATION_ANALYZER_AVAILABLE", False)
    monkeypatch.setattr(geniesim_job, "SIM2REAL_VALIDATION_AVAILABLE", False)
    monkeypatch.setattr(geniesim_job, "AUDIO_NARRATION_AVAILABLE", False)
    monkeypatch.setattr(geniesim_job, "get_metrics", lambda: DummyMetrics())

    def fake_generate_asset_provenance(
        scene_dir: Path,
        output_path: Path,
        scene_id: str,
        manifest_path: Path | None = None,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"scene_id": scene_id, "assets": []}, indent=2))

    monkeypatch.setattr(geniesim_job, "generate_asset_provenance", fake_generate_asset_provenance)

    class FakeGenieSimExporter:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def export(self, manifest_path: Path, output_dir: Path, usd_source_dir: Path | None = None):
            output_dir.mkdir(parents=True, exist_ok=True)
            scene_graph = output_dir / "scene_graph.json"
            asset_index = output_dir / "asset_index.json"
            task_config = output_dir / "task_config.json"
            scene_graph.write_text(json.dumps({"scene_id": scene_id, "nodes": [], "edges": []}, indent=2))
            asset_index.write_text(json.dumps({"assets": []}, indent=2))
            task_config.write_text(json.dumps({"tasks": []}, indent=2))
            return GenieSimExportResult(
                success=True,
                scene_id=scene_id,
                output_dir=output_dir,
                scene_graph_path=scene_graph,
                asset_index_path=asset_index,
                task_config_path=task_config,
                num_nodes=0,
                num_edges=0,
                num_assets=0,
                num_tasks=0,
            )

    monkeypatch.setattr(geniesim_job, "GenieSimExporter", FakeGenieSimExporter)

    def patched_path(*args, **kwargs):
        if args and args[0] == "/mnt/gcs":
            return tmp_path
        return Path(*args, **kwargs)

    monkeypatch.setattr(geniesim_job, "Path", patched_path)

    geniesim_prefix = f"scenes/{scene_id}/geniesim"
    monkeypatch.setenv("GENIESIM_PREFIX", geniesim_prefix)
    monkeypatch.setenv("REPLICATOR_PREFIX", f"scenes/{scene_id}/replicator")
    monkeypatch.setenv("VARIATION_ASSETS_PREFIX", f"scenes/{scene_id}/replicator/variation_assets")
    monkeypatch.setenv("FILTER_COMMERCIAL", "false")
    monkeypatch.setenv("ENABLE_MULTI_ROBOT", "false")
    monkeypatch.setenv("ENABLE_BIMANUAL", "false")
    monkeypatch.setenv("ENABLE_VLA_PACKAGES", "false")
    monkeypatch.setenv("ENABLE_RICH_ANNOTATIONS", "false")
    monkeypatch.setenv("ENABLE_PREMIUM_ANALYTICS", "false")
    monkeypatch.setenv("REQUIRE_QUALITY_GATES", "false")
    monkeypatch.setenv("COPY_USD", "false")

    with pytest.raises(SystemExit) as geniesim_exit:
        geniesim_job.main()
    assert geniesim_exit.value.code == 0

    geniesim_dir = tmp_path / geniesim_prefix
    merged_manifest = geniesim_dir / "merged_scene_manifest.json"
    assert merged_manifest.exists()
    merged_payload = json.loads(merged_manifest.read_text())
    assert merged_payload["scene_id"] == scene_id
    assert merged_payload["objects"]

    for expected_file in ["scene_graph.json", "asset_index.json", "task_config.json"]:
        assert (geniesim_dir / expected_file).exists()

    assert (geniesim_dir / "legal" / "asset_provenance.json").exists()

    assert (assets_dir / ".simready_complete").exists()
    assert (assets_dir / ".usd_assembly_complete").exists()
    assert (replicator_dir / ".replicator_complete").exists()
