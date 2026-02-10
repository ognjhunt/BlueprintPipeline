import json
from pathlib import Path

import tools.run_local_pipeline as run_local_pipeline
from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep


def _make_runner(tmp_path: Path) -> LocalPipelineRunner:
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("assets", "layout", "seg", "usd", "replicator"):
        (scene_dir / subdir).mkdir(parents=True, exist_ok=True)
    return LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="bedroom",
        enable_dwm=False,
        enable_dream2flow=False,
    )


def _write_minimal_manifest(assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene_id": "scene",
        "objects": [
            {
                "id": "obj_0",
                "physics": {"mass_kg": 0.0, "friction_static": 0.0},
                "transform": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "asset": {"path": "assets/obj_obj_0/asset.glb"},
            }
        ],
    }
    (assets_dir / "scene_manifest.json").write_text(json.dumps(payload))


def test_run_simready_converts_usdz_before_run(monkeypatch, tmp_path):
    import blueprint_sim.simready as simready_module

    runner = _make_runner(tmp_path)
    _write_minimal_manifest(runner.assets_dir)
    obj_dir = runner.assets_dir / "obj_obj_0"
    obj_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / "asset.glb").write_bytes(b"glb")

    call_order = []

    def fake_convert():
        call_order.append("convert")
        (obj_dir / "asset.usdz").write_bytes(b"usdz")

    def fake_run_from_env(root):
        assert root == runner.scene_dir
        assert (obj_dir / "asset.usdz").is_file()
        call_order.append("simready")
        (runner.assets_dir / ".simready_complete").write_text("ok")
        return 0

    monkeypatch.setattr(runner, "_convert_glb_assets_to_usdz", fake_convert)
    monkeypatch.setattr(simready_module, "run_from_env", fake_run_from_env)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("SIMREADY_PHYSICS_MODE", raising=False)

    result = runner._run_simready()

    assert result.success
    assert call_order == ["convert", "simready"]


def test_run_simready_fails_when_usdz_missing_after_conversion(monkeypatch, tmp_path):
    import blueprint_sim.simready as simready_module

    runner = _make_runner(tmp_path)
    _write_minimal_manifest(runner.assets_dir)
    obj_dir = runner.assets_dir / "obj_obj_0"
    obj_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / "asset.glb").write_bytes(b"glb")

    called = {"simready": False}

    def fake_run_from_env(root):
        called["simready"] = True
        return 0

    monkeypatch.setattr(runner, "_convert_glb_assets_to_usdz", lambda: None)
    monkeypatch.setattr(simready_module, "run_from_env", fake_run_from_env)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    result = runner._run_simready()

    assert not result.success
    assert "GLB→USDZ conversion incomplete before SimReady" in result.message
    assert called["simready"] is False


def test_run_simready_requires_gemini_key_in_gemini_mode(monkeypatch, tmp_path):
    import blueprint_sim.simready as simready_module

    runner = _make_runner(tmp_path)
    _write_minimal_manifest(runner.assets_dir)
    obj_dir = runner.assets_dir / "obj_obj_0"
    obj_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / "asset.glb").write_bytes(b"glb")
    (obj_dir / "asset.usdz").write_bytes(b"usdz")

    called = {"simready": False}

    def fake_run_from_env(root):
        called["simready"] = True
        return 0

    monkeypatch.setattr(runner, "_convert_glb_assets_to_usdz", lambda: None)
    monkeypatch.setattr(simready_module, "run_from_env", fake_run_from_env)
    monkeypatch.setattr(run_local_pipeline, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("SIMREADY_PHYSICS_MODE", "gemini")

    result = runner._run_simready()

    assert not result.success
    assert "requires GEMINI_API_KEY" in result.message
    assert called["simready"] is False


def test_generate_usda_prefers_simready_wrapper_and_skips_fallback_physics(tmp_path):
    runner = _make_runner(tmp_path)
    obj_dir = runner.assets_dir / "obj_obj_0"
    obj_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / "asset.usdz").write_text("usdz")
    (obj_dir / "simready.usda").write_text("#usda 1.0")

    manifest = {
        "objects": [
            {
                "id": "obj_0",
                "physics": {"mass": 123.0},
                "transform": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation_quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "asset": {"path": "assets/obj_obj_0/asset.glb"},
            }
        ]
    }

    usda = runner._generate_usda(manifest, layout={})
    assert "prepend references = @../assets/obj_obj_0/simready.usda@" in usda

    block_start = usda.index('def Xform "obj_obj_0"')
    block_end = usda.index("        }\n\n", block_start)
    obj_block = usda[block_start:block_end]
    assert "PhysicsRigidBodyAPI" not in obj_block
    assert "physics:mass" not in obj_block


def test_quality_gate_prefers_simready_physics_summary(tmp_path):
    runner = _make_runner(tmp_path)
    _write_minimal_manifest(runner.assets_dir)
    physics_summary = {
        "scene_id": runner.scene_id,
        "generated_at": "2026-02-10T00:00:00Z",
        "objects": [
            {
                "id": "obj_0",
                "mass_kg": 3.2,
                "static_friction": 0.7,
                "dynamic_friction": 0.6,
            }
        ],
    }
    (runner.assets_dir / "simready_physics.json").write_text(json.dumps(physics_summary))

    payload = runner._quality_gate_context_for_step(PipelineStep.SIMREADY)
    physics_props = payload["context"]["physics_properties"]

    assert physics_props["source"] == "simready_physics"
    assert physics_props["objects"][0]["id"] == "obj_0"
    assert physics_props["objects"][0]["mass"] == 3.2
    assert physics_props["objects"][0]["friction"] == 0.7
