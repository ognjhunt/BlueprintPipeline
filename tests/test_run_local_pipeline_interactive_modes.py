from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep


def _write_required_manifest(assets_dir: Path) -> None:
    payload = {
        "version": "1.0.0",
        "scene_id": "scene",
        "scene": {
            "coordinate_frame": "y_up",
            "meters_per_unit": 1.0,
        },
        "objects": [
            {
                "id": "obj_1",
                "category": "desk with drawers",
                "sim_role": "articulated_furniture",
                "asset": {"path": "assets/obj_1/asset.glb"},
                "transform": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
                },
                "articulation": {"required": True, "type": "prismatic"},
            }
        ],
    }
    (assets_dir / "scene_manifest.json").write_text(json.dumps(payload))


def _make_runner(tmp_path: Path) -> LocalPipelineRunner:
    scene_dir = tmp_path / "scene"
    assets_dir = scene_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    _write_required_manifest(assets_dir)
    return LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=False,
        skip_interactive=False,
        environment_type="bedroom",
    )


def test_run_interactive_allows_local_mode_without_remote_endpoint(tmp_path, monkeypatch) -> None:
    runner = _make_runner(tmp_path)
    assets_dir = runner.assets_dir

    monkeypatch.delenv("PARTICULATE_ENDPOINT", raising=False)
    monkeypatch.delenv("PARTICULATE_LOCAL_ENDPOINT", raising=False)
    monkeypatch.setenv("PARTICULATE_MODE", "local")
    monkeypatch.setenv("ARTICULATION_BACKEND", "particulate")

    def _fake_run(cmd, cwd, env, check):
        assert env["PARTICULATE_MODE"] == "local"
        assert env["PARTICULATE_ENDPOINT"] == "http://localhost:8080"
        results_dir = assets_dir / "interactive"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "interactive_results.json").write_text(
            json.dumps(
                {
                    "objects": [{"id": "obj_1", "is_articulated": True}],
                    "articulated_count": 1,
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("tools.run_local_pipeline.subprocess.run", _fake_run)

    result = runner._run_interactive()
    assert result.success is True
    assert result.outputs["articulated_count"] == 1


def test_run_interactive_rejects_required_articulation_with_mock_mode(tmp_path, monkeypatch) -> None:
    runner = _make_runner(tmp_path)
    monkeypatch.setenv("PARTICULATE_MODE", "mock")
    monkeypatch.setenv("ARTICULATION_BACKEND", "particulate")
    monkeypatch.delenv("PARTICULATE_ENDPOINT", raising=False)

    result = runner._run_interactive()
    assert result.success is False
    assert "PARTICULATE_MODE=mock" in result.message


def test_preflight_rejects_required_articulation_with_heuristic_backend(tmp_path, monkeypatch) -> None:
    runner = _make_runner(tmp_path)
    monkeypatch.setenv("ARTICULATION_BACKEND", "heuristic")
    monkeypatch.setenv("PARTICULATE_MODE", "local")

    ok = runner._preflight_articulation_requirements([PipelineStep.INTERACTIVE])
    assert ok is False
