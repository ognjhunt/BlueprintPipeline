from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_official_scenesmith_adapter_converts_house_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_module("scenesmith_paper_command_module", "scenesmith-service/scenesmith_paper_command.py")

    repo_dir = tmp_path / "scenesmith"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "main.py").write_text("# placeholder\n", encoding="utf-8")

    run_dir = tmp_path / "paper-run"
    monkeypatch.setattr(module, "_run_root", lambda _scene_id: run_dir)

    def _fake_process(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        output_dir = run_dir / "outputs" / "scene_demo_001" / "scene_000" / "combined_house"
        output_dir.mkdir(parents=True, exist_ok=True)
        house_state = {
            "objects": [
                {
                    "id": "table_001",
                    "semantic_class": "table",
                    "extent": {"x": 1.2, "y": 0.8, "z": 0.7},
                    "pose": {
                        "position": {"x": 0.3, "y": 0.0, "z": -0.2},
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                    "floor_object": True,
                },
                {
                    "id": "mug_001",
                    "semantic_class": "mug",
                    "extent": {"x": 0.08, "y": 0.1, "z": 0.08},
                    "pose": {
                        "position": {"x": 0.1, "y": 0.82, "z": 0.05},
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                },
            ]
        }
        house_state_path = output_dir / "house_state.json"
        house_state_path.write_text(json.dumps(house_state), encoding="utf-8")

        return {
            "returncode": 0,
            "stdout_tail": "ok",
            "stderr_tail": "",
            "stdout_log": str(run_dir / "stdout.log"),
            "stderr_log": str(run_dir / "stderr.log"),
            "house_state_path": str(house_state_path),
            "forced_exit_reason": "",
            "timed_out": False,
        }

    monkeypatch.setattr(module, "_run_scenesmith_process", _fake_process)
    monkeypatch.setenv("SCENESMITH_PAPER_REPO_DIR", str(repo_dir))
    monkeypatch.setenv("SCENESMITH_PAPER_PYTHON_BIN", "python3")
    monkeypatch.setenv("SCENESMITH_PAPER_KEEP_RUN_DIR", "true")

    response = module._run_official_scenesmith(
        {
            "scene_id": "scene_demo_001",
            "prompt": "A table with a mug",
            "seed": 3,
            "constraints": {"room_type": "kitchen"},
        }
    )

    assert response["schema_version"] == "v1"
    assert response["room_type"] == "kitchen"
    assert len(response["objects"]) == 2

    first = response["objects"][0]
    second = response["objects"][1]

    assert first["id"] == "table_001"
    assert first["sim_role"] == "static"
    assert second["id"] == "mug_001"
    assert second["sim_role"] == "manipulable_object"


@pytest.mark.unit
def test_official_scenesmith_adapter_requires_repo_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("scenesmith_paper_command_missing_repo_module", "scenesmith-service/scenesmith_paper_command.py")
    monkeypatch.delenv("SCENESMITH_PAPER_REPO_DIR", raising=False)

    with pytest.raises(RuntimeError, match="SCENESMITH_PAPER_REPO_DIR is required"):
        module._run_official_scenesmith(
            {
                "scene_id": "scene_demo_002",
                "prompt": "A kitchen",
                "seed": 1,
            }
        )


@pytest.mark.unit
def test_official_scenesmith_adapter_marks_generated_cabinet_articulation_optional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module("scenesmith_paper_command_optional_articulation_module", "scenesmith-service/scenesmith_paper_command.py")

    repo_dir = tmp_path / "scenesmith"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "main.py").write_text("# placeholder\n", encoding="utf-8")

    run_dir = tmp_path / "paper-run"
    monkeypatch.setattr(module, "_run_root", lambda _scene_id: run_dir)

    def _fake_process(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        output_dir = run_dir / "outputs" / "scene_demo_005" / "scene_000" / "combined_house"
        output_dir.mkdir(parents=True, exist_ok=True)
        house_state = {
            "objects": [
                {
                    "id": "cabinet_001",
                    "semantic_class": "cabinet",
                    "extent": {"x": 0.8, "y": 0.9, "z": 0.5},
                    "pose": {
                        "position": {"x": -1.0, "y": 0.0, "z": 1.2},
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                    "floor_object": True,
                }
            ]
        }
        house_state_path = output_dir / "house_state.json"
        house_state_path.write_text(json.dumps(house_state), encoding="utf-8")

        return {
            "returncode": 0,
            "stdout_tail": "ok",
            "stderr_tail": "",
            "stdout_log": str(run_dir / "stdout.log"),
            "stderr_log": str(run_dir / "stderr.log"),
            "house_state_path": str(house_state_path),
            "forced_exit_reason": "",
            "timed_out": False,
        }

    monkeypatch.setattr(module, "_run_scenesmith_process", _fake_process)
    monkeypatch.setenv("SCENESMITH_PAPER_REPO_DIR", str(repo_dir))
    monkeypatch.setenv("SCENESMITH_PAPER_PYTHON_BIN", "python3")
    monkeypatch.setenv("SCENESMITH_PAPER_KEEP_RUN_DIR", "true")
    monkeypatch.setenv("SCENESMITH_PAPER_ALL_SAM3D", "true")

    response = module._run_official_scenesmith(
        {
            "scene_id": "scene_demo_005",
            "prompt": "A kitchen with cabinets",
        }
    )

    assert len(response["objects"]) == 1
    obj = response["objects"][0]
    assert obj["sim_role"] == "articulated_furniture"
    assert obj["articulation"]["candidate"] is True
    assert obj["articulation"]["required"] is False
    assert obj["articulation"]["backend_hint"] == "particulate_optional"
    assert obj["articulation"]["requirement_source"] == "force_generated_assets"


@pytest.mark.unit
def test_official_scenesmith_adapter_preserves_required_joint_articulation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module("scenesmith_paper_command_joint_articulation_module", "scenesmith-service/scenesmith_paper_command.py")

    repo_dir = tmp_path / "scenesmith"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "main.py").write_text("# placeholder\n", encoding="utf-8")

    run_dir = tmp_path / "paper-run"
    monkeypatch.setattr(module, "_run_root", lambda _scene_id: run_dir)

    def _fake_process(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        output_dir = run_dir / "outputs" / "scene_demo_006" / "scene_000" / "combined_house"
        output_dir.mkdir(parents=True, exist_ok=True)
        house_state = {
            "objects": [
                {
                    "id": "door_001",
                    "semantic_class": "cabinet",
                    "joint_type": "revolute",
                    "extent": {"x": 0.6, "y": 0.8, "z": 0.4},
                    "pose": {
                        "position": {"x": 0.0, "y": 0.0, "z": 0.5},
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                }
            ]
        }
        house_state_path = output_dir / "house_state.json"
        house_state_path.write_text(json.dumps(house_state), encoding="utf-8")

        return {
            "returncode": 0,
            "stdout_tail": "ok",
            "stderr_tail": "",
            "stdout_log": str(run_dir / "stdout.log"),
            "stderr_log": str(run_dir / "stderr.log"),
            "house_state_path": str(house_state_path),
            "forced_exit_reason": "",
            "timed_out": False,
        }

    monkeypatch.setattr(module, "_run_scenesmith_process", _fake_process)
    monkeypatch.setenv("SCENESMITH_PAPER_REPO_DIR", str(repo_dir))
    monkeypatch.setenv("SCENESMITH_PAPER_PYTHON_BIN", "python3")
    monkeypatch.setenv("SCENESMITH_PAPER_KEEP_RUN_DIR", "true")
    monkeypatch.setenv("SCENESMITH_PAPER_ALL_SAM3D", "true")

    response = module._run_official_scenesmith(
        {
            "scene_id": "scene_demo_006",
            "prompt": "A kitchen cabinet with opening doors",
        }
    )

    assert len(response["objects"]) == 1
    obj = response["objects"][0]
    assert obj["articulation"]["candidate"] is True
    assert obj["articulation"]["required"] is True
    assert obj["articulation"]["backend_hint"] == "particulate_first"
    assert obj["articulation"]["requirement_source"] == "joint_type"


@pytest.mark.unit
def test_hydra_overrides_all_sam3d_and_gemini_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_module("scenesmith_paper_command_overrides_module", "scenesmith-service/scenesmith_paper_command.py")

    monkeypatch.setenv("SCENESMITH_PAPER_ALL_SAM3D", "true")
    monkeypatch.setenv("SCENESMITH_PAPER_ENABLE_GEMINI_CONTEXT_IMAGE", "true")
    monkeypatch.setenv("SCENESMITH_PAPER_EXTRA_OVERRIDES", '["experiment.pipeline.stop_stage=furniture"]')

    overrides = module._hydra_overrides(
        payload={
            "prompt": "A modern kitchen",
            "pipeline_stages": ["floor_plan", "furniture"],
        },
        run_dir=tmp_path,
        scene_name="scene_demo_003",
    )

    assert "experiment.pipeline.start_stage=floor_plan" in overrides
    assert "experiment.pipeline.stop_stage=furniture" in overrides
    assert "furniture_agent.context_image_generation.enabled=true" in overrides
    assert "furniture_agent.asset_manager.image_generation.backend=gemini" in overrides

    for prefix in module._PLACEMENT_AGENT_CONFIG_PREFIXES:
        assert f"{prefix}.asset_manager.general_asset_source=generated" in overrides
        assert f"{prefix}.asset_manager.backend=sam3d" in overrides
        assert f"{prefix}.asset_manager.router.strategies.articulated.enabled=false" in overrides
        assert f"{prefix}.asset_manager.articulated.sources.artvip.enabled=false" in overrides


@pytest.mark.unit
def test_hydra_overrides_supports_delimited_extra_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module("scenesmith_paper_command_extra_overrides_module", "scenesmith-service/scenesmith_paper_command.py")

    monkeypatch.setenv(
        "SCENESMITH_PAPER_EXTRA_OVERRIDES",
        "furniture_agent.context_image_generation.enabled=true;;experiment.num_workers=2",
    )

    overrides = module._hydra_overrides(
        payload={"prompt": "A compact kitchen"},
        run_dir=tmp_path,
        scene_name="scene_demo_004",
    )

    assert "furniture_agent.context_image_generation.enabled=true" in overrides
    assert "experiment.num_workers=2" in overrides


@pytest.mark.unit
def test_official_scenesmith_model_chain_retries_until_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module("scenesmith_paper_command_model_chain_module", "scenesmith-service/scenesmith_paper_command.py")

    repo_dir = tmp_path / "scenesmith"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "main.py").write_text("# placeholder\n", encoding="utf-8")

    run_root = tmp_path / "paper-run-chain"
    monkeypatch.setattr(module, "_run_root", lambda _scene_id: run_root)

    call_count = {"value": 0}

    def _fake_process(*, cmd, **kwargs):  # type: ignore[no-untyped-def]
        call_count["value"] += 1
        del kwargs
        run_dir = None
        for token in cmd:
            if isinstance(token, str) and token.startswith("hydra.run.dir="):
                run_dir = Path(token.split("=", 1)[1])
                break
        if run_dir is None:
            raise RuntimeError("hydra.run.dir override missing in command")

        if call_count["value"] == 1:
            return {
                "returncode": 1,
                "stdout_tail": "first attempt failed",
                "stderr_tail": "failure",
                "stdout_log": str(run_dir / "stdout.log"),
                "stderr_log": str(run_dir / "stderr.log"),
                "house_state_path": "",
                "forced_exit_reason": "",
                "timed_out": False,
            }

        output_dir = run_dir / "outputs" / "scene_demo_chain" / "scene_000" / "combined_house"
        output_dir.mkdir(parents=True, exist_ok=True)
        house_state = {
            "objects": [
                {
                    "id": "table_001",
                    "semantic_class": "table",
                    "extent": {"x": 1.2, "y": 0.8, "z": 0.7},
                    "pose": {
                        "position": {"x": 0.3, "y": 0.0, "z": -0.2},
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                    "floor_object": True,
                }
            ]
        }
        house_state_path = output_dir / "house_state.json"
        house_state_path.write_text(json.dumps(house_state), encoding="utf-8")

        return {
            "returncode": 0,
            "stdout_tail": "ok",
            "stderr_tail": "",
            "stdout_log": str(run_dir / "stdout.log"),
            "stderr_log": str(run_dir / "stderr.log"),
            "house_state_path": str(house_state_path),
            "forced_exit_reason": "",
            "timed_out": False,
        }

    monkeypatch.setattr(module, "_run_scenesmith_process", _fake_process)
    monkeypatch.setenv("SCENESMITH_PAPER_REPO_DIR", str(repo_dir))
    monkeypatch.setenv("SCENESMITH_PAPER_PYTHON_BIN", "python3")
    monkeypatch.setenv("SCENESMITH_PAPER_KEEP_RUN_DIR", "true")
    monkeypatch.delenv("SCENESMITH_PAPER_MODEL", raising=False)
    monkeypatch.setenv("SCENESMITH_PAPER_MODEL_CHAIN", "model-a,model-b")

    response = module._run_official_scenesmith(
        {
            "scene_id": "scene_demo_chain",
            "prompt": "A table in a room",
        }
    )

    assert response["paper_stack"]["model_selected"] == "model-b"
    attempts = response["paper_stack"]["model_attempts"]
    assert len(attempts) == 2
    assert attempts[0]["model"] == "model-a"
    assert attempts[0]["status"] == "failed"
    assert attempts[1]["model"] == "model-b"
    assert attempts[1]["status"] == "succeeded"


@pytest.mark.unit
def test_official_scenesmith_accepts_nonzero_exit_with_house_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module("scenesmith_paper_command_nonzero_resume_module", "scenesmith-service/scenesmith_paper_command.py")

    repo_dir = tmp_path / "scenesmith"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "main.py").write_text("# placeholder\n", encoding="utf-8")

    run_dir = tmp_path / "paper-run-nonzero"
    monkeypatch.setattr(module, "_run_root", lambda _scene_id: run_dir)

    def _fake_process(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        output_dir = run_dir / "outputs" / "scene_demo_nonzero" / "scene_000" / "combined_house"
        output_dir.mkdir(parents=True, exist_ok=True)
        house_state = {
            "objects": [
                {
                    "id": "table_001",
                    "semantic_class": "table",
                    "extent": {"x": 1.2, "y": 0.8, "z": 0.7},
                    "pose": {
                        "position": {"x": 0.3, "y": 0.0, "z": -0.2},
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                    "floor_object": True,
                }
            ]
        }
        house_state_path = output_dir / "house_state.json"
        house_state_path.write_text(json.dumps(house_state), encoding="utf-8")
        return {
            "returncode": 1,
            "stdout_tail": "Press Ctrl+C to exit",
            "stderr_tail": "",
            "stdout_log": str(run_dir / "stdout.log"),
            "stderr_log": str(run_dir / "stderr.log"),
            "house_state_path": str(house_state_path),
            "forced_exit_reason": "exit_prompt_after_house_state",
            "timed_out": False,
        }

    monkeypatch.setattr(module, "_run_scenesmith_process", _fake_process)
    monkeypatch.setenv("SCENESMITH_PAPER_REPO_DIR", str(repo_dir))
    monkeypatch.setenv("SCENESMITH_PAPER_PYTHON_BIN", "python3")
    monkeypatch.setenv("SCENESMITH_PAPER_KEEP_RUN_DIR", "true")

    response = module._run_official_scenesmith(
        {
            "scene_id": "scene_demo_nonzero",
            "prompt": "A room with a table",
        }
    )

    assert len(response["objects"]) == 1
    assert response["paper_stack"]["scenesmith_nonzero_exit_accepted"] is True
    assert response["paper_stack"]["scenesmith_exit_code"] == 1
    attempts = response["paper_stack"]["model_attempts"]
    assert attempts[0]["status"] == "succeeded_with_nonzero_exit"


@pytest.mark.unit
def test_official_scenesmith_reuses_existing_run_dir_without_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module("scenesmith_paper_command_existing_run_module", "scenesmith-service/scenesmith_paper_command.py")

    repo_dir = tmp_path / "scenesmith"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "main.py").write_text("# placeholder\n", encoding="utf-8")

    existing_run_dir = tmp_path / "existing-run"
    output_dir = existing_run_dir / "scene_demo_resume" / "scene_000" / "combined_house"
    output_dir.mkdir(parents=True, exist_ok=True)
    house_state = {
        "rooms": {
            "kitchen": {
                "objects": [
                    {
                        "id": "table_001",
                        "semantic_class": "table",
                        "extent": {"x": 1.2, "y": 0.8, "z": 0.7},
                        "pose": {
                            "position": {"x": 0.3, "y": 0.0, "z": -0.2},
                            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                        },
                        "floor_object": True,
                    }
                ]
            }
        }
    }
    (output_dir / "house_state.json").write_text(json.dumps(house_state), encoding="utf-8")

    def _unexpected_process_call(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise AssertionError("Subprocess should not be launched when reusing existing run dir")

    monkeypatch.setattr(module, "_run_scenesmith_process", _unexpected_process_call)
    monkeypatch.setenv("SCENESMITH_PAPER_REPO_DIR", str(repo_dir))
    monkeypatch.setenv("SCENESMITH_PAPER_PYTHON_BIN", "python3")
    monkeypatch.setenv("SCENESMITH_PAPER_EXISTING_RUN_DIR", str(existing_run_dir))
    monkeypatch.setenv("SCENESMITH_PAPER_KEEP_RUN_DIR", "false")

    response = module._run_official_scenesmith(
        {
            "scene_id": "scene_demo_resume",
            "prompt": "Reuse completed SceneSmith artifacts",
        }
    )

    assert len(response["objects"]) == 1
    assert response["paper_stack"]["run_dir"] == str(existing_run_dir.resolve())
    assert response["paper_stack"]["scenesmith_existing_run_reused"] is True
    attempts = response["paper_stack"]["model_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["status"] == "reused_existing_run"
    assert existing_run_dir.exists()


@pytest.mark.unit
def test_collect_raw_objects_handles_rooms_dict_schema() -> None:
    module = _load_module("scenesmith_paper_command_rooms_schema_module", "scenesmith-service/scenesmith_paper_command.py")
    house_state = {
        "rooms": {
            "kitchen": {
                "objects": [
                    {
                        "id": "salt_shaker_1",
                        "semantic_class": "salt_shaker",
                        "pose": {
                            "position": {"x": 1.2, "y": 0.87, "z": -0.4},
                            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                        },
                        "extent": {"x": 0.08, "y": 0.12, "z": 0.08},
                    }
                ]
            }
        }
    }
    objects = module._collect_raw_objects(house_state)
    assert len(objects) == 1
    assert objects[0]["id"] == "salt_shaker_1"
    assert objects[0]["semantic_class"] == "salt_shaker"
