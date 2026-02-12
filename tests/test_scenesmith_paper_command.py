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

    def _fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
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
        (output_dir / "house_state.json").write_text(json.dumps(house_state), encoding="utf-8")

        class _Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return _Result()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
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
