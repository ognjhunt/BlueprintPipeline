#!/usr/bin/env python3
"""
Contract tests for Genie Sim mock mode artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENIESIM_EXPORT_DIR = REPO_ROOT / "genie-sim-export-job"
GENIESIM_IMPORT_DIR = REPO_ROOT / "genie-sim-import-job"

sys.path.insert(0, str(GENIESIM_EXPORT_DIR))
sys.path.insert(0, str(GENIESIM_IMPORT_DIR))

from geniesim_client import GenieSimClient, GenerationParams
from import_from_geniesim import ImportConfig, run_import_job


def _build_scene_graph(scene_id: str) -> dict:
    return {
        "scene_id": scene_id,
        "coordinate_system": "y_up",
        "meters_per_unit": 1.0,
        "nodes": [],
        "edges": [],
        "metadata": {},
    }


def _build_task_config(scene_id: str) -> dict:
    return {
        "scene_id": scene_id,
        "environment_type": "kitchen",
        "tasks": [
            {"task_id": "pick", "task_name": "pick"},
            {"task_id": "place", "task_name": "place"},
        ],
    }


def test_mock_import_contracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene_id = "mock_scene"
    monkeypatch.setenv("GENIESIM_MOCK_MODE", "true")
    monkeypatch.setenv("SCENE_ID", scene_id)

    client = GenieSimClient(mock_mode=True, validate_on_init=False)
    generation_params = GenerationParams(
        episodes_per_task=1,
        num_variations=1,
        robot_type="franka",
        min_quality_score=0.85,
    )

    submit_result = client.submit_generation_job(
        scene_graph=_build_scene_graph(scene_id),
        asset_index={"assets": [], "metadata": {}},
        task_config=_build_task_config(scene_id),
        generation_params=generation_params,
        job_name="contract-test",
    )
    assert submit_result.success

    output_dir = tmp_path / "import_output"
    config = ImportConfig(
        job_id=submit_result.job_id or "mock-job",
        output_dir=output_dir,
        min_quality_score=0.85,
        enable_validation=True,
        filter_low_quality=True,
        require_lerobot=False,
        wait_for_completion=True,
        poll_interval=0,
    )
    result = run_import_job(config, client)
    client.close()

    assert result.success
    assert result.episodes_passed_validation == 1
    assert result.episodes_filtered == 1

    lerobot_dir = output_dir / "lerobot"
    assert (lerobot_dir / "dataset_info.json").is_file()
    assert (lerobot_dir / "episodes.jsonl").is_file()
    assert (lerobot_dir / "episode_000000.parquet").is_file()

    import_manifest_path = output_dir / "import_manifest.json"
    assert import_manifest_path.is_file()
    import_manifest = json.loads(import_manifest_path.read_text())

    assert import_manifest["schema_version"] == "1.2"
    assert import_manifest["provenance"]["source"] == "genie_sim"
    assert import_manifest["provenance"]["job_id"] == submit_result.job_id
    assert import_manifest["provenance"]["scene_id"] == scene_id

    assert import_manifest["quality"]["threshold"] == 0.85
    assert import_manifest["episodes"]["passed_validation"] == 1
    assert import_manifest["episodes"]["filtered"] == 1
    assert import_manifest["readme_path"] == "README.md"
    assert "sha256" in import_manifest["package"]

    episode_checksums = import_manifest["checksums"]["episodes"]
    assert episode_checksums
    assert all(len(entry["sha256"]) == 64 for entry in episode_checksums)
