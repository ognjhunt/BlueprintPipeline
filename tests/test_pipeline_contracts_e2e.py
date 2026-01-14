#!/usr/bin/env python3
"""
Contract test for the mock pipeline chain: regen3d -> manifest -> export -> import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fixtures.generate_mock_regen3d import generate_mock_regen3d
from tests.contract_utils import load_schema, validate_json_schema

REPO_ROOT = Path(__file__).resolve().parents[1]
GENIESIM_EXPORT_DIR = REPO_ROOT / "genie-sim-export-job"
GENIESIM_IMPORT_DIR = REPO_ROOT / "genie-sim-import-job"
for path in [GENIESIM_EXPORT_DIR, GENIESIM_IMPORT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_regen3d_to_import_bundle_contract(tmp_path: Path, monkeypatch) -> None:
    scene_id = "contract_chain"
    generate_mock_regen3d(
        output_dir=tmp_path,
        scene_id=scene_id,
        environment_type="kitchen",
    )

    scene_dir = tmp_path / "scenes" / scene_id
    monkeypatch.setenv("USE_GENIESIM", "true")
    monkeypatch.setenv("GENIESIM_MOCK_MODE", "true")
    monkeypatch.setenv("GENIESIM_FORCE_LOCAL", "false")
    monkeypatch.setenv("VARIATION_ASSETS_PREFIX", f"{scene_id}/variation_assets")

    from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep

    runner = LocalPipelineRunner(
        scene_dir=scene_dir,
        verbose=False,
        skip_interactive=True,
        environment_type="kitchen",
    )
    success = runner.run(
        steps=[
            PipelineStep.REGEN3D,
            PipelineStep.SIMREADY,
            PipelineStep.USD,
            PipelineStep.REPLICATOR,
            PipelineStep.VARIATION_GEN,
            PipelineStep.GENIESIM_EXPORT,
        ],
        run_validation=False,
    )
    assert success, "Pipeline failed before export"

    regen3d_dir = scene_dir / "regen3d"
    scene_info = _read_json(regen3d_dir / "scene_info.json")
    validate_json_schema(scene_info, load_schema("regen3d_scene_info.schema.json"))

    geniesim_dir = scene_dir / "geniesim"
    scene_graph = _read_json(geniesim_dir / "scene_graph.json")
    asset_index = _read_json(geniesim_dir / "asset_index.json")
    task_config = _read_json(geniesim_dir / "task_config.json")

    from geniesim_client import GenieSimClient, GenerationParams
    from import_from_geniesim import ImportConfig, run_import_job

    client = GenieSimClient(mock_mode=True, validate_on_init=False)
    submit_result = client.submit_generation_job(
        scene_graph=scene_graph,
        asset_index=asset_index,
        task_config=task_config,
        generation_params=GenerationParams(
            episodes_per_task=1,
            num_variations=1,
            robot_type="franka",
            min_quality_score=0.85,
        ),
        job_name="contract-chain",
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
    lerobot_dir = output_dir / "lerobot"
    dataset_info = _read_json(lerobot_dir / "dataset_info.json")
    validate_json_schema(dataset_info, load_schema("geniesim_local_dataset_info.schema.json"))

    index_schema = load_schema("geniesim_local_episodes_index.schema.json")
    for line in (lerobot_dir / "episodes.jsonl").read_text().splitlines():
        validate_json_schema(json.loads(line), index_schema)
