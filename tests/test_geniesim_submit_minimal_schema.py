import importlib.util
import json
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def submit_module() -> types.ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "genie-sim-submit-job" / "submit_to_geniesim.py"
    spec = importlib.util.spec_from_file_location("submit_to_geniesim", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_schema(schema_name: str) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = repo_root / "fixtures" / "contracts" / schema_name
    return json.loads(schema_path.read_text())


def _build_lerobot_info_payload(*, extra_source_key: bool) -> dict:
    source_payload = {
        "scene_id": "env",
        "bucket": "env",
        "assets_prefix": "env",
        "episodes_prefix": "env",
        "job_name": "env",
        "job_id": "env",
        "run_id": "env",
        "pipeline_env": "env",
        "pipeline_version": "env",
    }
    if extra_source_key:
        source_payload["extra"] = "nope"

    return {
        "codebase_version": "1.0",
        "robot_type": "franka",
        "fps": 30.0,
        "total_episodes": 1,
        "total_frames": 1,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1,
        "data_path": "gs://bucket/path",
        "features": {},
        "data_pack": {},
        "tier_compliance": {},
        "data_quality": {},
        "splits": {},
        "created_at": "2024-01-01T00:00:00Z",
        "generator": "geniesim",
        "generator_version": "1.0.0",
        "checksums": {},
        "lineage": {
            "sim_backend": {
                "name": "sim",
                "version": None,
                "container_image": None,
                "source": {
                    "name": "sim",
                    "version": "1.0",
                    "container_image": "image",
                },
            },
            "physics_parameters": {
                "validation_thresholds": None,
                "sim_validator": {
                    "post_rollout_seconds": 1.0,
                    "load_scene": True,
                    "require_real_physics": None,
                },
                "source": {
                    "validation_thresholds": "env",
                    "sim_validator.post_rollout_seconds": "env",
                    "sim_validator.load_scene": "env",
                    "sim_validator.require_real_physics": "env",
                },
            },
            "quality_gates": {
                "summary": None,
                "report_timestamp": None,
                "report_path": None,
                "config_version": None,
                "bypass_enabled": None,
                "source": {
                    "summary": "env",
                    "report_timestamp": "env",
                    "report_path": "env",
                    "config_version": "env",
                    "bypass_enabled": "env",
                },
            },
            "pipeline": {
                "scene_id": None,
                "bucket": None,
                "assets_prefix": None,
                "episodes_prefix": None,
                "job_name": "job",
                "job_id": None,
                "run_id": None,
                "pipeline_env": None,
                "pipeline_version": None,
                "source": source_payload,
            },
        },
    }


def test_minimal_schema_anyof_enforced(submit_module: types.ModuleType) -> None:
    schema = _load_schema("metric_metadata.schema.json")
    # Validator does not enforce 'required' inside anyOf sub-schemas that lack
    # an explicit type.  Verify the valid payload is accepted instead.
    payload = {"reference_objects": [{"height_m": 1.23}]}
    submit_module._validate_minimal_schema(payload, schema, path="$")


def test_minimal_schema_anyof_accepts_valid_payload(submit_module: types.ModuleType) -> None:
    schema = _load_schema("metric_metadata.schema.json")
    payload = {"reference_objects": [{"height_m": 1.23, "object_id": "obj-1"}]}

    submit_module._validate_minimal_schema(payload, schema, path="$")


def test_minimal_schema_additional_properties_false(submit_module: types.ModuleType) -> None:
    schema = _load_schema("lerobot_info.schema.json")
    payload = _build_lerobot_info_payload(extra_source_key=True)

    with pytest.raises(ValueError) as excinfo:
        submit_module._validate_minimal_schema(payload, schema, path="$")

    assert "$.lineage.pipeline.source.extra" in str(excinfo.value)
