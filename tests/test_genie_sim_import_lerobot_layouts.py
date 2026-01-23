from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_lerobot_v2_layout_validation(load_job_module, repo_root: Path) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")
    fixture_dir = repo_root / "genie-sim-import-job" / "fixtures" / "lerobot_v2"
    lerobot_dir = fixture_dir / "lerobot"

    validation = module._validate_lerobot_metadata_files(fixture_dir, lerobot_dir)

    assert validation["export_format"] == "lerobot_v2"
    assert validation["schema_errors"] == []

    dataset_info = _load_json(lerobot_dir / "dataset_info.json")
    assert (
        module._validate_schema_payload(
            dataset_info,
            "geniesim_local_dataset_info.schema.json",
            "dataset_info",
        )
        == []
    )


def test_lerobot_v3_layout_validation(load_job_module, repo_root: Path) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")
    fixture_dir = repo_root / "genie-sim-import-job" / "fixtures" / "lerobot_v3"
    lerobot_dir = fixture_dir / "lerobot"

    validation = module._validate_lerobot_metadata_files(fixture_dir, lerobot_dir)

    assert validation["export_format"] == "lerobot_v3"
    assert validation["schema_errors"] == []

    dataset_info = _load_json(lerobot_dir / "dataset_info.json")
    assert (
        module._validate_schema_payload(
            dataset_info,
            "geniesim_local_dataset_info.schema.json",
            "dataset_info",
        )
        == []
    )
