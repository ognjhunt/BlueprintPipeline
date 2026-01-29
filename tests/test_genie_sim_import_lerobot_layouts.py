from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_lerobot_v2_layout_validation(load_job_module, repo_root: Path) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")
    fixture_dir = repo_root / "genie-sim-import-job" / "fixtures" / "lerobot_v2"
    lerobot_dir = fixture_dir / "lerobot"

    validation = module._validate_lerobot_metadata_files(fixture_dir, lerobot_dir, require_parquet_validation=False)

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


def test_lerobot_v0_3_3_layout_validation(load_job_module, repo_root: Path) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")
    fixture_dir = repo_root / "genie-sim-import-job" / "fixtures" / "lerobot_v0_3_3"
    lerobot_dir = fixture_dir / "lerobot"

    validation = module._validate_lerobot_metadata_files(fixture_dir, lerobot_dir, require_parquet_validation=False)

    assert validation["export_format"] == "lerobot_v0.3.3"
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


def test_lerobot_v0_4_layout_validation(load_job_module, repo_root: Path) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")
    fixture_dir = repo_root / "genie-sim-import-job" / "fixtures" / "lerobot_v0_4"
    lerobot_dir = fixture_dir / "lerobot"

    validation = module._validate_lerobot_metadata_files(fixture_dir, lerobot_dir, require_parquet_validation=False)

    assert validation["export_format"] == "lerobot_v0.4"
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

    validation = module._validate_lerobot_metadata_files(fixture_dir, lerobot_dir, require_parquet_validation=False)

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


def test_detect_lerobot_export_format_versions(load_job_module) -> None:
    module = load_job_module("geniesim_import", "import_from_geniesim.py")

    assert (
        module._detect_lerobot_export_format({"version": "0.3.3"}).value
        == "lerobot_v0.3.3"
    )
    assert module._detect_lerobot_export_format({"version": "2.0"}).value == "lerobot_v2"
    assert module._detect_lerobot_export_format({"version": "3.0"}).value == "lerobot_v3"
