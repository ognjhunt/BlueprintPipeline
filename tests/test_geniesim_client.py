import importlib.util
from pathlib import Path

import pytest


def _load_geniesim_client_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "genie-sim-export-job" / "geniesim_client.py"
    spec = importlib.util.spec_from_file_location("geniesim_client", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_robot_type_success() -> None:
    geniesim_client = _load_geniesim_client_module()

    geniesim_client.validate_robot_type("franka")


def test_validate_robot_type_error() -> None:
    geniesim_client = _load_geniesim_client_module()

    with pytest.raises(ValueError):
        geniesim_client.validate_robot_type("invalid_robot")
