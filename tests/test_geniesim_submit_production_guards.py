import importlib.util
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


def _run_with_env(
    *,
    monkeypatch: pytest.MonkeyPatch,
    submit_module: types.ModuleType,
    env: dict[str, str],
    tmp_path: Path,
) -> object:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return submit_module._run_local_data_collection_with_handshake(
        scene_manifest={"usd_path": "/tmp/scene.usd"},
        task_config={"name": "test-task"},
        output_dir=tmp_path / "output",
        robot_type="franka",
        episodes_per_task=1,
        expected_server_version="3.0.0",
        required_capabilities={"data_collection"},
    )


def test_production_rejects_temp_dirs(
    monkeypatch: pytest.MonkeyPatch,
    submit_module: types.ModuleType,
    tmp_path: Path,
) -> None:
    result = _run_with_env(
        monkeypatch=monkeypatch,
        submit_module=submit_module,
        tmp_path=tmp_path,
        env={
            "PIPELINE_ENV": "production",
            "GENIESIM_RECORDINGS_DIR": "/tmp/geniesim_recordings",
            "GENIESIM_LOG_DIR": "/tmp/geniesim_logs",
        },
    )

    assert not result.success
    assert result.errors
    assert (
        "Refusing to use temporary directories for Genie Sim in production."
        in result.errors[0]
    )


def test_production_rejects_linear_fallback_flags(
    monkeypatch: pytest.MonkeyPatch,
    submit_module: types.ModuleType,
    tmp_path: Path,
) -> None:
    result = _run_with_env(
        monkeypatch=monkeypatch,
        submit_module=submit_module,
        tmp_path=tmp_path,
        env={
            "PIPELINE_ENV": "production",
            "GENIESIM_RECORDINGS_DIR": "/var/geniesim/recordings",
            "GENIESIM_LOG_DIR": "/var/geniesim/logs",
            "GENIESIM_ALLOW_LINEAR_FALLBACK": "true",
        },
    )

    assert not result.success
    assert result.errors
    assert (
        "Refusing to enable linear fallback in production."
        in result.errors[0]
    )
