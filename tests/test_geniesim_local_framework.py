import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.geniesim_adapter import local_framework as lf
from tools.geniesim_adapter import geniesim_server
from tools.geniesim_adapter.geniesim_grpc_pb2 import (
    GetIKStatusRequest,
    GetObservationRequest,
    TaskStatusRequest,
)


def test_curobo_missing_enables_fallback_non_production(monkeypatch):
    monkeypatch.setenv("GENIESIM_ENV", "development")
    monkeypatch.delenv("GENIESIM_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.setattr(lf, "CUROBO_INTEGRATION_AVAILABLE", False)

    config = lf.GenieSimConfig.from_env()
    framework = lf.GenieSimLocalFramework(config=config, verbose=False)

    assert framework.config.allow_linear_fallback is True


def test_curobo_missing_production_fails_fast(monkeypatch):
    monkeypatch.setenv("GENIESIM_ENV", "production")
    monkeypatch.delenv("GENIESIM_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.setattr(lf, "CUROBO_INTEGRATION_AVAILABLE", False)

    config = lf.GenieSimConfig.from_env()

    with pytest.raises(RuntimeError, match="pip install nvidia-curobo"):
        lf.GenieSimLocalFramework(config=config, verbose=False)


def test_production_temp_dirs_raise_for_defaults(monkeypatch):
    monkeypatch.setenv("GENIESIM_ENV", "production")
    monkeypatch.delenv("GENIESIM_RECORDINGS_DIR", raising=False)
    monkeypatch.delenv("GENIESIM_RECORDING_DIR", raising=False)
    monkeypatch.delenv("GENIESIM_LOG_DIR", raising=False)

    with pytest.raises(
        ValueError,
        match=r"GENIESIM_RECORDINGS_DIR.*GENIESIM_LOG_DIR",
    ):
        lf.GenieSimConfig.from_env()


def test_production_temp_dirs_raise_for_explicit_paths(monkeypatch):
    monkeypatch.setenv("GENIESIM_ENV", "production")
    monkeypatch.setenv("GENIESIM_RECORDINGS_DIR", "/tmp/custom_recordings")
    monkeypatch.setenv("GENIESIM_LOG_DIR", "/tmp/custom_logs")

    with pytest.raises(
        ValueError,
        match=r"GENIESIM_RECORDINGS_DIR.*GENIESIM_LOG_DIR",
    ):
        lf.GenieSimConfig.from_env()


@pytest.mark.unit
def test_check_geniesim_availability_allows_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("GENIESIM_ENV", "development")
    monkeypatch.setenv("ALLOW_GENIESIM_MOCK", "1")
    monkeypatch.setenv("GENIESIM_ROOT", str(tmp_path / "missing_geniesim"))
    isaac_path = tmp_path / "isaac"
    isaac_path.mkdir()
    (isaac_path / "python.sh").write_text("#!/bin/sh\necho isaac\n")
    monkeypatch.setenv("ISAAC_SIM_PATH", str(isaac_path))

    monkeypatch.setattr(lf.GenieSimGRPCClient, "_check_server_socket", lambda self: False)
    monkeypatch.setitem(sys.modules, "grpc", type("GrpcStub", (), {})())

    status = lf.check_geniesim_availability()

    assert status["mock_server_allowed"] is True
    assert status["isaac_sim_available"] is True
    assert status["available"] is True


@pytest.mark.unit
def test_local_servicer_get_ik_status_returns_success():
    servicer = geniesim_server.GenieSimLocalServicer(joint_count=6)

    response = servicer.GetIKStatus(GetIKStatusRequest(), context=None)

    assert response.success is True
    assert response.ik_solvable is True
    assert len(response.solution) == 6


@pytest.mark.unit
def test_local_servicer_get_task_status_returns_success():
    servicer = geniesim_server.GenieSimLocalServicer()

    response = servicer.GetTaskStatus(TaskStatusRequest(task_id="task-1"), context=None)

    assert response.success is True
    assert response.status
    assert response.progress >= 0.0


@pytest.mark.unit
def test_local_servicer_stream_observations_yields_responses():
    servicer = geniesim_server.GenieSimLocalServicer()

    responses = list(servicer.StreamObservations(GetObservationRequest(), context=None))

    assert responses
    assert all(response.success for response in responses)
