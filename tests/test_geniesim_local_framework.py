import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from enum import Enum

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
    monkeypatch.setenv("PIPELINE_ENV", "development")
    monkeypatch.delenv("GENIESIM_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.setattr(lf, "CUROBO_INTEGRATION_AVAILABLE", False)

    config = lf.GenieSimConfig.from_env()
    framework = lf.GenieSimLocalFramework(config=config, verbose=False)

    assert framework.config.allow_linear_fallback is True


def test_curobo_missing_production_fails_fast(monkeypatch):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.delenv("GENIESIM_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.setattr(lf, "CUROBO_INTEGRATION_AVAILABLE", False)

    config = lf.GenieSimConfig.from_env()

    with pytest.raises(RuntimeError, match="pip install nvidia-curobo"):
        lf.GenieSimLocalFramework(config=config, verbose=False)


def test_production_temp_dirs_raise_for_defaults(monkeypatch):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.delenv("GENIESIM_RECORDINGS_DIR", raising=False)
    monkeypatch.delenv("GENIESIM_RECORDING_DIR", raising=False)
    monkeypatch.delenv("GENIESIM_LOG_DIR", raising=False)

    with pytest.raises(
        ValueError,
        match=r"GENIESIM_RECORDINGS_DIR.*GENIESIM_LOG_DIR",
    ):
        lf.GenieSimConfig.from_env()


def test_production_temp_dirs_raise_for_explicit_paths(monkeypatch):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("GENIESIM_RECORDINGS_DIR", "/tmp/custom_recordings")
    monkeypatch.setenv("GENIESIM_LOG_DIR", "/tmp/custom_logs")

    with pytest.raises(
        ValueError,
        match=r"GENIESIM_RECORDINGS_DIR.*GENIESIM_LOG_DIR",
    ):
        lf.GenieSimConfig.from_env()


@pytest.mark.unit
def test_start_server_honors_startup_timeout(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("ALLOW_GENIESIM_MOCK", "1")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    recording_dir = tmp_path / "recordings"
    recording_dir.mkdir()

    config = lf.GenieSimConfig(
        geniesim_root=tmp_path / "missing_geniesim",
        log_dir=log_dir,
        recording_dir=recording_dir,
        server_startup_timeout_s=0.2,
        server_startup_poll_s=0.05,
    )
    framework = lf.GenieSimLocalFramework(config=config, verbose=False)

    class DummyProc:
        pid = 1234

        def poll(self):
            return None

    monkeypatch.setattr(lf.subprocess, "Popen", lambda *args, **kwargs: DummyProc())
    monkeypatch.setattr(framework._client, "_check_server_socket", lambda: True)
    monkeypatch.setattr(framework._client, "connect", lambda: True)
    monkeypatch.setattr(
        framework._client,
        "get_server_info",
        lambda timeout=None: lf.GrpcCallResult(
            success=False,
            available=True,
            error="not ready",
        ),
    )

    current_time = {"value": 0.0}

    def fake_time():
        current_time["value"] += 0.05
        return current_time["value"]

    monkeypatch.setattr(lf.time, "time", fake_time)
    monkeypatch.setattr(lf.time, "sleep", lambda *_: None)

    with caplog.at_level(logging.ERROR):
        result = framework.start_server(wait_for_ready=True)

    assert result is False
    assert (
        f"{config.server_startup_timeout_s}s" in caplog.text
        or f"{config.server_startup_timeout_s:.1f}s" in caplog.text
    )


@pytest.mark.unit
def test_check_geniesim_availability_allows_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_ENV", "development")
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


class DummyCircuitBreaker:
    def __init__(self):
        self.failures = 0
        self.successes = 0

    def allow_request(self) -> bool:
        return True

    def record_failure(self, exc: Exception) -> None:
        self.failures += 1

    def record_success(self) -> None:
        self.successes += 1

    def get_time_until_retry(self) -> float:
        return 0.0


@pytest.mark.unit
def test_call_grpc_retries_retryable_errors(monkeypatch):
    monkeypatch.setenv("GENIESIM_GRPC_MAX_RETRIES", "3")
    monkeypatch.setenv("GENIESIM_GRPC_RETRY_BASE_S", "0.1")
    monkeypatch.setenv("GENIESIM_GRPC_RETRY_MAX_S", "1.0")

    class StatusCode(Enum):
        UNAVAILABLE = 1
        DEADLINE_EXCEEDED = 2
        RESOURCE_EXHAUSTED = 3
        PERMISSION_DENIED = 4

    class FakeRpcError(Exception):
        def __init__(self, code):
            super().__init__(f"code={code}")
            self._code = code

        def code(self):
            return self._code

    fake_grpc = SimpleNamespace(RpcError=FakeRpcError, StatusCode=StatusCode)
    monkeypatch.setattr(lf, "grpc", fake_grpc)
    monkeypatch.setattr(lf, "GRPC_STUBS_AVAILABLE", True)

    sleep_calls = []
    monkeypatch.setattr(lf.time, "sleep", lambda delay: sleep_calls.append(delay))

    client = lf.GenieSimGRPCClient(host="localhost", port=1234, timeout=1.0)
    client._circuit_breaker = DummyCircuitBreaker()

    calls = {"count": 0}

    def flaky_call():
        calls["count"] += 1
        if calls["count"] < 3:
            raise FakeRpcError(StatusCode.UNAVAILABLE)
        return "ok"

    result = client._call_grpc("flaky", flaky_call, fallback="fallback")

    assert result == "ok"
    assert calls["count"] == 3
    assert sleep_calls == [0.1, 0.2]
    assert client._circuit_breaker.failures == 2
    assert client._circuit_breaker.successes == 1


@pytest.mark.unit
def test_call_grpc_non_retryable_error_no_retry(monkeypatch):
    monkeypatch.setenv("GENIESIM_GRPC_MAX_RETRIES", "3")
    monkeypatch.setenv("GENIESIM_GRPC_RETRY_BASE_S", "0.1")
    monkeypatch.setenv("GENIESIM_GRPC_RETRY_MAX_S", "1.0")

    class StatusCode(Enum):
        UNAVAILABLE = 1
        DEADLINE_EXCEEDED = 2
        RESOURCE_EXHAUSTED = 3
        PERMISSION_DENIED = 4

    class FakeRpcError(Exception):
        def __init__(self, code):
            super().__init__(f"code={code}")
            self._code = code

        def code(self):
            return self._code

    fake_grpc = SimpleNamespace(RpcError=FakeRpcError, StatusCode=StatusCode)
    monkeypatch.setattr(lf, "grpc", fake_grpc)
    monkeypatch.setattr(lf, "GRPC_STUBS_AVAILABLE", True)

    sleep_calls = []
    monkeypatch.setattr(lf.time, "sleep", lambda delay: sleep_calls.append(delay))

    client = lf.GenieSimGRPCClient(host="localhost", port=1234, timeout=1.0)
    client._circuit_breaker = DummyCircuitBreaker()

    calls = {"count": 0}

    def failing_call():
        calls["count"] += 1
        raise FakeRpcError(StatusCode.PERMISSION_DENIED)

    result = client._call_grpc("fail", failing_call, fallback="fallback")

    assert result == "fallback"
    assert calls["count"] == 1
    assert sleep_calls == []
    assert client._circuit_breaker.failures == 1
    assert client._circuit_breaker.successes == 0
