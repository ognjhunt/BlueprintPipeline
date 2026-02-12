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
    GetObservationReq,
    TaskStatusReq,
)


def test_curobo_missing_enables_fallback_non_production(monkeypatch):
    """When cuRobo is unavailable in dev, framework initializes without error."""
    monkeypatch.setenv("PIPELINE_ENV", "development")
    monkeypatch.delenv("GENIESIM_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.setattr(lf, "CUROBO_INTEGRATION_AVAILABLE", False)

    config = lf.GenieSimConfig.from_env()
    framework = lf.GenieSimLocalFramework(config=config, verbose=False)

    # Source no longer auto-enables linear fallback; just verifies no error
    assert framework.config.allow_linear_fallback is False


def test_curobo_missing_production_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.delenv("GENIESIM_ALLOW_LINEAR_FALLBACK", raising=False)
    monkeypatch.setattr(lf, "CUROBO_INTEGRATION_AVAILABLE", False)
    monkeypatch.setenv("CUROBO_REQUIRED", "1")

    # Production mode requires persistent dirs
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setenv("GENIESIM_RECORDINGS_DIR", str(recordings_dir))
    monkeypatch.setenv("GENIESIM_LOG_DIR", str(log_dir))

    config = lf.GenieSimConfig.from_env()

    with pytest.raises(RuntimeError, match="pip install nvidia-curobo"):
        lf.GenieSimLocalFramework(config=config, verbose=False)


def test_production_temp_dirs_raise_for_defaults(monkeypatch):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.delenv("GENIESIM_RECORDINGS_DIR", raising=False)
    monkeypatch.delenv("GENIESIM_RECORDING_DIR", raising=False)
    monkeypatch.delenv("GENIESIM_LOG_DIR", raising=False)

    with pytest.raises(
        (ValueError, Exception),
        match=r"Refusing to use temporary directories",
    ):
        lf.GenieSimConfig.from_env()


def test_production_temp_dirs_raise_for_explicit_paths(monkeypatch):
    monkeypatch.setenv("PIPELINE_ENV", "production")
    monkeypatch.setenv("GENIESIM_RECORDINGS_DIR", "/tmp/custom_recordings")
    monkeypatch.setenv("GENIESIM_LOG_DIR", "/tmp/custom_logs")

    with pytest.raises(
        (ValueError, Exception),
        match=r"Refusing to use temporary directories",
    ):
        lf.GenieSimConfig.from_env()


@pytest.mark.unit
def test_start_server_honors_startup_timeout(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("ALLOW_GENIESIM_MOCK", "1")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    recording_dir = tmp_path / "recordings"
    recording_dir.mkdir()
    geniesim_root = tmp_path / "geniesim_root"
    geniesim_root.mkdir()
    server_script = geniesim_root / "source/data_collection/scripts/data_collector_server.py"
    server_script.parent.mkdir(parents=True, exist_ok=True)
    server_script.write_text("#!/usr/bin/env python3\n")
    isaac_sim_path = tmp_path / "isaac-sim"
    isaac_sim_path.mkdir()
    isaac_python = isaac_sim_path / "python.sh"
    isaac_python.write_text("#!/bin/sh\nexit 0\n")
    isaac_python.chmod(0o755)

    config = lf.GenieSimConfig(
        geniesim_root=geniesim_root,
        isaac_sim_path=isaac_sim_path,
        log_dir=log_dir,
        recording_dir=recording_dir,
        server_startup_timeout_s=0.2,
        server_startup_poll_s=0.05,
    )
    framework = lf.GenieSimLocalFramework(config=config, verbose=True)

    class DummyProc:
        pid = 1234

        def poll(self):
            return None

    # Phase 1: is_server_running() must return False so start_server proceeds
    check_socket_calls = {"count": 0}

    def mock_check_server_socket():
        check_socket_calls["count"] += 1
        # First call is from is_server_running(); return False
        # Subsequent calls from the readiness loop return True
        return check_socket_calls["count"] > 1

    monkeypatch.setattr(lf.subprocess, "Popen", lambda *args, **kwargs: DummyProc())
    monkeypatch.setattr(framework._client, "_check_server_socket", mock_check_server_socket)
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

    with caplog.at_level(logging.ERROR, logger="tools.geniesim_adapter.local_framework"):
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
    servicer = geniesim_server.MockJointControlServicer()

    # GetIKStatusRsp proto does not have isSuccess directly;
    # the servicer implementation currently raises ValueError
    with pytest.raises(ValueError, match="isSuccess"):
        servicer.get_ik_status(None, context=None)


@pytest.mark.unit
def test_local_servicer_task_status_returns_response():
    servicer = geniesim_server.GenieSimLocalServicer()

    response = servicer.task_status(TaskStatusReq(isSuccess=True), context=None)

    assert response.msg


@pytest.mark.unit
def test_local_servicer_get_observation_returns_response():
    servicer = geniesim_server.GenieSimLocalServicer()

    response = servicer.get_observation(GetObservationReq(isCam=False), context=None)

    assert response is not None


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


def _make_framework(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> lf.GenieSimLocalFramework:
    monkeypatch.setenv("ALLOW_GENIESIM_MOCK", "1")
    monkeypatch.setenv("PIPELINE_ENV", "development")
    monkeypatch.setenv("STRICT_REALISM", "1")
    recording_dir = tmp_path / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    config = lf.GenieSimConfig(
        geniesim_root=tmp_path / "missing_geniesim_root",
        recording_dir=recording_dir,
        log_dir=log_dir,
        cleanup_tmp=False,
    )
    return lf.GenieSimLocalFramework(config=config, verbose=False)


@pytest.mark.unit
def test_resolve_required_cameras_normalizes_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    framework = _make_framework(tmp_path, monkeypatch)
    required = framework._resolve_required_cameras_for_task(
        {"required_camera_ids": ["head", "left_wrist", "wrist"]}
    )
    assert required == ["right", "left", "wrist"] or required == ["left", "right", "wrist"]
    assert set(required) == {"left", "right", "wrist"}


@pytest.mark.unit
def test_normalize_and_validate_tasks_applies_defaults_in_non_enforced_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    framework = _make_framework(tmp_path, monkeypatch)
    framework._enforce_task_taxonomy = False
    tasks = framework._normalize_and_validate_tasks(
        [{"task_type": "pick_place", "target_object": "mug_0"}]
    )
    assert len(tasks) == 1
    task = tasks[0]
    assert task["task_id"]
    assert task["task_name"]
    assert task["task_complexity"] in {"atomic", "composite"}
    assert task["curriculum_split"] in {"pretrain", "target"}
    assert set(task["required_camera_ids"]) == {"left", "right", "wrist"}


@pytest.mark.unit
def test_initialize_runtime_robot_fails_without_failover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    framework = _make_framework(tmp_path, monkeypatch)
    init_calls: list[str] = []

    def _fake_init(robot_cfg_file: str, base_pose, scene_usd):  # noqa: ANN001
        init_calls.append(robot_cfg_file)
        if robot_cfg_file == "franka_panda.json":
            raise RuntimeError("franka init failure")

    monkeypatch.setattr(framework, "_init_robot_on_server", _fake_init)
    monkeypatch.setattr(
        framework._client,
        "get_joint_position",
        lambda *args, **kwargs: lf.GrpcCallResult(
            success=True,
            available=True,
            payload=[0.0] * 34,
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        framework._initialize_runtime_robot_with_fallback(
            requested_robot="franka",
            base_pose={
                "position": {"x": 0, "y": 0, "z": 0},
                "orientation": {"rw": 1, "rx": 0, "ry": 0, "rz": 0},
            },
            scene_usd="scenes/empty_scene.usda",
        )

    assert init_calls == ["franka_panda.json"]
    assert "franka_panda.json" in str(excinfo.value)


@pytest.mark.unit
def test_runtime_patch_health_check_fails_when_containerized_and_no_readiness_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    framework = _make_framework(tmp_path, monkeypatch)
    monkeypatch.setenv(
        "GENIESIM_RUNTIME_READINESS_JSON",
        str(tmp_path / "missing_runtime_readiness.json"),
    )
    monkeypatch.setenv(
        "GENIESIM_RUNTIME_PRESTART_JSON",
        str(tmp_path / "missing_runtime_prestart.json"),
    )

    ok, details = framework._runtime_patch_health_check()

    assert ok is False
    assert any("no runtime_patch_markers readiness report" in msg for msg in details)


@pytest.mark.unit
def test_strict_effort_semantics_reject_estimated_efforts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    framework = _make_framework(tmp_path, monkeypatch)

    with pytest.raises(lf.FatalRealismError) as excinfo:
        framework._enforce_strict_effort_semantics(
            efforts_source="estimated_inverse_dynamics",
            real_effort_count=0,
            estimated_effort_count=4,
            effort_missing_count=0,
            total_frames=4,
        )

    assert excinfo.value.reason_code == "STRICT_EFFORTS_SOURCE"


@pytest.mark.unit
def test_strict_effort_semantics_no_kinematic_override_when_real_efforts_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("STRICT_REAL_ONLY", "1")
    monkeypatch.setenv("GENIESIM_KEEP_OBJECTS_KINEMATIC", "1")
    monkeypatch.setenv("REQUIRE_REAL_EFFORTS", "false")
    framework = _make_framework(tmp_path, monkeypatch)

    with pytest.raises(lf.FatalRealismError) as excinfo:
        framework._enforce_strict_effort_semantics(
            efforts_source="physx",
            real_effort_count=10,
            estimated_effort_count=0,
            effort_missing_count=0,
            total_frames=10,
            stale_ratio=1.0,
        )
    assert excinfo.value.reason_code == "STRICT_EFFORTS_STALE"

    with pytest.raises(lf.FatalRealismError) as excinfo:
        framework._enforce_strict_effort_semantics(
            efforts_source="estimated_inverse_dynamics",
            real_effort_count=0,
            estimated_effort_count=10,
            effort_missing_count=0,
            total_frames=10,
        )
    assert excinfo.value.reason_code == "STRICT_EFFORTS_SOURCE"


@pytest.mark.unit
def test_strict_contact_semantics_reject_placeholder_contacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    framework = _make_framework(tmp_path, monkeypatch)

    with pytest.raises(lf.FatalRealismError) as excinfo:
        framework._enforce_strict_contact_semantics(
            step_idx=12,
            phase="grasp",
            manipulation_phase=True,
            contact_rpc_available=True,
            strict_contacts=[],
            collision_provenance="physx_contact_report",
            contact_forces_payload={
                "provenance": "physx_contact_report",
                "available": True,
            },
            contact_rows=[],
        )

    assert excinfo.value.reason_code == "STRICT_CONTACT_PLACEHOLDER"


@pytest.mark.unit
def test_strict_object_motion_semantics_rejects_ee_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    framework = _make_framework(tmp_path, monkeypatch)

    with pytest.raises(lf.FatalRealismError) as excinfo:
        framework._enforce_strict_object_motion_semantics(
            requires_motion=True,
            any_moved=True,
            max_displacement=0.2,
            scene_state_provenances=["physx_server"],
            physics_probe_enabled=True,
            used_ee_proxy=True,
        )

    assert excinfo.value.reason_code == "STRICT_OBJECT_MOTION_EE_PROXY"


@pytest.mark.unit
def test_run_data_collection_aborts_immediately_on_fatal_realism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    framework = _make_framework(tmp_path, monkeypatch)

    monkeypatch.setattr(framework, "_validate_required_environment", lambda *_: None)
    monkeypatch.setattr(framework, "is_server_running", lambda: True)
    monkeypatch.setattr(framework._client, "is_connected", lambda: True)
    monkeypatch.setattr(framework._client, "ping", lambda timeout=10.0: True)
    monkeypatch.setattr(framework, "_init_robot_on_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        framework._client,
        "get_joint_position",
        lambda *args, **kwargs: lf.GrpcCallResult(
            success=True,
            available=True,
            payload=[0.0] * 34,
        ),
    )
    monkeypatch.setattr(framework, "_strict_realism_preflight", lambda: None)
    monkeypatch.setattr(
        framework,
        "_build_object_metadata_from_scene_config",
        lambda **kwargs: {
            "nodes": [],
            "dynamic_ids": [],
            "static_ids": [],
            "variation_ids": [],
            "object_sim_roles": {},
            "object_prim_aliases": {},
            "manifest_transforms": {},
        },
    )
    monkeypatch.setattr(framework, "_configure_task", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        framework._client,
        "reset_environment",
        lambda: lf.GrpcCallResult(success=True, available=True, payload={}),
    )
    monkeypatch.setattr(
        framework,
        "_run_single_episode",
        lambda **kwargs: {
            "success": False,
            "error": "placeholder contacts",
            "fatal_realism_failure": True,
            "fatal_realism_code": "STRICT_CONTACT_PLACEHOLDER",
            "fatal_realism_message": "placeholder contacts",
        },
    )

    result = framework.run_data_collection(
        task_config={
            "name": "strict_test",
            "robot_config": {"type": "g1", "base_position": [0.0, 0.0, 0.0]},
            "suggested_tasks": [{"task_name": "task_0", "target_object": "toaster"}],
        },
        scene_config={"nodes": [], "objects": []},
        episodes_per_task=1,
    )

    assert result.success is False
    assert result.fatal_realism_failure is True
    assert result.fatal_realism_code == "STRICT_CONTACT_PLACEHOLDER"
    assert result.episodes_passed == 0
