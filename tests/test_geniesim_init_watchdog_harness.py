import threading
import time
from types import SimpleNamespace

import pytest

from tools.geniesim_adapter import local_framework as lf


class _AlwaysOpenCircuitBreaker:
    def allow_request(self) -> bool:
        return True

    def record_failure(self, exc: Exception) -> None:
        del exc

    def record_success(self) -> None:
        return None

    def get_time_until_retry(self) -> float:
        return 0.0


@pytest.mark.unit
def test_call_grpc_abort_unblocks_without_waiting_for_worker(monkeypatch):
    """Harness: simulate a stuck gRPC function and verify abort returns quickly."""
    client = lf.GenieSimGRPCClient(host="localhost", port=1234, timeout=0.2)
    client._circuit_breaker = _AlwaysOpenCircuitBreaker()

    abort_event = threading.Event()
    worker_release = threading.Event()
    result_holder = {"value": None}

    def stuck_call():
        while not worker_release.is_set():
            time.sleep(0.01)
        return "unexpected"

    def _run_call():
        result_holder["value"] = client._call_grpc(
            "init_robot",
            stuck_call,
            fallback="fallback",
            abort_event=abort_event,
        )

    t = threading.Thread(target=_run_call, daemon=True)
    t.start()

    time.sleep(0.1)
    abort_event.set()
    t.join(timeout=1.0)

    # Cleanup in case assertion fails.
    worker_release.set()
    t.join(timeout=1.0)

    assert not t.is_alive(), "Abort path should return without blocking on worker teardown"
    assert result_holder["value"] == "fallback"


@pytest.mark.unit
def test_init_robot_returns_watchdog_abort_error_when_aborted(monkeypatch):
    """Harness: verify init_robot surfaces watchdog abort details to callers."""
    client = lf.GenieSimGRPCClient(host="localhost", port=1234, timeout=0.2)
    client._have_grpc = True
    client._stub = object()

    captured = {}

    def fake_call_grpc(
        action,
        func,
        fallback,
        success_checker=None,
        abort_event=None,
        lock_timeout=None,
    ):
        del action, func, fallback, success_checker, lock_timeout
        captured["abort_event_set"] = bool(abort_event and abort_event.is_set())
        return None

    monkeypatch.setattr(client, "_call_grpc", fake_call_grpc)

    abort_event = threading.Event()
    abort_event.set()

    result = client.init_robot(
        robot_type="G1_omnipicker_fixed.json",
        scene_usd_path="/workspace/scene.usda",
        abort_event=abort_event,
    )

    assert captured["abort_event_set"] is True
    assert result.success is False
    assert result.available is True
    assert "watchdog" in (result.error or "").lower()


@pytest.mark.unit
def test_init_watchdog_triggers_restart_and_retries(monkeypatch):
    """Harness: simulate init stall -> watchdog timeout -> restart -> successful retry."""
    monkeypatch.setenv("GENIESIM_INIT_ABORT_TIMEOUT_S", "0.05")

    class FakeClient:
        def __init__(self):
            self.timeout = 0.01
            self._first_call_timeout = 0.2
            self._joint_names = []
            self._robot_type = ""
            self.init_attempts = 0
            self.restart_attempts = 0
            self.restart_count_toward_budget_args = []

        def connect(self):
            return True

        def init_robot(self, robot_type, base_pose, scene_usd_path, abort_event=None):
            del robot_type, base_pose, scene_usd_path
            self.init_attempts += 1
            if self.init_attempts == 1:
                deadline = time.time() + 1.0
                while abort_event is not None and not abort_event.is_set() and time.time() < deadline:
                    time.sleep(0.005)
                return lf.GrpcCallResult(
                    success=False,
                    available=True,
                    error="init_robot aborted by watchdog (timeout/deadlock guard)",
                )
            return lf.GrpcCallResult(
                success=True,
                available=True,
                payload={"msg": "ok"},
            )

        def _attempt_server_restart(self, count_toward_budget=True):
            self.restart_attempts += 1
            self.restart_count_toward_budget_args.append(count_toward_budget)
            return True

        def set_gripper_state(self, width):
            del width
            return lf.GrpcCallResult(success=True, available=True, payload={"msg": "gripper open"})

        def get_joint_position(self, lock_timeout=None):
            del lock_timeout
            self._joint_names = ["joint_a", "joint_b"]
            return lf.GrpcCallResult(success=True, available=True, payload=[0.0, 0.1])

    framework = object.__new__(lf.GenieSimLocalFramework)
    framework.config = SimpleNamespace(robot_type="franka")
    framework._client = FakeClient()
    logs = []
    framework.log = lambda msg, level="INFO": logs.append((level, msg))
    framework._setup_default_lighting = lambda: None

    framework._init_robot_on_server(
        robot_cfg_file="franka_panda.json",
        base_pose={},
        scene_usd="/workspace/scene.usda",
    )

    assert framework._client.init_attempts >= 2
    assert framework._client.restart_attempts >= 1
    assert framework._client.restart_count_toward_budget_args[0] is False
    assert any("timed out" in msg.lower() for _, msg in logs)
    assert any("server restarted after init timeout" in msg.lower() for _, msg in logs)
