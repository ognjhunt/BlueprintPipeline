import pytest

import tools.error_handling.circuit_breaker as circuit_breaker

pytestmark = pytest.mark.usefixtures("add_repo_to_path")


class TimeController:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.unit
def test_circuit_breaker_transitions_closed_open_half_open_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = TimeController()
    monkeypatch.setattr(circuit_breaker.time, "time", controller.time)

    breaker = circuit_breaker.CircuitBreaker(
        "test-circuit-breaker-transitions",
        failure_threshold=2,
        success_threshold=2,
        recovery_timeout=5.0,
        failure_window=60.0,
    )

    assert breaker.state == circuit_breaker.CircuitState.CLOSED

    breaker.record_failure(RuntimeError("boom-1"))
    assert breaker.state == circuit_breaker.CircuitState.CLOSED

    breaker.record_failure(RuntimeError("boom-2"))
    assert breaker.state == circuit_breaker.CircuitState.OPEN
    assert breaker.allow_request() is False

    with pytest.raises(circuit_breaker.CircuitBreakerOpen):
        with breaker:
            pass

    assert breaker.get_time_until_retry() == pytest.approx(5.0)

    controller.advance(5.0)
    assert breaker.state == circuit_breaker.CircuitState.HALF_OPEN

    breaker.record_success()
    assert breaker.state == circuit_breaker.CircuitState.HALF_OPEN

    breaker.record_success()
    assert breaker.state == circuit_breaker.CircuitState.CLOSED


@pytest.mark.unit
def test_circuit_breaker_cooldown_and_failure_in_half_open(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = TimeController()
    monkeypatch.setattr(circuit_breaker.time, "time", controller.time)

    breaker = circuit_breaker.CircuitBreaker(
        "test-circuit-breaker-cooldown",
        failure_threshold=1,
        success_threshold=1,
        recovery_timeout=3.0,
        failure_window=60.0,
    )

    breaker.record_failure(ValueError("initial failure"))
    assert breaker.state == circuit_breaker.CircuitState.OPEN

    controller.advance(2.0)
    assert breaker.state == circuit_breaker.CircuitState.OPEN

    controller.advance(1.0)
    assert breaker.state == circuit_breaker.CircuitState.HALF_OPEN

    with pytest.raises(ValueError):
        with breaker:
            raise ValueError("half-open failure")

    assert breaker.state == circuit_breaker.CircuitState.OPEN


@pytest.mark.unit
def test_circuit_breaker_defaults_and_overrides() -> None:
    default_breaker = circuit_breaker.CircuitBreaker("test-circuit-breaker-defaults")

    assert default_breaker.config.failure_threshold == 5
    assert default_breaker.config.success_threshold == 2
    assert default_breaker.config.recovery_timeout == 30.0
    assert default_breaker.config.failure_window == 60.0
    assert Exception in default_breaker.config.failure_exceptions
    assert KeyboardInterrupt in default_breaker.config.excluded_exceptions

    override_breaker = circuit_breaker.CircuitBreaker(
        "test-circuit-breaker-overrides",
        failure_threshold=4,
        success_threshold=3,
        recovery_timeout=7.5,
        failure_window=15.0,
    )

    assert override_breaker.config.failure_threshold == 4
    assert override_breaker.config.success_threshold == 3
    assert override_breaker.config.recovery_timeout == 7.5
    assert override_breaker.config.failure_window == 15.0
