import pytest

import tools.error_handling.retry as retry_module


class TestRetryWithBackoff:
    def test_exhausted_retries_calls_callbacks_and_backoff(self, monkeypatch):
        delays = []
        attempts = []
        failures = []
        call_count = {"count": 0}

        def fake_sleep(delay):
            delays.append(delay)

        def on_retry(attempt, exc, delay):
            attempts.append((attempt, type(exc).__name__, delay))

        def on_failure(attempt, exc):
            failures.append((attempt, type(exc).__name__))

        def always_fail():
            call_count["count"] += 1
            raise ConnectionError("network down")

        monkeypatch.setattr(retry_module.time, "sleep", fake_sleep)

        wrapped = retry_module.retry_with_backoff(
            max_retries=2,
            base_delay=0.5,
            backoff_factor=2.0,
            jitter=False,
            on_retry=on_retry,
            on_failure=on_failure,
        )(always_fail)

        with pytest.raises(ConnectionError):
            wrapped()

        assert call_count["count"] == 2
        assert delays == [0.5]
        assert attempts == [(1, "ConnectionError", 0.5)]
        assert failures == [(2, "ConnectionError")]

    def test_retry_context_records_delay_and_stops(self, monkeypatch):
        delays = []

        def fake_sleep(delay):
            delays.append(delay)

        config = retry_module.RetryConfig(
            max_retries=2,
            base_delay=1.0,
            backoff_factor=2.0,
            jitter=False,
        )
        ctx = retry_module.RetryContext(config=config)
        monkeypatch.setattr(retry_module.time, "sleep", fake_sleep)

        assert ctx.record_failure(ConnectionError("fail-1")) is True
        assert ctx.record_failure(ConnectionError("fail-2")) is False

        assert ctx.attempt == 2
        assert ctx.total_delay == pytest.approx(1.0)
        assert delays == [1.0]
