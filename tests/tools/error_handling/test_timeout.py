import pytest

import tools.error_handling.timeout as timeout_module


class ImmediateTimer:
    def __init__(self, seconds, callback):
        self.seconds = seconds
        self.callback = callback
        self.cancelled = False

    def start(self):
        self.callback()

    def cancel(self):
        self.cancelled = True


class NoOpTimer:
    def __init__(self, seconds, callback):
        self.seconds = seconds
        self.callback = callback
        self.cancelled = False

    def start(self):
        return None

    def cancel(self):
        self.cancelled = True


class TestTimeoutThread:
    def test_timeout_thread_triggers(self, monkeypatch):
        monkeypatch.setattr(timeout_module.threading, "Timer", ImmediateTimer)

        with pytest.raises(timeout_module.TimeoutError) as exc_info:
            with timeout_module.timeout_thread(0.01, "Too slow"):
                return_value = "done"  # noqa: F841

        assert "Too slow" in str(exc_info.value)

    def test_with_timeout_propagates_exception(self, monkeypatch):
        monkeypatch.setattr(timeout_module.threading, "Timer", NoOpTimer)

        @timeout_module.with_timeout(5.0, use_thread=True)
        def raises_value_error():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            raises_value_error()
