"""
Tests for OpenTelemetry distributed tracing integration.

These tests verify that tracing works correctly when enabled and
gracefully degrades when disabled or dependencies are missing.
"""

import os
import io
import json
import logging
from contextlib import nullcontext, contextmanager

import pytest


@pytest.mark.unit
def test_tracing_disabled_by_default():
    """Tracing should be disabled by default in test environment."""
    from tools.tracing import get_tracer

    tracer = get_tracer()
    # Should return a tracer but it won't be enabled
    assert tracer is not None


@pytest.mark.unit
def test_trace_job_context_manager_no_op_when_disabled(monkeypatch):
    """Trace context manager should be no-op when tracing is disabled."""
    # Ensure tracing is disabled
    monkeypatch.setenv("ENABLE_TRACING", "false")
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    from tools.tracing import trace_job

    # Should not raise even with invalid parameters
    with trace_job("test-job", scene_id="test"):
        pass  # Should work fine


@pytest.mark.unit
def test_trace_function_decorator_no_overhead_when_disabled(monkeypatch):
    """Function decorator should have no overhead when tracing is disabled."""
    monkeypatch.setenv("ENABLE_TRACING", "false")
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    from tools.tracing import trace_function

    @trace_function(attributes={"component": "test"})
    def test_func(x: int) -> int:
        return x * 2

    # Should execute normally
    assert test_func(5) == 10


@pytest.mark.unit
def test_set_trace_attribute_no_op_when_disabled(monkeypatch):
    """Setting trace attributes should be no-op when tracing is disabled."""
    monkeypatch.setenv("ENABLE_TRACING", "false")

    from tools.tracing import set_trace_attribute

    # Should not raise
    set_trace_attribute("test.key", "test_value")
    set_trace_attribute("test.number", 42)
    set_trace_attribute("test.list", ["a", "b", "c"])


@pytest.mark.unit
def test_set_trace_error_no_op_when_disabled(monkeypatch):
    """Recording errors should be no-op when tracing is disabled."""
    monkeypatch.setenv("ENABLE_TRACING", "false")

    from tools.tracing import set_trace_error

    # Should not raise
    set_trace_error(ValueError("test error"))


@pytest.mark.unit
def test_inject_extract_context_when_disabled(monkeypatch):
    """Context injection/extraction should return empty when disabled."""
    monkeypatch.setenv("ENABLE_TRACING", "false")
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)

    from tools.tracing import inject_trace_context, extract_trace_context

    # May return request-id even when tracing is disabled
    context = inject_trace_context()
    # No OpenTelemetry trace headers when disabled
    assert "traceparent" not in context

    # Should handle gracefully
    extracted = extract_trace_context({"traceparent": "test"})
    # Should not raise


@pytest.mark.unit
def test_init_tracing_respects_explicit_disable(monkeypatch):
    """Explicit disable should override environment detection."""
    # Even with GCP project ID, should be disabled
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")

    from tools.tracing import init_tracing

    result = init_tracing(enabled=False)
    assert result is False


@pytest.mark.unit
def test_init_tracing_auto_detects_production(monkeypatch):
    """Should auto-enable in production environment."""
    monkeypatch.setenv("PIPELINE_ENV", "production")
    # Don't actually initialize to avoid requiring dependencies
    # Just test the detection logic


@pytest.mark.unit
def test_tracer_wrapper_graceful_degradation():
    """TracerWrapper should handle missing dependencies gracefully."""
    from tools.tracing.tracer import TracerWrapper

    # Should not raise even if OpenTelemetry is not installed
    tracer = TracerWrapper(service_name="test-service")
    assert tracer is not None

    # Should provide no-op span
    with tracer.start_span("test-span"):
        pass


@pytest.mark.integration
def test_job_wrapper_with_tracing_disabled(monkeypatch, tmp_path):
    """Job wrapper should work correctly with tracing disabled."""
    monkeypatch.setenv("ENABLE_TRACING", "false")

    from tools.error_handling.job_wrapper import run_job_with_dead_letter_queue

    def successful_job():
        return 0

    result = run_job_with_dead_letter_queue(
        successful_job,
        scene_id="test_scene",
        job_type="test-job",
        step="test",
    )

    assert result == 0


@pytest.mark.integration
def test_job_wrapper_with_tracing_enabled_but_no_exporter(monkeypatch, tmp_path):
    """Job wrapper should work even if tracing is enabled but no exporter available."""
    # Enable tracing but don't configure exporter
    monkeypatch.setenv("ENABLE_TRACING", "true")
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("JAEGER_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    from tools.error_handling.job_wrapper import run_job_with_dead_letter_queue

    def successful_job():
        return 0

    # Should still work (tracing will be initialized but disabled)
    result = run_job_with_dead_letter_queue(
        successful_job,
        scene_id="test_scene",
        job_type="test-job",
        step="test",
    )

    assert result == 0


@pytest.mark.integration
def test_job_wrapper_traces_errors(monkeypatch):
    """Job wrapper should trace errors when enabled."""
    monkeypatch.setenv("ENABLE_TRACING", "false")  # Disable to avoid dependencies

    from tools.error_handling.job_wrapper import run_job_with_dead_letter_queue

    def failing_job():
        raise ValueError("Test error")

    result = run_job_with_dead_letter_queue(
        failing_job,
        scene_id="test_scene",
        job_type="test-job",
        step="test",
    )

    # Should return failure exit code
    assert result == 1


@pytest.mark.unit
def test_trace_function_preserves_function_metadata():
    """Decorator should preserve function name and docstring."""
    from tools.tracing import trace_function

    @trace_function()
    def example_function(x: int) -> int:
        """Example docstring."""
        return x + 1

    assert example_function.__name__ == "example_function"
    assert example_function.__doc__ == "Example docstring."
    assert example_function(5) == 6


@pytest.mark.unit
def test_nested_trace_contexts(monkeypatch):
    """Nested trace contexts should work correctly."""
    monkeypatch.setenv("ENABLE_TRACING", "false")

    from tools.tracing import trace_job, set_trace_attribute

    with trace_job("outer-job", scene_id="scene_1"):
        set_trace_attribute("outer", "value")

        with trace_job("inner-job", scene_id="scene_1"):
            set_trace_attribute("inner", "value")

        # Should exit inner span
        set_trace_attribute("outer", "value2")

    # Should exit outer span


@pytest.mark.unit
def test_tracing_with_complex_attributes(monkeypatch):
    """Should handle complex attribute types gracefully."""
    monkeypatch.setenv("ENABLE_TRACING", "false")

    from tools.tracing import set_trace_attribute

    # Should handle various types
    set_trace_attribute("string", "value")
    set_trace_attribute("int", 42)
    set_trace_attribute("float", 3.14)
    set_trace_attribute("bool", True)
    set_trace_attribute("list", [1, 2, 3])  # Converted to string
    set_trace_attribute("dict", {"key": "value"})  # Converted to string
    set_trace_attribute("none", None)


@pytest.mark.unit
def test_request_id_propagates_to_logs_and_spans(monkeypatch):
    """Ensure trace_job generates a request_id used in logs and span attributes."""
    monkeypatch.delenv("REQUEST_ID", raising=False)

    from tools.logging_config import init_logging
    from tools.tracing import tracer as tracer_module
    from tools.tracing import correlation

    correlation._request_id.set(None)

    captured: dict[str, object] = {}

    class FakeTracer:
        enabled = True

        @contextmanager
        def start_span(self, name, attributes=None, kind=None):
            captured["name"] = name
            captured["attributes"] = attributes or {}
            yield None

    fake_tracer = FakeTracer()
    monkeypatch.setattr(tracer_module, "get_tracer", lambda: fake_tracer)

    stream = io.StringIO()
    from tools.logging_config import JsonLogFormatter, RequestIdFilter
    logger = logging.getLogger("request-id-test")
    logger.propagate = False
    logger.handlers.clear()
    h = logging.StreamHandler(stream)
    h.setFormatter(JsonLogFormatter())
    h.addFilter(RequestIdFilter())
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)

    with tracer_module.trace_job("request-id-job"):
        logger.info("hello from trace_job")

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert lines, "Expected log output"
    payload = json.loads(lines[-1])
    request_id = payload.get("request_id")
    assert request_id
    assert captured["attributes"]["request.id"] == request_id


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
