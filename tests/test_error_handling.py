"""Tests for error handling modules."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from tools.error_handling.errors import (
    PipelineError,
    ErrorCategory,
    ErrorSeverity,
    ErrorContext,
)
from tools.error_handling.circuit_breaker import CircuitBreaker, CircuitBreakerState
from tools.error_handling.timeout import timeout_handler, TimeoutError as TimeoutException
from tools.error_handling.dead_letter import (
    DeadLetterMessage,
    DeadLetterQueue,
    LocalDeadLetterQueue,
)
from tools.error_handling.retry import RetryConfig, retry_with_backoff
from tools.error_handling.partial_failure import PartialFailureHandler


class TestPipelineError:
    """Test PipelineError class."""

    def test_pipeline_error_creation(self):
        """Test creating a pipeline error."""
        error = PipelineError(
            message="Test error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
        )
        assert error.message == "Test error"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.ERROR

    def test_pipeline_error_with_context(self):
        """Test pipeline error with error context."""
        context = ErrorContext(
            scene_id="scene_001",
            step="episode_generation",
            attempt=2,
            max_attempts=3,
        )
        error = PipelineError(
            message="Scene generation failed",
            context=context,
            category=ErrorCategory.EXECUTION,
        )
        assert error.context.scene_id == "scene_001"
        assert error.context.step == "episode_generation"
        assert error.context.attempt == 2

    def test_pipeline_error_to_dict(self):
        """Test error serialization to dict."""
        error = PipelineError(
            message="Test error",
            category=ErrorCategory.VALIDATION,
        )
        error_dict = error.to_dict()
        assert error_dict["message"] == "Test error"
        assert error_dict["category"] == "validation"


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        assert breaker.state == CircuitBreakerState.CLOSED

        # Successful call
        result = breaker.call(lambda: "success")
        assert result == "success"

    def test_circuit_breaker_open_on_failure_threshold(self):
        """Test circuit breaker opens after threshold."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Trigger failures
        with pytest.raises(Exception):
            breaker.call(lambda: 1 / 0)

        assert breaker.state == CircuitBreakerState.CLOSED  # Still closed

        with pytest.raises(Exception):
            breaker.call(lambda: 1 / 0)

        assert breaker.state == CircuitBreakerState.OPEN  # Now open

    def test_circuit_breaker_rejects_calls_when_open(self):
        """Test circuit breaker rejects calls when open."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1)

        # Trigger failure to open circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1 / 0)

        # Circuit should reject further calls
        with pytest.raises(Exception) as exc_info:
            breaker.call(lambda: "should_not_run")
        assert "Circuit breaker is OPEN" in str(exc_info.value)

    def test_circuit_breaker_half_open_state(self):
        """Test circuit breaker half-open state and recovery."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.5)

        # Open circuit
        with pytest.raises(Exception):
            breaker.call(lambda: 1 / 0)

        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(0.6)

        # Circuit should be in half-open state
        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Successful call should close circuit
        result = breaker.call(lambda: "recovered")
        assert result == "recovered"
        assert breaker.state == CircuitBreakerState.CLOSED


class TestTimeoutHandler:
    """Test timeout handler decorator."""

    def test_timeout_handler_succeeds(self):
        """Test timeout handler with successful execution."""
        @timeout_handler(timeout=2)
        def slow_function(duration):
            time.sleep(duration)
            return "complete"

        result = slow_function(0.5)
        assert result == "complete"

    def test_timeout_handler_exceeds_timeout(self):
        """Test timeout handler with timeout exceeded."""
        @timeout_handler(timeout=0.5)
        def slow_function(duration):
            time.sleep(duration)
            return "complete"

        with pytest.raises(TimeoutException):
            slow_function(1)


class TestDeadLetterQueue:
    """Test DeadLetterQueue implementations."""

    @pytest.fixture
    def temp_dlq_dir(self):
        """Create temporary directory for DLQ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_local_dlq_publish(self, temp_dlq_dir):
        """Test publishing to local DLQ."""
        dlq = LocalDeadLetterQueue(directory=str(temp_dlq_dir))

        message = DeadLetterMessage(
            scene_id="scene_001",
            job_type="episode_generation",
            error_type="ValueError",
            error_message="Invalid input",
        )

        message_id = dlq.publish(message)
        assert message_id == message.message_id

        # Check file was created
        pending_file = temp_dlq_dir / "pending" / f"{message_id}.json"
        assert pending_file.exists()

    def test_local_dlq_get_pending(self, temp_dlq_dir):
        """Test retrieving pending messages from local DLQ."""
        dlq = LocalDeadLetterQueue(directory=str(temp_dlq_dir))

        # Publish multiple messages
        for i in range(3):
            message = DeadLetterMessage(
                scene_id=f"scene_{i:03d}",
                job_type="test",
                error_type="Error",
                error_message=f"Error {i}",
            )
            dlq.publish(message)

        # Retrieve pending
        pending = dlq.get_pending(limit=10)
        assert len(pending) == 3

    def test_local_dlq_mark_resolved(self, temp_dlq_dir):
        """Test marking message as resolved."""
        dlq = LocalDeadLetterQueue(directory=str(temp_dlq_dir))

        message = DeadLetterMessage(
            scene_id="scene_001",
            error_type="Error",
            error_message="Test error",
        )
        message_id = dlq.publish(message)

        # Mark as resolved
        success = dlq.mark_resolved(message_id)
        assert success

        # Check file moved to resolved
        resolved_file = temp_dlq_dir / "resolved" / f"{message_id}.json"
        assert resolved_file.exists()

        # Check not in pending
        pending_file = temp_dlq_dir / "pending" / f"{message_id}.json"
        assert not pending_file.exists()

    def test_local_dlq_get_stats(self, temp_dlq_dir):
        """Test getting DLQ statistics."""
        dlq = LocalDeadLetterQueue(directory=str(temp_dlq_dir))

        # Publish some messages
        for i in range(2):
            message = DeadLetterMessage(
                scene_id=f"scene_{i}",
                error_type="Error",
                error_message=f"Error {i}",
            )
            dlq.publish(message)

        stats = dlq.get_stats()
        assert stats["pending"] == 2
        assert stats["resolved"] == 0
        assert stats["total"] == 2


class TestRetryWithBackoff:
    """Test retry with backoff decorator."""

    def test_retry_succeeds_immediately(self):
        """Test retry when function succeeds immediately."""
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.1)
        def reliable_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = reliable_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_succeeds_after_failures(self):
        """Test retry succeeds after some failures."""
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 2

    def test_retry_exhausts_retries(self):
        """Test retry exhausts all retries."""
        @retry_with_backoff(max_retries=2, initial_delay=0.05)
        def always_fails():
            raise ValueError("Permanent error")

        with pytest.raises(ValueError):
            always_fails()


class TestPartialFailureHandler:
    """Test partial failure handler."""

    def test_partial_failure_handler_init(self):
        """Test initializing partial failure handler."""
        handler = PartialFailureHandler(scene_id="scene_001")
        assert handler.scene_id == "scene_001"
        assert len(handler.failed_items) == 0

    def test_track_failure(self):
        """Test tracking failures."""
        handler = PartialFailureHandler(scene_id="scene_001")

        handler.track_failure("item_1", ValueError("Invalid value"))
        handler.track_failure("item_2", RuntimeError("Runtime error"))

        assert len(handler.failed_items) == 2
        assert "item_1" in handler.failed_items
        assert "item_2" in handler.failed_items

    def test_get_failed_summary(self):
        """Test getting failure summary."""
        handler = PartialFailureHandler(scene_id="scene_001")

        handler.track_failure("item_1", ValueError("Error 1"))
        handler.track_failure("item_2", ValueError("Error 2"))
        handler.track_failure("item_3", RuntimeError("Error 3"))

        summary = handler.get_summary()
        assert summary["total_failures"] == 3
        assert summary["failure_types"]["ValueError"] == 2
        assert summary["failure_types"]["RuntimeError"] == 1

    def test_should_continue(self):
        """Test should_continue logic."""
        handler = PartialFailureHandler(scene_id="scene_001", max_failures=2)

        assert handler.should_continue()

        handler.track_failure("item_1", ValueError("Error"))
        assert handler.should_continue()

        handler.track_failure("item_2", ValueError("Error"))
        assert not handler.should_continue()


class TestDeadLetterMessageSerialization:
    """Test DeadLetterMessage serialization."""

    def test_message_to_dict(self):
        """Test converting message to dict."""
        message = DeadLetterMessage(
            scene_id="scene_001",
            job_type="test",
            error_type="ValueError",
            error_message="Test error",
        )

        message_dict = message.to_dict()
        assert message_dict["scene_id"] == "scene_001"
        assert message_dict["job_type"] == "test"
        assert message_dict["error_type"] == "ValueError"

    def test_message_from_dict(self):
        """Test creating message from dict."""
        data = {
            "message_id": "msg_001",
            "scene_id": "scene_001",
            "job_type": "test",
            "error_type": "ValueError",
            "error_message": "Test error",
            "attempt_count": 1,
            "max_attempts": 3,
            "status": "pending",
            "metadata": {},
            "original_payload": {},
            "traceback": None,
            "error_details": {},
            "first_failure_time": "2024-01-01T00:00:00Z",
            "last_failure_time": "2024-01-01T00:00:00Z",
        }

        message = DeadLetterMessage.from_dict(data)
        assert message.scene_id == "scene_001"
        assert message.message_id == "msg_001"

    def test_message_to_json(self):
        """Test converting message to JSON."""
        message = DeadLetterMessage(
            scene_id="scene_001",
            job_type="test",
            error_type="ValueError",
            error_message="Test error",
        )

        json_str = message.to_json()
        assert isinstance(json_str, str)
        assert "scene_id" in json_str
        assert "scene_001" in json_str
