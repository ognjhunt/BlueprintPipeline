"""
Circuit Breaker Pattern for BlueprintPipeline.

Prevents cascading failures by stopping calls to failing services
after a threshold of failures is reached.

States:
- CLOSED: Normal operation, calls pass through
- OPEN: Service is failing, calls are rejected immediately
- HALF_OPEN: Testing if service has recovered

Usage:
    breaker = CircuitBreaker("gemini_api", failure_threshold=5)

    try:
        with breaker:
            result = call_gemini_api()
    except CircuitBreakerOpen:
        result = use_fallback()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """States for the circuit breaker."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Rejecting calls
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and rejecting calls."""

    def __init__(self, name: str, time_until_retry: float):
        self.name = name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker '{name}' is open. "
            f"Retry in {time_until_retry:.1f}s"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    # Number of failures before opening the circuit
    failure_threshold: int = 5

    # Number of successes required to close circuit from half-open
    success_threshold: int = 2

    # Time to wait before moving from open to half-open (seconds)
    recovery_timeout: float = 30.0

    # Rolling window for tracking failures (seconds)
    failure_window: float = 60.0

    # Exceptions that count as failures
    failure_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {Exception}
    )

    # Exceptions that should NOT count as failures
    excluded_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {KeyboardInterrupt, SystemExit}
    )

    # Callback when circuit opens
    on_open: Optional[Callable[[str, int], None]] = None

    # Callback when circuit closes
    on_close: Optional[Callable[[str], None]] = None

    # Callback when circuit moves to half-open
    on_half_open: Optional[Callable[[str], None]] = None


@dataclass
class FailureRecord:
    """Record of a failure."""
    timestamp: float
    exception_type: str
    message: str


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting against cascading failures.

    Example:
        # Create breaker for external service
        breaker = CircuitBreaker("gemini_api", failure_threshold=5)

        # Use as context manager
        try:
            with breaker:
                result = call_gemini_api()
        except CircuitBreakerOpen as e:
            # Circuit is open, use fallback
            result = use_cached_response()

        # Use as decorator
        @breaker
        def call_gemini_api():
            ...
    """

    # Global registry of circuit breakers
    _registry: Dict[str, "CircuitBreaker"] = {}
    _registry_lock = threading.Lock()

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_timeout: float = 30.0,
        failure_window: float = 60.0,
        persistence_path: Optional[str | Path] = None,
        on_open: Optional[Callable[[str, int], None]] = None,
        on_close: Optional[Callable[[str], None]] = None,
        on_half_open: Optional[Callable[[str], None]] = None,
    ):
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout=recovery_timeout,
            failure_window=failure_window,
            on_open=on_open,
            on_close=on_close,
            on_half_open=on_half_open,
        )

        self._state = CircuitState.CLOSED
        self._failures: List[FailureRecord] = []
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        self._lock = threading.RLock()
        self._persistence_path = Path(persistence_path) if persistence_path else None

        if self._persistence_path:
            self._load_state()

        # Register in global registry
        with CircuitBreaker._registry_lock:
            CircuitBreaker._registry[name] = self

    @classmethod
    def get(cls, name: str) -> Optional["CircuitBreaker"]:
        """Get a circuit breaker by name from the registry."""
        with cls._registry_lock:
            return cls._registry.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, "CircuitBreaker"]:
        """Get all registered circuit breakers."""
        with cls._registry_lock:
            return dict(cls._registry)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN

    def _check_state_transition(self) -> None:
        """Check if we should transition to a new state."""
        now = time.time()

        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._opened_at and (now - self._opened_at) >= self.config.recovery_timeout:
                self._transition_to_half_open()

        # Clean up old failures outside the window
        self._failures = [
            f for f in self._failures
            if (now - f.timestamp) < self.config.failure_window
        ]

    def _transition_to_open(self, failure_count: int) -> None:
        """Transition to open state."""
        self._state = CircuitState.OPEN
        self._opened_at = time.time()
        self._success_count = 0

        logger.warning(
            f"Circuit breaker '{self.name}' opened after {failure_count} failures"
        )

        if self.config.on_open:
            self.config.on_open(self.name, failure_count)
        self._persist_state()

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0

        logger.info(f"Circuit breaker '{self.name}' is now half-open")

        if self.config.on_half_open:
            self.config.on_half_open(self.name)
        self._persist_state()

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self._state = CircuitState.CLOSED
        self._failures.clear()
        self._success_count = 0
        self._opened_at = None

        logger.info(f"Circuit breaker '{self.name}' is now closed")

        if self.config.on_close:
            self.config.on_close(self.name)
        self._persist_state()

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
                else:
                    self._persist_state()

    def record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            # Check if this exception should be counted
            if isinstance(exception, tuple(self.config.excluded_exceptions)):
                return

            is_failure = isinstance(exception, tuple(self.config.failure_exceptions))
            if not is_failure:
                return

            now = time.time()
            self._failures.append(FailureRecord(
                timestamp=now,
                exception_type=type(exception).__name__,
                message=str(exception)[:200],
            ))
            self._last_failure_time = now

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open moves back to open
                self._transition_to_open(len(self._failures))

            elif self._state == CircuitState.CLOSED:
                # Check if we should open
                if len(self._failures) >= self.config.failure_threshold:
                    self._transition_to_open(len(self._failures))
                else:
                    self._persist_state()
            else:
                self._persist_state()

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                return False

            # Half-open: allow request for testing
            return True

    def get_time_until_retry(self) -> float:
        """Get time until the circuit might allow requests."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return 0.0

            if self._opened_at is None:
                return 0.0

            elapsed = time.time() - self._opened_at
            remaining = self.config.recovery_timeout - elapsed
            return max(0.0, remaining)

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to_closed()

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": len(self._failures),
                "failure_threshold": self.config.failure_threshold,
                "success_count": self._success_count,
                "success_threshold": self.config.success_threshold,
                "last_failure_time": self._last_failure_time,
                "opened_at": self._opened_at,
                "time_until_retry": self.get_time_until_retry(),
            }

    def __enter__(self) -> "CircuitBreaker":
        """Context manager entry."""
        if not self.allow_request():
            raise CircuitBreakerOpen(
                self.name,
                self.get_time_until_retry(),
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        if exc_val is None:
            self.record_success()
        else:
            self.record_failure(exc_val)
        return False  # Don't suppress exceptions

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

    def _serialize_state(self) -> Dict[str, Any]:
        return {
            "state": self._state.value,
            "last_failure_time": self._last_failure_time,
            "opened_at": self._opened_at,
            "success_count": self._success_count,
            "failures": [
                {
                    "timestamp": failure.timestamp,
                    "exception_type": failure.exception_type,
                    "message": failure.message,
                }
                for failure in self._failures
            ],
        }

    def _deserialize_state(self, payload: Dict[str, Any]) -> None:
        state_value = payload.get("state", CircuitState.CLOSED.value)
        try:
            self._state = CircuitState(state_value)
        except ValueError:
            logger.warning(
                "Circuit breaker '%s' persisted state '%s' is invalid; defaulting to closed",
                self.name,
                state_value,
            )
            self._state = CircuitState.CLOSED

        self._last_failure_time = payload.get("last_failure_time")
        self._opened_at = payload.get("opened_at")
        self._success_count = int(payload.get("success_count", 0))
        self._failures = [
            FailureRecord(
                timestamp=item.get("timestamp", 0.0),
                exception_type=item.get("exception_type", "Exception"),
                message=item.get("message", ""),
            )
            for item in payload.get("failures", [])
            if isinstance(item, dict)
        ]

        self._check_state_transition()

    def _load_state(self) -> None:
        if not self._persistence_path or not self._persistence_path.exists():
            return
        try:
            with self._persistence_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                with self._lock:
                    self._deserialize_state(payload)
        except Exception:
            logger.exception(
                "Failed to load circuit breaker state from %s", self._persistence_path
            )

    def _persist_state(self) -> None:
        if not self._persistence_path:
            return
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._persistence_path.with_suffix(
                f"{self._persistence_path.suffix}.tmp"
            )
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(self._serialize_state(), handle)
            tmp_path.replace(self._persistence_path)
        except Exception:
            logger.exception(
                "Failed to persist circuit breaker state to %s", self._persistence_path
            )


class CircuitBreakerGroup:
    """
    Manage a group of circuit breakers for coordinated failure handling.

    Example:
        group = CircuitBreakerGroup()
        group.add("gemini", failure_threshold=5)
        group.add("openai", failure_threshold=3)

        # Get overall health
        health = group.get_health()
    """

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def add(self, name: str, **kwargs) -> CircuitBreaker:
        """Add a circuit breaker to the group."""
        with self._lock:
            breaker = CircuitBreaker(name, **kwargs)
            self._breakers[name] = breaker
            return breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_health(self) -> Dict[str, Any]:
        """Get health status of all circuit breakers."""
        with self._lock:
            total = len(self._breakers)
            closed = sum(1 for b in self._breakers.values() if b.is_closed)
            open_count = sum(1 for b in self._breakers.values() if b.is_open)
            half_open = total - closed - open_count

            return {
                "total": total,
                "closed": closed,
                "open": open_count,
                "half_open": half_open,
                "health_percentage": (closed / total * 100) if total > 0 else 100,
                "breakers": {
                    name: breaker.get_stats()
                    for name, breaker in self._breakers.items()
                },
            }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
