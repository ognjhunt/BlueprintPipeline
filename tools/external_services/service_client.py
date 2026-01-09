"""
Unified external service client wrapper with retry, circuit breaker, and timeout.

This module provides a robust wrapper for all external service calls with:
- Automatic retry with exponential backoff
- Circuit breaker pattern to prevent cascading failures
- Timeout handling
- Rate limiting
- Structured error handling
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, Type, TypeVar

import requests

from tools.error_handling.retry import retry_with_backoff, RetryConfig
from tools.error_handling.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from tools.error_handling.timeout import timeout_thread
from tools.error_handling.errors import PipelineError

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ServiceClientConfig:
    """Configuration for external service client."""

    # Service name for logging and circuit breaker
    service_name: str

    # Retry configuration
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0

    # Circuit breaker configuration
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: float = 30.0

    # Timeout configuration
    default_timeout: float = 30.0

    # Rate limiting
    rate_limit_enabled: bool = False
    calls_per_second: float = 10.0

    # HTTP status codes to retry
    retryable_status_codes: Set[int] = field(
        default_factory=lambda: {408, 429, 500, 502, 503, 504}
    )

    # Additional retryable exceptions
    retryable_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        }
    )


class ServiceClient:
    """
    Unified external service client with resilience patterns.

    Example:
        # Create client for Gemini API
        client = ServiceClient(ServiceClientConfig(
            service_name="gemini_api",
            max_retries=3,
            circuit_breaker_enabled=True,
        ))

        # Make request with automatic retry, circuit breaker, timeout
        response = client.call(
            func=lambda: requests.post(url, json=data),
            timeout=10.0,
        )
    """

    def __init__(self, config: ServiceClientConfig):
        self.config = config
        self.service_name = config.service_name

        # Setup circuit breaker
        if config.circuit_breaker_enabled:
            self.circuit_breaker = CircuitBreaker(
                name=f"{config.service_name}_circuit",
                failure_threshold=config.failure_threshold,
                recovery_timeout=config.recovery_timeout,
            )
        else:
            self.circuit_breaker = None

        # Setup rate limiter
        if config.rate_limit_enabled:
            self.rate_limiter = RateLimiter(config.calls_per_second)
        else:
            self.rate_limiter = None

        self._call_count = 0
        self._error_count = 0
        self._total_duration = 0.0

    def call(
        self,
        func: Callable[[], T],
        timeout: Optional[float] = None,
        operation_name: Optional[str] = None,
    ) -> T:
        """
        Execute a function call with retry, circuit breaker, and timeout.

        Args:
            func: Function to call
            timeout: Timeout in seconds (uses default if not specified)
            operation_name: Name for logging (uses service name if not specified)

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpen: If circuit breaker is open
            TimeoutError: If operation times out
            Exception: If all retries exhausted

        Example:
            result = client.call(
                func=lambda: requests.get("https://api.example.com/data"),
                timeout=10.0,
                operation_name="fetch_data",
            )
        """
        operation = operation_name or self.service_name
        timeout_seconds = timeout or self.config.default_timeout

        # Apply rate limiting
        if self.rate_limiter:
            self.rate_limiter.acquire()

        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.allow_request():
            time_until_retry = self.circuit_breaker.get_time_until_retry()
            raise CircuitBreakerOpen(self.circuit_breaker.name, time_until_retry)

        # Execute with retry and timeout
        start_time = time.time()

        try:
            result = self._execute_with_retry(func, timeout_seconds, operation)
            duration = time.time() - start_time

            # Record success
            self._call_count += 1
            self._total_duration += duration

            if self.circuit_breaker:
                self.circuit_breaker.record_success()

            logger.debug(
                f"{operation} succeeded in {duration:.2f}s",
                extra={
                    "service": self.service_name,
                    "operation": operation,
                    "duration": duration,
                },
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            self._error_count += 1

            if self.circuit_breaker:
                self.circuit_breaker.record_failure(e)

            logger.error(
                f"{operation} failed after {duration:.2f}s: {e}",
                extra={
                    "service": self.service_name,
                    "operation": operation,
                    "duration": duration,
                    "error": str(e),
                },
            )

            raise

    def _execute_with_retry(
        self,
        func: Callable[[], T],
        timeout_seconds: float,
        operation: str,
    ) -> T:
        """Execute function with retry logic."""
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                # Execute with timeout
                with timeout_thread(timeout_seconds, f"{operation} timed out"):
                    result = func()

                # Check if result is HTTP response with retryable status
                if hasattr(result, 'status_code'):
                    if result.status_code in self.config.retryable_status_codes:
                        raise requests.exceptions.HTTPError(
                            f"HTTP {result.status_code}",
                            response=result,
                        )

                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry
                should_retry = self._should_retry(e, attempt)

                if not should_retry or attempt >= self.config.max_retries:
                    raise

                # Calculate delay
                delay = self._calculate_delay(attempt)

                logger.warning(
                    f"Retry {attempt}/{self.config.max_retries} for {operation} "
                    f"after {delay:.2f}s: {e}",
                    extra={
                        "service": self.service_name,
                        "operation": operation,
                        "attempt": attempt,
                        "max_retries": self.config.max_retries,
                        "delay": delay,
                        "error": str(e),
                    },
                )

                time.sleep(delay)

        # Should not reach here
        if last_exception:
            raise last_exception

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if we should retry this exception."""
        # Never retry these
        if isinstance(exception, (KeyboardInterrupt, SystemExit)):
            return False

        # Check if it's a retryable exception type
        for exc_type in self.config.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True

        # Check HTTP status codes
        if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            if exception.response.status_code in self.config.retryable_status_codes:
                return True

        # Default: retry unknown exceptions
        return True

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        import random

        delay = self.config.base_delay * (self.config.backoff_factor ** (attempt - 1))
        delay = min(delay, self.config.max_delay)

        # Add jitter
        delay *= random.uniform(0.5, 1.5)

        return delay

    def get_stats(self) -> Dict[str, Any]:
        """Get service client statistics."""
        stats = {
            "service_name": self.service_name,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._call_count - self._error_count) / self._call_count
                if self._call_count > 0
                else 0.0
            ),
            "avg_duration": (
                self._total_duration / self._call_count
                if self._call_count > 0
                else 0.0
            ),
        }

        if self.circuit_breaker:
            stats["circuit_breaker"] = self.circuit_breaker.get_stats()

        return stats


class RateLimiter:
    """
    Token bucket rate limiter.

    Example:
        limiter = RateLimiter(calls_per_second=10.0)
        limiter.acquire()  # Blocks until token available
    """

    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.tokens = calls_per_second
        self.last_update = time.time()
        self._lock = __import__('threading').Lock()

    def acquire(self, tokens: int = 1) -> None:
        """Block until tokens available."""
        with self._lock:
            while True:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens += elapsed * self.calls_per_second
                self.tokens = min(self.tokens, self.calls_per_second)
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                time.sleep(0.1)


# Pre-configured clients for common services
def create_gemini_client() -> ServiceClient:
    """Create a pre-configured client for Gemini API."""
    return ServiceClient(ServiceClientConfig(
        service_name="gemini_api",
        max_retries=3,
        base_delay=2.0,
        circuit_breaker_enabled=True,
        failure_threshold=5,
        recovery_timeout=60.0,
        default_timeout=30.0,
        rate_limit_enabled=True,
        calls_per_second=5.0,
    ))


def create_genie_sim_client() -> ServiceClient:
    """Create a pre-configured client for Genie Sim API."""
    return ServiceClient(ServiceClientConfig(
        service_name="genie_sim_api",
        max_retries=5,
        base_delay=2.0,
        backoff_factor=2.0,
        circuit_breaker_enabled=True,
        failure_threshold=5,
        recovery_timeout=60.0,
        default_timeout=60.0,
        rate_limit_enabled=True,
        calls_per_second=10.0,
    ))


def create_gcs_client() -> ServiceClient:
    """Create a pre-configured client for Google Cloud Storage."""
    return ServiceClient(ServiceClientConfig(
        service_name="gcs",
        max_retries=3,
        base_delay=1.0,
        circuit_breaker_enabled=False,  # GCS has built-in retry
        default_timeout=120.0,
    ))


def create_particulate_client() -> ServiceClient:
    """Create a pre-configured client for Particulate service."""
    return ServiceClient(ServiceClientConfig(
        service_name="particulate",
        max_retries=3,
        base_delay=1.0,
        circuit_breaker_enabled=True,
        failure_threshold=3,
        recovery_timeout=30.0,
        default_timeout=15.0,
    ))
