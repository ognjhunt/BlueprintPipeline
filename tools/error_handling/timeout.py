"""
Timeout utilities for BlueprintPipeline.

Provides timeout handling for long-running operations, particularly
Isaac Sim operations that can hang indefinitely.
"""

from __future__ import annotations

import functools
import logging
import signal
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when an operation times out."""

    def __init__(self, message: str, timeout: float):
        self.timeout = timeout
        super().__init__(f"{message} (timeout: {timeout}s)")


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out", timeout=0)


@contextmanager
def timeout(seconds: float, error_message: str = "Operation timed out"):
    """
    Context manager for timeout handling using SIGALRM.

    Args:
        seconds: Timeout in seconds
        error_message: Error message to include in TimeoutError

    Raises:
        TimeoutError: If operation exceeds timeout

    Example:
        with timeout(10.0, "Physics step timed out"):
            physics_context.step(render=True)

    Note:
        Only works on Unix-like systems. On Windows, use timeout_thread instead.
    """
    # Check if we have signal support
    if not hasattr(signal, 'SIGALRM'):
        logger.warning(
            "SIGALRM not available (Windows?), timeout not enforced. "
            "Consider using timeout_thread instead."
        )
        yield
        return

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(seconds))

    try:
        yield
    except TimeoutError:
        raise TimeoutError(error_message, seconds)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@contextmanager
def timeout_thread(seconds: float, error_message: str = "Operation timed out"):
    """
    Context manager for timeout handling using threading.

    Works on all platforms including Windows. Uses a separate thread to
    monitor timeout, but note that this cannot interrupt blocking operations.

    Args:
        seconds: Timeout in seconds
        error_message: Error message to include in TimeoutError

    Example:
        with timeout_thread(10.0, "API call timed out"):
            result = requests.get(url)
    """
    timer = None
    timed_out = threading.Event()

    def timeout_callback():
        timed_out.set()

    timer = threading.Timer(seconds, timeout_callback)
    timer.start()

    try:
        yield
        if timed_out.is_set():
            raise TimeoutError(error_message, seconds)
    finally:
        if timer:
            timer.cancel()


def with_timeout(
    seconds: float,
    error_message: Optional[str] = None,
    use_thread: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for adding timeout to a function.

    Args:
        seconds: Timeout in seconds
        error_message: Custom error message (defaults to function name)
        use_thread: Use thread-based timeout (cross-platform) instead of signal

    Returns:
        Decorated function with timeout

    Example:
        @with_timeout(10.0)
        def slow_operation():
            time.sleep(20)  # Will timeout after 10s
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            msg = error_message or f"{func.__name__} timed out"

            timeout_ctx = timeout_thread if use_thread else timeout

            with timeout_ctx(seconds, msg):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class TimeoutManager:
    """
    Manages timeouts for multiple operations with different timeout values.

    Example:
        manager = TimeoutManager()
        manager.set("physics_step", 10.0)
        manager.set("render", 30.0)

        with manager("physics_step"):
            physics_context.step()
    """

    def __init__(self, default_timeout: float = 60.0):
        self.default_timeout = default_timeout
        self._timeouts: dict[str, float] = {}
        self._stats: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def set(self, operation: str, timeout_seconds: float) -> None:
        """Set timeout for a specific operation."""
        with self._lock:
            self._timeouts[operation] = timeout_seconds

    def get(self, operation: str) -> float:
        """Get timeout for a specific operation."""
        with self._lock:
            return self._timeouts.get(operation, self.default_timeout)

    def __call__(self, operation: str):
        """Use as context manager."""
        timeout_seconds = self.get(operation)
        return timeout(timeout_seconds, f"{operation} timed out")

    def record(self, operation: str, duration: float) -> None:
        """Record execution duration for an operation."""
        with self._lock:
            if operation not in self._stats:
                self._stats[operation] = []
            self._stats[operation].append(duration)

    def get_stats(self, operation: str) -> dict[str, Any]:
        """Get statistics for an operation."""
        with self._lock:
            durations = self._stats.get(operation, [])
            if not durations:
                return {"count": 0}

            return {
                "count": len(durations),
                "mean": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "timeout": self.get(operation),
            }


@contextmanager
def monitored_timeout(
    operation: str,
    timeout_seconds: float,
    manager: Optional[TimeoutManager] = None,
):
    """
    Context manager that enforces timeout and records execution time.

    Args:
        operation: Name of the operation
        timeout_seconds: Timeout in seconds
        manager: Optional TimeoutManager to record stats

    Example:
        with monitored_timeout("physics_step", 10.0, manager):
            physics_context.step()
    """
    start = time.time()

    with timeout(timeout_seconds, f"{operation} timed out"):
        try:
            yield
        finally:
            duration = time.time() - start
            if manager:
                manager.record(operation, duration)

            logger.debug(
                f"{operation} completed in {duration:.2f}s "
                f"(timeout: {timeout_seconds}s)"
            )
