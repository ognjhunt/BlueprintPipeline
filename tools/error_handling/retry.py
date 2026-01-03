"""
Retry Utilities for BlueprintPipeline.

Provides decorators and utilities for automatic retry with exponential backoff,
jitter, and intelligent error classification.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Set, Type, TypeVar, Union

from .errors import PipelineError, ErrorContext, classify_exception

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryableError(Exception):
    """Marker exception indicating the error is retryable."""
    pass


class NonRetryableError(Exception):
    """Marker exception indicating the error should NOT be retried."""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    # Maximum number of retry attempts
    max_retries: int = 3

    # Base delay between retries (seconds)
    base_delay: float = 1.0

    # Maximum delay between retries (seconds)
    max_delay: float = 60.0

    # Exponential backoff multiplier
    backoff_factor: float = 2.0

    # Add random jitter to prevent thundering herd
    jitter: bool = True
    jitter_range: tuple[float, float] = (0.5, 1.5)

    # Exceptions that should always be retried
    retryable_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {
            ConnectionError,
            TimeoutError,
            RetryableError,
        }
    )

    # Exceptions that should never be retried
    non_retryable_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {
            KeyboardInterrupt,
            SystemExit,
            NonRetryableError,
            ValueError,
            TypeError,
        }
    )

    # Callback for logging retry attempts
    on_retry: Optional[Callable[[int, Exception, float], None]] = None

    # Callback for final failure
    on_failure: Optional[Callable[[int, Exception], None]] = None

    # Context for error tracking
    context: Optional[ErrorContext] = None


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay for a given retry attempt."""
    delay = config.base_delay * (config.backoff_factor ** (attempt - 1))
    delay = min(delay, config.max_delay)

    if config.jitter:
        jitter_min, jitter_max = config.jitter_range
        delay *= random.uniform(jitter_min, jitter_max)

    return delay


def should_retry(
    exception: Exception,
    attempt: int,
    config: RetryConfig,
) -> bool:
    """Determine if an exception should be retried."""
    # Check max retries
    if attempt >= config.max_retries:
        return False

    # Check non-retryable exceptions
    for exc_type in config.non_retryable_exceptions:
        if isinstance(exception, exc_type):
            return False

    # Check retryable exceptions
    for exc_type in config.retryable_exceptions:
        if isinstance(exception, exc_type):
            return True

    # Check PipelineError retryable flag
    if isinstance(exception, PipelineError):
        return exception.retryable

    # Default to retrying unknown exceptions
    return True


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    non_retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    on_failure: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for automatic retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_factor: Exponential backoff multiplier
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Additional exceptions to retry
        non_retryable_exceptions: Exceptions to never retry
        on_retry: Callback for retry attempts
        on_failure: Callback for final failure

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_external_api():
            ...
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        on_retry=on_retry,
        on_failure=on_failure,
    )

    if retryable_exceptions:
        config.retryable_exceptions.update(retryable_exceptions)

    if non_retryable_exceptions:
        config.non_retryable_exceptions.update(non_retryable_exceptions)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(1, config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not should_retry(e, attempt, config):
                        logger.warning(
                            f"Non-retryable error in {func.__name__}: {e}"
                        )
                        if config.on_failure:
                            config.on_failure(attempt, e)
                        raise

                    if attempt >= config.max_retries:
                        logger.error(
                            f"Max retries ({config.max_retries}) exceeded for {func.__name__}: {e}"
                        )
                        if config.on_failure:
                            config.on_failure(attempt, e)
                        raise

                    delay = calculate_delay(attempt, config)

                    logger.warning(
                        f"Retry {attempt}/{config.max_retries} for {func.__name__} "
                        f"after {delay:.2f}s: {e}"
                    )

                    if config.on_retry:
                        config.on_retry(attempt, e, delay)

                    time.sleep(delay)

            # Should not reach here, but raise last exception if we do
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    non_retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    on_failure: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Async decorator for automatic retry with exponential backoff.

    Same as retry_with_backoff but for async functions.

    Example:
        @async_retry_with_backoff(max_retries=3)
        async def call_external_api():
            ...
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        on_retry=on_retry,
        on_failure=on_failure,
    )

    if retryable_exceptions:
        config.retryable_exceptions.update(retryable_exceptions)

    if non_retryable_exceptions:
        config.non_retryable_exceptions.update(non_retryable_exceptions)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(1, config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not should_retry(e, attempt, config):
                        logger.warning(
                            f"Non-retryable error in {func.__name__}: {e}"
                        )
                        if config.on_failure:
                            config.on_failure(attempt, e)
                        raise

                    if attempt >= config.max_retries:
                        logger.error(
                            f"Max retries ({config.max_retries}) exceeded for {func.__name__}: {e}"
                        )
                        if config.on_failure:
                            config.on_failure(attempt, e)
                        raise

                    delay = calculate_delay(attempt, config)

                    logger.warning(
                        f"Retry {attempt}/{config.max_retries} for {func.__name__} "
                        f"after {delay:.2f}s: {e}"
                    )

                    if config.on_retry:
                        config.on_retry(attempt, e, delay)

                    await asyncio.sleep(delay)

            if last_exception:
                raise last_exception

        return wrapper

    return decorator


class RetryContext:
    """
    Context manager for retry logic.

    Allows manual control over retry behavior.

    Example:
        retry_ctx = RetryContext(max_retries=3)
        while retry_ctx.should_continue():
            try:
                result = do_something()
                break
            except Exception as e:
                if not retry_ctx.record_failure(e):
                    raise
    """

    def __init__(self, config: Optional[RetryConfig] = None, **kwargs):
        self.config = config or RetryConfig(**kwargs)
        self.attempt = 0
        self.last_exception: Optional[Exception] = None
        self.total_delay = 0.0

    def should_continue(self) -> bool:
        """Check if we should continue trying."""
        return self.attempt < self.config.max_retries

    def record_failure(self, exception: Exception) -> bool:
        """
        Record a failure and determine if we should retry.

        Returns:
            True if should retry, False if should stop
        """
        self.attempt += 1
        self.last_exception = exception

        if not should_retry(exception, self.attempt, self.config):
            return False

        if self.attempt >= self.config.max_retries:
            return False

        delay = calculate_delay(self.attempt, self.config)
        self.total_delay += delay

        if self.config.on_retry:
            self.config.on_retry(self.attempt, exception, delay)

        time.sleep(delay)
        return True

    def get_stats(self) -> dict:
        """Get retry statistics."""
        return {
            "attempts": self.attempt,
            "max_retries": self.config.max_retries,
            "total_delay": self.total_delay,
            "last_exception": str(self.last_exception) if self.last_exception else None,
        }
