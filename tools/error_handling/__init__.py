"""
BlueprintPipeline Error Handling Module.

Provides robust error handling for production pipelines:
- Retry decorators with exponential backoff
- Dead letter queue integration
- Circuit breaker pattern for failing dependencies
- Structured error logging and alerting

Usage:
    from tools.error_handling import (
        retry_with_backoff,
        CircuitBreaker,
        DeadLetterQueue,
        PipelineError,
    )

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def process_scene(scene_id: str):
        ...

    breaker = CircuitBreaker("external_service", failure_threshold=5)
    with breaker:
        call_external_service()
"""

from .retry import (
    retry_with_backoff,
    RetryConfig,
    RetryableError,
    NonRetryableError,
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
)

from .dead_letter import (
    DeadLetterQueue,
    DeadLetterMessage,
    GCSDeadLetterQueue,
    PubSubDeadLetterQueue,
)

from .job_wrapper import (
    publish_failure,
    run_job_with_dead_letter_queue,
)

from .errors import (
    PipelineError,
    SceneProcessingError,
    EpisodeGenerationError,
    ValidationError,
    ExternalServiceError,
    ResourceExhaustedError,
    ConfigurationError,
)

from .timeout import (
    TimeoutError,
    timeout,
    timeout_thread,
    with_timeout,
    TimeoutManager,
    monitored_timeout,
)

from .partial_failure import (
    PartialFailureResult,
    PartialFailureError,
    PartialFailureHandler,
    process_with_partial_failure,
    save_successful_items,
)

__all__ = [
    # Retry
    "retry_with_backoff",
    "RetryConfig",
    "RetryableError",
    "NonRetryableError",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "CircuitState",
    # Dead Letter Queue
    "DeadLetterQueue",
    "DeadLetterMessage",
    "GCSDeadLetterQueue",
    "PubSubDeadLetterQueue",
    "publish_failure",
    "run_job_with_dead_letter_queue",
    # Errors
    "PipelineError",
    "SceneProcessingError",
    "EpisodeGenerationError",
    "ValidationError",
    "ExternalServiceError",
    "ResourceExhaustedError",
    "ConfigurationError",
    # Timeout
    "TimeoutError",
    "timeout",
    "timeout_thread",
    "with_timeout",
    "TimeoutManager",
    "monitored_timeout",
    # Partial Failure
    "PartialFailureResult",
    "PartialFailureError",
    "PartialFailureHandler",
    "process_with_partial_failure",
    "save_successful_items",
]
