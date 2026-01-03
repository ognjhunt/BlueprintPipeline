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

from .errors import (
    PipelineError,
    SceneProcessingError,
    EpisodeGenerationError,
    ValidationError,
    ExternalServiceError,
    ResourceExhaustedError,
    ConfigurationError,
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
    # Errors
    "PipelineError",
    "SceneProcessingError",
    "EpisodeGenerationError",
    "ValidationError",
    "ExternalServiceError",
    "ResourceExhaustedError",
    "ConfigurationError",
]
