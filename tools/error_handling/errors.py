"""
Custom Exception Classes for BlueprintPipeline.

Provides a hierarchy of exceptions for different failure modes,
enabling appropriate error handling and recovery strategies.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorSeverity(str, Enum):
    """Severity levels for pipeline errors."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Categories of pipeline errors."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    EXTERNAL_SERVICE = "external_service"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    INFRASTRUCTURE = "infrastructure"
    DATA = "data"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for an error."""
    scene_id: Optional[str] = None
    job_id: Optional[str] = None
    step: Optional[str] = None
    attempt: int = 1
    max_attempts: int = 3
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    additional: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "job_id": self.job_id,
            "step": self.step,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "timestamp": self.timestamp,
            "additional": self.additional,
        }


class PipelineError(Exception):
    """
    Base exception for all pipeline errors.

    Provides structured error information for logging, alerting,
    and dead letter queue processing.
    """

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        retryable: bool = True,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.retryable = retryable
        self.context = context or ErrorContext()
        self.cause = cause
        self._debug_enabled = os.getenv("BP_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
        self._path_redaction_regex = re.compile(r"(?:[A-Za-z]:\\\\|/)[^\\s]+")
        self.traceback_str = traceback.format_exc() if cause and self._debug_enabled else None

    def _sanitize_message(self, message: str) -> str:
        if not message:
            return message
        if self._debug_enabled:
            return message
        return self._path_redaction_regex.sub("<redacted-path>", message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize error for logging/storage."""
        debug_enabled = self._debug_enabled
        return {
            "error_type": self.__class__.__name__,
            "message": self._sanitize_message(self.message),
            "category": self.category.value,
            "severity": self.severity.value,
            "retryable": self.retryable,
            "context": self.context.to_dict(),
            "cause": self._sanitize_message(str(self.cause)) if self.cause else None,
            "traceback": self.traceback_str if debug_enabled else None,
        }

    def to_json(self) -> str:
        """Serialize error to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        context: Optional[ErrorContext] = None,
    ) -> "PipelineError":
        """Create a PipelineError from an existing exception."""
        return cls(
            message=str(exc),
            cause=exc,
            context=context,
        )


class SceneProcessingError(PipelineError):
    """Error during scene processing (regen3d, simready, usd-assembly)."""

    def __init__(
        self,
        message: str,
        scene_id: str,
        step: str,
        **kwargs,
    ):
        context = kwargs.pop("context", None) or ErrorContext()
        context.scene_id = scene_id
        context.step = step

        super().__init__(
            message,
            category=ErrorCategory.PROCESSING,
            context=context,
            **kwargs,
        )
        self.scene_id = scene_id
        self.step = step


class EpisodeGenerationError(PipelineError):
    """Error during episode generation."""

    def __init__(
        self,
        message: str,
        scene_id: str,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.pop("context", None) or ErrorContext()
        context.scene_id = scene_id
        context.step = "episode_generation"
        context.additional["episode_id"] = episode_id
        context.additional["task_name"] = task_name

        super().__init__(
            message,
            category=ErrorCategory.PROCESSING,
            context=context,
            **kwargs,
        )
        self.scene_id = scene_id
        self.episode_id = episode_id
        self.task_name = task_name


class ValidationError(PipelineError):
    """Error during validation (schema, physics, quality)."""

    def __init__(
        self,
        message: str,
        validation_type: str = "unknown",
        issues: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            retryable=False,  # Validation errors are typically not retryable
            **kwargs,
        )
        self.validation_type = validation_type
        self.issues = issues or []


class ExternalServiceError(PipelineError):
    """Error from external service (Gemini, Isaac Sim, etc.)."""

    def __init__(
        self,
        message: str,
        service_name: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs,
    ):
        # Most external service errors are retryable
        kwargs.setdefault("retryable", True)

        context = kwargs.pop("context", None) or ErrorContext()
        context.additional["service_name"] = service_name
        context.additional["status_code"] = status_code
        context.additional["response_body"] = response_body[:500] if response_body else None

        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            context=context,
            **kwargs,
        )
        self.service_name = service_name
        self.status_code = status_code
        self.response_body = response_body


class ResourceExhaustedError(PipelineError):
    """Error when resources are exhausted (GPU, memory, quota)."""

    def __init__(
        self,
        message: str,
        resource_type: str,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs,
    ):
        kwargs.setdefault("severity", ErrorSeverity.WARNING)
        kwargs.setdefault("retryable", True)  # Often retryable after waiting

        context = kwargs.pop("context", None) or ErrorContext()
        context.additional["resource_type"] = resource_type
        context.additional["current_usage"] = current_usage
        context.additional["limit"] = limit

        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            context=context,
            **kwargs,
        )
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


class ConfigurationError(PipelineError):
    """Error in configuration (missing env vars, invalid config)."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        **kwargs,
    ):
        kwargs.setdefault("retryable", False)  # Config errors require manual intervention

        context = kwargs.pop("context", None) or ErrorContext()
        context.additional["config_key"] = config_key
        context.additional["expected_type"] = expected_type

        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            **kwargs,
        )
        self.config_key = config_key
        self.expected_type = expected_type


class InfrastructureError(PipelineError):
    """Error in infrastructure (GCS, GKE, networking)."""

    def __init__(
        self,
        message: str,
        component: str,
        **kwargs,
    ):
        context = kwargs.pop("context", None) or ErrorContext()
        context.additional["component"] = component

        super().__init__(
            message,
            category=ErrorCategory.INFRASTRUCTURE,
            context=context,
            **kwargs,
        )
        self.component = component


class DataError(PipelineError):
    """Error in data (corrupt files, missing assets, invalid format)."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        expected_format: Optional[str] = None,
        **kwargs,
    ):
        kwargs.setdefault("retryable", False)  # Data errors usually require fixes

        context = kwargs.pop("context", None) or ErrorContext()
        context.additional["file_path"] = file_path
        context.additional["expected_format"] = expected_format

        super().__init__(
            message,
            category=ErrorCategory.DATA,
            context=context,
            **kwargs,
        )
        self.file_path = file_path
        self.expected_format = expected_format


def classify_exception(exc: Exception) -> PipelineError:
    """
    Classify a generic exception into the appropriate PipelineError type.

    Args:
        exc: The exception to classify

    Returns:
        An appropriate PipelineError subclass
    """
    error_message = str(exc)
    error_type = type(exc).__name__

    # Check for specific error patterns
    if "quota" in error_message.lower() or "rate limit" in error_message.lower():
        return ResourceExhaustedError(
            message=error_message,
            resource_type="api_quota",
            cause=exc,
        )

    if "timeout" in error_message.lower():
        return ExternalServiceError(
            message=error_message,
            service_name="unknown",
            cause=exc,
        )

    if "connection" in error_message.lower() or "network" in error_message.lower():
        return InfrastructureError(
            message=error_message,
            component="network",
            cause=exc,
        )

    if "permission" in error_message.lower() or "access denied" in error_message.lower():
        return ConfigurationError(
            message=error_message,
            cause=exc,
        )

    if "not found" in error_message.lower() or "missing" in error_message.lower():
        return DataError(
            message=error_message,
            cause=exc,
        )

    # Default to generic PipelineError
    return PipelineError.from_exception(exc)
