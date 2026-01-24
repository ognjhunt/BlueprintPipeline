"""Logging helpers for pipeline errors."""

from __future__ import annotations

import logging
from typing import Optional

from tools.error_handling.errors import PipelineError


def log_pipeline_error(
    pipeline_error: PipelineError,
    message: str,
    *,
    logger: Optional[logging.Logger | logging.LoggerAdapter] = None,
    level: int = logging.ERROR,
) -> None:
    """Log a pipeline error with structured taxonomy fields."""
    log = logger or logging.getLogger(__name__)
    payload = pipeline_error.to_dict()
    log.log(
        level,
        message,
        extra={
            "pipeline_error": payload,
            "error_category": payload.get("category"),
            "error_severity": payload.get("severity"),
            "error_context": payload.get("context"),
        },
    )
