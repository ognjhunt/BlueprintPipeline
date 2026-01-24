"""Shared logging configuration utilities."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from tools.config.env import parse_bool_env


DEFAULT_JSON_ENV_KEYS = ("LOG_JSON", "LOG_FORMAT")


def _should_use_json(env: dict[str, str]) -> bool:
    for key in DEFAULT_JSON_ENV_KEYS:
        if key in env:
            parsed = parse_bool_env(env.get(key))
            if parsed is not None:
                return parsed
    return True


def _resolve_context_value(record: logging.LogRecord, key: str, env_key: str) -> str:
    if hasattr(record, key):
        value = getattr(record, key)
        if value is not None:
            return str(value)
    env_value = os.getenv(env_key)
    return env_value if env_value is not None else ""


class JsonLogFormatter(logging.Formatter):
    """Formats logs as structured JSON with standard fields."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        pipeline_error = self._resolve_pipeline_error(record)
        error_category = getattr(record, "error_category", None)
        error_severity = getattr(record, "error_severity", None)
        error_context = getattr(record, "error_context", None)
        if pipeline_error:
            error_category = error_category or pipeline_error.get("category")
            error_severity = error_severity or pipeline_error.get("severity")
            error_context = error_context or pipeline_error.get("context")
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "job_id": _resolve_context_value(record, "job_id", "JOB_ID"),
            "scene_id": _resolve_context_value(record, "scene_id", "SCENE_ID"),
            "request_id": _resolve_context_value(record, "request_id", "REQUEST_ID"),
            "message": message,
        }
        if pipeline_error is not None:
            payload["pipeline_error"] = pipeline_error
        if error_category is not None:
            payload["error_category"] = self._serialize_enum(error_category)
        if error_severity is not None:
            payload["error_severity"] = self._serialize_enum(error_severity)
        if error_context is not None:
            payload["error_context"] = error_context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)

    @staticmethod
    def _serialize_enum(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        return value

    @staticmethod
    def _resolve_pipeline_error(record: logging.LogRecord) -> Optional[dict[str, Any]]:
        pipeline_error = getattr(record, "pipeline_error", None)
        if pipeline_error is None:
            return None
        if isinstance(pipeline_error, dict):
            return pipeline_error
        to_dict = getattr(pipeline_error, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        return {"detail": str(pipeline_error)}


class PlainTextFormatter(logging.Formatter):
    """Human-friendly formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        record.job_id = _resolve_context_value(record, "job_id", "JOB_ID")
        record.scene_id = _resolve_context_value(record, "scene_id", "SCENE_ID")
        record.request_id = _resolve_context_value(record, "request_id", "REQUEST_ID")
        return super().format(record)


def init_logging(
    *,
    level: Optional[int] = None,
    json_enabled: Optional[bool] = None,
    stream: Optional[Any] = None,
) -> None:
    """Initialize shared logging configuration."""

    if level is None:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = logging.getLevelName(level_name)
        if isinstance(level, str):
            level = logging.INFO

    if json_enabled is None:
        json_enabled = _should_use_json(os.environ)

    handler_stream = stream if stream is not None else sys.stdout
    handler = logging.StreamHandler(handler_stream)
    if json_enabled:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            PlainTextFormatter(
                "%(asctime)s %(levelname)s %(module)s "
                "[job_id=%(job_id)s scene_id=%(scene_id)s request_id=%(request_id)s] "
                "%(message)s"
            )
        )

    logging.basicConfig(level=level, handlers=[handler], force=True)
