"""Shared logging configuration utilities."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional


DEFAULT_JSON_ENV_KEYS = ("LOG_JSON", "LOG_FORMAT")


def _env_truthy(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "json"}:
        return True
    if normalized in {"0", "false", "no", "off", "plain", "text"}:
        return False
    return None


def _should_use_json(env: dict[str, str]) -> bool:
    for key in DEFAULT_JSON_ENV_KEYS:
        if key in env:
            parsed = _env_truthy(env.get(key))
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
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "job_id": _resolve_context_value(record, "job_id", "JOB_ID"),
            "scene_id": _resolve_context_value(record, "scene_id", "SCENE_ID"),
            "request_id": _resolve_context_value(record, "request_id", "REQUEST_ID"),
            "message": message,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


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
