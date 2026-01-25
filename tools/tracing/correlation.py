"""Correlation helpers for request IDs."""

from __future__ import annotations

import os
import uuid
from contextvars import ContextVar
from typing import Optional


_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Return the current request ID from context, if available."""
    return _request_id.get()


def ensure_request_id() -> str:
    """Ensure a request ID is set in context, generating one if needed."""
    current = _request_id.get()
    if current:
        return current

    env_request_id = os.getenv("REQUEST_ID")
    if env_request_id:
        _request_id.set(env_request_id)
        return env_request_id

    generated = str(uuid.uuid4())
    _request_id.set(generated)
    return generated
