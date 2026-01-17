"""Helpers for parsing boolean-like environment flags."""

from __future__ import annotations

from typing import Optional


_TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}


def env_flag(value: Optional[str], default: bool = False) -> bool:
    """Return a boolean for a string environment flag.

    Args:
        value: The raw environment variable value (or None).
        default: Value to return when the env var is unset.
    """
    if value is None:
        return default
    return str(value).strip().lower() in _TRUTHY_VALUES
