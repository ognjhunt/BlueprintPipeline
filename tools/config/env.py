"""Environment variable parsing helpers."""

from __future__ import annotations

from typing import Optional


_TRUTHY_VALUES = {"1", "true", "yes", "y", "on", "json"}
_FALSEY_VALUES = {"0", "false", "no", "off", "plain", "text"}


def parse_bool_env(value: Optional[str], *, default: Optional[bool] = None) -> Optional[bool]:
    """Parse a boolean-like environment variable.

    Args:
        value: Raw environment variable value.
        default: Value to return when the env var is unset or unrecognized.
    """
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in _TRUTHY_VALUES:
        return True
    if normalized in _FALSEY_VALUES:
        return False
    return default
