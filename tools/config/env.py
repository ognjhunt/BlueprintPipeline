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


def parse_int_env(
    value: Optional[str],
    *,
    default: Optional[int] = None,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    name: str = "environment variable",
) -> Optional[int]:
    """Parse an integer-like environment variable with optional bounds.

    Args:
        value: Raw environment variable value.
        default: Value to return when the env var is unset.
        min_value: Optional minimum allowed value.
        max_value: Optional maximum allowed value.
        name: Name of the env var for error messages.
    """
    if value is None:
        return default
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer (got {value!r}).") from exc

    if min_value is not None and parsed < min_value:
        raise ValueError(
            f"{name} must be >= {min_value} (got {parsed})."
        )
    if max_value is not None and parsed > max_value:
        raise ValueError(
            f"{name} must be <= {max_value} (got {parsed})."
        )
    return parsed
