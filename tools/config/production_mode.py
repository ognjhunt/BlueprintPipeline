"""Helpers for resolving production-mode flags across the pipeline."""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Mapping, Optional, Sequence, Tuple

_TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}
_FALSY_VALUES = {"0", "false", "no", "n", "off"}

_LEGACY_PRODUCTION_FLAG_REMOVAL_DATE = "2025-12-31"

_LEGACY_PRODUCTION_FLAGS: Tuple[str, ...] = ()

_LEGACY_PRODUCTION_ENV_VALUES: Mapping[str, Sequence[str]] = {}

_DEPRECATED_PRODUCTION_FLAGS = (
    "PRODUCTION_MODE",
    "SIMREADY_PRODUCTION_MODE",
    "PRODUCTION",
    "LABS_STAGING",
)

_DEPRECATED_PRODUCTION_ENV_NAMES = (
    "GENIESIM_ENV",
    "BP_ENV",
)

logger = logging.getLogger(__name__)


def _normalize_env_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def resolve_env_with_legacy(
    *,
    canonical_names: Sequence[str],
    legacy_names: Sequence[str] = (),
    env: Optional[Mapping[str, str]] = None,
    default: Optional[str] = None,
    preferred_name: Optional[str] = None,
    log: Optional[logging.Logger] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve an env value from canonical names, falling back to legacy names."""
    source = env or os.environ
    for name in canonical_names:
        value = _normalize_env_value(source.get(name))
        if value is not None:
            return value, name

    if legacy_names:
        canonical_label = preferred_name or (canonical_names[0] if canonical_names else None)
        for name in legacy_names:
            value = _normalize_env_value(source.get(name))
            if value is not None:
                if canonical_label:
                    (log or logger).warning(
                        "%s is deprecated; use %s instead. Planned removal after %s.",
                        name,
                        canonical_label,
                        _LEGACY_PRODUCTION_FLAG_REMOVAL_DATE,
                    )
                else:
                    (log or logger).warning(
                        "%s is deprecated. Planned removal after %s.",
                        name,
                        _LEGACY_PRODUCTION_FLAG_REMOVAL_DATE,
                    )
                return value, name

    return default, None


def resolve_pipeline_environment(
    env: Optional[Mapping[str, str]] = None,
    default: str = "development",
    log: Optional[logging.Logger] = None,
) -> str:
    """Resolve pipeline environment from PIPELINE_ENV only."""
    source = env or os.environ
    _enforce_legacy_production_flag_removal(source, log=log)
    value = _normalize_env_value(source.get("PIPELINE_ENV")) or default
    return (value or default).strip().lower()


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_VALUES


def _is_falsy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _FALSY_VALUES


def _set_env_value(env: Mapping[str, str], key: str, value: str) -> None:
    if env is os.environ:
        os.environ[key] = value
        return
    if hasattr(env, "__setitem__"):
        try:
            env[key] = value
            return
        except TypeError:
            pass
    os.environ[key] = value


def is_config_audit_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    source = env or os.environ
    return _is_truthy(source.get("BP_ENABLE_CONFIG_AUDIT"))


def ensure_config_audit_for_production(
    env: Optional[Mapping[str, str]] = None,
    log: Optional[logging.Logger] = None,
) -> bool:
    source = env or os.environ
    if resolve_production_mode(source):
        audit_value = _normalize_env_value(source.get("BP_ENABLE_CONFIG_AUDIT"))
        if audit_value is None:
            _set_env_value(source, "BP_ENABLE_CONFIG_AUDIT", "1")
            return True
        if _is_falsy(audit_value):
            return False
        _set_env_value(source, "BP_ENABLE_CONFIG_AUDIT", "1")
        return True
    return is_config_audit_enabled(source)


def _legacy_flag_cutoff_passed(today: Optional[date] = None) -> bool:
    cutoff = date.fromisoformat(_LEGACY_PRODUCTION_FLAG_REMOVAL_DATE)
    return (today or date.today()) > cutoff


def _find_legacy_production_settings(source: Mapping[str, str]) -> Tuple[str, ...]:
    found = []
    for name in _DEPRECATED_PRODUCTION_ENV_NAMES:
        if _normalize_env_value(source.get(name)) is not None:
            found.append(name)
    for name in _DEPRECATED_PRODUCTION_FLAGS:
        if _normalize_env_value(source.get(name)) is not None:
            found.append(name)
    return tuple(found)


def _enforce_legacy_production_flag_removal(
    source: Mapping[str, str],
    log: Optional[logging.Logger] = None,
) -> None:
    legacy_names = _find_legacy_production_settings(source)
    if not legacy_names:
        return
    if _legacy_flag_cutoff_passed():
        raise RuntimeError(
            "Legacy production flags are no longer supported after %s. "
            "Remove %s and set PIPELINE_ENV=production instead."
            % (_LEGACY_PRODUCTION_FLAG_REMOVAL_DATE, ", ".join(legacy_names))
        )
    (log or logger).warning(
        "Legacy production flags are deprecated and will be removed after %s. "
        "Remove %s and set PIPELINE_ENV=production instead.",
        _LEGACY_PRODUCTION_FLAG_REMOVAL_DATE,
        ", ".join(legacy_names),
    )


def resolve_production_mode_detail(
    env: Optional[Mapping[str, str]] = None,
    log: Optional[logging.Logger] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Resolve production-mode state and return the source env/value used."""
    source = env or os.environ
    _enforce_legacy_production_flag_removal(source, log=log)
    pipeline_env = _normalize_env_value(source.get("PIPELINE_ENV"))
    if (pipeline_env or "").strip().lower() in {"prod", "production"}:
        return True, "PIPELINE_ENV", pipeline_env
    return False, None, None


def resolve_production_mode(env: Optional[Mapping[str, str]] = None) -> bool:
    """Resolve production-mode state from PIPELINE_ENV.

    Canonical flag:
      - PIPELINE_ENV=production|prod
    Legacy production flags are no longer honored and will raise an error after
    _LEGACY_PRODUCTION_FLAG_REMOVAL_DATE.
    """
    return resolve_production_mode_detail(env=env, log=logger)[0]
