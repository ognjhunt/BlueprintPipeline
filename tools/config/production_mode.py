"""Helpers for resolving production-mode flags across the pipeline."""

from __future__ import annotations

import logging
import os
from typing import Mapping, Optional, Sequence, Tuple

_TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}
_FALSY_VALUES = {"0", "false", "no", "n", "off"}

_LEGACY_PRODUCTION_FLAGS = (
    "PRODUCTION_MODE",
    "SIMREADY_PRODUCTION_MODE",
    "ISAAC_SIM_REQUIRED",
    "REQUIRE_REAL_PHYSICS",
    "PRODUCTION",
    "LABS_STAGING",
)

_LEGACY_PRODUCTION_ENVS = ("BP_ENV",)

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
                        "%s is deprecated; use %s instead.",
                        name,
                        canonical_label,
                    )
                else:
                    (log or logger).warning("%s is deprecated.", name)
                return value, name

    return default, None


def resolve_pipeline_environment(
    env: Optional[Mapping[str, str]] = None,
    default: str = "development",
    log: Optional[logging.Logger] = None,
) -> str:
    """Resolve pipeline environment from canonical/legacy envs."""
    value, _ = resolve_env_with_legacy(
        canonical_names=("PIPELINE_ENV", "GENIESIM_ENV"),
        legacy_names=("BP_ENV",),
        env=env,
        default=default,
        preferred_name="PIPELINE_ENV",
        log=log,
    )
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


def resolve_production_mode(env: Optional[Mapping[str, str]] = None) -> bool:
    """Resolve production-mode state from canonical + legacy environment flags.

    Canonical flags (preferred):
      - PIPELINE_ENV=production|prod
      - GENIESIM_ENV=production|prod (GenieSim integrations)

    Legacy flags (deprecated, still honored for compatibility):
      - BP_ENV=production|prod
      - PRODUCTION_MODE=1/true/yes
      - SIMREADY_PRODUCTION_MODE=1/true/yes
      - DATA_QUALITY_LEVEL=production
      - ISAAC_SIM_REQUIRED=1/true/yes
      - REQUIRE_REAL_PHYSICS=1/true/yes
      - PRODUCTION=1/true/yes
      - LABS_STAGING=1/true/yes
    """

    env = env or os.environ
    pipeline_env, _ = resolve_env_with_legacy(
        canonical_names=("PIPELINE_ENV", "GENIESIM_ENV"),
        legacy_names=_LEGACY_PRODUCTION_ENVS,
        env=env,
        preferred_name="PIPELINE_ENV",
        log=logger,
    )
    if (pipeline_env or "").strip().lower() in {"prod", "production"}:
        return True

    data_quality_level = (env.get("DATA_QUALITY_LEVEL", "") or "").strip().lower()
    if data_quality_level == "production":
        return True

    for flag in _LEGACY_PRODUCTION_FLAGS:
        if _is_truthy(env.get(flag)):
            return True

    return False
