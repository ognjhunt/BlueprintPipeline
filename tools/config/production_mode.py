"""Helpers for resolving production-mode flags across the pipeline."""

from __future__ import annotations

import os
from typing import Mapping, Optional

_TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}

_LEGACY_PRODUCTION_FLAGS = (
    "PRODUCTION_MODE",
    "SIMREADY_PRODUCTION_MODE",
    "ISAAC_SIM_REQUIRED",
    "REQUIRE_REAL_PHYSICS",
    "PRODUCTION",
    "LABS_STAGING",
)

_LEGACY_PRODUCTION_ENVS = ("BP_ENV",)


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_VALUES


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
    pipeline_env = (env.get("PIPELINE_ENV", "") or "").strip().lower()
    geniesim_env = (env.get("GENIESIM_ENV", "") or "").strip().lower()
    if pipeline_env in {"prod", "production"}:
        return True

    if geniesim_env in {"prod", "production"}:
        return True

    for legacy_env in _LEGACY_PRODUCTION_ENVS:
        legacy_value = (env.get(legacy_env, "") or "").strip().lower()
        if legacy_value in {"prod", "production"}:
            return True

    data_quality_level = (env.get("DATA_QUALITY_LEVEL", "") or "").strip().lower()
    if data_quality_level == "production":
        return True

    for flag in _LEGACY_PRODUCTION_FLAGS:
        if _is_truthy(env.get(flag)):
            return True

    return False
