"""Helpers for resolving production-mode flags across the pipeline."""

from __future__ import annotations

import os
from typing import Mapping, Optional

_TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}

_LEGACY_PRODUCTION_FLAGS = (
    "SIMREADY_PRODUCTION_MODE",
    "PRODUCTION_MODE",
    "DATA_QUALITY_LEVEL",
    "ISAAC_SIM_REQUIRED",
    "REQUIRE_REAL_PHYSICS",
    "PRODUCTION",
    "LABS_STAGING",
)


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_VALUES


def resolve_production_mode(env: Optional[Mapping[str, str]] = None) -> bool:
    """Resolve production-mode state from canonical + legacy environment flags.

    Canonical flags:
      - PIPELINE_ENV=production|prod
      - PRODUCTION_MODE=1/true/yes
      - SIMREADY_PRODUCTION_MODE=1/true/yes

    Legacy flags (still honored for episode-generation compatibility):
      - DATA_QUALITY_LEVEL=production
      - ISAAC_SIM_REQUIRED=1/true/yes
      - REQUIRE_REAL_PHYSICS=1/true/yes
      - PRODUCTION=1/true/yes
      - LABS_STAGING=1/true/yes
    """

    env = env or os.environ
    pipeline_env = (env.get("PIPELINE_ENV", "") or "").strip().lower()
    geniesim_env = (env.get("GENIESIM_ENV", "") or "").strip().lower()
    if pipeline_env in {"prod", "production"} or geniesim_env == "production":
        return True

    data_quality_level = (env.get("DATA_QUALITY_LEVEL", "") or "").strip().lower()
    if data_quality_level == "production":
        return True

    for flag in _LEGACY_PRODUCTION_FLAGS:
        if _is_truthy(env.get(flag)):
            return True

    return False
