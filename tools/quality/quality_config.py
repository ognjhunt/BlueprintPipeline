from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from tools.config.env import parse_bool_env

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUALITY_CONFIG_PATH = REPO_ROOT / "genie-sim-import-job" / "quality_config.json"


@dataclass(frozen=True)
class QualityConfig:
    default_min_quality_score: float
    min_allowed: float
    max_allowed: float
    default_filter_low_quality: bool = True
    description: str = ""
    source_path: Optional[str] = None


@dataclass(frozen=True)
class ResolvedQualitySettings:
    min_quality_score: float
    filter_low_quality: bool
    config: QualityConfig


def load_quality_config(
    config_path: Optional[Path] = None,
    *,
    default_filter_low_quality: bool = True,
) -> QualityConfig:
    path = config_path or DEFAULT_QUALITY_CONFIG_PATH
    payload = json.loads(path.read_text())
    return QualityConfig(
        default_min_quality_score=float(payload["default_min_quality_score"]),
        min_allowed=float(payload["min_allowed"]),
        max_allowed=float(payload["max_allowed"]),
        default_filter_low_quality=default_filter_low_quality,
        description=str(payload.get("description", "")),
        source_path=str(path),
    )


def resolve_min_quality_score(
    env_value: Optional[str],
    config: QualityConfig,
) -> float:
    if env_value is None or env_value == "":
        return config.default_min_quality_score
    try:
        value = float(env_value)
    except ValueError as exc:
        raise ValueError(
            f"MIN_QUALITY_SCORE must be a number (got {env_value!r})"
        ) from exc
    if not config.min_allowed <= value <= config.max_allowed:
        raise ValueError(
            "MIN_QUALITY_SCORE must be between "
            f"{config.min_allowed} and {config.max_allowed} (got {value})"
        )
    return value


def resolve_filter_low_quality(
    env_value: Optional[str],
    config: QualityConfig,
) -> bool:
    parsed = parse_bool_env(env_value, default=config.default_filter_low_quality)
    if parsed is None:
        return config.default_filter_low_quality
    return parsed


def resolve_quality_settings(
    env: Optional[Mapping[str, str]] = None,
    config: Optional[QualityConfig] = None,
) -> ResolvedQualitySettings:
    env_values = os.environ if env is None else env
    resolved_config = config or load_quality_config()
    min_quality_score = resolve_min_quality_score(
        env_values.get("MIN_QUALITY_SCORE"),
        resolved_config,
    )
    filter_low_quality = resolve_filter_low_quality(
        env_values.get("FILTER_LOW_QUALITY"),
        resolved_config,
    )
    return ResolvedQualitySettings(
        min_quality_score=min_quality_score,
        filter_low_quality=filter_low_quality,
        config=resolved_config,
    )


QUALITY_CONFIG = load_quality_config()
DEFAULT_MIN_QUALITY_SCORE = QUALITY_CONFIG.default_min_quality_score
DEFAULT_FILTER_LOW_QUALITY = QUALITY_CONFIG.default_filter_low_quality
