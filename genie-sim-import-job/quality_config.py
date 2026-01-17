from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class QualityConfig:
    default_min_quality_score: float
    min_allowed: float
    max_allowed: float
    description: str = ""


def load_quality_config(config_path: Optional[Path] = None) -> QualityConfig:
    path = config_path or Path(__file__).with_name("quality_config.json")
    payload = json.loads(path.read_text())
    return QualityConfig(
        default_min_quality_score=float(payload["default_min_quality_score"]),
        min_allowed=float(payload["min_allowed"]),
        max_allowed=float(payload["max_allowed"]),
        description=str(payload.get("description", "")),
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


QUALITY_CONFIG = load_quality_config()
DEFAULT_MIN_QUALITY_SCORE = QUALITY_CONFIG.default_min_quality_score
