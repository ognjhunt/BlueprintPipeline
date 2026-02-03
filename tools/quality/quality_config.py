from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from tools.config.env import parse_bool_env

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUALITY_CONFIG_PATH = REPO_ROOT / "genie-sim-import-job" / "quality_config.json"


@dataclass(frozen=True)
class DiversityDivisors:
    """Robot-specific divisors for action/observation diversity scoring."""

    action: float = 0.05
    obs: float = 0.05


@dataclass(frozen=True)
class FrameCountScoring:
    """Configuration for frame count scoring (gradual vs cliff)."""

    min_frames_full_score: int = 10
    min_frames_nonzero: int = 3
    use_gradual_scoring: bool = True


@dataclass(frozen=True)
class QualityConfig:
    default_min_quality_score: float
    min_allowed: float
    max_allowed: float
    dimension_thresholds: Mapping[str, float] = field(default_factory=dict)
    diversity_divisors: Mapping[str, DiversityDivisors] = field(default_factory=dict)
    frame_count_scoring: FrameCountScoring = field(default_factory=FrameCountScoring)
    default_filter_low_quality: bool = True
    description: str = ""
    source_path: Optional[str] = None

    def get_diversity_divisors(self, robot_type: str) -> DiversityDivisors:
        """Get diversity divisors for a robot type, with fallback to default."""
        robot_key = robot_type.lower() if robot_type else "default"
        if robot_key in self.diversity_divisors:
            return self.diversity_divisors[robot_key]
        return self.diversity_divisors.get("default", DiversityDivisors())


@dataclass(frozen=True)
class ResolvedQualitySettings:
    min_quality_score: float
    filter_low_quality: bool
    dimension_thresholds: Mapping[str, float]
    config: QualityConfig


def _parse_diversity_divisors(
    payload: Mapping[str, Any]
) -> Mapping[str, DiversityDivisors]:
    """Parse diversity_divisors from config payload."""
    raw = payload.get("diversity_divisors", {})
    if not isinstance(raw, Mapping):
        return {"default": DiversityDivisors()}
    result: dict[str, DiversityDivisors] = {}
    for robot_key, divisors in raw.items():
        if isinstance(divisors, Mapping):
            result[str(robot_key).lower()] = DiversityDivisors(
                action=float(divisors.get("action", 0.05)),
                obs=float(divisors.get("obs", 0.05)),
            )
    if "default" not in result:
        result["default"] = DiversityDivisors()
    return result


def _parse_frame_count_scoring(payload: Mapping[str, Any]) -> FrameCountScoring:
    """Parse frame_count_scoring from config payload."""
    raw = payload.get("frame_count_scoring", {})
    if not isinstance(raw, Mapping):
        return FrameCountScoring()
    return FrameCountScoring(
        min_frames_full_score=int(raw.get("min_frames_full_score", 10)),
        min_frames_nonzero=int(raw.get("min_frames_nonzero", 3)),
        use_gradual_scoring=bool(raw.get("use_gradual_scoring", True)),
    )


def _parse_dimension_thresholds(payload: Mapping[str, Any]) -> Mapping[str, float]:
    raw = payload.get("dimension_thresholds", {})
    if not isinstance(raw, Mapping):
        return {}
    thresholds: dict[str, float] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, bool):
            raise ValueError(
                f"dimension_thresholds[{key!r}] must be a number, not a boolean."
            )
        try:
            thresholds[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"dimension_thresholds[{key!r}] must be a number (got {value!r})."
            ) from exc
    return thresholds


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
        dimension_thresholds=_parse_dimension_thresholds(payload),
        diversity_divisors=_parse_diversity_divisors(payload),
        frame_count_scoring=_parse_frame_count_scoring(payload),
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
        dimension_thresholds=resolved_config.dimension_thresholds,
        config=resolved_config,
    )


QUALITY_CONFIG = load_quality_config()
DEFAULT_MIN_QUALITY_SCORE = QUALITY_CONFIG.default_min_quality_score
DEFAULT_FILTER_LOW_QUALITY = QUALITY_CONFIG.default_filter_low_quality
