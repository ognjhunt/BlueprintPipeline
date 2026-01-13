#!/usr/bin/env python3
"""Shared loader for policy configuration values."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_CONFIG_PATH = REPO_ROOT / "policy_configs" / "planning_defaults.yaml"


class PolicyConfigError(ValueError):
    """Raised when policy config is missing or invalid."""


def load_policy_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load the policy configuration YAML file."""
    resolved_path = Path(
        os.getenv("PLANNING_DEFAULTS_CONFIG", str(config_path or DEFAULT_POLICY_CONFIG_PATH))
    )

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Policy config not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise PolicyConfigError("Policy config must be a YAML mapping")

    return data


def _require_section(config: Dict[str, Any], path: Iterable[str]) -> Dict[str, Any]:
    current: Any = config
    section_path = []
    for key in path:
        section_path.append(key)
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(section_path)
            raise PolicyConfigError(f"Missing required config section: {joined}")
        current = current[key]
    if not isinstance(current, dict):
        joined = ".".join(section_path)
        raise PolicyConfigError(f"Config section {joined} must be a mapping")
    return current


def _validate_number(name: str, value: Any, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)):
        raise PolicyConfigError(f"Config value '{name}' must be a number")
    float_value = float(value)
    if minimum is not None and float_value < minimum:
        raise PolicyConfigError(f"Config value '{name}' must be >= {minimum}")
    return float_value


def _validate_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise PolicyConfigError(f"Config value '{name}' must be a boolean")
    return value


def load_motion_planner_timing(config_path: Path | None = None) -> Dict[str, float]:
    config = load_policy_config(config_path)
    timing = _require_section(config, ("motion_planner", "timing"))

    required_keys = [
        "home",
        "approach",
        "pre_grasp",
        "grasp",
        "close_gripper",
        "lift",
        "transport",
        "pre_place",
        "place",
        "release",
        "retract",
        "return_home",
        "articulation_approach",
        "articulation_pre_grasp",
        "articulation_grasp",
        "articulation_close",
        "articulation_motion",
        "articulation_release",
        "articulation_retract",
        "push_home",
        "push_approach",
        "push_lower",
        "push_motion",
        "push_lift",
        "push_return_home",
        "simple_home",
        "simple_approach",
        "simple_reach",
        "simple_return_home",
    ]

    validated: Dict[str, float] = {}
    for key in required_keys:
        if key not in timing:
            raise PolicyConfigError(f"Missing motion_planner.timing value: {key}")
        validated[key] = _validate_number(f"motion_planner.timing.{key}", timing[key], minimum=0.0)

    return validated


def load_validation_thresholds(config_path: Path | None = None) -> Dict[str, Any]:
    config = load_policy_config(config_path)
    thresholds = _require_section(config, ("validation", "thresholds"))
    requirements = _require_section(config, ("validation", "requirements"))
    retries = _require_section(config, ("validation", "retries"))

    validated: Dict[str, Any] = {
        "max_unexpected_collisions": _validate_number(
            "validation.thresholds.max_unexpected_collisions",
            thresholds.get("max_unexpected_collisions"),
            minimum=0.0,
        ),
        "max_joint_violations": _validate_number(
            "validation.thresholds.max_joint_violations",
            thresholds.get("max_joint_violations"),
            minimum=0.0,
        ),
        "max_collision_force": _validate_number(
            "validation.thresholds.max_collision_force",
            thresholds.get("max_collision_force"),
            minimum=0.0,
        ),
        "max_joint_velocity": _validate_number(
            "validation.thresholds.max_joint_velocity",
            thresholds.get("max_joint_velocity"),
            minimum=0.0,
        ),
        "max_joint_acceleration": _validate_number(
            "validation.thresholds.max_joint_acceleration",
            thresholds.get("max_joint_acceleration"),
            minimum=0.0,
        ),
        "stability_threshold": _validate_number(
            "validation.thresholds.stability_threshold",
            thresholds.get("stability_threshold"),
            minimum=0.0,
        ),
        "min_quality_score": _validate_number(
            "validation.thresholds.min_quality_score",
            thresholds.get("min_quality_score"),
            minimum=0.0,
        ),
        "max_retries": int(
            _validate_number(
                "validation.thresholds.max_retries",
                thresholds.get("max_retries"),
                minimum=0.0,
            )
        ),
    }

    validated.update(
        {
            "require_task_success": _validate_bool(
                "validation.requirements.require_task_success",
                requirements.get("require_task_success"),
            ),
            "require_grasp_success": _validate_bool(
                "validation.requirements.require_grasp_success",
                requirements.get("require_grasp_success"),
            ),
            "require_placement_success": _validate_bool(
                "validation.requirements.require_placement_success",
                requirements.get("require_placement_success"),
            ),
            "require_object_stable": _validate_bool(
                "validation.requirements.require_object_stable",
                requirements.get("require_object_stable"),
            ),
            "retry_on_collision": _validate_bool(
                "validation.retries.retry_on_collision",
                retries.get("retry_on_collision"),
            ),
            "retry_on_joint_limit": _validate_bool(
                "validation.retries.retry_on_joint_limit",
                retries.get("retry_on_joint_limit"),
            ),
        }
    )

    return validated
