"""
Shared validation for Arena benchmark outputs.

Validates Arena evaluation results for required fields, types, ranges,
metadata consistency, and IsaacLab-Arena API version reporting.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


class ArenaBenchmarkValidationError(ValueError):
    """Raised when benchmark validation fails."""

    def __init__(self, errors: Sequence[str]) -> None:
        message = "Arena benchmark validation failed:\n" + "\n".join(
            f"- {error}" for error in errors
        )
        super().__init__(message)
        self.errors = list(errors)


def resolve_isaac_lab_arena_version(arena_dir: Optional[Path] = None) -> Optional[str]:
    """Resolve IsaacLab-Arena API version from manifest, module, or env."""
    if arena_dir is not None:
        manifest_path = arena_dir / "arena_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            return (
                manifest.get("isaac_lab_arena_version")
                or manifest.get("arena_version")
                or manifest.get("version")
            )

    spec = importlib.util.find_spec("isaaclab_arena")
    if spec is not None:
        module = importlib.import_module("isaaclab_arena")
        return getattr(module, "__version__", None) or getattr(module, "version", None)

    return os.getenv("ISAACLAB_ARENA_VERSION")


def validate_arena_benchmark_results(
    result: Mapping[str, Any],
    *,
    scene_id: Optional[str] = None,
    task_ids: Optional[Sequence[str]] = None,
    arena_dir: Optional[Path] = None,
    arena_version: Optional[str] = None,
) -> None:
    """Validate Arena benchmark results."""
    errors: list[str] = []

    if not isinstance(result, Mapping):
        raise ArenaBenchmarkValidationError([
            f"Benchmark results must be a mapping, got {type(result).__name__}."
        ])

    resolved_version = (
        arena_version
        or result.get("isaac_lab_arena_version")
        or result.get("arena_version")
        or resolve_isaac_lab_arena_version(arena_dir)
    )

    version_in_result = result.get("isaac_lab_arena_version")
    if not version_in_result:
        errors.append(
            "Missing required field: isaac_lab_arena_version (IsaacLab-Arena API version)."
        )
    elif not isinstance(version_in_result, str):
        errors.append(
            "Invalid type for isaac_lab_arena_version: expected string."
        )

    if resolved_version and version_in_result and resolved_version != version_in_result:
        errors.append(
            "isaac_lab_arena_version mismatch with available Arena metadata."
        )

    if "scene_id" in result:
        _validate_evaluation_runner_schema(result, errors)
        _validate_metadata_consistency(
            result,
            expected_scene_id=scene_id,
            expected_task_ids=task_ids,
            errors=errors,
        )
    elif "env_spec_id" in result:
        _validate_parallel_schema(result, errors)
        _validate_parallel_metadata_consistency(
            result,
            expected_scene_id=scene_id,
            expected_task_ids=task_ids,
            errors=errors,
        )
    else:
        errors.append(
            "Unrecognized benchmark schema: expected 'scene_id' or 'env_spec_id' in results."
        )

    if errors:
        raise ArenaBenchmarkValidationError(errors)


def _validate_evaluation_runner_schema(result: Mapping[str, Any], errors: list[str]) -> None:
    required_fields = {
        "success": bool,
        "scene_id": str,
        "policy_path": str,
        "policy_type": str,
        "embodiment": str,
        "timestamp": str,
        "total_episodes": int,
        "overall_success_rate": (int, float),
        "task_metrics": dict,
        "summary": dict,
        "errors": list,
    }
    _require_fields(result, required_fields, errors)

    _validate_rate(result, "overall_success_rate", errors)
    _validate_non_negative_int(result, "total_episodes", errors)

    task_metrics = result.get("task_metrics", {})
    if isinstance(task_metrics, Mapping):
        total_task_episodes = 0
        for task_id, metrics in task_metrics.items():
            if not isinstance(metrics, Mapping):
                errors.append(
                    f"task_metrics.{task_id} must be a mapping, got {type(metrics).__name__}."
                )
                continue
            _require_fields(
                metrics,
                {
                    "task_name": str,
                    "success_rate": (int, float),
                    "num_episodes": int,
                    "num_successes": int,
                    "timeout_rate": (int, float),
                },
                errors,
                prefix=f"task_metrics.{task_id}",
            )
            _validate_rate(metrics, "success_rate", errors, prefix=f"task_metrics.{task_id}")
            _validate_rate(metrics, "timeout_rate", errors, prefix=f"task_metrics.{task_id}")
            _validate_non_negative_int(metrics, "num_episodes", errors, prefix=f"task_metrics.{task_id}")
            _validate_non_negative_int(metrics, "num_successes", errors, prefix=f"task_metrics.{task_id}")
            if isinstance(metrics.get("num_episodes"), int):
                total_task_episodes += metrics["num_episodes"]

        total_episodes = result.get("total_episodes")
        if isinstance(total_episodes, int) and total_task_episodes != total_episodes:
            errors.append(
                "total_episodes does not match sum of task_metrics.num_episodes."
            )
    else:
        errors.append("task_metrics must be a mapping.")


def _validate_parallel_schema(result: Mapping[str, Any], errors: list[str]) -> None:
    required_fields = {
        "success": bool,
        "config": dict,
        "env_spec_id": str,
        "policy_id": str,
        "embodiment": str,
        "timing": dict,
        "scale": dict,
        "metrics": dict,
        "task_metrics": dict,
        "errors": list,
    }
    _require_fields(result, required_fields, errors)

    timing = result.get("timing", {})
    _require_fields(
        timing,
        {
            "start_time": str,
            "end_time": str,
            "total_wall_time_s": (int, float),
            "episodes_per_second": (int, float),
        },
        errors,
        prefix="timing",
    )

    scale = result.get("scale", {})
    _require_fields(
        scale,
        {
            "num_parallel_envs": int,
            "total_episodes": int,
            "total_steps": int,
        },
        errors,
        prefix="scale",
    )
    _validate_positive_int(scale, "num_parallel_envs", errors, prefix="scale")
    _validate_non_negative_int(scale, "total_episodes", errors, prefix="scale")
    _validate_non_negative_int(scale, "total_steps", errors, prefix="scale")

    metrics = result.get("metrics", {})
    _require_fields(
        metrics,
        {
            "overall_success_rate": (int, float),
            "overall_success_rate_ci": list,
            "overall_mean_return": (int, float),
            "overall_std_return": (int, float),
        },
        errors,
        prefix="metrics",
    )
    _validate_rate(metrics, "overall_success_rate", errors, prefix="metrics")
    _validate_ci(metrics.get("overall_success_rate_ci"), errors, "metrics.overall_success_rate_ci")

    task_metrics = result.get("task_metrics", {})
    if isinstance(task_metrics, Mapping):
        total_task_episodes = 0
        for task_id, metrics_entry in task_metrics.items():
            if not isinstance(metrics_entry, Mapping):
                errors.append(
                    f"task_metrics.{task_id} must be a mapping, got {type(metrics_entry).__name__}."
                )
                continue
            _require_fields(
                metrics_entry,
                {
                    "task_id": str,
                    "task_name": str,
                    "num_episodes": int,
                    "success_rate": (int, float),
                    "success_rate_ci": list,
                    "mean_return": (int, float),
                    "mean_length": (int, float),
                    "timeout_rate": (int, float),
                },
                errors,
                prefix=f"task_metrics.{task_id}",
            )
            _validate_rate(metrics_entry, "success_rate", errors, prefix=f"task_metrics.{task_id}")
            _validate_rate(metrics_entry, "timeout_rate", errors, prefix=f"task_metrics.{task_id}")
            _validate_ci(
                metrics_entry.get("success_rate_ci"),
                errors,
                f"task_metrics.{task_id}.success_rate_ci",
            )
            _validate_non_negative_int(
                metrics_entry,
                "num_episodes",
                errors,
                prefix=f"task_metrics.{task_id}",
            )
            if isinstance(metrics_entry.get("num_episodes"), int):
                total_task_episodes += metrics_entry["num_episodes"]

        total_episodes = scale.get("total_episodes")
        if isinstance(total_episodes, int) and total_task_episodes != total_episodes:
            errors.append(
                "scale.total_episodes does not match sum of task_metrics.num_episodes."
            )
    else:
        errors.append("task_metrics must be a mapping.")


def _validate_metadata_consistency(
    result: Mapping[str, Any],
    *,
    expected_scene_id: Optional[str],
    expected_task_ids: Optional[Sequence[str]],
    errors: list[str],
) -> None:
    if expected_scene_id and result.get("scene_id") != expected_scene_id:
        errors.append(
            f"scene_id mismatch: expected '{expected_scene_id}', got '{result.get('scene_id')}'."
        )

    if expected_task_ids:
        observed_task_ids = set(result.get("task_metrics", {}).keys())
        expected_task_ids_set = set(expected_task_ids)
        missing = expected_task_ids_set - observed_task_ids
        extra = observed_task_ids - expected_task_ids_set
        if missing:
            errors.append(
                f"task_metrics missing expected tasks: {sorted(missing)}."
            )
        if extra:
            errors.append(
                f"task_metrics contains unexpected tasks: {sorted(extra)}."
            )


def _validate_parallel_metadata_consistency(
    result: Mapping[str, Any],
    *,
    expected_scene_id: Optional[str],
    expected_task_ids: Optional[Sequence[str]],
    errors: list[str],
) -> None:
    if expected_scene_id and result.get("env_spec_id") != expected_scene_id:
        errors.append(
            f"env_spec_id mismatch: expected '{expected_scene_id}', got '{result.get('env_spec_id')}'."
        )

    if expected_task_ids:
        task_metrics = result.get("task_metrics", {})
        observed_task_ids = {
            entry.get("task_id")
            for entry in task_metrics.values()
            if isinstance(entry, Mapping)
        }
        expected_task_ids_set = set(expected_task_ids)
        missing = expected_task_ids_set - observed_task_ids
        extra = observed_task_ids - expected_task_ids_set
        if missing:
            errors.append(
                f"task_metrics missing expected task_ids: {sorted(missing)}."
            )
        if extra:
            errors.append(
                f"task_metrics contains unexpected task_ids: {sorted(extra)}."
            )


def _require_fields(
    data: Mapping[str, Any],
    required: Mapping[str, Any],
    errors: list[str],
    *,
    prefix: str = "",
) -> None:
    for field, expected_type in required.items():
        key = f"{prefix}.{field}" if prefix else field
        if field not in data:
            errors.append(f"Missing required field: {key}.")
            continue
        value = data.get(field)
        if expected_type is bool:
            if not isinstance(value, bool):
                errors.append(
                    f"Invalid type for {key}: expected bool, got {type(value).__name__}."
                )
            continue
        if not isinstance(value, expected_type):
            expected_name = (
                expected_type.__name__
                if isinstance(expected_type, type)
                else "/".join(t.__name__ for t in expected_type)
            )
            errors.append(
                f"Invalid type for {key}: expected {expected_name}, got {type(value).__name__}."
            )


def _validate_rate(
    data: Mapping[str, Any],
    field: str,
    errors: list[str],
    prefix: str = "",
) -> None:
    key = f"{prefix}.{field}" if prefix else field
    value = data.get(field)
    if isinstance(value, bool):
        errors.append(f"Invalid type for {key}: expected float in [0, 1].")
        return
    if isinstance(value, (int, float)):
        if value < 0 or value > 1:
            errors.append(f"{key} must be between 0 and 1.")
    elif value is not None:
        errors.append(f"Invalid type for {key}: expected float in [0, 1].")


def _validate_non_negative_int(
    data: Mapping[str, Any],
    field: str,
    errors: list[str],
    prefix: str = "",
) -> None:
    key = f"{prefix}.{field}" if prefix else field
    value = data.get(field)
    if isinstance(value, bool):
        errors.append(f"Invalid type for {key}: expected int, got bool.")
        return
    if isinstance(value, int):
        if value < 0:
            errors.append(f"{key} must be >= 0.")
    elif value is not None:
        errors.append(f"Invalid type for {key}: expected int.")


def _validate_positive_int(
    data: Mapping[str, Any],
    field: str,
    errors: list[str],
    prefix: str = "",
) -> None:
    key = f"{prefix}.{field}" if prefix else field
    value = data.get(field)
    if isinstance(value, bool):
        errors.append(f"Invalid type for {key}: expected int, got bool.")
        return
    if isinstance(value, int):
        if value <= 0:
            errors.append(f"{key} must be > 0.")
    elif value is not None:
        errors.append(f"Invalid type for {key}: expected int.")


def _validate_ci(value: Any, errors: list[str], key: str) -> None:
    if isinstance(value, list) and len(value) == 2:
        if all(isinstance(v, (int, float)) for v in value):
            if value[0] > value[1]:
                errors.append(f"{key} must be an increasing interval.")
            return
        errors.append(f"{key} must contain numeric values.")
        return
    if value is not None:
        errors.append(f"{key} must be a list of two numeric values.")
