from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_quality_thresholds(
    *,
    min_quality_score: float,
    filter_low_quality: bool,
    dimension_thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "min_quality_score": min_quality_score,
        "filter_low_quality": filter_low_quality,
    }
    if dimension_thresholds:
        payload["dimension_thresholds"] = dict(dimension_thresholds)
    return payload


def build_geniesim_idempotency_inputs(
    *,
    scene_id: str,
    task_config: Mapping[str, Any],
    robot_types: Sequence[str],
    episodes_per_task: int,
    quality_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "task_config": dict(task_config),
        "robot_types": list(robot_types),
        "episodes_per_task": episodes_per_task,
        "quality_thresholds": dict(quality_thresholds),
    }
