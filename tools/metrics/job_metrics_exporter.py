"""Export Genie Sim job.json metrics summaries to pipeline metrics."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from tools.metrics.pipeline_metrics import get_metrics

logger = logging.getLogger(__name__)


def _parse_duration_seconds(
    created_at: Optional[str],
    completed_at: Optional[str],
) -> Optional[float]:
    if not created_at or not completed_at:
        return None
    try:
        created_dt = datetime.fromisoformat(created_at.replace("Z", ""))
        completed_dt = datetime.fromisoformat(completed_at.replace("Z", ""))
    except ValueError:
        return None
    return max(0.0, (completed_dt - created_dt).total_seconds())


def _resolve_robot_type(job_payload: Dict[str, Any]) -> str:
    generation_params = job_payload.get("generation_params")
    if isinstance(generation_params, dict):
        robot_type = generation_params.get("robot_type")
        if isinstance(robot_type, str) and robot_type:
            return robot_type
        robot_types = generation_params.get("robot_types")
        if isinstance(robot_types, list) and robot_types:
            return str(robot_types[0])
    return "unknown"


def _resolve_episode_count(metrics_summary: Dict[str, Any]) -> Optional[int]:
    for key in ("episodes_collected", "total_episodes", "episodes_passed"):
        value = metrics_summary.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return None


def _emit_metrics_for_summary(
    *,
    scene_id: str,
    job_id: str,
    robot_type: str,
    metrics_summary: Dict[str, Any],
) -> None:
    metrics = get_metrics()
    labels = {
        "scene_id": scene_id,
        "job_id": job_id,
        "robot_type": robot_type,
    }
    metrics.geniesim_jobs.inc(labels=labels)

    episode_count = _resolve_episode_count(metrics_summary)
    if episode_count is not None:
        metrics.geniesim_episodes_generated.inc(episode_count, labels=labels)

    duration_seconds = metrics_summary.get("duration_seconds")
    if not isinstance(duration_seconds, (int, float)):
        duration_seconds = _parse_duration_seconds(
            metrics_summary.get("created_at"),
            metrics_summary.get("completed_at"),
        )
    if isinstance(duration_seconds, (int, float)):
        metrics.geniesim_job_duration_seconds.observe(float(duration_seconds), labels=labels)

    if metrics.enable_logging:
        logger.info(
            "[METRICS] Exported Genie Sim job.json summary for %s/%s (%s)",
            scene_id,
            job_id,
            robot_type,
        )


def export_job_metrics(
    *,
    job_payload: Optional[Dict[str, Any]] = None,
    job_json_path: Optional[Path | str] = None,
    scene_id: Optional[str] = None,
) -> bool:
    """Emit job.json metrics summaries to the configured pipeline metrics backend."""
    if job_payload is None:
        if not job_json_path:
            raise ValueError("job_payload or job_json_path is required")
        path = Path(job_json_path)
        if not path.exists():
            logger.warning("Job metrics export skipped; missing job.json at %s", path)
            return False
        try:
            job_payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("Job metrics export skipped; invalid job.json: %s", exc)
            return False

    if not isinstance(job_payload, dict):
        logger.warning("Job metrics export skipped; payload is not a dict")
        return False

    resolved_scene_id = scene_id or job_payload.get("scene_id") or "unknown"
    job_id = job_payload.get("job_id") or "unknown"

    job_metrics_by_robot = job_payload.get("job_metrics_by_robot")
    if isinstance(job_metrics_by_robot, dict) and job_metrics_by_robot:
        for robot_type, summary in job_metrics_by_robot.items():
            if not isinstance(summary, dict):
                continue
            summary_job_id = summary.get("job_id") or job_id
            _emit_metrics_for_summary(
                scene_id=resolved_scene_id,
                job_id=summary_job_id,
                robot_type=str(robot_type),
                metrics_summary=summary,
            )
        return True

    job_metrics = job_payload.get("job_metrics")
    if isinstance(job_metrics, dict):
        _emit_metrics_for_summary(
            scene_id=resolved_scene_id,
            job_id=job_metrics.get("job_id") or job_id,
            robot_type=_resolve_robot_type(job_payload),
            metrics_summary=job_metrics,
        )
        return True

    job_metrics_summary = job_payload.get("job_metrics_summary")
    if isinstance(job_metrics_summary, dict):
        _emit_metrics_for_summary(
            scene_id=resolved_scene_id,
            job_id=job_metrics_summary.get("job_id") or job_id,
            robot_type=_resolve_robot_type(job_payload),
            metrics_summary=job_metrics_summary,
        )
        return True

    logger.warning("Job metrics export skipped; no metrics summary found")
    return False
