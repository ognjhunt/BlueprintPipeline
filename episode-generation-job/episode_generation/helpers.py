import json
import logging
import os
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.config.production_mode import resolve_production_mode

logger = logging.getLogger(__name__)


def _should_bypass_quality_gates() -> bool:
    return os.getenv("BYPASS_QUALITY_GATES", "").lower() in {"1", "true", "yes", "y"}


def _is_service_mode() -> bool:
    return (
        os.getenv("SERVICE_MODE", "").lower() in {"1", "true", "yes", "y"}
        or os.getenv("K_SERVICE") is not None
        or os.getenv("KUBERNETES_SERVICE_HOST") is not None
    )


def _gate_report_path(root: Path, scene_id: str, job_name: str) -> Path:
    report_path = root / f"scenes/{scene_id}/{job_name}/quality_gate_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return report_path


def _compute_collision_free_rate(
    validation_report_path: Optional[Path],
    fallback_rate: float,
) -> float:
    if not validation_report_path or not validation_report_path.is_file():
        return fallback_rate
    try:
        report = json.loads(validation_report_path.read_text())
        episodes = report.get("episodes", [])
        if not episodes:
            return fallback_rate
        collision_free = sum(1 for ep in episodes if not ep.get("collision_events"))
        return collision_free / len(episodes)
    except Exception:
        logger.warning(
            "[EPISODE-GEN-JOB] Failed to compute collision-free rate from %s; using fallback.",
            validation_report_path,
            exc_info=True,
        )
        return fallback_rate


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} {units[-1]}"


def _parse_headroom_pct() -> float:
    raw = os.getenv("EXPORT_DISK_HEADROOM_PCT")
    if not raw:
        return 0.15
    try:
        pct = float(raw)
        if pct > 1:
            pct /= 100.0
        return max(pct, 0.0)
    except ValueError:
        logger.warning(
            "[EPISODE-GEN-JOB] Invalid EXPORT_DISK_HEADROOM_PCT value %r; using default.",
            raw,
        )
        return 0.15


def _is_production_run() -> bool:
    if resolve_production_mode():
        return True
    try:
        from quality_certificate import get_data_quality_level
    except Exception:
        return False
    try:
        return get_data_quality_level().value == "production"
    except Exception:
        logger.warning(
            "[EPISODE-GEN-JOB] Failed to determine data quality level; "
            "falling back to production env flags.",
            exc_info=True,
        )
    return False


def _estimate_export_requirements(
    episodes: List["GeneratedEpisode"],
    config: "EpisodeGenerationConfig",
) -> Dict[str, Any]:
    expected_episodes = len(episodes)
    resolution = config.image_resolution
    width, height = resolution
    num_cameras = max(config.num_cameras, 1)

    frame_counts = []
    for episode in episodes:
        if not episode.trajectory:
            continue
        if getattr(episode.trajectory, "num_frames", 0):
            frame_counts.append(int(episode.trajectory.num_frames))
        elif getattr(episode.trajectory, "total_duration", 0):
            frame_counts.append(int(ceil(episode.trajectory.total_duration * config.fps)))

    if frame_counts:
        frames_per_episode = max(frame_counts)
        episode_seconds = frames_per_episode / max(config.fps, 1.0)
    else:
        episode_seconds = 10.0
        frames_per_episode = int(ceil(episode_seconds * max(config.fps, 1.0)))

    bytes_per_pixel = {
        "core": 3,  # RGB
        "plus": 8,  # RGB + depth + segmentation
        "full": 16,  # RGB + depth + segmentation + normals/metadata
    }.get(config.data_pack_tier.lower(), 3)

    pixels_per_frame = width * height
    per_frame_overhead = 64 * 1024
    per_episode_overhead = 5 * 1024 * 1024
    bytes_per_frame = num_cameras * pixels_per_frame * bytes_per_pixel + per_frame_overhead
    required_bytes = expected_episodes * (frames_per_episode * bytes_per_frame + per_episode_overhead)

    return {
        "required_bytes": required_bytes,
        "expected_episodes": expected_episodes,
        "frames_per_episode": frames_per_episode,
        "episode_seconds": episode_seconds,
        "num_cameras": num_cameras,
        "resolution": resolution,
        "bytes_per_pixel": bytes_per_pixel,
        "bytes_per_frame": bytes_per_frame,
        "per_frame_overhead": per_frame_overhead,
        "per_episode_overhead": per_episode_overhead,
    }
