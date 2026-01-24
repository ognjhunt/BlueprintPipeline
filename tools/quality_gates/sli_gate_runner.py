#!/usr/bin/env python3
"""SLI quality gate runner for production workflows.

Reads episode generation artifacts, emits metrics, and enforces quality gates
for production runs based on configurable SLIs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.config import load_quality_config
from tools.metrics.pipeline_metrics import get_metrics
from tools.quality_gates import build_notification_service
from tools.quality_gates.quality_gate import QualityGateCheckpoint, QualityGateRegistry


def _load_json_from_path(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def _load_json_from_gcs(bucket: str, object_name: str) -> Dict[str, Any]:
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage is required to load GCS artifacts. "
            "Install google-cloud-storage or mount the bucket locally."
        ) from exc

    client = storage.Client()
    blob = client.bucket(bucket).blob(object_name)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: gs://{bucket}/{object_name}")
    return json.loads(blob.download_as_text())


def _load_artifact(
    bucket: Optional[str],
    object_name: str,
    data_root: Path,
    scene_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if scene_dir:
        if "/episodes/" in object_name:
            relative = object_name.split("/episodes/", 1)[1]
            local_path = scene_dir / "episodes" / relative
        else:
            local_path = scene_dir / object_name
        if local_path.exists():
            return _load_json_from_path(local_path)

    local_path = data_root / object_name
    if local_path.exists():
        return _load_json_from_path(local_path)

    if not bucket:
        raise RuntimeError(
            f"Missing BUCKET for GCS lookup and local path not found: {local_path}"
        )
    return _load_json_from_gcs(bucket, object_name)


def _build_quality_context(
    quality_manifest: Dict[str, Any],
    generation_manifest: Dict[str, Any],
    config: Any,
    episode_metadata_path: Optional[Path] = None,
    lerobot_dataset_path: Optional[Path] = None,
) -> Dict[str, Any]:
    episodes = quality_manifest.get("episodes", [])
    total = len(episodes)
    stats = generation_manifest.get("statistics", {})
    average_quality_score = float(stats.get("average_quality_score", 0.0) or 0.0)
    pass_rate = float(stats.get("pass_rate", 0.0) or 0.0)

    quality_threshold = config.episodes.quality_score_min
    passed_quality = sum(
        1 for ep in episodes if (ep.get("quality_score") or 0.0) >= quality_threshold
    )

    allowed_sensor_sources = set(config.data_quality.allowed_sensor_sources)
    allowed_physics_backends = set(config.data_quality.allowed_physics_backends)

    sensor_sources = [ep.get("sensor_backend") for ep in episodes if ep.get("sensor_backend")]
    physics_backends = [ep.get("physics_backend") for ep in episodes if ep.get("physics_backend")]

    sensor_capture_rate = (
        sum(1 for source in sensor_sources if source in allowed_sensor_sources) / total
        if total
        else 0.0
    )
    physics_validation_rate = (
        sum(1 for backend in physics_backends if backend in allowed_physics_backends) / total
        if total
        else 0.0
    )

    return {
        "episode_stats": {
            "total_generated": total,
            "passed_quality_filter": passed_quality,
            "average_quality_score": average_quality_score,
            "collision_free_rate": pass_rate,
        },
        "data_quality": {
            "sensor_capture_rate": sensor_capture_rate,
            "sensor_sources": sorted(set(sensor_sources)),
            "physics_validation_rate": physics_validation_rate,
            "physics_backends": sorted(set(physics_backends)),
        },
        "episode_metadata_path": str(episode_metadata_path) if episode_metadata_path else None,
        "lerobot_dataset_path": str(lerobot_dataset_path) if lerobot_dataset_path else None,
    }


def _emit_metrics(
    scene_id: str,
    quality_context: Dict[str, Any],
    quality_manifest: Dict[str, Any],
) -> None:
    metrics = get_metrics()
    episode_stats = quality_context["episode_stats"]
    data_quality = quality_context["data_quality"]

    metrics.episode_quality_score.observe(
        episode_stats["average_quality_score"],
        labels={"scene_id": scene_id},
    )
    metrics.physics_validation_score.observe(
        data_quality["physics_validation_rate"],
        labels={"scene_id": scene_id},
    )

    source_counts: Dict[str, int] = {}
    for entry in quality_manifest.get("episodes", []):
        source = entry.get("sensor_backend") or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in source_counts.items():
        metrics.sensor_capture_source_total.inc(
            count,
            labels={"scene_id": scene_id, "source": source},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run production SLI quality gates.")
    parser.add_argument("--scene-id", default=os.getenv("SCENE_ID", ""))
    parser.add_argument("--bucket", default=os.getenv("BUCKET"))
    parser.add_argument("--episodes-prefix", default=os.getenv("EPISODES_PREFIX"))
    parser.add_argument("--data-root", default=os.getenv("DATA_ROOT", "/mnt/gcs"))
    parser.add_argument("--scene-dir", default=os.getenv("SCENE_DIR"))
    parser.add_argument("--report-path", default=os.getenv("QUALITY_GATE_REPORT_PATH"))

    args = parser.parse_args()

    if not args.scene_id:
        raise RuntimeError("SCENE_ID is required to run quality gates.")

    episodes_prefix = args.episodes_prefix or f"scenes/{args.scene_id}/episodes"
    data_root = Path(args.data_root)
    scene_dir = Path(args.scene_dir).expanduser().resolve() if args.scene_dir else None

    quality_manifest = _load_artifact(
        bucket=args.bucket,
        object_name=f"{episodes_prefix}/dataset_quality_manifest.json",
        data_root=data_root,
        scene_dir=scene_dir,
    )
    generation_manifest = _load_artifact(
        bucket=args.bucket,
        object_name=f"{episodes_prefix}/manifests/generation_manifest.json",
        data_root=data_root,
        scene_dir=scene_dir,
    )

    config = load_quality_config()
    notification_service = build_notification_service(config, verbose=True)
    episode_metadata_path = None
    lerobot_dataset_path = None
    if scene_dir:
        lerobot_candidate = scene_dir / "episodes" / "lerobot"
        if lerobot_candidate.is_dir():
            lerobot_dataset_path = lerobot_candidate
        info_candidate = lerobot_candidate / "meta" / "info.json" if lerobot_candidate else None
        metadata_candidate = lerobot_candidate / "metadata.json" if lerobot_candidate else None
        if info_candidate and info_candidate.is_file():
            episode_metadata_path = info_candidate
        elif metadata_candidate and metadata_candidate.is_file():
            episode_metadata_path = metadata_candidate

    quality_context = _build_quality_context(
        quality_manifest,
        generation_manifest,
        config,
        episode_metadata_path=episode_metadata_path,
        lerobot_dataset_path=lerobot_dataset_path,
    )

    _emit_metrics(args.scene_id, quality_context, quality_manifest)

    registry = QualityGateRegistry(verbose=True, config=config)
    registry.run_checkpoint(
        checkpoint=QualityGateCheckpoint.EPISODES_GENERATED,
        context=quality_context,
        notification_service=notification_service,
    )

    if args.report_path:
        registry.save_report(args.scene_id, Path(args.report_path))

    if not registry.can_proceed():
        print("[QUALITY-GATE] Blocking failures detected. Halting pipeline.")
        return 1

    print("[QUALITY-GATE] All production SLIs satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
