#!/usr/bin/env python3
"""Parallel batch runner for BlueprintPipeline scenes.

Supports scene lists or manifests, parallel execution, retries, checkpoints,
progress tracking, and quality-gate reports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.batch_processing.parallel_runner import ParallelPipelineRunner, SceneStatus
from tools.checkpoint import should_skip_step, write_checkpoint
from tools.checkpoint.hash_config import resolve_checkpoint_hash_setting
from tools.metrics import track_pipeline_run, update_pipeline_status
from tools.metrics.pipeline_metrics import get_metrics
from tools.quality_gates import QualityGateCheckpoint, QualityGateRegistry
from tools.run_local_pipeline import LocalPipelineRunner, PipelineStep
from tools.scene_batch_reporting import _summarize_batch_results
from tools.locking.gcs_lock import DEFAULT_HEARTBEAT_SECONDS, DEFAULT_TTL_SECONDS, GCSLock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneBatchItem:
    scene_id: str
    scene_dir: Path
    environment_type: str = "generic"


@dataclass
class SceneBatchConfig:
    steps: Optional[List[PipelineStep]]
    resume_from: Optional[PipelineStep]
    run_validation: bool
    skip_interactive: bool
    enable_dwm: bool
    enable_dream2flow: bool
    enable_inventory_enrichment: Optional[bool]
    disable_articulated_assets: bool


def _parse_scene_list_file(path: Path) -> List[str]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict) and "scenes" in payload:
            return [str(item) for item in payload["scenes"]]
        return []

    lines = []
    for line in path.read_text().splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        lines.append(cleaned)
    return lines


def _parse_scene_manifest(path: Path, scene_root: Path) -> List[SceneBatchItem]:
    payload = json.loads(path.read_text())
    items: List[SceneBatchItem] = []

    if isinstance(payload, dict) and "scenes" in payload:
        scenes = payload["scenes"]
    else:
        scenes = payload

    if isinstance(scenes, list):
        for entry in scenes:
            if isinstance(entry, str):
                items.append(SceneBatchItem(scene_id=entry, scene_dir=scene_root / entry))
                continue
            if not isinstance(entry, dict):
                continue
            scene_id = entry.get("scene_id") or entry.get("id")
            if not scene_id:
                continue
            scene_dir = Path(entry.get("scene_dir", scene_root / scene_id)).expanduser().resolve()
            environment_type = entry.get("environment_type", "generic")
            items.append(SceneBatchItem(scene_id=scene_id, scene_dir=scene_dir, environment_type=environment_type))
        return items

    if isinstance(payload, dict) and payload.get("scene_id"):
        scene_id = payload["scene_id"]
        scene_dir = (scene_root / scene_id).expanduser().resolve()
        if path.name == "scene_manifest.json" and path.parent.name == "assets":
            scene_dir = path.parents[1]
        items.append(SceneBatchItem(scene_id=scene_id, scene_dir=scene_dir))
    return items


def _parse_scene_ids(raw_ids: Sequence[str], scene_root: Path) -> List[SceneBatchItem]:
    items: List[SceneBatchItem] = []
    for entry in raw_ids:
        if not entry:
            continue
        path = Path(entry)
        if path.exists() and path.is_dir():
            scene_dir = path.expanduser().resolve()
            scene_id = scene_dir.name
        else:
            scene_id = entry
            scene_dir = (scene_root / scene_id).expanduser().resolve()
        items.append(SceneBatchItem(scene_id=scene_id, scene_dir=scene_dir))
    return items


def _dedupe_items(items: Iterable[SceneBatchItem]) -> List[SceneBatchItem]:
    seen = set()
    deduped = []
    for item in items:
        key = (item.scene_id, str(item.scene_dir))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _parse_steps(value: Optional[str]) -> Optional[List[PipelineStep]]:
    if not value:
        return None
    steps = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            steps.append(PipelineStep(entry))
        except ValueError as exc:
            raise ValueError(f"Unknown pipeline step: {entry}") from exc
    return steps


def _parse_resume_from(value: Optional[str]) -> Optional[PipelineStep]:
    if not value:
        return None
    try:
        return PipelineStep(value)
    except ValueError as exc:
        raise ValueError(f"Unknown resume-from step: {value}") from exc




def _dir_has_files(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _build_readiness_checklist(
    scene_dir: Path,
    results: List[Any],
    *,
    enable_dwm: bool,
) -> Dict[str, bool]:
    steps_success = {result.step.value: result.success for result in results}
    checklist = {
        "usd_valid": (scene_dir / "usd" / "scene.usda").exists(),
        "physics_stable": steps_success.get(PipelineStep.SIMREADY.value, False),
        "episodes_generated": _dir_has_files(scene_dir / "episodes"),
        "replicator_ready": _dir_has_files(scene_dir / "replicator"),
        "isaac_lab_ready": _dir_has_files(scene_dir / "isaac_lab"),
    }
    if enable_dwm:
        checklist["dwm_ready"] = _dir_has_files(scene_dir / "dwm")
    return checklist


def _run_quality_gates(
    scene_id: str,
    scene_dir: Path,
    results: List[Any],
    report_path: Path,
    *,
    enable_dwm: bool,
) -> Dict[str, Any]:
    checklist = _build_readiness_checklist(scene_dir, results, enable_dwm=enable_dwm)
    registry = QualityGateRegistry(verbose=True)
    registry.run_checkpoint(
        checkpoint=QualityGateCheckpoint.SCENE_READY,
        context={
            "scene_id": scene_id,
            "readiness_checklist": checklist,
        },
    )
    registry.save_report(scene_id, report_path)
    return registry.to_report(scene_id)


def _scene_report_path(base_dir: Path, scene_id: str) -> Path:
    return base_dir / scene_id / "quality_gate_report.json"


def _build_scene_processor(
    config: SceneBatchConfig,
    reports_dir: Path,
    checkpoint_step: str,
    skip_completed: bool,
    attempt_counts: Dict[str, int],
) -> Tuple[Any, Dict[str, int]]:
    metrics = get_metrics()
    job_name = "scene-batch"
    attempt_lock = Lock()

    def process_scene(scene: SceneBatchItem) -> Dict[str, Any]:
        with attempt_lock:
            attempt_counts[scene.scene_id] = attempt_counts.get(scene.scene_id, 0) + 1
            attempt = attempt_counts[scene.scene_id]
        if attempt > 1:
            metrics.retries_total.inc(labels={"scene_id": scene.scene_id, "job": job_name})

        report_path = _scene_report_path(reports_dir, scene.scene_id)
        if skip_completed and should_skip_step(scene.scene_dir, checkpoint_step, expected_outputs=[report_path]):
            logger.info("Skipping %s (checkpoint found)", scene.scene_id)
            return {
                "scene_id": scene.scene_id,
                "scene_dir": str(scene.scene_dir),
                "skipped": True,
                "quality_gate_report": str(report_path),
            }

        delivery_id = track_pipeline_run(
            scene_id=scene.scene_id,
            customer_id="batch_runner",
            environment_type=scene.environment_type,
        )
        update_pipeline_status(delivery_id, "processing")

        start_time = time.time()
        success = False
        error_message = ""
        quality_report: Optional[Dict[str, Any]] = None

        with metrics.track_job(job_name, scene.scene_id):
            runner = LocalPipelineRunner(
                scene_dir=scene.scene_dir,
                verbose=True,
                skip_interactive=config.skip_interactive,
                environment_type=scene.environment_type,
                enable_dwm=config.enable_dwm,
                enable_dream2flow=config.enable_dream2flow,
                enable_inventory_enrichment=config.enable_inventory_enrichment,
                disable_articulated_assets=config.disable_articulated_assets,
            )
            try:
                success = runner.run(
                    steps=config.steps,
                    run_validation=config.run_validation,
                    resume_from=config.resume_from,
                )
            except Exception as exc:
                success = False
                error_message = str(exc)

            if success:
                update_pipeline_status(delivery_id, "delivered")
            else:
                if not error_message:
                    error_message = "Pipeline run failed"
                update_pipeline_status(delivery_id, "failed", error=error_message)

            if reports_dir:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                quality_report = _run_quality_gates(
                    scene.scene_id,
                    scene.scene_dir,
                    runner.results,
                    report_path,
                    enable_dwm=config.enable_dwm,
                )

            if success:
                write_checkpoint(
                    scene.scene_dir,
                    checkpoint_step,
                    status="completed",
                    started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    outputs={
                        "quality_gate_summary": quality_report.get("summary") if quality_report else {},
                        "delivery_id": delivery_id,
                    },
                    output_paths=[report_path] if report_path else [],
                    scene_id=scene.scene_id,
                    store_output_hashes=resolve_checkpoint_hash_setting(),
                )

        duration = time.time() - start_time
        metadata = {
            "scene_id": scene.scene_id,
            "scene_dir": str(scene.scene_dir),
            "delivery_id": delivery_id,
            "duration_seconds": duration,
            "attempt": attempt,
            "quality_gate_report": str(report_path),
            "quality_gate_summary": quality_report.get("summary") if quality_report else {},
            "steps": [
                {
                    "step": result.step.value,
                    "success": result.success,
                    "message": result.message,
                }
                for result in runner.results
            ],
        }

        if not success:
            failure_details = error_message or "Pipeline run failed"
            raise RuntimeError(failure_details)

        return metadata

    return process_scene, attempt_counts


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BlueprintPipeline scenes in parallel.")
    parser.add_argument("--scene-root", default="./scenes", help="Base directory containing scenes.")
    parser.add_argument("--scene-ids", nargs="*", help="Scene IDs or scene directories to process.")
    parser.add_argument("--scene-list", type=Path, help="Path to a text or JSON scene list.")
    parser.add_argument("--manifest", type=Path, help="Path to a batch manifest or scene_manifest.json.")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent scenes.")
    parser.add_argument("--retry-attempts", type=int, default=3, help="Retry attempts per scene.")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="Retry delay seconds.")
    parser.add_argument("--rate-limit", type=float, default=10.0, help="Rate limit per second.")
    parser.add_argument("--steps", help="Comma-separated pipeline steps to run.")
    parser.add_argument("--resume-from", help="Resume from pipeline step.")
    parser.add_argument("--validate", action="store_true", help="Run validation step.")
    parser.add_argument("--skip-interactive", action="store_true", help="Skip interactive step.")
    parser.add_argument("--enable-dwm", action="store_true", help="Enable DWM steps by default.")
    parser.add_argument("--enable-dream2flow", action="store_true", help="Enable Dream2Flow steps by default.")
    parser.add_argument(
        "--enable-experimental",
        action="store_true",
        help="Enable experimental steps (DWM + Dream2Flow) by default.",
    )
    parser.add_argument("--enable-inventory-enrichment", action="store_true")
    parser.add_argument("--disable-articulated-assets", action="store_true")
    parser.add_argument("--reports-dir", type=Path, default=Path("./batch_reports"))
    parser.add_argument("--dlq-path", type=Path, help="Override dead-letter queue output path.")
    parser.add_argument("--checkpoint-step", default="batch-run", help="Checkpoint name for batch runs.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip scenes with checkpoints.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _build_scene_batch_lock() -> tuple[Optional[GCSLock], float]:
    bucket = os.getenv("GCS_LOCK_BUCKET")
    prefix = os.getenv("GCS_LOCK_PREFIX")
    if not bucket or not prefix:
        return None, 0.0
    ttl_seconds = int(os.getenv("GCS_LOCK_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))
    heartbeat_seconds = int(os.getenv("GCS_LOCK_HEARTBEAT_SECONDS", str(DEFAULT_HEARTBEAT_SECONDS)))
    wait_timeout = float(os.getenv("GCS_LOCK_WAIT_SECONDS", "0"))
    lock_name = os.getenv("GCS_LOCK_NAME", "scene-batch.lock").lstrip("/")
    object_name = f"{prefix.rstrip('/')}/{lock_name}".lstrip("/")
    lock = GCSLock(
        bucket_name=bucket,
        object_name=object_name,
        ttl_seconds=ttl_seconds,
        heartbeat_seconds=heartbeat_seconds,
    )
    return lock, wait_timeout


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.enable_experimental:
        args.enable_dwm = True
        args.enable_dream2flow = True

    scene_root = Path(args.scene_root).expanduser().resolve()
    items: List[SceneBatchItem] = []

    if args.scene_ids:
        items.extend(_parse_scene_ids(args.scene_ids, scene_root))
    if args.scene_list:
        items.extend(_parse_scene_ids(_parse_scene_list_file(args.scene_list), scene_root))
    if args.manifest:
        items.extend(_parse_scene_manifest(args.manifest, scene_root))

    items = _dedupe_items(items)

    if not items:
        logger.error("No scenes found to process. Provide --scene-ids, --scene-list, or --manifest.")
        return 1

    steps = _parse_steps(args.steps)
    resume_from = _parse_resume_from(args.resume_from)

    config = SceneBatchConfig(
        steps=steps,
        resume_from=resume_from,
        run_validation=args.validate,
        skip_interactive=args.skip_interactive,
        enable_dwm=args.enable_dwm,
        enable_dream2flow=args.enable_dream2flow,
        enable_inventory_enrichment=True if args.enable_inventory_enrichment else None,
        disable_articulated_assets=args.disable_articulated_assets,
    )

    attempt_counts: Dict[str, int] = {}
    process_scene, attempt_counts = _build_scene_processor(
        config=config,
        reports_dir=args.reports_dir,
        checkpoint_step=args.checkpoint_step,
        skip_completed=args.skip_completed,
        attempt_counts=attempt_counts,
    )

    runner = ParallelPipelineRunner(
        max_concurrent=args.max_concurrent,
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay,
        rate_limit_per_second=args.rate_limit,
    )

    def progress_callback(completed: int, total: int) -> None:
        logger.info("Batch progress: %s/%s scenes complete", completed, total)

    scene_map = {item.scene_id: item for item in items}

    lock, wait_timeout = _build_scene_batch_lock()
    if lock:
        logger.info("Attempting to acquire GCS lock %s", lock.gcs_uri)
        if not lock.acquire(wait_timeout=wait_timeout):
            logger.error("Failed to acquire GCS lock %s; another worker is active.", lock.gcs_uri)
            return 3
    else:
        logger.info("GCS lock not configured; proceeding without global lock.")

    try:
        async def _run() -> Dict[str, Any]:
            batch_result = await runner.process_batch(
                scene_ids=[item.scene_id for item in items],
                process_fn=lambda scene_id: process_scene(scene_map[scene_id]),
                progress_callback=progress_callback,
            )

            summary = _summarize_batch_results(batch_result.results, args.reports_dir, dlq_path=args.dlq_path)
            logger.info(
                "Batch complete: %s success, %s failed, %s cancelled",
                summary["success"],
                summary["failed"],
                summary["cancelled"],
            )
            return summary

        summary = asyncio.run(_run())
    finally:
        if lock:
            lock.release()

    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
