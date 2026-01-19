#!/usr/bin/env python3
"""
assemble_scene.py - Orchestrate the full USD scene assembly pipeline.

This script:
  1. Reads scene_assets.json to find all objects
  2. Converts GLB assets to USDZ format
  3. Builds the scene.usda with proper references
  4. Validates the final scene

Usage:
  python assemble_scene.py

Environment variables:
  BUCKET         - GCS bucket name
  SCENE_ID       - Scene identifier
  LAYOUT_PREFIX  - Path prefix for layout files
  ASSETS_PREFIX  - Path prefix for asset files
  USD_PREFIX     - Path prefix for USD output (defaults to ASSETS_PREFIX)
  BYPASS_QUALITY_GATES - Skip quality gate evaluation (dev-only)
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from blueprint_sim.assembly import assemble_from_env
from tools.metrics.pipeline_metrics import get_metrics
from tools.quality_gates.quality_gate import QualityGateCheckpoint, QualityGateRegistry
from tools.validation.entrypoint_checks import (
    validate_required_env_vars,
    validate_scene_manifest,
)
from tools.workflow.failure_markers import FailureMarkerWriter
from tools.logging_config import init_logging

JOB_NAME = "usd-assembly-job"
GCS_ROOT = Path("/mnt/gcs")

logger = logging.getLogger(__name__)


def _should_bypass_quality_gates() -> bool:
    return os.getenv("BYPASS_QUALITY_GATES", "").lower() in {"1", "true", "yes", "y"}


def _gate_report_path(scene_id: str) -> Path:
    report_path = GCS_ROOT / f"scenes/{scene_id}/{JOB_NAME}/quality_gate_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return report_path


def main() -> None:
    validate_required_env_vars(
        {
            "BUCKET": "GCS bucket name",
            "SCENE_ID": "Scene identifier",
            "LAYOUT_PREFIX": "Path prefix for layout files (scenes/<sceneId>/layout)",
            "ASSETS_PREFIX": "Path prefix for assets (scenes/<sceneId>/assets)",
        },
        label="[USD-ASSEMBLY]",
    )
    bucket = os.environ["BUCKET"]
    scene_id = os.environ["SCENE_ID"]
    assets_prefix = os.environ["ASSETS_PREFIX"]
    assets_root = GCS_ROOT / assets_prefix
    validate_scene_manifest(assets_root / "scene_manifest.json", label="[USD-ASSEMBLY]")
    exit_code = assemble_from_env()

    if exit_code != 0:
        FailureMarkerWriter(bucket, scene_id, JOB_NAME).write_failure(
            exception=RuntimeError("USD assembly failed during scene build"),
            failed_step="assemble_scene",
            input_params={
                "scene_id": scene_id,
                "assets_prefix": assets_prefix,
                "layout_prefix": os.environ.get("LAYOUT_PREFIX", ""),
                "usd_prefix": os.getenv("USD_PREFIX") or assets_prefix,
            },
            partial_results={"exit_code": exit_code},
            recommendations=["Check usd-assembly-job logs for object add failures."],
        )
        sys.exit(exit_code)

    if _should_bypass_quality_gates():
        logger.warning("[USD-ASSEMBLY] ⚠️  BYPASS_QUALITY_GATES enabled - skipping quality gates")
        sys.exit(exit_code)

    usd_prefix = os.getenv("USD_PREFIX") or assets_prefix
    usd_path = GCS_ROOT / usd_prefix / "scene.usda"

    quality_gates = QualityGateRegistry(verbose=True)
    quality_gates.run_checkpoint(
        QualityGateCheckpoint.USD_ASSEMBLED,
        context={"usd_path": str(usd_path), "scene_id": scene_id},
    )
    report_path = _gate_report_path(scene_id)
    quality_gates.save_report(scene_id, report_path)

    if not quality_gates.can_proceed():
        logger.error("[USD-ASSEMBLY] ❌ Quality gates blocked downstream pipeline")
        FailureMarkerWriter(bucket, scene_id, JOB_NAME).write_failure(
            exception=RuntimeError("Quality gates blocked: USD validation failed"),
            failed_step="quality_gates",
            input_params={
                "scene_id": scene_id,
                "usd_prefix": usd_prefix,
                "usd_path": str(usd_path),
            },
            partial_results={"quality_gate_report": str(report_path)},
            recommendations=[
                "Fix USD validation errors before proceeding.",
                f"Review quality gate report: {report_path}",
            ],
        )
        sys.exit(1)
    metrics = get_metrics()
    with metrics.track_job("usd-assembly-job", scene_id):
        exit_code = assemble_from_env()

    usd_prefix = os.getenv("USD_PREFIX") or assets_prefix
    completion_manifest_path = Path("/mnt/gcs") / usd_prefix / "usd_assembly_manifest.json"
    metrics_summary = {
        "backend": metrics.backend.value,
        "stats": metrics.get_stats(),
    }
    completion_manifest = {
        "scene_id": scene_id,
        "assets_prefix": assets_prefix,
        "layout_prefix": os.getenv("LAYOUT_PREFIX", ""),
        "usd_prefix": usd_prefix,
        "convert_only": os.getenv("CONVERT_ONLY", "false").lower() in {"1", "true", "yes"},
        "status": "completed" if exit_code == 0 else "failed",
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "metrics_summary": metrics_summary,
    }
    completion_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    completion_manifest_path.write_text(json.dumps(completion_manifest, indent=2))

    sys.exit(exit_code)


if __name__ == "__main__":
    from tools.startup_validation import validate_and_fail_fast

    init_logging()
    validate_and_fail_fast(job_name="USD-ASSEMBLY", validate_gcs=True)
    main()
