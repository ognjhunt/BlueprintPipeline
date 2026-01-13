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
"""

import json
import os
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from blueprint_sim.assembly import assemble_from_env
from tools.validation.entrypoint_checks import (
    validate_required_env_vars,
    validate_scene_manifest,
)
from tools.metrics.pipeline_metrics import get_metrics


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
    assets_prefix = os.getenv("ASSETS_PREFIX", "")
    assets_root = Path("/mnt/gcs") / assets_prefix
    validate_scene_manifest(assets_root / "scene_manifest.json", label="[USD-ASSEMBLY]")
    scene_id = os.getenv("SCENE_ID", "")
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
    main()
