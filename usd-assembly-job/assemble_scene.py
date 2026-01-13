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

import os
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
    sys.exit(assemble_from_env())


if __name__ == "__main__":
    main()
