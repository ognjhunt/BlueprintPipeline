from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from .manifest import DEFAULT_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMREADY_DIR = REPO_ROOT / "simready-job"
if str(SIMREADY_DIR) not in sys.path:
    sys.path.append(str(SIMREADY_DIR))

from prepare_simready_assets import prepare_simready_assets_job  # noqa: E402


def run_from_env(root: Path = DEFAULT_ROOT) -> int:
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    assets_prefix = os.getenv("ASSETS_PREFIX")
    allow_heuristic_fallback = os.getenv("SIMREADY_ALLOW_HEURISTIC_FALLBACK")
    production_mode = os.getenv("SIMREADY_PRODUCTION_MODE")
    return prepare_simready_assets_job(
        bucket,
        scene_id,
        assets_prefix,
        root=root,
        allow_heuristic_fallback=None if allow_heuristic_fallback is None else allow_heuristic_fallback.lower() in {"1", "true", "yes"},
        production_mode=None if production_mode is None else production_mode.lower() in {"1", "true", "yes"},
    )


def prepare_simready_assets(
    assets_prefix: str,
    bucket: str = "",
    scene_id: str = "",
    root: Path = DEFAULT_ROOT,
    allow_heuristic_fallback: Optional[bool] = None,
    production_mode: Optional[bool] = None,
) -> int:
    return prepare_simready_assets_job(
        bucket,
        scene_id,
        assets_prefix,
        root=root,
        allow_heuristic_fallback=allow_heuristic_fallback,
        production_mode=production_mode,
    )


__all__ = ["run_from_env", "prepare_simready_assets"]
