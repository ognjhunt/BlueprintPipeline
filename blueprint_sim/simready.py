from __future__ import annotations

import os
import sys
from pathlib import Path

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
    return prepare_simready_assets_job(bucket, scene_id, assets_prefix, root=root)


def prepare_simready_assets(
    assets_prefix: str,
    bucket: str = "",
    scene_id: str = "",
    root: Path = DEFAULT_ROOT,
) -> int:
    return prepare_simready_assets_job(bucket, scene_id, assets_prefix, root=root)


__all__ = ["run_from_env", "prepare_simready_assets"]
