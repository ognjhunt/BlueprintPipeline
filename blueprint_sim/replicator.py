from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from .manifest import DEFAULT_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
REPLICATOR_DIR = REPO_ROOT / "replicator-job"
if str(REPLICATOR_DIR) not in sys.path:
    sys.path.append(str(REPLICATOR_DIR))

from generate_replicator_bundle import generate_replicator_bundle_job  # noqa: E402


def run_from_env(root: Path = DEFAULT_ROOT) -> int:
    bucket = os.getenv("BUCKET", "")
    scene_id = os.getenv("SCENE_ID", "")
    seg_prefix = os.getenv("SEG_PREFIX", f"scenes/{scene_id}/seg")
    assets_prefix = os.getenv("ASSETS_PREFIX", f"scenes/{scene_id}/assets")
    usd_prefix = os.getenv("USD_PREFIX", f"scenes/{scene_id}/usd")
    replicator_prefix = os.getenv("REPLICATOR_PREFIX", f"scenes/{scene_id}/replicator")
    requested_policies_str = os.getenv("REQUESTED_POLICIES", "")
    requested_policies = [p.strip() for p in requested_policies_str.split(",") if p.strip()] or None
    if not scene_id:
        print("[REPLICATOR] ERROR: SCENE_ID environment variable is required", file=sys.stderr)
        return 1

    return generate_replicator_bundle_job(
        bucket=bucket,
        scene_id=scene_id,
        seg_prefix=seg_prefix,
        assets_prefix=assets_prefix,
        usd_prefix=usd_prefix,
        replicator_prefix=replicator_prefix,
        requested_policies=requested_policies,
        root=root,
    )


def generate_replicator_bundle(
    scene_id: str,
    seg_prefix: str,
    assets_prefix: str,
    usd_prefix: str,
    replicator_prefix: str,
    bucket: str = "",
    requested_policies: Optional[List[str]] = None,
    root: Path = DEFAULT_ROOT,
) -> int:
    return generate_replicator_bundle_job(
        bucket=bucket,
        scene_id=scene_id,
        seg_prefix=seg_prefix,
        assets_prefix=assets_prefix,
        usd_prefix=usd_prefix,
        replicator_prefix=replicator_prefix,
        requested_policies=requested_policies,
        root=root,
    )


__all__ = ["run_from_env", "generate_replicator_bundle"]
