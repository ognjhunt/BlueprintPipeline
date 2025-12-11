from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.scene_manifest.loader import load_manifest_or_scene_assets  # noqa: E402


DEFAULT_ROOT = Path("/mnt/gcs")


def load_canonical_manifest(assets_prefix: str, root: Path = DEFAULT_ROOT) -> Dict:
    """Load the canonical ``scene_manifest.json`` (or legacy fallback).

    Args:
        assets_prefix: Prefix to the assets directory relative to the storage root.
        root: Storage root (defaults to ``/mnt/gcs``).

    Raises:
        FileNotFoundError: if neither manifest nor legacy scene_assets.json is present.
    """

    assets_root = root / assets_prefix
    manifest = load_manifest_or_scene_assets(assets_root)
    if manifest is None:
        raise FileNotFoundError(
            f"scene manifest not found at {assets_root / 'scene_manifest.json'} "
            f"or legacy plan at {assets_root / 'scene_assets.json'}"
        )
    return manifest


__all__ = ["load_canonical_manifest", "DEFAULT_ROOT"]
