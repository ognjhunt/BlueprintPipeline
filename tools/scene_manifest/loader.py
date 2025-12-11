"""Utilities for loading and converting scene manifest data.

The canonical input is ``scene_manifest.json`` which is described by
``tools/scene_manifest/manifest_schema.json``. Some downstream jobs still expect
legacy ``scene_assets.json``-style payloads; ``load_manifest_or_scene_assets``
handles both cases and normalizes manifests into the legacy structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

SIM_ROLE_TO_TYPE = {
    "manipulable_object": "interactive",
    "articulated_furniture": "interactive",
    "articulated_appliance": "interactive",
    "scene_shell": "static",
    "background": "static",
    "static": "static",
}


def _canonical_to_legacy_object(obj: Dict) -> Dict:
    source = obj.get("source", {}) or {}
    from_scene_assets = source.get("scene_assets") or {}
    generation = obj.get("asset_generation") or {}
    generation_inputs = generation.get("inputs") or {}

    sim_role = obj.get("sim_role", "unknown")
    legacy_type = SIM_ROLE_TO_TYPE.get(sim_role, "static")

    entry = {
        "id": obj.get("id"),
        "class_name": (obj.get("semantics") or {}).get("category")
        or from_scene_assets.get("class_name"),
        "type": legacy_type,
        "pipeline": generation.get("pipeline") or from_scene_assets.get("pipeline"),
        "multiview_dir": generation_inputs.get("multiview_dir")
        or from_scene_assets.get("multiview_dir"),
        "crop_path": generation_inputs.get("crop_path") or from_scene_assets.get("crop_path"),
        "preferred_view": generation_inputs.get("preferred_view")
        or from_scene_assets.get("preferred_view"),
        "approx_location": (obj.get("placement") or {}).get("approx_location")
        or from_scene_assets.get("approx_location"),
        "asset_path": (obj.get("asset") or {}).get("path")
        or from_scene_assets.get("asset_path"),
        "interactive_output": generation.get("output")
        or from_scene_assets.get("interactive_output"),
        "physx_endpoint": (obj.get("articulation") or {}).get("physx_endpoint")
        or from_scene_assets.get("physx_endpoint"),
        "polygon": (obj.get("placement") or {}).get("polygon")
        or from_scene_assets.get("polygon"),
    }

    # Drop empty values to mirror scene_assets.json more closely
    return {k: v for k, v in entry.items() if v is not None}


def _manifest_to_legacy(manifest: Dict) -> Dict:
    objects = [_canonical_to_legacy_object(o) for o in manifest.get("objects", [])]
    return {
        "scene_id": manifest.get("scene_id"),
        "objects": objects,
        "schema_version": manifest.get("schema_version"),
    }


def load_manifest_or_scene_assets(assets_root: Path) -> Optional[Dict]:
    """Load ``scene_manifest.json`` when present, otherwise fall back to
    ``scene_assets.json``.

    Downstream jobs can continue to operate on the familiar scene-assets shape
    while the pipeline migrates to the canonical manifest.
    """

    manifest_path = assets_root / "scene_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r") as f:
            manifest = json.load(f)
        return _manifest_to_legacy(manifest)

    legacy_path = assets_root / "scene_assets.json"
    if legacy_path.is_file():
        with legacy_path.open("r") as f:
            return json.load(f)

    return None


def load_manifest(manifest_path: Path) -> Dict:
    with manifest_path.open("r") as f:
        return json.load(f)


__all__ = [
    "load_manifest",
    "load_manifest_or_scene_assets",
]
