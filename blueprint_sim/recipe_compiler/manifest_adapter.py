from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from tools.scene_manifest.validate_manifest import (
    manifest_from_blueprint_recipe,
)


class SceneManifestAdapter:
    """Adapt scene plans and matched assets into the canonical manifest.

    This is a lightweight version of the adapter from BlueprintRecipe. It
    relies on the shared ``manifest_from_blueprint_recipe`` helper in
    ``tools/scene_manifest`` so the structure matches Pipeline's schema.
    """

    def __init__(
        self,
        asset_root: str,
        meters_per_unit: float = 1.0,
        coordinate_frame: str = "y_up",
        up_axis: str = "Y",
    ) -> None:
        self.asset_root = asset_root
        self.meters_per_unit = meters_per_unit
        self.coordinate_frame = coordinate_frame
        self.up_axis = up_axis

    def build_manifest(
        self,
        scene_plan: Dict[str, Any],
        matched_assets: Dict[str, Dict[str, Any]],
        recipe: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        manifest = manifest_from_blueprint_recipe(scene_plan, matched_assets)
        manifest.setdefault("metadata", {}).update(metadata or {})
        manifest["meters_per_unit"] = self.meters_per_unit
        manifest["coordinate_frame"] = self.coordinate_frame
        manifest["up_axis"] = self.up_axis
        manifest["asset_root"] = self.asset_root
        return manifest

    def save_manifest(self, manifest: Dict[str, Any], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return path


__all__ = ["SceneManifestAdapter"]
