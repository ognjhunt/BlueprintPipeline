"""
Asset Provenance Generator for BlueprintPipeline.

Tracks the source, license, and commercial viability of all assets in a scene.
Essential for legal compliance when selling datasets to labs.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AssetProvenanceEntry:
    """Legal provenance for a single asset."""
    asset_id: str
    name: str
    source: str
    license: str
    commercial_ok: bool
    description: str = ""
    original_url: Optional[str] = None
    attribution: Optional[str] = None


class AssetProvenanceGenerator:
    """Generates legal provenance reports for scene assets."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, msg: str):
        if self.verbose:
            print(f"[PROVENANCE] {msg}")

    def generate(self, manifest: Dict[str, Any], output_path: Path):
        """
        Generate a provenance report from a scene manifest.

        Args:
            manifest: The scene manifest dictionary.
            output_path: Where to save the provenance JSON.
        """
        self.log(f"Generating provenance report for scene: {manifest.get('scene_id')}")

        assets = []
        objects = manifest.get("objects", [])

        for obj in objects:
            asset_data = obj.get("asset", {})
            # Skip if no asset info (e.g. background)
            if not asset_data and not obj.get("asset_path"):
                continue

            entry = self._build_entry(obj)
            assets.append(entry)

        report = {
            "scene_id": manifest.get("scene_id"),
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_assets": len(assets),
            "commercial_summary": {
                "all_commercial_ok": all(a["commercial_ok"] for a in assets),
                "commercial_count": sum(1 for a in assets if a["commercial_ok"]),
                "non_commercial_count": sum(1 for a in assets if not a["commercial_ok"]),
            },
            "assets": assets
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        self.log(f"Provenance report saved to {output_path}")
        if not report["commercial_summary"]["all_commercial_ok"]:
            self.log("WARNING: Scene contains non-commercially usable assets!")

    def _build_entry(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Build a provenance entry for an object."""
        asset_data = obj.get("asset", {})

        # Determine source and license with sensible fallbacks
        source = asset_data.get("source") or obj.get("source") or "internal"
        license_name = asset_data.get("license") or obj.get("license") or "proprietary"

        # Commercial OK logic:
        # - internal/proprietary is always OK
        # - CC0/Public Domain is always OK
        # - Some sources like Objaverse may have mixed licenses
        commercial_ok = asset_data.get("commercial_ok")
        if commercial_ok is None:
            # Heuristic
            if license_name.lower() in ["cc0", "public_domain", "proprietary"]:
                commercial_ok = True
            elif source.lower() == "blueprintpipeline":
                commercial_ok = True
            else:
                commercial_ok = False

        return {
            "asset_id": obj.get("id"),
            "name": obj.get("name", obj.get("category")),
            "source": source,
            "license": license_name,
            "commercial_ok": commercial_ok,
            "description": obj.get("description", ""),
            "attribution": asset_data.get("attribution"),
        }
