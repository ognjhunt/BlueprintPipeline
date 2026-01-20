#!/usr/bin/env python3
"""
Asset Provenance Generator.

Generates `asset_provenance.json` files for legal/procurement compliance.

This is CRITICAL for enterprise sales - legal teams block purchases without:
1. Clear license information per asset
2. Source attribution (where did each asset come from?)
3. Commercial use clearance
4. Audit trail of transformations

Output: legal/asset_provenance.json with:
- Per-asset license and source
- Commercial use flags
- Transformation chain (what processing was applied)
- Generation timestamps
- Pipeline version tracking

Reference: NVIDIA Omniverse Asset Licensing Guidelines
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class LicenseType(str, Enum):
    """Standard license types for assets."""

    # Commercial-friendly
    CC0 = "cc0"  # Public domain
    CC0_1_0 = "cc0-1.0"  # SPDX identifier for CC0 1.0
    CC_BY = "cc-by"  # Attribution only
    CC_BY_4_0 = "cc-by-4.0"  # SPDX identifier for CC-BY 4.0
    CC_BY_SA = "cc-by-sa"  # Attribution + ShareAlike
    MIT = "mit"
    APACHE_2 = "apache-2.0"
    BSD_2_CLAUSE = "bsd-2-clause"
    BSD_3_CLAUSE = "bsd-3-clause"
    GPL_3_0 = "gpl-3.0"
    LGPL_3_0 = "lgpl-3.0"
    MPL_2_0 = "mpl-2.0"
    PROPRIETARY_COMMERCIAL = "proprietary-commercial"  # Our own, commercial OK
    NVIDIA_OMNIVERSE = "nvidia-omniverse"  # NVIDIA asset store
    SIMREADY = "simready"  # SimReady assets

    # Non-commercial / Restricted
    CC_BY_NC = "cc-by-nc"  # Non-commercial
    CC_BY_NC_SA = "cc-by-nc-sa"  # Non-commercial + ShareAlike
    ACADEMIC_ONLY = "academic-only"
    RESEARCH_ONLY = "research-only"
    UNKNOWN = "unknown"


# Licenses that are OK for commercial use
COMMERCIAL_OK_LICENSES = {
    LicenseType.CC0,
    LicenseType.CC0_1_0,
    LicenseType.CC_BY,
    LicenseType.CC_BY_4_0,
    LicenseType.CC_BY_SA,
    LicenseType.MIT,
    LicenseType.APACHE_2,
    LicenseType.BSD_2_CLAUSE,
    LicenseType.BSD_3_CLAUSE,
    LicenseType.GPL_3_0,
    LicenseType.LGPL_3_0,
    LicenseType.MPL_2_0,
    LicenseType.PROPRIETARY_COMMERCIAL,
    LicenseType.NVIDIA_OMNIVERSE,
    LicenseType.SIMREADY,
}


@dataclass
class AssetSource:
    """Source information for an asset."""

    # Primary source
    source_type: str  # "reconstruction", "gemini", "simready", "objaverse", "manual"
    source_id: Optional[str] = None  # Original ID in source system
    source_url: Optional[str] = None  # URL if applicable

    # Attribution
    creator: Optional[str] = None
    creation_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.source_type,
            "id": self.source_id,
            "url": self.source_url,
            "creator": self.creator,
            "creation_date": self.creation_date,
        }


@dataclass
class TransformationStep:
    """A single transformation applied to an asset."""

    step_name: str  # "reconstruction", "scaling", "physics_authoring", "collision_gen"
    tool_name: str  # "3d-re-gen", "blueprint-scaler", etc.
    tool_version: str
    timestamp: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_hash: Optional[str] = None  # Hash of input
    output_hash: Optional[str] = None  # Hash of output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_name,
            "tool": self.tool_name,
            "version": self.tool_version,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
        }


@dataclass
class AssetProvenance:
    """Complete provenance record for a single asset."""

    asset_id: str
    asset_path: str

    # License information
    license: LicenseType = LicenseType.UNKNOWN
    license_url: Optional[str] = None
    commercial_use_ok: bool = False

    # Source information
    source: AssetSource = field(default_factory=lambda: AssetSource(source_type="unknown"))

    # Transformation chain
    transformations: List[TransformationStep] = field(default_factory=list)

    # Verification
    content_hash: Optional[str] = None  # SHA-256 of current file
    verification_timestamp: Optional[str] = None

    # Metadata
    category: str = "unknown"
    description: str = ""

    def __post_init__(self):
        # Auto-determine commercial status from license
        if isinstance(self.license, str):
            self.license = LicenseType(self.license)
        self.commercial_use_ok = self.license in COMMERCIAL_OK_LICENSES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "asset_path": self.asset_path,
            "license": {
                "type": self.license.value,
                "url": self.license_url,
                "commercial_ok": self.commercial_use_ok,
            },
            "source": self.source.to_dict(),
            "transformations": [t.to_dict() for t in self.transformations],
            "verification": {
                "content_hash": self.content_hash,
                "timestamp": self.verification_timestamp,
            },
            "metadata": {
                "category": self.category,
                "description": self.description,
            },
        }


@dataclass
class SceneProvenance:
    """Complete provenance record for a scene."""

    scene_id: str
    version: str = "1.0.0"
    generated_at: str = ""

    # Scene-level license (most restrictive of all assets)
    scene_license: LicenseType = LicenseType.PROPRIETARY_COMMERCIAL
    commercial_use_ok: bool = True

    # Asset provenances
    assets: List[AssetProvenance] = field(default_factory=list)

    # Pipeline information
    pipeline_name: str = "BlueprintPipeline"
    pipeline_version: str = "1.0.0"

    # Summary
    total_assets: int = 0
    commercial_ok_assets: int = 0
    non_commercial_assets: int = 0
    unknown_license_assets: int = 0

    # Blocking issues for commercial use
    commercial_blockers: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat() + "Z"

    def compute_scene_license(self) -> None:
        """Compute the most restrictive license across all assets."""
        self.total_assets = len(self.assets)
        self.commercial_ok_assets = sum(1 for a in self.assets if a.commercial_use_ok)
        self.non_commercial_assets = sum(
            1 for a in self.assets
            if not a.commercial_use_ok and a.license != LicenseType.UNKNOWN
        )
        self.unknown_license_assets = sum(
            1 for a in self.assets if a.license == LicenseType.UNKNOWN
        )

        # Scene is commercial OK only if ALL assets are commercial OK
        self.commercial_use_ok = self.non_commercial_assets == 0 and self.unknown_license_assets == 0

        # Find blocking issues
        self.commercial_blockers = []
        for asset in self.assets:
            if asset.license in [LicenseType.CC_BY_NC, LicenseType.CC_BY_NC_SA]:
                self.commercial_blockers.append(
                    f"Asset '{asset.asset_id}' has non-commercial license ({asset.license.value})"
                )
            elif asset.license in [LicenseType.ACADEMIC_ONLY, LicenseType.RESEARCH_ONLY]:
                self.commercial_blockers.append(
                    f"Asset '{asset.asset_id}' is restricted to academic/research use"
                )
            elif asset.license == LicenseType.UNKNOWN:
                self.commercial_blockers.append(
                    f"Asset '{asset.asset_id}' has unknown license - requires verification"
                )

        # Determine scene license
        if self.commercial_use_ok:
            self.scene_license = LicenseType.PROPRIETARY_COMMERCIAL
        elif self.non_commercial_assets > 0:
            self.scene_license = LicenseType.CC_BY_NC  # Most restrictive commercial-blocking
        else:
            self.scene_license = LicenseType.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "version": self.version,
            "generated_at": self.generated_at,
            "license": {
                "type": self.scene_license.value,
                "commercial_ok": self.commercial_use_ok,
                "blockers": self.commercial_blockers,
            },
            "summary": {
                "total_assets": self.total_assets,
                "commercial_ok": self.commercial_ok_assets,
                "non_commercial": self.non_commercial_assets,
                "unknown_license": self.unknown_license_assets,
            },
            "pipeline": {
                "name": self.pipeline_name,
                "version": self.pipeline_version,
            },
            "assets": [a.to_dict() for a in self.assets],
        }

    def save(self, output_path: Path) -> None:
        """Save provenance to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class AssetProvenanceGenerator:
    """
    Generates asset provenance records from scene manifest.

    Usage:
        generator = AssetProvenanceGenerator(scene_dir)
        provenance = generator.generate()
        provenance.save(scene_dir / "legal" / "asset_provenance.json")
    """

    def __init__(
        self,
        scene_dir: Path,
        scene_id: Optional[str] = None,
        manifest_path: Optional[Path] = None,
        pipeline_version: str = "1.0.0",
        verbose: bool = True,
    ):
        self.scene_dir = Path(scene_dir)
        self.scene_id = scene_id or self.scene_dir.name
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.pipeline_version = pipeline_version
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[PROVENANCE] {msg}")

    def generate(self) -> SceneProvenance:
        """Generate complete provenance record for the scene."""
        self.log(f"Generating provenance for: {self.scene_id}")

        provenance = SceneProvenance(
            scene_id=self.scene_id,
            pipeline_version=self.pipeline_version,
        )

        # Load manifest
        manifest = self._load_manifest()

        # Generate provenance for each asset
        for obj in manifest.get("objects", []):
            asset_provenance = self._generate_asset_provenance(obj)
            if asset_provenance:
                provenance.assets.append(asset_provenance)

        # Compute scene-level license
        provenance.compute_scene_license()

        self.log(
            f"Provenance complete: {provenance.total_assets} assets, "
            f"{provenance.commercial_ok_assets} commercial OK"
        )

        return provenance

    def _load_manifest(self) -> Dict[str, Any]:
        """Load scene manifest."""
        manifest_path = self.manifest_path or (self.scene_dir / "assets" / "scene_manifest.json")
        self.log(f"Using manifest path: {manifest_path}")
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f)
        self.log(f"Manifest not found at {manifest_path}; proceeding with empty manifest")
        return {}

    def _generate_asset_provenance(self, obj: Dict[str, Any]) -> Optional[AssetProvenance]:
        """Generate provenance for a single asset."""
        obj_id = obj.get("id", "")
        if not obj_id:
            return None

        # Skip background/shell objects
        sim_role = obj.get("sim_role", "unknown")
        if sim_role in ["background", "scene_shell"]:
            return None

        # Get asset info
        asset_info = obj.get("asset", {})
        asset_path = asset_info.get("path", "")

        # Determine source
        source = self._determine_source(obj, asset_info)

        # Determine license
        license_type, license_url = self._determine_license(obj, asset_info, source)

        # Build transformation chain
        transformations = self._build_transformation_chain(obj)

        # Compute content hash if file exists
        content_hash = None
        full_path = self.scene_dir / asset_path if asset_path else None
        if full_path and full_path.exists():
            content_hash = self._compute_file_hash(full_path)

        return AssetProvenance(
            asset_id=obj_id,
            asset_path=asset_path,
            license=license_type,
            license_url=license_url,
            source=source,
            transformations=transformations,
            content_hash=content_hash,
            verification_timestamp=datetime.utcnow().isoformat() + "Z",
            category=obj.get("category", "unknown"),
            description=obj.get("description", ""),
        )

    def _determine_source(
        self,
        obj: Dict[str, Any],
        asset_info: Dict[str, Any],
    ) -> AssetSource:
        """Determine the source of an asset."""
        # Check explicit source in manifest
        source_type = asset_info.get("source", "unknown")

        # Map common sources
        source_mapping = {
            "gemini": ("gemini", "Generated by Google Gemini"),
            "blueprint_recipe": ("reconstruction", "3D-RE-GEN reconstruction"),
            "simready": ("simready", "NVIDIA SimReady asset"),
            "objaverse": ("objaverse", "Objaverse open dataset"),
            "manual": ("manual", "Manual authoring"),
        }

        source_type_norm, creator = source_mapping.get(
            source_type.lower(),
            (source_type, None)
        )

        # Check for reconstruction metadata
        if "reconstruction" in obj:
            recon = obj["reconstruction"]
            return AssetSource(
                source_type="reconstruction",
                source_id=recon.get("reconstruction_id"),
                source_url=recon.get("source_url"),
                creator="3D-RE-GEN",
                creation_date=recon.get("timestamp"),
            )

        return AssetSource(
            source_type=source_type_norm,
            source_id=asset_info.get("asset_id"),
            source_url=asset_info.get("url"),
            creator=creator,
        )

    def _determine_license(
        self,
        obj: Dict[str, Any],
        asset_info: Dict[str, Any],
        source: AssetSource,
    ) -> tuple[LicenseType, Optional[str]]:
        """Determine the license for an asset."""
        # Check explicit license
        explicit_license = asset_info.get("license", "").lower()

        # Map string licenses to enum
        license_map = {
            "cc0": LicenseType.CC0,
            "cc-by": LicenseType.CC_BY,
            "cc by": LicenseType.CC_BY,
            "cc-by-sa": LicenseType.CC_BY_SA,
            "cc-by-nc": LicenseType.CC_BY_NC,
            "cc-by-nc-sa": LicenseType.CC_BY_NC_SA,
            "mit": LicenseType.MIT,
            "apache": LicenseType.APACHE_2,
            "apache-2.0": LicenseType.APACHE_2,
            "proprietary": LicenseType.PROPRIETARY_COMMERCIAL,
            "nvidia": LicenseType.NVIDIA_OMNIVERSE,
            "simready": LicenseType.SIMREADY,
        }

        for key, license_type in license_map.items():
            if key in explicit_license:
                return license_type, asset_info.get("license_url")

        # Infer from source
        source_licenses = {
            "gemini": LicenseType.PROPRIETARY_COMMERCIAL,  # Our generated content
            "reconstruction": LicenseType.PROPRIETARY_COMMERCIAL,  # Our processed content
            "simready": LicenseType.SIMREADY,
            "objaverse": LicenseType.CC_BY,  # Most Objaverse is CC-BY
        }

        if source.source_type in source_licenses:
            return source_licenses[source.source_type], None

        return LicenseType.UNKNOWN, None

    def _build_transformation_chain(self, obj: Dict[str, Any]) -> List[TransformationStep]:
        """Build the transformation chain for an asset."""
        transformations = []
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Check for reconstruction
        if "reconstruction" in obj:
            transformations.append(TransformationStep(
                step_name="reconstruction",
                tool_name="3D-RE-GEN",
                tool_version="1.0",
                timestamp=obj["reconstruction"].get("timestamp", timestamp),
                parameters={
                    "method": obj["reconstruction"].get("method", "unknown"),
                },
            ))

        # Check for scaling
        transform = obj.get("transform", {})
        if transform.get("scale") and transform["scale"] != [1, 1, 1]:
            transformations.append(TransformationStep(
                step_name="scaling",
                tool_name="BlueprintPipeline",
                tool_version=self.pipeline_version,
                timestamp=timestamp,
                parameters={"scale": transform["scale"]},
            ))

        # Check for physics authoring
        if obj.get("physics"):
            transformations.append(TransformationStep(
                step_name="physics_authoring",
                tool_name="BlueprintPipeline",
                tool_version=self.pipeline_version,
                timestamp=timestamp,
                parameters={
                    "mass": obj["physics"].get("mass"),
                    "friction": obj["physics"].get("friction"),
                },
            ))

        # Check for collision generation
        if obj.get("physics", {}).get("collision_shape"):
            transformations.append(TransformationStep(
                step_name="collision_generation",
                tool_name="BlueprintPipeline",
                tool_version=self.pipeline_version,
                timestamp=timestamp,
                parameters={
                    "shape_type": obj["physics"]["collision_shape"],
                },
            ))

        # If no transformations, add a passthrough
        if not transformations:
            transformations.append(TransformationStep(
                step_name="import",
                tool_name="BlueprintPipeline",
                tool_version=self.pipeline_version,
                timestamp=timestamp,
            ))

        return transformations

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


def generate_asset_provenance(
    scene_dir: Path,
    output_path: Optional[Path] = None,
    scene_id: Optional[str] = None,
    manifest_path: Optional[Path] = None,
    verbose: bool = True,
) -> SceneProvenance:
    """
    Convenience function to generate asset provenance.

    Args:
        scene_dir: Path to scene directory
        output_path: Optional path to save provenance (defaults to scene_dir/legal/asset_provenance.json)
        scene_id: Optional scene ID
        manifest_path: Optional path to scene manifest JSON
        verbose: Print progress

    Returns:
        SceneProvenance
    """
    generator = AssetProvenanceGenerator(
        scene_dir,
        scene_id,
        manifest_path=manifest_path,
        verbose=verbose,
    )
    provenance = generator.generate()

    if output_path is None:
        output_path = scene_dir / "legal" / "asset_provenance.json"

    provenance.save(output_path)

    return provenance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate asset provenance")
    parser.add_argument("scene_dir", type=Path, help="Path to scene directory")
    parser.add_argument("--output", type=Path, help="Output path for provenance")
    parser.add_argument("--scene-id", help="Scene identifier")
    parser.add_argument("--manifest-path", type=Path, help="Path to scene manifest JSON")

    args = parser.parse_args()

    provenance = generate_asset_provenance(
        scene_dir=args.scene_dir,
        output_path=args.output,
        scene_id=args.scene_id,
        manifest_path=args.manifest_path,
    )

    print(f"\nAsset Provenance Summary")
    print("=" * 50)
    print(f"Scene ID: {provenance.scene_id}")
    print(f"Commercial Use OK: {provenance.commercial_use_ok}")
    print(f"Scene License: {provenance.scene_license.value}")
    print(f"\nAsset Summary:")
    print(f"  Total: {provenance.total_assets}")
    print(f"  Commercial OK: {provenance.commercial_ok_assets}")
    print(f"  Non-Commercial: {provenance.non_commercial_assets}")
    print(f"  Unknown License: {provenance.unknown_license_assets}")

    if provenance.commercial_blockers:
        print(f"\nCommercial Blockers ({len(provenance.commercial_blockers)}):")
        for blocker in provenance.commercial_blockers:
            print(f"  ❌ {blocker}")
